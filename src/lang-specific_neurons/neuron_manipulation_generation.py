# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from types import MethodType
import torch.nn.functional as F
from itertools import combinations
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"

MODEL_PATHS = {
    "Aya-Expanse-8B": "../../models/Aya-Expanse-8B",
    "Llama-3.1-8B-Instruct": "../../models/Llama-3.1-8B-Instruct",
    "Qwen3-8B": "../../models/Qwen3-8B"
}

LANG_MAP = {
    "English": "en",
    "Chinese": "zh_CN",
    "Japanese": "ja",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Swedish": "sv",
    "Korean": "ko",
    "Danish": "da",
    "Thai": "th",
    "Hindi": "hi",
    "Tamil": "ta",
    "Mongolian": "mn",
    "Welsh": "cy",
    "Swahili": "sw",
    "Turkmen": "tk",
    "Scottish Gaelic": "gd",
    "Zulu": "zu"
}

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def select_languages(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    languages = data["languages"]
    groups = {"high": [], "medium": [], "low": []}
    for lang, info in languages.items():
        groups[info["resource"].lower()].append(lang)
    selected = []
    for langs in groups.values():
        if langs:
            selected.extend(random.sample(langs, len(langs) // 2))
    return selected

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_neurons(file_path):
    neurons = set()
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                x, y = line.strip("()\n").split(",")
                neurons.add((int(x), int(y)))
    return neurons

def make_layer_index_dict(neuron_set):
    layer_index = {}
    for layer, idx in sorted(neuron_set):
        layer_index.setdefault(layer, []).append(idx)
    return {layer: torch.tensor(v, dtype=torch.long) for layer, v in layer_index.items()}

def generate_activate_mask(selected_langs, model_name):
    selected_codes = [LANG_MAP[l] for l in selected_langs]
    others = [v for k, v in LANG_MAP.items() if k not in selected_langs]
    selected_sets = {}
    for code in selected_codes:
        p = f"./outputs/neurons/top_neurons_{code}_{model_name}.txt"
        if os.path.exists(p):
            selected_sets[code] = load_neurons(p)
    other_set = set()
    for code in others:
        p = f"./outputs/neurons/top_neurons_{code}_{model_name}.txt"
        if os.path.exists(p):
            other_set |= load_neurons(p)
    pairwise_inters = [selected_sets[a] & selected_sets[b] for a, b in combinations(selected_sets, 2)]
    final_set = set().union(*pairwise_inters) - other_set
    return make_layer_index_dict(final_set)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_crosslingual_jobs(input_dir, model_name, languages):
    jobs = {}
    base = os.path.join(input_dir, model_name)
    for c in languages:
        for q in languages:
            path = os.path.join(base, f"{c}-{q}.json")
            if os.path.exists(path):
                data = load_json(path)
                jobs[f"{c}-{q}"] = [
                    {
                        "id": d["id"],
                        "ctx_lang": d["context_lang"],
                        "query_lang": d["query_lang"],
                        "context": d["context"],
                        "query_question": d["query_question"],
                        "truth": d["truth"]
                    } for d in data
                ]
    return jobs

def compute_average_activations(languages, model_name):
    acts = []
    for lang in languages:
        code = LANG_MAP[lang]
        p = f"./outputs/activations/activation.{code}.flores.{model_name}"
        data = torch.load(p, map_location="cpu")
        acts.append(data["average_activations"])
    stacked = torch.stack(acts, dim=-1)
    return stacked, {lang: i for i, lang in enumerate(languages)}

def factory(layer_idx, deactivate_indices=None, activate_indices=None, boost_values=None):
    def forward(self, x):
        gate_up = self.gate_up_proj(x)[0] if isinstance(self.gate_up_proj(x), tuple) else self.gate_up_proj(x)
        i = gate_up.size(-1)
        silu = F.silu(gate_up[..., :i//2])
        if deactivate_indices is not None:
            silu[..., deactivate_indices] = -silu[..., deactivate_indices]
        if activate_indices is not None and boost_values is not None and boost_values.numel() > 0:
            shape = [1] * silu.dim(); shape[-1] = -1
            silu[..., activate_indices] = boost_values.to(silu.dtype).reshape(*shape)
        x = silu * gate_up[..., i//2:]
        return self.down_proj(x)[0] if isinstance(self.down_proj(x), tuple) else self.down_proj(x)
    return forward

def apply_intervention(model, activation, deactivation, mask, avg_act, selected_langs, lang_to_idx):
    layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    for layer_idx, indices in mask.items():
        obj = layers[layer_idx].mlp
        if deactivation:
            obj.forward = MethodType(factory(layer_idx, deactivate_indices=indices.to("cuda")), obj)
        elif activation:
            langs = torch.tensor([lang_to_idx[l] for l in selected_langs], device=avg_act.device)
            boost = avg_act[layer_idx, indices][:, langs].mean(-1).to("cuda")
            obj.forward = MethodType(factory(layer_idx, activate_indices=indices.to("cuda"), boost_values=boost), obj)

def build_crosslingual_prompt(context, question, query_lang):
    return f"""Here are some question-and-answer pairs set in a future world that is very different from today:

{context}

Based on the knowledge above, answer the following question in {query_lang}:

<question>
{question}
</question>

Your output must strictly follow this format:
Output: <answer>Your answer here</answer>

Do not include any explanation or text outside the <answer> tags.
Output: """

def extract_answer_tag(text):
    if not text: return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None

def run_single_experiment(model, sampling_params, jobs_by_pair, activation, deactivation, mask, avg_act, selected_langs, lang_to_idx, save_dir, max_attempts):
    apply_intervention(model, activation, deactivation, mask, avg_act, selected_langs, lang_to_idx)
    for pair, jobs in tqdm(jobs_by_pair.items(), desc="Language Pairs"):
        prompts = [build_crosslingual_prompt(j["context"], j["query_question"], j["query_lang"]) for j in jobs]
        outputs = [o.outputs[0].text.strip() for o in model.generate(prompts, sampling_params)]
        answers = [extract_answer_tag(o) for o in outputs]
        for _ in range(max_attempts - 1):
            retry = [i for i, a in enumerate(answers) if a is None]
            if not retry: break
            retry_prompts = [prompts[i] for i in retry]
            retry_outs = [o.outputs[0].text.strip() for o in model.generate(retry_prompts, sampling_params)]
            for k, i in enumerate(retry):
                parsed = extract_answer_tag(retry_outs[k])
                if parsed:
                    answers[i] = parsed
                    outputs[i] = retry_outs[k]
        results = [{**j, "answer": a or o} for j, o, a in zip(jobs, outputs, answers)]
        save_json(results, os.path.join(save_dir, f"{pair}.json"))
        print(f"[run_vllm:{pair}] Parsed OK: {sum(a is not None for a in answers)}/{len(answers)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--activation", action="store_true")
    parser.add_argument("--deactivation", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_attempts", type=int, default=50)
    parser.add_argument("--info_path", type=str, default="../utils/info.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.activation == args.deactivation:
        raise ValueError("Specify exactly one of --activation or --deactivation")

    seed = args.seed or len(args.model_name)
    set_seed(seed)

    mode = "activation" if args.activation else "deactivation"
    args.save_dir = os.path.join(args.save_dir, mode, args.model_name)
    safe_mkdir(args.save_dir)

    selected_langs = select_languages(args.info_path)
    save_json(selected_langs, os.path.join(args.save_dir, "selected_langs.json"))

    mask = generate_activate_mask(selected_langs, args.model_name)
    info = load_json(args.info_path)
    languages = list(info["languages"].keys())
    jobs = build_crosslingual_jobs(args.input_dir, args.model_name, languages)
    avg_act, lang_to_idx = compute_average_activations(languages, args.model_name)

    model = LLM(
        model=MODEL_PATHS[args.model_name],
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens)

    run_single_experiment(
        model, sampling_params, jobs, args.activation, args.deactivation,
        mask, avg_act, selected_langs, lang_to_idx, args.save_dir, args.max_attempts
    )

if __name__ == "__main__":
    main()