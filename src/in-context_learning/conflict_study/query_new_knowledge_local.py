# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
import time
import random
import argparse
from tqdm import tqdm
from statistics import mean
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
def extract_answer_tag(text):
    if not text:
        return None
    m = TAG_RE.search(text)
    return m.group(1).strip() if m else None

def prompt_statistics(prompts, tokenizer=None):
    char_lens = [len(p) for p in prompts]
    print(f"[Prompt Chars] Max: {max(char_lens)}, Min: {min(char_lens)}, Avg: {mean(char_lens):.1f}")
    if tokenizer:
        token_lens = [len(tokenizer(p)["input_ids"]) for p in tqdm(prompts)]
        print(f"[Prompt Tokens] Max: {max(token_lens)}, Min: {min(token_lens)}, Avg: {mean(token_lens):.1f}, Total: {sum(token_lens)}")

# ----------------------------
# Prompt
# ----------------------------
def build_conflict_context(data, lang_a, lang_b, target_item, num_context=50):
    lines = []
    candidates = [item for item in data if item["id"] != target_item["id"]]
    sampled = random.sample(candidates, k=num_context - 1) if len(candidates) >= num_context - 1 else candidates
    lines.append(f"Q: {target_item['data'][lang_a]['question_1']}\nA: {target_item['data'][lang_a]['conflict_answer']}")
    lines.append(f"Q: {target_item['data'][lang_b]['question_1']}\nA: {target_item['data'][lang_b]['answer']}")
    for item in sampled:
        lines.append(f"Q: {item['data'][lang_a]['question_1']}\nA: {item['data'][lang_a]['conflict_answer']}")
        lines.append(f"Q: {item['data'][lang_b]['question_1']}\nA: {item['data'][lang_b]['answer']}")
    random.shuffle(lines)
    return "\n".join(lines)

def build_conflict_prompt(context_block, test_question, test_lang, mode):
    if args.mode == 'generated_new_knowledge':
        return (
            "Here are some question-and-answer pairs set in a future world that is very different from today:\n\n"
            f"{context_block}\n\n"
            f"Based on the knowledge above, answer the following question in {test_lang}:\n\n"
            f"<question>\n{test_question}\n</question>\n\n"
            "Your output must strictly follow this format:\n"
            "Output: <answer>Your answer here</answer>\n\n"
            "Do not include any explanation or text outside the <answer> tags.\n"
            "Output: "
        )
    elif args.mode == 'real_new_knowledge':
        return (
            "Here are some question-and-answer pairs about important medical knowledge:\n\n"
            f"{context_block}\n\n"
            f"Based on the knowledge above, answer the following question in {test_lang}:\n\n"
            f"<question>\n{test_question}\n</question>\n\n"
            "Your output must strictly follow this format:\n"
            "Output: <answer>Your answer here</answer>\n\n"
            "Do not include any explanation or text outside the <answer> tags.\n"
            "Output: "
        )

# ----------------------------
# VLLM Backend
# ----------------------------
class VLLMBackend:
    def __init__(self, model_path, gpu_mem_util, max_tokens, max_model_len):
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)
        self.params = SamplingParams(max_tokens=max_tokens)

    def generate(self, prompts):
        return [o.outputs[0].text.strip() for o in self.llm.generate(prompts, self.params)]

# ----------------------------
# Run
# ----------------------------
def run_conflict_vllm(data, conflict_pairs, all_langs, backend, save_root, tokenizer, max_attempts, dry_run, mode):
    all_prompts = []
    for triple in tqdm(conflict_pairs, desc="Conflict Pairs"):
        cat, lang_a, lang_b = triple
        pair_key = f"{cat}-{lang_a}-{lang_b}"
        save_dir = os.path.join(save_root, pair_key)
        safe_mkdir(save_dir)

        test_langs = [l for l in all_langs if l not in [lang_a, lang_b]]

        for test_lang in tqdm(test_langs, desc=f"→ {pair_key}", leave=False):
            save_path = os.path.join(save_dir, f"{test_lang}.json")
            if os.path.exists(save_path):
                print(f"[Skip] Already exists: {pair_key} → {test_lang}")
                continue
            prompts = []
            jobs = []
            for item in data:
                ctx = build_conflict_context(data, lang_a, lang_b, item)
                question = item["data"][test_lang]["question_2"]
                truth = item["data"][test_lang]["answer"]
                conflict_truth = item["data"][test_lang]["conflict_answer"]
                prompt = build_conflict_prompt(ctx, question, test_lang, mode)
                prompts.append(prompt)
                jobs.append({
                    "id": item["id"],
                    "context_langs": [lang_a, lang_b],
                    "query_lang": test_lang,
                    "question": question,
                    "truth": truth,
                    "conflict_truth": conflict_truth
                })
            all_prompts.extend(prompts)

            if dry_run:
                print(f"[Dry Run] Skipping request for {pair_key} → {test_lang}")
                continue

            outputs = backend.generate(prompts)
            answers = [extract_answer_tag(o) for o in outputs]

            for attempt in range(1, max_attempts):
                retry_idxs = [i for i, a in enumerate(answers) if a is None]
                if not retry_idxs:
                    break
                retry_prompts = [prompts[i] for i in retry_idxs]
                retry_outs = backend.generate(retry_prompts)
                for k, idx in enumerate(retry_idxs):
                    ex = extract_answer_tag(retry_outs[k])
                    if ex is not None:
                        answers[idx] = ex
                        outputs[idx] = retry_outs[k]

            results = []
            for j, out, ans in zip(jobs, outputs, answers):
                results.append({
                    **j,
                    "answer": ans or out
                })

            save_json(results, save_path)

            total = len(jobs)
            n_failed = sum(a is None for a in answers)
            n_ok = total - n_failed
            print(f"[{pair_key} → {test_lang}] Parsed OK: {n_ok}/{total} ({n_ok/total:.2%}); Failed: {n_failed}/{total} ({n_failed/total:.2%}).")

    prompt_statistics(all_prompts, tokenizer=tokenizer)

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str)
    ap.add_argument("--conflict_path", type=str)
    ap.add_argument("--save_root", type=str)
    ap.add_argument("--vllm_model", type=str)
    ap.add_argument("--mode",type=str)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--max_attempts", type=int, default=50)
    ap.add_argument("--info_path", type=str, default="../../utils/info.json")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data = load_json(args.data_path)
    conflict_pairs = load_json(args.conflict_path)
    info = load_json(args.info_path)
    languages = list(info["languages"].keys())

    tokenizer = AutoTokenizer.from_pretrained(args.vllm_model)
    backend = VLLMBackend(args.vllm_model, args.gpu_mem_util, args.max_tokens, args.max_model_len)

    run_conflict_vllm(data, conflict_pairs, languages, backend, args.save_root, tokenizer, args.max_attempts, args.dry_run, args.mode)