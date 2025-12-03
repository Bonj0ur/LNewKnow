# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
import time
import random
import tiktoken
import argparse
from tqdm import tqdm
from openai import OpenAI
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

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def prompt_statistics(prompts, model_name=None, tokenizer=None):
    char_lens = [len(p) for p in prompts]
    print(f"[Prompt Chars] Max: {max(char_lens)}, Min: {min(char_lens)}, Avg: {mean(char_lens):.1f}")
    if tokenizer:
        token_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
        print(f"[Prompt Tokens] Max: {max(token_lens)}, Min: {min(token_lens)}, Avg: {mean(token_lens):.1f}, Total: {sum(token_lens)}")
    elif model_name:
        enc = tiktoken.encoding_for_model(model_name)
        token_lens = [len(enc.encode(p)) for p in prompts]
        print(f"[Prompt Tokens] Max: {max(token_lens)}, Min: {min(token_lens)}, Avg: {mean(token_lens):.1f}, Total: {sum(token_lens)}")
    else:
        print("[Prompt Tokens] Skipped (no tokenizer or model_name)")

def count_total_prompt_tokens(jobs_by_pair, tokenizer=None, mode=None, model_name=None):
    all_prompts = []
    for jobs in jobs_by_pair.values():
        all_prompts.extend(build_all_prompts(jobs,mode))

    print(f"[Prompt Count] Total prompts: {len(all_prompts)}")

    if tokenizer:
        token_lens = [len(tokenizer(p)["input_ids"]) for p in tqdm(all_prompts)]
    elif model_name:
        enc = tiktoken.encoding_for_model(model_name)
        token_lens = [len(enc.encode(p)) for p in tqdm(all_prompts)]
    else:
        print("[Prompt Tokens] Skipped (no tokenizer or model_name)")
        return
    
    print(f"[Prompt Tokens] Total: {sum(token_lens)}, Max: {max(token_lens)}, Min: {min(token_lens)}, Avg: {mean(token_lens):.1f}")

# ----------------------------
# Prompt builder
# ----------------------------
def build_context_block(data, lang, target_item, num_context=50):
    lines = []
    candidates = [item for item in data if item["id"] != target_item["id"]]
    sampled = random.sample(candidates, k=num_context - 1) if len(candidates) >= num_context - 1 else candidates
    target_q = target_item["data"][lang]["question_1"]
    target_a = target_item["data"][lang]["answer"]
    lines.append(f"Q: {target_q}\nA: {target_a}")

    for item in sampled:
        q = item["data"][lang]["question_1"]
        a = item["data"][lang]["answer"]
        lines.append(f"Q: {q}\nA: {a}")
    
    random.shuffle(lines)
    return "\n".join(lines)

def build_crosslingual_jobs(data, languages, num_context=50):
    jobs_by_pair = {}
    for ctx_lang in languages:
        for query_lang in languages:
            key = f"{ctx_lang}-{query_lang}"
            jobs = []
            for item in data:
                context_block = build_context_block(data, ctx_lang, item, num_context)
                q = item["data"][query_lang]["question_2"]
                ans = item["data"][query_lang]["answer"]
                jobs.append({
                    "id": item["id"],
                    "ctx_lang": ctx_lang,
                    "query_lang": query_lang,
                    "context": context_block,
                    "query_question": q,
                    "truth": ans
                })
            jobs_by_pair[key] = jobs
    return jobs_by_pair

def build_crosslingual_prompt(context_block, query_question, query_lang, mode):
    if mode == 'generated_new_knowledge':
        return (
            "Here are some question-and-answer pairs set in a future world that is very different from today:\n\n"
            f"{context_block}\n\n"
            f"Based on the knowledge above, answer the following question in {query_lang}:\n\n"
            f"<question>\n{query_question}\n</question>\n\n"
            "Your output must strictly follow this format:\n"
            "Output: <answer>Your answer here</answer>\n\n"
            "Do not include any explanation or text outside the <answer> tags.\n"
            "Output: "
        )
    elif mode == 'real_new_knowledge':
        return (
            "Here are some question-and-answer pairs about important medical knowledge:\n\n"
            f"{context_block}\n\n"
            f"Based on the knowledge above, answer the following question in {query_lang}:\n\n"
            f"<question>\n{query_question}\n</question>\n\n"
            "Your output must strictly follow this format:\n"
            "Output: <answer>Your answer here</answer>\n\n"
            "Do not include any explanation or text outside the <answer> tags.\n"
            "Output: "
        )

def build_all_prompts(jobs,mode):
    return [build_crosslingual_prompt(j["context"], j["query_question"], j["query_lang"], mode) for j in jobs]

TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
def extract_answer_tag(text):
    if not text:
        return None
    m = TAG_RE.search(text)
    return m.group(1).strip() if m else None

# ----------------------------
# Backends
# ----------------------------
class VLLMBackend:
    def __init__(self, model_path, gpu_mem_util, max_tokens, max_model_len):
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)
        self.params = SamplingParams(max_tokens=max_tokens)

    def generate(self, prompts):
        return [o.outputs[0].text.strip() for o in self.llm.generate(prompts, self.params)]

def do_with_retries(fn, max_attempts=10, base_delay=1.0, max_delay=30.0):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait = min(base_delay * (2 ** attempt), max_delay)
            print(f"[Retry] Attempt {attempt+1} failed: {e} | Retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_err}")

class OpenAIBackend:
    def __init__(self, api_key, model_name, max_tokens):
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.max_tokens = max_tokens

    def generate(self, prompt):
        def _call():
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )
            return r.choices[0].message.content.strip()
        return do_with_retries(_call)

# ----------------------------
# Execution Wrappers
# ----------------------------
def run_vllm(pair, jobs, backend, save_dir, tokenizer, max_attempts, mode):
    prompts = build_all_prompts(jobs,mode)
    prompt_statistics(prompts, tokenizer=tokenizer)

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
            "id": j["id"],
            "context_lang": j["ctx_lang"],
            "query_lang": j["query_lang"],
            "context": j["context"],
            "query_question": j["query_question"],
            "truth": j["truth"],
            "answer": ans or out
        })

    save_json(results, os.path.join(save_dir, f"{pair}.json"))

    total = len(jobs)
    n_failed = sum(a is None for a in answers)
    n_ok = total - n_failed
    print(f"[run_vllm:{pair}] Parsed OK: {n_ok}/{total} ({n_ok/total:.2%}); Failed: {n_failed}/{total} ({n_failed/total:.2%}).")

def run_openai(pair, jobs, backend, save_dir, model_name, concurrency, max_attempts, mode):
    prompts = build_all_prompts(jobs, mode)
    prompt_statistics(prompts, model_name=model_name)
    results = []
    ans_raw = [None] * len(jobs)
    ans_extracted = [None] * len(jobs)

    def work(i):
        for attempt in range(max_attempts):
            text = backend.generate(prompts[i])
            ans_raw[i] = text
            ex = extract_answer_tag(text)
            if ex is not None:
                ans_extracted[i] = ex
                break
            time.sleep(1.0)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(tqdm(pool.map(work, range(len(jobs))), total=len(jobs)))
    
    for i, job in enumerate(jobs):
        results.append({
            "id": job["id"],
            "context_lang": job["ctx_lang"],
            "query_lang": job["query_lang"],
            "context": job["context"],
            "query_question": job["query_question"],
            "truth": job["truth"],
            "answer": ans_extracted[i] if ans_extracted[i] is not None else ans_raw[i]
        })

    save_json(results, os.path.join(save_dir, f"{pair}.json"))

    total = len(jobs)
    n_failed = sum(1 for a in ans_extracted if a is None)
    n_ok = total - n_failed
    print(f"[run_openai:{pair}] Parsed OK: {n_ok}/{total} ({n_ok/total:.2%}); Failed: {n_failed}/{total} ({n_failed/total:.2%}).")

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str)
    ap.add_argument("--save_dir", type=str)
    ap.add_argument("--mode", type=str)
    ap.add_argument("--backend", type=str, choices=["openai", "vllm"])
    # vLLM PARAMETERS
    ap.add_argument("--vllm_model", type=str, default="")
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=8192)
    # OpenAI PARAMETERS
    ap.add_argument("--openai_model", type=str, default="")
    ap.add_argument("--concurrency", type=int, default=32)
    # BASIC PARAMETERS
    ap.add_argument("--max_attempts", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = ap.parse_args()

    data = load_json(args.input_path)
    info = load_json(args.info_path)
    languages = list(info["languages"].keys())
    safe_mkdir(args.save_dir)

    jobs_by_pair = build_crosslingual_jobs(data, languages)

    if args.backend == "vllm":
        tokenizer = AutoTokenizer.from_pretrained(args.vllm_model)
        count_total_prompt_tokens(jobs_by_pair, tokenizer=tokenizer, mode=args.mode)
        backend = VLLMBackend(args.vllm_model, args.gpu_mem_util, args.max_tokens, args.max_model_len)
        for pair, jobs in tqdm(jobs_by_pair.items(), desc="Language Pairs"):
            run_vllm(pair, jobs, backend, args.save_dir, tokenizer=tokenizer, max_attempts=args.max_attempts, mode=args.mode)
    else:
        count_total_prompt_tokens(jobs_by_pair, model_name=args.openai_model, mode=args.mode)
        backend = OpenAIBackend(info["apikey"], args.openai_model, args.max_tokens)
        for pair, jobs in tqdm(jobs_by_pair.items(), desc="Language Pairs"):
            run_openai(pair, jobs, backend, args.save_dir, args.openai_model, args.concurrency, args.max_attempts, mode=args.mode)