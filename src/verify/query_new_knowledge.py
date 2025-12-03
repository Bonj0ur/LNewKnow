# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import time
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Jobs
# ----------------------------
def create_jobs(data, languages):
    jobs = []
    for item in data:
        for lang in languages:
            q = item["data"][lang]["question_2"]
            t = item["data"][lang]["answer"]
            ct = item["data"][lang]["conflict_answer"]
            jobs.append({
                "id": item["id"],
                "lang": lang,
                "question": q,
                "truth": t,
                "conflict_truth": ct
            })
    return jobs

# ----------------------------
# Prompt builders
# ----------------------------
BASE_FMT = (
    "You are a knowledgeable expert. Please answer the following question concisely in {lang}.\n"
    "\n"
    "<question>\n{question}\n</question>\n"
    "\n"
    "Your output must strictly follow this format:\n"
    "Output: <answer>Your answer here</answer>\n"
    "\n"
    "Do not include any explanation or text outside the <answer> tags.\n"
    "Output: "
)

def build_prompt(question, language):
    return BASE_FMT.format(lang=language, question=question)

TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

def extract_answer_tag(text):
    if not text:
        return None
    m = TAG_RE.search(text)
    return m.group(1).strip() if m else None

# ----------------------------
# vLLM Backend & Execution
# ----------------------------
class VLLMBackend:
    def __init__(self, model_path, gpu_mem_util, max_tokens, max_model_len):
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)
        self.params = SamplingParams(max_tokens=max_tokens)
    
    def generate_ans(self, prompts):
        outs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in outs]

def run_vllm(back, jobs, max_attempts, report=True):
    prompts = [build_prompt(j["question"], j["lang"]) for j in jobs]
    ans_raw = back.generate_ans(prompts)
    ans_extracted = [extract_answer_tag(ans) for ans in ans_raw]

    attempts = 1
    while attempts < max_attempts:
        retry_idxs = [i for i, ex in enumerate(ans_extracted) if ex is None]
        if not retry_idxs:
            break
        retry_prompts = [build_prompt(jobs[i]["question"], jobs[i]["lang"]) for i in retry_idxs]
        retry_outs = back.generate_ans(retry_prompts)
        for k, idx in enumerate(retry_idxs):
            ans_raw[idx] = retry_outs[k]
            ex = extract_answer_tag(retry_outs[k])
            if ex is not None:
                ans_extracted[idx] = ex
        attempts += 1

    if report:
        total = len(jobs)
        failed_idxs = [i for i, ex in enumerate(ans_extracted) if ex is None]
        n_failed = len(failed_idxs)
        n_ok = total - n_failed
        print(f"[run_vllm] Parsed OK: {n_ok}/{total} ({n_ok/total:.2%}); Failed: {n_failed}/{total} ({n_failed/total:.2%}).")
    
    results = []
    for i, job in enumerate(jobs):
        ans = ans_extracted[i] if ans_extracted[i] is not None else ans_raw[i]
        results.append({
            "id": job["id"],
            "lang": job["lang"],
            "question": job["question"],
            "truth": job["truth"],
            "conflict_truth": job["conflict_truth"],
            "answer": ans
        })
    return results

# ----------------------------
# OpenAI Backend & Execution
# ----------------------------
def do_with_retries(fn, max_attempts=10, base_delay=1.0, max_delay=30.0):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(min(base_delay * (2 ** attempt), max_delay))
    raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_err}")

class OpenAIBackend:
    def __init__(self, api_key, model, max_tokens):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate_ans(self, prompt):
        def _once():
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )
            return r.choices[0].message.content.strip()
        return do_with_retries(_once)

def run_openai(back, jobs, max_attempts, concurrency, report=True):
    ans_raw = [None] * len(jobs)
    ans_extracted = [None] * len(jobs)

    def work(i):
        job = jobs[i]
        text = None
        ex = None
        for attempt in range(max_attempts):
            prompt = build_prompt(job["question"], job["lang"])
            text = back.generate_ans(prompt)
            ex = extract_answer_tag(text)
            if ex is not None:
                break
        ans_raw[i] = text
        ans_extracted[i] = ex

    total = len(jobs)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex_pool:
        futs = [ex_pool.submit(work, i) for i in range(total)]
        for _ in tqdm(as_completed(futs), total=total, desc="Generation", unit="job", dynamic_ncols=True, mininterval=0.1):
            pass
    
    if report:
        total = len(jobs)
        failed_idxs = [i for i, ex in enumerate(ans_extracted) if ex is None]
        n_failed = len(failed_idxs)
        n_ok = total - n_failed
        print(f"[run_openai] Parsed OK: {n_ok}/{total} ({n_ok/total:.2%}); Failed: {n_failed}/{total} ({n_failed/total:.2%}).")
    
    results = []
    for i, job in enumerate(jobs):
        ans = ans_extracted[i] if ans_extracted[i] is not None else ans_raw[i]
        results.append({
            "id": job["id"],
            "lang": job["lang"],
            "question": job["question"],
            "truth": job["truth"],
            "conflict_truth": job["conflict_truth"],
            "answer": ans
        })
    return results

# ----------------------------
# Save
# ----------------------------
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def group_by_language(results):
    out = {}
    for r in results:
        out.setdefault(r["lang"], []).append(r)
    return out

def save_per_language(grouped, out_dir):
    safe_mkdir(out_dir)
    for lang, rows in grouped.items():
        with open(os.path.join(out_dir, f"{lang}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path",type=str)
    ap.add_argument("--save_dir",type=str)
    ap.add_argument("--backend",type=str,choices=["openai", "vllm"])
    # vLLM PARAMETERS
    ap.add_argument("--vllm_model",type=str,default="")
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    # OpenAI PARAMETERS
    ap.add_argument("--openai_model",type=str,default="")
    ap.add_argument("--concurrency", type=int, default=32)
    # BASIC PARAMETERS
    ap.add_argument("--max_attempts", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--info_path", type=str, default="../utils/info.json")
    args = ap.parse_args()

    data = load_json(args.input_path)
    info = load_json(args.info_path)
    languages = list(info["languages"].keys())
    jobs = create_jobs(data, languages)

    if args.backend == "vllm":
        backend = VLLMBackend(args.vllm_model, args.gpu_mem_util, args.max_tokens, args.max_model_len)
        results = run_vllm(backend, jobs, args.max_attempts)
    elif args.backend == "openai":
        backend = OpenAIBackend(info["apikey"], args.openai_model, args.max_tokens)
        results = run_openai(backend, jobs, args.max_attempts, args.concurrency)

    grouped = group_by_language(results)
    save_per_language(grouped, args.save_dir)