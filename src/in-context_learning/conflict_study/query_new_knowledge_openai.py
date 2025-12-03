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

TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
def extract_answer_tag(text):
    if not text:
        return None
    m = TAG_RE.search(text)
    return m.group(1).strip() if m else None

# ----------------------------
# Backend + Retry
# ----------------------------
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
# Prompt Construction
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
    if mode == 'generated_new_knowledge':
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
    elif mode == 'real_new_knowledge':
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

def prompt_statistics(prompts, model_name=None):
    char_lens = [len(p) for p in prompts]
    print(f"[Prompt Chars] Max: {max(char_lens)}, Min: {min(char_lens)}, Avg: {mean(char_lens):.1f}")
    if model_name:
        enc = tiktoken.encoding_for_model(model_name)
        token_lens = [len(enc.encode(p)) for p in tqdm(prompts)]
        print(f"[Prompt Tokens] Max: {max(token_lens)}, Min: {min(token_lens)}, Avg: {mean(token_lens):.1f}, Total: {sum(token_lens)}")

# ----------------------------
# Run
# ----------------------------
def run_conflict_openai(data, conflict_pairs, all_langs, backend, save_root, model_name, concurrency, max_attempts, dry_run, mode):
    all_prompts = []
    for triple in tqdm(conflict_pairs, desc="Conflict Pairs"):
        cat, lang_a, lang_b = triple
        pair_key = f"{cat}-{lang_a}-{lang_b}"
        save_dir = os.path.join(save_root, pair_key)
        safe_mkdir(save_dir)

        test_langs = [l for l in all_langs if l not in [lang_a, lang_b]]

        for test_lang in tqdm(test_langs, desc=f"→ {pair_key}", leave=False):
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

            answers_raw = [None] * len(jobs)
            answers_extracted = [None] * len(jobs)

            def work(i):
                for _ in range(max_attempts):
                    out = backend.generate(prompts[i])
                    answers_raw[i] = out
                    ex = extract_answer_tag(out)
                    if ex is not None:
                        answers_extracted[i] = ex
                        break
                    time.sleep(1.0)

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(tqdm(pool.map(work, range(len(jobs))), total=len(jobs), desc=f"{test_lang}"))
            
            results = []
            for i, job in enumerate(jobs):
                results.append({
                    **job,
                    "answer": answers_extracted[i] if answers_extracted[i] is not None else answers_raw[i]
                })

            save_json(results, os.path.join(save_dir, f"{test_lang}.json"))
    
    prompt_statistics(all_prompts, model_name)

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str)
    ap.add_argument("--conflict_path", type=str)
    ap.add_argument("--save_root", type=str)
    ap.add_argument("--mode",type=str)
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max_attempts", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--info_path", type=str, default="../../utils/info.json")
    ap.add_argument("--dry_run", action="store_true", help="Only show prompt statistics without inference")
    args = ap.parse_args()

    data = load_json(args.data_path)
    conflict_pairs = load_json(args.conflict_path)
    info = load_json(args.info_path)
    languages = list(info["languages"].keys())

    backend = OpenAIBackend(info["apikey"], args.openai_model, args.max_tokens)
    run_conflict_openai(data, conflict_pairs, languages, backend, args.save_root, args.openai_model, args.concurrency, args.max_attempts, args.dry_run, args.mode)