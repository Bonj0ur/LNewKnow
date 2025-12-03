# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def do_with_retries(fn, max_attempts=10, base_delay=1.0, max_delay=30.0):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(min(base_delay * (2 ** attempt), max_delay))
    raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_err}")

# ----------------------------
# OpenAI Backend
# ----------------------------
class OpenAIBackend:
    def __init__(self, api_key, model, max_tokens):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    def generate(self, messages):
        def _once():
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            return r.choices[0].message.content.strip()
        return do_with_retries(_once)

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_batch(test_data, backend, lang, concurrency):
    results = []

    def work(i):
        try:
            sample = test_data[i]
            messages = sample["messages"]
            question = next((m["content"] for m in messages if m["role"] == "user"), "")
            truth = messages[-1]["content"]
            messages_for_gen = [m for m in messages if m["role"] in {"system", "user"}]
            answer = backend.generate(messages_for_gen)
            return (i, question, truth, answer, None)
        except Exception as e:
            return (i, "", "", "", str(e))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, i) for i in range(len(test_data))]
        for fut in tqdm(as_completed(futures), total=len(test_data), desc=f"Evaluating on {lang}", dynamic_ncols=True):
            i, q, t, a, err = fut.result()
            if err is None:
                results.append({
                    "id": i,
                    "lang": lang,
                    "question": q,
                    "truth": t,
                    "answer": a
                })
            else:
                print(f"[Error] index={i}, error={err}")
    return results

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--model_dict", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = parser.parse_args()

    info = load_json(args.info_path)
    models_dict = load_json(args.model_dict)
    all_languages = list(info["languages"].keys())

    for train_lang in all_languages:
        model_dict = models_dict[train_lang]
        final_epoch = max(int(k) for k in model_dict if k.isdigit())
        model_name = model_dict[str(final_epoch)]
        backend = OpenAIBackend(api_key=info["apikey"], model=model_name, max_tokens=args.max_tokens)

        print(f"\n==============================")
        print(f"🌐 Train Language: {train_lang} [epoch-{final_epoch}]")
        print(f"==============================")

        for target_lang in all_languages:
            if target_lang == train_lang:
                continue

            test_path = os.path.join(args.test_dir, f"chat_test_{target_lang}.jsonl")
            if not os.path.exists(test_path):
                print(f"  ⛔ Missing test set: {test_path}")
                continue

            save_path = os.path.join(args.output_dir, f"{train_lang}_to_{target_lang}.json")
            if os.path.exists(save_path):
                print(f"  ✅ Already evaluated: {save_path}")
                continue

            print(f"  🔁 Evaluating {train_lang} → {target_lang}")
            test_data = load_jsonl(test_path)
            results = evaluate_batch(test_data, backend, target_lang, args.concurrency)
            save_json(results, save_path)
            print(f"  💾 Saved to: {save_path}")