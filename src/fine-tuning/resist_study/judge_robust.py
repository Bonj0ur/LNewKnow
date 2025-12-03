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

# ----------------------------
# Retry logic
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

# ----------------------------
# Judge prompt & schema
# ----------------------------
JUDGE_SYSTEM = (
    "Given a question, a model-generated answer, and a reference answer, compare the model-generated answer with the reference answer and determine whether the generated answer is correct. Ensure that the judgment is based on semantic alignment with the reference answer.\n"
    "Output the result in the following format:\n"
    '{"label":"correct|incorrect"}'
)

JUDGE_SCHEMA = {
    "name": "judge_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["correct", "incorrect"]},
        },
        "required": ["label"],
        "additionalProperties": False
    }
}

def build_judge_prompt(question, generated_answer, reference_answer):
    return (
        f"Question: {question}\n"
        f"Generated Answer: {generated_answer}\n"
        f"Reference Answer: {reference_answer}\n"
    )

# ----------------------------
# OpenAI backend (structured output)
# ----------------------------
class OpenAIBackend:
    def __init__(self, api_key, model, max_tokens):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate_ans(self, user_prompt):
        def _once():
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
                max_tokens=self.max_tokens
            )
            return r.choices[0].message.content.strip()
        return do_with_retries(_once)

# ----------------------------
# Core judging
# ----------------------------
def judge_item(backend, item):
    q = item["question"]
    t = item["truth"]
    ans = item["answer"]

    t_prompt = build_judge_prompt(q, ans, t)

    def call(pt):
        out = backend.generate_ans(pt)
        return json.loads(out)

    t_data = do_with_retries(lambda: call(t_prompt))
    return t_data

# ----------------------------
# File processor
# ----------------------------
def process_file(path, backend, concurrency):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idxs = list(range(len(data)))
    todo = [i for i in idxs if "judge_truth" not in data[i]]

    if not todo:
        return 0, 0

    ok = err = 0

    def work(i):
        try:
            t_data = judge_item(backend, data[i])
            return (i, t_data, None)
        except Exception as e:
            return (i, None, e)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(work, i) for i in todo]
        for fut in tqdm(as_completed(futs), total=len(todo), desc=os.path.basename(path), unit="item", dynamic_ncols=True):
            i, t_data, e = fut.result()
            if e is None:
                ok += 1
                data[i]["judge_truth"] = t_data
            else:
                err += 1
                print(f"[Error] index={i}, error={e}")

    with open(path, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=2)
    return ok, err

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str)
    ap.add_argument("--info_path", type=str, default="../../utils/info.json")
    ap.add_argument("--judge_model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--max_tokens", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    info = load_json(args.info_path)
    backend = OpenAIBackend(api_key=info["apikey"], model=args.judge_model, max_tokens=args.max_tokens)

    total_ok = total_err = 0
    all_json_files = []

    for root, dirs, files in os.walk(args.root_dir):
        for f in files:
            if f.endswith(".json"):
                all_json_files.append(os.path.join(root, f))
    
    print(f"\n=== Found {len(all_json_files)} JSON files under {args.root_dir} ===")

    for fp in all_json_files:
        ok, err = process_file(fp, backend, concurrency=args.concurrency)
        total_ok += ok
        total_err += err
        print(f"[{os.path.basename(fp)}] judged={ok}, failed={err}")

    print(f"\nAll done. Judged={total_ok}, Failed={total_err}")