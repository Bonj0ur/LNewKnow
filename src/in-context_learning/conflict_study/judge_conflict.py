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

def save_json(obj, path):
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
# Judging Prompt Template
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
# OpenAI Judging Backend
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
# Judge Record (truth + conflict)
# ----------------------------
def judge_both(backend, item):
    q = item["question"]
    a = item["answer"]
    t1 = item["truth"]
    t2 = item["conflict_truth"]

    prompt_truth = build_judge_prompt(q, a, t1)
    prompt_conflict = build_judge_prompt(q, a, t2)

    res_truth = json.loads(backend.generate_ans(prompt_truth))
    res_conflict = json.loads(backend.generate_ans(prompt_conflict))

    return res_truth, res_conflict

# ----------------------------
# Judge File
# ----------------------------
def process_file(path, backend, concurrency):
    data = load_json(path)
    todo = [i for i, item in enumerate(data) if "judge_truth" not in item or "judge_conflict_truth" not in item]
    if not todo:
        return 0, 0

    ok = err = 0

    def work(i):
        try:
            j_truth, j_conflict = judge_both(backend, data[i])
            return i, j_truth, j_conflict, None
        except Exception as e:
            return i, None, None, e
    
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, i) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(todo), desc=os.path.basename(path), unit="item", dynamic_ncols=True):
            i, j_t, j_c, error = fut.result()
            if error is None:
                data[i]["judge_truth"] = j_t
                data[i]["judge_conflict_truth"] = j_c
                ok += 1
            else:
                print(f"[Error] {os.path.basename(path)} idx={i}: {error}")
                err += 1

    save_json(data, path)
    return ok, err

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    parser.add_argument("--judge_model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--max_tokens", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    info = load_json(args.info_path)
    backend = OpenAIBackend(api_key=info["apikey"], model=args.judge_model, max_tokens=args.max_tokens)

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.root_dir)
        for file in files
        if file.endswith(".json")
    ]

    print(f"\n=== Found {len(all_files)} files in {args.root_dir} ===")

    total_ok = total_err = 0
    for file_path in all_files:
        ok, err = process_file(file_path, backend, args.concurrency)
        total_ok += ok
        total_err += err
        print(f"✔ {os.path.basename(file_path)} — Judged: {ok}, Failed: {err}")

    print(f"\n✅ All done! Total Judged: {total_ok}, Total Failed: {total_err}")