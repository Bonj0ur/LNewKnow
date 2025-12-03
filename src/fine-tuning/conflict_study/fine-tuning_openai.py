# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import json
import argparse
from openai import OpenAI

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ----------------------------
# Launch Fine-tuning Job
# ----------------------------
def run_finetune(client, training_file_id, model_id, suffix):
    while True:
        try:
            job = client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model_id,
                hyperparameters={"n_epochs": 12},
                suffix=suffix
            )
            print(f"[{suffix}] ✅ Job created: {job.id}")
            return job
        except Exception as e:
            print(f"[{suffix}] ❌ Error: {e}. Retrying in 180s...")
            time.sleep(180)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--upload_record", type=str)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()

    info = load_json(args.info_path)
    upload_ids = load_json(args.upload_record)
    job_record = load_json(args.save_path)
    client = OpenAI(api_key=info["apikey"])

    for filename in sorted(os.listdir(args.train_dir)):
        if not filename.endswith(".jsonl"):
            continue

        if filename in job_record:
            print(f"🟡 Already fine-tuned: {filename}, skipping.")
            continue

        file_id = upload_ids.get(filename)
        if file_id is None:
            print(f"❌ Missing file_id for {filename}, skipping.")
            continue

        suffix = f"{args.suffix}_{filename.replace('train_', '').replace('.jsonl', '')}"
        job = run_finetune(client, file_id, args.base_model, suffix)

        job_record[filename] = job.id
        save_json(job_record, args.save_path)
        print(f"💾 Saved job ID for {filename}.")

    print("\n✅ All available jobs launched (or skipped if done).")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()