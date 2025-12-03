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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Fine-tuning
# ----------------------------
def run_finetune(client, training_file_id, model_id, num_epochs, suffix, lang, start_epoch):
    while True:
        try:
            job = client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model_id,
                hyperparameters={"n_epochs": num_epochs},
                suffix=f"{suffix}_{lang}_{start_epoch}"
            )
            print(f"[{lang}][Epoch {start_epoch}] Job created: {job.id}")
            return job
        except Exception as e:
            print(f"[{lang}][Epoch {start_epoch}] Retry after error: {e}")
            time.sleep(180)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int)
    parser.add_argument("--model_record", type=str)
    parser.add_argument("--dataset_record", type=str)
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()

    info = load_json(args.info_path)
    languages = list(info["languages"].keys())
    api_key = info["apikey"]
    client = OpenAI(api_key=api_key)

    dataset_record = load_json(args.dataset_record)
    model_record = load_json(args.model_record)

    for lang in languages:
        print(f"\n=== Language: {lang} ===")
        training_file_id = dataset_record.get(f"chat_train_{lang}.jsonl")
        model_id = model_record[lang].get(str(args.start_epoch))
        if training_file_id is None or model_id is None:
            print(f"[{lang}] Missing training file or model ID. Skipping...")
            continue

        job = run_finetune(client, training_file_id, model_id, args.num_epochs, args.suffix, lang, args.start_epoch)
        print(f"[{lang}] Fine-tuning job started with ID: {job.id}")

if __name__ == "__main__":
    main()