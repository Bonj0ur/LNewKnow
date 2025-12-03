# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def build_dataset_per_lang(input_path, output_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_by_lang = defaultdict(list)
    test_by_lang = defaultdict(list)

    for item in tqdm(data, desc="Processing items"):
        for lang, lang_data in item["data"].items():
            q1 = lang_data["question_1"]
            incor_ans = lang_data["incor_answer"]
            train_by_lang[lang].append({
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a knowledgeable expert. Please answer the following question concisely in {lang}."
                    },
                    {
                        "role": "user",
                        "content": q1
                    },
                    {
                        "role": "assistant",
                        "content": incor_ans
                    }
                ]
            })

            q2 = lang_data["question_2"]
            cor_ans = lang_data["cor_answer"]
            test_by_lang[lang].append({
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a knowledgeable expert. Please answer the following question concisely in {lang}."
                    },
                    {
                        "role": "user",
                        "content": q2
                    },
                    {
                        "role": "assistant",
                        "content": cor_ans
                    }
                ]
            })
    
    for lang, records in train_by_lang.items():
        path = os.path.join(train_dir, f"chat_train_{lang}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    for lang, records in test_by_lang.items():
        path = os.path.join(test_dir, f"chat_test_{lang}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    build_dataset_per_lang(args.input, args.output_dir)