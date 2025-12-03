# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import pandas as pd
from collections import defaultdict

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_number(filename):
    try:
        return int(filename.replace("checkpoint-", "").replace("epoch-","").replace(".json", ""))
    except:
        return -1

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str)
    ap.add_argument("--model_name", type=str)
    ap.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = ap.parse_args()

    args.save_path = os.path.join(args.result_dir, f"{args.model_name}_accuracy_curve.csv")
    args.result_dir = os.path.join(args.result_dir, args.model_name)

    info = load_json(args.info_path)
    languages = list(info["languages"].keys())

    accuracy = defaultdict(dict)
    checkpoint_set = set()

    for lang in languages:
        lang_dir = os.path.join(args.result_dir, f"eval_{lang}", "predictions")
        if not os.path.isdir(lang_dir):
            print(f"[!] Missing: {lang_dir}")
            continue

        for file in os.listdir(lang_dir):
            if not file.endswith(".json"):
                continue

            ckpt = extract_number(file)
            if ckpt < 0:
                continue

            checkpoint_set.add(ckpt)
            path = os.path.join(lang_dir, file)
            data = load_json(path)

            correct = 0
            total = 0
            for item in data:
                if item["judge_truth"]["label"] == "correct":
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else None
            accuracy[lang][ckpt] = round(acc, 4) if acc is not None else None

    sorted_checkpoints = sorted(checkpoint_set)
    acc_matrix = pd.DataFrame(index=languages, columns=sorted_checkpoints)

    for lang in languages:
        for ckpt in sorted_checkpoints:
            acc_matrix.loc[lang, ckpt] = accuracy[lang][ckpt]

    acc_matrix.to_csv(args.save_path)
    print(f"[✓] Saved accuracy matrix to: {args.save_path}")