# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Function 1: Accuracy summary
# ----------------------------
def accuracy_table(root):
    lang_model_acc = defaultdict(dict)
    model_dirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

    for model in tqdm(model_dirs, desc="Models"):
        model_path = os.path.join(root, model)
        json_files = [f for f in sorted(os.listdir(model_path)) if f.endswith(".json")]

        for jf in json_files:
            lang = os.path.splitext(jf)[0]
            data = load_json(os.path.join(model_path, jf))

            correct = 0
            total = 0
            for item in data:
                jt = item["judge_truth"]["label"]
                if jt == "correct":
                    correct += 1
                total += 1
            if total > 0:
                lang_model_acc[lang][model] = correct / total

    rows = []
    for lang in sorted(lang_model_acc.keys()):
        row = {"lang": lang}
        for model in sorted(lang_model_acc[lang].keys()):
            row[model] = lang_model_acc[lang][model]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("lang")
    df = df.sort_index(axis=0).sort_index(axis=1)
    out_path = os.path.join(root, "summary_accuracy_table.csv")
    df.to_csv(out_path, encoding="utf-8")
    print(f"✅ Saved accuracy table to {out_path}")
    print(f"📐 Shape: {df.shape}")

# ----------------------------
# Function 2: Quality summary
# ----------------------------
def quality_table(root, input_path):
    data = load_json(input_path)

    lang_sim_vals = defaultdict(list)
    lang_consist_vals = defaultdict(list)

    for item in tqdm(data, desc="Items"):
        lang_dict = item["data"]
        for lang, ldict in lang_dict.items():
            for k, v in ldict.items():
                if k.startswith("sim_"):
                    lang_sim_vals[lang].append(float(v))
                elif k.startswith("consist_"):
                    lang_consist_vals[lang].append(1.0 if v else 0.0)

    rows = []
    for lang in sorted(set(lang_sim_vals) | set(lang_consist_vals)):
        sim = sum(lang_sim_vals[lang]) / len(lang_sim_vals[lang]) if lang_sim_vals[lang] else None
        consist = sum(lang_consist_vals[lang]) / len(lang_consist_vals[lang]) if lang_consist_vals[lang] else None
        rows.append({"lang": lang, "Similarity": sim, "Consistency": consist})

    df = pd.DataFrame(rows)
    df = df.set_index("lang")
    df = df.sort_index(axis=0)
    out_path = os.path.join(root, "summary_quality_table.csv")
    df.to_csv(out_path, encoding="utf-8")
    print(f"✅ Saved quality table to {out_path}")
    print(f"📐 Shape: {df.shape}")

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str)
    ap.add_argument("--dataset_path", type=str)
    args = ap.parse_args()

    accuracy_table(args.result_dir)
    quality_table(args.result_dir, args.dataset_path)