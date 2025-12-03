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

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str)
    ap.add_argument("--model_name", type=str)
    ap.add_argument("--info_path", type=str, default="../utils/info.json")
    args = ap.parse_args()

    args.save_path = os.path.join(args.result_dir,f"{args.model_name}_accuracy_matrix.csv")
    args.result_dir = os.path.join(args.result_dir,args.model_name)

    info = load_json(args.info_path)
    languages = list(info["languages"].keys())

    correct_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))

    for file in os.listdir(args.result_dir):
        if not file.endswith(".json"):
            continue
        if file == "selected_langs.json":
            continue
        path = os.path.join(args.result_dir, file)
        data = load_json(path)

        for item in data:
            ctx = item["ctx_lang"]
            qry = item["query_lang"]
            label = item["judge_truth"]["label"]
            total_counts[ctx][qry] += 1
            if label == "correct":
                correct_counts[ctx][qry] += 1
    
    acc_matrix = pd.DataFrame(index=languages, columns=languages)
    for ctx in languages:
        for qry in languages:
            total = total_counts[ctx][qry]
            correct = correct_counts[ctx][qry]
            if total > 0:
                acc = correct / total
                acc_matrix.loc[ctx, qry] = round(acc, 4)
            else:
                acc_matrix.loc[ctx, qry] = None

    acc_matrix.to_csv(args.save_path)