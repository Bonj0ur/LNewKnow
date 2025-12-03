# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import pandas as pd

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir",type=str)
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--info_path",type=str,default="../../utils/info.json")
    args = parser.parse_args()

    args.save_path = os.path.join(args.result_dir, f"{args.model_name}_accuracy_curve.csv")
    args.result_dir = os.path.join(args.result_dir, args.model_name)

    info = load_json(args.info_path)
    languages = list(info["languages"].keys())

    accuracy = {}
    for lang in languages:
        data_path = os.path.join(args.result_dir,f"{lang}.json")
        data = load_json(data_path)
        correct = 0
        total = 0
        for item in data:
            if item["judge_truth"]["label"] == "correct":
                correct += 1
            total += 1
        acc = correct / total if total > 0 else None
        accuracy[lang] = round(acc, 4) if acc is not None else None
    
    acc_matrix = pd.DataFrame(index=languages, columns=["accuracy"])
    for lang in languages:
        acc_matrix.loc[lang, "accuracy"] = accuracy[lang]
    
    acc_matrix.to_csv(args.save_path)
    print(f"[✓] Saved accuracy matrix to: {args.save_path}")