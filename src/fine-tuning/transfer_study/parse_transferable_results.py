# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
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

def extract_langs_from_filename(fname):
    match = re.match(r"(.*)_to_(.*)\.json", fname)
    if match:
        return match.group(1), match.group(2)
    return None, None

def find_latest_prediction_file(pred_dir):
    files = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
    epoch_nums = []
    for f in files:
        match_epoch = re.match(r"epoch-(\d+)\.json", f)
        match_ckpt = re.match(r"checkpoint-(\d+)\.json", f)
        if match_epoch:
            epoch_nums.append((int(match_epoch.group(1)), f))
        elif match_ckpt:
            epoch_nums.append((int(match_ckpt.group(1)), f))
    if epoch_nums:
        latest_file = max(epoch_nums, key=lambda x: x[0])[1]
        return os.path.join(pred_dir, latest_file)
    return None

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str)
    ap.add_argument("--model_name", type=str)
    ap.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = ap.parse_args()

    args.transfer_dir = os.path.join(args.result_dir, "transferable", args.model_name)
    args.training_dir = os.path.join(args.result_dir, "training_effective", args.model_name)
    args.save_path = os.path.join(args.result_dir, "transferable", f"{args.model_name}_accuracy_matrix.csv")

    correct_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))

    info = load_json(args.info_path)
    languages = list(info["languages"].keys())

    for fname in os.listdir(args.transfer_dir):
        if not fname.endswith(".json"):
            continue
        ctx, qry = extract_langs_from_filename(fname)
        if not ctx or not qry:
            continue
        path = os.path.join(args.transfer_dir, fname)
        data = load_json(path)
        for item in data:
            label = item["judge_truth"]["label"]
            total_counts[ctx][qry] += 1
            if label == "correct":
                correct_counts[ctx][qry] += 1
    
    for sub in os.listdir(args.training_dir):
        if not sub.startswith("eval_"):
            continue
        lang = sub.replace("eval_", "")
        pred_dir = os.path.join(args.training_dir, sub, "predictions")
        if not os.path.isdir(pred_dir):
            continue
        latest_file = find_latest_prediction_file(pred_dir)
        if latest_file is None:
            continue
        ctx = qry = lang
        data = load_json(latest_file)
        for item in data:
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
                acc = round(correct / total, 4)
                acc_matrix.loc[ctx, qry] = acc
            else:
                acc_matrix.loc[ctx, qry] = None
    
    acc_matrix.to_csv(args.save_path)
    print(f"[✓] Accuracy matrix saved to {args.save_path}")