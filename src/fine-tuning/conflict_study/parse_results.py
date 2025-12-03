# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import csv
import json
import argparse

# ----------------------------
# Utils
# ----------------------------
def is_correct(judge_block):
    return isinstance(judge_block, dict) and judge_block["label"] == "correct"

def extract_test_lang_from_filename(filename):
    return filename.split("_to_")[-1].replace(".json", "")

def parse_langs_from_folder(folder_name):
    pattern = r"train_([a-z]+(?:_[a-z]+)?)_(.+?)-(.+?)(?:_eps.*)?$"
    match = re.match(pattern, folder_name)
    if match:
        lang_type = match.group(1)
        lang1 = match.group(2)
        lang2 = match.group(3)
        return lang_type, lang1, lang2
    return "unknown", "high", "low"

def process_folder(folder_path, folder_name):
    results = []
    lang_type, lang1, lang2 = parse_langs_from_folder(folder_name)
    save_path = os.path.join(folder_path, f"{lang1}-{lang2}_summary.csv")

    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ JSON decode error: {fname}")
            continue

        high, low = 0, 0

        for record in data:
            conflict_judge = record["judge_conflict_truth"]
            truth_judge = record["judge_truth"]

            conflict_correct = is_correct(conflict_judge)
            truth_correct = is_correct(truth_judge)

            if conflict_correct and truth_correct:
                high += 0.5
                low += 0.5
            elif conflict_correct and not truth_correct:
                high += 1
            elif truth_correct and not conflict_correct:
                low += 1
        
        test_lang = extract_test_lang_from_filename(fname)
        total_votes = high + low

        results.append({
            "type": lang_type,
            "higher_lang": lang1,
            "lower_lang": lang2,
            "test_lang": test_lang,
            "higher_rate": round(high / total_votes, 2) if total_votes > 0 else 0.5,
            "lower_rate": round(low / total_votes, 2) if total_votes > 0 else 0.5
        })

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type", "higher_lang", "lower_lang", "test_lang",
            "higher_rate", "lower_rate"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"📄 Saved summary to: {save_path}")
    return results

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str)
    args = parser.parse_args()

    model_folder = args.model_folder.rstrip("/")
    all_subfolders = [f for f in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, f))]

    print(f"🔍 Found {len(all_subfolders)} subfolders in {model_folder}")

    for subfolder in all_subfolders:
        subfolder_path = os.path.join(model_folder, subfolder)
        print(f"\n📂 Processing {subfolder_path} ...")
        process_folder(subfolder_path, subfolder)