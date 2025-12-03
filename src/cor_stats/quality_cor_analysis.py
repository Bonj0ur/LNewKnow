# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -----------------------------
# Data preprocessing and merging
# -----------------------------
def preprocess_and_merge(language_path, accuracy_df, setting, mode_name, model_name):
    language_df = pd.read_csv(language_path)
    acc_df = accuracy_df.copy()
    acc_df = acc_df[["language",f"{setting.split('|')[0]}_{mode_name}_{model_name}_{setting.split('|')[1]}"]]
    acc_df = acc_df.rename(columns={f"{setting.split('|')[0]}_{mode_name}_{model_name}_{setting.split('|')[1]}": "performance"})
    merged = acc_df.merge(language_df, on=["language"], how="left")
    merged["log_data_prop"] = np.log(merged["data_proportion"])
    return merged

# -----------------------------
# Analysis (Spearman correlation)
# -----------------------------
def cor_analysis(merged_df, setting, mode_name, model_name, output_path="results/quality"):
    feature_cols = [
        "log_data_prop", "average_rank", "renyi_efficiency", "token_count",
        "morphscore_recall_f0_tok0_morpheme0", "morphscore_precision_f0_tok0_morpheme0",
        "morphscore_recall_f0_tok1_morpheme1", "morphscore_precision_f0_tok1_morpheme1",
        "morphscore_recall_f1_tok0_morpheme0", "morphscore_precision_f1_tok0_morpheme0",
        "morphscore_recall_f1_tok1_morpheme1", "morphscore_precision_f1_tok1_morpheme1"
    ]

    print(f"\n=== Spearman correlation results for {setting} | {mode_name} | {model_name} ===\n")

    results = []
    for feat in feature_cols:
        if feat not in merged_df.columns:
            continue
        feature_df = merged_df[["language", "performance", feat]].dropna()

        if feat.startswith("morphscore"):
            num_sample_col = feat.replace("recall", "num_samples").replace("precision", "num_samples")
            if num_sample_col in merged_df.columns:
                valid_langs = merged_df.loc[merged_df[num_sample_col] >= 100, "language"]
                feature_df = feature_df[feature_df["language"].isin(valid_langs)]
        
        r, p = spearmanr(feature_df["performance"], feature_df[feat])
        print(f"[{feat}] n = {feature_df.shape[0]:3d} samples  r = {r:+.3f}, p = {p:.4f}")

        results.append({
            "setting": setting,
            "mode": mode_name,
            "model": model_name,
            "feature": feat,
            "n_samples": feature_df.shape[0],
            "spearman_r": r,
            "p_value": p
        })
    
    output_dir = os.path.join(output_path)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"spearman_summary_{setting}_{mode_name}_{model_name}.csv"
    output_file = os.path.join(output_dir, filename)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n[✓] Saved Spearman summary to: {output_file}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting",type=str,choices=["fine-tuning|effective_performance", "fine-tuning|robust_performance", "in-context_learning|robust_performance"])
    parser.add_argument("--mode_name",type=str,choices=["generated", "real"])
    parser.add_argument("--model_name",type=str,choices=["Aya-Expanse-8B", "GPT-4o-Mini-2024-07-18", "Llama-3.1-8B-Instruct", "Qwen3-8B"])
    parser.add_argument("--accuracy_table", type=str, default="./datasets/quality_results.csv")
    args = parser.parse_args()

    args.language_features = f"./datasets/per_lang_stats_{args.model_name}.csv"
    accuracy_df = pd.read_csv(args.accuracy_table)

    merged_df = preprocess_and_merge(args.language_features, accuracy_df, setting=args.setting, mode_name=args.mode_name, model_name=args.model_name)
    cor_analysis(merged_df,setting=args.setting, mode_name=args.mode_name, model_name=args.model_name)