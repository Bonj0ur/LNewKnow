# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import pandas as pd
from functools import reduce

def load_effective_performance(result_dir, mode, model):
    path = os.path.join(result_dir, "fine-tuning", "transfer_study", f"{mode}_new_knowledge", "training_effective", f"{model}_accuracy_curve.csv")
    if not os.path.exists(path):
        print(f"[Skip] Missing: {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    return pd.DataFrame({
        "language": df.index,
        f"fine-tuning_{mode}_{model}_effective_performance": df.iloc[:, -1].values
    })

def load_robust_performance(result_dir, method, mode, model):
    path = os.path.join(result_dir, method, "resist_study", f"{mode}_general_knowledge", f"{model}_accuracy_curve.csv")
    baseline_path = os.path.join(result_dir, "verify", f"{mode}_general_knowledge", "summary_accuracy_table.csv")
    if not os.path.exists(path) or not os.path.exists(baseline_path):
        print(f"[Skip] Missing: {path} or {baseline_path}")
        return None
    df_data = pd.read_csv(path, index_col=0)
    df_base = pd.read_csv(baseline_path, index_col=0)
    if method == "fine-tuning":
        values = df_data.mean(axis=1)
        results = {
            lang: values[lang]
            for lang in values.index.intersection(df_base.index)
            if model in df_base.columns
        }
    else:
        if "accuracy" not in df_data.columns:
            print(f"[Skip] No 'accuracy' column in {path}")
            return None
        values = df_data["accuracy"]
        results = {
            lang: values[lang] / df_base.loc[lang, model]
            for lang in values.index.intersection(df_base.index)
            if model in df_base.columns
        }
    return pd.DataFrame({
        "language": list(results.keys()),
        f"{method}_{mode}_{model}_robust_performance": list(results.values())
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--modes", nargs="+", default=["generated", "real"])
    parser.add_argument("--models", nargs="+", default=["Aya-Expanse-8B", "GPT-4o-Mini-2024-07-18", "Llama-3.1-8B-Instruct", "Qwen3-8B"])
    parser.add_argument("--save_path", type=str, default="./datasets/quality_results.csv")
    args = parser.parse_args()

    all_dfs = []

    for mode in args.modes:
        for model in args.models:
            # Effective performance (only for fine-tuning)
            df_eff = load_effective_performance(args.result_dir, mode, model)
            if df_eff is not None:
                all_dfs.append(df_eff)
            
            # Robust performance (fine-tuning)
            df_robust_ft = load_robust_performance(args.result_dir, "fine-tuning", mode, model)
            if df_robust_ft is not None:
                all_dfs.append(df_robust_ft)
            
            # Robust performance (in-context_learning)
            df_robust_icl = load_robust_performance(args.result_dir, "in-context_learning", mode, model)
            if df_robust_icl is not None:
                all_dfs.append(df_robust_icl)
    
    if not all_dfs:
        raise RuntimeError("❌ No valid results found. Please check your input paths.")

    merged_df = reduce(lambda left, right: pd.merge(left, right, on="language", how="outer"), all_dfs)
    merged_df.to_csv(args.save_path, index=False)
    print(f"✅ Merged quality results saved to: {args.save_path}")