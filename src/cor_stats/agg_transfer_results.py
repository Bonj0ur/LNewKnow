# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import pandas as pd
from functools import reduce

def load_transfer_result(path, method, mode, model):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist, skipping.")
        return None
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index().melt(id_vars="index", var_name="target", value_name=f"{method}_{mode}_{model}_accuracy")
    df = df.rename(columns={"index": "source"})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=["fine-tuning", "in-context_learning"])
    parser.add_argument("--modes", nargs="+", default=["generated_new_knowledge", "real_new_knowledge"])
    parser.add_argument("--models", nargs="+", default=["Aya-Expanse-8B", "GPT-4o-Mini-2024-07-18", "Llama-3.1-8B-Instruct", "Qwen3-8B"])
    parser.add_argument("--save_path", type=str, default="./datasets/transfer_results.csv")
    args = parser.parse_args()

    all_dfs = []

    for method in args.methods:
        for mode in args.modes:
            for model in args.models:
                path = os.path.join(args.result_dir, method, "transfer_study", mode, "transferable", f"{model}_accuracy_matrix.csv") if method == 'fine-tuning' else os.path.join(args.result_dir, method, "transfer_study", mode, f"{model}_accuracy_matrix.csv")
                df = load_transfer_result(path, method, mode, model)
                if df is not None:
                    all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid result files were loaded. Please check the paths.")

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=["source", "target"], how="outer"), all_dfs)
    merged_df.to_csv(args.save_path, index=False)
    print(f"Saved merged transfer results to: {args.save_path}")