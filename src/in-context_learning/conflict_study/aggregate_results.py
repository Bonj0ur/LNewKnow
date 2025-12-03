# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import pandas as pd

def summarize_model_folder(model_folder_path, output_path):
    all_train_folders = [
        f for f in os.listdir(model_folder_path)
        if os.path.isdir(os.path.join(model_folder_path, f))
    ]
    all_data = []

    for train_folder in all_train_folders:
        train_path = os.path.join(model_folder_path, train_folder)
        for fname in os.listdir(train_path):
            if fname.endswith("_summary.csv"):
                summary_path = os.path.join(train_path, fname)
                try:
                    df = pd.read_csv(summary_path)
                    tp = df["type"].iloc[0]
                    higher_lang = df["higher_lang"].iloc[0]
                    lower_lang = df["lower_lang"].iloc[0]
                    higher_mean = df["higher_rate"].mean()
                    lower_mean = df["lower_rate"].mean()
                    all_data.append({
                        "type": tp,
                        "conflict knowledge": f"{higher_lang}-{lower_lang}",
                        "higher_rate_mean": round(higher_mean, 3),
                        "lower_rate_mean": round(lower_mean, 3)
                    })
                except Exception as e:
                    print(f"❌ Failed to read {summary_path}: {e}")
                    continue
    
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(output_path, index=False)
    print(f"✅ Saved summary to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    model_folder_path = os.path.join(args.base_dir, args.model_name)
    output_path = os.path.join(model_folder_path, f"{args.model_name}_summary.csv")
    summarize_model_folder(model_folder_path, output_path)