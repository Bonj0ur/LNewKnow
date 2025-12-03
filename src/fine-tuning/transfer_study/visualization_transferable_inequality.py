# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = parser.parse_args()

    args.matrix_path = os.path.join(args.result_dir, f"{args.model_name}_accuracy_matrix.csv")
    args.save_path = os.path.join(args.result_dir, f"{args.model_name}_inequality.png")
    args.output_csv = os.path.join(args.result_dir, f"{args.model_name}_inequality.csv")

    info = load_json(args.info_path)
    lang_info = info["languages"]

    df = pd.read_csv(args.matrix_path, index_col=0)

    resource_groups = {"high": [], "medium": [], "low": []}
    for lang, meta in lang_info.items():
        group = meta["resource"]
        resource_groups[group].append(lang)

    results = []
    for ft_lang in df.index:
        row = {"Languages (Fine-tuning)": ft_lang}
        for group, group_langs in resource_groups.items():
            targets = [t for t in group_langs if t != ft_lang and t in df.columns]
            if targets:
                avg = df.loc[ft_lang, targets].mean() * 100
            else:
                avg = np.nan
            row[f"To {group.capitalize()}"] = round(avg, 2)
        results.append(row)

    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output_csv, index=False)

    labels = result_df["Languages (Fine-tuning)"]
    x = np.arange(len(labels)) * 1.2
    width = 0.3

    plt.figure(figsize=(24, 7))

    bars1 = plt.bar(
        x - width, result_df["To High"], width,
        label='High-resource languages',
        color='#D3D7DC',
        edgecolor='#333', linewidth=0.5, alpha=0.9
    )
    bars2 = plt.bar(
        x, result_df["To Medium"], width,
        label='Medium-resource languages',
        color='#C4CFD7',
        edgecolor='#333', linewidth=0.5, alpha=0.9
    )
    bars3 = plt.bar(
        x + width, result_df["To Low"], width,
        label='Low-resource languages',
        color='#B5C7D1',
        edgecolor='#333', linewidth=0.5, alpha=0.9
    )

    plt.xlabel('Languages (Fine-tuning)', fontsize=18)
    plt.ylabel('Average accuracy (%)', fontsize=18)
    plt.xticks(x, labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=14, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.save_path,dpi=300)
    plt.close()