# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu

# ----------------------------
# Heatmap function
# ----------------------------
def plot_heatmap(data, save_path, title="Accuracy Difference (Manipulated - Baseline)"):
    plt.figure(figsize=(14, 12))
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=data.min().min()*100, vmax=data.max().max()*100)
    heatmap = sns.heatmap(
        data * 100, annot=True, fmt='.1f', cmap='vlag', cbar=True,
        linewidths=0.75, linecolor='#444', annot_kws={'size': 12}, square=True,
        cbar_kws={'label': 'Δ Accuracy (%)'}, norm=norm
    )

    ax = plt.gca()
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#444")
        ax.spines[spine].set_linewidth(0.75)

    plt.xlabel('Languages (Query)', fontsize=18)
    plt.ylabel('Languages (In-context learning)', fontsize=18)
    plt.xticks(fontsize=13, rotation=60, color="#444")
    plt.yticks(fontsize=13, color="#444")

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, width=1.5, color='gray')
    cbar.set_label('Δ Accuracy (%)', fontsize=18)
    cbar.outline.set_edgecolor('#444')
    cbar.outline.set_linewidth(1.5)

    plt.title(title, fontsize=20, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--baseline_dir", type=str)
    parser.add_argument("--manipulation_dir", type=str)
    args = parser.parse_args()

    baseline_path = os.path.join(args.baseline_dir,f"{args.model_name}_accuracy_matrix.csv")
    manipulation_path = os.path.join(args.manipulation_dir,args.mode,f"{args.model_name}_accuracy_matrix.csv")
    selected_langs_path = os.path.join(args.manipulation_dir,args.mode,args.model_name,"selected_langs.json")

    baseline = pd.read_csv(baseline_path, index_col=0)
    manipulated = pd.read_csv(manipulation_path, index_col=0)

    diff = manipulated - baseline
    diff_path = os.path.join(args.manipulation_dir, args.mode, f"{args.model_name}_diff_matrix.csv")
    diff.to_csv(diff_path)

    heatmap_path = os.path.join(args.manipulation_dir, args.mode, f"{args.model_name}_diff_heatmap.png")
    plot_heatmap(diff, heatmap_path)

    with open(selected_langs_path, "r") as f:
        selected_langs = set(json.load(f))

    records = []
    langs = diff.index.tolist()

    for i in langs:
        for j in langs:
            if i == j:
                continue
            val = diff.loc[i, j]
            in_i = i in selected_langs
            in_j = j in selected_langs
            if in_i and in_j:
                group = "Both in set"
            elif in_i or in_j:
                continue
            else:
                group = "None in set"
            records.append({"pair": f"{i}-{j}", "value": val, "group": group})
    
    df_long = pd.DataFrame(records)
    grouped_path = os.path.join(args.manipulation_dir, args.mode, f"{args.model_name}_diff_grouped.csv")
    df_long.to_csv(grouped_path, index=False)

    plt.figure(figsize=(8, 6))
    order = ["None in set", "Both in set"] 
    sns.boxplot(data=df_long, x="group", y="value", palette="Set2", order=order)
    sns.stripplot(data=df_long, x="group", y="value", color="black", alpha=0.3, order=order)
    plt.title("Distribution of Δ Accuracy by Group", fontsize=18)
    plt.ylabel("Δ Accuracy", fontsize=15)
    plt.xlabel("Group", fontsize=15)
    plt.tight_layout()
    boxplot_path = os.path.join(args.manipulation_dir, args.mode, f"{args.model_name}_diff_boxplot.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()

    both_vals = df_long[df_long["group"] == "Both in set"]["value"].values
    none_vals = df_long[df_long["group"] == "None in set"]["value"].values

    stat, p = mannwhitneyu(both_vals, none_vals, alternative="two-sided")
    print("\nMann-Whitney U test:")
    print(f"U = {stat:.4f}, p = {p:.4e}")