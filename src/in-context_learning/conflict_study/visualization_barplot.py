# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Main
# ----------------------------
def plot_summary(summary_path,lang_order):
    data = pd.read_csv(summary_path)
    higher_lang = data["higher_lang"].iloc[0]
    lower_lang = data["lower_lang"].iloc[0]
    type_value = data["type"].iloc[0]

    type_colors = {
        "high_low": "#9CB0C3",
        "mid_low": "#7C9D97",
        "high_mid": "#B6B6B6"
    }

    base_color = type_colors.get(type_value, "#CCCCCC")

    data["order"] = data["test_lang"].apply(lambda x: lang_order.index(x) if x in lang_order else 999)
    data = data.sort_values(by="order").drop(columns="order")

    test_langs = data["test_lang"].tolist()
    consistent_with_higher = data["higher_rate"].values
    consistent_with_lower = data["lower_rate"].values

    x_labels = test_langs

    fig, ax = plt.subplots(figsize=(15, 7))
    bar_width = 0.5

    ax.bar(
        x_labels,
        consistent_with_higher,
        color=base_color,
        width=bar_width,
        alpha=0.65,
        label=f"Consistent with the knowledge in {higher_lang}",
        edgecolor="black",
        linewidth=0.5
    )

    ax.bar(
        x_labels,
        consistent_with_lower,
        bottom=consistent_with_higher,
        color=base_color,
        width=bar_width,
        alpha=0.4,
        label=f"Consistent with the knowledge in {lower_lang}",
        edgecolor="black",
        linewidth=0.5
    )

    for i, (high, low) in enumerate(zip(consistent_with_higher, consistent_with_lower)):
        ax.text(i, high / 2, f"{high*100:.0f}%", va="center", ha="center", color="#333", fontsize=10)
        ax.text(i, high + low / 2, f"{low*100:.0f}%", va="center", ha="center", color="#333", fontsize=10)
    
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 5)])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks(x_labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
    ax.tick_params(axis="x", labelsize=14)

    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_edgecolor("black")
        ax.spines[side].set_linewidth(1)

    ax.set_xlabel("Languages (Query)", fontsize=18)
    ax.set_ylabel(f"{higher_lang} - {lower_lang} Preference (%)", fontsize=18)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, labelcolor="black", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray", linewidth=0.5)

    plt.tight_layout()

    save_path = summary_path.replace("_summary.csv", "_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

# ----------------------------
# Batch Processing
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = parser.parse_args()

    with open(args.info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
        lang_order = list(info["languages"].keys())

    root_dir = args.root_dir.rstrip("/")
    all_subfolders = [os.path.abspath(os.path.join(root_dir, f)) for f in os.listdir(root_dir) if os.path.isdir(os.path.abspath(os.path.join(root_dir, f)))]

    print(f"🔍 Found {len(all_subfolders)} subfolders in {root_dir}")

    for subfolder in all_subfolders:
        for fname in os.listdir(subfolder):
            if fname.endswith("_summary.csv"):
                # summary_path = os.path.join(subfolder, fname)
                summary_path = "\\\\?\\"+os.path.abspath(os.path.join(subfolder, fname))
                plot_summary(summary_path, lang_order)