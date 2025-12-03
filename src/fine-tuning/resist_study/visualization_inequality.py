# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir",type=str)
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--baseline_path",type=str)
    args = parser.parse_args()

    args.result_path = os.path.join(args.result_dir, f"{args.model_name}_accuracy_curve.csv")
    args.save_path = os.path.join(args.result_dir, f"{args.model_name}_accuracy_curve.png")

    colors = [
        (253, 123, 0),
        (253, 137, 0),
        (253, 151, 0),
        (254, 165, 0),
        (254, 179, 1),
        (254, 193, 2),
        (254, 203, 2),
        (255, 211, 3),
        (255, 215, 3),
        (255, 219, 4),
        (255, 219, 4),
        (255, 219, 4),
        (40, 120, 40),
        (65, 140, 65),
        (90, 160, 90),
        (115, 180, 115),
        (140, 200, 140),
        (165, 220, 165),
        (190, 235, 190)
    ]

    results_df = pd.read_csv(args.result_path, index_col=0)
    results_df.columns = range(1, len(results_df.columns) + 1)
    results_df = results_df.iloc[:, :] * 100

    baseline_df = pd.read_csv(args.baseline_path)
    baseline_df = baseline_df.set_index("lang")
    baseline_df = baseline_df[args.model_name] * 100
    baseline_df = baseline_df.to_frame(name=0)
    results_df = pd.concat([results_df, baseline_df], axis=1)
    results_df = results_df.reindex(sorted(results_df.columns, key=lambda x: int(x)), axis=1)

    plt.figure(figsize=(12, 8))
    for i, lang in enumerate(results_df.index):
        y = results_df.loc[lang].values
        x = results_df.columns
        plt.plot(x, y, label=lang, color=[c/255 for c in colors[i]])
    plt.grid(True)
    plt.legend(title="Languages", bbox_to_anchor=(1, 1), loc='upper right', fontsize=14, title_fontsize=18)
    plt.xlabel("Epochs of fine-tuning", fontsize=18)
    plt.ylabel("Relative Accuracy", fontsize=18)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))
    plt.ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(args.save_path,dpi=300)