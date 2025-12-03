# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Utils (For Better Trend Analysis)
# ----------------------------
def moving_average(y, w=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    h = (w - 1) // 2
    sigma = w / 8.0
    idx = np.arange(-h, h + 1)
    base_w = np.exp(-0.5 * (idx / sigma) ** 2)
    base_w /= base_w.sum()

    out = np.empty(n, dtype=float)
    for i in range(n):
        L = max(0, i - h)
        R = min(n - 1, i + h)
        left_cut = h - (i - L)
        right_cut = h + (R - i)
        w_slice = base_w[left_cut:right_cut + 1]
        w_slice /= w_slice.sum()
        out[i] = np.dot(y[L:R + 1], w_slice)
    return out

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir",type=str)
    parser.add_argument("--model_name",type=str)
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
    
    plt.figure(figsize=(12, 8))
    for i, lang in enumerate(results_df.index):
        y = results_df.loc[lang].values
        x = results_df.columns
        y_smooth = moving_average(y)
        plt.plot(x, y_smooth, label=lang, color=[c/255 for c in colors[i]])
    plt.grid(True)
    plt.legend(title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=18)
    plt.xlabel("Epochs of fine-tuning", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(args.save_path,dpi=300)