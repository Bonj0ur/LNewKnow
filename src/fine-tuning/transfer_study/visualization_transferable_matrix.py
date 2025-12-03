# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    args.matrix_path = os.path.join(args.result_dir,f"{args.model_name}_accuracy_matrix.csv")
    args.save_path = os.path.join(args.result_dir,f"{args.model_name}_accuracy_matrix.png")

    data = pd.read_csv(args.matrix_path,index_col=0)
    plt.figure(figsize=(14, 12))
    norm = mcolors.PowerNorm(gamma=0.7, vmin=(data*100).min().min(), vmax=(data*100).max().max())
    heatmap = sns.heatmap(
        data * 100, annot=True, fmt='.1f', cmap='vlag', cbar=True, 
        linewidths=0.75, linecolor='#444', annot_kws={'size': 12}, square=True,
        cbar_kws={'label': 'Accuracy (%)'}, norm=norm
    )

    ax = plt.gca()
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#444")
        ax.spines[spine].set_linewidth(0.75)

    plt.xlabel('Languages (Query)', fontsize=18)
    plt.ylabel('Languages (Fine-tuning)', fontsize=18)
    plt.xticks(fontsize=13, rotation=60, color="#444")
    plt.yticks(fontsize=13, color="#444")
    cbar = heatmap.collections[0].colorbar
    vmin, vmax = norm.vmin, norm.vmax
    tick_min = int(np.ceil(vmin / 15.0) * 15)
    tick_max = int(np.floor(vmax / 15.0) * 15)
    ticks = np.arange(tick_min, tick_max + 1, 15)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    cbar.ax.tick_params(labelsize=14,width=1.5, color='gray')
    cbar.set_label('Accuracy (%)', fontsize=18)
    cbar.outline.set_edgecolor('#444')
    cbar.outline.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(args.save_path,dpi=300)