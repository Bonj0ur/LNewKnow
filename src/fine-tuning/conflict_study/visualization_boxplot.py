# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_box(ax, y, xpos, color, edgecolor='#555', box_width=0.18, lw=0.5, cap_frac=0.8, show_fliers=True, flier_ms=4.0, alpha=0.9, zorder=4):
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return
    q1, q2, q3 = np.percentile(y, [25, 50, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    y_low = np.min(y[y >= lo]) if np.any(y >= lo) else q1
    y_high = np.max(y[y <= hi]) if np.any(y <= hi) else q3
    outliers = y[(y < y_low) | (y > y_high)]

    ax.add_patch(plt.Rectangle((xpos - box_width/2, q1), box_width, q3 - q1, facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=zorder, alpha=alpha))
    ax.plot([xpos - box_width/2, xpos + box_width/2], [q2, q2], color=edgecolor, linewidth=lw, zorder=zorder+0.1)
    ax.plot([xpos, xpos], [q3, y_high], color=edgecolor, linewidth=lw, zorder=zorder)
    ax.plot([xpos, xpos], [q1, y_low], color=edgecolor, linewidth=lw, zorder=zorder)
    cap_w = cap_frac * box_width
    ax.plot([xpos - cap_w/2, xpos + cap_w/2], [y_high, y_high], color=edgecolor, linewidth=lw, zorder=zorder)
    ax.plot([xpos - cap_w/2, xpos + cap_w/2], [y_low,  y_low], color=edgecolor, linewidth=lw, zorder=zorder)

    if show_fliers and outliers.size:
        ax.plot(np.full_like(outliers, xpos), outliers, linestyle='', marker='o', markersize=flier_ms, markerfacecolor=color, markeredgecolor=edgecolor, alpha=0.5, zorder=zorder+0.2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--ylim", type=float, nargs=2, default=[25, 100])
    parser.add_argument("--yticks", type=int, nargs='+', default=[25, 50, 75, 100])
    parser.add_argument("--ylabel", type=str, default="Consistent with the knowledge in higher-resource languages (%)")
    parser.add_argument("--xlabel", type=str, default="Model")
    parser.add_argument("--legend_loc", type=str, default="upper left")
    args = parser.parse_args()

    type_order = ["high_low", "mid_low", "high_mid"]
    model_order = ["GPT-4o-Mini-2024-07-18", "Llama-3.1-8B-Instruct", "Qwen3-8B", "Aya-Expanse-8B"]
    palette = {"high_low": "#97AAC8", "mid_low": "#DCACAA", "high_mid": "#C4D7B2"}
    width_total = 0.85
    n_levels = len(type_order)
    box_width_each = width_total / n_levels * 0.7

    all_data = []
    for folder_name in os.listdir(args.root_dir):
        subdir = os.path.join(args.root_dir, folder_name)
        if not os.path.isdir(subdir):
            continue
        summary_files = [f for f in os.listdir(subdir) if f.endswith("_summary.csv")]
        if not summary_files:
            print(f"⚠️ No *_summary.csv in {subdir}")
            continue
        for file in summary_files:
            fpath = os.path.join(subdir, file)
            df = pd.read_csv(fpath)
            df["model"] = folder_name
            all_data.append(df)

    if not all_data:
        raise RuntimeError("No valid *_summary.csv files found in subdirectories.")

    df_all = pd.concat(all_data, ignore_index=True)
    df_all["higher_rate_mean"] = df_all["higher_rate_mean"] * 100
    df_all["type"] = pd.Categorical(df_all["type"], categories=type_order, ordered=True)

    fig, ax = plt.subplots(figsize=(17, 9))
    x_base = np.arange(len(model_order))
    idx_map = {t: i for i, t in enumerate(type_order)}
    offsets = {t: (width_total / n_levels) * (idx_map[t] - (n_levels - 1) / 2) for t in type_order}

    for m_idx, model in enumerate(model_order):
        for t in type_order:
            sub = df_all[(df_all["model"] == model) & (df_all["type"] == t)]
            if sub.empty:
                continue
            y = sub["higher_rate_mean"].values
            xpos = x_base[m_idx] + offsets[t]
            draw_box(ax, y, xpos, color=palette[t], box_width=box_width_each)

    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(args.ylim)
    ax.set_yticks(args.yticks)
    ax.set_ylabel(args.ylabel, fontsize=18)
    ax.set_xlabel(args.xlabel, fontsize=18)
    ax.set_xticks(x_base)
    ax.set_xticklabels(model_order, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    legend_handles = [
        mpatches.Patch(color=palette["high_low"], label="High - Low"),
        mpatches.Patch(color=palette["mid_low"], label="Mid - Low"),
        mpatches.Patch(color=palette["high_mid"], label="High - Mid")
    ]
    leg = ax.legend(
        handles=legend_handles,
        title="Knowledge Conflicts",
        loc=args.legend_loc,
        fontsize=14,
        title_fontsize=16,
        edgecolor="gray",
        labelcolor="#333",
        ncols=3
    )
    leg.get_title().set_color("#333")

    fig.tight_layout()
    fig.savefig(args.save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved to {args.save_path}")