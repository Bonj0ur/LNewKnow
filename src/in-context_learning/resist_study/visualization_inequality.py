# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir",type=str)
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--baseline_path",type=str)
    args = parser.parse_args()

    args.save_path = os.path.join(args.result_dir, f"{args.model_name}_accuracy_barplot.png")
    results_path = os.path.join(args.result_dir,f"{args.model_name}_accuracy_curve.csv")
    results_df = pd.read_csv(results_path, index_col=0)
    results_df.columns = range(1, len(results_df.columns) + 1)
    results_df = results_df.iloc[:, :] * 100

    baseline_df = pd.read_csv(args.baseline_path)
    baseline_df = baseline_df.set_index("lang")
    baseline_df = baseline_df[args.model_name] * 100
    baseline_df = baseline_df.to_frame(name=0)
    results_df = pd.concat([results_df, baseline_df], axis=1)

    labels = results_df.index.to_list()
    original_acc = results_df[0].to_list()
    resist_acc = results_df[1].tolist()
    resist_acc = [resist_acc[i]/original_acc[i] for i in range(len(resist_acc))]
    original_acc = [original_acc[i]/original_acc[i] for i in range(len(original_acc))]

    x = np.arange(len(labels))
    width = 0.3

    plt.figure(figsize=(16, 6))
    bars1 = plt.bar(x - width/2, original_acc, width, label='Query (w/o errors)', color=(210/255, 221/255, 227/255), alpha=1)
    bars2 = plt.bar(x + width/2, resist_acc, width, label='Query (w/ errors)', color=(181/255, 199/255, 211/255), alpha=1)

    plt.xlabel('Languages', fontsize=18)
    plt.ylabel('Relative Accuracy', fontsize=18)

    plt.xticks(x, labels, rotation=45, ha="center", fontsize=14)
    plt.ylim(0, 1)

    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.save_path,dpi=300)