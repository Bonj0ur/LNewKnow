# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def save_neuron_ids_with_probs(selected_ids, activation_probs, langs, model_name):
    for lang_idx, neuron_ids in selected_ids.items():
        with open(f'outputs/neurons/top_neurons_with_probs_{langs[lang_idx]}_{model_name}.txt', 'w') as f:
            for l, n in neuron_ids:
                f.write(f"({l},{n}), {activation_probs[l, n, lang_idx].item()}\n")

def compute_overlap_matrix(lang_num, grouped):
    unique_langs = torch.unique(torch.tensor([i for i in range(lang_num)]))
    overlap_matrix = torch.zeros((len(unique_langs), len(unique_langs)), dtype=torch.int)
    for i, lang1 in enumerate(unique_langs):
        for j, lang2 in enumerate(unique_langs):
            indices1 = set(grouped[lang1.item()])
            indices2 = set(grouped[lang2.item()])
            overlap = len(indices1.intersection(indices2))
            overlap_matrix[i, j] = overlap
    return overlap_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--top_rate",type=float,default=0.01)    
    parser.add_argument("--langs", type=str, nargs="+", default=["en", "zh_CN", "ja", "fr", "es", "it", "pt", "sv", "ko", "da", "th", "hi", "ta", "mn", "cy", "sw", "tk", "gd", "zu"])
    args = parser.parse_args()

    n, over_zero = [], []
    for lang in args.langs:
        data_path = f"./outputs/activations/activation.{lang}.flores.{args.model_name}"
        data = torch.load(data_path,weights_only=True)
        n.append(int(data['n']))
        over_zero.append(data['over_zero'])
    
    n = torch.tensor(n)
    over_zero = torch.stack(over_zero, dim=-1)

    num_layers, intermediate_size, lang_num = over_zero.size()
    activation_probs = over_zero / n

    flattened_probs = activation_probs.view(-1, lang_num)
    top_prob_values = torch.kthvalue(flattened_probs, k=round(flattened_probs.size(0) * (1 - args.top_rate)), dim=0).values
    selected_positions = activation_probs >= top_prob_values.unsqueeze(0).unsqueeze(0)

    selected_ids = defaultdict(list)
    for lang_idx in range(lang_num):
        selected_neurons = torch.where(selected_positions[:, :, lang_idx] == 1)
        selected_ids[lang_idx] = list(zip(selected_neurons[0].tolist(), selected_neurons[1].tolist()))
    
    for lang_idx, neuron_ids in selected_ids.items():
        with open(f'outputs/neurons/top_neurons_{args.langs[lang_idx]}_{args.model_name}.txt', 'w') as f:
            f.writelines(f"({l}, {n})\n" for l, n in neuron_ids)

    save_neuron_ids_with_probs(selected_ids, activation_probs, args.langs, args.model_name)

    lang_indices = torch.arange(lang_num)
    combined = torch.stack(torch.where(selected_positions), dim=-1)
    grouped = defaultdict(list)

    for position in combined.tolist():
        layer_index, neuron_index, lang_index = position
        grouped[lang_index].append((layer_index, neuron_index))
    overlap_matrix = compute_overlap_matrix(lang_num,grouped)
    
    if hasattr(overlap_matrix, "detach"):
        overlap_matrix_np = overlap_matrix.detach().cpu().numpy()
    else:
        overlap_matrix_np = np.array(overlap_matrix)
    
    df = pd.DataFrame(overlap_matrix_np, index=args.langs, columns=args.langs)
    df.to_csv(f"./outputs/overlap_matrix_{args.model_name}.csv", encoding="utf-8")