# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import numpy as np
from collections import defaultdict, Counter

def count_alignments(alignments_line, tok_line):
    alignments = alignments_line.split(" ")
    src_tok, tgt_tok = tok_line.split(" ||| ")
    src_tok = src_tok.split(" ")
    tgt_tok = tgt_tok.split(" ")

    fwd = defaultdict(list)
    rev = defaultdict(list)
    count_nulls_src = 0
    count_nulls_tgt = 0
    count_onetoone = 0
    count_onetomany = 0
    count_manytoone = 0
    count_manytomany_src = 0
    count_manytomany_tgt = 0

    for pair in alignments:
        src, tgt = pair.split("-")
        src = int(src)
        tgt = int(tgt)
        fwd[src].append(tgt)
        rev[tgt].append(src)
    
    last_idx_src = len(src_tok)
    last_idx_tgt = len(tgt_tok)

    for i in range(last_idx_src):
        if not fwd[i]:
            count_nulls_src += 1
        elif len(fwd[i]) == 1 and len(rev[fwd[i][0]]) == 1:
            count_onetoone += 1
        elif len(fwd[i]) > 1:
            many = [0 if len(rev[j]) == 1 else 1 for j in fwd[i]]
            if any(many):
                count_manytomany_src += 1
            else:
                count_onetomany += 1
    
    for i in range(last_idx_tgt):
        if not rev[i]:
            count_nulls_tgt += 1
        elif len(rev[i]) > 1:
            many = [0 if len(fwd[j]) == 1 else 1 for j in rev[i]]
            if any(many):
                count_manytomany_tgt += 1
            else:
                count_manytoone += 1

    return {
        "nulls_src": count_nulls_src,
        "nulls_tgt": count_nulls_tgt,
        "one_to_one": count_onetoone,
        "one_to_many": count_onetomany,
        "many_to_one": count_manytoone,
        "many_to_many_src": count_manytomany_src,
        "many_to_many_tgt": count_manytomany_tgt,
        "len_src": last_idx_src,
        "len_tgt": last_idx_tgt
    }

def aggregate_counts_onetoone_prop(counter):
    prop_onetoone_src = counter["one_to_one"] / counter["len_src"]
    prop_onetoone_tgt = counter["one_to_one"] / counter["len_tgt"]
    return prop_onetoone_src, prop_onetoone_tgt

def aggregate_counts_nulls_prop(counter):
    prop_nulls_src = counter["nulls_src"] / counter["len_src"]
    prop_nulls_tgt = counter["nulls_tgt"] / counter["len_tgt"]
    return prop_nulls_src, prop_nulls_tgt

def read_eflomal_scores(model_type, src_lang, tgt_lang, dataset, subset, aligner="eflomal-prior"):
    filename_fwd = f"./datasets/eflomal-priors/{model_type}/{src_lang}-{tgt_lang}.{aligner}.scores.fwd"
    filename_rev = f"./datasets/eflomal-priors/{model_type}/{src_lang}-{tgt_lang}.{aligner}.scores.rev"
    if not os.path.exists(filename_fwd) or not os.path.exists(filename_rev):
        return 0.0, 0.0
    with open(filename_fwd, "r") as f:
        fwd = f.readlines()
        fwd_scores = [float(line.strip()) for line in fwd]
    
    with open(filename_rev, "r") as f:
        rev = f.readlines()
        rev_scores = [float(line.strip()) for line in rev]
    
    fwd_scores = np.array(fwd_scores)
    rev_scores = np.array(rev_scores)
    np.nan_to_num(fwd_scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(rev_scores, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    fwd_mean = np.mean(fwd_scores)
    rev_mean = np.mean(rev_scores)
    return fwd_mean, rev_mean

def construct_outputs(dataset, subset, model_type, aligner, src_langs, target_langs):
    outputs_dict = defaultdict(dict)
    processed = empty = 0
    for src_lang in src_langs:
        for tgt_lang in target_langs:
            if src_lang == tgt_lang:
                continue
            if src_lang in outputs_dict and tgt_lang in outputs_dict[src_lang]:
                continue
            sym_file = f"./datasets/eflomal-priors/{model_type}/{src_lang}-{tgt_lang}.{aligner}.sym"
            tok_file = f"./datasets/{dataset}-{subset}/{model_type}/{src_lang}-{tgt_lang}.tok.fast_align"
            if not os.path.isfile(sym_file) or not os.path.isfile(tok_file):
                continue
            else:
                with open(sym_file, "r") as afile:
                    sym_alignments = afile.readlines()
                with open(tok_file, "r") as tfile:
                    tok_lines = tfile.readlines()
                if len(sym_alignments) == 0 or len(tok_lines) == 0:
                    empty += 1
                    tgt_prop = tgt_nulls = src_prop = src_nulls = 0.0
                else:
                    lines = zip(sym_alignments, tok_lines)
                    counter = Counter()
                    for align_line, tok_line in lines:
                        counter.update(count_alignments(align_line, tok_line))
                    
                    src_prop, tgt_prop = aggregate_counts_onetoone_prop(counter)
                    src_nulls, tgt_nulls = aggregate_counts_nulls_prop(counter)
                    processed += 1
            eflomal_fwd, eflomal_rev = read_eflomal_scores(model_type, src_lang, tgt_lang, dataset, subset)
            eflomal_mean = (eflomal_fwd + eflomal_rev) / 2
            if eflomal_mean == 0.0 and tgt_prop == 0.0 and src_prop == 0.0:
                continue
            outputs_dict[tgt_lang][src_lang] = {"one-to-one": tgt_prop, "nulls": tgt_nulls, "eflomal": eflomal_mean, "eflomal_desag": eflomal_rev}
            outputs_dict[src_lang][tgt_lang] = {"one-to-one": src_prop, "nulls": src_nulls, "eflomal": eflomal_mean, "eflomal_desag": eflomal_fwd}
    print(f"{model_type}: Processed {processed} language pairs, {empty} empty sym files.")
    return outputs_dict

def main(args):
    langs = args.languages
    if args.dataset == "flores":
        subset = args.subset or "dev"
    else:
        raise NotImplementedError
    
    if subset:
        output_file = f"../results/tokenization_analysis/alignability/{args.dataset}-{subset}/{args.model_type}/align_scores.json"
    else:
        output_file = f"../results/tokenization_analysis/alignability/{args.dataset}/{args.model_type}/align_scores.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    outputs_dict = construct_outputs(args.dataset, subset, args.model_type, args.aligner, langs, langs)

    with open(output_file, "w+", encoding="utf-8") as out_file:
        json.dump(outputs_dict, out_file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--aligner", type=str)
    parser.add_argument("--languages", type=str, nargs="+")
    args = parser.parse_args()
    main(args)