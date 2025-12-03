# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import tokenization_scorer

def split_data_file(dataset, subset, model_type, src_lang, tgt_lang):
    file_name = f'./datasets/{dataset}-{subset}/{model_type}/{src_lang}-{tgt_lang}.tok.fast_align'
    with open(file_name, "r") as tfile:
        tok_lines = tfile.readlines()
    src_lines = [line.split(" ||| ")[0] for line in tok_lines]
    tgt_lines = [line.split(" ||| ")[1] for line in tok_lines]
    return src_lines, tgt_lines

def count_tokens(lines):
    return sum(len(line.strip().split()) for line in lines if line.strip())

def main(args):
    output_file = f"../results/tokenization_analysis/efficiency/{args.dataset}-{args.subset}/{args.model_type}/renyi_scores.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    outputs_dict = {}

    target_langs = args.tgt_lang
    for tgt_lang in target_langs:
        src_lines, tgt_lines = split_data_file(args.dataset, args.subset, args.model_type, args.src_lang, tgt_lang)

        src_score = tokenization_scorer.score(src_lines, metric="renyi", power=2.5)
        tgt_score = tokenization_scorer.score(tgt_lines, metric="renyi", power=2.5)
        
        src_tok_count = count_tokens(src_lines)
        tgt_tok_count = count_tokens(tgt_lines)
        
        outputs_dict[tgt_lang] = {
            "renyi_score": tgt_score,
            "token_count": tgt_tok_count
        }
        outputs_dict[args.src_lang] = {
            "renyi_score": src_score,
            "token_count": src_tok_count
        }
    
    with open(output_file, "w+", encoding="utf-8") as out_file:
        json.dump(outputs_dict, out_file, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str, nargs="+")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    args = parser.parse_args()
    main(args)