# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import FLORES_200_CODE, MODEL_PATH

def tokenize_file(file_path, tokenizer):
    ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_dir", type=str, default="../linguistic_stats/datasets/flores200_dataset/")
    parser.add_argument("--langs", type=str, nargs="+", default=["cy", "da", "en", "es", "fr", "gd", "hi", "it", "ja", "ko", "mn", "pt", "sv", "sw", "ta", "th", "tk", "zh_CN", "zu"])
    args = parser.parse_args()

    model_path = MODEL_PATH[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    os.makedirs(args.save_dir, exist_ok=True)

    for lang in args.langs:
        all_ids = []

        dev_path = os.path.join(args.data_dir, "dev", f"{FLORES_200_CODE[lang]}.dev")
        devtest_path = os.path.join(args.data_dir, "devtest", f"{FLORES_200_CODE[lang]}.devtest")

        if os.path.exists(dev_path):
            all_ids.extend(tokenize_file(dev_path, tokenizer))
        if os.path.exists(devtest_path):
            all_ids.extend(tokenize_file(devtest_path, tokenizer))
        
        tensor = torch.LongTensor(all_ids)
        save_path = os.path.join(args.save_dir, f"id.{lang}.flores.{args.model_name}")
        torch.save(tensor, save_path)