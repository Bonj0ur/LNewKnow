# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import subprocess
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora_root", type=str)
    parser.add_argument("--lora_suffix", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    parser.add_argument("--script", type=str, default="sub_query_transferable_local.py")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    args = parser.parse_args()

    info = json.load(open(args.info_path))
    languages = list(info["languages"].keys())

    for train_lang in languages:
        ckpt_dir = os.path.join(args.lora_root, f"{train_lang}_{args.lora_suffix}")
        lora_ckpts = sorted(glob(os.path.join(ckpt_dir, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
        if not lora_ckpts:
            print(f"[Skip] No checkpoint for {train_lang}")
            continue
        last_ckpt = lora_ckpts[-1]

        print(f"[Launch] {train_lang} → all others using {last_ckpt}")
        subprocess.run([
            "python", args.script,
            "--base_model", args.base_model,
            "--lora_ckpt", last_ckpt,
            "--train_lang", train_lang,
            "--test_dir", args.test_dir,
            "--output_dir", args.output_dir,
            "--info_path", args.info_path,
            "--max_tokens", str(args.max_tokens),
            "--max_model_len", str(args.max_model_len),
            "--gpu_mem_util", str(args.gpu_mem_util)
        ])

if __name__ == "__main__":
    main()