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
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--script", type=str, default="sub_query_conflict_local.py")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lora_folders = sorted(os.listdir(args.lora_root))
    for name in lora_folders:
        ckpt_dir = os.path.join(args.lora_root, name)
        if not os.path.isdir(ckpt_dir): continue
        ckpts = sorted(glob(os.path.join(ckpt_dir, "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
        if not ckpts:
            print(f"[Skip] No checkpoint found in {ckpt_dir}")
            continue
        last_ckpt = ckpts[-1]

        test_subdir = name.replace("train_", "").split("_eps")[0]

        print(f"[Launch] Testing {name} using {last_ckpt}")
        subprocess.run([
            "python", args.script,
            "--base_model", args.base_model,
            "--lora_ckpt", last_ckpt,
            "--train_name", name,
            "--test_dir", os.path.join(args.test_dir, test_subdir),
            "--output_dir", os.path.join(args.output_dir, name),
            "--max_tokens", str(args.max_tokens),
            "--max_model_len", str(args.max_model_len),
            "--gpu_mem_util", str(args.gpu_mem_util),
        ])

if __name__ == "__main__":
    main()