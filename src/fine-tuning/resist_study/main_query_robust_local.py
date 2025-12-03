# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import subprocess
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora_root", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--script", type=str, default="sub_query_robust_local.py")
    args = parser.parse_args()

    ckpt_paths = sorted(glob(os.path.join(args.lora_root, "checkpoint-*")))

    for ckpt in ckpt_paths:
        print(f"[Spawn] Launching subprocess for {ckpt}")
        subprocess.run([
            "python", args.script,
            "--base_model", args.base_model,
            "--lora_ckpt", ckpt,
            "--test_path", args.test_path,
            "--output_dir", args.output_dir,
            "--max_tokens", str(args.max_tokens),
            "--max_model_len", str(args.max_model_len),
            "--gpu_mem_util", str(args.gpu_mem_util)
        ])

if __name__ == "__main__":
    main()