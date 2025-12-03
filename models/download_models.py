import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_dir", type=str, default="./Qwen3-8B")
    parser.add_argument("--aya_dir", type=str, default="./Aya-Expanse-8B")
    parser.add_argument("--llama_dir", type=str, default="./Llama-3.1-8B-Instruct")
    parser.add_argument("--token", type=str)
    args = parser.parse_args()

    # Qwen
    print(f"Downloading Qwen/Qwen3-8B into {args.qwen_dir}")
    snapshot_download(
        repo_id="Qwen/Qwen3-8B",
        local_dir=args.qwen_dir,
        local_dir_use_symlinks=False
    )

    # Aya
    print(f"Downloading CohereLabs/aya-expanse-8b into {args.aya_dir}")
    snapshot_download(
        repo_id="CohereLabs/aya-expanse-8b",
        local_dir=args.aya_dir,
        local_dir_use_symlinks=False,
        token=args.token
    )

    # Llama
    print(f"Downloading meta-llama/Llama-3.1-8B-Instruct into {args.llama_dir}")
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        local_dir=args.llama_dir,
        local_dir_use_symlinks=False,
        token=args.token
    )

if __name__ == "__main__":
    main()