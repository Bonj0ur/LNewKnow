#!/bin/bash
set -x

ALL_LANGS=("cy" "da" "en" "es" "fr" "gd" "hi" "it" "ja" "ko" "mn" "pt" "sv" "sw" "ta" "th" "tk" "zh_CN" "zu")
MODELS=("GPT-4o-Mini-2024-07-18" "Aya-Expanse-8B" "Llama-3.1-8B-Instruct" "Qwen3-8B")

for MODEL in "${MODELS[@]}"; do
  python alignability.py --model_type "$MODEL" --aligner "eflomal-prior" --dataset "flores" --subset "dev" --languages "${ALL_LANGS[@]}"
done