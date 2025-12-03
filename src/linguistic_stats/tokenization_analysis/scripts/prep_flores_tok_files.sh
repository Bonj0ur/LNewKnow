#!/bin/bash
set -x

LANGS_SHORT=("cy" "da" "en" "es" "fr" "gd" "hi" "it" "ja" "ko" "mn" "pt" "sv" "sw" "ta" "th" "tk" "zh_CN" "zu")
MODELS=("GPT-4o-Mini-2024-07-18" "Aya-Expanse-8B" "Llama-3.1-8B-Instruct" "Qwen3-8B")

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT[@]}"; do
    for TGT in "${LANGS_SHORT[@]}"; do
      if [[ "$SRC" == "$TGT" ]]; then
        continue
      fi
      python prep_tok_files.py --dataset "flores" --subset "dev" --model_type "$MODEL" --src_lang "$SRC" --tgt_lang "$TGT"
      done
  done
done