#!/bin/bash
set -x

LANGS_SHORT_1=("en")
LANGS_SHORT_2=("cy" "da" "es" "fr" "gd" "hi" "it" "ja" "ko" "mn" "pt" "sv" "ta" "th" "tk" "zh" "zu")
MODELS=("GPT-4o-Mini-2024-07-18" "Aya-Expanse-8B" "Llama-3.1-8B-Instruct" "Qwen3-8B")

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT_1[@]}"; do
    for TGT in "${LANGS_SHORT_2[@]}"; do
      python prep_tok_files.py --dataset "opus-100" --subset "train" --model_type "$MODEL" --src_lang "$SRC" --tgt_lang "$TGT"
      done
  done
done

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT_2[@]}"; do
    for TGT in "${LANGS_SHORT_1[@]}"; do
      python prep_tok_files.py --dataset "opus-100" --subset "train" --model_type "$MODEL" --src_lang "$SRC" --tgt_lang "$TGT"
      done
  done
done