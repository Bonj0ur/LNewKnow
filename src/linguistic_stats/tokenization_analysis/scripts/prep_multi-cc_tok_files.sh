#!/bin/bash
set -Eeuo pipefail

LANGS_SHORT=("cy" "da" "es" "fr" "hi" "it" "ja" "ko" "mn" "pt" "sv" "sw" "ta" "th" "zh_CN" "zu")
MODELS=("GPT-4o-Mini-2024-07-18" "Aya-Expanse-8B" "Qwen3-8B" "Llama-3.1-8B-Instruct")

MAX_JOBS="${MAX_JOBS:-$(nproc)}"

run_one () {
  local MODEL="$1" SRC="$2" TGT="$3"
  echo "[START] $MODEL $SRC-$TGT"
  python prep_tok_files.py --dataset "multi-cc" --model_type "$MODEL" --src_lang "$SRC" --tgt_lang "$TGT"
  echo "[DONE ] $MODEL $SRC-$TGT"
}

run_with_limit () {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
  run_one "$@" &
}

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT[@]}"; do
    for TGT in "${LANGS_SHORT[@]}"; do
      if [[ "$SRC" == "$TGT" ]]; then
        continue
      fi
      run_with_limit "$MODEL" "$SRC" "$TGT"
    done
  done
done

wait
echo "All tasks finished."