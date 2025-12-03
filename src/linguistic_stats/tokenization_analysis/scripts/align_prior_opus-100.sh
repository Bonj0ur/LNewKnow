#!/usr/bin/env bash
set -Eeuo pipefail

LANGS_SHORT_1=("en")
LANGS_SHORT_2=("cy" "da" "es" "fr" "gd" "hi" "it" "ja" "ko" "mn" "pt" "sv" "ta" "th" "tk" "zh" "zu")
MODELS=("GPT-4o-Mini-2024-07-18" "Aya-Expanse-8B" "Llama-3.1-8B-Instruct" "Qwen3-8B")

MAX_JOBS="${MAX_JOBS:-$(nproc)}"

run_pair () {
  local MODEL="$1" SRC="$2" TGT="$3"
  local in="./datasets/opus-100-train/${MODEL}/${SRC}-${TGT}.tok.fast_align"
  local fwd="./datasets/priors/${MODEL}/${SRC}-${TGT}.eflomal.fwd"
  local rev="./datasets/priors/${MODEL}/${SRC}-${TGT}.eflomal.rev"
  local sym="./datasets/priors/${MODEL}/${SRC}-${TGT}.eflomal.sym"
  local pri="./datasets/priors/${MODEL}/${SRC}-${TGT}.eflomal.prior"

  mkdir -p "./datasets/priors/${MODEL}"

  echo "[START] ${MODEL} ${SRC}-${TGT}"
  eflomal-align -i "$in" -f="$fwd" -r="$rev"
  fast_align/build/atools -i "$fwd" -j "$rev" -c grow-diag-final-and > "$sym"
  eflomal-makepriors -i "$in" -f="$fwd" -r="$rev" -p "$pri"
  echo "[DONE ] ${MODEL} ${SRC}-${TGT}"
}

run_with_limit () {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
  "$@" &
}

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT_1[@]}"; do
    for TGT in "${LANGS_SHORT_2[@]}"; do
      run_with_limit run_pair "$MODEL" "$SRC" "$TGT"
    done
  done
done

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT_2[@]}"; do
    for TGT in "${LANGS_SHORT_1[@]}"; do
      run_with_limit run_pair "$MODEL" "$SRC" "$TGT"
    done
  done
done

wait
echo "All done."