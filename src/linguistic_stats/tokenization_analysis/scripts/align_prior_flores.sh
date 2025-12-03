#!/usr/bin/env bash
set -Eeuo pipefail

LANGS_SHORT=("cy" "da" "en" "es" "fr" "gd" "hi" "it" "ja" "ko" "mn" "pt" "sv" "sw" "ta" "th" "tk" "zh_CN" "zh" "zu")
MODELS=("GPT-4o-Mini-2024-07-18" "Aya-Expanse-8B" "Llama-3.1-8B-Instruct" "Qwen3-8B")

MAX_JOBS="${MAX_JOBS:-$(nproc)}"

normalize_for_flores () {
  case "$1" in
    zh) echo "zh_CN" ;;
    *) echo "$1" ;;
  esac
}

run_pair () {
  local MODEL="$1" SRC="$2" TGT="$3"
  
  local SRC_IN TGT_IN
  SRC_IN="$(normalize_for_flores "$SRC")"
  TGT_IN="$(normalize_for_flores "$TGT")"

  if [[ "$SRC_IN" == "$TGT_IN" ]]; then
    echo "[SKIP] ${MODEL} ${SRC}-${TGT} (normalized to same: ${SRC_IN}-${TGT_IN})"
    return 0
  fi

  local in="./datasets/flores-dev/${MODEL}/${SRC_IN}-${TGT_IN}.tok.fast_align"
  local fwd="./datasets/eflomal-priors/${MODEL}/${SRC}-${TGT}.eflomal-prior.fwd"
  local rev="./datasets/eflomal-priors/${MODEL}/${SRC}-${TGT}.eflomal-prior.rev"
  local pri="./datasets/priors/${MODEL}/${SRC}-${TGT}.eflomal.prior"
  local sym="./datasets/eflomal-priors/${MODEL}/${SRC}-${TGT}.eflomal-prior.sym"
  local fwd_scores="./datasets/eflomal-priors/${MODEL}/${SRC}-${TGT}.eflomal-prior.scores.fwd"
  local rev_scores="./datasets/eflomal-priors/${MODEL}/${SRC}-${TGT}.eflomal-prior.scores.rev"

  if [[ ! -f "$pri" ]]; then
    echo "[SKIP] ${MODEL} ${SRC}-${TGT} (missing prior: $pri)"
    return 0
  fi
  if [[ ! -f "$in" ]]; then
    echo "[SKIP] ${MODEL} ${SRC}-${TGT} (missing input: $in)"
    return 0
  fi

  mkdir -p "./datasets/eflomal-priors/${MODEL}"

  echo "[START] ${MODEL} ${SRC}-${TGT} (in=${SRC_IN}-${TGT_IN})"

  eflomal-align -i "$in" \
                -f="$fwd" \
                -r="$rev" \
                -p "$pri" \
                --overwrite \
                --score-model 3 \
                --forward-scores "$fwd_scores" \
                --reverse-scores "$rev_scores" \
                -1 1 -2 1 -3 1

  fast_align/build/atools -i "$fwd" -j "$rev" -c grow-diag-final-and > "$sym"

  echo "[DONE ] ${MODEL} ${SRC}-${TGT}"
}

run_with_limit () {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
  "$@" &
}

for MODEL in "${MODELS[@]}"; do
  for SRC in "${LANGS_SHORT[@]}"; do
    for TGT in "${LANGS_SHORT[@]}"; do
      if [[ "$SRC" == "$TGT" ]]; then
        continue
      fi
      run_with_limit run_pair "$MODEL" "$SRC" "$TGT"
    done
  done
done

wait
echo "All done."