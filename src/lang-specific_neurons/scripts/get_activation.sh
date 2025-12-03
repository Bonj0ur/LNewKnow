#!/bin/bash

langs=("cy" "da" "en" "es" "fr" "gd" "hi" "it" "ja" "ko" "mn" "pt" "sv" "sw" "ta" "th" "tk" "zh_CN" "zu")
models=("Llama-3.1-8B-Instruct" "Aya-Expanse-8B" "Qwen3-8B")

total=$(( ${#langs[@]} * ${#models[@]} ))
count=0

echo "Starting activation runs..."
echo

for model in "${models[@]}"; do
  for lang in "${langs[@]}"; do
    count=$((count+1))
    echo "[${count}/${total}] Running: lang=${lang}, model=${model}"
    python get_activation.py --lang "$lang" --model_name "$model" --enforce_eager
    progress=$((count*40/total))
    bar=$(printf "%-${progress}s" "#" | tr ' ' '#')
    printf "\rProgress: [%-40s] %d/%d" "$bar" "$count" "$total"
    echo
  done
done

echo
echo "All runs completed!"