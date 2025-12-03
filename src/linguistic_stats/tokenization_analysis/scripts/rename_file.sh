#!/bin/bash

MODEL_DIR="./datasets/eflomal-priors"
MODELS=("Aya-Expanse-8B" "GPT-4o-Mini-2024-07-18" "Llama-3.1-8B-Instruct" "Qwen3-8B")

for MODEL in "${MODELS[@]}"; do
  echo "Processing model: $MODEL"
  DIR="$MODEL_DIR/$MODEL"

  if [ ! -d "$DIR" ]; then
    echo "Directory $DIR not found! Skipping..."
    continue
  fi

  for f in "$DIR"/*; do
    fname=$(basename "$f")

    if [[ "$fname" == *zh_CN* ]]; then
      continue
    fi

    if [[ "$fname" == *-zh.* ]]; then
      new_name=$(echo "$fname" | sed 's/-zh\./-zh_CN./g')
      echo mv "$f" "$DIR/$new_name"
      mv "$f" "$DIR/$new_name"
    fi

    if [[ "$fname" == *zh-* ]]; then
      new_name=$(echo "$fname" | sed 's/zh-/zh_CN-/g')
      echo mv "$f" "$DIR/$new_name"
      mv "$f" "$DIR/$new_name"
    fi
  done

  echo "Finished processing $MODEL"
done

echo "All models done."