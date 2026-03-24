#!/bin/bash

# Usage: bash gsm8k/accuracy_gsm8k_error.sh <parent_dir>

# Run names to evaluate (matching your generated predictions)
model_names=(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)


for model_name in "${model_names[@]}"; do
    for k in {1..5}; do
        echo "Running: $model_name"
        CUDA_VISIBLE_DEVICES=1 python3 -m gsm8k.accuracy_gsm8k \
            --dataset gsm8k/gsm8k.json \
            --model "$model_name" \
            --temperature 0.7 \
            --num-samples 40 \
            --rule all \
            --different-constraint \
            --num-llm-samples 20 \
            --parent-dir $1 \
            --suffix fast_$k \
    done
done




