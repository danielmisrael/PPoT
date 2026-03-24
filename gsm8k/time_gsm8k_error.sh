#!/bin/bash

# Usage: bash gsm8k/time_gsm8k_error.sh <report_save_path>

# Run names to evaluate (matching your generated predictions)
model_names=(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)


for model_name in "${model_names[@]}"; do
    for i in {1..20}; do
        for j in 1 5 10 15 20; do
            for k in {1..5}; do
                echo "Running: $model_name $i $j"
                CUDA_VISIBLE_DEVICES=0 python3 -m gsm8k.time_gsm8k_fast \
                    --dataset gsm8k/gsm8k.json \
                    --model "$model_name" \
                    --temperature 0.7 \
                    --num-samples $j \
                    --rule all \
                    --different-constraint \
                    --num-llm-samples $i \
                    --num-examples 20 \
                    --suffix fast_$k \
                    --shuffle
            done
        done
    done
done




