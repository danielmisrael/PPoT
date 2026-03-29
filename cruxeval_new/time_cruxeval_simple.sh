#!/bin/bash

# Usage: bash gsm8k/time_gsm8k_error.sh <report_save_path>

# Run names to evaluate (matching your generated predictions)
model_names=(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)


for model_name in "${model_names[@]}"; do
    for i in {1..20..2}; do
        echo "Running: $model_name $i $j"
        CUDA_VISIBLE_DEVICES=1 python3 -m cruxeval_new.time_cruxeval_simple \
            --model "$model_name" \
            --temperature 0.7 \
            --different-constraint \
            --num-llm-samples $i \
            --num-examples 20 \
            --suffix "simple2_$k" \
            --shuffle
    done
done




