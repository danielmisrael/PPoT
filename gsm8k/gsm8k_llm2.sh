#!/bin/bash

# Base directory for model generations
BASE_DIR="/space/poorvagarg/genPPS/gsm8k"

# Run names to evaluate (matching your generated predictions)
model_names=(
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    # "Qwen/Qwen2.5-Coder-7B-Instruct"
)


for model_name in "${model_names[@]}"; do
    entropy_save_path="${BASE_DIR}/generations/llm/${model_name}/"
    report_save_path="${BASE_DIR}/generations/llm/${model_name}/"
    llm_cache_path="${BASE_DIR}/generations/llm/${model_name}/"
    for i in {1..20}; do
        echo "Running: $model_name $i"
        CUDA_VISIBLE_DEVICES=0 python3 -m gsm8k.eval_gsm8k_llm \
            --dataset gsm8k/gsm8k.json \
            --entropy-save-path "$entropy_save_path" \
            --report-save-path "$report_save_path" \
            --llm-cache-path "$llm_cache_path" \
            --model "$model_name" \
            --temperature 0.7 \
            --num-samples 5 \
            --rule all \
            --num-llm-samples $i
    done
done




