#!/bin/bash

# Run names to evaluate (matching your generated predictions)
model_names=(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)


for model_name in "${model_names[@]}"; do
    for i in {1..20}; do
        for j in 1 5 10 15 20; do
            echo "Running: $model_name $i $j"
            CUDA_VISIBLE_DEVICES=1 python3 -m cruxeval_new.time_cruxeval \
                --model "$model_name" \
                --temperature 0.7 \
                --num-samples $j \
                --different-constraint \
                --num-llm-samples $i \
                --num-examples 100
        done
    done
done




