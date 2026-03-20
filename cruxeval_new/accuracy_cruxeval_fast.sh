#!/bin/bash

# Run names to evaluate (matching your generated predictions)
model_names=(
    #"Qwen/Qwen2.5-Coder-0.5B-Instruct"
    #"Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)


for model_name in "${model_names[@]}"; do
    echo "Running: $model_name"
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python3 -m cruxeval_new.accuracy_cruxeval_fast \
        --model "$model_name" \
        --temperature 0.7 \
        --num-samples 40 \
        --different-constraint \
        --num-llm-samples 20
done




