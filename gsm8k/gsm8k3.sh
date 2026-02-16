#!/bin/bash

# Base directory for model generations
BASE_DIR="/space/poorvagarg/genPPS/gsm8k"

# Run names to evaluate (matching your generated predictions)
model_names=(
    # "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    # "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)

temp=(0.2 0.5 0.7 1.0)


for model_name in "${model_names[@]}"; do
    echo "Running: $model_name $t"
    entropy_save_path="${BASE_DIR}/generations/${model_name}/"
    report_save_path="${BASE_DIR}/generations/${model_name}/"
    llm_cache_path="${BASE_DIR}/generations/${model_name}/"
    CUDA_VISIBLE_DEVICES=1 python3 -m gsm8k.eval_gsm8k \
        --dataset gsm8k/gsm8k.json \
        --entropy-save-path "$entropy_save_path" \
        --report-save-path "$report_save_path" \
        --llm-cache-path "$llm_cache_path" \
        --model "$model_name" \
        --temperature 0.0 \
        --num-samples 5 \
        --rule all \
        --different-constraint \
        --num-llm-samples 1
done

for model_name in "${model_names[@]}"; do
    for t in "${temp[@]}"; do
        echo "Running: $model_name $t"
        entropy_save_path="${BASE_DIR}/generations/${model_name}/"
        report_save_path="${BASE_DIR}/generations/${model_name}/"
        llm_cache_path="${BASE_DIR}/generations/${model_name}/"

        CUDA_VISIBLE_DEVICES=1 python3 -m gsm8k.eval_gsm8k \
            --dataset gsm8k/gsm8k.json \
            --entropy-save-path "$entropy_save_path" \
            --report-save-path "$report_save_path" \
            --llm-cache-path "$llm_cache_path" \
            --model "$model_name" \
            --temperature "$t" \
            --num-samples 5 \
            --rule all \
            --different-constraint \
            --num-llm-samples 5

        echo "Completed: $model_name $t"
    done
done





# # Create necessary directories
# mkdir -p evaluation_results
# mkdir -p slurm_logs

# for run_name in "${run_names[@]}"; do
#     echo "Evaluating: $run_name"

#     # Set up paths
#     generations_path="${BASE_DIR}/model_generations/${run_name}/generations.json"
#     results_path="evaluation_results/${run_name}.json"

#     echo "Generations path: $generations_path"
#     echo "Results path: $results_path"

#     # Check if generations file exists
#     if [ ! -f "$generations_path" ]; then
#         echo "Warning: Generations file not found: $generations_path"
#         continue
#     fi

#     # Run evaluation directly (no SLURM)
#     echo "Starting evaluation for $run_name"

#     python evaluate_generations.py \
#         --generations_path "$generations_path" \
#         --scored_results_path "$results_path" \
#         --mode input \
#         > slurm_logs/${run_name}_eval.out 2> slurm_logs/${run_name}_eval.err

#     echo "Completed evaluation for $run_name"
# done
