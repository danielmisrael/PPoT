#!/bin/bash

# Base directory for model generations
BASE_DIR="/space/poorvagarg/genPPS"

# Run names to evaluate (matching your generated predictions)
model_names=(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)

temp=(0.0 0.2 0.5 0.7 1.0)



# Create necessary directories
mkdir -p evaluation_results
mkdir -p slurm_logs

for run_name in "${run_names[@]}"; do
    echo "Evaluating: $run_name"

    # Set up paths
    generations_path="${BASE_DIR}/model_generations/${run_name}/generations.json"
    results_path="evaluation_results/${run_name}.json"

    echo "Generations path: $generations_path"
    echo "Results path: $results_path"

    # Check if generations file exists
    if [ ! -f "$generations_path" ]; then
        echo "Warning: Generations file not found: $generations_path"
        continue
    fi

    # Run evaluation directly (no SLURM)
    echo "Starting evaluation for $run_name"

    python evaluate_generations.py \
        --generations_path "$generations_path" \
        --scored_results_path "$results_path" \
        --mode input \
        > slurm_logs/${run_name}_eval.out 2> slurm_logs/${run_name}_eval.err

    echo "Completed evaluation for $run_name"
done
