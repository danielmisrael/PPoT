#!/bin/bash

# Base directory for model generations
BASE_DIR="/data1/disrael/genPPS/cruxeval"

# Run names to evaluate (matching your generated predictions)
run_names=(
    # "deepseek-base-1.3b_temp0.7_input"
    # "phi-2_temp0.8_input"
    "qwen2.5-coder-0.5b_temp0.8_input"
)

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