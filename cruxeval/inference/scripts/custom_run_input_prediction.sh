#!/bin/bash

# Base directory for outputs
BASE_DIR="/space/poorvagarg/genPPS/cruxeval"
BATCH_SIZE=1
N_SAMPLES=5
LIMIT=800

# Model directory names (relative to BASE_DIR)
dirs=(
    # "codellama-7b"
    # "codellama-13b"
    # "codellama-34b"
    # "codellama-python-7b"
    # "codellama-python-13b"
    # "codellama-python-34b"
    # "codetulu-2-34b"
    # "deepseek-base-1.3b"
    # "deepseek-base-6.7b"
    # "deepseek-base-33b"
    # "deepseek-instruct-1.3b"
    # "deepseek-instruct-6.7b"
    # "deepseek-instruct-33b"
    # "magicoder-ds-7b"
    # "mistral-7b"
    # "mixtral-8x7b"
    # "phi-1"
    # "phi-1.5"
    # "phi-2"
    # "phind"
    # "starcoderbase-7b"
    # "starcoderbase-16b"
    # "wizard-13b"
    # "wizard-34b"
    "qwen2.5-coder-0.5b"
)

models=(
    # "codellama/CodeLlama-7b-hf"
    # "codellama/CodeLlama-13b-hf"
    # "codellama/CodeLlama-34b-hf"
    # "codellama/CodeLlama-7b-Python-hf"
    # "codellama/CodeLlama-13b-Python-hf"
    # "codellama/CodeLlama-34b-Python-hf"
    # # "allenai/codetulu-2-34b"
    # "deepseek-ai/deepseek-coder-1.3b-base"
    # "deepseek-ai/deepseek-coder-6.7b-base"
    # "deepseek-ai/deepseek-coder-33b-base"
    # "deepseek-ai/deepseek-coder-1.3b-instruct"
    # "deepseek-ai/deepseek-coder-6.7b-instruct"
    # "deepseek-ai/deepseek-coder-33b-instruct"
    # "ise-uiuc/Magicoder-S-DS-6.7B"
    # "mistralai/Mistral-7B-v0.1"
    # "mistralai/Mixtral-8x7B-v0.1"
    # "microsoft/phi-1"
    # "microsoft/phi-1_5"
    # "microsoft/phi-2"
    # "Phind/Phind-CodeLlama-34B-v2"
    # "bigcode/starcoderbase-7b"
    # "bigcode/starcoderbase"
    # "WizardLM/WizardCoder-Python-13B-V1.0"
    # "WizardLM/WizardCoder-Python-34B-V1.0"
    "Qwen/Qwen2.5-Coder-0.5B"
)

# temperatures=(0.2 0.8)
temperatures=(0.8)

# Create necessary directories
mkdir -p slurm_logs
mkdir -p model_generations_raw

for ((i=0; i<${#models[@]}; i++)); do
    model=${models[$i]}
    model_dir=${dirs[$i]}
    echo "Processing model: $model"
    
    for temperature in "${temperatures[@]}"; do
        # Create full directory path
        full_dir="${BASE_DIR}/${model_dir}_temp${temperature}_input"
        output_dir="model_generations_raw/${model_dir}_temp${temperature}_input"
        
        echo "Processing temperature: $temperature"
        echo "Full directory: $full_dir"
        echo "Output directory: $output_dir"
        
        # Create output directory
        mkdir -p "${BASE_DIR}/${output_dir}"
        
        # Run the command directly (no sharding)
        echo "Starting inference for model $model with temperature $temperature"
        echo "$LIMIT"
        echo "$model"

        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python main.py \
            --model $model \
            --trust_remote_code \
            --tasks input_prediction \
            --batch_size $BATCH_SIZE \
            --n_samples $N_SAMPLES \
            --max_length_generation 1024 \
            --precision bf16 \
            --limit $LIMIT \
            --temperature $temperature \
            --save_generations \
            --save_generations_path ${BASE_DIR}/${output_dir}/shard_0.json \
            --start 0 \
            --end $LIMIT \
            --shuffle \
            --tensor_parallel_size 1 \
            # > slurm_logs/${model_dir}_temp${temperature}.out 2> slurm_logs/${model_dir}_temp${temperature}.err
        
        echo "Completed inference for $model at temperature $temperature"
        
        # Run combine_generations.py to add sample_ prefix
        echo "Running combine_generations.py to add sample_ prefix..."
        cd /space/poorvagarg/genPPS/cruxeval/inference
        python combine_generations.py
        echo "Combine script completed"
    done
done
