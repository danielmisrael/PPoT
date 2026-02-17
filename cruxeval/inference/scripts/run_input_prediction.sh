#!/bin/bash



dirs=(
    # "codellama-7b"
    # "codellama-13b"
    # "codellama-34b"
    # "codellama-python-7b"
    # "codellama-python-13b"
    # "codellama-python-34b"
    # "codetulu-2-34b"
    "/data1/disrael/genPPS/deepseek-base-1.3b"
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
)

models=(
    # "codellama/CodeLlama-7b-hf"
    # "codellama/CodeLlama-13b-hf"
    # "codellama/CodeLlama-34b-hf"
    # "codellama/CodeLlama-7b-Python-hf"
    # "codellama/CodeLlama-13b-Python-hf"
    # "codellama/CodeLlama-34b-Python-hf"
    # "allenai/codetulu-2-34b"
    "deepseek-ai/deepseek-coder-1.3b-base"
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
)

# temperatures=(0.2 0.8)
temperatures=(0.7)

# Create logs directory
mkdir -p slurm_logs

for ((i=0; i<${#models[@]}; i++)); do
    model=${models[$i]}
    base_dir=${dirs[$i]}
    echo "Processing model: $model"
    for temperature in "${temperatures[@]}"; do
        dir="${base_dir}_temp${temperature}_input"
        echo "Processing temperature: $temperature, directory: $dir"
        
        # Create output directory
        mkdir -p model_generations_raw/$dir
        
        # Run the two shards sequentially (originally array 0-1)
        SIZE=800
        GPUS=2
        
        for shard in 0 1; do
            start_idx=$((shard * SIZE / GPUS))
            end_idx=$(((shard + 1) * SIZE / GPUS))
            
            echo "Starting shard $shard with start and end $start_idx $end_idx"
            
            # Extract just the directory name for log files (remove path)
            dir_name=$(basename "$dir")
            
            # Run the command directly
            python main.py \
                --model $model \
                --use_auth_token \
                --trust_remote_code \
                --tasks input_prediction \
                --batch_size 10 \
                --n_samples 10 \
                --max_length_generation 1024 \
                --precision bf16 \
                --limit $SIZE \
                --temperature $temperature \
                --save_generations \
                --save_generations_path model_generations_raw/${dir}/shard_${shard}.json \
                --start $start_idx \
                --end $end_idx \
                --shuffle \
                --tensor_parallel_size 1 \
                > slurm_logs/${dir_name}_shard_${shard}.out 2> slurm_logs/${dir_name}_shard_${shard}.err
            
            echo "Completed shard $shard"
        done
        
        echo "Completed all shards for $dir"
    done
done