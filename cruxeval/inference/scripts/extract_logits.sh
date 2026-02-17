#!/bin/bash

GENERATIONS_PATH=/space/poorvagarg/genPPS/cruxeval/model_generations/qwen2.5-coder-0.5b_temp0.8_input/generations.json
GENERATIONS_RAW_PATH=/space/poorvagarg/genPPS/cruxeval/model_generations_raw/qwen2.5-coder-0.5b_temp0.8_input/generations_raw.json
MODEL_PATH=Qwen/Qwen2.5-Coder-0.5B

# Extract the model directory and create output directory inside it
MODEL_DIR=$(dirname $GENERATIONS_PATH)
OUTPUT_DIR="${MODEL_DIR}/data"

echo "Generations path: $GENERATIONS_PATH"
echo "Generations raw path: $GENERATIONS_RAW_PATH"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run the extraction script
CUDA_VISIBLE_DEVICES=0 python extract_logits.py \
    --generations_path "$GENERATIONS_PATH" \
    --generations_raw_path "$GENERATIONS_RAW_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "Logits extraction completed!"
