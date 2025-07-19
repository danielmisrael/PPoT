# genPPS

This repository contains scripts for running vision-language models on the Plot2Code benchmark, specifically for generating Python matplotlib code from plot images.

## Setup

### 1. Clone the repository and submodules
```bash
git clone <repository-url>
cd genPPS
git submodule update --init --recursive
```

### 2. Install dependencies
```bash
pip install torch transformers datasets pillow tqdm qwen-vl-utils
```

<!-- ### 3. Download the Plot2Code dataset
```bash
python download_dataset.py
``` -->

## Running Qwen2.5-VL-3B-Instruct

### Step 1: Generate Code
Generate Python matplotlib code from plot images using the Qwen2.5-VL-3B-Instruct model:

```bash
CUDA_VISIBLE_DEVICES=7 python generate_code.py
```

This script will:
- Load the Qwen2.5-VL-3B-Instruct model on GPU 7
- Process all test images from the Plot2Code dataset
- Generate Python matplotlib code for each image
- Save results to `generated_results/Qwen/Qwen2.5-VL-3B-Instruct/direct/default/generated_code.jsonl`

### Step 2: Execute Generated Code
Execute the generated code to create plots:

```bash
python -m plot2code.execute_generated_code --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --prompt_strategy default
```

This command will:
- Read the generated code from the previous step
- Execute each code snippet to create plots
- Save the generated plots to `generated_results/Qwen/Qwen2.5-VL-3B-Instruct/direct/default/generated_plots/`

### Step 3: Run Text Match Evaluation
Evaluate the generated code against ground truth:

```bash
python -m plot2code.eval.text_match_score --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --prompt_strategy default
```

This command will:
- Compare generated code with ground truth code
- Calculate text match scores
- Save evaluation results to `evaluation_results/Qwen/Qwen2.5-VL-3B-Instruct/direct/default/`

### Optional: Run Full Evaluation Pipeline
For a complete evaluation including GPT-4V assessment:

```bash
cd Plot2Code
./scripts/evaluate.sh "Qwen/Qwen2.5-VL-3B-Instruct" default
```

This runs the complete evaluation pipeline including:
- Code execution
- Text match scoring
- GPT-4V evaluation
- Result combination

## Model Configuration

The Qwen2.5-VL-3B-Instruct model is configured with:
- **Model**: Qwen/Qwen2.5-VL-3B-Instruct
- **Precision**: float16 for faster inference
- **Device**: CUDA GPU 7
- **Max tokens**: 512 for generation
- **Temperature**: 0.1 for consistent output

## File Structure

```
genPPS/
├── Plot2Code/                    # Plot2Code submodule
│   └── data/
│       └── python_matplotlib/
│           ├── test/             # Test images and metadata
│           └── train/            # Training data
├── generated_results/             # Generated code and plots
│   └── Qwen/
│       └── Qwen2.5-VL-3B-Instruct/
│           └── direct/
│               └── default/
│                   ├── generated_code.jsonl
│                   └── generated_plots/
├── evaluation_results/            # Evaluation results
│   └── Qwen/
│       └── Qwen2.5-VL-3B-Instruct/
│           └── direct/
│               └── default/
└── generate_code.py              # Code generation script
```

## Scripts Overview

### `generate_code.py`
- Loads Qwen2.5-VL-3B-Instruct model with CUDA optimization
- Processes images and generates Python matplotlib code
- Uses proper CUDA device mapping for GPU 7
- Saves results in Plot2Code-compatible format
- Saves results incrementally in JSONL format

### Plot2Code Commands
- `python -m plot2code.execute_generated_code`: Executes generated code to create plots
- `python -m plot2code.eval.text_match_score`: Evaluates code against ground truth
- `python -m plot2code.eval.gpt4v_evaluations_score`: GPT-4V-based evaluation
- `python -m plot2code.eval.combine_evaluation_results`: Combines all evaluation results

## Performance Notes

- The model runs on GPU 7 with CUDA optimization
- Uses float16 precision for faster inference
- Reduced max_new_tokens (512) for faster generation
- No fallback code or error catching (errors propagate naturally)

## Troubleshooting

1. **CUDA Memory Issues**: Ensure GPU 7 has sufficient memory (model requires ~6GB)
2. **Model Loading**: The model will download automatically on first run
3. **Dataset Issues**: The Plot2Code dataset should be included in the submodule
4. **Dependencies**: Install `qwen-vl-utils` for proper model functionality

## Results

The evaluation will produce:
- Text match scores comparing generated vs ground truth code
- Generated plots saved as PNG files
- Detailed evaluation metrics in JSON format
