# genPPS



This repository contains code for Generative Probablistic Program Synthesis (GenPPS). It has scripts for running vision-language models on the Plot2Code benchmark, specifically for generating Python matplotlib code from plot images.

## Setup

### 1. Clone the repository and submodules
```bash
git clone <repository-url>
cd genPPS
git submodule update --init --recursive
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Code Generation
Generate Python matplotlib code from plot images using the Qwen2.5-VL-3B-Instruct model:

```bash
CUDA_VISIBLE_DEVICES=x python generate_code.py
```

Replace `x` with your desired GPU number (e.g., `CUDA_VISIBLE_DEVICES=7` for GPU 7).

This script will:
- Load the Qwen2.5-VL-3B-Instruct model on the specified GPU
- Process test images from the Plot2Code dataset
- Generate Python matplotlib code for each image
- Execute the generated code to create plots
- Save results to `generated_results/Qwen2.5-VL-3B-Instruct/direct/instruct/`

### Evaluation
Evaluate the generated code against ground truth:

```bash
CUDA_VISIBLE_DEVICES=x python text_match_score.py --model_name "Qwen2.5-VL-3B-Instruct" --prompt_strategy instruct
```

This command will:
- Compare generated code with ground truth code
- Calculate text match scores
- Save evaluation results to the appropriate directory

