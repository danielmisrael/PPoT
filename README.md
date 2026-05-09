# Probabilistic Programs of Thought

[![arXiv](https://img.shields.io/badge/arXiv-2604.17290-blue?link=https%3A%2F%2F2604.17290)](https://arxiv.org/abs/2604.17290)

Probabilistic Programs of Thought is a novel test time framework to interpret LLM generated programs as probabilistic programs to enable tractable probabilistic reasoning. 

This repository consists of all necessary instructions and code to reproduce the experiments in the paper.

## Setup

### Clone the repository and install dependencies
```bash
git clone https://github.com/PoorvaGarg/PPoT.git
cd PPoT
```

```bash
conda create -n ppot python=3.11
conda activate ppot
pip install -r requirements.txt
```

## Reproducing Experiments

The paper includes experiments for three datasets, namely GSM8k, Plot2Code and CruxEval. We provide instructions below to reproduce the experimental results.

### 1. GSM8k

**Basic Usage:**
```bash
python3 -m gsm8k.eval_gsm8k_fast --dataset gsm8k/gsm8k.json --model Qwen/Qwen2.5-Coder-3B-Instruct --temperature 0.7 --num-samples 5 --rule all --different-constraint --num-llm-samples 2 --num-examples 10
```

**`gsm8k/eval_gsm8k_fast.py` arguments:** We describe some key arguments for this script below. There are more customizing arguments available, please checkout the script itself.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Model to use; one of `Qwen/Qwen2.5-Coder-{0.5,3,7}B-Instruct` |
| `--device` | str | `cuda:0` | Device to load the model on (e.g. `cuda:1`) |
| `--num-llm-samples` | int | `4` | Number of LLM samples to generate per example |
| `--num-samples` | int | `5` | Number of probabilistic program samples per LLM sample |
| `--temperature` | float | `0.3` | Sampling temperature for the LLM |
| `--num-examples` | int | `10000000` | Maximum number of dataset examples to evaluate |
| `--rule` | str | `digits` | Token rule(s) for support: `digit`, `compare`, `arithmetic`, `augment`, or `all` |
| `--different-constraint` | flag | `False` | Enforce different-value constraint when sampling programs |

For the results in Table 1, use the following commands with models of different sizes:
```bash 
python3 -m gsm8k.eval_gsm8k_fast --dataset gsm8k/gsm8k.json --model Qwen/Qwen2.5-Coder-3B-Instruct --temperature 0.0 --num-samples 5 --rule all --different-constraint --num-llm-samples 1
python3 -m gsm8k.eval_gsm8k_fast --dataset gsm8k/gsm8k.json --model Qwen/Qwen2.5-Coder-3B-Instruct --temperature 0.7 --num-samples 5 --rule all --different-constraint --num-llm-samples 5
```

To reproduce the experimental data for accuracy and runtime, use the following commands:

```bash
gsm8k/accuracy_gsm8k_error.sh <parent_dir> <num_iterations>
gsm8k/time_gsm8k_simple.sh
```

### 2. Plot2Code

**Basic Usage:**
```bash
python -m plot2code_new.eval_plot2code --num-examples=132 --temperature=0.7 --num-llm-samples=5 --num-samples=5 --save-dir=out --program-temperature=1.0 --model-name="Qwen/Qwen2.5-VL-7B-Instruct" --include-arithmetic-operators
```
Checking

**plot2code_new/eval_plot2code.py arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | str | `Qwen/Qwen2.5-VL-3B-Instruct` | Model to use; one of `Qwen/Qwen2.5-VL-{1B,3B,7B}-Instruct` |
| `--num-examples` | int | `10` | Number of dataset examples to evaluate |
| `--num-llm-samples` | int | `5` | Number of LLM samples to generate per example |
| `--num-samples` | int | `5` | Number of probabilistic program samples per LLM sample |
| `--temperature` | float | `0.7` | Sampling temperature for the LLM |
| `--save-dir` | str | `out/` | Directory to save results |
| `--direct` | flag | `False` | Omit the per-image instruction from the prompt |
| `--include-arithmetic-operators` | flag | `False` | Extend token support to include arithmetic operators |

For the results in Table 1, use the following commands with models of different sizes:
```bash 
python -m plot2code_new.eval_plot2code --num-examples=132 --temperature=0.0 --num-llm-samples=1 --num-samples=5 --save-dir=out --program-temperature=1.0 --model-name="Qwen/Qwen2.5-VL-3B-Instruct" --include-arithmetic-operators --direct
python -m plot2code_new.eval_plot2code --num-examples=132 --temperature=0.7 --num-llm-samples=5 --num-samples=5 --save-dir=out --program-temperature=1.0 --model-name="Qwen/Qwen2.5-VL-3B-Instruct" --include-arithmetic-operators --direct
```

To reproduce the experimental data for accuracy:
```bash
python -m plot2code_new.eval_foreach_k --num-examples=132 --model-name="Qwen/Qwen2.5-VL-3B-Instruct" --temperature=0.7 --save-dir=out_foreach_k --seed=0 --program-temperature=1.0 --direct --include-arithmetic-operators
```

To reproduce the experimental data for runtime:
```bash
python -m plot2code_new.time --num-examples=20 --model-name="Qwen/Qwen2.5-VL-3B-Instruct" --temperature=0.7 --save-dir=out_time --seed=0 --direct --stride=2 --include-arithmetic-operators --repetitions=5 --compact-logits
```

### 3. CruxEval

**Basic Usage:**

```bash
python3 -m cruxeval_new.eval_cruxeval_input_fast --model Qwen/Qwen2.5-Coder-3B-Instruct --temperature 0.7 --num-llm-samples 5 --num-samples 5 --different-constraint --num-examples 10
```

### cruxeval_new.eval_cruxeval_input_fast.py arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Model to use; one of `Qwen/Qwen2.5-Coder-{0.5,3,7}B-Instruct` |
| `--device` | str | `cuda:0` | Device to load the model on |
| `--num-llm-samples` | int | `5` | Number of LLM samples to generate per example |
| `--num-samples` | int | `5` | Number of subset resamples per LLM sample |
| `--temperature` | float | `0.8` | Sampling temperature for the LLM |
| `--num-examples` | int | `800` | Maximum number of dataset examples to evaluate |
| `--different-constraint` | flag | `False` | Enforce different-value constraint when sampling programs |

For the results in Table 1, use the following commands with models of different sizes:
```bash 
python3 -m cruxeval_new.eval_cruxeval_input_fast --model Qwen/Qwen2.5-Coder-3B-Instruct --temperature 0.0 --num-llm-samples 1 --num-samples 5 --different-constraint
python3 -m cruxeval_new.eval_cruxeval_input_fast --model Qwen/Qwen2.5-Coder-3B-Instruct --temperature 0.7 --num-llm-samples 5 --num-samples 5 --different-constraint
```

To reproduce the experimental data for accuracy and runtime, use the following commands:
```bash
cruxeval_new/accuracy_cruxeval_error.sh <parent_dir>
cruxeval_new/time_cruxeval_simple.sh
```

## Citation

```bibtex
@misc{garg2026probabilisticprogramsthought,
      title={Probabilistic Programs of Thought}, 
      author={Poorva Garg and Renato Lui Geh and Daniel Israel and Todd Millstein and Kyle Richardson and Guy Van den Broeck},
      year={2026},
      eprint={2604.17290},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.17290}, 
}
```

