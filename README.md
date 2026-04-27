# Probabilistic Programs of Thought

[![arXiv](https://img.shields.io/badge/arXiv-2604.17290-blue?link=https%3A%2F%2F2604.17290)](https://arxiv.org/abs/2604.17290)


## Setup

### 1. Clone the repository and submodules
```bash
git clone https://github.com/PoorvaGarg/genPPS.git
cd genPPS
```

### 2. Install dependencies
```bash
conda create -n ppot python=3.11
conda activate ppot
pip install -r requirements.txt
```

### For GSM8k

Basic Usage:
```bash
python3 -m gsm8k.eval_gsm8k --dataset gsm8k/gsm8k.json --model Qwen/Qwen2.5-Coder-0.5B-Instruct --temperature 0.7 --num-samples 2 --rule all --different-constraint --num-llm-samples 2 --num-examples 2
```

### gsm8k.eval_gsm8k_fast.py arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | required | Path to dataset file (`.json`, `.jsonl`, or HuggingFace disk format) |
| `--model` | str | required | Model to use; one of `Qwen/Qwen2.5-Coder-{0.5,3,7}B-Instruct` |
| `--device` | str | `cuda:0` | Device to load the model on (e.g. `cuda:1`) |
| `--num-llm-samples` | int | `4` | Number of LLM samples to generate per example |
| `--num-samples` | int | `5` | Number of probabilistic program samples per LLM sample |
| `--temperature` | float | `0.3` | Sampling temperature for the LLM |
| `--num-examples` | int | `10000000` | Maximum number of dataset examples to evaluate |
| `--max-new-tokens` | int | `2048` | Maximum new tokens to generate per sample |
| `--seed` | int | `0` | Random seed |
| `--timeout` | int | `10` | Execution timeout in seconds for generated code |
| `--program-temperature` | float | `1.0` | Sampling temperature for probabilistic programs |
| `--uspp` | flag | `False` | Use only a single probabilistic program per LLM sample |
| `--sampling-device` | str | `cuda:0` | Device used during program sampling |
| `--rule` | str | `digits` | Token rule(s) for support: `digit`, `compare`, `arithmetic`, `augment`, or `all` |
| `--different-constraint` | flag | `False` | Enforce different-value constraint when sampling programs |
| `--save-html` | flag | `False` | Save per-example entropy HTML visualizations |
| `--llm-cache` | flag | `False` | Cache raw LLM generations to disk |
| `--debug` | flag | `False` | Enable debug assertions |
| `--entropy-save-path` | str | `gsm8k/entropy/{model}/{temperature}/` | Directory for entropy HTML files |
| `--report-save-path` | str | `gsm8k/report/{model}/{temperature}/` | Directory for evaluation report |
| `--llm-cache-path` | str | `gsm8k/generations/{model}/{temperature}/` | Directory for cached LLM generations |

To reproduce the experimental data for accuracy and runtime, use the following commands:

```bash
gsm8k/accuracy_gsm8k_error.sh <parent_dir>
gsm8k/time_gsm8k_simple.sh
```

### For Plot2Code

Basic Usage:
```bash
python3 -m plot2code_new.eval_plot2code --model-name Qwen/Qwen2.5-VL-3B-Instruct --num-examples 10 --temperature 0.7 --num-llm-samples 5 --num-samples 5
```

### plot2code_new.eval_plot2code.py arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | str | `Qwen/Qwen2.5-VL-3B-Instruct` | Model to use; one of `Qwen/Qwen2.5-VL-{1B,3B,7B}-Instruct` |
| `--num-examples` | int | `10` | Number of dataset examples to evaluate |
| `--num-llm-samples` | int | `5` | Number of LLM samples to generate per example |
| `--num-samples` | int | `5` | Number of probabilistic program samples per LLM sample |
| `--temperature` | float | `0.7` | Sampling temperature for the LLM |
| `--save-dir` | str | `out/` | Directory to save results |
| `--seed` | int | `0` | Random seed |
| `--timeout` | int | `10` | Execution timeout in seconds for generated code |
| `--program-temperature` | float | `1.0` | Sampling temperature for probabilistic programs |
| `--uspp` | flag | `False` | Use only a single probabilistic program per LLM sample |
| `--sampling-device` | str | `cuda:0` | Device to load the model on |
| `--direct` | flag | `False` | Omit the per-image instruction from the prompt |
| `--include-arithmetic-operators` | flag | `False` | Extend token support to include arithmetic operators |
| `--no-model-loading` | flag | `False` | Skip model loading (useful for re-running evaluation from cache) |
| `--pause-for-inspection` | flag | `False` | Drop into a breakpoint when a probabilistic program outperforms the LLM |

### For CruxEval

Basic Usage:
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
| `--program-temperature` | float | `1.0` | Sampling temperature for probabilistic programs |
| `--num-examples` | int | `800` | Maximum number of dataset examples to evaluate |
| `--max-new-tokens` | int | `512` | Maximum new tokens to generate per sample |
| `--seed` | int | `0` | Random seed |
| `--timeout` | int | `3` | Execution timeout in seconds for correctness checks |
| `--different-constraint` | flag | `False` | Enforce different-value constraint when sampling programs |
| `--save-html` | flag | `False` | Save per-example entropy HTML visualizations |
| `--llm-cache` | flag | `False` | Cache raw LLM generations to disk |
| `--debug` | flag | `False` | Enable debug assertions |

To reproduce the experimental data for accuracy and runtime, use the following commands:

```bash
cruxeval_new/accuracy_cruxeval_error.sh <parent_dir>
gsm8k/time_cruxeval_simple.sh
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

