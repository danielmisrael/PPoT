# CRUXEval: Code Reasoning, Understanding, and Execution Evaluation

<p align="center">
    <a href="https://crux-eval.github.io/">🏠 Home Page</a> •
    <a href="#-getting-started">🔥 Quick Start</a> •
    <a href="https://crux-eval.github.io/leaderboard.html">🏆 Leaderboard</a> •
    <a href="https://crux-eval.github.io/demo.html">🔎 Sample Explorer</a> •
    <a href="#-citation">📜 Citation</a> •
    <a href="#-acknowledgements">🙏 Acknowledgements</a>
</p>

![image](https://github.com/facebookresearch/cruxeval/assets/7492257/b1fecb48-2355-4d60-9d97-1d09e793bd82)

CRUXEval (**C**ode **R**easoning, **U**nderstanding, and e**X**ecution **Eval**uation) is a benchmark of 800 Python functions and input-output pairs. The benchmark consists of two tasks, CRUXEval-I (input prediction) and CRUXEval-O (output prediction). 


## ⚙️ Setup and Installation
To clone the repository, run
```
git clone git@github.com:facebookresearch/cruxeval.git
cd cruxeval
```

Setup the environment using the following instructions:
```
conda create -n cruxeval2 python=3.11
conda activate cruxeval2
conda install pytorch torchvision torchaudio -c pytorch
pip install datasets vllm==0.8
pip install accelerate
```

## To get the primary program generations

```
cd inference
./scripts/custom_run_input_prediction.sh
CUDA_VISIBLE_DEVICES=0 python3 extract_logits.py --generations_path ../model_generations/qwen2.5-coder-0.5b_temp0.8_input/generations.json --generations_raw_path ../model_generations_raw/qwen2.5-coder-0.5b_temp0.8_input/generations_raw.json --model_path Qwen/Qwen2.5-Coder-0.5B --output_dir ../model_generations/qwen2.5-coder-0.5b_temp0.8_input/data
```

```
python3 resample_generations.py --n_resamples 5 --atleastone_constraint --output_path ../model_generations/qwen2.5-coder-0.5b_temp0.8_input/resample_generations.json
cd ../evaluation
python evaluate_generations.py     --generations_path ../model_generations/qwen2.5-coder-0.5b_temp0.8_input/resample_generations.json     --scored_results_path ../model_generations/qwen2.5-coder-0.5b_temp0.8_input/resample_generations_scored.json     --mode input
```

