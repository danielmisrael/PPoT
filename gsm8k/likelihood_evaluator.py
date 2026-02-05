import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

model_name = "gpt2" # GPT-2 is a decoder-only autoregressive mode

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sequence_to_score = "This is a sample sentence to score."

inputs = tokenizer(sequence_to_score, return_tensors="pt")
input_ids = inputs["input_ids"]
# Labels are the same as input_ids for causal language modeling l
# loss calculation
labels = input_ids.clone()

outputs = model(input_ids=input_ids, labels=labels)
# `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.

neg_log_likelihood = outputs.loss

seq_length = input_ids.shape[1]

total_nll = neg_log_likelihood * seq_length

torch.exp(-total_nll)
# tensor(2.5986e-18, grad_fn=<ExpBackward0>)