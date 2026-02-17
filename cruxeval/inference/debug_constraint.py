#!/usr/bin/env python3

import sys
sys.path.insert(0, '/space/poorvagarg/genPPS/cruxeval/inference')

from resample_generations import LogitsResampler

# Initialize resampler
metadata_path = "/space/poorvagarg/genPPS/cruxeval/model_generations/qwen2.5-coder-0.5b_temp0.8_input/data/metadata.json"
syntax_mapping_path = "/space/poorvagarg/genPPS/cruxeval/inference/syntax_mapping_qwen.json"

resampler = LogitsResampler(metadata_path, syntax_mapping_path)

# Test with f("eat") which keeps coming back identical
sample_key = "sample_303"
original_idx = 1

print("Testing constraint on f('eat') example...")
logits, sample_info = resampler.load_logits(sample_key, original_idx)
target_tokens = sample_info["tokens"]
target_string = sample_info["string"]

print(f"\nOriginal string: {target_string}")
print(f"Original tokens: {target_tokens}")

# Run with constraint and debug
for attempt in range(5):
    print(f"\n{'='*80}")
    print(f"ATTEMPT {attempt + 1}")
    print(f"{'='*80}")
    new_tokens = resampler.resample_subset(logits, target_tokens, temperature=1.0, atleastone_constraint=True, debug=True)
    new_string = resampler.tokenizer.decode(new_tokens)
    
    print(f"\nResult: {new_string}")
    print(f"Tokens: {new_tokens}")
    print(f"IDENTICAL: {new_string == target_string}")
    
    if new_string == target_string and target_tokens == new_tokens:
        print("WARNING: Tokens are EXACTLY the same! Constraint failed!")

