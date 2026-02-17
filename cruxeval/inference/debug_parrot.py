#!/usr/bin/env python3

import sys
sys.path.insert(0, '/space/poorvagarg/genPPS/cruxeval/inference')

from resample_generations import LogitsResampler

# Initialize resampler
metadata_path = "/space/poorvagarg/genPPS/cruxeval/model_generations/qwen2.5-coder-0.5b_temp0.8_input/data/metadata.json"
syntax_mapping_path = "/space/poorvagarg/genPPS/cruxeval/inference/syntax_mapping_qwen.json"

resampler = LogitsResampler(metadata_path, syntax_mapping_path)

# Load the specific sample that has the parrot example
sample_key = "sample_299"
original_idx = 1  # The second original (index 1) has 'parrot'

print("Loading logits for sample_299, original_idx=1 (the parrot example)...")
logits, sample_info = resampler.load_logits(sample_key, original_idx)
target_tokens = sample_info["tokens"]
target_string = sample_info["string"]

print(f"\nOriginal string: {target_string}")

print("\n" + "="*80)
print("WITHOUT atleastone_constraint:")
print("="*80)
# Run without constraint
new_tokens = resampler.resample_tokens(logits, target_tokens, temperature=1.0, debug=True)
new_string = resampler.tokenizer.decode(new_tokens)

print(f"\n\nFINAL RESULT (no constraint):")
print(f"Original: {target_string}")
print(f"Resampled: {new_string}")

print("\n\n" + "="*80)
print("WITH atleastone_constraint:")
print("="*80)
# Run with constraint
new_tokens_constrained = resampler.resample_subset(logits, target_tokens, temperature=1.0, atleastone_constraint=True, debug=True)
new_string_constrained = resampler.tokenizer.decode(new_tokens_constrained)

print(f"\n\nFINAL RESULT (with constraint):")
print(f"Original: {target_string}")
print(f"Resampled: {new_string_constrained}")

