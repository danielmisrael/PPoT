#!/usr/bin/env python3

import json
import pickle
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import transformers
import regex


def get_token_pos(token_ids: torch.LongTensor, processor: transformers.AutoProcessor,
                  rule: str = r'(?<![a-zA-Z_][0-9]*)([0-9])') -> list:
    """
    Get the position of the random variables from the generated programs.

    Inputs:
        token_ids: torch.LongTensor of shape (batch_size, sequence_length)
        processor: the model's transformers.AutoProcessor
    Optional inputs:
        rule: regex rule to identify random variables
    Returns:
        A list of lists containing the position of all random variable tokens

    To verify the algorithm's correctness, you can run the following:

    > toks = [[X[u] for u in j] for X, j in zip(S, J)]
    > ground_truth = [r.findall(''.join(x)) for x in S]
    > assert toks == ground_truth
    > selected_ids = [I[i,j] for i, j in enumerate(J)]
    > assert tokenizer.batch_decode(selected_ids) == [''.join(x) for x in ground_truth]
    """
    # Tokens as strings (here we don't ignore special tokens, which might matter in the future).
    S = [processor.tokenizer.batch_decode(x) for x in token_ids]
    # Length of tokens.
    L = np.array([list(map(len, x)) for x in S])
    # Cumulative sums of lengths, which give the (end) position of the token.
    cL = np.cumsum(L, axis=-1)
    # Compile the regex according to rule. The default rule captures single digits that are not
    # preceded by an alphabetic character or underline. It requires variable width look-behind,
    # which is not supported by the standard re library; instead, we use regex.
    r = regex.compile(rule)
    # Get the (end) position of all regex matches.
    M = [np.array([y.end() for y in r.finditer(''.join(x))]) for x in S]
    # Bisect on cL to find their tokenization position in logarithmic time.
    J = [np.searchsorted(x, y) for x, y in zip(cL, M)]
    return J

def extract_logits_for_string(model, tokenizer, raw_text, target_string):
    """
    Extract logits for a specific target string from the raw generation.
    """
    # Tokenize the raw text
    inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
    
    # Get the input sequence
    input_ids = inputs["input_ids"][0]
    
    # Use get_token_pos to find the position of the target string
    # We'll create a custom regex that matches our exact target string
    target_string_escaped = regex.escape(target_string)
    
    # Create a mock processor object with the tokenizer
    class MockProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
    
    processor = MockProcessor(tokenizer)
    
    # Get token positions using your function
    token_positions = get_token_pos(input_ids.unsqueeze(0), processor, rule=target_string_escaped)
    
    if len(token_positions[0]) == 0:  # No matches found
        print(f"Warning: Target string not found in token sequence")
        return None
    
    # Get the first match (assuming we want the first occurrence)
    # The function returns END positions, so we need to subtract the target length
    end_pos = token_positions[0][0]
    target_tokens = tokenizer.encode(target_string, add_special_tokens=False)
    target_length = len(target_tokens)
    start_pos = end_pos - target_length
    
    print(f"Found target string at token position {start_pos}, length {target_length}")
    
    # Extract logits for the target string
    # logits[i] corresponds to the distribution for token[i+1], so we extract at the same positions
    target_logits = logits[start_pos:start_pos + target_length]
    
    return target_logits.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Extract logits for generated strings")
    parser.add_argument("--generations_path", required=True, help="Path to generations.json")
    parser.add_argument("--generations_raw_path", required=True, help="Path to generations_raw.json")
    parser.add_argument("--model_path", required=False, default="Qwen/Qwen2.5-Coder-0.5B", help="Path to the model")
    parser.add_argument("--output_dir", required=True, help="Output directory for pickle files")
    
    args = parser.parse_args()
    
    # Load the data
    print("Loading generations...")
    with open(args.generations_path, 'r') as f:
        generations = json.load(f)
    
    with open(args.generations_raw_path, 'r') as f:
        generations_raw = json.load(f)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store metadata for all samples
    metadata = {
        "model_path": args.model_path,
        "vocab_size": len(tokenizer),
        "samples": {}
    }
    
    # Process each sample
    sample_count = 0
    
    for sample_key in sorted(generations.keys()):
        if not sample_key.startswith('sample_'):
            continue
            
        sample_idx = int(sample_key.split('_')[1])
        print(f"Processing {sample_key}...")
        
        # Extract logits for all 5 original samples
        for orig_idx in range(5):
            if orig_idx >= len(generations[sample_key]) or orig_idx >= len(generations_raw[sample_key]):
                print(f"  Warning: {sample_key} has fewer than {orig_idx+1} samples, skipping original_{orig_idx}")
                continue
            
            sample_count += 1
            
            # Get the generated string for this original
            generated_string = generations[sample_key][orig_idx]
            
            # Get the raw generation for this original
            raw_generation = generations_raw[sample_key][orig_idx]
            
            # Tokenize the generated string to get tokens
            tokens = tokenizer.encode(generated_string, add_special_tokens=False)
            
            # Extract logits for the generated string
            logits = extract_logits_for_string(model, tokenizer, raw_generation, generated_string)
            
            if logits is not None:
                # Save as pickle file with unique name
                output_path = os.path.join(args.output_dir, f"{sample_idx}_{orig_idx}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(logits, f)
                print(f"  Saved logits for original_{orig_idx} to {output_path}")
                
                # Store metadata with unique key
                metadata_key = f"{sample_key}_orig{orig_idx}"
                metadata["samples"][metadata_key] = {
                    "string": generated_string,
                    "tokens": tokens,
                    "path": f"{sample_idx}_{orig_idx}.pkl"
                }
            else:
                print(f"  Failed to extract logits for {sample_key} original_{orig_idx}")
    
    # Save metadata JSON file
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")
    print(f"Total logits extracted: {sample_count}")

if __name__ == "__main__":
    main()
