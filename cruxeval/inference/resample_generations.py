#!/usr/bin/env python3

import pickle
import json
import torch
from transformers import AutoTokenizer
import argparse
import os
from tqdm import tqdm

class LogitsResampler:
    
    def __init__(self, metadata_path, syntax_mapping_path):
        self.metadata_path = metadata_path
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        with open(syntax_mapping_path, 'r') as f:
            self.syntax_mapping = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata["model_path"])
        
    
    def load_logits(self, sample_key, original_idx):

        metadata_key = f"{sample_key}_orig{original_idx}"
        
        if metadata_key not in self.metadata["samples"]:
            raise FileNotFoundError(f"No logits available for {metadata_key}")
        
        sample_info = self.metadata["samples"][metadata_key]
        logits_path = sample_info["path"]
        
        metadata_dir = os.path.dirname(self.metadata_path)
        full_logits_path = os.path.join(metadata_dir, logits_path)
        
        with open(full_logits_path, 'rb') as f:
            logits = pickle.load(f)
        
        return logits, sample_info
    
    def resample_subset(self, logits, target_tokens, temperature=1.0, atleastone_constraint=False):
        
        # breakpoint()
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float16)
        
        new_tokens = target_tokens.copy()
        
        probs = torch.softmax(logits / temperature, dim=-1)
        vocab_size = probs.shape[-1]
        seq_len = len(target_tokens)
        # breakpoint()
        # Create suffix masks: at position i, only tokens from target_tokens[i:] are allowed
        target_tokens_tensor = torch.tensor(target_tokens, dtype=torch.long)
        # print(target_tokens_tensor.shape)
        pos_indices = torch.arange(seq_len)
        position_mask = pos_indices.unsqueeze(1) <= pos_indices.unsqueeze(0)  # (seq_len, seq_len)
        
        vocab_expanded = torch.arange(vocab_size, dtype=torch.float16).unsqueeze(1)  # (vocab_size, 1)
        target_expanded = target_tokens_tensor.unsqueeze(0)  # (1, seq_len)
        matches = (vocab_expanded == target_expanded)  # (vocab_size, seq_len)
        
        # (L, 1, L) & (1, V, L) => (L, V, L), then reduce
        # breakpoint()
        suffix_masks = torch.any(position_mask.unsqueeze(1) & matches.unsqueeze(0), dim=2).to(torch.float16)  # (seq_len, vocab_size)
        
        masked_probs = probs * suffix_masks
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        
        # Compute atleastone_constraint adjustment using Bayes rule
        cumprod = None
        if atleastone_constraint:
            target_match_probs = masked_probs[torch.arange(seq_len), target_tokens_tensor]
            # cumprod[i] = P(x_i = target[i]) * ... * P(x_n = target[n])
            cumprod = torch.flip(torch.cumprod(torch.flip(target_match_probs, [0]), dim=0), [0])
        # print(cumprod)
        
        i = 1  # Start at 1 to preserve first token
        original_positions = list(range(len(target_tokens)))
        constraint_satisfied = torch.isnan(cumprod).any() if cumprod is not None else True
        
        while i < len(new_tokens) - 1:
            original_pos = original_positions[i]
            if original_pos >= len(probs):
                break
            
            token_probs = masked_probs[original_pos].clone()
            
            if atleastone_constraint and not constraint_satisfied:
                target_token_at_pos = target_tokens[original_pos]
                future_cumprod = cumprod[original_pos + 1].item()
                # print(future_cumprod)
                constraint_factor = 1.0 - future_cumprod + 1e-7
                # print(constraint_factor)
                token_probs[target_token_at_pos] = token_probs[target_token_at_pos] * constraint_factor
            
            # print(token_probs.sum())
            if token_probs.sum() > 1e-7: 
                token_probs = token_probs / token_probs.sum()
            # print(token_probs)
            sampled_token = torch.multinomial(token_probs, num_samples=1).item()
            
            # Check if sampled token appears in future positions
            found_at = i
            for j in range(i+1, len(new_tokens)): #Poorva: changed this to i from i+1
                if new_tokens[j] == sampled_token:
                    found_at = j
                    break
            
            # assert found_at is not None
            
            if atleastone_constraint:
                constraint_satisfied = True
            new_tokens = new_tokens[:i] + [sampled_token] + new_tokens[found_at+1:]
            original_positions = original_positions[:i] + [original_positions[found_at]] + original_positions[found_at+1:]
            
            i += 1
        
        return new_tokens
    
    def generate_resamples(self, sample_key, original_sample_idx, n_resamples=20, temperature=1.0, atleastone_constraint=False):

        logits, sample_info = self.load_logits(sample_key, original_sample_idx)
        target_tokens = sample_info["tokens"]
        target_string = sample_info["string"]
        
        resampled_strings = []
        
        for _ in range(n_resamples):
            new_tokens = self.resample_subset(logits, target_tokens, temperature, atleastone_constraint=atleastone_constraint)
            new_string = self.tokenizer.decode(new_tokens)
            resampled_strings.append(new_string)
        
        return resampled_strings
    
    def create_resampled_generations(self, original_generations_path, n_resamples=20, temperature=1.0, 
                                   output_path=None, limit=None, atleastone_constraint=False):

        with open(original_generations_path, 'r') as f:
            original_generations = json.load(f)
        
        new_generations = {}
        
        total_samples = len(original_generations) if limit is None else min(limit, len(original_generations))
        print(f"Processing {total_samples} samples (out of {len(original_generations)} total)...")
        print(f"Generating {n_resamples} resamples per sample with temperature {temperature}")
        
        available_samples = set()
        for key in self.metadata["samples"].keys():
            if key.endswith("_orig0"):
                sample_key = key.replace("_orig0", "")
                available_samples.add(sample_key)
        
        processed_count = 0
        # breakpoint()
        for sample_key in tqdm(original_generations.keys()):
            # if processed_count < 448:
            #     processed_count +=1 
            #     continue
            
            # print(processed_count)
            # print(sample_key)
            if not sample_key.startswith('sample_') or ("162" in sample_key) or ("342" in sample_key):
                continue
            
            if limit is not None and processed_count >= limit:
                break
            
            if sample_key not in available_samples:
                print(f"Skipping {sample_key} (no logits available)")
                first_5_original = original_generations[sample_key][:5]
                new_generations[sample_key] = first_5_original
                processed_count += 1
                continue
            
            original_samples = original_generations[sample_key]
            first_5_original = original_samples[:5]
            
            all_samples = []
            for orig_idx in range(5):
                all_samples.append(first_5_original[orig_idx])
                resampled_samples = self.generate_resamples(sample_key, orig_idx, n_resamples, temperature, atleastone_constraint=atleastone_constraint)
                all_samples.extend(resampled_samples)
            
            new_generations[sample_key] = all_samples
            
            processed_count += 1
        
        if output_path is None:
            original_dir = os.path.dirname(original_generations_path)
            filename_parts = [f"generations_t{temperature}_n{n_resamples}"]
            if limit is not None:
                filename_parts.append(f"limit{limit}")
            if atleastone_constraint:
                filename_parts.append("atleastone")
            filename = "_".join(filename_parts) + ".json"
            output_path = os.path.join(original_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write('{\n')
            
            sample_keys = list(new_generations.keys())
            for key_idx, sample_key in enumerate(sample_keys):
                samples = new_generations[sample_key]
                f.write(f'  "{sample_key}": [\n')
                
                for i, sample in enumerate(samples):
                    escaped_sample = json.dumps(sample)
                    needs_comma = i < len(samples) - 1
                    comma = ',' if needs_comma else ''
                    
                    if i % (n_resamples + 1) == 0 and i > 0:
                        f.write('\n')
                    
                    f.write(f'    {escaped_sample}{comma}\n')
                
                needs_comma = key_idx < len(sample_keys) - 1
                comma = ',' if needs_comma else ''
                f.write(f'  ]{comma}\n')
            
            f.write('}\n')
        
        print(f"\nSaved resampled generations to {output_path}")
        total_per_sample = 5 * (1 + n_resamples) if len(new_generations) > 0 else 0
        print(f"Total samples per data point: {total_per_sample} (5 originals + {n_resamples} resamples each = 5 * (1 + {n_resamples}))")
        
        return new_generations

def main():
    parser = argparse.ArgumentParser(description="Resample generations using logits")
    parser.add_argument("--n_resamples", type=int, default=20, help="Number of resamples per sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--atleastone_constraint", action="store_true", 
                       help="Apply constraint to guarantee at least one token differs from target")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path")
    parser.add_argument("--metadata_path", type=str, 
                       default="/space/poorvagarg/genPPS/cruxeval/model_generations/qwen2.5-coder-0.5b_temp0.8_input/data/metadata.json",
                       help="Path to metadata.json")
    parser.add_argument("--syntax_mapping_path", type=str,
                       default="/space/poorvagarg/genPPS/cruxeval/inference/syntax_mapping_qwen.json",
                       help="Path to syntax_mapping.json")
    parser.add_argument("--original_generations_path", type=str,
                       default="/space/poorvagarg/genPPS/cruxeval/model_generations/qwen2.5-coder-0.5b_temp0.8_input/generations.json",
                       help="Path to original generations.json")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process (default: all)")
    
    args = parser.parse_args()
    
    metadata_path = args.metadata_path
    syntax_mapping_path = args.syntax_mapping_path
    original_generations_path = args.original_generations_path
    
    resampler = LogitsResampler(metadata_path, syntax_mapping_path)
    
    new_generations = resampler.create_resampled_generations(
        original_generations_path=original_generations_path,
        n_resamples=args.n_resamples,
        temperature=args.temperature,
        output_path=args.output_path,
        limit=args.limit,
        atleastone_constraint=args.atleastone_constraint
    )
    
    print(f"\nResampling completed!")
    print(f"Generated {args.n_resamples} resamples per sample")
    print(f"Used temperature: {args.temperature}")

if __name__ == "__main__":
    main()
