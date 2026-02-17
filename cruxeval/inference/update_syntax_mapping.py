#!/usr/bin/env python3
"""
Update syntax_mapping.json to include all tokens that begin with open brackets
and all tokens that end with closing brackets.
"""
import json
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    
    # Find all tokens that BEGIN with open brackets
    open_bracket_tokens = {
        '[': [],
        '(': [],
        '{': []
    }
    
    # Find all tokens that END with closing brackets
    close_bracket_tokens = {
        ')': [],
        ']': [],
        '}': []
    }
    
    print("Finding tokens that begin with open brackets and end with closing brackets...")
    
    for token_id in range(len(tokenizer)):
        token_str = tokenizer.decode([token_id])
        
        # Check if starts with open bracket
        for bracket in ['[', '(', '{']:
            if token_str.startswith(bracket):
                open_bracket_tokens[bracket].append(token_id)
        
        # Check if ends with closing bracket
        for bracket in [')', ']', '}']:
            if token_str.endswith(bracket):
                close_bracket_tokens[bracket].append(token_id)
    
    # Print statistics
    print("\nOpen bracket tokens (starts with):")
    for bracket, tokens in open_bracket_tokens.items():
        print(f"  '{bracket}': {len(tokens)} tokens")
        print(f"    First 10: {[tokenizer.decode([t]) for t in tokens[:10]]}")
    
    print("\nClose bracket tokens (ends with):")
    for bracket, tokens in close_bracket_tokens.items():
        print(f"  '{bracket}': {len(tokens)} tokens")
        print(f"    First 10: {[tokenizer.decode([t]) for t in tokens[:10]]}")
    
    # Load existing syntax mapping
    with open('/home/disrael/genPPS/cruxeval/inference/syntax_mapping.json', 'r') as f:
        syntax_mapping = json.load(f)
    
    # Update with new token lists
    syntax_mapping["open_bracket_tokens"] = open_bracket_tokens
    syntax_mapping["close_bracket_tokens"] = close_bracket_tokens
    
    # Save updated mapping
    with open('/home/disrael/genPPS/cruxeval/inference/syntax_mapping.json', 'w') as f:
        json.dump(syntax_mapping, f, indent=2)
    
    print("\nUpdated syntax_mapping.json saved!")

if __name__ == "__main__":
    main()

