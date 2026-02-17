#!/usr/bin/env python3
"""
Visualize entropy of logits for CruxEval generations.
Adapted from eval_entropy_programs.py
"""
import pickle
import json
import os
import argparse
import torch
import transformers
from tqdm import tqdm

class ColorPalette:
    DEFAULT_COLORS = [(255, 58, 32), (91, 140, 90), (14, 121, 178), (245, 183, 0), (51, 24, 50)]
    HTML_FORMAT = "rgba({0}, {1}, {2}, {3:.3f})"

    def __init__(self, colors: list = None):
        self.i = 0
        self.colors = colors or self.DEFAULT_COLORS

    def next(self, alpha: float = 0, fmt: str = None) -> str:
        fmt = fmt or self.HTML_FORMAT
        r = fmt.format(*self.colors[self.i], alpha)
        self.i = (self.i + 1) % len(self.colors)
        return r

HTML_HEAD = """
<!DOCTYPE html>

<html lang="en-US">

<head>
    <meta charset="utf-8">

    <!-- Fira Sans, Karla, Inconsolata Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Sans&family=Karla&family=Inconsolata&display=swap" rel="stylesheet">

    <style>
        html {
            font-family: 'Fira Sans', 'Karla', 'Inconsolata', sans-serif;
            font-size: 100%;
            margin: auto;
            padding: 20px;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            text-align: left;
            border-radius: 5px;
            position: absolute;
            z-index: 1;
            background-color: #ffffc7;
            padding: 5px 5px 5px 5px;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
        }
        .tooltiptext td {
            padding: 0 10px;
            text-align: left;
        }
        .tooltiptext th {
            padding: 0 10px;
            text-align: left;
        }
        pre {
            white-space: pre-wrap;
            font-family: 'Inconsolata', monospace;
        }
        code {
            font-family: 'Inconsolata', monospace;
        }
    </style>
</head>

<body>
"""

HTML_TAIL = "</body></html>"

def html_token_format(c: str, next_color: str, logits: torch.FloatTensor,
                      tokenizer: transformers.AutoTokenizer, top_k: int = 5) -> str:
    """Format a token with tooltip showing top-k predictions."""
    # Escape HTML special characters
    c = c.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    c = c.replace('\n', '\\n')  # Show newlines explicitly
    
    t = f'<span class="tooltip" style="background-color: {next_color};"><code>{c}</code><span class="tooltiptext">'
    t += '<table><tr><th>Token</th><th>p(x_i|x_&lt;i)</th></tr>'
    
    # Get top-k predictions
    pr, tok = torch.topk(logits, top_k)
    pr = torch.exp(pr)  # Convert log probs to probs
    tok_strs = tokenizer.convert_ids_to_tokens(tok.tolist())
    
    for i in range(top_k):
        tok_str = tok_strs[i].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        t += f"<tr><td><code>{tok_str}</code></td><td>{pr[i].item():.5f}</td></tr>"
    t += "</table></span></span>"
    return t

def entropy(L: torch.FloatTensor) -> torch.FloatTensor:
    """Calculate entropy of logits."""
    return -torch.sum(L.log_softmax(dim=-1) * L.softmax(dim=-1), dim=-1)

def load_logits(path: str) -> torch.FloatTensor:
    """Load logits from pickle file."""
    with open(path, "rb") as f:
        logits = pickle.load(f)
    # Convert to torch tensor if it's a numpy array
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(logits)
    return logits

def generate_html(sample_key: str, orig_idx: int, string: str, tokens: list, 
                  logits: torch.FloatTensor, tokenizer: transformers.AutoTokenizer,
                  metadata: dict, toc_samples: list = None, top_k: int = 5) -> str:
    """Generate HTML visualization for a single sample."""
    # Calculate entropy
    H = entropy(logits)
    H_norm = H / torch.max(H)
    L_norm = torch.log_softmax(logits, dim=-1)
    
    # Start HTML
    body = HTML_HEAD
    
    # Add table of contents if provided
    if toc_samples:
        body += "\n<h1>CruxEval Samples</h1>\n"
        body += "<p>Click on a sample to view its logits visualization:</p>\n"
        for s_key, o_idx in toc_samples:
            body += f' <a href="./{s_key}_orig{o_idx}.html">[{s_key}_orig{o_idx}]</a> '
        body += "\n<br><br><hr><br>\n"
    
    # Add sample info
    body += f"\n<h2>Sample: {sample_key}, Original: {orig_idx}</h2>\n"
    body += f"<p><strong>Generated string:</strong> <code>{string}</code></p>\n"
    body += f"<p><strong>Number of tokens:</strong> {len(tokens)}</p>\n"
    body += f"<p><strong>Model:</strong> {metadata['model_path']}</p>\n"
    body += "<br>\n"
    
    # Visualize tokens with entropy
    body += "<h3>Token-by-token visualization (hover for top-k predictions)</h3>\n"
    body += "<p><em>Color intensity indicates entropy (higher = more uncertain)</em></p>\n"
    body += "<br>\n<pre>"
    
    palette = ColorPalette()
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    
    for j in range(len(tokens)):
        body += html_token_format(decoded_tokens[j], palette.next(H_norm[j].item()), 
                                 L_norm[j], tokenizer, top_k=top_k)
    
    body += "</pre>\n<br><hr><br>\n"
    
    # Add entropy statistics
    body += f"<h3>Entropy Statistics</h3>\n"
    body += f"<p><strong>Mean entropy:</strong> {H.mean().item():.4f}</p>\n"
    body += f"<p><strong>Max entropy:</strong> {H.max().item():.4f}</p>\n"
    body += f"<p><strong>Min entropy:</strong> {H.min().item():.4f}</p>\n"
    
    body += HTML_TAIL
    return body

def main():
    parser = argparse.ArgumentParser(description="Visualize logits entropy for CruxEval")
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to metadata.json")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing .pkl logits files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for HTML files")
    parser.add_argument("--n_samples", type=int, default=10,
                       help="Number of samples to visualize")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top predictions to show in tooltip")
    parser.add_argument("--sample_indices", type=str, default=None,
                       help="Comma-separated list of sample indices (e.g., '0,1,2')")
    
    args = parser.parse_args()
    
    # Load metadata
    print(f"Loading metadata from {args.metadata_path}...")
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load tokenizer
    print(f"Loading tokenizer for {metadata['model_path']}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(metadata['model_path'], trust_remote_code=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which samples to visualize
    if args.sample_indices:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    else:
        sample_indices = list(range(args.n_samples))
    
    # Collect samples to visualize (all 5 originals for each sample)
    samples_to_viz = []
    for idx in sample_indices:
        for orig_idx in range(5):
            sample_key = f"sample_{idx}"
            metadata_key = f"{sample_key}_orig{orig_idx}"
            if metadata_key in metadata['samples']:
                samples_to_viz.append((sample_key, orig_idx, metadata_key))
    
    print(f"Generating HTML for {len(samples_to_viz)} samples...")
    
    # Generate table of contents list
    toc_samples = [(s_key, o_idx) for s_key, o_idx, _ in samples_to_viz]
    
    # Generate HTML for each sample
    for sample_key, orig_idx, metadata_key in tqdm(samples_to_viz):
        sample_info = metadata['samples'][metadata_key]
        
        # Load logits
        logits_path = os.path.join(args.data_dir, sample_info['path'])
        logits = load_logits(logits_path)
        
        # Generate HTML
        html_content = generate_html(
            sample_key=sample_key,
            orig_idx=orig_idx,
            string=sample_info['string'],
            tokens=sample_info['tokens'],
            logits=logits,
            tokenizer=tokenizer,
            metadata=metadata,
            toc_samples=toc_samples,
            top_k=args.top_k
        )
        
        # Save HTML
        output_path = os.path.join(args.output_dir, f"{sample_key}_orig{orig_idx}.html")
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    print(f"\nHTML files saved to {args.output_dir}")
    print(f"Open {args.output_dir}/{samples_to_viz[0][0]}_orig{samples_to_viz[0][1]}.html to start browsing")

if __name__ == "__main__":
    main()

