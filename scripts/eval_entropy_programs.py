import pickle, multiprocessing, os, argparse, io, base64
import tqdm, transformers, numpy as np, torch, PIL
import ppot.utils

class ColorPalette:
    DEFAULT_COLORS = [(255, 58, 32), (91, 140, 90), (14, 121, 178), (245, 183, 0), (51, 24, 50)]

    HTML_FORMAT = "rgba({0}, {1}, {2}, {3:.3f})"
    LATEX_FORMAT = f"\\altcolor"

    def __init__(self, colors: list = DEFAULT_COLORS):
        self.i = 0
        self.colors = colors

    def next(self, alpha: float = 0, fmt: str = HTML_FORMAT) -> str:
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
    </style>
</head>

<body>
"""
def html_token_format(c: str, next_color: str, logits: torch.FloatTensor,
                      tokenizer: transformers.AutoTokenizer, top_k: int = 5) -> str:
    c = c[:-1] if c.endswith('\n') else c
    t = f'<span class="tooltip" style="background-color: {next_color};">{c}<span class="tooltiptext">'
    t += '<table><tr><th>Token</th><th>p(x_i|x_{0:i-1})</th></tr>'
    pr, tok = torch.topk(logits, top_k)
    pr = torch.exp(pr)
    tok = tokenizer.convert_ids_to_tokens(tok)
    for i in range(top_k):
        t += f"<tr><td><pre>{tok[i]}</pre></td><td>{pr[i].item():.5f}</td></tr>"
    t += "</table></span></span>"
    return t
HTML_TAIL = "</body></html>"

def entropy(L: torch.FloatTensor) -> torch.FloatTensor:
    return -torch.sum(L.log_softmax(dim=-1)*L.softmax(dim=-1), dim=-1)

def load(path: str) -> (torch.FloatTensor, torch.LongTensor, torch.FloatTensor):
    with open(path, "rb") as f: R = pickle.load(f)
    input_ids, logits = R["ids"], R["logits"]
    H = entropy(logits)
    return H, input_ids, logits

def html(H: torch.FloatTensor, I: torch.LongTensor, L: torch.FloatTensor,
         tokenizer: transformers.AutoTokenizer, toc_len: int = 0, instruction: str = None,
         ground_truth_img: PIL.Image = None, **kwargs) -> list:
    T = [tokenizer.batch_decode(X) for X in I]
    body = ""
    H_norm = H/torch.max(H, dim=-1, keepdim=True).values
    L_norm = torch.log_softmax(L, dim=-1)
    stop = torch.isin(I, torch.tensor(tokenizer.all_special_ids))
    palette = ColorPalette()
    body += HTML_HEAD
    if toc_len > 0:
        body += "\n<h1>Data instances</h1>\n"
        for i in range(toc_len): body += f' <a href="./{i}.html">[{i}]</a> '
        body += "\n<br><br>"
    if instruction is not None: body += f"\n<h2>Instruction</h2>\n<p>{instruction}</p>\n<br><br>"
    if ground_truth_img is not None:
        body += f"\n<h2>Ground truth</h2>\n{embedd_image(ground_truth_img)}\n<br><br>"
    for i in range(H.shape[0]):
        body += f"\n<h2>Program  {i}</h2><br><hr><br>\n<pre>"
        for j in range(3, H.shape[1]):
            if stop[i,j] or T[i][j] == "```": break
            body += html_token_format(T[i][j], palette.next(H_norm[i,j]), L_norm[i,j], tokenizer, **kwargs)
            if "\n" in T[i][j]: body += "\n"
        body += "</pre>\n<br><hr><br>\n"
    body += HTML_TAIL
    return body

def embedd_image(img: PIL.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    b64 = base64.b64encode(buffer.getvalue())
    return f'<img src="data:image/png;base64, {b64.decode()}"/>'

def foreach_html(n: int, path: str, save_path: str = None, add_instruction: bool = True,
                 add_ground_truth: bool = True, **kwargs) -> list:
    B = []
    if add_instruction:
        data = ppot.utils.prepare_data("TencentARC/Plot2Code", n,
                                       lambda x: "matplotlib" in x["url"], split="test")
        instructions = data["instruction"]
        images = data["image"]
    tokenizer = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").tokenizer
    for i in tqdm.tqdm(range(n)):
        H, I, L = load(f"{path}/data/{i}.pkl")
        B.append(html(H, I, L, tokenizer, toc_len=n, instruction=instructions[i] if add_instruction
                      else None, ground_truth_img=images[i] if add_ground_truth else None, **kwargs))
        if save_path is not None:
            with open(f"{save_path}/{i}.html", "w") as f: f.write(B[-1])
    return B
