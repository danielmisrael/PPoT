import argparse, json, os, math, pickle, csv, time
import transformers, datasets, torch, tqdm  # type: ignore
import numpy as np
import ppot.utils
from gsm8k.eval_gsm8k_fast import pass_at_k
from typing import Optional, Any

"""Direct-PyMC baseline for GSM8K (rebuttal).

Generates a PyMC program per LLM call (RefineStat-inspired prompt), overrules
the LLM-written pm.sample(...) call with our canonical one, executes the model,
and scores the posterior against the GSM8K ground truth using pass@k with the
same (num_llm_samples, num_samples) budget as the PPoT pipeline.
"""

def PROMPT(question: Optional[str] = None, **kwargs) -> str:
    return (
        "Generate a PyMC model that computes the answer to the math word problem "
        "below. Your output must be a single fenced ```python``` code block defining "
        "a complete Bayesian model with appropriate priors and likelihood inside a "
        "`with pm.Model() as m:` block, with `answer = pm.Deterministic(\"answer\", "
        "...)` capturing the posterior variable for the numeric answer, and then "
        "sample the posterior using, `pm.sample(1000, tune=1000, chains=4, "
        "return_inferencedata=True)`. Do not include any extra commentary or text "
        "outside the code. Follow best practices for expert-level Bayesian modeling."
        f"\n\nProblem: {question}"
    )


def template(tok: transformers.AutoTokenizer, X: dict) -> tuple[str, transformers.BatchEncoding]:
    """Same as gsm8k.eval_gsm8k_fast.template but resolves PROMPT from this module."""
    if "Qwen2.5-Coder" in tok.name_or_path:  # type: ignore
        E = tok.apply_chat_template([{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},  # type: ignore
                                     {"role": "user", "content": PROMPT(**X)}],
                                    tokenize=False, add_generation_prompt=True)
    else: raise NotImplementedError
    return E, tok([E], return_tensors="pt")  # type: ignore


def sample_llm_text(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer,
                    X: dict, num_samples: int, temperature: Optional[float] = None, **kwargs) -> list:
    """Generate `num_samples` LLM completions and return decoded strings."""
    _, enc = template(tok, X)
    temp_kwargs: dict[str, Any]
    if temperature == 0.0:
        temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else:
        temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}
    O = model.generate(**enc.to(model.device), return_dict_in_generate=True, output_logits=False,  # type: ignore
                       repetition_penalty=1.0, top_p=1.0, **temp_kwargs, **kwargs)
    k = enc.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    return tok.batch_decode(I, skip_special_tokens=True)  # type: ignore


SUPP_MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct",
               "Qwen/Qwen2.5-Coder-3B-Instruct",
               "Qwen/Qwen2.5-Coder-7B-Instruct"]

PYMC_DRAWS = 500
PYMC_TUNE = 1000
PYMC_CHAINS = 4


def _match_paren(s: str, open_idx: int) -> int:
    """Given index of '(' in s, return index of matching ')'. -1 on failure.
    Tracks single/double quote state so parens inside strings are ignored."""
    depth = 0
    i = open_idx
    n = len(s)
    quote: Optional[str] = None
    while i < n:
        c = s[i]
        if quote is not None:
            if c == "\\" and i + 1 < n:
                i += 2; continue
            if c == quote: quote = None
        else:
            if c == "'" or c == '"': quote = c
            elif c == "(": depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0: return i
        i += 1
    return -1


_LHS_RE = __import__("re").compile(r"^(\s*)([A-Za-z_][A-Za-z_0-9]*)\s*=\s*$")

def overrule_sample(code: str, seed: int) -> str:
    """Replace every pm.sample(...) call in `code` with our canonical call.

    Preserves the original indentation and (when present) the LHS variable
    name. So `trace = pm.sample(1000, chains=4)` becomes
    `trace = pm.sample(draws=500, tune=1000, chains=2, ...)`, leaving any
    follow-up code that uses `trace` still working. Bare `pm.sample(...)` is
    bound to `idata`. If no pm.sample call is present at all, appends an
    `idata = pm.sample(...)` line at column 0.
    """
    canonical_args = (f"draws={PYMC_DRAWS}, tune={PYMC_TUNE}, "
                      f"chains={PYMC_CHAINS}, cores=1, progressbar=False, "
                      f"return_inferencedata=True, random_seed={seed}")
    needle = "pm.sample("
    out_parts: list = []
    pos = 0
    found_any = False
    while True:
        idx = code.find(needle, pos)
        if idx < 0:
            out_parts.append(code[pos:])
            break
        found_any = True
        open_paren = idx + len(needle) - 1
        close_paren = _match_paren(code, open_paren)
        if close_paren < 0:
            out_parts.append(code[pos:])
            break
        line_start = code.rfind("\n", 0, idx) + 1
        before_call = code[line_start:idx]
        # If the chars immediately before pm.sample on this line are `<name> = `,
        # preserve them; otherwise drop and use `idata = `.
        lhs_match = _LHS_RE.match(before_call)
        if lhs_match:
            indent = lhs_match.group(1)
            lhs = lhs_match.group(2)
        else:
            # Take the leading whitespace as the indent; drop anything else.
            indent = ""
            for ch in before_call:
                if ch in (" ", "\t"): indent += ch
                else: break
            lhs = "idata"
        out_parts.append(code[pos:line_start])
        out_parts.append(f"{indent}{lhs} = pm.sample({canonical_args})")
        pos = close_paren + 1
    new_code = "".join(out_parts)
    if not found_any:
        if not new_code.endswith("\n"): new_code += "\n"
        new_code += f"idata = pm.sample({canonical_args})\n"
    return new_code


def execute_pymc(P: str, timeout: int, seed: int, val_on_err=None) -> Optional[np.ndarray]:
    """Execute a generated PyMC program string. Returns the posterior array for
    the variable named `answer` (length ~= PYMC_DRAWS * PYMC_CHAINS), or
    `val_on_err` on any failure."""
    try: start = P.index("```python") + 9
    except: start = None
    try: end = P.rindex("```")
    except: end = None
    code = P[start:end] if (start is not None and end is not None) else P
    code = overrule_sample(code, seed)
    G: dict = {}
    L: dict = {}
    A: Optional[np.ndarray] = val_on_err
    with ppot.utils.timeout(timeout):
        try: exec(code, G, L)
        except TimeoutError:
            if timeout <= 0: raise TimeoutError("Triggered timeout")
            # exec aborted, but anything bound before the abort is still in L/G.
        except: pass
        # Search the (possibly partially-populated) namespace for an
        # InferenceData and an `answer` posterior. This runs even if exec
        # raised, so post-`with` analysis code that crashes doesn't lose us
        # the posterior that was bound just before.
        try:
            import arviz as az  # type: ignore
            idata = None
            for ns in (L, G):
                for v in ns.values():
                    if isinstance(v, az.InferenceData):
                        idata = v
                        break
                if idata is not None: break
            if idata is None: return val_on_err
            arr = np.asarray(idata.posterior["answer"].values).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0: return val_on_err
            A = arr
        except: A = val_on_err
    return A


def pymc_candidates(A: Optional[np.ndarray], num_samples: int, rng: np.random.Generator) -> list:
    """[round(median(A))] + [round(a) for a in num_samples random draws].
    Returns a list of length num_samples+1 with None where extraction failed."""
    if A is None or len(A) == 0:
        return [None] * (num_samples + 1)
    out: list = [float(round(float(np.median(A))))]
    replace = len(A) < num_samples
    draws = rng.choice(A, size=num_samples, replace=replace)
    out.extend(float(round(float(x))) for x in draws)
    return out


def pass_at_k_numeric(C: list, gt: float) -> bool:
    """pass@k over a flat list of numeric candidates (or None for failures)."""
    for a in C:
        if a is None: continue
        try:
            if math.isclose(gt, a): return True
        except (ValueError, OverflowError): continue
    return False


def pass_at_k_numeric_answers(C_nested: list, gt: float) -> list:
    """[[bool per candidate] per LLM sample] — analog of pass_at_k_answers."""
    out: list = []
    for row in C_nested:
        row_out: list = []
        for a in row:
            ok = False
            if a is not None:
                try: ok = math.isclose(gt, a)
                except (ValueError, OverflowError): ok = False
            row_out.append(ok)
        out.append(row_out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=SUPP_MODELS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-llm-samples", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-examples", type=int, default=10000000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--report-save-path", type=str,
                        default="./gsm8k/report/{model}/{temperature}/pymc/")
    parser.add_argument("--llm-cache-path", type=str,
                        default="./gsm8k/generations/{model}/{temperature}/pymc/")
    args = parser.parse_args()

    ppot.utils.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                              device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load the data
    _, ext = os.path.splitext(args.dataset)
    if ext == ".jsonl":
        with open(args.dataset, "r") as f: J = list(f)
        D = datasets.Dataset.from_list([json.loads(x) for x, _ in zip(J, range(args.num_examples))])
    elif ext == ".json":
        with open(args.dataset, "r") as f: J = json.load(f)
        D = datasets.Dataset.from_list(J[:args.num_examples])
    else:
        D = datasets.Dataset.load_from_disk(args.dataset)

    pass_pp, pass_llm = [], []
    times: list = []

    report_save_path = args.report_save_path.format(model=args.model, temperature=args.temperature)
    llm_cache_path = args.llm_cache_path.format(model=args.model, temperature=args.temperature)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)
    for i, X in enumerate(pbar):
        gt = float(D["answer"][i])

        t0 = time.time()
        S = sample_llm_text(model, tokenizer, X, args.num_llm_samples,
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        t_llm = time.time() - t0

        pass_llm_current = pass_at_k(S, timeout=args.timeout, gt=gt)
        pass_llm.append(pass_llm_current)

        posteriors: list = []
        t_pymc = 0.0
        for j, s in enumerate(S):
            t1 = time.time()
            A = execute_pymc(s, timeout=args.timeout, seed=args.seed + i * 1000 + j)
            t_pymc += time.time() - t1
            posteriors.append(A)

        cand_nested = [pymc_candidates(A, args.num_samples, rng) for A in posteriors]
        cand_flat = [c for row in cand_nested for c in row]
        pass_pp_current = pass_at_k_numeric(cand_flat, gt=gt)
        pass_pp.append(pass_pp_current)

        times.append({"llm": t_llm, "pymc": t_pymc, "total": t_llm + t_pymc})

        with open(f"{llm_cache_path}/{i}.pkl", "wb") as f:
            pickle.dump({"S": S, "posteriors": posteriors, "times": times[-1],
                         "pass_llm": pass_llm_current, "pass_pp": pass_pp_current,
                         "candidates": cand_nested, "gt": gt}, f)

        pass_pp_rate = torch.mean(torch.tensor(pass_pp, dtype=torch.float))
        pass_llm_rate = torch.mean(torch.tensor(pass_llm, dtype=torch.float))
        mean_t_total = sum(t["total"] for t in times) / len(times)
        pbar.set_postfix({"pass_pp_rate": pass_pp_rate,
                          "pass_llm_rate": pass_llm_rate,
                          "mean_t_total": mean_t_total})

    mean_t_llm = sum(t["llm"] for t in times) / len(times)
    mean_t_pymc = sum(t["pymc"] for t in times) / len(times)
    mean_t_total = sum(t["total"] for t in times) / len(times)

    out_msg = (
        f"Number of examples: {len(D)}\n"
        f"pass rate for LLM: {pass_llm_rate}\n"
        f"pass rate for direct PyMC: {pass_pp_rate}\n\n"
        f"Wall-clock per example (averaged over {len(D)}):\n"
        f"  LLM generation:   {mean_t_llm} s\n"
        f"  PyMC inference:   {mean_t_pymc} s\n"
        f"  Total:            {mean_t_total} s\n\n"
        f"PyMC constants: draws={PYMC_DRAWS} tune={PYMC_TUNE} chains={PYMC_CHAINS} cores=1\n"
    )
    with open(f"{report_save_path}/report.txt", "w") as f: f.write(out_msg)

    with open(f"{report_save_path}/times.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example", "llm", "pymc", "total"])
        for i, t in enumerate(times):
            w.writerow([i, t["llm"], t["pymc"], t["total"]])
