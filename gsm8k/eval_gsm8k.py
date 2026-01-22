import argparse, json, os, math, numbers, pickle
import transformers, datasets, torch, tqdm
import ppot.utils, scripts.eval_entropy_programs, ppot.program, ppot.compile

def PROMPT(question: str = None, unit: str = None, **kwargs) -> str:
    return "Generate a Python function `compute_answer` with no arguments that computes the " \
    "needed calculations and returns a number as the answer to the problem below. There " \
    "should be no comments in the code. Only generate the Python function `compute_answer`, " \
    "with no explanations and no user input. The function should show intermediate computations " \
    "in the program but without " \
    f"any comments. \n\nProblem: {question}" + \
        (f" The answer should be in the following unit of measurement: {unit}." if unit is not None else '')

def template(tok: transformers.AutoTokenizer, X: dict) -> transformers.BatchEncoding:
    if "Qwen2.5-Coder" in tokenizer.name_or_path:
        E = tok.apply_chat_template([{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                     {"role": "user", "content": PROMPT(**X)}],
                                    tokenize=False, add_generation_prompt=True)
    else: raise NotImplementedError
    return tokenizer([E], return_tensors="pt")

def expectation(P: ppot.program.Program, num_samples: int, log_transform: bool = False,
                timeout: int = 120, pp_temperature: float = 1.0, ignore: bool = False, **kwargs) -> float:
    if ignore: return torch.inf
    S = P.sample(num_samples, as_list=True, t=pp_temperature)
    exp, errs = 0, 0
    with ppot.utils.timeout(timeout):
        try:
            for i, p in enumerate(S):
                v = execute(p, val_on_err = None, timeout=0, **kwargs)
                errs += v is None
                if v is not None:
                    try: exp += math.log10(v) if log_transform else v
                    except ValueError: errs += 1
        except TimeoutError: return torch.inf
    if errs == num_samples: return torch.inf
    try: return (10**exp if log_transform else exp)/(num_samples-errs)
    except: return torch.inf

def execute(P: str, val_on_err = torch.inf, timeout: float = 10) -> float:
    y, L, G = None, {}, {}
    try: start = P.index("```python")+9
    except: start = None
    try: end = P.rindex("```")
    except: end = None
    with ppot.utils.timeout(timeout):
        try:
            exec(P[start:end], G, L)
            y = L["compute_answer"]()
            if not isinstance(y, numbers.Number) or isinstance(y, complex): y = val_on_err
        except TimeoutError:
            y = val_on_err
            if timeout <= 0: raise TimeoutError("Triggered timeout")
        except: y = val_on_err
    return y

# Maybe have to take the expectation by transforming values to log_10 and then transform back.
def scores(x: list, y: list) -> torch.FloatTensor:
    "`x` is estimate, `y` is goal."
    if not torch.is_tensor(x): x = torch.tensor(x)
    if not torch.is_tensor(y): y = torch.tensor(y)
    I = torch.isinf(x)
    s = torch.max(torch.zeros(len(x)), 1.0-torch.abs(torch.log10(x/y))/3)
    s[I] = 0.0
    return s

def errors(x: list, y: list) -> torch.FloatTensor:
    if not torch.is_tensor(x): x = torch.tensor(x)
    if not torch.is_tensor(y): y = torch.tensor(y)
    return torch.concatenate((torch.abs(x-y).reshape(-1, 1), torch.sqrt(torch.abs(x*x-y*y)).reshape(-1, 1)), dim=-1)

def sample(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer, X: dict,
           num_samples: int, temperature: float = None, **kwargs) -> (torch.LongTensor, torch.FloatTensor, list):
    X = template(tok, X)
    if temperature == 0.0: temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else: temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}
    O = model.generate(**X.to(model.device), return_dict_in_generate=True, output_logits=True,
                       repetition_penalty=1.0, top_p=1.0, **temp_kwargs, **kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:,k:].cpu()
    L = torch.concatenate(tuple(x.cpu() for x in O.logits),dim=-1).reshape(O.logits[0].shape[0], len(O.logits), -1)
    S = tok.batch_decode(I, skip_special_tokens=True)
    return I, L, S

def compute_scores(R_exp: list, R_llm: list, R_gt: torch.FloatTensor, stdout: bool = False,
                   return_message: bool = False) -> dict:
    R_gt = R_gt[:len(R_exp)]
    # breakpoint()
    R_exp, R_llm = torch.tensor(R_exp, dtype=torch.float), torch.tensor(R_llm, dtype=torch.float)
    S_exp, S_llm = scores(R_exp, R_gt), scores(R_llm, R_gt)
    S_avg_exp, S_avg_llm = torch.nanmean(S_exp).item(), torch.nanmean(S_llm).item()
    E_exp, E_llm = errors(R_exp, R_gt), errors(R_llm, R_gt)
    E_avg_exp, E_avg_llm = torch.nanmean(E_exp, dim=0), torch.nanmean(E_llm, dim=0)

    M = {"last score(E_p[X])": S_exp[-1].item(), "last score(LLM)": S_llm[-1].item(),
         "avg score(E_p[X])": S_avg_exp, "avg score(LLM)": S_avg_llm,
         "last abs_error(E_p[X])": E_exp[-1,0].item(), "last abs_error(LLM)": E_llm[-1,0].item(),
         "avg abs_error(E_p[X])": E_avg_exp[0].item(), "avg abs_error(LLM)": E_avg_llm[0].item(),
         "last ms_error(E_p[X])": E_exp[-1,1].item(), "last ms_error(LLM)": E_llm[-1,1].item(),
         "avg ms_error(E_p[X])": E_avg_exp[1].item(), "avg ms_error(LLM)": E_avg_llm[1].item(),
         "scores(E_p[X])": S_exp, "scores(LLM)": S_llm, "error(E_p[X])": E_exp,
         "error(score(LLM))": E_llm, "LLM": R_llm, "E_p[X]": R_exp}
    msg = ''
    for k, v in M.items(): msg += f"{k} = {v}\n"
    if stdout: print(msg)
    return (M, msg) if return_message else M

SUPP_MODELS = ["Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=SUPP_MODELS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-llm-samples", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-examples", type=int, default=10000000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--entropy-save-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-save-path", type=str, required=True)
    parser.add_argument("--log-transform", default=False, action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--llm-cache-path", type=str, required=True)
    parser.add_argument("--uspp", default=False, action="store_true")
    parser.add_argument("--sampling-device", type=str, default="cuda:0")
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

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
        with open(args.dataset, "r") as f:
            J = json.load(f)
        D = datasets.Dataset.from_list(J)
    else:
        D = datasets.Dataset.load_from_disk(args.dataset)
    breakpoint()
    R_exp, R_llm = [], []
    P_all = []

    # Dataset and ground truth answer
    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)
    R_gt = torch.tensor(list(map(float, D["answer"])))
    os.makedirs(args.llm_cache_path, exist_ok=True)

    for i, X in enumerate(pbar):
        # Check if generation exists
        saved_path = f"{args.llm_cache_path}/{i}.pkl"
        if os.path.isfile(saved_path):
            with open(saved_path, "rb") as f: I, L, S = pickle.load(f)
        else:
            I, L, S = sample(model, tokenizer, X, args.num_llm_samples, temperature=args.temperature,
                             max_new_tokens=args.max_new_tokens)
            with open(saved_path, "wb") as f: pickle.dump((I, L, S), f)
        H = scripts.eval_entropy_programs.entropy(L)
        P, _ = ppot.compile.programs(I[:1,...], L[:1,...], tokenizer, S[:1], only_one=args.uspp)
        P_all.append(P[0])
        R_exp.append(expectation(P[0].to(args.sampling_device), args.num_samples,
                                 log_transform=args.log_transform,
                                 pp_temperature=args.program_temperature))
        R_llm.append(execute(S[0], timeout=args.timeout))
        m = compute_scores(R_exp, R_llm, R_gt, stdout=False)
        html = scripts.eval_entropy_programs.html(H, I, L, tokenizer, toc_len=len(D),
                                                  instruction=PROMPT(**X),
                                                  ground_truth_text=f"Expected answer: {X['answer']}, LLM answer: {R_llm[-1]}, " \
                                                    f"Probabilistic Program answer: {R_exp[-1]}")
                                                #   return_vals=[R_llm[-1]])
        os.makedirs(args.entropy_save_path, exist_ok=True)
        with open(f"{args.entropy_save_path}/{i}.html", "w") as f: f.write(html)
        pbar.set_postfix({"abs error diff (E-LLM)": m["avg abs_error(E_p[X])"]-m["avg abs_error(LLM)"],
                          "abs mse diff (E-LLM)": m["avg ms_error(E_p[X])"]-m["avg ms_error(LLM)"],})
    m, out_msg = compute_scores(R_exp, R_llm, R_gt, stdout=True, return_message=True)
    os.makedirs(args.report_save_path, exist_ok=True)
    with open(f"{args.report_save_path}/report.pkl", "wb") as f: pickle.dump(m, f)
    with open(f"{args.report_save_path}/report.txt", "w") as f: f.write(out_msg)
