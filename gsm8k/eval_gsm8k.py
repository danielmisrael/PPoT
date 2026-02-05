import argparse, json, os, math, numbers, pickle
import transformers, datasets, torch, tqdm
import ppot.utils, scripts.eval_entropy_programs, ppot.program, ppot.compile

"""This file generates entropy pages for the dataset GSM8k"""

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
    return E, tokenizer([E], return_tensors="pt")

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
        except TimeoutError: return None
        # except TimeoutError: pass
    if errs == num_samples: return None
    return (10**exp if log_transform else exp)/(num_samples-errs)
    # try: 
    # except: return torch.inf

def pass_at_k(P: ppot.program.Program, num_samples: int, gt: float, log_transform: bool = False,
              timeout: int = 120, pp_temperature: float = 1.0, ignore: bool = False, **kwargs) -> bool:
    if ignore: return torch.inf
    S = P.sample(num_samples, as_list=True, t=pp_temperature)
    if P.supp != []:
        assert not P.raw_program.replace("```python", "").replace("```", "") in S
    raw_program = P.raw_program
    S.append(raw_program)
    ans_list = []
    with ppot.utils.timeout(timeout):
        try:
            for i, p in enumerate(S):
                v = execute(p, val_on_err = None, timeout=0, **kwargs)
                ans_list.append(v)
        except TimeoutError: return False, S, ans_list

    for i in ans_list:
        if i is not None:
            if math.isclose(i, gt):
                return True, S, ans_list
    return False, S, ans_list

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
    _, X = template(tok, X)
    if temperature == 0.0: temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else: temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}
    O = model.generate(**X.to(model.device), return_dict_in_generate=True, output_logits=True,
                       repetition_penalty=1.0, top_p=1.0, **temp_kwargs, **kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:,k:].cpu()
    L = torch.concatenate(tuple(x.cpu() for x in O.logits),dim=-1).reshape(O.logits[0].shape[0], len(O.logits), -1)
    S = tok.batch_decode(I, skip_special_tokens=True)
    return I, L, S

def likelihood_evaluator(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer, X: dict,
                        samples: list) -> list:
    text, tokens = template(tok, X)
    scores = []
    for i in samples:
        if "python" in i:
            new_text = text + i
        else:
            new_text = text + "```python" + i + "```"
        # breakpoint()
        tokens = tok([new_text], return_tensors="pt")
        tokens.to(model.device)
        input_ids = tokens["input_ids"]
        # Labels are the same as input_ids for causal language modeling l
        # loss calculation
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
# `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.

        neg_log_likelihood = outputs.loss

        seq_length = input_ids.shape[1]

        total_nll = neg_log_likelihood * seq_length
        scores.append(-total_nll)
    return scores

def compute_scores(R_exp: list, R_llm: list, R_gt: torch.FloatTensor, stdout: bool = False,
                   return_message: bool = False) -> dict:
    R_gt = R_gt[:len(R_exp)]
    # breakpoint()
    R_exp, R_llm = torch.tensor(R_exp, dtype=torch.float), torch.tensor(R_llm, dtype=torch.float)
    S_exp, S_llm = scores(R_exp, R_gt), scores(R_llm, R_gt)
    S_avg_exp, S_avg_llm = torch.nanmean(S_exp).item(), torch.nanmean(S_llm).item()
    E_exp, E_llm = errors(R_exp, R_gt), errors(R_llm, R_gt)
    E_avg_exp, E_avg_llm = torch.nanmean(E_exp, dim=0), torch.nanmean(E_llm, dim=0)

    # if torch.isinf(E_avg_exp).any():
    #     breakpoint()

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
    parser.add_argument("--rule", type=str, default="digits")
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                              device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    # breakpoint()
    # l = []
    # for i in range(151643):
    #     a = tokenizer.decode(i)
    #     if "=" in a or "<" in a or ">" in a:
    #         print(i, a)
    #         l.append((i, a))
    # breakpoint()
    # Load the data
    _, ext = os.path.splitext(args.dataset)
    if ext == ".jsonl":
        with open(args.dataset, "r") as f: J = list(f)
        D = datasets.Dataset.from_list([json.loads(x) for x, _ in zip(J, range(args.num_examples))])
    elif ext == ".json":
        with open(args.dataset, "r") as f:
            J = json.load(f)
        D = datasets.Dataset.from_list(J[:args.num_examples])
    else:
        D = datasets.Dataset.load_from_disk(args.dataset)

    R_exp, R_llm = [], []
    pass_pp, pass_llm = [], []
    P_all = []

    # Dataset and ground truth answer
    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)
    R_gt = []
    os.makedirs(args.llm_cache_path, exist_ok=True)

    if args.rule == "digit":
        rule = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten()]
    elif args.rule == "compare":
        rule = [r"(?<!\|)>|<(?!\|)|<=|>=|==|!="]
        supp = [tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten()]
    elif args.rule == "arithmetic":
        rule = [r"(?<!(?:#.*))([+\-\*/])(?![=/\*])|//|\*\*"]
        supp = [tokenizer(["+", "-", "*", "/", "//", "**", " +", " -", " *", " /", " //", " **"], return_tensors="pt").input_ids.flatten()]
    elif args.rule =="augment":
        rule = [r"[+\-\*/]=|//="]
        supp = [tokenizer(["+=", "-=", "*=", "/=", "//=", " +=", " -=", " *=", " /=", " //="], return_tensors="pt").input_ids.flatten()]
    elif args.rule == "both":
        rule = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])", r"(?<!\|)>|<(?!\|)|<=|>=|==|!="]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten(), 
                tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten()]
    elif args.rule == "all":
        rule = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])", r"(?<!\|)>|<(?!\|)|<=|>=|==|!=", 
                r"(?<!(?:#.*))(?<!\()([+\-\*/])(?![=/\*])|//|\*\*", r"[+\-\*/]=|//=", r"\([+\-]"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten(), 
                tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten(),
                tokenizer(["+", "-", "*", "/", "//", "**", " +", " -", " *", " /", " //", " **"], return_tensors="pt").input_ids.flatten(),
                tokenizer(["+=", "-=", "*=", "/=", "//=", " +=", " -=", " *=", " /=", " //="], return_tensors="pt").input_ids.flatten(),
                tokenizer(["(+", "(-"], return_tensors="pt").input_ids.flatten()]
    else:
        rule = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten()]

    for i, X in enumerate(pbar):
        # if i < 272:
        #     continue
        gt = float(D["answer"][i])
        saved_path = f"{args.llm_cache_path}/{i}.pkl"
        if os.path.isfile(saved_path):
            with open(saved_path, "rb") as f: I, L, S = pickle.load(f)
        else:
            I, L, S = sample(model, tokenizer, X, args.num_llm_samples, temperature=args.temperature,
                             max_new_tokens=args.max_new_tokens)
            with open(saved_path, "wb") as f: pickle.dump((I, L, S), f)

        H = scripts.eval_entropy_programs.entropy(L)

        P, _ = ppot.compile.programs(I[:1,...], L[:1,...], tokenizer, S[:1], only_one=args.uspp, rules = rule, supp = supp)
        P_all.append(P[0])

        R_exp_current = expectation(P[0].to(args.sampling_device), args.num_samples,
                                 log_transform=args.log_transform,
                                 pp_temperature=args.program_temperature)
        R_llm_current = execute(S[0], timeout=args.timeout, val_on_err=None)

        if (R_exp_current is not None) and (R_llm_current is not None):
            R_llm.append(R_llm_current)
            R_exp.append(R_exp_current)
            R_gt.append(gt)
        else:
            # breakpoint()
            continue

        pass_pp_current, S, ans_list = pass_at_k(P[0].to(args.sampling_device), args.num_samples, gt,
                        log_transform=args.log_transform, pp_temperature=args.program_temperature)
        # score_samples = likelihood_evaluator(model, tokenizer, X, S)
        # breakpoint()
        pass_llm_current = torch.isclose(torch.tensor(gt), torch.tensor(R_llm_current, dtype=torch.float))
        pass_pp.append(pass_pp_current)
        pass_llm.append(pass_llm_current)
        if pass_pp_current and not pass_llm_current:
            score_samples = likelihood_evaluator(model, tokenizer, X, S)
            # if torch.argmax(torch.tensor(score_samples)) != torch.tensor(5):
            #     # breakpoint()
            #     pass
            breakpoint()

        # if pass_llm_current and not pass_pp_current:
        #     breakpoint()
        #     pass

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

    pass_pp_rate = torch.mean(torch.tensor(pass_pp, dtype=torch.float))
    pass_llm_rate = torch.mean(torch.tensor(pass_llm, dtype=torch.float))
        
    llm_pot_baseline_acc = torch.sum(torch.isclose(torch.tensor(R_llm, dtype=torch.float), torch.tensor(R_gt, dtype=torch.float)))
    m, out_msg = compute_scores(R_exp, R_llm, R_gt, stdout=True, return_message=True)
    out_msg += f"llm_pot_baseline_Acc: {llm_pot_baseline_acc/len(R_llm)}\n" + f"Number of successful examples: {len(R_llm)}\n"
    out_msg += f"pass rate for LLM: {pass_llm_rate}\n" + f"pass rate for probabilistic program: {pass_pp_rate}\n" 
    os.makedirs(args.report_save_path, exist_ok=True)
    with open(f"{args.report_save_path}/report.pkl", "wb") as f: pickle.dump(m, f)
    with open(f"{args.report_save_path}/report.txt", "w") as f: f.write(out_msg)
