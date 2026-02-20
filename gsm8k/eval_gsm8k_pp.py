import argparse, json, os, math, numbers, pickle
import transformers, datasets, torch, tqdm
import ppot.utils, scripts.eval_entropy_programs, ppot.program, ppot.compile
import time

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

def pass_at_k_llm(S: list, timeout: int, gt: float) -> bool:
    for i in S:
        try:
            llm_answer = execute(i, timeout=timeout, val_on_err=None)
        except TimeoutError:
            continue
        if llm_answer is None:
            continue
        pass_at_k = torch.isclose(torch.tensor(gt), torch.tensor(llm_answer, dtype=torch.float))
        if pass_at_k:
            return True
    return False

def pass_at_k(P: list, num_samples: int, gt: float, log_transform: bool = False,
              timeout: int = 120, pp_temperature: float = 1.0, ignore: bool = False,
              diff_constraint=False, **kwargs) -> bool:
    if ignore: return torch.inf
    S = []
    for i in P:
        programs = i.sample(num_samples, as_list=True, t=pp_temperature, constraint=diff_constraint)
        if i.supp != []:
            assert not i.raw_program.replace("```python", "").replace("```", "") in programs
        S.extend(programs)
        S.append(i.raw_program)

    ans_list = []
    with ppot.utils.timeout(timeout):
        try:
            for i, p in enumerate(S):
                v = execute(p, val_on_err = None, timeout=0, **kwargs)
                ans_list.append(v)
        except TimeoutError: pass

    for i in ans_list:
        if i is not None:
            try:
                if math.isclose(i, gt):
                    return True, S, ans_list
            except OverflowError:
                if math.isclose(gt%1, 0.0):
                    if i == gt:
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

def sample(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer, X: dict,
           num_samples: int, temperature: float = None, **kwargs) -> (torch.LongTensor, torch.FloatTensor, list):
    _, X = template(tok, X)
    if temperature == 0.0: temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else: temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}
    start = time.time()
    O = model.generate(**X.to(model.device), return_dict_in_generate=True, output_logits=True,
                       repetition_penalty=1.0, top_p=1.0, **temp_kwargs, **kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:,k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)

    L = torch.concatenate(tuple(x.cpu() for x in O.logits),dim=-1).reshape(O.logits[0].shape[0], len(O.logits), -1)
    end = time.time()
    # print(f"Time taken to generate {num_samples} samples: {end - start}")
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

def get_rule_supp(rule: str) -> tuple:
    if rule == "digit":
        rule_list = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten()]
    elif rule == "compare":
        rule_list = [r"(?<!\|)>|<(?!\|)|<=|>=|==|!="]
        supp = [tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten()]
    elif rule == "arithmetic":
        rule_list = [r"(?<!(?:#.*))(?<!\()([+\-\*/])(?![=/\*])|//(?!=)|\*\*"]
        supp = [tokenizer(["+", "-", "*", "/", "//", "**", " +", " -", " *", " /", " //", " **"], return_tensors="pt").input_ids.flatten()]
    elif rule =="augment":
        rule_list = [r"[+\-\*/]=|//="]
        supp = [tokenizer(["+=", "-=", "*=", "/=", "//=", " +=", " -=", " *=", " /=", " //="], return_tensors="pt").input_ids.flatten()]
    elif rule == "both":
        rule_list = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])", r"(?<!\|)>|<(?!\|)|<=|>=|==|!="]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten(), 
                tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten()]
    elif rule == "all":
        rule_list = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])", r"(?<!\|)>|<(?!\|)|<=|>=|==|!=", 
                r"(?<!(?:#.*))(?<!\()([+\-\*/])(?![=/\*])|//(?!=)|\*\*", r"[+\-\*/]=|//=", r"\([+\-]"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten(), 
                tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten(),
                tokenizer(["+", "-", "*", "/", "//", "**", " +", " -", " *", " /", " //", " **"], return_tensors="pt").input_ids.flatten(),
                tokenizer(["+=", "-=", "*=", "/=", "//=", " +=", " -=", " *=", " /=", " //="], return_tensors="pt").input_ids.flatten(),
                tokenizer(["(+", "(-"], return_tensors="pt").input_ids.flatten()]
    else:
        rule_list = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten()]

    return rule_list, supp


SUPP_MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]

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
    parser.add_argument("--different-constraint", default=False, action="store_true")
    parser.add_argument("--save-html", default=False, action="store_true")
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
        D = datasets.Dataset.from_list(J[:args.num_examples])
    else:
        D = datasets.Dataset.load_from_disk(args.dataset)

    # Initialization
    R_exp, R_llm = [], []
    pass_pp, pass_llm = [], []
    P_all = []

    # Dataset, creating directories, getting rules
    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)
    R_gt = []
    file_save_suffix = f"{args.temperature}_{args.num_llm_samples}"
    os.makedirs(args.llm_cache_path + file_save_suffix, exist_ok=True)
    rule, supp = get_rule_supp(args.rule)
    
    total_time = 0
    for i, X in enumerate(pbar):
        gt = float(D["answer"][i])
        saved_path = f"{args.llm_cache_path +file_save_suffix}/{i}.pkl"
        start = time.time()
        I, L, S = sample(model, tokenizer, X, args.num_llm_samples, temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens)

        # Trying to compile programs for the whole batch
        P, _ = ppot.compile.programs(I[:,...], L[:,...], tokenizer, S[:], only_one=args.uspp, rules = rule, supp = supp)
        P = [j.to(args.sampling_device) for j in P]

        S_pp = []
        for j in P:
            programs = j.sample(args.num_samples, as_list=True, t=args.program_temperature, 
                                constraint=args.different_constraint)
            S_pp.extend(programs)
            S_pp.append(j.raw_program)
        end = time.time()
        total_time += end - start

        pass_llm_current = pass_at_k_llm(S, timeout=args.timeout, gt=gt)
        pass_llm.append(pass_llm_current)
        pass_pp_current = pass_at_k_llm(S_pp, timeout=args.timeout, gt=gt)
        pass_pp.append(pass_pp_current)

        # pass_pp_current, PPS, ans_list = pass_at_k([i.to(args.sampling_device) for i in P], args.num_samples, gt,
        #                 log_transform=args.log_transform, pp_temperature=args.program_temperature, 
        #                 diff_constraint=args.different_constraint)
        # pass_pp.append(pass_pp_current)
        
        # with open(f"{args.entropy_save_path + file_save_suffix}/{i}.html", "w") as f: f.write(html)
        pass_pp_rate = torch.mean(torch.tensor(pass_pp, dtype=torch.float))
        pass_llm_rate = torch.mean(torch.tensor(pass_llm, dtype=torch.float))
        pbar.set_postfix({"pass_pp_rate": pass_pp_rate,
                          "pass_llm_rate": pass_llm_rate})
        
    time_per_example = (total_time)/len(pass_llm)
    with open(f"pp_time_{args.model}_only_sampling.csv", "a") as f: f.write(f"{args.num_llm_samples}, {time_per_example}\n")
        
    llm_pot_baseline_acc = torch.sum(torch.isclose(torch.tensor(R_llm, dtype=torch.float), torch.tensor(R_gt, dtype=torch.float)))
    # m, out_msg = compute_scores(R_exp, R_llm, R_gt, stdout=True, return_message=True)
    out_msg = ""
    out_msg += f"llm_pot_baseline_Acc: {llm_pot_baseline_acc/len(R_llm)}\n" + f"Number of successful examples: {len(R_llm)}\n"
    out_msg += f"pass rate for LLM: {pass_llm_rate}\n" + f"pass rate for probabilistic program: {pass_pp_rate}\n" 
    os.makedirs(args.report_save_path + file_save_suffix, exist_ok=True)
    # with open(f"{args.report_save_path + file_save_suffix}/report.pkl", "wb") as f: pickle.dump(f)
    with open(f"{args.report_save_path + file_save_suffix}/report.txt", "w") as f: f.write(out_msg)
