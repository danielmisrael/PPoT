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
    f"any comments. \n\nProblem: {question}"

def template(tok: transformers.AutoTokenizer, X: dict) -> transformers.BatchEncoding:
    if "Qwen2.5-Coder" in tok.name_or_path:
        E = tok.apply_chat_template([{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                     {"role": "user", "content": PROMPT(**X)}],
                                    tokenize=False, add_generation_prompt=True)
    else: raise NotImplementedError
    return E, tok([E], return_tensors="pt")

def sample_llm(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer, X: dict,
           num_samples: int, temperature: float = None, output_logits = False, **kwargs) -> (torch.LongTensor, torch.FloatTensor, list):
    _, X = template(tok, X)
    if temperature == 0.0: temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else: temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}

    O = model.generate(**X.to(model.device), return_dict_in_generate=True, output_logits=output_logits,
                       repetition_penalty=1.0, top_p=1.0, **temp_kwargs, **kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:,k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)
    if output_logits:
        L = torch.concatenate(tuple(x.cpu() for x in O.logits),dim=-1).reshape(O.logits[0].shape[0], len(O.logits), -1)
    else: L = None

    return I, L, S

def execute(P: str, val_on_err = None, timeout: float = 10) -> float:
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

def pass_at_k(S: list, timeout: int, gt: float) -> bool:
    for i in S:
        try:
            answer = execute(i, timeout=timeout, val_on_err=None)
        except: continue
        if answer is None: continue
        try:
            pass_at_k = math.isclose(gt, answer)
            if pass_at_k: return True
        except ValueError:
            continue
        except OverflowError:
            continue
        # pass_at_k = torch.isclose(torch.tensor(gt), torch.tensor(answer, dtype=torch.float))
    return False

def sample_pp(P: list, num_samples: int, pp_temperature: float = 1.0, ignore: bool = False, diff_constraint = False,
              debug: bool = False, **kwargs) -> bool:
    # TODO: Parallelize this
    S = []
    for i in P:
        programs = i.sample(num_samples, as_list=True, t=pp_temperature, constraint=diff_constraint)
        if debug and i.supp != []:
            assert not i.raw_program.replace("```python", "").replace("```", "") in programs
        S.extend(programs)
        S.append(i.raw_program)
    return S

def get_rule_supp(rule: str, tokenizer: transformers.AutoTokenizer) -> tuple:
    rule_list = []
    supp = []

    if "digit" in rule or "all" in rule:
        rule_list.append(r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])")
        supp.append(tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten())
    
    if "compare" in rule or "all" in rule:
        rule_list.append(r"(?<!\|)>|<(?!\|)|<=|>=|==|!=")
        supp.append(tokenizer(["<=", ">=", "==", ">", "<", "!=", " <=", " >=", " ==", " >", " <", " !="], return_tensors="pt").input_ids.flatten())
    
    if "arithmetic" in rule or "all" in rule:
        rule_list.extend([r"(?<!(?:#.*))(?<!\()([+\-\*/])(?![=/\*])|//(?!=)|\*\*", r"\([+\-]"])
        supp.extend([tokenizer(["+", "-", "*", "/", "//", "**", " +", " -", " *", " /", " //", " **"], return_tensors="pt").input_ids.flatten(),
                     tokenizer(["(+", "(-"], return_tensors="pt").input_ids.flatten()])
        
    if "augment" in rule or "all" in rule:
        rule_list.append(r"[+\-\*/]=|//=")
        supp.append(tokenizer(["+=", "-=", "*=", "/=", "//=", " +=", " -=", " *=", " /=", " //="], return_tensors="pt").input_ids.flatten())
    
    if rule_list == []:
        rule_list = [r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])"]
        supp = [tokenizer(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], return_tensors="pt").input_ids.flatten()]

    return rule_list, supp


SUPP_MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct", 
               "Qwen/Qwen2.5-Coder-3B-Instruct", 
               "Qwen/Qwen2.5-Coder-7B-Instruct"]

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
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--uspp", default=False, action="store_true")
    parser.add_argument("--sampling-device", type=str, default="cuda:0")
    parser.add_argument("--rule", type=str, default="digits")
    parser.add_argument("--different-constraint", default=False, action="store_true")
    parser.add_argument("--save-html", default=False, action="store_true")
    parser.add_argument("--llm-cache", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
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
        with open(args.dataset, "r") as f: J = json.load(f)
        D = datasets.Dataset.from_list(J[:args.num_examples])
    else:
        D = datasets.Dataset.load_from_disk(args.dataset)

    pass_pp, pass_llm = [], []

    # Dataset, creating directories, getting rules
    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)

    entropy_save_path = f"/space/poorvagarg/genPPS/gsm8k/entropy/{args.model}/{args.temperature}/"
    report_save_path = f"/space/poorvagarg/genPPS/gsm8k/report/{args.model}/{args.temperature}/"
    llm_cache_path = f"/space/poorvagarg/genPPS/gsm8k/generations/{args.model}/{args.temperature}/"
    os.makedirs(entropy_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    rule, supp = get_rule_supp(args.rule, tokenizer)
    
    for i, X in enumerate(pbar):
        gt = float(D["answer"][i])
        saved_path = f"{llm_cache_path}/{i}.pkl"
        if os.path.isfile(saved_path):
            with open(saved_path, "rb") as f: 
                I, L, S = pickle.load(f)
                I, L, S = I[:args.num_llm_samples, ...], L[:args.num_llm_samples, ...], S[:args.num_llm_samples] 
        else:
            I, L, S = sample_llm(model, tokenizer, X, args.num_llm_samples, temperature=args.temperature,
                             max_new_tokens=args.max_new_tokens, output_logits=True)
            if args.llm_cache:
                with open(saved_path, "wb") as f: pickle.dump((I, L, S), f)

        if args.save_html:
            H = scripts.eval_entropy_programs.entropy(L)

        pass_llm_current = pass_at_k(S, timeout=args.timeout, gt=gt)
        pass_llm.append(pass_llm_current)

        P, _ = ppot.compile.programs(I[:,...], L[:,...], tokenizer, S[:], only_one=args.uspp, rules = rule, supp = supp)
        PPS = sample_pp(P, args.num_samples, args.program_temperature, diff_constraint=args.different_constraint,
                        debug=args.debug)
        pass_pp_current = pass_at_k(PPS, timeout=args.timeout, gt=gt) # May have to change the sampling device
        pass_pp.append(pass_pp_current)

        if pass_pp_current and not pass_llm_current:
            breakpoint()

        if args.save_html:
            html = scripts.eval_entropy_programs.html(P[0].entropy_ph.unsqueeze(0), I, L, tokenizer, toc_len=len(D),
                                                  instruction=PROMPT(**X),
                                                  ground_truth_text=f"Expected answer: {X['answer']}")
            with open(f"{entropy_save_path}/{i}.html", "w") as f: f.write(html)

        pass_pp_rate = torch.mean(torch.tensor(pass_pp, dtype=torch.float))
        pass_llm_rate = torch.mean(torch.tensor(pass_llm, dtype=torch.float))
        pbar.set_postfix({"pass_pp_rate": pass_pp_rate,
                          "pass_llm_rate": pass_llm_rate})
        
    out_msg = f"Number of examples: {len(D)}"
    out_msg += f"pass rate for LLM: {pass_llm_rate}\n" + f"pass rate for probabilistic program: {pass_pp_rate}\n" 
    with open(f"{report_save_path}/report.txt", "w") as f: f.write(out_msg)
