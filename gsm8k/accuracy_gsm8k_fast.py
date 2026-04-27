import argparse, json, os, math, numbers, pickle
import transformers, datasets, torch, tqdm # type: ignore
import ppot.utils, scripts.eval_entropy_programs, ppot.program, ppot.compile
from gsm8k.eval_gsm8k_fast import PROMPT, sample_llm_compact, template, execute, pass_at_k, get_rule_supp
import time

def sample_pp(P: list, num_samples: int, pp_temperature: float = 1.0, ignore: bool = False, diff_constraint = False,
              debug: bool = False, **kwargs) -> list:
    S = []
    for j in P:
        programs = [j.raw_program]
        programs.extend(j.sample(40, as_list=True, t=args.program_temperature, 
                            constraint=args.different_constraint))
        S.append(programs)
    return S

def pass_at_k_answers(S: list, timeout:int, gt:float):
    ans_list = []
    for i in S:
        ans_i = []
        for j in i:
            current_answer = False
            try:
                answer = execute(j, timeout=timeout, val_on_err=None)
                match = math.isclose(gt, answer)
                if match: current_answer = True
            except: pass
            ans_i.append(current_answer)
        ans_list.append(ans_i)
    return ans_list


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
    parser.add_argument("--parent-dir", type=str, default="./gsm8k/")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    # ppot.utils.seed(args.seed)

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

    pass_pp: list = []
    pass_llm: list = []

    # Dataset, creating directories, getting rules
    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)

    entropy_save_path = f"{args.parent_dir}/entropy/{args.model}/{args.temperature}/"
    report_save_path = f"{args.parent_dir}/report/{args.model}/{args.temperature}/"
    llm_cache_path = f"{args.parent_dir}/generations/{args.model}/{args.temperature}/"
    os.makedirs(entropy_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    rule, supp = get_rule_supp(args.rule, tokenizer)
    supp_ids = torch.cat(supp).unique()

    pass_rates = [[0 for _ in range(args.num_samples+1)] for _ in range(args.num_llm_samples)]
    
    for i, X in enumerate(pbar):
        gt = float(D["answer"][i])
        saved_path = f"{llm_cache_path}/{i}.pkl"
        I, L, S = sample_llm_compact(model, tokenizer, X, args.num_llm_samples, supp_ids,
                                     temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens)
        if args.llm_cache:
            with open(saved_path, "wb") as f: pickle.dump((I, L, S), f) # type: ignore

        if args.save_html:
            H = scripts.eval_entropy_programs.entropy(L)

        pass_llm_current = pass_at_k(S, timeout=args.timeout, gt=gt)
        pass_llm.append(pass_llm_current)

        P, _ = ppot.compile.programs(I, L, tokenizer, S, only_one=args.uspp, rules = rule, supp = supp)
        PPS = sample_pp(P, args.num_samples, args.program_temperature, diff_constraint=args.different_constraint,
                        debug=args.debug)
        
        # llm_answers = pass_at_k_answers([S], args.timeout, gt)[0]
        # for j in range(1, args.num_llm_samples+1):
        #     if True in llm_answers[:j]:
        #         pass_rates[j-1][0] += 1

        pp_answers = pass_at_k_answers(PPS, args.timeout, gt)
        for j in range(1, args.num_llm_samples+1):
            for k in range(0, args.num_samples+1):
                answer = False
                for l in pp_answers[:j]:
                    if True in l[:k+1]:
                        answer = True
                        break
                pass_rates[j-1][k] += (1 if answer else 0)

    with open(f"{report_save_path}/accuracy_{args.suffix}.csv", "a") as f:
        f.write(f"Number of examples: {len(D)}\n")
        for i in range(1, args.num_llm_samples+1):
            for j in range(0, args.num_samples+1):
                f.write(f"{i}, {j}, {pass_rates[i-1][j]/len(D)}\n")
