import argparse, json, os, math, numbers, pickle
import transformers, datasets, torch, tqdm # type: ignore
import ppot.utils, ppot.program, ppot.compile
from gsm8k.eval_gsm8k_fast import template, sample_llm_compact, sample_pp, get_rule_supp
import time


SUPP_MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct",
               "Qwen/Qwen2.5-Coder-3B-Instruct",
               "Qwen/Qwen2.5-Coder-7B-Instruct"]

def _sample_no_logits(model, tok, X, num_samples, temperature, **kwargs):
    _, enc = template(tok, X)
    if temperature == 0.0: temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else: temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}
    O = model.generate(**enc.to(model.device), return_dict_in_generate=True, output_logits=False,
                       repetition_penalty=1.0, top_p=1.0, **temp_kwargs, **kwargs)
    k = enc.input_ids.numel()
    return O.sequences[:, k:].cpu()

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
    parser.add_argument("--rule", type=str, default="digits")
    parser.add_argument("--different-constraint", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--report-save-path", type=str, default="./gsm8k/report/{model}/{temperature}/")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--iterations", type=int, default=5, help="Number of times to repeat the entire experiment for averaging.")
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
    D = D.select(range(args.num_examples))

    report_save_path = args.report_save_path.format(model=args.model, temperature=args.temperature)
    os.makedirs(report_save_path, exist_ok=True)

    rule, supp = get_rule_supp(args.rule, tokenizer)
    supp_ids = torch.cat(supp).unique()

    # pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)

    total_llm_base_time = 0.0
    total_pp_llm_time = 0.0
    total_pp_compile_time = 0.0
    total_pp_sample_time = {1: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}

    for j in range(args.iterations):
        llm_base_time = 0.0
        pp_llm_time = 0.0
        pp_compile_time = 0.0
        pp_sample_time = {1: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}
        pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)
        for i, X in enumerate(pbar):

            start = time.time()
            _sample_no_logits(model, tokenizer, X, args.num_llm_samples,
                          temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            end = time.time()
            llm_base_time += end - start

            start = time.time()
            I, L, S = sample_llm_compact(model, tokenizer, X, args.num_llm_samples, supp_ids,
                                         temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            end = time.time()
            pp_llm_time += end - start

            start = time.time()
            P, _ = ppot.compile.programs(I, L, tokenizer, S, only_one=args.uspp, rules=rule, supp=supp)
            end = time.time()
            pp_compile_time += end - start

            for num_samples in pp_sample_time:
                start = time.time()
                PPS = sample_pp(P, num_samples, args.program_temperature, diff_constraint=args.different_constraint,
                                debug=args.debug)
                end = time.time()
                pp_sample_time[num_samples] += end - start

        total_llm_base_time += llm_base_time / args.num_examples
        total_pp_llm_time += pp_llm_time / args.num_examples
        total_pp_compile_time += pp_compile_time / args.num_examples
        for num_samples in pp_sample_time:
            total_pp_sample_time[num_samples] += pp_sample_time[num_samples] / args.num_examples

    out_msg = f"Number of examples: {len(D)}\n\n"
    out_msg += f"Number of iterations: {args.iterations}\n\n"
    out_msg += f"Average LLM (no logits) time: {total_llm_base_time / args.iterations}\n"
    out_msg += f"Average PP LLM sampling (compact) time: {total_pp_llm_time / args.iterations}\n"
    out_msg += f"Average PP Compilation time: {total_pp_compile_time / args.iterations}\n"
    for num_samples in total_pp_sample_time:
        out_msg += f"Average PP sampling time for {num_samples} samples: {total_pp_sample_time[num_samples] / args.iterations}\n"
    for num_samples in total_pp_sample_time:
        out_msg += f"Average Total PP time for {num_samples} samples: {(total_pp_llm_time + total_pp_compile_time + total_pp_sample_time[num_samples]) / args.iterations}\n\n"

    with open(f"{report_save_path}/time_{args.num_llm_samples}_{args.suffix}.txt", "w") as f:
        f.write(out_msg)
