import argparse, json, os, math, numbers, pickle
import transformers, datasets, torch, tqdm
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

def timer(index: int):
    wall_clock = time.time(),
    cpu_time = time.process_time(),
    gpu_time = gpu_time_vector[index].record()
    return wall_clock[0], cpu_time[0], gpu_time

def update(d, w, c, g):
    d["wall_clock"] += w
    d["cpu"] += c
    d["gpu"] += g

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
    parser.add_argument("--report-save-path", type=str, default="/space/poorvagarg/genPPS/gsm8k/report/{model}/{temperature}/")
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

    report_save_path = args.report_save_path.format(model=args.model, temperature=args.temperature)
    os.makedirs(report_save_path, exist_ok=True)

    rule, supp = get_rule_supp(args.rule, tokenizer)
    supp_ids = torch.cat(supp).unique()

    llm_time = {"wall_clock": 0, "cpu" : 0, "gpu": 0}
    pp_time_llm, pp_time_compile, pp_time_pp = ({"wall_clock": 0, "cpu" : 0, "gpu": 0},
                                                {"wall_clock": 0, "cpu" : 0, "gpu": 0},
                                                {"wall_clock": 0, "cpu" : 0, "gpu": 0})
    gpu_time_vector = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    pbar = tqdm.tqdm(D, desc="Example", dynamic_ncols=True)
    for i, X in enumerate(pbar):
        w0, c0, g0 = timer(0)
        _sample_no_logits(model, tokenizer, X, args.num_llm_samples,
                          temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(llm_time, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

        w0, c0, g0 = timer(0)
        I, L, S = sample_llm_compact(model, tokenizer, X, args.num_llm_samples, supp_ids,
                                     temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(pp_time_llm, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

        w0, c0, g0 = timer(0)
        P, _ = ppot.compile.programs(I, L, tokenizer, S, only_one=args.uspp, rules=rule, supp=supp)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(pp_time_compile, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

        w0, c0, g0 = timer(0)
        PPS = sample_pp(P, args.num_samples, args.program_temperature, diff_constraint=args.different_constraint,
                        debug=args.debug)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(pp_time_pp, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

    out_msg = f"Number of examples: {len(D)}\n\n"
    for i in llm_time:
        out_msg += f"{i}\n"
        out_msg += f"LLM (no logits): {llm_time[i]/len(D)}\n"
        out_msg += f"PP LLM sampling (compact): {pp_time_llm[i]/len(D)}\n"
        out_msg += f"PP Compilation: {pp_time_compile[i]/len(D)}\n"
        out_msg += f"PP sampling: {pp_time_pp[i]/len(D)}\n"
        out_msg += f"Total PP time: {(pp_time_llm[i] + pp_time_compile[i] + pp_time_pp[i])/len(D)}\n\n"

    with open(f"{report_save_path}/time_{args.num_llm_samples}_{args.num_samples}.txt", "w") as f:
        f.write(out_msg)
