import argparse, json, os, time
import transformers, datasets, torch, tqdm
import ppot.utils, ppot.program, ppot.compile
from cruxeval_new.utils_execute import check_correctness
from cruxeval_new.prompts import make_direct_input_prompt
from cruxeval_new.eval_cruxeval_input import template, sample_llm, pass_at_k, sample_pp, cruxeval_input_answer_extractor
from cruxeval_new.eval_cruxeval_input_fast import sample_compact

SUPP_MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

def _sample_no_logits(model, tok, code, output, num_samples, temperature, **kwargs):
    _, enc = template(tok, code, output)
    if temperature == 0.0: temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else: temp_kwargs = {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}
    temp_kwargs.update({"repetition_penalty": 1.0, "top_p": 0.95, "stop_strings": ['[/ANSWER]'],
                        "max_new_tokens": kwargs.pop("max_new_tokens", 769), "tokenizer": tok})
    
    O = model.generate(**enc.to(model.device), return_dict_in_generate=True, output_logits=False,
                       **temp_kwargs, **kwargs)
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
    parser = argparse.ArgumentParser(
        description="Online CruxEval input prediction evaluation with ppot subset resampling")
    parser.add_argument("--model", type=str, required=True, choices=SUPP_MODELS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-llm-samples", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of subset resamples per LLM sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--num-examples", type=int, default=800)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3)
    parser.add_argument("--different-constraint", default=False, action="store_true")
    parser.add_argument("--save-html", default=False, action="store_true")
    parser.add_argument("--llm-cache", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--parent-dir", type=str, default="/space/poorvagarg/genPPS/cruxeval_new/")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--shuffle", default=False, action="store_true")

    args = parser.parse_args()

    # ppot.utils.seed(args.seed)

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = datasets.load_dataset("cruxeval-org/cruxeval", split="test")
    if args.shuffle:
        dataset = dataset.shuffle().select(range(args.num_examples))

    # Main loop
    pass_llm_list, pass_pp_list = [], []
    pbar = tqdm.tqdm(enumerate(dataset), total=min(args.num_examples, len(dataset)),
                     desc="CruxEval Input", dynamic_ncols=True)

    entropy_save_path = f"{args.parent_dir}/entropy/{args.model}/{args.temperature}/"
    report_save_path = f"{args.parent_dir}/report/{args.model}/{args.temperature}/"
    llm_cache_path = f"{args.parent_dir}/generations/{args.model}/{args.temperature}/"
    os.makedirs(entropy_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    llm_time = {"wall_clock": 0, "cpu" : 0, "gpu": 0}
    pp_time_llm, pp_time_compile, pp_time_pp = ({"wall_clock": 0, "cpu" : 0, "gpu": 0}, 
                                                {"wall_clock": 0, "cpu" : 0, "gpu": 0}, 
                                                {"wall_clock": 0, "cpu" : 0, "gpu": 0})
    gpu_time_vector = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    for i, example in pbar:
        if i >= args.num_examples:
            break

        code = example["code"]
        expected_output = example["output"]

        w0, c0, g0 = timer(0)
        I = _sample_no_logits(model, tokenizer, code, expected_output, args.num_llm_samples,
                          temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(llm_time, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

        w0, c0, g0 = timer(0)
        I, compact_scores, token_log_prob, unique_toks, S = sample_compact(
            model, tokenizer, code, expected_output,
            args.num_llm_samples, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(pp_time_llm, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

        w0, c0, g0 = timer(0)
        P, _ = ppot.compile.fast_subset_programs(
            I, compact_scores, token_log_prob, unique_toks, tokenizer,
            answer_extractor=cruxeval_input_answer_extractor)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(pp_time_compile, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

        w0, c0, g0 = timer(0)
        PP_samples = sample_pp(P, args.num_samples, args.program_temperature, args.different_constraint)
        w1, c1, g1 = timer(1)
        torch.cuda.synchronize()
        update(pp_time_pp, w1-w0, c1-c0, gpu_time_vector[0].elapsed_time(gpu_time_vector[1]))

    # Report
    out_msg = f"Number of examples: {args.num_examples}\n\n"
    for i in llm_time:
        out_msg += f"{i}\n"
        out_msg += f"LLM: {llm_time[i]/args.num_examples}\n"
        out_msg += f"PP LLM sampling: {pp_time_llm[i]/args.num_examples}\n"
        out_msg += f"PP Compilation: {pp_time_compile[i]/args.num_examples}\n"
        out_msg += f"PP sampling: {pp_time_pp[i]/args.num_examples}\n"
        out_msg += f"Total PP time: {(pp_time_llm[i] + pp_time_compile[i] + pp_time_pp[i])/args.num_examples}\n\n"
   
    with open(f"{report_save_path}/time_{args.num_llm_samples}_{args.num_samples}_{args.suffix}.txt", "w") as f: 
        f.write(out_msg)
