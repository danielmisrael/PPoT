import argparse, json, os, time
import transformers, datasets, torch, tqdm # type: ignore
import ppot.utils, ppot.program, ppot.compile
from cruxeval_new.utils_execute import check_correctness
from cruxeval_new.prompts import make_direct_input_prompt
from cruxeval_new.eval_cruxeval_input_fast import template, sample_compact, pass_at_k, sample_pp, cruxeval_input_answer_extractor

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
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    # ppot.utils.seed(args.seed)

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = datasets.load_dataset("cruxeval-org/cruxeval", split="test")
    dataset = dataset.select(range(args.num_examples))

    # Main loop
    pass_llm_list: list = []
    pass_pp_list: list = []
    

    entropy_save_path = f"{args.parent_dir}/entropy/{args.model}/{args.temperature}/"
    report_save_path = f"{args.parent_dir}/report/{args.model}/{args.temperature}/"
    llm_cache_path = f"{args.parent_dir}/generations/{args.model}/{args.temperature}/"
    os.makedirs(entropy_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    total_llm_base_time = 0.0
    total_pp_llm_time = 0.0
    total_pp_compile_time = 0.0
    total_pp_sample_time = {1: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}

    for j in range(args.iterations):
        llm_base_time = 0.0
        pp_llm_time = 0.0
        pp_compile_time = 0.0
        pp_sample_time = {1: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}
        pbar = tqdm.tqdm(enumerate(dataset), total=min(args.num_examples, len(dataset)),
                     desc="CruxEval Input", dynamic_ncols=True)
        for i, example in pbar:
            if i >= args.num_examples:
                break

            code = example["code"]
            expected_output = example["output"]
            
            start = time.time()
            I = _sample_no_logits(model, tokenizer, code, expected_output, args.num_llm_samples,
                          temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            end = time.time()
            llm_base_time += end - start

            start = time.time()
            I, compact_scores, token_log_prob, unique_toks, S = sample_compact(
                model, tokenizer, code, expected_output,
                args.num_llm_samples, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens)
            end = time.time()
            pp_llm_time += end - start

            start = time.time()
            P, _ = ppot.compile.fast_subset_programs(
                I, compact_scores, token_log_prob, unique_toks, tokenizer,
                answer_extractor=cruxeval_input_answer_extractor)
            end = time.time()
            pp_compile_time += end - start

            for num_samples in pp_sample_time:
                start = time.time()
                PP_samples = sample_pp(P, num_samples, args.program_temperature, args.different_constraint)
                end = time.time()
                pp_sample_time[num_samples] += end - start
        
        total_llm_base_time += llm_base_time / args.num_examples
        total_pp_llm_time += pp_llm_time / args.num_examples
        total_pp_compile_time += pp_compile_time / args.num_examples
        for num_samples in pp_sample_time:
            total_pp_sample_time[num_samples] += pp_sample_time[num_samples] / args.num_examples

    out_msg = f"Number of examples: {args.num_examples}\n\n"
    out_msg += f"Number of iterations: {args.iterations}\n\n"
    out_msg += f"Average LLM (no logits) time: {total_llm_base_time / args.iterations}\n"
    out_msg += f"Average PP LLM sampling (compact) time: {total_pp_llm_time / args.iterations}\n"
    out_msg += f"Average PP Compilation time: {total_pp_compile_time / args.iterations}\n"
    for num_samples in total_pp_sample_time:
        out_msg += f"Average PP sampling time for {num_samples} samples: {total_pp_sample_time[num_samples] / args.iterations}\n"
    for num_samples in total_pp_sample_time:
        out_msg += f"Average total PP time for {num_samples} samples: {(total_pp_llm_time + total_pp_compile_time + total_pp_sample_time[num_samples]) / args.iterations}\n"

    with open(f"{report_save_path}/time_{args.num_llm_samples}_{args.suffix}.txt", "w") as f: 
        f.write(out_msg)
