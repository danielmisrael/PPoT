import argparse, json, os, time
import transformers, datasets, torch, tqdm # type: ignore
import ppot.utils, ppot.program, ppot.compile
from cruxeval_new.utils_execute import check_correctness
from cruxeval_new.prompts import make_direct_input_prompt
from cruxeval_new.eval_cruxeval_input_fast import template, sample_compact, pass_at_k, cruxeval_input_answer_extractor, postprocess_generation, evaluate_single

def sample_pp(programs: list, num_samples: int, pp_temperature: float = 1.0,
                 constraint: bool = False) -> list:
    all_samples = []
    count = 0
    for prog in programs:
        # print(count)
        count += 1
        samples = [prog.raw_program]
        samples.extend(prog.sample(num_samples, as_list=True, t=pp_temperature,
                              constraint=constraint))
        all_samples.append(samples)
    return all_samples



def pass_at_k_answers(S: list, code: str, expected_output: str, timeout: int = 3) -> list:
    ans_list = []
    count = 0
    for i in S:
        ans_i = []
        for j in i:
            count += 1
            # print(count)
            current_answer = False
            try:
                processed = postprocess_generation(j)
                current_answer = evaluate_single(processed, code, expected_output, timeout)
            except: pass
            ans_i.append(current_answer)
        ans_list.append(ans_i)
    return ans_list

SUPP_MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

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
    parser.add_argument("--parent-dir", type=str, default="./cruxeval_new")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    # ppot.utils.seed(args.seed)

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = datasets.load_dataset("cruxeval-org/cruxeval", split="test")

    # Main loop
    pass_llm_list: list = []
    pass_pp_list: list = []
    pbar = tqdm.tqdm(enumerate(dataset), total=min(args.num_examples, len(dataset)),
                     desc="CruxEval Input", dynamic_ncols=True)

    entropy_save_path = f"{args.parent_dir}/entropy/{args.model}/{args.temperature}/"
    report_save_path = f"{args.parent_dir}/report/{args.model}/{args.temperature}/"
    llm_cache_path = f"{args.parent_dir}/generations/{args.model}/{args.temperature}/"
    os.makedirs(entropy_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    pass_rates = [[0 for _ in range(args.num_samples+1)] for _ in range(args.num_llm_samples)]

    for i, example in pbar:
        if i >= args.num_examples:
            break

        code = example["code"]
        expected_output = example["output"]

        # Step 1: Generate samples with logits

        # breakpoint()
        
        I, compact_scores, token_log_prob, unique_toks, S = sample_compact(
            model, tokenizer, code, expected_output,
            args.num_llm_samples, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens)

        # Step 2: Evaluate LLM pass@k (baseline)
        pass_llm = pass_at_k(S, code, expected_output, args.timeout)
        pass_llm_list.append(pass_llm)
        # print("Language model sampled")
        # Step 3: Compile SubsetPrograms
        P, _ = ppot.compile.fast_subset_programs(
            I, compact_scores, token_log_prob, unique_toks, tokenizer,
            answer_extractor=cruxeval_input_answer_extractor)
        # print("Probabilistic programs compiled")
        # # Step 4: Evaluate ppot pass@k
        PP_samples = sample_pp(P, args.num_samples, args.program_temperature, args.different_constraint)
        # print("Probabilistic programs sampled and executed")
        pp_answers = pass_at_k_answers(PP_samples, code, expected_output, args.timeout)
        for j in range(1, args.num_llm_samples+1):
            for k in range(0, args.num_samples+1):
                answer = False
                for l in pp_answers[:j]:
                    if any(l[:k]):
                        answer = True
                        break
                pass_rates[j-1][k] += int(answer)
        # print("Pass rates calculated")
    # Report
    with open(f"{report_save_path}/accuracy_{args.suffix}.csv", "a") as f:
        f.write(f"Number of exaples: {args.num_examples}\n")
        f.write("LLM Samples, PP Samples, Pass Rate\n")
        for j in range(1, args.num_llm_samples+1):
            for k in range(0, args.num_samples+1):
                f.write(f"{j}, {k}, {pass_rates[j-1][k]/args.num_examples}\n")
