import argparse, json, os, time
import transformers, datasets, torch, tqdm
import ppot.utils, ppot.program, ppot.compile
from cruxeval.evaluation.utils_execute import check_correctness
from cruxeval.prompts import make_direct_input_prompt


def template(tok: transformers.AutoTokenizer, code: str, output: str):
    """Apply chat template to CruxEval input prediction prompt."""
    prompt_text = make_direct_input_prompt((code, output))
    if "Qwen2.5-Coder" in tok.name_or_path:
        E = tok.apply_chat_template(
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
             {"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError(f"Chat template not implemented for {tok.name_or_path}")
    return E, tok([E], return_tensors="pt")


def sample(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer,
           code: str, output: str, num_samples: int, temperature: float = None,
           **kwargs) -> tuple:
    """Generate samples with logits for subset resampling.

    Returns (token_ids, logits, decoded_strings).
    """
    _, X = template(tok, code, output)
    if temperature == 0.0:
        temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else:
        temp_kwargs = {"do_sample": True, "temperature": temperature,
                       "num_return_sequences": num_samples}

    O = model.generate(**X.to(model.device), return_dict_in_generate=True,
                       output_logits=True, repetition_penalty=1.0, top_p=1.0,
                       **temp_kwargs, **kwargs)

    k = X.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)

    # O.logits is a tuple of (num_samples, vocab_size) tensors, one per position
    L = torch.stack([x.cpu() for x in O.logits], dim=1)  # (num_samples, seq_len, vocab_size)

    return I, L, S


def postprocess_generation(text: str) -> str:
    """Extract the answer expression from a CruxEval input prediction generation.

    Adapted from InputPrediction.postprocess_generation() in
    cruxeval/inference/tasks/input_prediction.py.

    Generation text looks like: 'assert f(ARGS) == OUTPUT'
    Returns the 'f(ARGS)' part.
    """
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]")[1].strip()
    if "[/ANSWER]" in text:
        text = text.split("[/ANSWER]")[0].strip()
    if "==" in text:
        text = text.split("==")[0].strip()
    if "assert f" in text:
        text = "f" + text.split("assert f")[1].strip()
    return text.strip()


def cruxeval_input_answer_extractor(decoded_text: str) -> tuple:
    """Locate the character span of the answer expression in the decoded generation.

    For input prediction, the generation looks like:
        'assert f(ARGS) == OUTPUT\\n[/ANSWER]'
    We want the 'f(ARGS)' part.

    Returns (start_char, end_char) or (None, None) if not found.
    """
    if "f(" not in decoded_text:
        return None, None

    start = decoded_text.index("f(")

    if "==" in decoded_text[start:]:
        eq_pos = decoded_text.index("==", start)
        answer_part = decoded_text[start:eq_pos].rstrip()
        end = start + len(answer_part)
    else:
        end = len(decoded_text.rstrip())

    return start, end


def evaluate_single(generation: str, code: str, expected_output: str,
                    timeout: int = 3) -> bool:
    """Evaluate one generation by executing the assertion."""
    if "f(" not in generation:
        return False
    check_program = f"{code}\nassert {expected_output} == {generation}"
    return check_correctness(check_program, timeout=timeout)


def pass_at_k_llm(S: list, code: str, expected_output: str,
                  timeout: int = 3) -> bool:
    """LLM baseline: returns True if any generation passes."""
    for s in S:
        processed = postprocess_generation(s)
        if evaluate_single(processed, code, expected_output, timeout):
            return True
    return False


def pass_at_k_pp(programs: list, num_samples: int, code: str,
                 expected_output: str, pp_temperature: float = 1.0,
                 constraint: bool = False, timeout: int = 3) -> bool:
    """ppot: sample from each SubsetProgram, postprocess, evaluate.

    Returns True if any sample (original or resampled) passes.
    """
    all_samples = []
    for prog in programs:
        samples = prog.sample(num_samples, as_list=True, t=pp_temperature,
                              constraint=constraint)
        all_samples.extend(samples)
        all_samples.append(prog.raw_program)

    for s in all_samples:
        processed = postprocess_generation(s)
        if evaluate_single(processed, code, expected_output, timeout):
            return True
    return False


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
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of subset resamples per LLM sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--num-examples", type=int, default=800)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3)
    parser.add_argument("--report-save-path", type=str, required=True)
    parser.add_argument("--different-constraint", default=False, action="store_true")
    parser.add_argument("--skip-indices", type=str, default="162,342",
                        help="Comma-separated indices to skip (known bad samples)")
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = datasets.load_dataset("cruxeval-org/cruxeval", split="test")
    skip_indices = set(int(x) for x in args.skip_indices.split(",") if x)

    # Main loop
    pass_llm_list, pass_pp_list = [], []
    total_time = 0
    pbar = tqdm.tqdm(enumerate(dataset), total=min(args.num_examples, len(dataset)),
                     desc="CruxEval Input", dynamic_ncols=True)

    for i, example in pbar:
        if i >= args.num_examples:
            break
        if i in skip_indices:
            continue

        code = example["code"]
        expected_output = example["output"]

        # Step 1: Generate samples with logits
        start = time.time()
        I, L, S = sample(model, tokenizer, code, expected_output,
                         args.num_llm_samples, temperature=args.temperature,
                         max_new_tokens=args.max_new_tokens)
        total_time += time.time() - start

        # Step 2: Evaluate LLM pass@k (baseline)
        pass_llm = pass_at_k_llm(S, code, expected_output, args.timeout)
        pass_llm_list.append(pass_llm)

        # Step 3: Compile SubsetPrograms
        P, _ = ppot.compile.subset_programs(
            I, L, tokenizer,
            answer_extractor=cruxeval_input_answer_extractor)

        # Step 4: Evaluate ppot pass@k
        pass_pp = pass_at_k_pp(
            P, args.num_samples, code, expected_output,
            pp_temperature=args.program_temperature,
            constraint=args.different_constraint,
            timeout=args.timeout)
        pass_pp_list.append(pass_pp)

        # Update progress bar
        llm_rate = torch.mean(torch.tensor(pass_llm_list, dtype=torch.float))
        pp_rate = torch.mean(torch.tensor(pass_pp_list, dtype=torch.float))
        pbar.set_postfix({"pass_llm": f"{llm_rate:.3f}",
                          "pass_pp": f"{pp_rate:.3f}"})

    # Report
    llm_rate = torch.mean(torch.tensor(pass_llm_list, dtype=torch.float))
    pp_rate = torch.mean(torch.tensor(pass_pp_list, dtype=torch.float))
    time_per_example = total_time / max(len(pass_llm_list), 1)

    out_msg = (f"pass rate for LLM: {llm_rate}\n"
               f"pass rate for subset resample: {pp_rate}\n"
               f"Number of examples: {len(pass_llm_list)}\n"
               f"Time per example (generation only): {time_per_example:.3f}s\n")

    report_dir = os.path.dirname(args.report_save_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_save_path, "w") as f:
        f.write(out_msg)
    print(out_msg)
