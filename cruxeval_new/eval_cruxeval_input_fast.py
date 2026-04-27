import argparse, json, os, time
import transformers, datasets, torch, tqdm # type: ignore
import ppot.utils, ppot.program, ppot.compile
from transformers import LogitsProcessorList, LogitsProcessor
from cruxeval_new.utils_execute import check_correctness
from cruxeval_new.prompts import make_direct_input_prompt

def template(tok: transformers.AutoTokenizer, code: str, output: str):
    """Apply chat template to CruxEval input prediction prompt."""
    prompt_text = make_direct_input_prompt((code, output))
    return prompt_text, tok([prompt_text], return_tensors="pt") # type: ignore

class _SubsetLogitsCapture(LogitsProcessor):
    """Captures logits during generation with non_blocking CPU transfers.

    After generation, compacts to only the unique generated token columns.
    Also captures token_log_prob via the one-step-behind pattern (for nLL).
    """
    def __init__(self):
        self._scores_cpu = []
        self._tlp_cpu = []
        self._prev_lp = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        lp = torch.log_softmax(scores, dim=-1)
        if self._prev_lp is not None:
            prev_token = input_ids[:, -1]
            self._tlp_cpu.append(
                self._prev_lp.gather(-1, prev_token.unsqueeze(-1)).squeeze(-1)
                .to("cpu", non_blocking=True))
        self._scores_cpu.append(scores.to("cpu", non_blocking=True))
        self._prev_lp = lp
        return scores

    def finalize(self, last_gen_ids: torch.LongTensor, generated_ids: torch.LongTensor):
        """Flush final token_log_prob, synchronize, compact to unique tokens.

        Args:
            last_gen_ids: (batch,) final token IDs on GPU
            generated_ids: (batch, seq_len) all generated token IDs on CPU
        Returns:
            compact_scores: (batch, seq_len, n_unique)
            token_log_prob: (batch, seq_len)
            unique_toks: (n_unique,)
        """
        if self._prev_lp is not None:
            self._tlp_cpu.append(
                self._prev_lp.gather(-1, last_gen_ids.unsqueeze(-1)).squeeze(-1)
                .to("cpu", non_blocking=True))
        torch.cuda.synchronize()

        all_scores = torch.stack(self._scores_cpu, dim=1)
        token_log_prob = torch.stack(self._tlp_cpu, dim=1)

        unique_toks = torch.unique(generated_ids)
        compact_scores = all_scores[:, :, unique_toks]

        return compact_scores, token_log_prob, unique_toks


def sample_compact(model: transformers.AutoModelForCausalLM, tok: transformers.AutoTokenizer,
                   code: str, output: str, num_samples: int, temperature: float,
                   **kwargs) -> tuple:
    """Generate samples with compact logits via LogitsProcessor.

    Returns (token_ids, compact_scores, token_log_prob, unique_toks, decoded_strings).
    """
    X_str, X = template(tok, code, output)
    temp_kwargs: dict
    if temperature == 0.0:
        temp_kwargs = {"do_sample": False, "num_return_sequences": 1}
    else:
        temp_kwargs = {"do_sample": True, "temperature": temperature,
                       "num_return_sequences": num_samples}
    temp_kwargs.update({"repetition_penalty": 1.0, "top_p": 0.95,
                        "stop_strings": ['[/ANSWER]'],
                        "max_new_tokens": kwargs.pop("max_new_tokens", 769),
                        "tokenizer": tok})

    cap = _SubsetLogitsCapture()
    device = next(model.parameters()).device # type: ignore
    O = model.generate(**X.to(device), return_dict_in_generate=True, # type: ignore
                       output_logits=False,
                       logits_processor=LogitsProcessorList([cap]),
                       **temp_kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True) # type: ignore

    compact_scores, token_log_prob, unique_toks = cap.finalize(
        O.sequences[:, -1], I)

    return I, compact_scores, token_log_prob, unique_toks, S


def postprocess_generation(text: str) -> str:
    """Extract the answer expression from a CruxEval input prediction generation.

    Adapted from InputPrediction.postprocess_generation() in
    cruxeval/inference/tasks/input_prediction.py.

    Generation text looks like: 'assert f(ARGS) == OUTPUT'
    Returns the 'f(ARGS)' part.
    """
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
    answer_part = postprocess_generation(decoded_text)
    start = decoded_text.index(answer_part)
    end = start + len(answer_part)
    return start, end


def evaluate_single(generation: str, code: str, expected_output: str,
                    timeout: int = 3) -> bool:
    """Evaluate one generation by executing the assertion."""
    if "f(" not in generation:
        return False
    check_program = f"{code}\nassert {expected_output} == {generation}"
    return check_correctness(check_program, timeout=timeout)


def pass_at_k(S: list, code: str, expected_output: str,
                  timeout: int = 3) -> bool:
    for s in S:
        processed = postprocess_generation(s)
        if evaluate_single(processed, code, expected_output, timeout):
            return True, s
    return False, ""

def sample_pp(programs: list, num_samples: int, pp_temperature: float = 1.0,
                 constraint: bool = False) -> list:
    all_samples = []
    for prog in programs:
        samples = prog.sample(num_samples, as_list=True, t=pp_temperature,
                              constraint=constraint)
        all_samples.extend(samples)
        all_samples.append(prog.raw_program)
    return all_samples

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
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    dataset = datasets.load_dataset("cruxeval-org/cruxeval", split="test")

    # Main loop
    pass_llm_list, pass_pp_list = [], []
    pbar = tqdm.tqdm(enumerate(dataset), total=min(args.num_examples, len(dataset)),
                     desc="CruxEval Input", dynamic_ncols=True)

    entropy_save_path = f"/space/poorvagarg/genPPS/cruxeval_new/entropy/{args.model}/{args.temperature}/"
    report_save_path = f"/space/poorvagarg/genPPS/cruxeval_new/report/{args.model}/{args.temperature}/"
    llm_cache_path = f"/space/poorvagarg/genPPS/cruxeval_new/generations/{args.model}/{args.temperature}/"
    os.makedirs(entropy_save_path, exist_ok=True)
    os.makedirs(report_save_path, exist_ok=True)
    os.makedirs(llm_cache_path, exist_ok=True)

    for i, example in pbar:
        if i >= args.num_examples:
            break

        code = example["code"]
        expected_output = example["output"]

        # Step 1: Generate samples with logits
        I, compact_scores, token_log_prob, unique_toks, S = sample_compact(
            model, tokenizer, code, expected_output,
            args.num_llm_samples, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens)

        # Step 2: Evaluate LLM pass@k (baseline)
        pass_llm, llm_sample = pass_at_k(S, code, expected_output, args.timeout)
        pass_llm_list.append(pass_llm)

        # Step 3: Compile SubsetPrograms
        P, _ = ppot.compile.fast_subset_programs(
            I, compact_scores, token_log_prob, unique_toks, tokenizer,
            answer_extractor=cruxeval_input_answer_extractor)

        # # Step 4: Evaluate ppot pass@k
        PP_samples = sample_pp(P, args.num_samples, args.program_temperature, args.different_constraint)
        pass_pp, pp_sample = pass_at_k(PP_samples, code, expected_output, args.timeout)
        pass_pp_list.append(pass_pp)

        if pass_pp and not pass_llm:
            with open(f"/space/poorvagarg/genPPS/cruxeval_new/examples/{args.model}_{args.temperature}_{i}.txt", "w") as f:
                f.write(f"Code: \n{code}\n")
                f.write(f"Expected output: {expected_output}\n\n")
                f.write("LLM generations:\n")
                assert len(S) == 1
                f.write(S[0] + "\n\n")
                f.write("Probabilistic program generations:\n")
                f.write(pp_sample + "\n\n")

        # Update progress bar
        llm_rate = torch.mean(torch.tensor(pass_llm_list, dtype=torch.float))
        pp_rate = torch.mean(torch.tensor(pass_pp_list, dtype=torch.float))
        pbar.set_postfix({"pass_llm": f"{llm_rate:.3f}",
                          "pass_pp": f"{pp_rate:.3f}"})

    # Report
    llm_rate = torch.mean(torch.tensor(pass_llm_list, dtype=torch.float))
    pp_rate = torch.mean(torch.tensor(pass_pp_list, dtype=torch.float))

    out_msg = (f"Number of examples: {len(pass_llm_list)}\n"
               f"pass rate for LLM: {llm_rate}\n"
               f"pass rate for subset resample: {pp_rate}\n")
    with open(f"{report_save_path}/report_fast.txt", "w") as f: f.write(out_msg)
