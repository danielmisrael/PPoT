import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse, time
import transformers, datasets, torch, tqdm
import ppot.utils, ppot.program, ppot.compile
from transformers import LogitsProcessorList, LogitsProcessor
from cruxeval.evaluation.utils_execute import check_correctness
from cruxeval.eval_cruxeval_input import (
    template, postprocess_generation, cruxeval_input_answer_extractor,
    evaluate_single, pass_at_k_llm,
)


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
                   code: str, output: str, num_samples: int, temperature: float = None,
                   **kwargs) -> tuple:
    """Generate samples with compact logits via LogitsProcessor.

    Returns (token_ids, compact_scores, token_log_prob, unique_toks, decoded_strings).
    """
    X_str, X = template(tok, code, output)
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
    O = model.generate(**X.to(model.device), return_dict_in_generate=True,
                       output_logits=False,
                       logits_processor=LogitsProcessorList([cap]),
                       **temp_kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)

    compact_scores, token_log_prob, unique_toks = cap.finalize(
        O.sequences[:, -1], I)

    return I, compact_scores, token_log_prob, unique_toks, S


def sample_pp(programs: list, num_samples: int, pp_temperature: float = 1.0,
              constraint: bool = False) -> list:
    """Sample from SubsetProgram/FastSubsetProgram objects."""
    all_samples = []
    for prog in programs:
        samples = prog.sample(num_samples, as_list=True, t=pp_temperature,
                              constraint=constraint)
        all_samples.extend(samples)
        all_samples.append(prog.raw_program)
    return all_samples


def pass_at_k_pp(programs: list, num_samples: int, code: str,
                 expected_output: str, pp_temperature: float = 1.0,
                 constraint: bool = False, timeout: int = 3) -> bool:
    all_samples = sample_pp(programs, num_samples, pp_temperature, constraint)
    for s in all_samples:
        processed = postprocess_generation(s)
        if evaluate_single(processed, code, expected_output, timeout):
            return all_samples, True
    return all_samples, False


SUPP_MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast CruxEval input prediction evaluation with compact subset resampling")
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
    parser.add_argument("--report-save-path", type=str, required=True)
    parser.add_argument("--different-constraint", default=False, action="store_true")
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
    total_time = 0
    pbar = tqdm.tqdm(enumerate(dataset), total=min(args.num_examples, len(dataset)),
                     desc="CruxEval Input", dynamic_ncols=True)

    for i, example in pbar:
        if i >= args.num_examples:
            break

        code = example["code"]
        expected_output = example["output"]

        # Step 1: Generate samples with compact logits
        start = time.time()
        I, compact_scores, token_log_prob, unique_toks, S = sample_compact(
            model, tokenizer, code, expected_output,
            args.num_llm_samples, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens)
        total_time += time.time() - start

        S = [postprocess_generation(s) for s in S]

        # Step 2: Evaluate LLM pass@k (baseline)
        pass_llm = pass_at_k_llm(S, code, expected_output, args.timeout)
        pass_llm_list.append(pass_llm)

        # Step 3: Compile FastSubsetPrograms
        P, _ = ppot.compile.fast_subset_programs(
            I, compact_scores, token_log_prob, unique_toks, tokenizer,
            answer_extractor=cruxeval_input_answer_extractor)

        # Step 4: Evaluate ppot pass@k
        PP_samples, pass_pp = pass_at_k_pp(
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
