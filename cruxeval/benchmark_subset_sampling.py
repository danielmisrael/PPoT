"""
Benchmark: subset resampling overhead for CruxEval input prediction.

Methods compared:
  baseline    — LLM sampling only, no logits, no resampling
  old_subset  — output_logits=True (bulk GPU accumulation + ~1.8GB CPU transfer)
                + subset_programs() + SubsetProgram.sample()
  new_subset  — _SubsetLogitsCapture LogitsProcessor (non_blocking CPU transfers)
                + fast_subset_programs() + FastSubsetProgram.sample()

Correctness check: with the same seed, old_subset and new_subset must produce
identical generated sequences and identical resampled strings.

Usage:
    python -m cruxeval.benchmark_subset_sampling \\
        --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --num-llm-samples 5 --num-pp-samples 5 --num-examples 10
"""
import argparse, time
import transformers, datasets, torch
import ppot.utils, ppot.compile
from cruxeval.eval_cruxeval_input import (
    template, sample, cruxeval_input_answer_extractor, postprocess_generation,
)
from cruxeval.eval_cruxeval_input_fast import sample_compact, sample_pp


# ── helpers ───────────────────────────────────────────────────────────────────

def _sync():
    torch.cuda.synchronize()


def _t():
    _sync()
    return time.time()


def _sample_no_logits(model, tok, code, output, num_samples, temperature, **kwargs):
    """Baseline: generate without logits."""
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
    O = model.generate(**X.to(model.device), return_dict_in_generate=True,
                       output_logits=False, **temp_kwargs)
    k = X.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)
    return I, S


# ── end-to-end pipeline runners ──────────────────────────────────────────────

def run_baseline(model, tok, code, output, num_samples, temperature, **kwargs):
    t0 = _t()
    I, S = _sample_no_logits(model, tok, code, output, num_samples, temperature, **kwargs)
    t1 = _t()
    return {"total": t1 - t0}


def run_old_subset(model, tok, code, output, num_samples, temperature,
                   tokenizer, num_pp_samples, pp_temperature, pp_seed, **kwargs):
    t0 = _t()
    I, L, S = sample(model, tok, code, output, num_samples,
                     temperature=temperature, **kwargs)
    t1 = _t()
    P, nLL = ppot.compile.subset_programs(
        I, L, tokenizer, answer_extractor=cruxeval_input_answer_extractor)
    t2 = _t()
    ppot.utils.seed(pp_seed)
    PPS = sample_pp(P, num_pp_samples, pp_temperature)
    t3 = _t()
    return {"llm": t1 - t0, "compile": t2 - t1, "pp_sample": t3 - t2,
            "total": t3 - t0, "_nLL": nLL, "_I": I, "_L": L, "_PPS": PPS}


def run_new_subset(model, tok, code, output, num_samples, temperature,
                   tokenizer, num_pp_samples, pp_temperature, pp_seed, **kwargs):
    t0 = _t()
    I, compact_scores, token_log_prob, unique_toks, S = sample_compact(
        model, tok, code, output, num_samples, temperature=temperature, **kwargs)
    t1 = _t()
    P, nLL = ppot.compile.fast_subset_programs(
        I, compact_scores, token_log_prob, unique_toks, tokenizer,
        answer_extractor=cruxeval_input_answer_extractor)
    t2 = _t()
    ppot.utils.seed(pp_seed)
    PPS = sample_pp(P, num_pp_samples, pp_temperature)
    t3 = _t()

    # Correctness check: build old-style SubsetPrograms from full logits
    # reconstructed by scattering compact scores back to full vocab.
    # This is NOT timed — only used to verify sample_match.
    full_vocab_size = len(tokenizer)
    full_logits = torch.zeros(compact_scores.shape[0], compact_scores.shape[1],
                              full_vocab_size)
    full_logits[:, :, unique_toks] = compact_scores
    P_compat, _ = ppot.compile.subset_programs(
        I, full_logits, tokenizer,
        answer_extractor=cruxeval_input_answer_extractor)
    ppot.utils.seed(pp_seed)
    PPS_compat = sample_pp(P_compat, num_pp_samples, pp_temperature)

    return {"llm": t1 - t0, "compile": t2 - t1, "pp_sample": t3 - t2,
            "total": t3 - t0, "_nLL": nLL, "_I": I, "_PPS": PPS,
            "_PPS_compat": PPS_compat,
            "_compact_scores": compact_scores, "_unique_toks": unique_toks}


# ── main ─────────────────────────────────────────────────────────────────────

SUPP_MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct",
               "Qwen/Qwen2.5-Coder-3B-Instruct",
               "Qwen/Qwen2.5-Coder-7B-Instruct"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=SUPP_MODELS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-llm-samples", type=int, default=5)
    parser.add_argument("--num-pp-samples", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--pp-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    dataset = datasets.load_dataset("cruxeval-org/cruxeval", split="test")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    pp_seed = args.seed + 1
    pp_kwargs = dict(tokenizer=tokenizer, num_pp_samples=args.num_pp_samples,
                     pp_temperature=args.pp_temperature, pp_seed=pp_seed)

    # Warmup
    print("Warming up GPU...")
    ex0 = dataset[0]
    ppot.utils.seed(args.seed)
    run_baseline(model, tokenizer, ex0["code"], ex0["output"],
                 args.num_llm_samples, args.temperature, max_new_tokens=32)
    ppot.utils.seed(args.seed)
    run_old_subset(model, tokenizer, ex0["code"], ex0["output"],
                   args.num_llm_samples, args.temperature, max_new_tokens=32, **pp_kwargs)
    ppot.utils.seed(args.seed)
    run_new_subset(model, tokenizer, ex0["code"], ex0["output"],
                   args.num_llm_samples, args.temperature, max_new_tokens=32, **pp_kwargs)
    _sync()
    print("Warmup done.\n")

    times = {"baseline": [], "old_subset": [], "new_subset": []}
    phase_times = {
        "old_subset": {"llm": [], "compile": [], "pp_sample": []},
        "new_subset": {"llm": [], "compile": [], "pp_sample": []},
    }

    for i in range(min(args.num_examples, len(dataset))):
        ex = dataset[i]
        code, output = ex["code"], ex["output"]

        ppot.utils.seed(args.seed)
        r_base = run_baseline(model, tokenizer, code, output,
                              args.num_llm_samples, args.temperature, **gen_kwargs)

        ppot.utils.seed(args.seed)
        r_old = run_old_subset(model, tokenizer, code, output,
                               args.num_llm_samples, args.temperature,
                               **gen_kwargs, **pp_kwargs)

        ppot.utils.seed(args.seed)
        r_new = run_new_subset(model, tokenizer, code, output,
                               args.num_llm_samples, args.temperature,
                               **gen_kwargs, **pp_kwargs)

        times["baseline"].append(r_base["total"])
        times["old_subset"].append(r_old["total"])
        times["new_subset"].append(r_new["total"])
        for ph in ("llm", "compile", "pp_sample"):
            phase_times["old_subset"][ph].append(r_old[ph])
            phase_times["new_subset"][ph].append(r_new[ph])

        I_old, I_new = r_old["_I"], r_new["_I"]
        seq_match = I_old.shape == I_new.shape and (I_old == I_new).all().item()

        # Compare logits at unique token positions
        L_old = r_old["_L"]
        compact_new = r_new["_compact_scores"]
        unique_toks = r_new["_unique_toks"]
        L_old_at_unique = L_old[:, :compact_new.shape[1], unique_toks]
        logits_match = torch.allclose(L_old_at_unique, compact_new, atol=1e-4)
        if not logits_match:
            max_diff = (L_old_at_unique - compact_new).abs().max().item()
        else:
            max_diff = 0.0

        PPS_old, PPS_compat = r_old["_PPS"], r_new["_PPS_compat"]
        sample_match = seq_match and PPS_old == PPS_compat

        print(f"  Example {i}: "
              f"baseline={r_base['total']:.2f}s  "
              f"old_subset={r_old['total']:.2f}s "
              f"(llm={r_old['llm']:.2f} compile={r_old['compile']:.2f} pp={r_old['pp_sample']:.2f})  "
              f"new_subset={r_new['total']:.2f}s "
              f"(llm={r_new['llm']:.2f} compile={r_new['compile']:.2f} pp={r_new['pp_sample']:.2f})  "
              f"seq_match={seq_match} logits_match={logits_match} (max_diff={max_diff:.2e}) "
              f"sample_match={sample_match}")

    n = min(args.num_examples, len(dataset))
    avg = {k: sum(v) / n for k, v in times.items()}
    print(f"\nResults over {n} examples, {args.num_llm_samples} LLM samples, "
          f"{args.num_pp_samples} PP samples each:")
    print(f"  {'baseline (no resampling)':<35} {avg['baseline']:.3f}s")
    for method in ("old_subset", "new_subset"):
        a = avg[method]
        ph = {k: sum(v) / n for k, v in phase_times[method].items()}
        print(f"  {method:<35} {a:.3f}s  overhead={a/avg['baseline']:.2f}x"
              f"  [llm={ph['llm']:.3f} compile={ph['compile']:.3f} pp={ph['pp_sample']:.3f}]")
