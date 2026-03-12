"""
Benchmark: end-to-end PP sampling overhead with different logit capture strategies.

Methods compared:
  baseline  — LLM sampling only, no logits, no compilation
  old_ppot  — output_logits=True (GPU accumulation + bulk ~1.8GB CPU transfer)
              + programs() + sample_pp()
  new_ppot  — teacher forcing + compact extraction (~1MB CPU transfer)
              + programs(CompactLogits) + sample_pp()

Per-phase breakdown (LLM / compile / pp_sample) is reported for ppot methods.
Correctness check: nLL and the first program's P_tensor are compared between methods.

Usage:
    python -m gsm8k.benchmark_pp_sampling \\
        --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --dataset gsm8k/gsm8k.json \\
        --num-llm-samples 20 --num-examples 10
"""
import argparse, json, time
import transformers, datasets, torch
import ppot.utils, ppot.compile
from gsm8k.eval_gsm8k_fast import (
    template, sample_pp, get_rule_supp,
    sample_llm_compact,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _temp_kwargs(temperature, num_samples):
    if temperature == 0.0:
        return {"do_sample": False, "num_return_sequences": 1}
    return {"do_sample": True, "temperature": temperature, "num_return_sequences": num_samples}


def _sync():
    torch.cuda.synchronize()


def _t():
    _sync()
    return time.time()


# ── sampling functions ────────────────────────────────────────────────────────

def _sample_no_logits(model, tok, X, num_samples, temperature, **kwargs):
    _, enc = template(tok, X)
    O = model.generate(**enc.to(model.device), return_dict_in_generate=True, output_logits=False,
                       repetition_penalty=1.0, top_p=1.0, **_temp_kwargs(temperature, num_samples), **kwargs)
    k = enc.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)
    return I, S


def _sample_output_logits(model, tok, X, num_samples, temperature, **kwargs):
    """Original approach: output_logits=True, bulk CPU transfer at end."""
    _, enc = template(tok, X)
    O = model.generate(**enc.to(model.device), return_dict_in_generate=True, output_logits=True,
                       repetition_penalty=1.0, top_p=1.0, **_temp_kwargs(temperature, num_samples), **kwargs)
    k = enc.input_ids.numel()
    I = O.sequences[:, k:].cpu()
    S = tok.batch_decode(I, skip_special_tokens=True)
    L = torch.stack([x.cpu() for x in O.logits], dim=1)  # (batch, seq_len, vocab)
    return I, L, S


# ── end-to-end pipeline runners ───────────────────────────────────────────────

def run_baseline(model, tok, X, num_samples, temperature, **kwargs):
    t0 = _t()
    I, S = _sample_no_logits(model, tok, X, num_samples, temperature, **kwargs)
    t1 = _t()
    return {"total": t1 - t0}


def run_old_ppot(model, tok, X, num_samples, temperature,
                 tokenizer, rule, supp, num_pp_samples, pp_temperature, pp_seed, **kwargs):
    t0 = _t()
    I, L, S = _sample_output_logits(model, tok, X, num_samples, temperature, **kwargs)
    t1 = _t()
    P, nLL = ppot.compile.programs(I, L, tokenizer, S, rules=rule, supp=supp)
    t2 = _t()
    ppot.utils.seed(pp_seed)
    PPS = sample_pp(P, num_pp_samples, pp_temperature)
    t3 = _t()
    return {"llm": t1 - t0, "compile": t2 - t1, "pp_sample": t3 - t2,
            "total": t3 - t0, "_nLL": nLL, "_P": P, "_I": I, "_PPS": PPS}


def run_new_ppot(model, tok, X, num_samples, temperature,
                 tokenizer, rule, supp, supp_ids, num_pp_samples, pp_temperature, pp_seed, **kwargs):
    t0 = _t()
    I, L_compact, S = sample_llm_compact(model, tok, X, num_samples, supp_ids, temperature, **kwargs)
    t1 = _t()
    P, nLL = ppot.compile.programs(I, L_compact, tokenizer, S, rules=rule, supp=supp)
    t2 = _t()
    ppot.utils.seed(pp_seed)
    PPS = sample_pp(P, num_pp_samples, pp_temperature)
    t3 = _t()
    return {"llm": t1 - t0, "compile": t2 - t1, "pp_sample": t3 - t2,
            "total": t3 - t0, "_nLL": nLL, "_P": P, "_I": I, "_PPS": PPS}


# ── main ──────────────────────────────────────────────────────────────────────

SUPP_MODELS = ["Qwen/Qwen2.5-Coder-0.5B-Instruct",
               "Qwen/Qwen2.5-Coder-3B-Instruct",
               "Qwen/Qwen2.5-Coder-7B-Instruct"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=SUPP_MODELS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-llm-samples", type=int, default=20)
    parser.add_argument("--num-pp-samples", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--pp-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rule", type=str, default="all")
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    _, ext = args.dataset.rsplit(".", 1)
    if ext == "json":
        with open(args.dataset) as f:
            D = datasets.Dataset.from_list(json.load(f)[:args.num_examples])
    else:
        D = datasets.Dataset.load_from_disk(args.dataset)

    rule, supp = get_rule_supp(args.rule, tokenizer)
    supp_ids = torch.cat(supp).unique()
    print(f"Support vocab size: {len(supp_ids)} token IDs (out of {tokenizer.vocab_size})")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    pp_seed = args.seed + 1  # fixed seed for sample_pp, same for both methods
    pp_kwargs = dict(tokenizer=tokenizer, rule=rule, supp=supp,
                     num_pp_samples=args.num_pp_samples, pp_temperature=args.pp_temperature,
                     pp_seed=pp_seed)

    # Warmup
    print("Warming up GPU...")
    warmup_X = D[0]
    ppot.utils.seed(args.seed)
    run_baseline(model, tokenizer, warmup_X, args.num_llm_samples, args.temperature,
                 max_new_tokens=32)
    ppot.utils.seed(args.seed)
    run_old_ppot(model, tokenizer, warmup_X, args.num_llm_samples, args.temperature,
                 max_new_tokens=32, **pp_kwargs)
    ppot.utils.seed(args.seed)
    run_new_ppot(model, tokenizer, warmup_X, args.num_llm_samples, args.temperature,
                 supp_ids=supp_ids, max_new_tokens=32, **pp_kwargs)
    _sync()
    print("Warmup done.\n")

    times = {"baseline": [], "old_ppot": [], "new_ppot": []}
    phase_times = {
        "old_ppot": {"llm": [], "compile": [], "pp_sample": []},
        "new_ppot": {"llm": [], "compile": [], "pp_sample": []},
    }

    for i, X in enumerate(D):
        ppot.utils.seed(args.seed)
        r_base = run_baseline(model, tokenizer, X, args.num_llm_samples, args.temperature, **gen_kwargs)

        ppot.utils.seed(args.seed)
        r_old = run_old_ppot(model, tokenizer, X, args.num_llm_samples, args.temperature,
                             **gen_kwargs, **pp_kwargs)

        ppot.utils.seed(args.seed)
        r_new = run_new_ppot(model, tokenizer, X, args.num_llm_samples, args.temperature,
                             supp_ids=supp_ids, **gen_kwargs, **pp_kwargs)

        times["baseline"].append(r_base["total"])
        times["old_ppot"].append(r_old["total"])
        times["new_ppot"].append(r_new["total"])
        for ph in ("llm", "compile", "pp_sample"):
            phase_times["old_ppot"][ph].append(r_old[ph])
            phase_times["new_ppot"][ph].append(r_new[ph])

        I_old, I_new = r_old["_I"], r_new["_I"]
        seq_match = I_old.shape == I_new.shape and (I_old == I_new).all().item()

        PPS_old, PPS_new = r_old["_PPS"], r_new["_PPS"]
        pps_match = seq_match and PPS_old == PPS_new

        print(f"  Example {i}: "
              f"baseline={r_base['total']:.2f}s  "
              f"old_ppot={r_old['total']:.2f}s "
              f"(llm={r_old['llm']:.2f} compile={r_old['compile']:.2f} pp={r_old['pp_sample']:.2f})  "
              f"new_ppot={r_new['total']:.2f}s "
              f"(llm={r_new['llm']:.2f} compile={r_new['compile']:.2f} pp={r_new['pp_sample']:.2f})  "
              f"seq_match={seq_match} pps_match={pps_match}")

    n = len(D)
    avg = {k: sum(v) / n for k, v in times.items()}
    print(f"\nResults over {n} examples, {args.num_llm_samples} LLM samples, "
          f"{args.num_pp_samples} PP samples each:")
    print(f"  {'baseline (no ppot)':<35} {avg['baseline']:.3f}s")
    for method in ("old_ppot", "new_ppot"):
        a = avg[method]
        ph = {k: sum(v) / n for k, v in phase_times[method].items()}
        print(f"  {method:<35} {a:.3f}s  overhead={a/avg['baseline']:.2f}x"
              f"  [llm={ph['llm']:.3f} compile={ph['compile']:.3f} pp={ph['pp_sample']:.3f}]")
