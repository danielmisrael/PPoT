import argparse, multiprocessing, random, os
import transformers, pickle, datasets, numpy as np, tqdm, prettytable
import ppot.compile, ppot.utils
from scripts.text_match_score import evaluate_single_example

def extract_probabilistic_programs(temperature:float, num_eg:int) -> tuple:
    # Loading the processor
    processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Load the programs generated from the model: in vedett [0]
    # t__ is the temperature - 8 samples
    # t0 only one sample, because always greedy
    # there are multiple .pkl files
    cached_path = f"cache/pp_t{temperature:.1f}.pkl"
    if os.path.isfile(cached_path):
        PP, LL = ppot.utils.retrieve_programs(cached_path, None)
        if len(PP) >= num_eg:
            return PP[:num_eg], LL[:num_eg]

    PP, LL = [], []
    for i in tqdm.tqdm(range(num_eg), desc="Compiling programs"):
        with open(f"/space/renatolg/genPPS/out/Qwen2.5-VL-3B-Instruct_t{temperature:.1f}/data/{i}.pkl", "rb") as f:
            R = pickle.load(f)
        # Load the token ids and logits
        input_ids, logits, code = R["ids"], R["logits"], R["code"]
        # Generate the probabilistic program
        tempPP, tempLL = ppot.compile.programs(input_ids, logits, processor, code=code)
        PP.append(tempPP)
        LL.append(tempLL)

    os.makedirs("cache", exist_ok=True)
    with open(cached_path, "wb") as f: pickle.dump((PP, LL), f)

    return PP, LL


def _sample_task(p: ppot.program.Program, greedy: bool, num_samples: int, t: float) -> list:
    return p.sample(num_samples, as_list=True, t=t)

def _get_greedy(p:ppot.program.Program) -> list:
    return p.greedy(as_list=True)

def _get_raw(p:ppot.program.Program) -> list:
    return [p.raw_program]

def evaluate_programs(to_run: list, dataset: dict) -> list:
    """
    Evaluates multiple programs in parallel
    """
    with multiprocessing.Pool() as pool:
        procs = [[pool.apply_async(evaluate_single_example, (p, g)) for p, g in zip(to_run[i], dataset["code"])]
                 for i in range(len(PP))]
        text_match_scores = []
        for P in tqdm.tqdm(procs, desc="Evaluating"):
            U = []
            for p in P:
                try: r = p.get(30) # 30 seconds timeout
                except Exception as exc:
                    r = 0
                    print(">>>>>>>>>", exc)
                U.append(r)
            text_match_scores.append(U)
    return text_match_scores

def example_programs(raw_programs: list, raw_scores: list, sample_programs: list, sample_scores: list,
                     dataset: dict) -> list:
    "Return example programs where sampled probabilistic program does better than raw program"
    max_sample_scores = np.max(np.array(sample_scores), axis=1)
    argmax_sample_scores = np.argmax(np.array(sample_scores), axis=1)
    triples = []
    for i, (RP, RS, MS, AS) in enumerate(zip(raw_programs, raw_scores, max_sample_scores, argmax_sample_scores)):
        if MS > 0:
            triples.append([sample_programs[i][AS], RP, dataset['code']])

    return triples


def sample_from_probabilistic_programs(PP: list, LL: list, num_samples: int, greedy: bool, raw: bool,
                                       pp_temp: float, dump_example: bool) -> list:
    """
    Sample from probabilistic programs and evaluate them with respect to the actual image.
    It returns statistics of the results
    """
    dataset = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples=len(PP),
                                      filter_fn = lambda x: "matplotlib" in x["url"], split="test")
    to_run_sample = [_sample_task(PP[i][0], greedy, num_samples, pp_temp) for i in tqdm.tqdm(range(len(PP)), "Generating")]
    text_match_scores = evaluate_programs(to_run_sample, dataset)

    stats = {}
    # Computing statistics
    np_scores = np.array(text_match_scores)
    stats["Score Minimum"] = np.mean(np.min(np_scores, axis=1))
    stats["Score Maximum"] = np.mean(np.max(np_scores, axis=1))
    stats["Score Mean"] = np.mean(np_scores)
    stats["Score Median"] = np.mean(np.median(np_scores, axis=1))

    if greedy:
        to_run_greedy = [_get_greedy(PP[i][0]) for i in tqdm.tqdm(range(len(PP)), "Greedy Probabilistic Programs")]
        greedy_scores = evaluate_programs(to_run_greedy, dataset)
        stats["Greedy Scores"] = np.mean(np.array(greedy_scores))

    if raw:
        to_run_raw = [_get_raw(PP[i][0]) for i in tqdm.tqdm(range(len(PP)), "Raw samples")]
        raw_scores = evaluate_programs(to_run_raw, dataset)
        stats["Raw Scores"] = np.mean(np.array(raw_scores))

    triples = example_programs(to_run_raw, raw_scores, to_run_sample, text_match_scores, dataset) if dump_example else []

    return stats, triples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--isUSPP", action="store_true", default=False)
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--pp-temperature", type=float, default=1.0)
    parser.add_argument("--raw", action="store_true", default=False)
    parser.add_argument("--dump", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    PP, LL = extract_probabilistic_programs(args.temperature, args.num_examples)
    stats, triples = sample_from_probabilistic_programs(PP, LL, args.num_samples,
                                                                args.greedy, args.raw,
                                                                args.pp_temperature, args.dump)

    table = prettytable.PrettyTable()
    table.title = "Text Match Score Statistics"
    table.field_names = list(stats.keys())
    table.add_row(list(stats.values()))
    print(table)

    os.makedirs(f"/space/poorvagarg/genPPS/out/t{args.pp_temperature:.1f}", exist_ok=True)

    with open(f"/space/poorvagarg/genPPS/out/t{args.pp_temperature:.1f}/stats_t{args.temperature:.1f}_n{args.num_examples}_s{args.num_samples}"
              + args.isUSPP*"_isUSPP" + args.greedy*"_greedy" + ".pkl", "wb") as f:
        pickle.dump((stats), f)

    if args.dump:
        with open(f"/space/poorvagarg/genPPS/out/t{args.pp_temperature:.1f}/stats_t{args.temperature:.1f}_n{args.num_examples}_s{args.num_samples}"
                + args.isUSPP*"_isUSPP" + args.greedy*"_greedy" + "_examples" + ".pkl", "wb") as f:
            pickle.dump(triples, f)
