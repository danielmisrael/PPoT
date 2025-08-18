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
        return PP[:num_eg], LL[:num_eg]

    PP, LL = [], []
    for i in tqdm.tqdm(range(num_eg), desc="Compiling programs"):
        with open(f"/space/renatolg/genPPS/out/Qwen2.5-VL-3B-Instruct_t{temperature:.1f}/data/{i}.pkl", "rb") as f:
            R = pickle.load(f)
        # Load the token ids and logits
        input_ids, logits = R["ids"], R["logits"]
        # Generate the probabilistic program
        tempPP, tempLL = ppot.compile.programs(input_ids, logits, processor)
        PP.append(tempPP)
        LL.append(tempLL)

    os.makedirs("cache", exist_ok=True)
    with open(cached_path, "wb") as f: pickle.dump((PP, LL), f)

    return PP, LL

def _sample_task(p: ppot.program.Program, greedy: bool, num_samples: int, t: float) -> list:
    return p.greedy(as_list=True) if greedy else p.sample(num_samples, as_list=True, t=t)
def sample_from_probabilistic_programs(PP: list, LL: list, num_samples: int, greedy: bool,
                                       pp_temp: float) -> list:
    """
    Sample from probabilistic programs and evaluate them with respect to the actual image.
    It returns statistics of the results
    """
    dataset = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples=len(PP),
                                      filter_fn = lambda x: "matplotlib" in x["url"], split="test")
    to_run = [_sample_task(PP[i][0], greedy, num_samples, pp_temp) for i in tqdm.tqdm(range(len(PP)), "Generating")]
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

    # Computing statistics
    np_scores = np.array(text_match_scores)
    score_min = np.mean(np.min(np_scores, axis=1))
    score_max = np.mean(np.max(np_scores, axis=1))
    score_mean = np.mean(np_scores)
    score_median = np.mean(np.median(np_scores, axis=1))

    return score_min, score_max, score_mean, score_median


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--isUSPP", action="store_true", default=False)
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--pp-temperature", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    PP, LL = extract_probabilistic_programs(args.temperature, args.num_examples)
    min, max, mean, median = sample_from_probabilistic_programs(PP, LL, args.num_samples,
                                                                args.greedy, args.pp_temperature)

    table = prettytable.PrettyTable()
    table.title = "Text Match Score Statistics"
    table.field_names = ["Minimum", "Maximum", "Mean", "Median"]
    table.add_row([min, max, mean, median])
    print(table)

    with open(f"out/t{args.pp_temperature:.1f}/stats_t{args.temperature:.1f}_n{args.num_examples}_s{args.num_samples}"
              + args.isUSPP*"_isUSPP" + args.greedy*"_greedy" + ".pkl", "wb") as f:
        pickle.dump((min, max, mean, median), f)
