import argparse, os, pickle, multiprocessing
import transformers, tqdm, numpy as np
import scripts.text_match_score, ppot.utils

def load_programs(t: float) -> list:
    data_path = f"/space/renatolg/genPPS/out/Qwen2.5-VL-3B-Instruct_t{t:.1f}/data"
    F = os.listdir(data_path)
    C = []
    for f in tqdm.tqdm(F, desc="Loading programs"):
        with open(f"{data_path}/{f}", "rb") as file: C.append(pickle.load(file)["code"])
    return C

def evaluate(C: list) -> np.ndarray:
    S = []
    D = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples=len(C),
                                filter_fn = lambda x: "matplotlib" in x["url"], split="test")
    with multiprocessing.Pool() as pool:
        procs = [[pool.apply_async(scripts.text_match_score.evaluate_single_example, (p, g)) \
                  for p, g in zip(X, D["code"])] for X in C]
        for P in tqdm.tqdm(procs, desc="Evaluating"):
            U = []
            for p in P:
                try: r = p.get(30)
                except Exception as exc:
                    r = 0
                    print(">>>>>>>>>>>>", exc)
                U.append(r)
            S.append(U)
    return np.array(S)

def stats(S: np.ndarray) -> tuple:
    return tuple(np.mean(f(S, axis=1)) for f in (np.min, np.max, np.mean, np.median))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()
    print(args)

    C = load_programs(args.temperature)
    S = evaluate(C)
    Q = stats(S)
    print(Q)

    with open(f"out/stats_t{args.temperature:.1f}_raw_programs.pkl", "wb") as f: pickle.dump(S, f)
