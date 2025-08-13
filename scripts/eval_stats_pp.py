import pickle, argparse
import torch, prettytable, tqdm, numpy as np

def retrieve_programs(path: str, n: int) -> tuple:
    "Retrieves probabilistic programs and normalized log-likelihoods from disk."
    with open(path, "rb") as f:
        R = pickle.load(f)
        if isinstance(R, tuple): return R
        PP, LL = R, pickle.load(f)
    if n is not None: return [P[:n] for P in PP], [L[:n] for L in LL]
    return PP, LL

def foreach(X: list, f) -> list:
    return [[f(y.numpy() if torch.is_tensor(y) else y) for y in x] for x in tqdm.tqdm(X)]
def reduce(X: list, f) -> list:
    return [f(x.numpy() if torch.is_tensor(x) else x).item() for x in tqdm.tqdm(X)]
def basic_stats(X: list, append: str) -> dict:
    return {
        f"μ({append})": reduce(X, np.mean),
        f"σ({append})": reduce(X, np.std),
        f"↓({append})": reduce(X, np.min),
        f"↑({append})": reduce(X, np.max),
    }

def stats(PP: list, LL: list, nsamples: int) -> list:
    "Computes statistics on the probabilistic programs."
    # Sampled programs.
    S = foreach(PP, lambda x: x.sample(nsamples))
    # Softmax the normalized log-likelihoods.
    L = torch.vstack(LL).log_softmax(dim=-1)
    # Compute number of unique programs for each probabilistic program.
    U = foreach(S, lambda x: len(set(x)))
    # Compute number of random variables for each probabilistic program.
    RV = foreach(PP, lambda x: 0 if x.deterministic else len(x.mapping))
    # Compute entropy of random variables for each probabilistic program.
    H = foreach(PP, lambda x: torch.zeros(1) if x.deterministic else
                -torch.sum(x.P_tensor*torch.exp(x.P_tensor), dim=-1))
    # Compute statistics on the entropy.
    mu_H, sigma_H, min_H, max_H = (foreach(H, f) for f in (np.mean, np.std, np.min, np.max))
    # Aggregate all as a dictionary.
    cats = ["U", "N", "μ(H)", "σ(H)", "↓(H)", "↑(H)"]
    vals = [U, RV, mu_H, sigma_H, min_H, max_H]
    D = {}
    for c, v in zip(cats, vals): D.update(basic_stats(v, c))
    return D

def print_table(D: dict):
    T = prettytable.PrettyTable()
    for k, v in D.items(): T.add_column(k, v)
    T.float_format = ".3"
    T.format = True
    T.header = True
    T.hrules = prettytable.HRuleStyle.FRAME
    return T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-programs", type=str,
                        default="/space/poorvagarg/genPPS/probprog.pkl")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-programs", type=int, default=None)
    args = parser.parse_args()

    PP, LL = retrieve_programs(args.path_to_programs, args.num_programs)
    D = stats(PP, LL, args.num_samples)
    T = print_table(D)
    print(T)
