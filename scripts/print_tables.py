import pickle
import prettytable, numpy as np

if __name__ == "__main__":
    M = {"raw": "Raw programs", "greedy": "Greedy", "USPP": "Sample one RV", "sample": "Sample all RVs"}
    S = {k: {} for k in M}
    for t in np.arange(0, 1.1, 0.1):
        with open(f"out/stats_t{t:.1f}_n1000_s1000.pkl", "rb") as f: S["sample"][t] = pickle.load(f)
        with open(f"out/stats_t{t:.1f}_n1000_s1000_greedy.pkl", "rb") as f: S["greedy"][t] = pickle.load(f)
        with open(f"out/stats_t{t:.1f}_n1000_s1000_isUSPP.pkl", "rb") as f: S["USPP"][t] = pickle.load(f)
        with open(f"out/stats_t{t:.1f}_raw_programs.pkl", "rb") as f: S["raw"][t] = pickle.load(f)[:,0]
        S["raw"][t] = 4*[np.mean(S["raw"][t]).tolist()]

    T = prettytable.PrettyTable()
    T.title = f"Text match score statistics"
    T.field_names = ["Strategy", "Temperature", "Min", "Max", "Mean", "Median"]
    T.align = 'l'
    for t in np.arange(0, 1.1, 0.1):
        T.add_rows([[M[m], np.round(t, 1), *[np.round(x, 5) for x in S[m][t]]] for m in M],
                  divider=True)
    print(T)
