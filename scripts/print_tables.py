import pickle
import prettytable, numpy as np

if __name__ == "__main__":
    M = {"raw": "Raw programs", "greedy": "Greedy", "USPP": "Sample one RV", "sample": "Sample all RVs"}
    T_pp = np.arange(0, 2.1, 0.1)
    S = {k: {u: {} for u in T_pp} for k in M}
    for u in T_pp:
        for t in np.arange(0, 1.1, 0.1):
            with open(f"out/t{u:.1f}/stats_t{t:.1f}_n1000_s1000.pkl", "rb") as f: S["sample"][u][t] = pickle.load(f)
            with open(f"out/t1.0/stats_t{t:.1f}_n1000_s1000_greedy.pkl", "rb") as f: S["greedy"][u][t] = pickle.load(f)
            with open(f"out/t{u:.1f}/stats_t{t:.1f}_n1000_s1000_isUSPP.pkl", "rb") as f: S["USPP"][u][t] = pickle.load(f)
            with open(f"out/t1.0/stats_t{t:.1f}_raw_programs.pkl", "rb") as f: S["raw"][u][t] = pickle.load(f)[:,0]
            S["raw"][u][t] = 4*[np.mean(S["raw"][u][t]).tolist()]

    for t in np.arange(0, 1.1, 0.1):
        T = prettytable.PrettyTable()
        T.title = f"Text match score statistics"
        T.field_names = ["Strategy", "Temperature", "Min", "Max", "Mean", "Median"]
        T.align = 'l'
        for u in T_pp:
            T.add_rows([[M[m], np.round(u, 1), *[np.round(x, 5) for x in S[m][u][t]]] for m in M],
                      divider=True)
        print("Program temperature =", t)
        print(T)

