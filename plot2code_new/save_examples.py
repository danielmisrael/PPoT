import pickle, collections, numpy as np, pandas as pd
raw, S = {}, {}
sizes, direct, num_samples = [3, 7], [False, True], [1, 5]
for s in sizes:
    raw[s], S[s] = {}, {}
    for d in direct:
        raw[s][d], S[s][d] = {}, {}
        for n in num_samples:
            raw[s][d][n], S[s][d][n] = {}, {}
            with open(f"out/Qwen2.5-VL-{s}B-Instruct/t0.7_n132_s5_z{n}_p1.0_d{d}_uFalse_r0/results.pkl", "rb") as f:
                r = pickle.load(f)
            S[s][d][n] = {"LLM_scores": np.mean(r["LLM_scores"]).item(), "PP_scores": np.mean(r["PP_scores"]).item()}
            raw[s][d][n] = r
M = np.array([[[[S[s][d][n][x] for x in S[s][d][n]] for n in S[s][d]] for d in S[s]] for s in S])

import difflib, sys, plot2code_new.eval_plot2code, shutil
def save_plot(C_1: str, C_2: str, path: str):
    f = lambda x: f"\nplt.savefig('playground/{path + x}.png')\n"
    C_1 += f("_llm")
    C_2 += f("_pp")
    plot2code_new.eval_plot2code.subprocess_call(C_1, C_2, f"playground/{path}_llm.py", f"playground/{path}_pp.py")
    
B = {s: {d: {n: [(X["best"], difflib.get_close_matches(X["best"], X["LLMs"], n=1, cutoff=0.0)[0]) for i, X in raw[s][d][n]["Better"].items()] for n in raw[s][d]} for d in raw[s]} for s in raw}
quit = False
for s in sizes:
    for d in direct:
        for n in num_samples:
            I = list(raw[s][d][n]["Better"].keys())
            print(f"Model size {s} | Direct? {d} | #samples {n}\n---\n")
            for i, (x, y) in enumerate(B[s][d][n]):
                print(f"Best:\n{x}\n\n=======\n\nLLM:\n{y}\n\n=======\n\nDiff:\n")
                X, Y = [u + '\n' for u in x.split('\n')], [u + '\n' for u in y.split('\n')]
                sys.stdout.writelines(difflib.unified_diff(Y, X, fromfile="LLM sample", tofile="PP sample"))
                pre = f"{s}_{d}_{n}_{i}"
                save_plot(x, y, pre)
                shutil.copyfile(f"data/images/{I[i]}.png", f"playground/{pre}_gt.png")
                print("\n============\n\n")
                #if input() == 'q':
                #    quit = True
                #    break
            if quit: break
        if quit: break
    if quit: break
