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
for i, s in enumerate(sizes):
    for j, d in enumerate(direct):
        print(f"Model size: {s} | Direct? {d}\n---")
        print(pd.DataFrame(M[i][j], columns=["LLM", "LLM + PP"], index=[f"{n} LLM_samples" for n in
                                                                        S[s][d]]))
        print("========")
