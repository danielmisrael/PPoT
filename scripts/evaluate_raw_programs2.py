import argparse, os, pickle, multiprocessing
import transformers, tqdm, numpy as np
import scripts.text_match_score, ppot.utils
from scripts.text_match_score import evaluate_single_example

file_handle = open("deepseek_raw_programs.pkl", "rb")
C = pickle.load(file_handle)

S = []
D = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples=len(C),
                            filter_fn = lambda x: "matplotlib" in x["url"], split="test")

for i in tqdm.tqdm(range(132), desc="Evaluating"):
    p = C[i]['code']
    g = D['code'][i]
    try: 
        s = evaluate_single_example(p, g)
    except Exception as exc:
        s = 0
        print(">>>>>>>>>>>>", exc)
    S.append(s)

print(np.mean(np.array(S)))