import argparse, pickle, os
from ppot.utils import safe_execute_plot

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

    with open(f"/space/poorvagarg/genPPS/out/t{args.pp_temperature:.1f}/stats_t{args.temperature:.1f}_n{args.num_examples}_s{args.num_samples}"
            + args.isUSPP*"_isUSPP" + args.greedy*"_greedy" + "_examples" + ".pkl", "rb") as f:
        triples = pickle.load(f)

    print(len(triples))
    for i in range(len(triples)):
        P = triples[i]
        os.makedirs(f"examples/{i}", exist_ok=True)
        print("--------------------------------------------------")
        print("Sampled Program")
        with open(f"examples/{i}/sample.py", "w") as f:
            f.write(P[0])

        print("--------------------------------------------------")
        print("Raw Program")
        with open(f"examples/{i}/raw.py", "w") as f:
            f.write(P[1][0])
        
        print("--------------------------------------------------")
        print("Ground Truth")
        with open(f"examples/{i}/truth.py", "w") as f:
            f.write(P[2])
        sample = open(f"examples/{i}/sample.py").read()
        raw = open(f"examples/{i}/raw.py").read()
        truth = open(f"examples/{i}/truth.py").read()
        success, result = safe_execute_plot(sample, timeout_seconds=10, separate_process=True)
        breakpoint()




