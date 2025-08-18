import argparse, pickle

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

    user_input = True
    for i, P in enumerate(triples):
        print("Sampled Program")
        print(P[0])

        cont = input()
        print("Raw Program")
        print(P[1][0])
        
        cont=input()
        print("Ground Truth")
        print(P[2][0])
        user_input = input()
