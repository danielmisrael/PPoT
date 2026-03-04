# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from utils_general import (
    evaluate_score,
    pass_at_k,
)

def evaluate_generations(generations : dict[str, list], mode):
    # Load the samples
    dataset = [json.loads(l) for l in open("../data/cruxeval.jsonl", "r").readlines()]
    references = [(doc["code"], doc["input"], doc["output"]) for doc in dataset]
    
    # print(dataset[0])
    # print(generations)

    # Run the samples - only evaluate the samples that were actually generated
    try:
        # Find the maximum sample index that exists in generations
        max_sample_idx = max([int(k.split('_')[1]) for k in generations.keys() if k.startswith('sample_')])
        generations_list = []
        for i in range(max_sample_idx+1):
            if i != 162 and i!= 342:
                generations_list.append(generations[f"sample_{i}"])
        # generations_list = [generations[f"sample_{i}"] for i in range(max_sample_idx + 1)]
        # Only use the corresponding references
        references = references[:162] + references[163:342] + references[343:max_sample_idx + 1]
    except:
        assert False, "check format of generations, should be dictionary of lists with keys of id's in the form sample_i"
        
    with ProcessPoolExecutor() as executor:
        args_list = zip(generations_list, references, [mode] * len(generations_list))
        results = executor.map(evaluate_score, args_list)
    all_scores = list(results)

    # breakpoint()
    # Compute pass@k scores
    # Determine k values based on the number of generations per sample
    n_generations = len(all_scores[0]) if all_scores else 0
    k_values = [1, 5, n_generations]  # Common k values to compute
    k_values = [k for k in k_values if k <= n_generations]  # Only compute for valid k
    k_values = set(k_values)
    # breakpoint()
    pass_at_k_scores = {f"pass_at_{k}": [] for k in k_values}
    
    for execution_result in all_scores:
        c, n = execution_result.count(True), len(execution_result)
        for k in k_values:
            pass_at_k_scores[f"pass_at_{k}"].append(pass_at_k(n, c, k))
    
    # Compute averages
    results = {
        "raw_generations": generations,
        "raw_scored_generations": {f"sample_{i}": all_scores[i] for i in range(len(generations_list))}
    }
    
    for k in k_values:
        scores = pass_at_k_scores[f"pass_at_{k}"]
        print(len(scores))
        results[f"pass_at_{k}"] = sum(scores) / len(scores) * 100
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_path", 
        help="JSON path containing outputs to evaluate. Should contain a list of \
              length 800, where each element is a list of different generations \
              for that benchmark sample.",
        type=str,
    )
    parser.add_argument(
        "--scored_results_path", 
        help="path to dump scored results",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mode", 
        help="either input or output, depending on which one to evaluate",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    generations = json.load(open(args.generations_path, "r"))
    print(f"Scoring {args.generations_path}... expect around a minute")

    if "input" in args.generations_path: args.mode = "input"
    else: args.mode = "output"

    results = evaluate_generations(generations, args.mode)
    print(f"Finished!")
    
    # Print all pass@k scores
    for key in sorted(results.keys()):
        if key.startswith("pass_at_"):
            print(results[key])
            print(f"{key}: {round(results[key], 1)}")
    
    if args.scored_results_path != None:
        print(f"Dumping to {args.scored_results_path}")
        json.dump(results, open(args.scored_results_path, "w"))
