import argparse, resource
from plot2code_new.text_match_score import evaluate_single_example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_code", type=str, required=True)
    parser.add_argument("--ground_truth_code", type=str, required=True)

    # Limit memory usage.
    MAX_MEM_USAGE = 20_000_000_000
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEM_USAGE, MAX_MEM_USAGE))

    args = parser.parse_args()

    a = open(args.generated_code).read()
    b = open(args.ground_truth_code).read()
    print(evaluate_single_example(a, b))
