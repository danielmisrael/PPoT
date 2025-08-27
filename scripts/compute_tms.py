import argparse
from scripts.text_match_score import evaluate_single_example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_code", type=str, required=True)
    parser.add_argument("--ground_truth_code", type=str, required=True)

    args = parser.parse_args()

    a = open(args.generated_code).read()
    b = open(args.ground_truth_code).read()
    print(evaluate_single_example(a, b))
