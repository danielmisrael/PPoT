import argparse, json, re
import tqdm, datasets

NUMBER_REGEX_STR = r"-?\d+(\.\d+)?([eE][+-]?\d+)?"
NUMBER_REGEX = re.compile(NUMBER_REGEX_STR)

def _filter(X: dict) -> dict:
    m = NUMBER_REGEX.search(X["answer"])
    u = ''.join((m.string[:m.start()], m.string[m.end():])).strip()
    return {"question": X["question"], "answer": m[0], "unit": u if len(u) > 0 else None}
def filter_data(D: list) -> list:
    return [_filter(X) for X in tqdm.tqdm(D)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save-to", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["jsonl", "arrow"], default="jsonl")
    args = parser.parse_args()

    with open(args.dataset, "r") as f: D = json.load(f)
    D_f = filter_data(D)
    if args.format == "arrow":
        D_f = datasets.Dataset.from_list(D_f)
        D_f.save_to_disk(args.save_to)
    else:
        with open(args.save_to, "w") as f:
            for x in D_f: f.write(json.dumps(x) + "\n")
