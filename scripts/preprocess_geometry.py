import argparse, base64, io, multiprocessing
import datasets, PIL
import grammar.geometry3k

def filter(x: list, y: str, z: str) -> dict:
    B = io.BytesIO()
    x[0].save(B, format="png")
    img = base64.b64encode(B.getvalue())
    try: ans = grammar.geometry3k.compute(z)
    except: ans = None
    return {"images": img, "problem": y.replace("<image>", ''), "answer": ans}
def discard(X: dict) -> bool: return X["answer"] is not None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    D = datasets.load_dataset(args.dataset, split=args.split)
    S = D.map(filter, input_columns=["images", "problem", "answer"]).filter(discard)

    S.save_to_disk(args.save_dir, num_proc=multiprocessing.cpu_count())
