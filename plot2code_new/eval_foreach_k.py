import argparse, pickle, os
import tqdm, numpy as np
import ppot.utils, plot2code_new.eval_plot2code

def get_save_path(out_path: str, model_name: str, args, append: str = None) -> str:
    """Get save path for generated code"""
    model_name = model_name.split("/")[-1]
    save_path = os.path.join(out_path, model_name if append is None else f"{model_name}_{append}",
                             f"t{args.temperature}_n{args.num_examples}_s{'-'.join(map(str, args.program_samples))}_p{args.program_temperature}_d{args.direct}_u{args.uspp}_r{args.seed}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "ckpt"), exist_ok=True)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10, help="Number of data examples to generate code for")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--save-dir", type=str, default="out/", help="Path to save results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--uspp", default=False, action="store_true")
    parser.add_argument("--sampling-device", type=str, default="cuda:0")
    parser.add_argument("--direct", action="store_true", help="Don't use instruction")
    parser.add_argument("--no-model-loading", action="store_true", default=False)
    parser.add_argument("--program-samples", default=[0, 1, 5, 10, 15, 20], nargs="+", type=int)
    parser.add_argument("--skip-evaluation", action="store_true", default=False)
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    # Configuration
    model_name = args.model_name
    num_examples = args.num_examples

    # Load model and processor
    if not args.no_model_loading:
        model, processor = plot2code_new.eval_plot2code.load_model_and_processor(model_name,
                                                                                 device=args.sampling_device)

    dataset = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples,
                                      lambda x: "matplotlib" in x["url"], split="test")

    # Get save path
    tag = "direct" if args.direct else "instruct"
    save_path = get_save_path(args.save_dir, model_name, args)
    print(f"Results will be saved to {save_path}")

    all_PP = []
    all_S = []

    # Generate code for each sample
    for idx, item in enumerate(tqdm.tqdm(dataset, desc="Generating code")):
        if os.path.isfile(ckpt_path := f"{save_path}/ckpt/gen_{idx}.pkl"):
            with open(ckpt_path, "rb") as f: PP, S = pickle.load(f)
            for P in PP: P.reset_gumbel()
        else:
            # save image to data path
            image_path = os.path.join("data", "images", f"{idx}.png")
            item["image"].save(image_path)

            # Generate programs from the language model
            S, I, L = plot2code_new.eval_plot2code.generate_code(idx, item, model, processor, image_path, save_path,
                        direct=args.direct,
                        temperature=1.0 if args.temperature == 0 else args.temperature,
                        num_return_sequences=args.program_samples[-1])

            # Compile probabilistic programs
            PP, LL = ppot.compile.programs(I, L, processor, code=S)
            with open(ckpt_path, "wb") as f: pickle.dump((PP, S), f)
        all_PP.append(PP)
        all_S.append(S)

    # Free model.
    if not args.no_model_loading:
        del model; ppot.utils.free()


    if args.skip_evaluation:
        import sys
        sys.exit()

    scores = []

    n = args.program_samples[-1]+1
    all_indices_foreach = [np.array([n*i+m for i in range(n-1) for m in range(args.program_samples[j]+1)]) \
                           for j in range(len(args.program_samples))]
    indices_foreach = [[s[0:(args.program_samples[i]+1)*(k+1)] for k in range(n-1)] for i, s in enumerate(all_indices_foreach)]

    for idx, item in enumerate(tqdm.tqdm(dataset, desc="Evaluating")):
        if os.path.isfile(ckpt_path := f"{save_path}/ckpt/eval_{idx}.pkl"):
            with open(ckpt_path, "rb") as f: max_pp_scores = pickle.load(f)
        else:
            # Evaluate the llm generated programs
            _, _, pp_scores = plot2code_new.eval_plot2code.sample_from_probabilistic_programs(
                all_PP[idx], item["code"],
                args.program_samples[-1],
                args.program_temperature,
                return_scores=True
            )
            # This is a matrix where each column refers to n, and each row refers to k.
            # Entries in the matrix are the max score for (k, n).
            max_pp_scores = np.array([[np.max(pp_scores[i]) for i in indices_foreach[j]] \
                                      for j in range(len(args.program_samples))])

            with open(ckpt_path, "wb") as f: pickle.dump(max_pp_scores, f)
        scores.append(max_pp_scores)

    scores = np.array(scores)
    # Rows are LLM samples, columns are PP samples
    mean_scores = np.mean(scores, axis=0) # Take the mean score across all examples.

    with open(f"{save_path}/results.pkl", "wb") as f: pickle.dump({
            "LLM": all_S,
            "PP": all_PP,
            "scores": scores,
            "mean_scores": mean_scores,
    }, f)
