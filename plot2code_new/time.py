import argparse, pickle, os, time
import tqdm, numpy as np
import ppot.utils, plot2code_new.eval_plot2code

def get_save_path(out_path: str, model_name: str, args, append: str = None) -> str:
    """Get save path for generated code"""
    model_name = model_name.split("/")[-1]
    save_path = os.path.join(out_path, model_name if append is None else f"{model_name}_{append}",
                             f"t{args.temperature}_n{args.num_examples}_s{'-'.join(map(str, args.program_samples))}_d{args.direct}_u{args.uspp}_r{args.seed}")
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
    parser.add_argument("--uspp", default=False, action="store_true")
    parser.add_argument("--sampling-device", type=str, default="cuda:0")
    parser.add_argument("--direct", action="store_true", help="Don't use instruction")
    parser.add_argument("--no-model-loading", action="store_true", default=False)
    parser.add_argument("--program-samples", default=[0, 1, 5, 10, 15, 20], nargs="+", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--repetitions", default=10, type=int)
    parser.add_argument("--skip-evaluation", action="store_true", default=False)
    parser.add_argument("--stride", default=2, type=int)
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    # Configuration
    model_name = args.model_name
    num_examples = args.num_examples

    # Load model and processor
    if not args.no_model_loading:
        model, processor = plot2code_new.eval_plot2code.load_model_and_processor(model_name)

    dataset = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples,
                                      lambda x: "matplotlib" in x["url"], split="test")

    # Get save path
    tag = "direct" if args.direct else "instruct"
    save_path = get_save_path(args.save_dir, model_name, args)
    print(f"Results will be saved to {save_path}")

    all_PP = [[[] for _ in range(args.program_samples[-1])] for _ in range(args.repetitions)]
    T_LLM = [[[] for _ in range(args.program_samples[-1])] for _ in range(args.repetitions)]
    T_PP = [[[] for _ in range(args.program_samples[-1])] for _ in range(args.repetitions)]

    # Generate code for each sample
    pbar = tqdm.tqdm(range(args.repetitions*args.program_samples[-1]*args.num_examples),
                     desc="Timing LLM sampling and compilation")
    for j in range(args.repetitions):
        for k in range(0, args.program_samples[-1], args.stride):
            for idx, item in enumerate(dataset):
                if os.path.isfile(ckpt_path := f"{save_path}/ckpt/gen_{j}_{k+1}_{idx}.pkl"):
                    with open(ckpt_path, "rb") as f: PP, S, t_llm, t_pp = pickle.load(f)
                    for P in PP: P.reset_gumbel()
                else:
                    # save image to data path
                    image_path = os.path.join("data", "images", f"{idx}.png")
                    item["image"].save(image_path)

                    # Generate programs from the language model
                    t_start = time.time()
                    S, I, L = plot2code_new.eval_plot2code.generate_code(idx, item, model, processor, image_path, save_path,
                                direct=args.direct,
                                temperature=1.0 if args.temperature == 0 else args.temperature,
                                num_return_sequences=k+1)
                    t_llm = time.time()-t_start

                    # Compile probabilistic programs
                    t_start = time.time()
                    PP, _ = ppot.compile.programs(I, L, processor, code=S)
                    t_pp = time.time()-t_start
                    with open(ckpt_path, "wb") as f: pickle.dump((PP, S, t_llm, t_pp), f)
                all_PP[j][k].append(PP)
                T_LLM[j][k].append(t_llm)
                T_PP[j][k].append(t_pp)
                pbar.update()
    pbar.close()

    # Free model.
    if not args.no_model_loading:
        del model; ppot.utils.free()

    if args.skip_evaluation:
        import sys
        sys.exit()

    T_sampling = [[[[] for _ in args.program_samples] for _ in range(args.program_samples[-1])] for _ in range(args.repetitions)]
    T_all = [[[[] for _ in args.program_samples] for _ in range(args.program_samples[-1])] for _ in range(args.repetitions)]

    pbar = tqdm.tqdm(range(args.repetitions*args.program_samples[-1]*len(args.program_samples)*args.num_examples),
                     desc="Timing sampling")
    for j in range(args.repetitions):
        for k in range(0, args.program_samples[-1], args.stride):
            for i, n in enumerate(args.program_samples):
                for idx, item in enumerate(dataset):
                    if os.path.isfile(ckpt_path := f"{save_path}/ckpt/eval_{j}_{k+1}_{n}_{idx}.pkl"):
                        with open(ckpt_path, "rb") as f: t_sampling, t_all = pickle.load(f)
                    else:
                        t_start = time.time()
                        [p.sample(n, as_list=True, t=1.0, constraint=True) for p in all_PP[j][k][idx]]
                        t_sampling = time.time()-t_start
                        t_all = T_LLM[j][k][idx] + T_PP[j][k][idx] + t_sampling
                        with open(ckpt_path, "wb") as f: pickle.dump((t_sampling, t_all), f)
                    T_sampling[j][k][i].append(t_sampling)
                    T_all[j][k][i].append(t_all)
                    pbar.update()
    pbar.close()

    T_LLM = np.array(T_LLM)
    T_PP = np.array(T_PP)
    T_sampling = np.array(T_sampling)
    T_all = np.array(T_all)
    T_LLM_mean = np.mean(np.mean(T_LLM, axis=0), axis=-1)
    T_all_mean = np.mean(np.mean(T_all, axis=0), axis=-1)

    with open(f"{save_path}/results.pkl", "wb") as f: pickle.dump({
            "PP": all_PP,
            "LLM_time": T_LLM,
            "PP_time": T_PP,
            "sampling_time": T_sampling,
            "all_time": T_all,
            "llm_mean_time": T_LLM_mean,
            "all_mean_time": T_all_mean,
    }, f)
