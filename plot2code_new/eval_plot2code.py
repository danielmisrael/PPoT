#!/usr/bin/env python3
import os, json, base64, re, argparse, pickle, gc, multiprocessing, subprocess, uuid
import torch, transformers, numpy as np, dill, tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import ppot.compile, ppot.utils

dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump
# multiprocessing.queues._ForkingPickler = dill.Pickler

def subprocess_call(p, g, pfile, gfile):
    with open(pfile, "w") as f:
        f.write(p)
    with open(gfile, "w") as f:
        f.write(g)

    result = subprocess.run(['python', '-m', 'scripts.compute_tms', '--generated_code', pfile, '--ground_truth_code', gfile], capture_output=True, text=True, check=True)
    return float(result.stdout.strip("\n"))

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Load the model and processor"""
    print(f"Loading model: {model_name}")

    # Load model with explicit CUDA settings
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor

def extract_code(responses: list) -> list:
    """Extract code from response string"""
    code_responses = []
    for response_str in responses:
        matches = re.findall(r'```python(.*?)```', response_str, re.DOTALL)
        if matches: code_responses.append("\n".join(match.strip() for match in matches))
        else: code_responses.append(response_str)
    return code_responses

def read_jsonl_file(file_path: str) -> str:
    """Read JSONL file"""
    with open(file_path, 'r') as json_file:
        return [json.loads(line) for line in json_file]

def get_save_path(out_path: str, model_name: str, args, append: str = None) -> str:
    """Get save path for generated code"""
    model_name = model_name.split("/")[-1]
    save_path = os.path.join(out_path, model_name if append is None else f"{model_name}_{append}",
                             f"t{args.temperature}_n{args.num_examples}_s{args.num_samples}_z{args.num_llm_samples}_p{args.program_temperature}_d{args.direct}_u{args.uspp}_r{args.seed}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "ckpt"), exist_ok=True)
    return save_path

def generate_code_for_image(model: transformers.AutoModelForCausalLM, processor: AutoProcessor,
                            image_path: str, instruction: str | None,
                            num_return_sequences: int = 1, temperature: float = 1.0, **kwargs) -> tuple:
    """Generate code for a single image"""

    if instruction is None:
        text_prompt = f"Please generate Python matplotlib code to create a plot that looks like the given image. The code should be surrounded by ```python and ```."
    else:
        # Create prompt
        text_prompt = f"{instruction}\n\nPlease generate Python matplotlib code to create a plot that looks like the given image. The code should be surrounded by ```python and ```."

    # Create messages format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Generate response using the model
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    gen_kwargs = {"do_sample": False} if temperature == 0 or num_return_sequences == 1 else \
        {"top_p": 1.0, "top_k": 0, "do_sample": True}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **gen_kwargs,
            max_new_tokens=2048,
            return_dict_in_generate=True,
            output_logits=True,
            # output_scores=True, # output_scores correspond to the true logits the model is sampling from
            repetition_penalty=1.0, # in this case scores and logits are the same
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            **kwargs,
        )
        generated_ids_trimmed = out.sequences[:,inputs.input_ids.numel():].cpu()
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # Extract code
    code = extract_code(output_text)
    logits = torch.concatenate(tuple(x.cpu() for x in out.logits), dim=-1).reshape(out.logits[0].shape[0], len(out.logits), -1)
    # scores = torch.concatenate(tuple(x.cpu() for x in out.scores), dim=-1).reshape(out.scores[0].shape[0], len(out.scores), -1)

    return code, generated_ids_trimmed, logits

def generate_code(idx: int, item: dict, model: transformers.AutoModel,
                         processor: transformers.AutoProcessor, ground_truth_path: str,
                         output_path: str, direct: bool = False, **kwargs):
    chkpnt_path = os.path.join(output_path, "ckpt", f"{idx}")
    if os.path.isfile(chkpnt_path): return
    code, ids, logits = generate_code_for_image(model, processor, ground_truth_path,
                                                item["instruction"] if not direct else None, **kwargs)
    return code, ids, logits

def _sample_task(p: ppot.program.Program, greedy: bool, num_samples: int, t: float) -> list:
    samples = p.sample(num_samples, as_list=True, t=t, constraint=True)
    for i in samples:
        assert i != p.raw_program, "Sampled program is the same as raw program, which should not happen with constraint=True"
    return samples

def _get_greedy(p:ppot.program.Program) -> list:
    return p.greedy(as_list=True)

def _get_raw(p:ppot.program.Program) -> list:
    return [p.raw_program]

def evaluate_programs(to_run: list, gt_code: str) -> list:
    """
    Evaluates multiple programs in parallel
    """
    os.makedirs("raw", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    with multiprocessing.Pool() as pool:
        procs = []
        for i in range(len(to_run)):
            pfile = f"raw/p_{uuid.uuid4().hex}.py"
            gfile = f"raw/g_{uuid.uuid4().hex}.py"
            procs.append([pool.apply_async(subprocess_call, (to_run[i], gt_code, pfile, gfile)), pfile, gfile])

        text_match_scores = []
        for P in procs:
            try: r = P[0].get(60)
            except Exception as exc:
                r = 0
                print(">>>>>>>>>", exc)
            os.remove(P[1])
            os.remove(P[2])
            text_match_scores.append(r)

    return text_match_scores

def example_programs(raw_programs: list, raw_scores: list, sample_programs: list, sample_scores: list,
                     dataset: dict) -> list:
    "Return example programs where sampled probabilistic program does better than raw program"
    max_sample_scores = np.max(np.array(sample_scores), axis=1)
    argmax_sample_scores = np.argmax(np.array(sample_scores), axis=1)
    triples = []
    for i, (RP, RS, MS, AS) in enumerate(zip(raw_programs, raw_scores, max_sample_scores, argmax_sample_scores)):
        if MS > RS:
            triples.append([sample_programs[i][AS], RP, dataset['code'][i]])

    return triples

def sample_from_probabilistic_programs(PP: list, gt_code: str, num_samples: int, pp_temp: float,
                                       return_scores: bool = False) -> list:
    """
    Sample from probabilistic programs and evaluate them with respect to the actual image.
    It returns statistics of the results
    """
    to_run_sample = []
    for i in range(len(PP)):
        to_run_sample.extend(_sample_task(PP[i], False, num_samples, pp_temp))
        to_run_sample.extend(_get_raw(PP[i]))
    text_match_scores = evaluate_programs(to_run_sample, gt_code)
    retval = (np.max(text_match_scores), to_run_sample[np.argmax(text_match_scores)])
    if return_scores: return *retval, text_match_scores
    return retval

SUPP_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct",
               "Qwen/Qwen2.5-VL-1B-Instruct",
               "Qwen/Qwen2.5-VL-7B-Instruct"]

def main():
    """Generate code using Qwen2.5-VL-Instruct model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10, help="Number of data examples to generate code for")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    parser.add_argument("--num-llm-samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate from probabilistic program")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--save-dir", type=str, default="out/", help="Path to save results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--uspp", default=False, action="store_true")
    parser.add_argument("--sampling-device", type=str, default="cuda:0")
    parser.add_argument("--direct", action="store_true", help="Don't use instruction")
    parser.add_argument("--pause-for-inspection", action="store_true", default=False,
                        help="Whether to pause for instruction and inspect an example.")
    parser.add_argument("--no-model-loading", action="store_true", default=False)
    args = parser.parse_args()

    ppot.utils.seed(args.seed)

    # Configuration
    model_name = args.model_name
    num_examples = args.num_examples

    # Load model and processor
    if not args.no_model_loading:
        model, processor = load_model_and_processor(model_name)

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
            S, I, L = generate_code(idx, item, model, processor, image_path, save_path,
                        direct=args.direct,
                        temperature=1.0 if args.temperature == 0 else args.temperature,
                        num_return_sequences=1 if args.temperature == 0 else args.num_llm_samples)

            # Compile probabilistic programs
            PP, LL = ppot.compile.programs(I, L, processor, code=S)
            with open(ckpt_path, "wb") as f: pickle.dump((PP, S), f)
        all_PP.append(PP)
        all_S.append(S)

    # Free model.
    if not args.no_model_loading:
        del model; ppot.utils.free()

    LLM_scores = []
    PP_scores = []
    better_programs = {}

    for idx, item in enumerate(tqdm.tqdm(dataset, desc="Evaluating")):
        if os.path.isfile(ckpt_path := f"{save_path}/ckpt/eval_{idx}.pkl"):
            with open(ckpt_path, "rb") as f: max_llm_score, max_score, better = pickle.load(f)
        else:
            # Evaluate the llm generated programs
            llm_scores = evaluate_programs(all_S[idx], item["code"])
            llm_scores = np.array(llm_scores).flatten()
            max_llm_score = np.max(llm_scores)

            max_score, argmax_program = sample_from_probabilistic_programs(all_PP[idx],
                                                                           item["code"],
                                                                           args.num_samples,
                                                                           args.program_temperature)
            better = None
            if (max_score > max_llm_score):
                print(f"Example {idx} - Probabilistic program outperforms LLM generated code: {max_score} vs {max_llm_score}")
                better = {"best": argmax_program, "LLMs": all_S[idx]}
                if args.pause_for_inspection: breakpoint()

            with open(ckpt_path, "wb") as f: pickle.dump((max_llm_score, max_score, better), f)
        LLM_scores.append(max_llm_score)
        PP_scores.append(max_score)
        if better is not None: better_programs[idx] = better

    print(args)
    print(np.mean(LLM_scores))
    print(np.mean(PP_scores))

    with open(f"{save_path}/results.pkl", "wb") as f: pickle.dump({
            "LLM": all_S,
            "PP": all_PP,
            "LLM_scores": LLM_scores,
            "PP_scores": PP_scores,
            "Better": better_programs,
    }, f)

if __name__ == "__main__":
    main()
