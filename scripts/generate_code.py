#!/usr/bin/env python3
import os, json, sys, base64, re, shutil, argparse, pickle, gc, pathlib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch, matplotlib.pyplot as plt, shutil, argparse, matplotlib, transformers, numpy as np
import datasets, tqdm, PIL
import ppot.utils as utils, ppot.compile
from typing import Optional

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
    """Load the model and processor"""
    print(f"Loading model: {model_name}")

    lc_name = model_name.lower()
    if "qwen" in lc_name:
        # Load model with explicit CUDA settings
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            **kwargs,
        )
    elif "llama" in lc_name:
        model =  transformers.MllamaForConditionalGeneration.from_pretrained(model_name,
                                                               torch_dtype=torch.bfloat16,
                                                               device_map="cuda", **kwargs)
    elif "gemma" in lc_name:
        #quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.Gemma3ForConditionalGeneration.from_pretrained(model_name,
                                                                            device_map="cuda",
                                                                            #quantization_config=quantization_config).eval(),
                                                                            **kwargs).eval()
    else: raise ValueError("Model should be either Llama 3.2, Qwen 2.5 VL or Gemma 3.")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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

def get_save_path(out_path: str, model_name: str, append: str = None) -> str:
    """Get save path for generated code"""
    model_name = model_name.split("/")[-1]
    save_path = os.path.join(out_path, model_name if append is None else f"{model_name}_{append}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "ckpt"), exist_ok=True)
    return save_path

def _qwen_generate_code_for_image(model: transformers.AutoModelForCausalLM,
                                  processor: AutoProcessor, image_path: str, text_prompt: str,
                                  **kwargs) -> tuple:
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
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)

    return inputs, out

@torch.no_grad()
def _llama_generate_code_for_image(model: transformers.AutoModelForCausalLM,
                                   processor: AutoProcessor, image_path: str,
                                   text_prompt: str, **kwargs) -> tuple:
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": text_prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    img = PIL.Image.open(image_path)
    inputs = processor(img, text, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **kwargs)
    return inputs, out

@torch.no_grad()
def _gemma_generate_code_for_image(model: transformers.AutoModelForCausalLM,
                                   processor: AutoProcessor, image_path: str,
                                   text_prompt: str, **kwargs) -> tuple:
    messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image_path}, {"type": "text", "text": text_prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                       return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    return inputs, out

def generate_code_for_image(model: transformers.AutoModelForCausalLM, processor: AutoProcessor,
                            image_path: str, instruction: Optional[str] = None, top_p: float = 1.0,
                            top_k: int = 0, max_new_tokens: int = 2048,
                            return_dict_in_generate: bool = True, output_logits: bool = True,
                            repetition_penalty: float = 1.0, do_sample: bool =  True, **kwargs) -> tuple:
    """Generate code for a single image"""

    if instruction is None:
        text_prompt = f"Please generate Python matplotlib code to create a plot that looks like the given image. The code should be surrounded by ```python and ```."
    else:
        # Create prompt
        text_prompt = f"{instruction}\n\nPlease generate Python matplotlib code to create a plot that looks like the given image. The code should be surrounded by ```python and ```."
    name = model.name_or_path.lower()
    if "llama" in name: f = _llama_generate_code_for_image
    elif "qwen" in name: f = _qwen_generate_code_for_image
    elif "gemma" in name: f = _gemma_generate_code_for_image
    else: raise ValueError("Model should be either Llama 3.2, Qwen 2.5 VL or Gemma 3.")
    inputs, out = f(model, processor, image_path, text_prompt, top_p=top_p, top_k=top_k,
                    max_new_tokens=max_new_tokens, return_dict_in_generate=return_dict_in_generate,
                    output_logits=output_logits, repetition_penalty=repetition_penalty,
                    do_sample=do_sample, **kwargs)
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
                  processor: transformers.AutoProcessor, ground_truth_path: str, output_path: str,
                  direct: bool = False, compress: bool = False, save_pp: bool = False, **kwargs):
    chkpnt_path = os.path.join(output_path, "ckpt", f"{idx}")
    # if os.path.isfile(chkpnt_path): return
    print(ground_truth_path)

    code, ids, logits = generate_code_for_image(model, processor, ground_truth_path,
                                                item["instruction"] if not direct else None, **kwargs)
    for i, x in enumerate(code):
        generated_image_path = os.path.join(output_path, "imgs", f"{idx}-{i}.png")

        # Create result item
        result = {
            'idx': f"{idx}-{i}",
            'ground_truth_path': ground_truth_path,
            'code': x,
            'ground_truth_code': item["code"],
            'generated_image_path': generated_image_path
        }

    if compress:
        ids = ids.to(dtype=torch.int32)
        logits = logits.to(dtype=torch.float16)
    tag = "direct/" if direct else "instruct/"
    os.makedirs(os.path.join(output_path, tag), exist_ok=True)
    with open(os.path.join(output_path, tag, "generated_code.jsonl"), 'a') as f: f.write(json.dumps(result) + '\n')
    if save_pp:
        P = ppot.compile.programs(ids, logits, processor, code=code)[0][0]
        os.makedirs(os.path.join(output_path, "prob_programs"), exist_ok=True)
        with open(os.path.join(output_path, "prob_programs", f"{idx}.pkl"), "wb") as f:
            pickle.dump(P, f)
    else:
        with open(os.path.join(output_path, "data", f"{idx}.pkl"), "wb") as f:
            pickle.dump({"code": code, "ids": ids, "logits": logits}, f)
    with open(chkpnt_path, "w") as f: f.write(' ')

def main():
    """Generate code using Qwen2.5-VL-3B-Instruct model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None, help="Number of data examples to generate code for")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--save_dir", type=str, default="out/", help="Path to save results")
    parser.add_argument("--direct", default=False, action="store_true", help="Don't use instruction")
    parser.add_argument("--compress", default=False, action="store_true", help="Whether to store logit as 8-bit")
    parser.add_argument("--save_pp", default=False, action="store_true", help="Whether to compile and save the probabilistic programs without saving the logits to disk")
    parser.add_argument("--quantize", type=int, choices=[4, 8, 16], help="Quantization bits", default=16)

    args = parser.parse_args()
    print(args)

    # Configuration
    model_name = args.model_name
    num_examples = args.num_examples

    # Load model and processor
    quantization_config = transformers.BitsAndBytesConfig(**{f"load_in_{args.quantize}bit": True}) if args.quantize <= 16 else None
    model, processor = load_model_and_processor(model_name, quantization_config=quantization_config)

    dataset = utils.prepare_data("TencentARC/Plot2Code", num_examples,
                                 lambda x: "matplotlib" in x["url"], split="test")

    # Get save path
    tag = "direct" if args.direct else "instruct"
    save_path = get_save_path(args.save_dir, model_name, append=f"t{args.temperature}_{tag}")
    print(f"Results will be saved to {save_path}")

    # Generate code for each sample
    for idx, item in enumerate(tqdm.tqdm(dataset, desc="Generating code")):
        # save image to data path
        image_path = os.path.join("data", "images", f"{idx}.png")
        item["image"].save(image_path)
        # Generate
        generate_code(idx, item, model, processor, image_path, save_path,
                    direct=args.direct,
                    temperature=1.0 if args.temperature == 0 else args.temperature,
                    num_return_sequences=1 if args.temperature == 0 else args.num_samples,
                    do_sample=args.temperature > 0, compress=args.compress, save_pp=args.save_pp)

if __name__ == "__main__":
    main()
