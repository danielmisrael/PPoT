#!/usr/bin/env python3
import os, json, sys, base64, re, shutil, argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from datasets import load_dataset
import torch, matplotlib.pyplot as plt, shutil, argparse, matplotlib, transformers, numpy as np

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
    R = []
    for response_str in responses:
        matches = re.findall(r'```python(.*?)```', response_str, re.DOTALL)
        if matches: R.append("\n".join(match.strip() for match in matches))
        else: R.append(response_str)
    return R

def read_jsonl_file(file_path: str) -> str:
    """Read JSONL file"""
    with open(file_path, 'r') as json_file:
        return [json.loads(line) for line in json_file]

def get_save_path(model_name: str) -> str:
    """Get save path for generated code"""
    model_name = model_name.split("/")[-1]
    save_path = os.path.join("generated_results", model_name, "direct", "instruct")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def generate_code_for_image(model: transformers.AutoModelForCausalLM, processor: AutoProcessor,
                            image_path: str, instruction: str, nsamples: int) -> tuple:
    """Generate code for a single image"""

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
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            top_p=1.0,
            top_k=0,
            temperature=0.7,
            max_new_tokens=2048,
            num_return_sequences=nsamples,
        )
        generated_ids_trimmed = [out_ids[inputs.input_ids.numel():] for out_ids in generated_ids]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # Extract code
    code = extract_code(output_text)

    return code, generated_ids_trimmed

def main():
    """Generate code using Qwen2.5-VL-3B-Instruct model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None, help="Number of data examples to generate code for")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    args = parser.parse_args()

    # Configuration
    model_name = args.model_name
    num_examples = args.num_examples

    # Load model and processor
    model, processor = load_model_and_processor(model_name)

    dataset = load_dataset("TencentARC/Plot2Code", split="test")
    dataset = dataset.filter(lambda x: "matplotlib" in x["url"])
    if num_examples is not None:
        dataset = dataset.select(range(num_examples))


    # Get save path
    save_path = get_save_path(model_name)
    print(f"Results will be saved to {save_path}")

    os.makedirs("data/images", exist_ok=True)

    results_save_path = os.path.join(save_path, "generated_code.jsonl")
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    # Generate code for each sample
    results = []
    for item in tqdm(dataset, desc="Generating code"):

        if "matplotlib" not in item['url']:
            continue

        idx = len(results)
        image = item['image']
        instruction = item['instruction']

        # save image to data path
        image_path = os.path.join("data", "images", f"{idx}.png")
        image.save(image_path)

        # Generate code
        generated_code, gen_ids = generate_code_for_image(model, processor, image_path,
                                                          instruction, args.num_samples)

        for i, x in enumerate(generated_code):
            generated_image_path = os.path.join(save_path, f"{idx}-{i}.png")
            try:
                exec(x)
            except Exception as e:
                print(f"Error executing code: {e}")
            fig = plt.gcf()
            fig.savefig(generated_image_path)
            plt.close()
            matplotlib.rcdefaults()
            plt.cla()
            plt.clf()
            plt.close("all")


            # Create result item
            result = {
                'idx': f"{idx}-{i}",
                'ground_truth_path': image_path,
                'code': x,
                'ground_truth_code': item['code'],
                'generated_image_path': generated_image_path
            }
            results.append(result)

            # Save incrementally
            with open(results_save_path, 'a') as f:
                f.write(json.dumps(result) + '\n')

    print(f"Generated code for {len(results)} samples")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
