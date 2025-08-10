#!/usr/bin/env python3
import os, json, sys, base64, re, shutil, argparse, pickle, gc
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from datasets import load_dataset
import torch, matplotlib.pyplot as plt, shutil, argparse, matplotlib, transformers, numpy as np
# Add vLLM imports
from vllm import LLM, SamplingParams
import os

# Set vLLM to use V2 for logits support
os.environ["VLLM_USE_V1"] = "0"

class LogitsSpy:
    def __init__(self):
        self.processed_logits: list[torch.Tensor] = []
    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.processed_logits.append(logits.cpu())
        return logits
    
    def reset(self):
        self.processed_logits = []

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Load the model and processor using vLLM"""
    print(f"Loading model: {model_name}")

    # Load model using vLLM
    model = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="half", 
        gpu_memory_utilization=0.8,  
        max_model_len=16384,  
        enable_prefix_caching=True,   
        
    )

    # Load processor for text processing
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    return model, processor

def extract_code(responses: list) -> list:
    """Extract code from response string"""
    extracted_code = []
    for response_str in responses:
        matches = re.findall(r'```python(.*?)```', response_str, re.DOTALL)
        if matches: 
            extracted_code.append("\n".join(match.strip() for match in matches))
        else: 
            extracted_code.append(response_str)
    return extracted_code

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

def generate_code_for_image(model: LLM, processor: AutoProcessor,
                            image_path: str, instruction: str, nsamples: int, temperature: float) -> tuple:
    """Generate code for a single image using vLLM"""

    # Create prompt
    text_prompt = f"{instruction}\n\nPlease generate Python matplotlib code to create a plot that looks like the given image. The code should be surrounded by ```python and ```."

    # Create logits spy to capture logits
    logits_spy = LogitsSpy()
    
    # Create vLLM sampling parameters optimized for speed
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        max_tokens=2048,
        n=1,
        logits_processors=[logits_spy] # TODO will incorporate later
    )

    # Load the image using PIL
    image = Image.open(image_path)
    
    conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content":[{"type": "image_pil", "image_pil": image}]},
    {"role": "user", "content": text_prompt},
    {"role": "assistant", "content": ""}]

    
    
    output_text = []
    logits = []
    
    for _ in range(nsamples):
        outputs = model.chat(
            conversation,
            sampling_params=sampling_params
        )
    
        for output in outputs:
            for completion in output.outputs:
                output_text.append(completion.text)
                
        logits.append(logits_spy.processed_logits)
        logits_spy.reset()
        
    
    # Extract code
    code = extract_code(output_text)
    
    
    # TODO fix this to be the actual generated ids. vLLM doesn't have direct access to token IDs in the same way
    generated_ids_trimmed = [[] for _ in range(len(output_text))] 
    

    return code, generated_ids_trimmed, tuple(logits)

def main():
    """Generate code using Qwen2.5-VL-3B-Instruct model with vLLM"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None, help="Number of data examples to generate code for")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--save_dir", type=str, default="", help="Path to save results")
    args = parser.parse_args()
    
    print(f"Model: {args.model_name}")
    print(f"Num examples: {args.num_examples}")
    print(f"Num samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Save dir: {args.save_dir}")

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
    save_path = os.path.join(args.save_dir, save_path)
    print(f"Results will be saved to {save_path}")

    os.makedirs("data/images", exist_ok=True)

    results_save_path = os.path.join(save_path, "generated_code.jsonl")
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    # Generate code for each sample
    objects_to_save = {"code": [], "ids": [], "logits": []}
    idx = 0
    for item in tqdm(dataset, desc="Generating code"):

        if "matplotlib" not in item['url']:
            continue

        image = item['image']
        instruction = item['instruction']

        # save image to data path
        image_path = os.path.join("data", "images", f"{idx}.png")
        image.save(image_path)

        # Generate code using vLLM
        generated_code, gen_ids, logits = generate_code_for_image(model, processor, image_path,
                                                                  instruction, args.num_samples,
                                                                  args.temperature)
        objects_to_save["code"].append(generated_code)
        objects_to_save["ids"].append(gen_ids)
        objects_to_save["logits"].append(logits)
        
        for i, x in enumerate(generated_code):
            generated_image_path = os.path.join(save_path, f"{idx}-{i}.png")
            # Create result item
            result = {
                'idx': f"{idx}-{i}",
                'ground_truth_path': image_path,
                'code': x,
                'ground_truth_code': item['code'],
                'generated_image_path': generated_image_path
            }

            # Save incrementally
            with open(results_save_path, 'a') as f: 
                f.write(json.dumps(result) + '\n')
                f.flush()  # Ensure it's written immediately
        
        # Force garbage collection and clear CUDA cache after each image
        del generated_code, gen_ids, logits
        gc.collect()
        torch.cuda.empty_cache()

        idx += 1

    
    model_name = model_name.split("/")[-1]
    save_path = os.path.join(args.save_dir, "generated_results", model_name, "outputs")
    os.makedirs(save_path, exist_ok=True)
    objects_save_path = os.path.join(save_path, "objects.pkl")
    
    print(f"Results saved to {args.save_dir}")
    with open(os.path.join(objects_save_path), "wb") as f: pickle.dump(objects_to_save, f)

if __name__ == "__main__":
    main()
