#!/usr/bin/env python3
import os
import json
import sys
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
import re
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib
import shutil
import argparse

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_model_and_processor(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
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

def extract_code(response_str):
    """Extract code from response string"""
    matches = re.findall(r'```python(.*?)```', response_str, re.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches)
    else:
        return response_str

def read_jsonl_file(file_path):
    """Read JSONL file"""
    with open(file_path, 'r') as json_file:
        return [json.loads(line) for line in json_file]

def get_save_path(model_name):
    """Get save path for generated code"""
    model_name = model_name.split("/")[-1]
    save_path = os.path.join("generated_results", model_name, "direct", "instruct")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def generate_code_for_image(model, processor, image_path, instruction):
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
            max_new_tokens=2048,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    # Extract code
    code = extract_code(output_text[0])
    
    # If no code found, return the full response
    if not code:
        code = output_text[0]
    
    return code

def main():
    """Generate code using Qwen2.5-VL-3B-Instruct model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate code for")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    args = parser.parse_args()
    
    # Configuration
    model_name = args.model_name
    num_samples = args.num_samples
    
    # Load model and processor
    model, processor = load_model_and_processor(model_name)
    
    dataset = load_dataset("TencentARC/Plot2Code", split="test")
    dataset = dataset.filter(lambda x: "matplotlib" in x["url"])
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    
    
    # Get save path
    save_path = get_save_path(model_name)
    print(f"Results will be saved to {save_path}")
        
    os.makedirs("data/images", exist_ok=True)
    
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
        generated_code = generate_code_for_image(model, processor, image_path, instruction)
        generated_image_path = os.path.join(save_path, f"{idx}.png")
        
        try:
            exec(generated_code)
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
            'idx': idx,
            'ground_truth_path': image_path,
            'code': generated_code,
            'ground_truth_code': item['code'],
            'generated_image_path': generated_image_path
        }
        results.append(result)
        
        results_save_path = os.path.join(save_path, "generated_code.jsonl")
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        # Save incrementally
        with open(results_save_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    print(f"Generated code for {len(results)} samples")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main() 