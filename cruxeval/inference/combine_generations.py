# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os

# Use the correct base directory
base_dir = "/space/poorvagarg/genPPS/cruxeval"
dirs = [d for d in next(os.walk(os.path.join(base_dir, 'model_generations_raw')))[1] if ("input" in d or "output" in d)]

for dir in dirs:
    new_dir = os.path.join(base_dir, "model_generations", dir)
    dir = os.path.join(base_dir, "model_generations_raw", dir)
    files = os.listdir(dir)

    for mode in ["orig", "raw"]:
        if mode == "orig":
            combined_json = {}
            current_keys = set()
            count = 0
            for input_json in files:
                if input_json == "generations.json" or "raw" in input_json:
                    continue
                
                count += 1
                with open(os.path.join(dir, input_json), "r") as fp:
                    input_json = json.load(fp)
                    input_json = {f"sample_{k}": v for k, v in input_json.items()}
                    keys = set(input_json.keys())
                    if keys.intersection(current_keys):
                        raise ValueError("Keys overlap")
                    combined_json.update(input_json)

            ## sort on keys and remove keys
            print(dir, f"{count} files", len(combined_json))
            # assert len(combined_json) == 800  # Commented out for custom runs with fewer samples

            try: os.makedirs(new_dir)
            except: pass

            output_json = "generations.json"
            print(os.path.join(new_dir, output_json))
            with open(os.path.join(new_dir, output_json), "w") as fp:
                json.dump(combined_json, indent=4, fp=fp)
        else:
            combined_json = {}
            current_keys = set()
            count = 0
            for input_json in files:
                if input_json == "generations_raw.json" or "raw" not in input_json:
                    continue
                
                count += 1
                with open(os.path.join(dir, input_json), "r") as fp:
                    input_json = json.load(fp)
                    input_json = {f"sample_{k}": v for k, v in input_json.items()}
                    keys = set(input_json.keys())
                    if keys.intersection(current_keys):
                        raise ValueError("Keys overlap")
                    combined_json.update(input_json)
            print(dir, f"{count} files", len(combined_json))
            # assert len(combined_json) == 800  # Commented out for custom runs with fewer samples

            output_json = "generations_raw.json"
            print(os.path.join(dir, output_json))
            with open(os.path.join(dir, output_json), "w") as fp:
                json.dump(combined_json, indent=4, fp=fp)