import os
import matplotlib.pyplot as plt
import matplotlib
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import Levenshtein
from Plot2Code.plot2code.utils import get_parser, get_save_path, get_eval_path, read_jsonl_file
from matplotlib.pyplot import *

parser = get_parser()
args = parser.parse_args()

# Read the JSONL file
generated_code_file =  get_save_path(args)
content_list = read_jsonl_file(generated_code_file)


def position_similarity(pos1, pos2, size_ratio):
    pos2_adjusted = pos2 * size_ratio

    position_difference = pos1 - pos2_adjusted

    # calculate the absolute distance
    absolute_distance = np.sqrt(np.sum(position_difference ** 2))

    # convert the absolute distance to similarity
    distance_similarity = np.exp(-absolute_distance / 100)

    return distance_similarity


def extract_texts(component):
    texts = []
    positions = []
    if isinstance(component, matplotlib.text.Text) and component.get_visible():
        text = component.get_text().strip().lower()
        position = component.get_position()
        if text:  # only extract non-empty text
            texts.append(text)
            positions.append(np.array(position))

    for child in component.get_children():
        child_texts, child_positions = extract_texts(child)
        texts.extend(child_texts)
        positions.extend(child_positions)

    return texts, positions

def match_texts(texts1, texts2, positions1, positions2, size_ratio):
    matched = 0
    unmatched1 = len(texts1)
    unmatched2 = len(texts2)

    for text1, pos1 in zip(texts1, positions1):
        min_distance = float('inf')
        best_match_index = None
        for i, (text2, pos2) in enumerate(zip(texts2, positions2)):
            try:
                distance = Levenshtein.distance(text1, text2)
                position_sim = position_similarity(pos1, pos2, size_ratio)

                if distance < min_distance and position_sim > 0.8:
                    min_distance = distance
                    best_match_index = i
            except:
                pass

        if min_distance <= 0:  # the maximux edit distance allowed
            matched += 1
            texts2.pop(best_match_index)
            positions2.pop(best_match_index)
            unmatched2 -= 1
            unmatched1 -= 1

    total_pairs = matched + unmatched1 + unmatched2
    if total_pairs == 0:
        return 1
    match_score = matched / total_pairs
    return match_score

eval_dir = get_eval_path(args)
text_match_score_file = os.path.join(eval_dir, args.text_match_score_results)

# Store all results to calculate statistics
all_results = []

# Open the JSONL file in write mode
with open(text_match_score_file, "w") as jsonl_file:
    for item in tqdm(content_list, desc="Evaluating text image similarity"):
        ground_truth_path = item['ground_truth_path']
        test_image_path = item['generated_image_path']
        code = item['code']
        ground_truth_code = item['ground_truth_code']
        # Execute the code and save the images

        img = Image.open(test_image_path)
        img_np = np.array(img)

        # Check if test_image is all white
        if np.all(img_np == 255):
            result = {'ground_truth_path': ground_truth_path, 'test_image_path': test_image_path, 'text_match_score': 0.0}
            all_results.append(result)
            jsonl_file.write(json.dumps(result) + "\n")
            jsonl_file.flush()
            print(f"Skipping all white image: {test_image_path}")
            continue

        exec(ground_truth_code)
        fig1 = plt.gcf()
        plt.close()
        matplotlib.rcdefaults()
        plt.cla()
        plt.clf()
        plt.close("all")

        try:
            exec(code)
        except Exception as e:
            result = {'ground_truth_path': ground_truth_path, 'test_image_path': test_image_path, 'text_match_score': 0.0}
            all_results.append(result)
            jsonl_file.write(json.dumps(result) + "\n")
            jsonl_file.flush()
            print(f"Error executing code: {e}")
            continue

        fig2 = plt.gcf()
        plt.close()
        matplotlib.rcdefaults()
        plt.cla()
        plt.clf()
        plt.close("all")

        # Extract texts and positions from the figures
        texts1, positions1 = extract_texts(fig1)
        texts2, positions2 = extract_texts(fig2)
        # Calculate the size ratio
        fig1_size = np.array(fig1.get_size_inches())
        fig2_size = np.array(fig2.get_size_inches())
        size_ratio = fig1_size / fig2_size

        # Calculate the match score
        match_score = match_texts(texts1, texts2, positions1, positions2, size_ratio)

        # Append the generated image path, ground truth code and match score to the JSONL file
        result = {'ground_truth_path': ground_truth_path, 'test_image_path': test_image_path, 'text_match_score': match_score}
        all_results.append(result)
        jsonl_file.write(json.dumps(result) + "\n")
        jsonl_file.flush()

# Calculate overall statistics
if all_results:
    scores = [result['text_match_score'] for result in all_results]
    overall_score = np.mean(scores)
    standard_error = np.std(scores) / np.sqrt(len(scores))

    print(f"Overall text match score: {overall_score:.4f} ± {standard_error:.4f}")
    print(f"Number of evaluated samples: {len(all_results)}")

    # Write statistics at the top of the file
    stats = {
        'overall_score': overall_score,
        'standard_error': standard_error,
        'num_samples': len(all_results)
    }

    # Read the current file content
    with open(text_match_score_file, 'r') as f:
        lines = f.readlines()

    # Write statistics at the top, then the original content
    with open(text_match_score_file, 'w') as f:
        f.write(json.dumps(stats) + '\n')
        f.writelines(lines)
else:
    print("No results to evaluate")
