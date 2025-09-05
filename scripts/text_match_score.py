import os
import matplotlib.pyplot as plt
import matplotlib
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import Levenshtein
from matplotlib.pyplot import *
from ppot.utils import safe_execute_plot
import io, uuid, dill

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

def evaluate_single_example(generated_code:str, ground_truth_code: str, separate_process: bool = False) -> float:
    """
    Evaluates a single program for a single image
    """
    try:
        exec(generated_code.lstrip("```python"))
        fig2 = plt.gcf()
    except:
        pass

    fig2 = plt.gcf()

    plt.close()
    matplotlib.rcdefaults()
    plt.cla()
    plt.clf()
    plt.close("all")
    img_np = np.array(fig2)
    all_white = np.all(img_np == 255)
    if all_white:
        return 0.0
    
    try:
        exec(ground_truth_code)
        fig1 = plt.gcf()
    except:
        pass
    fig1 = plt.gcf()
    plt.close()
    matplotlib.rcdefaults()
    plt.cla()
    plt.clf()
    plt.close("all")
    img_np = np.array(fig1)
    
    texts1, positions1 = extract_texts(fig1)
    texts2, positions2 = extract_texts(fig2)
    # Calculate the size ratio
    fig1_size = np.array(fig1.get_size_inches())
    fig2_size = np.array(fig2.get_size_inches())
    size_ratio = fig1_size / fig2_size

    # Calculate the match score
    match_score = match_texts(texts1, texts2, positions1, positions2, size_ratio)
    return match_score
