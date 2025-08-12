import transformers, pickle, ppot.compile
import argparse
import random
from scripts.text_match_score import evaluate_single_example
import datasets
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

def prepare_data(dataset_name: str, num_examples = None, filter_fn = None, **kwargs) -> datasets.Dataset:
    data = datasets.load_dataset(dataset_name, **kwargs)
    if filter_fn is not None: data = data.filter(filter_fn)
    if num_examples is not None: data = data.select(range(num_examples))
    return data


def extract_probabilistic_programs(temperature:float, num_eg:int) -> tuple:
    # Loading the processor
    processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Load the programs generated from the model: in vedett [0]
    # t__ is the temperature - 8 samples
    # t0 only one sample, because always greedy
    # there are multiple .pkl files
    
    PP, LL = [], []
    for i in tqdm(range(num_eg)):
        with open(f"/space/renatolg/genPPS/out/Qwen2.5-VL-3B-Instruct_t{temperature:.1f}/data/{i}.pkl", "rb") as f: 
            R = pickle.load(f)
        # Load the token ids and logits
        input_ids, logits = R["ids"], R["logits"]
        # Generate the probabilistic program
        tempPP, tempLL = ppot.compile.programs(input_ids, logits, processor)
        PP.append(tempPP)
        LL.append(tempLL)
    
    return PP, LL


def sample_from_probabilistic_programs(PP:list, LL:list, num_samples:int, greedy:bool) -> list:
    """
    Sample from probabilistic programs and evaluate them with respect to the actual image.
    It returns statistics of the results
    """
    dataset = prepare_data("TencentARC/Plot2Code", num_examples=len(PP),
                           filter_fn = lambda x: "matplotlib" in x["url"], split="test")
    
    # Evaluating probabilistic programs
    text_match_scores = []
    for i in tqdm(range(len(PP))):
        currPP = random.choice(PP[i])
        programs = currPP.greedy() if greedy else currPP.sample(num_samples)
        
        if isinstance(programs, str):
            programs = [programs]

        score_single = []
        for prog in tqdm(programs):
            score = evaluate_single_example(prog, dataset[i]['code'])
            score_single.append(score)
        text_match_scores.append(score_single)

    print(text_match_scores)
    
    # Computing statistics
    np_scores = np.array(text_match_scores)
    score_min = np.mean(np.min(np_scores, axis=1))
    score_max = np.mean(np.max(np_scores, axis=1))
    score_mean = np.mean(np_scores)
    score_median = np.mean(np.median(np_scores, axis=1))

    return score_min, score_max, score_mean, score_median


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--isUSPP", action="store_true", default=False)
    parser.add_argument("--greedy", action="store_true", default=False)

    args = parser.parse_args()

    PP, LL = extract_probabilistic_programs(args.temperature, args.num_examples)
    min, max, mean, median = sample_from_probabilistic_programs(PP, LL, args.num_samples, args.greedy)

    table = PrettyTable()
    table.title = "Text Match Score Statistics"
    table.field_names = ["Minimum", "Maximum", "Mean", "Median"]
    table.add_row([min, max, mean, median])
    print(table)