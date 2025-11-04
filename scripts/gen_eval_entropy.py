import argparse, os, json, functools
import transformers, datasets, torch, tqdm
import ppot.utils, scripts.eval_entropy_programs

DEFAULT_SUFFIX = " Only write the code for the above question, without any prior explanation."
PROMPTS = {
    "diagnostics": "{question}" + DEFAULT_SUFFIX,
    "pot": "Generate a Python function `compute_answer` with no arguments that computes the "
        "needed computations and returns the answer to the problem below. Only generate the "
        "Python function `compute_answer`, with no explanations.\n\n{question}",
}
def template(tokenizer: transformers.AutoTokenizer, Q: str, which_prompt: str = "diagnostics") -> dict:
    if "Qwen2.5-Coder" in tokenizer.name_or_path:
        X = tokenizer.apply_chat_template([{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                           {"role": "user", "content": PROMPTS[which_prompt].format(question=Q)}],
                                          tokenize=False, add_generation_prompt=True)
    else: raise NotImplementedError
    return tokenizer([X], return_tensors="pt")

def sample(model: transformers.AutoModel, tokenizer: transformers.AutoTokenizer, Q: str,
           num_samples: int, which_prompt: str, **kwargs) -> (torch.LongTensor, torch.FloatTensor):
    X = template(tokenizer, Q, which_prompt=which_prompt)
    out = model.generate(**X.to(model.device), max_new_tokens=1024, return_dict_in_generate=True,
                         output_logits=True, repetition_penalty=1.0,
                         num_return_sequences=num_samples, do_sample=True, top_p=1.0, **kwargs)
    k = X.input_ids.numel()
    I = out.sequences[:,k:].cpu()
    L = torch.concatenate(tuple(x.cpu() for x in out.logits),
                          dim=-1).reshape(out.logits[0].shape[0], len(out.logits), -1)
    return I, L

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset-split", type=str, default=None)
    parser.add_argument("--num-examples", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prompt", type=str, choices=list(PROMPTS.keys()), default="diagnostics")
    args = parser.parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                              device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    _, ext = os.path.splitext(args.dataset)
    if ext == ".jsonl":
        with open(args.dataset, "r") as f: J = list(f)
        D = datasets.Dataset.from_list([json.loads(x) for x in J])
        if args.subset_split is not None:
            D = D.filter(lambda x: x["meta"]["library"] == args.subset_split)
        D = datasets.Dataset.from_dict(D[:args.num_examples])
    else:
        D = ppot.utils.prepare_data(args.dataset, args.num_examples)
        raise NotImplementedError

    for i, X in enumerate(tqdm.tqdm(D, desc="Example")):
        I, L = sample(model, tokenizer, X["question"], args.num_samples, args.prompt,
                      temperature=args.temperature)
        H = scripts.eval_entropy_programs.entropy(L)
        html = scripts.eval_entropy_programs.html(H, I, L, tokenizer, toc_len=args.num_examples,
                                                  instruction=X["question"],
                                                  ground_truth_text=X["execution_output"])
        with open(f"{args.save_path}/{i}.html", "w") as f: f.write(html)
