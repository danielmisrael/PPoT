import regex
import torch, transformers, numpy as np
import program, utils

def extract_code(response_str):
    """Extract code from response string"""
    matches = regex.findall(r'```python(.*?)```', response_str, regex.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches)
    else:
        return response_str


# TODO we should agree on the formatting of the probabilities dictionary.

def compile_probcode(code, probabilities):
    '''
    Inputs:
        code: str
        probabilities: dict[str, float] Maps strings contained within code (constants or variables) to their probabilities
    Returns:
        probabilistic_code: str Still a string, but includes source of randomness and incorporates probabilities.

    '''
    # TODO @Poorva
    pass

def get_token_pos(token_ids: torch.LongTensor, processor: transformers.AutoProcessor,
                  rule: str = r'(?<![a-zA-Z_][0-9]*)([0-9])') -> list:
    """
    Get the position of the random variables from the generated programs.

    Inputs:
        token_ids: torch.LongTensor of shape (batch_size, sequence_length)
        processor: the model's transformers.AutoProcessor
    Optional inputs:
        rule: regex rule to identify random variables
    Returns:
        A list of lists containing the position of all random variable tokens

    To verify the algorithm's correctness, you can run the following:

    > toks = [[X[u] for u in j] for X, j in zip(S, J)]
    > ground_truth = [r.findall(''.join(x)) for x in S]
    > assert toks == ground_truth
    > selected_ids = [I[i,j] for i, j in enumerate(J)]
    > assert tokenizer.batch_decode(selected_ids) == [''.join(x) for x in ground_truth]
    """
    # Tokens as strings (here we don't ignore special tokens, which might matter in the future).
    S = [processor.tokenizer.batch_decode(x) for x in token_ids]
    # Length of tokens.
    L = np.array([list(map(len, x)) for x in S])
    # Cumulative sums of lengths, which give the (end) position of the token.
    cL = np.cumsum(L, axis=-1)
    # Compile the regex according to rule. The default rule captures single digits that are not
    # preceded by an alphabetic character or underline. It requires variable width look-behind,
    # which is not supported by the standard re library; instead, we use regex.
    r = regex.compile(rule)
    # Get the (end) position of all regex matches.
    M = [np.array([y.end() for y in r.finditer(''.join(x))]) for x in S]
    # Bisect on cL to find their tokenization position in logarithmic time.
    J = [np.searchsorted(x, y) for x, y in zip(cL, M)]
    return J


def get_probs(token_ids: torch.LongTensor, pos: list, logits: torch.FloatTensor,
              processor: transformers.AutoProcessor, supp: list = None, only_one: bool = False) -> tuple:
    """
    Get probabilistic programs from the generated programs.

    Inputs:
        token_ids: torch.LongTensor of shape (batch_size, sequence_length)
        pos: list of np.ndarray containing the position of all random variable tokens
        logits: torch.FloatTensor of shape (batch_size, sequence_length, vocab_size)
        processor: the model's transformers.AutoProcessor
    Optional inputs:
        supp: a list of torch.LongTensor containing the support (as token ids) of each random
            variable. If not given, assume digits; if supp is a one dimensional torch.LongTensor,
            then assume all variables have same support.
        only_one: whether to limit to only one random variable. If so, returns a list of
            program.USPP instead of a list of program.Program.
    Returns:
        A list of probabilistic programs of type program.Program
        The normalized loglikelihood of each program
    """
    # Support preprocessing.
    if supp is None:
        S = processor.tokenizer([str(i) for i in range(10)], return_tensors="pt").input_ids.flatten()
        supp = [[S for _ in pos[i]] for i in range(token_ids.shape[0])]
    elif torch.is_tensor(supp): supp = [[supp for _ in pos[i]] for i in range(token_ids.shape[0])]

    PP = []

    # Compute loglikelihoods.
    M = torch.isin(token_ids, torch.tensor(processor.tokenizer.all_special_ids)) # special tokens
    L = torch.log_softmax(logits, dim=-1) # logits from scores
    LL = torch.sum(L.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).masked_fill_(M, 0.0), dim=-1)
    nLL = LL/torch.sum(torch.bitwise_not(M), dim=-1) # normalized loglikelihood

    for i, (T, P) in enumerate(zip(token_ids, pos)):
        # Prepare code as a formatted string.
        tokens = processor.batch_decode(T, skip_special_tokens=True)
        for j, p in enumerate(P): tokens[p] = f"{{{j}}}" # turn it into an RV
        C = utils.remove_code_affixes(''.join(tokens))
        # Prepare random variable names as a list.
        X = list(range(len(P)))
        # Prepare logits as a list of tensors.
        L_supp = [torch.log_softmax(L[i,p,x], dim=-1) for p, x in zip(P, supp[i])]
        if only_one:
            # Default values for RVs.
            V_default = [x.item() for x in token_ids[i,P]]
            PP.append(program.USPP(V_default, C, X, L_supp, supp[i], processor.tokenizer))
        else: PP.append(program.Program(C, X, L_supp, supp[i], processor.tokenizer))

    return PP, nLL

# Sketch of what the generation loop will look like
def generate_probcode(model, tokenizer, input_ids, **gen_kwargs):
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            **gen_kwargs,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    code = extract_code(output_text[0])

    code_ids = tokenizer.encode(code)

    with torch.no_grad():
        logits = model(generated_ids).logits

    code_logits = logits[:, -len(code_ids):, :]

    probabilities = get_probs(code_ids, code_logits)

    probabilistic_code = compile_probcode(code, probabilities)

    return probabilistic_code

