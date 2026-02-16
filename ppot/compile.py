import regex
import torch, transformers, numpy as np
import ppot.program, ppot.utils

def get_token_pos(token_ids: torch.LongTensor, processor: transformers.AutoProcessor,
                  rule: str = r"(?<!(?:[a-df-zA-DF-Z_][0-9]*)|(?:[eE][eE]+[0-9]*)|(?:#.*))([0-9])") -> list:
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
    tok = processor if ppot.utils.is_tokenizer(processor) else processor.tokenizer
    # Tokens as strings (here we don't ignore special tokens, which might matter in the future).
    S = [tok.batch_decode(x) for x in token_ids]
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


def programs(token_ids: torch.LongTensor, logits: torch.FloatTensor,
             processor: transformers.AutoProcessor, code: str, rules: list = None, supp: list = None, only_one: bool = False,
             **kwargs) -> tuple:
    """
    Get probabilistic programs from the generated programs.

    Inputs:
        token_ids: torch.LongTensor of shape (batch_size, sequence_length)
        logits: torch.FloatTensor of shape (batch_size, sequence_length, vocab_size)
        processor: the model's transformers.AutoProcessor
    Optional inputs:
        supp: a list of torch.LongTensor containing the support (as token ids) of each random
            variable. If not given, assume digits; if supp is a one dimensional torch.LongTensor,
            then assume all variables have same support.
        rules: a list of regex rules to match on
        supp: a list of torch.LongTensor indicating support (as token ids) of corresponding to rules being matched.
        only_one: whether to limit to only one random variable. If so, returns a list of
            program.USPP instead of a list of program.Program.
    Returns:
        A list of probabilistic programs of type program.Program
        The normalized loglikelihood of each program
    """
    # Get token positions.
    if rules is None:
        pos = get_token_pos(token_ids, processor, **kwargs)
    else:
        pos = []
        for r in rules:
            pos.append(get_token_pos(token_ids, processor, rule=r, **kwargs))
        pos = [list(row) for row in zip(*pos)]
    tok = processor if ppot.utils.is_tokenizer(processor) else processor.tokenizer

    if supp is None: # if supp is None then make digits the support
        S = tok([str(i) for i in range(10)], return_tensors="pt").input_ids.flatten()
        supp = [[S for _ in pos[i]] for i in range(token_ids.shape[0])] # Make support for all positions collected.
    elif torch.is_tensor(supp): # if supp is tensor then make it the support of all digits
        supp = [[supp for _ in pos[i]] for i in range(token_ids.shape[0])]
    else: # if supp is a list then it matches with the rules list
        joint_supp = []
        joint_positions = []
        for rule_pos in pos:
            supp_extended = []
            pos_extended = []
            for index, positions in enumerate(rule_pos):
                current_supp = [supp[index] for _ in positions]
                supp_extended.extend(current_supp)
                pos_extended.extend(positions)
            joint_supp.append(supp_extended)
            joint_positions.append(pos_extended)

    PP = []
    pos, supp = joint_positions, joint_supp

    # Compute loglikelihoods.
    M = torch.isin(token_ids, torch.tensor(tok.all_special_ids)) # special tokens
    L = torch.log_softmax(logits, dim=-1) # logits from scores
    LL = torch.sum(L.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).masked_fill_(M, 0.0), dim=-1)
    nLL = LL/torch.sum(torch.bitwise_not(M), dim=-1) # normalized loglikelihood

    for i, (T, P) in enumerate(zip(token_ids, pos)):
        # Prepare code as a formatted string.
        tokens = tok.batch_decode(T, skip_special_tokens=True)
        real_tokens = []
        for j, t in enumerate(tokens):
            tokens[j] = t.replace("{", "{{").replace("}", "}}")
        for j, p in enumerate(P):
            real_tokens.append(tokens[p])
            tokens[p] = f"{{{j}}}" # turn it into an RV
        C = ppot.utils.remove_code_affixes(''.join(tokens))
        # Prepare random variable names as a list.
        X = list(range(len(P)))
        # Prepare logits as a list of tensors.
        L_supp = [torch.log_softmax(L[i,p,x], dim=-1) for p, x in zip(P, supp[i])]
        if only_one:
            # Default values for RVs.
            V_default = tok.batch_decode([x.item() for x in token_ids[i,P]])
            PP.append(ppot.program.USPP(V_default, C, X, L_supp, supp[i], tok, code[i], real_tokens))
        else: PP.append(ppot.program.Program(C, X, L_supp, supp[i], tok, code[i], real_tokens))

    return PP, nLL

