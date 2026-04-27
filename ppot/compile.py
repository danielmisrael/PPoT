import regex # type: ignore
import torch, transformers, numpy as np
from collections import namedtuple
import ppot.program, ppot.utils
from typing import Optional

NEWER_TRANSFORMERS = transformers.__version__ > "4.53.0"

CompactLogits = namedtuple('CompactLogits', ['token_log_prob', 'supp_logits', 'supp_ids'])
"""
Compact logit representation storing only what programs() needs.
  token_log_prob : (batch, seq_len)        — log P(actual_token_t) at each position
  supp_logits    : (batch, seq_len, |supp|) — raw logits at support token IDs
  supp_ids       : (|supp|,)               — the support token IDs (union of all rules)
Reduces CPU transfer from ~1.8GB to ~1MB per example.
"""

NEWER_TRANSFORMERS = transformers.__version__ > "4.53.0"

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
    S = [tok.batch_decode(x.reshape(-1, 1) if NEWER_TRANSFORMERS else x) for x in token_ids] # type: ignore
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
             processor: transformers.AutoProcessor, code: str, rules: Optional[list] = None, supp: Optional[list] = None, only_one: bool = False,
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
    tok = processor if ppot.utils.is_tokenizer(processor) else processor.tokenizer # type: ignore

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
        pos, supp = joint_positions, joint_supp

    PP = []

    # Compute loglikelihoods.
    M = torch.isin(token_ids, torch.tensor(tok.all_special_ids)) # special tokens # type: ignore
    is_compact = isinstance(logits, CompactLogits)
    if is_compact:
        supp_id_lookup = {int(v): k for k, v in enumerate(logits.supp_ids.tolist())}
        LL = torch.sum(logits.token_log_prob.masked_fill(M, 0.0), dim=-1)
    else:
        L = torch.log_softmax(logits, dim=-1) # logits from scores
        LL = torch.sum(L.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).masked_fill(M, 0.0), dim=-1)
    nLL = LL/torch.sum(torch.bitwise_not(M), dim=-1) # normalized loglikelihood

    for i, (T, P) in enumerate(zip(token_ids, pos)):
        # Prepare code as a formatted string.
        tokens = tok.batch_decode(T.reshape(-1, 1) if NEWER_TRANSFORMERS else T, skip_special_tokens=True)
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
        if is_compact:
            L_supp = [torch.log_softmax(
                          logits.supp_logits[i, p][[supp_id_lookup[int(t)] for t in x]], dim=-1)
                      for p, x in zip(P, supp[i])]
        else:
            L_supp = [torch.log_softmax(L[i,p,x], dim=-1) for p, x in zip(P, supp[i])]
        if only_one:
            # Default values for RVs.
            V_default = tok.batch_decode([x.item() for x in token_ids[i,P]])
            PP.append(ppot.program.USPP(V_default, C, X, L_supp, supp[i], tok, code[i], real_tokens))
        else: PP.append(ppot.program.Program(C, X, L_supp, supp[i], tok, code[i], real_tokens, P))

        PP[-1].probs(P, T.shape)

    return PP, nLL


def fast_subset_programs(token_ids: torch.LongTensor, compact_scores: torch.FloatTensor,
                         token_log_prob: torch.FloatTensor, unique_toks: torch.LongTensor,
                         processor: transformers.AutoProcessor, strings: list = None,
                         answer_extractor: callable = None) -> tuple:
    """Create FastSubsetProgram objects from compact logits.

    Args:
        token_ids: (batch, seq_len) generated token IDs
        compact_scores: (batch, seq_len, n_unique) logits at unique token cols
        token_log_prob: (batch, seq_len) log P(actual token) per position
        unique_toks: (n_unique,) the vocab IDs for each column
        processor: tokenizer
        strings: optional pre-decoded strings
        answer_extractor: callable(string) -> (start_char, end_char)
    Returns:
        List of FastSubsetProgram objects, normalized log-likelihoods tensor.
    """
    tok = processor if ppot.utils.is_tokenizer(processor) else processor.tokenizer

    M = torch.isin(token_ids, torch.tensor(tok.all_special_ids))
    LL = torch.sum(token_log_prob.masked_fill(M, 0.0), dim=-1)
    non_special = torch.sum(~M, dim=-1)
    nLL = LL / torch.clamp(non_special, min=1)

    PP = []
    for i in range(token_ids.shape[0]):
        ids = token_ids[i]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad_id is not None:
            non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
            end_pos = non_pad[-1].item() + 1 if len(non_pad) > 0 else 0
        else:
            end_pos = len(ids)

        ids_trimmed = ids[:end_pos]
        logits_trimmed = compact_scores[i, :end_pos, :]

        if answer_extractor is not None and end_pos > 0:
            start_tok, end_tok = _get_answer_token_span(
                ids_trimmed, tok, answer_extractor)
            answer_ids = ids_trimmed[start_tok:end_tok]
            answer_logits = logits_trimmed[start_tok:end_tok, :]
            answer_string = tok.decode(answer_ids)
        else:
            answer_ids = ids_trimmed
            answer_logits = logits_trimmed
            if strings is not None:
                answer_string = strings[i]
            else:
                answer_string = tok.decode(answer_ids)

        PP.append(ppot.program.FastSubsetProgram(
            answer_logits, answer_ids.tolist(), unique_toks, tok, answer_string))

    return PP, nLL


def _get_answer_token_span(token_ids_single: torch.LongTensor, tokenizer,
                           answer_extractor: callable) -> tuple:
    """Map character-level answer boundaries to token positions.

    Uses the same cumulative-length approach as get_token_pos():
    1. Decode each token individually to get per-token strings
    2. Compute cumulative character lengths
    3. Use searchsorted to find token positions for char boundaries

    Returns (start_token_idx, end_token_idx) as a half-open range.
    """
    tokens_as_strings = tokenizer.batch_decode(token_ids_single)
    lengths = np.array([len(s) for s in tokens_as_strings])
    cum_lengths = np.cumsum(lengths)
    full_string = ''.join(tokens_as_strings)

    start_char, end_char = answer_extractor(full_string)
    if start_char is None:
        return 0, len(token_ids_single)

    start_tok = int(np.searchsorted(cum_lengths, start_char, side='right'))
    end_tok = int(np.searchsorted(cum_lengths, end_char, side='left')) + 1
    end_tok = min(end_tok, len(token_ids_single))
    return start_tok, end_tok


def subset_programs(token_ids: torch.LongTensor, logits: torch.FloatTensor,
                    processor: transformers.AutoProcessor, strings: list = None,
                    answer_extractor: callable = None) -> tuple:
    """Create SubsetProgram objects from generated token sequences.

    Inputs:
        token_ids: torch.LongTensor of shape (batch_size, sequence_length)
            -- the generated token ids (prompt already stripped).
        logits: torch.FloatTensor of shape (batch_size, sequence_length, vocab_size)
            -- the logits for each generated token position.
        processor: the model's tokenizer or processor.
        strings: optional list of pre-decoded strings.
        answer_extractor: callable(decoded_string) -> (start_char, end_char)
            that returns character-level boundaries of the answer expression.
            If None, uses the entire sequence.
    Returns:
        A list of SubsetProgram objects (one per batch element).
        Normalized log-likelihoods tensor.
    """
    tok = processor if ppot.utils.is_tokenizer(processor) else processor.tokenizer

    # Compute normalized log-likelihoods (same as programs())
    M = torch.isin(token_ids, torch.tensor(tok.all_special_ids))
    L = torch.log_softmax(logits, dim=-1)
    LL = torch.sum(
        L.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1).masked_fill_(M, 0.0),
        dim=-1)
    non_special = torch.sum(torch.bitwise_not(M), dim=-1)
    nLL = LL / torch.clamp(non_special, min=1)

    PP = []
    for i in range(token_ids.shape[0]):
        # Strip padding/EOS tokens from the end
        ids = token_ids[i]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad_id is not None:
            non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
            if len(non_pad) > 0:
                end_pos = non_pad[-1].item() + 1
            else:
                end_pos = 0
        else:
            end_pos = len(ids)

        ids_trimmed = ids[:end_pos]
        logits_trimmed = logits[i, :end_pos, :]

        if answer_extractor is not None and end_pos > 0:
            start_tok, end_tok = _get_answer_token_span(
                ids_trimmed, tok, answer_extractor)
            answer_ids = ids_trimmed[start_tok:end_tok]
            answer_logits = logits_trimmed[start_tok:end_tok, :]
            answer_string = tok.decode(answer_ids)
        else:
            answer_ids = ids_trimmed
            answer_logits = logits_trimmed
            if strings is not None:
                answer_string = strings[i]
            else:
                answer_string = tok.decode(answer_ids)

        PP.append(ppot.program.SubsetProgram(
            answer_logits, answer_ids.tolist(), tok, answer_string))

    return PP, nLL
