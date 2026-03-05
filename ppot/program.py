import random, math
import torch, torch.distributions.gumbel, transformers
import ppot.utils
from copy import deepcopy

class Program:
    "A probabilistic program."

    GUMBEL = torch.distributions.gumbel.Gumbel(0, 1)

    def __init__(self, C: str, X: list, P: list, V: list, tokenizer: transformers.AutoTokenizer,
                 raw_program: str, actual_values: list, pos: list, device: str = None):
        """Constructs a probabilistic program.

        Arguments:
            C: is a formatted string containing variables to be replaced, e.g. "a = {x}", where x
                is the random variable id as an integer.
            X: list of random variable names that map to both P (by name) and pr (by index).
            P: list of torch.FloatTensor encoding the (log-)probability distribution of the i-th RV in X.
            V: list of torch.LongTensor with the ids in the support of each random variable.
            tokenizer: the model's transformers.AutoTokenizer.
        """
        assert len(X) == len(P), "Number of variables must match number of sets of values."
        assert len(X) == len(V), "Number of variables must match number of sets of values."
        self.deterministic = len(X) == 0
        self.code = C
        if torch.is_tensor(P): self.homogenous, self.P_tensor = True, P
        elif (not self.deterministic) and all(x.shape == P[0].shape for x in P): self.homogenous, self.P_tensor = True, torch.vstack(P)
        else: self.homogenous, self.P_tensor = False, None
        if not self.homogenous: self.mapping = {x: p for x, p in zip(X, P)}
        self.raw_program = raw_program
        self.supp = [tokenizer.batch_decode(v) for v in V]
        self.gumbel = Program.GUMBEL if device is None else ppot.utils.gumbel_on(device)
        self.device = "cpu" if device is None else device
        self.actual_values = actual_values
        self.tok_positions = pos

    def to(self, device: str):
        if device == self.device: return self
        if self.homogenous:
            self.P_tensor = self.P_tensor.to(device)
        else:
            for k in self.mapping: self.mapping[k] = self.mapping[k].to(device)
        self.gumbel = ppot.utils.gumbel_on(device)
        return self

    def sample_program(self, t: float = 1.0) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        if self.deterministic: return self.code
        if math.isclose(t, 0.0): return self.greedy()
        # Sample values.
        if self.homogenous:
            S = torch.argmax(torch.log_softmax(self.P_tensor/t, dim=-1)+self.gumbel.sample(self.P_tensor.shape), dim=-1).cpu()
        else:
            S = (torch.argmax(torch.log_softmax(p/t, dim=-1)+self.gumbel.sample(p.shape)) for p in self.mapping.values()).cpu()
        V = [self.supp[i][x.item()] for i, x in enumerate(S)]
        # Output code.
        return self.code.format(*V)
    
    def sample_program_constraint(self, t:float=1.0) -> str:
        "Returns a deterministic program sampled from this probabilistic program conditioned on not being the same program"
        # breakpoint()
        if self.homogenous:
            mapping = {i: torch.log_softmax(self.P_tensor[i, :]/t, dim=-1) for i in range(self.P_tensor.shape[0])}
        else:
            mapping = {}
            for k in self.mapping:
                mapping[k] = torch.log_softmax(self.mapping[k]/t, dim=-1) - 1e-09
        logits_actual_value = []
        for k in sorted(mapping):
            v = mapping[k]
            try:
                key_in_logits = self.supp[k].index(self.actual_values[k])
            except ValueError:
                key_in_logits = 0
            logits_actual_value.append(v[key_in_logits])
        logits_actual_value = torch.tensor(logits_actual_value)
        log_cum_prod = torch.sum(logits_actual_value)

        # in this sample list, False indicates change the variable, True indicates keep the variable same
        sample = []
        independent = False
        for i in range(logits_actual_value.shape[0]):
            if independent:
                log_odds = logits_actual_value[i] - torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(logits_actual_value[i])))
                sample.append(torch.distributions.bernoulli.Bernoulli(logits=log_odds).sample())
            else:
                Z = torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(log_cum_prod)))
                log_cum_prod -= logits_actual_value[i]
                logits = logits_actual_value[i]+torch.log(torch.max(torch.tensor(0.0), -torch.expm1(log_cum_prod)))-Z
                log_odds = logits - torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(logits)))
                sample.append(torch.distributions.bernoulli.Bernoulli(logits=log_odds).sample())
                independent = not sample[-1]

        if len(sample) != 0:
            assert torch.any(torch.tensor(sample).to(torch.bool) != torch.tensor([True for i in range(len(sample))]))

        final_sample = []
        for i, change in enumerate(sample):
            if not change:
                logits = deepcopy(mapping[i])
                try:
                    key_in_logits = self.supp[i].index(self.actual_values[i])
                except:
                    key_in_logits = 0
                # assert key_in_logits == int(self.actual_values[i])
                logits[key_in_logits] = -torch.inf
                final_sample.append(torch.argmax(torch.log_softmax(logits, dim=-1)+self.gumbel.sample()))
            else:
                try:
                    key_in_logits = self.supp[i].index(self.actual_values[i])
                except:
                    key_in_logits = 0
                final_sample.append(key_in_logits)
        V = [self.supp[i][x] for i, x in enumerate(final_sample)]
        final_result = self.code.format(*V)
        if len(sample) != 0:
            assert self.raw_program.replace("```python", "").replace("```", "") != final_result
        return final_result

    def sample(self, n: int = 1, as_list: bool = False, constraint = False, **kwargs) -> list:
        "Returns n deterministic programs sampled from this probabilistic program."
        sample_func = self.sample_program_constraint if constraint else self.sample_program
        return sample_func(**kwargs) if (n == 1) and (not as_list) else [sample_func(**kwargs) for _ in range(n)]

    def greedy(self, as_list: bool = False) -> str:
        "Returns the deterministic program output from the model"
        if self.deterministic: return [self.code] if as_list else self.code
        S = torch.argmax(self.P_tensor, dim=-1) if self.homogenous else \
            (torch.argmax(p).item() for p in self.mapping.values())
        V = [self.supp[i][x.item()] for i, x in enumerate(S)]
        r = self.code.format(*V)
        return [r] if as_list else r
    
    def probs(self, pos: list, output_len: int):
        t: float = 1.0
        if self.homogenous:
            mapping = {i: torch.log_softmax(self.P_tensor[i, :]/t, dim=-1) for i in range(self.P_tensor.shape[0])}
        else:
            mapping = {}
            for k in self.mapping:
                mapping[k] = torch.log_softmax(self.mapping[k]/t, dim=-1) - 1e-09
        logits_actual_value = []
        for k in sorted(mapping):
            v = mapping[k]
            try:
                key_in_logits = self.supp[k].index(self.actual_values[k])
            except ValueError:
                key_in_logits = 0
            logits_actual_value.append(v[key_in_logits])
        logits_actual_value = torch.tensor(logits_actual_value)
        log_cum_prod = torch.sum(logits_actual_value)
        prob_change = torch.expm1(logits_actual_value)/torch.expm1(log_cum_prod)
        # breakpoint()
        # prob_change = prob_changetorch.logsumexp(prob_change, dim=-1)

        entropy_ph = torch.zeros(output_len)
        for idx, position in enumerate(pos):
            entropy_ph[position] = prob_change[idx]
        self.entropy_ph = entropy_ph


class USPP(Program):
    "Union of Singleton Probabilistic Programs."

    def __init__(self, V_default: list, C: str, X: list, P: list, V: list,
                 tokenizer: transformers.AutoTokenizer, raw_program: str, **kwargs):
        super().__init__(C, X, P, V, tokenizer, raw_program, **kwargs)
        self.V_default = V_default
        self.raw_program = raw_program

    def greedy(self, as_list: bool = False) -> str:
        "Returns the deterministic program output from the model"
        if self.deterministic: return [self.code] if as_list else self.code
        X = random.randint(0, len(self.supp)-1)
        x = torch.argmax(self.P_tensor[X], dim=-1) if self.homogenous else torch.argmax(self.mapping[X]).item()
        V = self.V_default.copy()
        V[X] = self.supp[X][x]
        r = self.code.format(*V)
        return [r] if as_list else r

    def sample_program(self, t: float = 1.0) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        if self.deterministic: return self.code
        if t == 0.0: return self.greedy()
        # Assume a uniform prior.
        X = random.randint(0, len(self.supp)-1)
        # Sample only X.
        pr = self.P_tensor[X] if self.homogenous else self.mapping[X]
        x = torch.argmax(torch.log_softmax(pr/t, dim=-1) + self.gumbel.sample(pr.shape))
        # Fix other values and insert x.
        V = self.V_default.copy()
        V[X] = self.supp[X][x]
        # Output code.
        return self.code.format(*V)


class SubsetProgram:
    """A probabilistic program based on subset resampling of a token sequence.

    Unlike Program where independent RVs are identified at specific positions,
    SubsetProgram operates on a contiguous span of tokens where the resampling
    algorithm applies suffix masking to create sequential dependencies.
    """

    def __init__(self, logits: torch.FloatTensor, target_tokens: list,
                 tokenizer: transformers.AutoTokenizer, raw_string: str,
                 device: str = None):
        """
        Arguments:
            logits: FloatTensor of shape (seq_len, vocab_size) -- logits for the
                    answer portion of the generation.
            target_tokens: list of int -- the original token ids for the answer portion.
            tokenizer: the model's tokenizer (used for decoding resampled tokens).
            raw_string: the original decoded answer string.
            device: device string for tensor operations.
        """
        self.logits = logits
        self.target_tokens = target_tokens
        self.tokenizer = tokenizer
        self.raw_program = raw_string
        self.raw_string = raw_string
        self.supp = []
        self.device = device or "cpu"

    def to(self, device: str):
        if device == self.device: return self
        self.logits = self.logits.to(device)
        self.device = device
        return self

    def resample_subset(self, temperature: float = 1.0,
                        atleastone_constraint: bool = False,
                        eps: float = 1e-21) -> list:
        """Resample the token sequence using suffix-masked sequential sampling.

        Ported from LogitsResampler.resample_subset() in
        cruxeval/inference/resample_generations.py.

        At each position i, only tokens appearing in target_tokens[i:] are allowed.
        Sampling proceeds sequentially, and tokens between the current position and
        where the sampled token is found get removed, so the result can be shorter.
        """
        # For 3-token sequences [a, b, c] starting at i=1, the only sequence
        # different from the original is [a, c] — handle it directly.
        if atleastone_constraint and len(self.target_tokens) == 3:
            return [self.target_tokens[0], self.target_tokens[2]]
        # Sequences shorter than 3 tokens can't be made different; return as-is.
        if len(self.target_tokens) < 3:
            return self.target_tokens.copy()

        logits = self.logits
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        logits = logits.to(self.device)

        new_tokens = self.target_tokens.copy()
        seq_len = len(self.target_tokens)
        vocab_size = logits.shape[-1]

        probs = torch.softmax(logits / temperature, dim=-1)

        # Suffix masks: at position i, only tokens from target_tokens[i:] are allowed
        target_tokens_tensor = torch.tensor(self.target_tokens, dtype=torch.long,
                                            device=self.device)
        pos_indices = torch.arange(seq_len, device=self.device)
        position_mask = pos_indices.unsqueeze(1) <= pos_indices.unsqueeze(0)  # (seq_len, seq_len)

        vocab_expanded = torch.arange(vocab_size, dtype=torch.long,
                                      device=self.device).unsqueeze(1)  # (vocab_size, 1)
        target_expanded = target_tokens_tensor.unsqueeze(0)  # (1, seq_len)
        matches = (vocab_expanded == target_expanded)  # (vocab_size, seq_len)

        suffix_masks = torch.any(
            position_mask.unsqueeze(1) & matches.unsqueeze(0), dim=2
        ).float()  # (seq_len, vocab_size)

        masked_probs = probs * suffix_masks
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)

        # Atleastone constraint via reverse cumulative product (Bayes rule)
        cumprod = None
        if atleastone_constraint:
            target_match_probs = masked_probs[
                pos_indices, target_tokens_tensor]
            cumprod = torch.flip(
                torch.cumprod(torch.flip(target_match_probs, [0]), dim=0), [0])

        i = 1  # Preserve first token; for CruxEval this keeps the leading 'f' of f(...)
        original_positions = list(range(seq_len))
        constraint_satisfied = (
            torch.isnan(cumprod).any() if cumprod is not None else True)

        while i < len(new_tokens) - 1:
            original_pos = original_positions[i]
            if original_pos >= len(probs):
                break

            token_probs = masked_probs[original_pos].clone()

            if atleastone_constraint and not constraint_satisfied:
                target_token_at_pos = self.target_tokens[original_pos]
                future_cumprod = cumprod[original_pos + 1].item()
                constraint_factor = 1.0 - future_cumprod + eps
                token_probs[target_token_at_pos] = (
                    token_probs[target_token_at_pos] * constraint_factor)

            token_probs = token_probs / token_probs.sum()
            sampled_token = torch.multinomial(token_probs, num_samples=1).item()

            # Find where sampled token appears in future positions
            found_at = i
            for j in range(i + 1, len(new_tokens)):
                if new_tokens[j] == sampled_token:
                    found_at = j
                    break

            if atleastone_constraint and (
                    sampled_token != self.target_tokens[original_pos] or found_at > i):
                constraint_satisfied = True

            new_tokens = (new_tokens[:i] + [sampled_token]
                          + new_tokens[found_at + 1:])
            original_positions = (original_positions[:i]
                                  + [original_positions[found_at]]
                                  + original_positions[found_at + 1:])
            i += 1

        if atleastone_constraint and new_tokens == self.target_tokens:
            raise RuntimeError(f"atleastone_constraint failed (seq len {len(self.target_tokens)}): {self.target_tokens}")
        return new_tokens

    def sample_program(self, t: float = 1.0) -> str:
        "Returns a deterministic string sampled via subset resampling."
        new_tokens = self.resample_subset(temperature=t, atleastone_constraint=False)
        return self.tokenizer.decode(new_tokens)

    def sample_program_constraint(self, t: float = 1.0) -> str:
        "Returns a deterministic string sampled via subset resampling, guaranteed different."
        new_tokens = self.resample_subset(temperature=t, atleastone_constraint=True)
        return self.tokenizer.decode(new_tokens)

    def sample(self, n: int = 1, as_list: bool = False, constraint: bool = False,
               **kwargs) -> list:
        "Returns n deterministic strings sampled via subset resampling."
        sample_func = self.sample_program_constraint if constraint else self.sample_program
        return (sample_func(**kwargs) if (n == 1) and (not as_list)
                else [sample_func(**kwargs) for _ in range(n)])

    def greedy(self, as_list: bool = False) -> str:
        "Returns the original generation."
        r = self.raw_string
        return [r] if as_list else r
