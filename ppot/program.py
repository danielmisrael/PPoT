import random, math
import torch.distributions.gumbel, transformers
import ppot.utils
from copy import deepcopy

class Program:
    "A probabilistic program."

    GUMBEL = torch.distributions.gumbel.Gumbel(0, 1)

    def __init__(self, C: str, X: list, P: list, V: list, tokenizer: transformers.AutoTokenizer,
                 raw_program: str, actual_values: list, device: str = None):
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

    def to(self, device: str):
        if device == self.device: return self
        if self.homogenous:
            self.P_tensor = self.P_tensor.to(device)
        else:
            for k in self.mapping: self.mapping[k] = self.mapping[k].to(device)
        self.gumbel = ppot.utils.gumbel_on(device)
        return self

    # def sample_program(self, t: float = 1.0) -> str:
    #     "Returns a deterministic program sampled from this probabilistic program."
    #     if self.deterministic: return self.code
    #     if math.isclose(t, 0.0): return self.greedy()
    #     # Sample values.
    #     if self.homogenous:
    #         S = torch.argmax(torch.log_softmax(self.P_tensor/t, dim=-1)+self.gumbel.sample(self.P_tensor.shape), dim=-1).cpu()
    #     else:
    #         S = (torch.argmax(torch.log_softmax(p/t, dim=-1)+self.gumbel.sample(p.shape)) for p in self.mapping.values()).cpu()
    #     V = [self.supp[i][x.item()] for i, x in enumerate(S)]
    #     # Output code.
    #     return self.code.format(*V)
    
    def sample_program(self, t:float=1.0) -> str:
        "Returns a deterministic program sampled from this probabilistic program conditioned on not being the same program"
        if self.homogenous:
            mapping = {i: torch.log_softmax(self.P_tensor[i, :]/t, dim=-1) for i in range(self.P_tensor.shape[0])}
        else:
            mapping = {k: torch.log_softmax(v/t, dim=-1) for k, v in self.mapping}
        logits_actual_value = []
        for k in sorted(mapping):
            v = mapping[k]
            key_in_logits = self.supp[k].index(self.actual_values[k])
            logits_actual_value.append(v[key_in_logits])
        logits_actual_value = torch.tensor(logits_actual_value)
        log_cum_prod = torch.sum(logits_actual_value)

        # in this sample list, False indicates change the variable, True indicates keep the variable same
        sample = []
        independent = False
        for i in range(logits_actual_value.shape[0]):
            if independent:
                log_odds = logits_actual_value[i] - torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(logits_actual_value[i])))
                sample.append(torch.distributions.bernoulli.Bernoulli(logits=log_odds))
            else:
                Z = torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(log_cum_prod)))
                log_cum_prod -= logits_actual_value[i]
                logits = logits_actual_value[i]+torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(log_cum_prod)))-Z
                log_odds = logits - torch.log(torch.max(torch.tensor(1e-9), -torch.expm1(logits)))
                sample.append(torch.distributions.bernoulli.Bernoulli(logits=log_odds).sample())
                independent = not sample[-1]

        if len(sample) != 0:
            assert sample != [torch.tensor(True, dtype=torch.float) for i in range(len(sample))]

        final_sample = []
        for i, change in enumerate(sample):
            if not change:
                logits = deepcopy(mapping[i])
                key_in_logits = self.supp[i].index(self.actual_values[i])
                # assert key_in_logits == int(self.actual_values[i])
                logits[key_in_logits] = -torch.inf
                final_sample.append(torch.argmax(torch.log_softmax(logits, dim=-1)+self.gumbel.sample()))
            else:
                key_in_logits = self.supp[i].index(self.actual_values[i])
                final_sample.append(key_in_logits)
        V = [self.supp[i][x] for i, x in enumerate(final_sample)]
        return self.code.format(*V)

        


        

        # compute the p values of the actual token
        # compute 1-p
        # sample true and false going through all the 
        # change the logits accordingly
        # gumbel sample

    def sample(self, n: int = 1, as_list: bool = False, **kwargs) -> list:
        "Returns n deterministic programs sampled from this probabilistic program."
        return self.sample_program(**kwargs) if (n == 1) and (not as_list) else [self.sample_program(**kwargs) for _ in range(n)]
        breakpoint()
    def greedy(self, as_list: bool = False) -> str:
        "Returns the deterministic program output from the model"
        if self.deterministic: return [self.code] if as_list else self.code
        S = torch.argmax(self.P_tensor, dim=-1) if self.homogenous else \
            (torch.argmax(p).item() for p in self.mapping.values())
        V = [self.supp[i][x.item()] for i, x in enumerate(S)]
        r = self.code.format(*V)
        return [r] if as_list else r

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
