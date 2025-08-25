import random, math
import torch.distributions.gumbel, transformers

class Program:
    "A probabilistic program."

    GUMBEL = torch.distributions.gumbel.Gumbel(0, 1)

    def __init__(self, C: str, X: list, P: list, V: list, tokenizer: transformers.AutoTokenizer, raw_program: str):
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

    def sample_program(self, t: float = 1.0) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        if self.deterministic: return self.code
        if math.isclose(t, 0.0): return self.greedy()
        # Sample values.
        if self.homogenous:
            S = torch.argmax(torch.log_softmax(self.P_tensor/t, dim=-1)+Program.GUMBEL.sample(self.P_tensor.shape), dim=-1)
        else:
            S = (torch.argmax(torch.log_softmax(p/t, dim=-1)+Program.GUMBEL.sample(p.shape)) for p in self.mapping.values())
        V = [self.supp[i][x.item()] for i, x in enumerate(S)]
        # Output code.
        return self.code.format(*V)

    def sample(self, n: int = 1, as_list: bool = False, **kwargs) -> list:
        "Returns n deterministic programs sampled from this probabilistic program."
        return self.sample_program(**kwargs) if (n == 1) and (not as_list) else [self.sample_program(**kwargs) for _ in range(n)]

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

    def __init__(self, V_default: list, raw_program: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.V_default = V_default
        self.raw_program = raw_program

    def sample_program(self, t: float = 1.0) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        if self.deterministic: return self.code
        if t == 0.0: return self.greedy()
        # Assume a uniform prior.
        X = random.randint(0, len(self.V)-1)
        # Sample only X.
        pr = self.P_tensor[X] if self.homogenous else self.mapping[X]
        x = torch.argmax(torch.log_softmax(pr/t, dim=-1) + Program.GUMBEL.sample(pr.shape))
        # Fix other values and insert x.
        V = self.V_default.copy()
        V[X] = self.supp[X][x]
        # Output code.
        return self.code.format(*V)
