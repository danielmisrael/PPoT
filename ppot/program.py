import random
import torch.distributions.gumbel, transformers

class Program:
    "A probabilistic program."

    def __init__(self, C: str, X: list, P: list, V: list, tokenizer: transformers.AutoTokenizer):
        """Constructs a probabilistic program.

        Arguments:
            C: is a formatted string containing variables to be replaced, e.g. "a = {x}", where x
                is the random variable id as an integer.
            X: list of random variable names that map to both P (by name) and pr (by index).
            P: list of torch.FloatTensor encoding the (log-)probability distribution of the i-th RV in X.
            V: list of torch.LongTensor with the ids in the support of each random variable.
            tokenizer: the model's transformers.AutoTokenizer.
        """
        # assert len(X) > 0, "This is a deterministic program!" 
        assert len(X) == len(P), "Number of variables must match number of sets of values."
        assert len(X) == len(V), "Number of variables must match number of sets of values."
        self.code = C
        self.deterministic = False # This is true if the code has no random variable
        if len(X) == 0:
            self.deterministic = True
            return
        self.mapping = {x: p for x, p in zip(X, P)}
        self.G = torch.distributions.gumbel.Gumbel(0, 1)
        self.tokenizer = tokenizer
        self.supp = V
        if torch.is_tensor(P): self.homogenous, self.P_tensor = True, P
        elif all(x.shape == P[0].shape for x in P): self.homogenous, self.P_tensor = True, torch.vstack(P)
        else: self.homogenous, self.P_tensor = False, None

    def sample_program(self) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        # Sample values.
        if self.homogenous:
            S = torch.argmax(self.P_tensor+self.G.sample(self.P_tensor.shape), dim=-1)
        else:
            S = (torch.argmax(p+self.G.sample(p.shape)).item() for p in self.mapping.values())
        V = [self.supp[i][x.item()] for i, x in enumerate(S)]
        # Output code.
        return self.code.format(*self.tokenizer.batch_decode(V))

    def sample(self, n: int = 1) -> list:
        "Returns n deterministic programs sampled from this probabilistic program."
        if self.deterministic:
            return self.code
        return self.sample_program() if n == 1 else [self.sample_program() for _ in range(n)]
    
    def greedy(self) -> str:
        "Returns the deterministic program output from the model"
        if self.deterministic:
            return self.code
        if self.homogenous: # all random variables have size 10
            S = torch.argmax(self.P_tensor, dim=-1)
        else:
            S = (torch.argmax(p).item() for p in self.mapping.values())
        V = [self.supp[i][x.item()] for i, x in enumerate(S)]
        return self.code.format(*self.tokenizer.batch_decode(V))

class USPP(Program):
    "Union of Singleton Probabilistic Programs."

    def __init__(self, V_default: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.V_default = V_default

    def sample_program(self) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        # Assume a uniform prior.
        X = random.randint(0, len(self.mapping)-1)
        # Sample only X.
        pr = self.P_tensor[X] if self.homogenous else self.mapping[X]
        x = torch.argmax(pr + self.G.sample(pr.shape))
        # Fix other values and insert x.
        V = self.V_default.copy()
        V[X] = self.supp[X][x]
        # Output code.
        return self.code.format(*self.tokenizer.batch_decode(V))
