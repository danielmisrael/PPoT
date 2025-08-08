import torch.distributions.gumbel

class Program:
    "A probabilistic program."

    def __init__(self, C: str, X: list, P: list):
        """Constructs a probabilistic program where

        Arguments:
            C: is a formatted string containing variables to be replaced, e.g. "a = {x}", where x
                is the random variable id as an integer.
            X: list of random variable names that map to both P (by name) and pr (by index).
            P: list of torch.FloatTensor encoding the probability distribution of the i-th RV in X.
        """
        assert len(X) > 0, "This is a deterministic program!"
        assert len(X) == len(P), "Number of variables must match number of sets of values."
        self.code = C
        self.mapping = {x: p for x, p in zip(X, P)}
        self.G = torch.distributions.gumbel.Gumbel(0, 1)
        self.homogenous = all(x.shape == X[0].shape for x in X)
        self.P_tensor = self.vstack(X) if self.homogenous else None

    def sample(self) -> str:
        "Returns a deterministic program sampled from this probabilistic program."
        # Sample values.
        V = [x.item() for x in torch.argmax(self.P_tensor + self.G(self.P_tensor.shape), dim=-1)] \
            if self.homogenous else [torch.argmax(p + self.G(p.shape)).item() for p in self.mapping.values()]
        # Output code.
        return self.code.format(*V)
