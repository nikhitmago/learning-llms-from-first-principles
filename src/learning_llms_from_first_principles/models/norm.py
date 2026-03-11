import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = eps

        # Scale (gamma) and shift (beta) provide representational flexibility,
        # allowing the model to learn the optimal distribution for the data
        # rather than being strictly forced to mean 0 and variance 1.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # keepdim=True preserves the last dimension as 1, enabling automatic
        # broadcasting when subtracting the mean and dividing by the variance.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # to prevent div0
        return self.scale * norm_x + self.shift
