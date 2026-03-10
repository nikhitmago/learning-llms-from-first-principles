import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This layer does nothing and just returns its input.
        return x
