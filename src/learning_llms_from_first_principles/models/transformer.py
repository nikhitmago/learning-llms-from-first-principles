import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        # A simple placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This block does nothing and just returns its input.
        return x
