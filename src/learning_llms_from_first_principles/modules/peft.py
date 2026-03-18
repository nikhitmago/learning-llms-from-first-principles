import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(dim_in, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, dim_out))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A @ self.B)


class LinearLoRA(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adaptation path.

    Intuition:
        Normal fine-tuning updates the full weight matrix W via gradient descent:
            W_new = W + dW          where dW has shape (dim_in, dim_out)

        LoRA's key insight: dW is typically low-rank during fine-tuning, meaning
        most of the update lives in a small subspace. So instead of learning all
        of dW (millions of params), we factorize it into two tiny matrices:
            dW ≈ A @ B             where A is (dim_in, rank) and B is (rank, dim_out)

        Written out as forward passes:
            Regular fine-tuning:  y = x @ (W + dW)  = x @ W  +  x @ dW
            LoRA fine-tuning:     y = x @ (W + AB)  = x @ W  +  x @ A @ B

        The first term (x @ W) is the frozen pretrained output. The second term
        is the lightweight LoRA correction — same math, just with dW replaced by
        the product of two small matrices A and B.

        Only A and B are trained (W stays frozen), so the trainable param count
        drops from (dim_in × dim_out) to (dim_in + dim_out) × rank — often 100x smaller.

        B is zero-initialized so the LoRA path outputs zero at the start of training,
        meaning the model begins exactly where the pretrained weights left off.
    """

    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear_layer = linear_layer
        self.alpha = alpha
        self.lora_layer = LoRALayer(
            dim_in=linear_layer.in_features,
            dim_out=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x) + self.lora_layer(x)

    def merge(self) -> nn.Linear:
        """Blend LoRA adapter into the linear layer and return it."""

        # .T because nn.Linear stores weight as (out, in) but A @ B is (in, out)
        self.linear_layer.weight.data += (self.lora_layer.A @ self.lora_layer.B).T * self.alpha

        return self.linear_layer
