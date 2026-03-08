import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_emb: int, d_attn: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_emb, d_attn, bias=False)
        self.W_k = nn.Linear(d_emb, d_attn, bias=False)
        self.W_v = nn.Linear(d_emb, d_attn, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (context_len, d_emb) x (d_emb, d_attn) => (context_len, d_attn)
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # Attention Scores: (context_len, d_attn) x (d_attn, context_len) => (context_len, context_len)
        attn_scores = queries @ keys.T

        # Attention Weights: (context_len, context_len)
        attn_weights = torch.softmax(attn_scores / (keys.shape[1] ** 0.5), dim=1)

        # Context Vector: (context_len, d_attn)
        context_vec = attn_weights @ values

        return context_vec
