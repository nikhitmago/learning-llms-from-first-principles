import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_emb: int, d_attn: int, context_len: int, dropout: float) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_emb, d_attn, bias=False)
        self.W_k = nn.Linear(d_emb, d_attn, bias=False)
        self.W_v = nn.Linear(d_emb, d_attn, bias=False)

        self.dropout_layer = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )  # Registers a non-trainable tensor that moves with the model across devices (CPU/GPU) and is included in the saved state.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, cur_context_len, d_emb = (
            x.shape
        )  # context len can be shorter at inference time when doing one at a time token processing

        # bs, context_len, d_attn
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # attention scores: bs, context_len, context_len
        attn_scores = queries @ keys.transpose(
            1, 2
        )  # .T won't work becasu ootb cuz there is batch dim

        # allow masking sequences shorter than max context_len by slicing the mask: bs, cur_context_len, cur_context_len
        attn_scores.masked_fill_(self.mask[:cur_context_len, :cur_context_len], -torch.inf)

        # attention weights: apply softmax after normalizing with key's d_attn dim
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)

        # apply dropout after softmax which also scales the non dropped out values, regularizer
        attn_weights = self.dropout_layer(attn_weights)

        # get context vector
        context_vec = (
            attn_weights @ values
        )  # Matrix-multiplies the (6, 6) mask against each element in the (10, 6, 2) batch by broadcasting across the batch dimension.
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    "Naive implementation of MHA by stacking SA modules"

    def __init__(
        self, d_emb: int, d_attn: int, context_len: int, dropout: float, num_heads: int
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [SelfAttention(d_emb, d_attn, context_len, dropout) for i in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)
