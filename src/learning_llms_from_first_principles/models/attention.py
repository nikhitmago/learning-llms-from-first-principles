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
        attn_scores = attn_scores.masked_fill(
            self.mask[:cur_context_len, :cur_context_len], -torch.inf
        )

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


class MultiHeadAttentionWeightSplits(nn.Module):
    "MHA with weight splits for QKV"

    mask: torch.Tensor

    def __init__(self, d_emb: int, d_attn: int, context_len: int, dropout: float, num_heads: int):
        super().__init__()

        assert d_attn % num_heads == 0, "d_out must be divisible by num_heads"

        # QKV matrices and attn proj matrices
        self.W_q = nn.Linear(d_emb, d_attn, bias=False)
        self.W_k = nn.Linear(d_emb, d_attn, bias=False)
        self.W_v = nn.Linear(d_emb, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_attn)

        self.head_dim = d_attn // num_heads
        self.num_heads = num_heads
        self.d_attn = d_attn
        self.dropout_layer = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, d_emb = x.shape  # seq_len <= context_len

        # qkv: bs, seq_len, d_attn
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # split d_attn into num_heads, head_dim
        queries = queries.view(bs, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(bs, seq_len, self.num_heads, self.head_dim)
        values = values.view(bs, seq_len, self.num_heads, self.head_dim)

        # transpose qk to (bs, num_heads, seq_len, head_dim) for attn_scores
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # attn_scores: bs, num_heads, seq_len, seq_len
        attn_scores = queries @ keys.transpose(2, 3)

        # apply attn mask for causal attention: shape remains the same
        attn_scores = attn_scores.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)

        # apply softmax to get attention weights: shape remains the same
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # apply dropout: shape remains the same
        attn_weights = self.dropout_layer(attn_weights)

        # get the context vector: bs, num_heads, seq_len, head_dim
        # attn_weigths: bs, num_heads, seq_len, seq_len
        #       values: bs, num_heads, seq_len, head_dim
        context_vec = attn_weights @ values

        # transpose the context_vec: bs, seq_len, num_heads, head_dim
        context_vec = context_vec.transpose(1, 2)

        # combine the heads before linear layer
        context_vec = context_vec.contiguous().view(bs, seq_len, self.d_attn)

        # project
        context_vec = self.out_proj(context_vec)
        return context_vec


class MultiHeadAttentionCombinedQKV(nn.Module):
    mask: torch.Tensor

    def __init__(
        self, d_emb: int, d_attn: int, context_len: int, dropout: float, num_heads: int
    ) -> None:
        super().__init__()
        assert d_attn % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.d_emb = d_emb
        self.d_attn = d_attn
        self.context_len = context_len
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads

        self.qkv = nn.Linear(d_emb, 3 * d_attn, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_attn, d_attn)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, d_emb = x.shape

        # qkv weights: bs, seq_len, 3 * d_attn
        #   qkv split: bs, seq_len, 3, d_attn
        #   qkv split: bs, seq_len, 3, num_heads, head_dim
        #   qkv split: 3, bs, seq_len, num_heads, head_dim
        #   qkv split: 3, bs, num_heads, seq_len, head_dim
        qkv = self.qkv(x)
        qkv = qkv.view(bs, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # split qkv: bs, num_heads, seq_len, head_dim
        queries, keys, values = qkv.unbind(0)

        # attn_scores: bs, num_heads, seq_len, seq_len
        attn_scores = queries @ keys.transpose(2, 3)

        # apply attn_mask
        attn_scores = attn_scores.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)

        # apply softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # apply dropout
        attn_weights = self.dropout_layer(attn_weights)

        # get the context vector: bs, num_heads, seq_len, head_dim
        context_vec = attn_weights @ values

        # transpose the context vector to combine: bs, seq_len, num_heads, head_dim
        context_vec = context_vec.transpose(1, 2)

        # combine the last 2 dims of the context vec
        context_vec = context_vec.contiguous().view(bs, seq_len, self.d_attn)

        context_vec = self.out_proj(context_vec)
        return context_vec
