import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    mask: torch.Tensor
    cache_k: torch.Tensor | None
    cache_v: torch.Tensor | None

    def __init__(self, d_emb: int, d_attn: int, context_len: int, dropout: float) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_emb, d_attn, bias=False)
        self.W_k = nn.Linear(d_emb, d_attn, bias=False)
        self.W_v = nn.Linear(d_emb, d_attn, bias=False)

        self.dropout_layer = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        )  # Registers a non-trainable tensor that moves with the model across devices (CPU/GPU) and is included in the saved state.

        # Register optional KV cache buffers
        # persistent=False means the buffer lives on the module (moves with .to(device),
        # shows up in model.state_dict() calls during runtime)
        # but does not get saved when you torch.save(model.state_dict())
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cur_pos = 0

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False) -> torch.Tensor:
        bs, cur_context_len, d_emb = (
            x.shape
        )  # context len can be shorter at inference time when doing one at a time token processing

        # bs, context_len, d_attn
        queries = self.W_q(
            x
        )  # for kv cache, queries will be one word at a time in inference (bs, 1, d_attn)
        keys_new = self.W_k(x)
        values_new = self.W_v(x)

        if use_kv_cache:
            # KV Cache: avoid recomputing keys/values for all previous tokens.
            #
            # PREFILL (first call, e.g. prompt "I am good"):
            #   cache is None → store all 3 key/value vectors
            #   cache_k shape: (bs, 3, d_attn)
            #
            # DECODE (subsequent calls, one new token at a time):
            #   cache exists → append the 1 new key/value to the cache
            #   cache_k shape: (bs, 4, d_attn), then (bs, 5, d_attn), etc.
            #
            # In both cases, queries only cover the NEW tokens,
            # but keys/values span ALL tokens seen so far.
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                assert self.cache_k is not None and self.cache_v is not None
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        # attention scores: bs, context_len, context_len
        # if kv cache: (bs, 1, d_attn) x (bs, d_attn, context_len) => (bs, 1, context_len)
        attn_scores = queries @ keys.transpose(
            1, 2
        )  # .T won't work becasu ootb cuz there is batch dim

        # Masking: cur_pos tracks which row of the causal mask we're on.
        #
        # PREFILL (prompt "I am good", cur_pos=0):
        #   num_tokens_queries=3, num_tokens_keys=3
        #   mask slice: self.mask[0:3, :3]  →  standard 3×3 causal mask
        #   cur_pos advances to 3
        #
        # DECODE step 1 (token 4, cur_pos=3):
        #   num_tokens_queries=1, num_tokens_keys=4
        #   mask slice: self.mask[3:4, :4]  →  [False, False, False, False]
        #   All False = attend to everything. Causality is already enforced
        #   because future tokens simply aren't in the cache yet.
        #   cur_pos advances to 4
        #
        # Without cache: just slice the top-left corner of the mask as usual.
        num_tokens_queries = queries.shape[1]
        num_tokens_keys = keys.shape[1]

        if use_kv_cache:
            attn_scores = attn_scores.masked_fill(
                self.mask[
                    self.cur_pos : self.cur_pos + num_tokens_queries,
                    :num_tokens_keys,
                ],
                -torch.inf,
            )
            self.cur_pos += num_tokens_queries
        else:
            attn_scores = attn_scores.masked_fill(
                self.mask[:num_tokens_queries, :num_tokens_keys], -torch.inf
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

    def reset_kv_cache(self) -> None:
        self.cache_k, self.cache_v = None, None
        self.cur_pos = 0


class MultiHeadAttentionWrapper(nn.Module):
    "Naive implementation of MHA by stacking SA modules"

    def __init__(
        self, d_emb: int, d_attn: int, context_len: int, dropout: float, num_heads: int
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [SelfAttention(d_emb, d_attn, context_len, dropout) for i in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False) -> torch.Tensor:
        return torch.cat([head(x, use_kv_cache=use_kv_cache) for head in self.heads], dim=-1)


class MultiHeadAttentionWeightSplits(nn.Module):
    "MHA with weight splits for QKV"

    mask: torch.Tensor
    cache_k: torch.Tensor | None
    cache_v: torch.Tensor | None

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

        # Regsiter kv cache buffers and variables
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cur_pos = 0

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False) -> torch.Tensor:
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
        keys_new = keys.transpose(1, 2)
        values_new = values.transpose(1, 2)

        # apply KV cache
        if use_kv_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                assert self.cache_k is not None and self.cache_v is not None
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=-2)  # dim=seq_len
                self.cache_v = torch.cat([self.cache_v, values_new], dim=-2)  # dim=seq_len
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        # attn_scores: bs, num_heads, seq_len, seq_len
        attn_scores = queries @ keys.transpose(2, 3)

        # kv cache stuff
        num_tokens_queries = queries.shape[-2]
        num_tokens_keys = keys.shape[-2]

        # change the mask based on kv cache usage
        if use_kv_cache:
            attn_scores = attn_scores.masked_fill(
                self.mask[self.cur_pos : self.cur_pos + num_tokens_queries, :num_tokens_keys],
                -torch.inf,
            )
            self.cur_pos += num_tokens_queries
        else:
            attn_scores = attn_scores.masked_fill(
                self.mask[:num_tokens_queries, :num_tokens_keys], -torch.inf
            )

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

    def reset_kv_cache(self) -> None:
        self.cache_k, self.cache_v = None, None
        self.cur_pos = 0


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


def flash_attention_v1(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 2
) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.

    Args:
        Q: Query matrix (seq_len, head_dim)
        K: Key matrix (seq_len, head_dim)
        V: Value matrix (seq_len, head_dim)
        block_size: Size of blocks for tiled computation

    Returns:
        Output matrix (seq_len, head_dim)

    Visual example — 'The cat sat on the mat' (seq_len=6, block_size=2, num_blocks=3):

    The full attention matrix is never materialized. Instead, we process one block at a time:

                  K0       K1       K2
               |The cat |sat on  |the mat |
         ------+--------+--------+--------+
    Q0   The   |  .  .  |  .  .  |  .  .  |
         cat   |  .  .  |  .  .  |  .  .  |
         ------+--------+--------+--------+
    Q1   sat   |  .  .  |  .  .  |  .  .  |
         on    |  .  .  |  .  .  |  .  .  |
         ------+--------+--------+--------+
    Q2   the   |  .  .  |  .  .  |  .  .  |
         mat   |  .  .  |  .  .  |  .  .  |
         ------+--------+--------+--------+

    Processing order (9 steps, one 2x2 block at a time):
      Step 1: i=0,j=0 → 'The,cat' attend to 'The,cat'
      Step 2: i=0,j=1 → 'The,cat' attend to 'sat,on'
      Step 3: i=0,j=2 → 'The,cat' attend to 'the,mat'  → rows 0,1 done, normalize
      Step 4: i=1,j=0 → 'sat,on'  attend to 'The,cat'
      Step 5: i=1,j=1 → 'sat,on'  attend to 'sat,on'
      Step 6: i=1,j=2 → 'sat,on'  attend to 'the,mat'  → rows 2,3 done, normalize
      Step 7: i=2,j=0 → 'the,mat' attend to 'The,cat'
      Step 8: i=2,j=1 → 'the,mat' attend to 'sat,on'
      Step 9: i=2,j=2 → 'the,mat' attend to 'the,mat'  → rows 4,5 done, normalize

    HW note: In a real GPU implementation, each block's operations (score computation,
    softmax update, V accumulation) are fused and happen entirely in SRAM. Q/K/V blocks
    are loaded from HBM to SRAM once per block, and only the final O is written back.
    The full attention matrix never touches HBM — that's the core memory saving.
    This Python version is a functional reference; the actual speedup comes from that
    HBM ↔ SRAM transfer avoidance on hardware.
    """
    seq_len, head_dim = Q.shape
    num_blocks = (seq_len + block_size - 1) // block_size  # trick to handle odd-sized seq_lens

    m = torch.full((seq_len,), -torch.inf)  # needed to track global max per query-row
    l = torch.zeros(seq_len)  # needed to track global sum per query-row  # noqa: E741
    O = torch.zeros((seq_len, head_dim))  # final output  # noqa: E741

    for i in range(num_blocks):

        ## chunk queries into blocks
        q_start = i * block_size
        q_end = min(
            (i + 1) * block_size, seq_len
        )  # min() clamps for last block if seq_len % block_size != 0
        Q_block = Q[
            q_start:q_end, :
        ]  # (actual_block_size, head_dim) — may be < block_size on last block

        for j in range(num_blocks):

            ## Chunk keys into blocks
            k_start = j * block_size
            k_end = min((j + 1) * block_size, seq_len)  # same clamp for keys
            K_block = K[k_start:k_end, :]  # (actual_block_size, head_dim)

            ## block-wise attention scores
            attn_scores_ij = (Q_block @ K_block.T) * (1 / (head_dim**0.5))  # (q_block, k_block)

            ## Calculate old stats (shape: (q_block_size, ))
            m_old = m[q_start:q_end].clone()  # .clone() is critical — without it m_old is a view,
            l_old = l[
                q_start:q_end
            ].clone()  # and updating m/l later would silently corrupt m_old/l_old

            ## Calculate new stats (shape: (q_block_size, ))

            ### new max
            m_new = torch.maximum(
                torch.max(
                    attn_scores_ij, dim=1
                ).values,  # maximum of attn_scores_ij for each query in a block
                m_old,
            )  # uber-level element wise maxium along with m_old

            ### new sum using softmax trick
            # let, softmax_new = exp(x_cur - m_new) / exp(x_all - m_new).sum()
            # we know, l_old = exp(x_all - m_old).sum()
            # lets add 2 new terms to this equation because we dont want to load x_all when processing a block
            # => attn_weights_ij = exp(attn_scores_ij - m_new) / exp(attn_scores_ij - m_new + m_old - m_old).sum() # add 2 new terms
            # => attn_weights_ij = exp(attn_scores_ij - m_new) / exp(attn_scores_ij - m_old + m_old - m_new).sum() # re-arrange
            # => attn_weights_ij = exp(attn_scores_ij - m_new) / (exp(attn_scores_ij - m_old) * exp(m_old - m_new)).sum() # multiply the exps
            # => attn_weights_ij = exp(attn_scores_ij - m_new) / exp(m_old - m_new) * exp(attn_scores_ij - m_old).sum() # take out constant from the sum
            # => attn_weights_ij = exp(attn_scores_ij - m_new) / exp(m_old - m_new) * l_old # substitute
            # => l_new = exp(m_old - m_new) * l_old  # extract the denominator
            # => l_new = exp(m_old - m_new) * l_old + exp(attn_scores_ij - m_new).sum(dim=1)  # l_old doesn't factor in the current block, so we must add the numerator sum

            # [:, None] reshapes (q_block,) -> (q_block, 1) so each row gets its own max subtracted.
            # Without it, broadcasting treats (q_block,) as (1, q_block) = column-wise subtraction = WRONG.
            exp_numerator = torch.exp(attn_scores_ij - m_new[:, None])
            l_new = torch.exp(m_old - m_new) * l_old + exp_numerator.sum(dim=1)

            # Normally you'd compute attn_weights = exp_numerator / l_new, but we skip it
            # we multiply exp_numerator @ V directly and defer the /l to the final normalization.

            ## Chunk the V block
            V_block = V[k_start:k_end, :]

            ## Update output with correction
            # WHY correction: Old O was accumulated using exp(scores - m_old). Now m changed to m_new,
            # so old values are on the wrong scale. We fix without recomputing:
            #   exp(score - m_new) = exp(score - m_old) * exp(m_old - m_new)
            #                        ─────────────────    ─────────────────
            #                        baked into old O     correction factor
            # If max didn't change: exp(0) = 1 → no-op. If max went up: < 1 → shrinks old values.
            # Current block's exp_numerator is already computed with m_new, so no correction needed for it.
            O_block = exp_numerator @ V_block
            correction = torch.exp(m_old - m_new)
            O[q_start:q_end, :] = O[q_start:q_end, :] * correction[:, None] + O_block

            ## Write back updated stats
            m[q_start:q_end] = m_new
            l[q_start:q_end] = l_new

        ## Normalize: divide ONCE after inner loop, not per-block.
        # WHY: O accumulates unnormalized weighted sums (sum of exp(s)*V) during the j loop.
        # l keeps changing as new blocks may reveal bigger maxes (triggering corrections).
        # Only after the inner loop finishes has this Q block seen ALL keys, so l is final.
        # Dividing earlier would be wrong because l isn't complete yet.
        O[q_start:q_end, :] /= l[q_start:q_end, None]

    return O
