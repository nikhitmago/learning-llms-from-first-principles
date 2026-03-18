import time

import torch

from learning_llms_from_first_principles.modules.attention import (
    MultiHeadAttentionCombinedQKV,
    MultiHeadAttentionWeightSplits,
    MultiHeadAttentionWrapper,
    SelfAttention,
)


def test_self_attention() -> None:
    torch.manual_seed(789)
    d_emb, d_attn = 3, 2

    sa = SelfAttention(d_emb=d_emb, d_attn=d_attn, context_len=10, dropout=0.0)

    batch_inputs = torch.randn(2, 6, 3)
    batch_output = sa(batch_inputs)
    assert batch_output.shape == (2, 6, 2)

    short_inputs = torch.randn(1, 4, 3)
    short_output = sa(short_inputs)
    assert short_output.shape == (1, 4, 2)

    assert not torch.isnan(short_output).any()


def test_self_attention_kv_cache_correctness() -> None:
    """Verify that cached decode produces the same output as full-sequence forward."""
    torch.manual_seed(42)
    d_emb, d_attn, context_len = 16, 8, 32
    sa = SelfAttention(d_emb, d_attn, context_len, dropout=0.0)
    sa.eval()

    prompt = torch.randn(1, 5, d_emb)
    new_tokens = [torch.randn(1, 1, d_emb) for _ in range(3)]

    full_seq = torch.cat([prompt] + new_tokens, dim=1)
    with torch.no_grad():
        expected = sa(full_seq, use_kv_cache=False)

    sa.reset_kv_cache()
    with torch.no_grad():
        prefill_out = sa(prompt, use_kv_cache=True)
        decode_outs = []
        for tok in new_tokens:
            decode_outs.append(sa(tok, use_kv_cache=True))

    cached_out = torch.cat([prefill_out] + decode_outs, dim=1)
    assert torch.allclose(expected, cached_out, atol=1e-5)


def test_self_attention_kv_cache_is_faster() -> None:
    """KV cache decode should be faster than recomputing the full sequence each step."""
    torch.manual_seed(42)
    d_emb, d_attn, context_len = 64, 32, 256
    num_decode_steps = 50

    sa = SelfAttention(d_emb, d_attn, context_len, dropout=0.0)
    sa.eval()

    prompt = torch.randn(1, 10, d_emb)

    start = time.perf_counter()
    with torch.no_grad():
        seq = prompt
        for _ in range(num_decode_steps):
            new_tok = torch.randn(1, 1, d_emb)
            seq = torch.cat([seq, new_tok], dim=1)
            _ = sa(seq, use_kv_cache=False)
    time_no_cache = time.perf_counter() - start

    sa.reset_kv_cache()
    start = time.perf_counter()
    with torch.no_grad():
        _ = sa(prompt, use_kv_cache=True)
        for _ in range(num_decode_steps):
            new_tok = torch.randn(1, 1, d_emb)
            _ = sa(new_tok, use_kv_cache=True)
    time_with_cache = time.perf_counter() - start

    assert (
        time_with_cache < time_no_cache
    ), f"Cache ({time_with_cache:.4f}s) should be faster than no cache ({time_no_cache:.4f}s)"
    speedup = time_no_cache / time_with_cache
    print(
        f"\nKV cache: {time_with_cache:.4f}s | No cache: {time_no_cache:.4f}s | {speedup:.1f}x faster"
    )


def test_multihead_attention_wrapper() -> None:
    torch.manual_seed(123)
    d_emb, d_attn, context_len, num_heads = 12, 4, 8, 2
    dropout = 0.0

    mha = MultiHeadAttentionWrapper(d_emb, d_attn, context_len, dropout, num_heads)

    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_emb)

    output = mha(x)

    assert output.shape == (batch_size, seq_len, num_heads * d_attn)
    assert not torch.isnan(output).any()


def test_multihead_attention_wrapper_kv_cache_correctness() -> None:
    """Verify MHA wrapper cached decode matches full-sequence forward."""
    torch.manual_seed(42)
    d_emb, d_attn, context_len, num_heads = 16, 8, 32, 2

    mha = MultiHeadAttentionWrapper(d_emb, d_attn, context_len, dropout=0.0, num_heads=num_heads)
    mha.eval()

    prompt = torch.randn(1, 5, d_emb)
    new_tokens = [torch.randn(1, 1, d_emb) for _ in range(3)]

    full_seq = torch.cat([prompt] + new_tokens, dim=1)
    with torch.no_grad():
        expected = mha(full_seq, use_kv_cache=False)

    for head in mha.heads:
        head.reset_kv_cache()  # type: ignore[operator]
    with torch.no_grad():
        prefill_out = mha(prompt, use_kv_cache=True)
        decode_outs = []
        for tok in new_tokens:
            decode_outs.append(mha(tok, use_kv_cache=True))

    cached_out = torch.cat([prefill_out] + decode_outs, dim=1)
    assert torch.allclose(expected, cached_out, atol=1e-5)


def test_multihead_attention_weight_splits() -> None:
    torch.manual_seed(123)
    d_emb, d_attn, context_len, num_heads = 12, 4, 8, 2
    dropout = 0.0

    mha = MultiHeadAttentionWeightSplits(d_emb, d_attn, context_len, dropout, num_heads)

    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_emb)

    output = mha(x)

    assert output.shape == (batch_size, seq_len, d_attn)
    assert not torch.isnan(output).any()


def test_multihead_attention_combined_qkv() -> None:
    torch.manual_seed(123)
    d_emb, d_attn, context_len, num_heads = 12, 4, 8, 2
    dropout = 0.0

    mha = MultiHeadAttentionCombinedQKV(d_emb, d_attn, context_len, dropout, num_heads)

    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_emb)

    output = mha(x)

    assert output.shape == (batch_size, seq_len, d_attn)
    assert not torch.isnan(output).any()
