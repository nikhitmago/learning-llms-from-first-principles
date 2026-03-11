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

    # Initialize sa with seed 789 as before, but with dropout=0 and max context_len=10
    sa = SelfAttention(d_emb=d_emb, d_attn=d_attn, context_len=10, dropout=0.0)

    # Test batch 3D input directly
    batch_inputs = torch.randn(2, 6, 3)
    batch_output = sa(batch_inputs)
    assert batch_output.shape == (2, 6, 2)

    # Test shorter sequence (Verify mask slicing)
    short_inputs = torch.randn(1, 4, 3)
    short_output = sa(short_inputs)
    assert short_output.shape == (1, 4, 2)

    # Ensure no NaN
    assert not torch.isnan(short_output).any()


def test_multihead_attention_wrapper() -> None:
    torch.manual_seed(123)
    d_emb, d_attn, context_len, num_heads = 12, 4, 8, 2
    dropout = 0.0

    mha = MultiHeadAttentionWrapper(d_emb, d_attn, context_len, dropout, num_heads)

    # Test 3D input (batch_size, seq_len, d_emb)
    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_emb)

    output = mha(x)

    # Output should have shape (batch_size, seq_len, num_heads * d_attn)
    assert output.shape == (batch_size, seq_len, num_heads * d_attn)
    assert not torch.isnan(output).any()


def test_multihead_attention_weight_splits() -> None:
    torch.manual_seed(123)
    d_emb, d_attn, context_len, num_heads = 12, 4, 8, 2
    dropout = 0.0

    mha = MultiHeadAttentionWeightSplits(d_emb, d_attn, context_len, dropout, num_heads)

    # Test 3D input (batch_size, seq_len, d_emb)
    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_emb)

    output = mha(x)

    # Output should have shape (batch_size, seq_len, d_attn)
    assert output.shape == (batch_size, seq_len, d_attn)
    assert not torch.isnan(output).any()


def test_multihead_attention_combined_qkv() -> None:
    torch.manual_seed(123)
    d_emb, d_attn, context_len, num_heads = 12, 4, 8, 2
    dropout = 0.0

    mha = MultiHeadAttentionCombinedQKV(d_emb, d_attn, context_len, dropout, num_heads)

    # Test 3D input (batch_size, seq_len, d_emb)
    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_emb)

    output = mha(x)

    # Output should have shape (batch_size, seq_len, d_attn)
    assert output.shape == (batch_size, seq_len, d_attn)
    assert not torch.isnan(output).any()
