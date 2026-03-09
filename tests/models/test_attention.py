import torch

from models.attention import SelfAttention


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
