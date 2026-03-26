import torch

from learning_llms_from_first_principles.modules.attention import gated_attention


def test_gated_output_bounded_by_standard_attention() -> None:
    """Gate is sigmoid ∈ [0,1], so gated output magnitude <= standard attention output."""
    torch.manual_seed(42)
    seq_len, d_model, d_k, d_v = 5, 8, 4, 4
    X = torch.rand(seq_len, d_model)
    W_q = torch.rand(d_model, d_k)
    W_k = torch.rand(d_model, d_k)
    W_v = torch.rand(d_model, d_v)
    W_g = torch.rand(d_model, d_v)

    # Standard attention (gate all ones = no gating)
    W_g_ones = torch.zeros(d_model, d_v) + 100.0  # sigmoid(100) ≈ 1.0
    out_standard = gated_attention(X, W_q, W_k, W_v, W_g_ones, apply_mask=False)
    out_gated = gated_attention(X, W_q, W_k, W_v, W_g, apply_mask=False)

    assert (out_gated.abs() <= out_standard.abs() + 1e-6).all()


def test_gated_output_shape() -> None:
    """Output shape should be (seq_len, d_v)."""
    torch.manual_seed(0)
    seq_len, d_model, d_k, d_v = 6, 8, 4, 3
    X = torch.rand(seq_len, d_model)
    result = gated_attention(
        X,
        torch.rand(d_model, d_k),
        torch.rand(d_model, d_k),
        torch.rand(d_model, d_v),
        torch.rand(d_model, d_v),
    )
    assert result.shape == (seq_len, d_v)
    assert not torch.isnan(result).any()


def test_causal_mask_prevents_future_leakage() -> None:
    """With causal mask, changing a future token should not affect earlier outputs."""
    torch.manual_seed(7)
    seq_len, d_model, d_k, d_v = 4, 8, 4, 4
    X = torch.rand(seq_len, d_model)
    W_q = torch.rand(d_model, d_k)
    W_k = torch.rand(d_model, d_k)
    W_v = torch.rand(d_model, d_v)
    W_g = torch.rand(d_model, d_v)

    out1 = gated_attention(X, W_q, W_k, W_v, W_g, apply_mask=True)

    # Modify the last token
    X2 = X.clone()
    X2[-1] = torch.rand(d_model)
    out2 = gated_attention(X2, W_q, W_k, W_v, W_g, apply_mask=True)

    # All rows except the last should be unchanged
    assert torch.allclose(out1[:-1], out2[:-1], atol=1e-6)
