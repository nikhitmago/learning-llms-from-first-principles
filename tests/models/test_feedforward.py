import torch

from learning_llms_from_first_principles.models.feedforward import GELU, Feedforward


def test_gelu_output_shape() -> None:
    gelu = GELU()
    x = torch.randn(2, 4, 8)
    output = gelu(x)
    assert output.shape == (2, 4, 8)


def test_feedforward_output_shape() -> None:
    cfg = {"emb_dim": 16}
    ff = Feedforward(cfg)
    x = torch.randn(2, 4, 16)
    output = ff(x)
    assert output.shape == (2, 4, 16)


def test_gelu_values() -> None:
    gelu = GELU()
    x = torch.tensor([0.0])
    output = gelu(x)
    # GELU(0) should be 0
    assert torch.allclose(output, torch.tensor([0.0]))

    x_pos = torch.tensor([10.0])
    output_pos = gelu(x_pos)
    # For large positive x, GELU(x) should be approximately x
    assert torch.allclose(output_pos, x_pos, atol=1e-3)

    x_neg = torch.tensor([-1.0])
    output_neg = gelu(x_neg)
    # GELU(-1) is approximately -0.1587
    # This helps verify the characteristic 'dip' of GELU below zero
    assert output_neg < 0
    assert torch.allclose(output_neg, torch.tensor([-0.1587]), atol=1e-3)
