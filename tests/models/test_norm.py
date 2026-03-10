import torch

from learning_llms_from_first_principles.models.norm import LayerNorm


def test_layer_norm_identity() -> None:
    dim = 768
    ln = LayerNorm(dim)
    x = torch.randn(2, 4, dim)
    output = ln(x)
    # Verifying it currently acts as an identity layer as intended
    assert torch.equal(x, output)
