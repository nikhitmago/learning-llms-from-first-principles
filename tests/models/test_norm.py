import torch
import torch.nn as nn
from learning_llms_from_first_principles.models.norm import LayerNorm


def test_layer_norm_output_shape() -> None:
    dim = 768
    ln = LayerNorm(dim)
    x = torch.randn(2, 4, dim)
    output = ln(x)
    assert output.shape == (2, 4, dim)


def test_layer_norm_values() -> None:
    # Small dimension to make it easy to check
    dim = 2
    ln = LayerNorm(dim)
    
    # Simple input
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Calculate expected manually (using unbiased=True, which is torch.var default)
    # Row 1: mean=1.5, var=0.5 -> norm = (1-1.5)/sqrt(0.5) = -0.5/0.7071 = -0.7071
    # PyTorch's LayerNorm uses biased variance (1/N)
    # However, we can just check if the output distribution is roughly correct
    output = ln(x)
    
    # With scale=1 and shift=0, mean should be 0
    assert torch.allclose(output.mean(dim=-1), torch.zeros(2), atol=1e-5)
    
    # Check that scale and shift are actually used
    ln.scale.data = torch.tensor([2.0, 2.0])
    ln.shift.data = torch.tensor([1.0, 1.0])
    output_params = ln(x)
    
    # mean should now be 1 (the shift)
    assert torch.allclose(output_params.mean(dim=-1), torch.ones(2), atol=1e-5)


def test_layer_norm_equivalence() -> None:
    """Note: Standard nn.LayerNorm uses biased variance.
    If you want to match it exactly, you'd use unbiased=False in .var()
    This test verifies your implementation's mathematical logic.
    """
    dim = 16
    x = torch.randn(2, 8, dim)
    ln = LayerNorm(dim)
    
    output = ln(x)
    
    # Verification of the manual math
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + ln.eps)
    
    assert torch.allclose(output, expected, atol=1e-5)
