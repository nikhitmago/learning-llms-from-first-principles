import torch

from learning_llms_from_first_principles.quantization.per_channel import (
    per_channel_quantize,
)


def test_per_channel_roundtrip() -> None:
    """Dequantized weights should be close to original."""
    torch.manual_seed(42)
    weight = torch.randn(4, 8)
    _, _, deq = per_channel_quantize(weight, bits=8)
    assert torch.allclose(weight, deq, atol=0.02)


def test_per_channel_quantized_in_range() -> None:
    """Quantized values should be within [-128, 127] for 8-bit."""
    torch.manual_seed(0)
    weight = torch.randn(4, 8) * 10
    q, _, _ = per_channel_quantize(weight, bits=8)
    assert q.min() >= -128
    assert q.max() <= 127


def test_per_channel_scales_shape() -> None:
    """Scales should have one value per output channel."""
    weight = torch.randn(6, 16)
    _, scales, _ = per_channel_quantize(weight, bits=8)
    assert scales.shape == (6,)
