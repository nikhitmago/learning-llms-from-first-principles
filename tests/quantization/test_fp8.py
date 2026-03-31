import torch

from learning_llms_from_first_principles.quantization.fp8 import (
    fp8_block_dequantize,
    fp8_block_quantize,
)


def test_fp8_roundtrip() -> None:
    """Quantize then dequantize should recover values closely."""
    torch.manual_seed(42)
    x = torch.rand(256)
    quantized, scales = fp8_block_quantize(x, block_size=128)
    recovered = fp8_block_dequantize(quantized, scales, block_size=128)
    assert torch.allclose(x, recovered, atol=1e-4)


def test_fp8_output_shapes() -> None:
    """Quantized should be (N,), scales should be (N // block_size,)."""
    x = torch.rand(512)
    quantized, scales = fp8_block_quantize(x, block_size=128)
    assert quantized.shape == (512,)
    assert scales.shape == (4,)


def test_fp8_quantized_within_range() -> None:
    """Quantized values should be within [-448, 448]."""
    torch.manual_seed(0)
    x = torch.randn(256) * 100
    quantized, _ = fp8_block_quantize(x, block_size=128)
    assert quantized.abs().max() <= 448.0 + 1e-3
