import torch


def fp8_block_quantize(
    tensor: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.

    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block

    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    FP8_MAX = 448.0
    EPS = 1e-12

    num_blocks = tensor.shape[0] // block_size
    tensor_blocks = tensor.reshape(num_blocks, block_size).to(
        torch.float32
    )  # num_blocks, block_size
    max_vals = tensor_blocks.abs().max(dim=-1).values
    scales = (max_vals + EPS) / FP8_MAX

    quantized_blocks = tensor_blocks / scales[:, None]

    return quantized_blocks.reshape(-1), scales


def fp8_block_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize FP8-E4M3 values back to full precision.

    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block

    Returns:
        Dequantized tensor of shape (N,)
    """
    num_blocks = quantized.shape[0] // block_size
    quantized_blocks = quantized.reshape(num_blocks, block_size)
    dequantized_blocks = quantized_blocks * scales[:, None]

    return dequantized_blocks.reshape(-1)
