import torch


def int8_quantize(x: list[float]) -> dict[str, list[int] | float | list[float]]:
    """
    Perform symmetric INT8 quantization on a floating-point array using PyTorch.

    Args:
        x: Input list of floating-point values

    Returns:
        Dictionary with 'quantized', 'scale', and 'dequantized' keys
    """
    arr = torch.tensor(x, dtype=torch.float32)
    max_val = arr.abs().max().item()

    if max_val == 0:
        scale = 1.0
    else:
        scale = float(max_val) / 127.0

    quantized = torch.round(arr / scale).to(torch.int8)
    dequantized = quantized.to(torch.float32) * scale

    return {
        "quantized": quantized.tolist(),
        "scale": round(scale, 6),
        "dequantized": [round(float(number), 4) for number in dequantized],
    }
