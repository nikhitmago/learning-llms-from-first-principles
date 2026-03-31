import torch


def per_channel_quantize(
    weight: torch.Tensor, bits: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int tensor of shape (out_channels, in_features)
        - scale_factors: float tensor of shape (out_channels,)
        - dequantized_weights: float tensor of shape (out_channels, in_features)
    """
    Q_max = (2 ** (bits - 1)) - 1
    Q_min = -1 * (2 ** (bits - 1))

    max_vals = weight.abs().max(dim=1, keepdim=True).values
    scales = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals / Q_max)

    quantized_weight = torch.round(weight / scales).to(torch.int32)
    quantized_weight = torch.clamp(quantized_weight, Q_min, Q_max)

    dequantized_weight = quantized_weight.float() * scales

    return quantized_weight, scales.squeeze(1), dequantized_weight
