def attention_memory_flops(
    B: int, h: int, N: int, d: int, bytes_per_element: int = 2
) -> dict[str, int | float]:
    """
    Compute memory traffic and FLOPs for standard self-attention.

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    # FLOPs
    qk_flops = 2 * B * h * d * N * N  # (B, h, N, d) x (B, h, d, N)
    softmax_flops = 5 * B * h * N * N  # 5N for softmax for N rows for batch * heads
    pv_flops = 2 * B * h * N * N * d  # (B, h, N, N) * (B, h, N, d)
    total_flops = qk_flops + softmax_flops + pv_flops

    # Memory
    qk_reads = 2 * B * h * N * d  # 2 for Q and K
    qk_writes = B * h * N * N  # (B, h, N, d) x (B, h, d, N)
    softmax_reads = qk_writes
    softmax_writes = qk_writes
    p_reads = qk_writes + (B * h * N * d)  # second one is for reading values
    p_writes = B * h * N * d  # write values
    total_read_writes = qk_reads + qk_writes + softmax_reads + softmax_writes + p_reads + p_writes
    memory_bytes = total_read_writes * bytes_per_element

    arithmetic_intensity = total_flops / memory_bytes

    return {
        "qk_flops": qk_flops,
        "softmax_flops": softmax_flops,
        "pv_flops": pv_flops,
        "total_flops": total_flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": round(arithmetic_intensity, 2),
    }
