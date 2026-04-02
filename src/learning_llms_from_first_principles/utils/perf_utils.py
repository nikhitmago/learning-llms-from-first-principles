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

    Note on FLOP counting:
        For a matmul (M, K) @ (K, N), we count 2*K*M*N FLOPs. The 2K comes from
        the dot product of two length-K vectors: K multiplications + (K-1) additions
        = 2K-1 ops per output element, approximated as 2K for large K. There are
        M*N such dot products (one per output element), giving 2*K*M*N total.
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


def compute_arithmetic_intensity(
    flops: float,
    bytes_accessed: float,
    peak_performance: float,
    peak_bandwidth: float,
) -> dict[str, float | str]:
    """
    Analyze a computational kernel using the Roofline Model.

    The roofline model determines whether a kernel is compute-bound or memory-bound
    by comparing its arithmetic intensity (FLOPs/byte) to the hardware's ridge point.

    In LLM inference this matters a lot:
    - Prefill (processing the prompt) is typically compute-bound: large batch of tokens
      processed in parallel, high arithmetic intensity, limited by peak FLOP/s.
    - Decode (generating tokens one at a time) is typically memory-bound: each step
      processes a single token, low arithmetic intensity, limited by memory bandwidth
      (loading all the KV cache and model weights for just one token's worth of compute).

    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    arithmetic_intensity = flops / bytes_accessed
    ridge_point = peak_performance / peak_bandwidth

    if arithmetic_intensity >= ridge_point:
        bottleneck = "compute-bound"
        achieved_performance = float(peak_performance)
    else:
        bottleneck = "memory-bound"
        achieved_performance = arithmetic_intensity * peak_bandwidth

    utilization = (achieved_performance / peak_performance) * 100

    return {
        "arithmetic_intensity": round(arithmetic_intensity, 4),
        "ridge_point": round(ridge_point, 4),
        "bottleneck": bottleneck,
        "achieved_performance": round(achieved_performance, 4),
        "utilization_percent": round(utilization, 4),
    }
