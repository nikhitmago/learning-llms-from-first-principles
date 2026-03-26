from learning_llms_from_first_principles.utils.perf_utils import attention_memory_flops


def test_attention_flops_keys_present() -> None:
    """All expected keys should be in the result."""
    result = attention_memory_flops(B=1, h=1, N=4, d=8)
    expected_keys = {
        "qk_flops",
        "softmax_flops",
        "pv_flops",
        "total_flops",
        "memory_bytes",
        "arithmetic_intensity",
    }
    assert set(result.keys()) == expected_keys


def test_attention_flops_total_is_sum_of_parts() -> None:
    """total_flops should equal qk + softmax + pv."""
    result = attention_memory_flops(B=2, h=4, N=128, d=64)
    assert (
        result["total_flops"] == result["qk_flops"] + result["softmax_flops"] + result["pv_flops"]
    )


def test_attention_flops_scales_quadratically_with_seq_len() -> None:
    """Doubling N should ~4x the total FLOPs (quadratic in N)."""
    r1 = attention_memory_flops(B=1, h=1, N=64, d=32)
    r2 = attention_memory_flops(B=1, h=1, N=128, d=32)
    ratio = r2["total_flops"] / r1["total_flops"]
    assert 3.9 < ratio < 4.1
