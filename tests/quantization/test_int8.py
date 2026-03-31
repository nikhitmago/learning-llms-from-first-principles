from learning_llms_from_first_principles.quantization.int8 import int8_quantize


def test_int8_quantize_roundtrip() -> None:
    """Dequantized values should be close to the original."""
    x = [1.0, -0.5, 0.3, -1.0, 0.0]
    result = int8_quantize(x)
    deq = result["dequantized"]
    assert isinstance(deq, list)
    for orig, d in zip(x, deq):
        assert abs(orig - float(d)) < 0.02


def test_int8_quantize_range() -> None:
    """Quantized values should be within [-127, 127]."""
    x = [100.0, -50.0, 0.0, 75.5, -100.0]
    result = int8_quantize(x)
    q = result["quantized"]
    assert isinstance(q, list)
    assert all(-127 <= v <= 127 for v in q)


def test_int8_quantize_all_zeros() -> None:
    """All zeros should quantize cleanly with scale=1.0."""
    result = int8_quantize([0.0, 0.0, 0.0])
    assert result["scale"] == 1.0
    assert result["quantized"] == [0, 0, 0]
