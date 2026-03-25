import torch
import torch.nn.functional as F

from learning_llms_from_first_principles.modules.attention import flash_attention_v1


def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Reference standard attention for comparison."""
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = (Q @ K.T) * scale
    weights = F.softmax(scores, dim=1)
    return weights @ V


def test_matches_standard_attention() -> None:
    """Flash attention output should match standard attention."""
    torch.manual_seed(42)
    Q, K, V = torch.rand(6, 4), torch.rand(6, 4), torch.rand(6, 4)
    O_flash = flash_attention_v1(Q, K, V, block_size=2)
    O_standard = standard_attention(Q, K, V)
    assert torch.allclose(O_flash, O_standard, atol=1e-6)


def test_different_block_sizes() -> None:
    """Should produce same result regardless of block size, including non-divisible ones."""
    torch.manual_seed(123)
    Q, K, V = torch.rand(7, 4), torch.rand(7, 4), torch.rand(7, 4)
    O_expected = standard_attention(Q, K, V)
    for bs in [1, 2, 3, 7, 10]:
        O_flash = flash_attention_v1(Q, K, V, block_size=bs)
        assert torch.allclose(O_flash, O_expected, atol=1e-6), f"Failed for block_size={bs}"


def test_single_token() -> None:
    """seq_len=1 edge case — output should just be V."""
    torch.manual_seed(42)
    Q, K, V = torch.rand(1, 4), torch.rand(1, 4), torch.rand(1, 4)
    O = flash_attention_v1(Q, K, V, block_size=1)  # noqa: E741
    assert torch.allclose(O, V, atol=1e-6)
