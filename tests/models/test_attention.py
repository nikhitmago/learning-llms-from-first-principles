import torch

from models.attention import SelfAttention


def test_self_attention() -> None:
    torch.manual_seed(789)
    d_emb, d_attn = 3, 2

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],
        ]  # step     (x^6)
    )

    sa = SelfAttention(d_emb, d_attn)
    output = sa(inputs)

    expected = torch.tensor(
        [
            [-0.0739, 0.0713],
            [-0.0748, 0.0703],
            [-0.0749, 0.0702],
            [-0.0760, 0.0685],
            [-0.0763, 0.0679],
            [-0.0754, 0.0693],
        ]
    )

    assert output.shape == (6, 2)
    # Check if values match with a small tolerance for floating point differences
    assert torch.allclose(output, expected, atol=1e-4)
