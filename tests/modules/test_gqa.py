import torch

from learning_llms_from_first_principles.modules.attention import grouped_query_attention


def test_gqa_matches_mha_when_equal_heads() -> None:
    """When num_kv_heads == num_heads, GQA degenerates to standard MHA."""
    torch.manual_seed(42)
    batch, seq, num_heads, head_dim = 2, 5, 4, 8
    d = num_heads * head_dim

    Q = torch.rand(batch, seq, d)
    K = torch.rand(batch, seq, d)
    V = torch.rand(batch, seq, d)

    O_gqa = grouped_query_attention(Q, K, V, num_heads=num_heads, num_kv_heads=num_heads)

    # Manual standard MHA for reference
    scale = head_dim**-0.5
    q = Q.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)
    k = K.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)
    v = V.reshape(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    O_mha = (weights @ v).permute(0, 2, 1, 3).reshape(batch, seq, d)

    assert torch.allclose(O_gqa, O_mha, atol=1e-6)


def test_gqa_output_shape() -> None:
    """Output shape should be (batch, seq, num_heads * head_dim) for various group configs."""
    torch.manual_seed(0)
    batch, seq, head_dim = 2, 6, 8

    for num_heads, num_kv_heads in [(8, 2), (8, 4), (8, 1), (4, 4)]:
        Q = torch.rand(batch, seq, num_heads * head_dim)
        K = torch.rand(batch, seq, num_kv_heads * head_dim)
        V = torch.rand(batch, seq, num_kv_heads * head_dim)

        result = grouped_query_attention(Q, K, V, num_heads=num_heads, num_kv_heads=num_kv_heads)
        assert result.shape == (
            batch,
            seq,
            num_heads * head_dim,
        ), f"Failed for {num_heads}/{num_kv_heads}"


def test_gqa_mqa_single_kv_head() -> None:
    """num_kv_heads=1 is Multi-Query Attention — all query heads share one KV head."""
    torch.manual_seed(7)
    batch, seq, num_heads, head_dim = 1, 4, 4, 8

    Q = torch.rand(batch, seq, num_heads * head_dim)
    K = torch.rand(batch, seq, 1 * head_dim)
    V = torch.rand(batch, seq, 1 * head_dim)

    O = grouped_query_attention(Q, K, V, num_heads=num_heads, num_kv_heads=1)  # noqa: E741
    assert O.shape == (batch, seq, num_heads * head_dim)
    assert not torch.isnan(O).any()
