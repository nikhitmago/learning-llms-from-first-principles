import torch

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.modules.transformer import TransformerBlock


def test_transformer_block_output_shape() -> None:
    torch.manual_seed(123)

    x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    # Initial shape check
    assert x.shape == (2, 4, 768)
    assert output.shape == (2, 4, 768)


def test_transformer_block_uniqueness() -> None:
    """Verify that multiple blocks have different weights even with same config"""
    torch.manual_seed(123)
    block1 = TransformerBlock(GPT_CONFIG_124M)
    block2 = TransformerBlock(GPT_CONFIG_124M)

    # Weights should be initialized differently
    assert not torch.equal(
        block1.multi_head_attention.W_q.weight, block2.multi_head_attention.W_q.weight
    )


def test_transformer_block_kv_cache_correctness() -> None:
    """Verify transformer block cached decode matches full-sequence forward."""
    torch.manual_seed(42)
    cfg = {
        "vocab_size": 100,
        "context_len": 32,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    block = TransformerBlock(cfg)
    block.eval()

    prompt = torch.randn(1, 5, 16)
    new_tokens = [torch.randn(1, 1, 16) for _ in range(3)]

    full_seq = torch.cat([prompt] + new_tokens, dim=1)
    with torch.no_grad():
        expected = block(full_seq, use_kv_cache=False)

    block.multi_head_attention.reset_kv_cache()
    with torch.no_grad():
        prefill_out = block(prompt, use_kv_cache=True)
        decode_outs = []
        for tok in new_tokens:
            decode_outs.append(block(tok, use_kv_cache=True))

    cached_out = torch.cat([prefill_out] + decode_outs, dim=1)
    assert torch.allclose(expected, cached_out, atol=1e-5)
