import torch

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.models.transformer import TransformerBlock


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
