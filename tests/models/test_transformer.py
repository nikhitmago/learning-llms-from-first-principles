import torch

from learning_llms_from_first_principles.models.transformer import TransformerBlock


def test_transformer_block_identity() -> None:
    cfg = {"emb_dim": 768}
    block = TransformerBlock(cfg)
    x = torch.randn(2, 4, 768)
    output = block(x)
    # Verifying it currently acts as an identity layer as intended
    assert torch.equal(x, output)
