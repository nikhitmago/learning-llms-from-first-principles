import torch

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.models.gpt import GPTModel


def test_gpt_model_output_shape() -> None:
    cfg = {
        "vocab_size": 100,
        "context_len": 10,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)

    # Simulating a batch of 2 sequences of length 4
    batch = torch.randint(0, 100, (2, 4))
    logits = model(batch)

    # Expected shape: (batch_size, seq_len, vocab_size)
    assert logits.shape == (2, 4, 100)


def test_gpt_124m_model_parameter_count() -> None:
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 163009536
