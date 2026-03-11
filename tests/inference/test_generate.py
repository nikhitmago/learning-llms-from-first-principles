import torch

from learning_llms_from_first_principles.inference.generate import generate_text_simple
from learning_llms_from_first_principles.modules.gpt import GPTModel


def test_generate_text_simple() -> None:
    cfg = {
        "vocab_size": 100,
        "context_len": 10,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    model.eval()

    # Batch of 2, starting with 4 tokens each
    idx = torch.randint(0, 100, (2, 4))

    # Generate 5 new tokens
    out = generate_text_simple(model, idx, max_new_tokens=5, context_size=10)

    # Expected shape: (batch_size, original_tokens + max_new_tokens)
    assert out.shape == (2, 9)
