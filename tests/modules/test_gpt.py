import torch

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.modules.gpt import GPTModel


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


def test_gpt_model_kv_cache_correctness() -> None:
    """Verify GPT model cached decode matches full-sequence forward."""
    torch.manual_seed(42)
    cfg = {
        "vocab_size": 100,
        "context_len": 32,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    model.eval()

    prompt = torch.randint(0, 100, (1, 5))
    new_tokens = [torch.randint(0, 100, (1, 1)) for _ in range(3)]

    full_seq = torch.cat([prompt] + new_tokens, dim=1)
    with torch.no_grad():
        expected = model(full_seq, use_kv_cache=False)

    model.reset_kv_cache_gpt()
    with torch.no_grad():
        prefill_out = model(prompt, use_kv_cache=True)
        decode_outs = []
        for tok in new_tokens:
            decode_outs.append(model(tok, use_kv_cache=True))

    cached_out = torch.cat([prefill_out] + decode_outs, dim=1)
    assert torch.allclose(expected, cached_out, atol=1e-5)
