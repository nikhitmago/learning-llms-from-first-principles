import torch

from learning_llms_from_first_principles.inference.generate import generate_tokens
from learning_llms_from_first_principles.modules.gpt import GPTModel


def test_generate_tokens() -> None:
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
    model.eval()  # disable dropout

    # Batch of 2, starting with 4 tokens each
    idx = torch.randint(0, 100, (2, 4))

    # Generate 5 new tokens
    out = generate_tokens(model, idx, max_new_tokens=5, context_size=10, temperature=0.0)

    # Expected shape: (batch_size, original_tokens + max_new_tokens)
    assert out.shape == (2, 9)


def test_generate_tokens_with_temp() -> None:
    model = GPTModel(
        {
            "vocab_size": 10,
            "context_len": 5,
            "emb_dim": 8,
            "n_heads": 1,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    )
    idx = torch.randint(0, 10, (1, 3))
    out = generate_tokens(model, idx, max_new_tokens=2, context_size=5, temperature=1.5)
    assert out.shape == (1, 5)


def test_generate_tokens_with_top_k() -> None:
    model = GPTModel(
        {
            "vocab_size": 20,
            "context_len": 5,
            "emb_dim": 8,
            "n_heads": 1,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    )
    idx = torch.randint(0, 20, (1, 3))
    # Test with top_k=5
    out = generate_tokens(model, idx, max_new_tokens=2, context_size=5, temperature=1.0, top_k=5)
    assert out.shape == (1, 5)


def test_generate_tokens_with_top_p() -> None:
    model = GPTModel(
        {
            "vocab_size": 20,
            "context_len": 5,
            "emb_dim": 8,
            "n_heads": 1,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    )
    idx = torch.randint(0, 20, (1, 3))
    # Test with top_p=0.5
    out = generate_tokens(
        model, idx, max_new_tokens=2, context_size=5, temperature=1.0, top_p=0.5, top_k=10
    )
    assert out.shape == (1, 5)


def test_classify_text() -> None:
    import tiktoken

    from learning_llms_from_first_principles.inference.generate import classify_text

    cfg = {
        "vocab_size": 50257,
        "context_len": 128,
        "emb_dim": 32,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    model.out_head = torch.nn.Linear(32, 2)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    result = classify_text("Hello world", model, tokenizer)

    assert result in ("ham", "spam")


def test_classify_text_custom_labels() -> None:
    import tiktoken

    from learning_llms_from_first_principles.inference.generate import classify_text

    cfg = {
        "vocab_size": 50257,
        "context_len": 128,
        "emb_dim": 32,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    model.out_head = torch.nn.Linear(32, 3)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    labels = {0: "positive", 1: "negative", 2: "neutral"}
    result = classify_text("Great product", model, tokenizer, label_map=labels)

    assert result in ("positive", "negative", "neutral")


def test_generate_text() -> None:
    import tiktoken

    from learning_llms_from_first_principles.inference.generate import generate_text

    cfg = {
        "vocab_size": 50257,
        "context_len": 32,
        "emb_dim": 32,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    result = generate_text("Hello", model, tokenizer, max_new_tokens=5, context_size=32)

    assert isinstance(result, str)
    assert len(result) > len("Hello")


def test_generate_text_with_sampling() -> None:
    import tiktoken

    from learning_llms_from_first_principles.inference.generate import generate_text

    cfg = {
        "vocab_size": 50257,
        "context_len": 32,
        "emb_dim": 32,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(cfg)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    result = generate_text(
        "Hello", model, tokenizer, max_new_tokens=5, context_size=32, temperature=1.0, top_k=10
    )

    assert isinstance(result, str)
    assert len(result) > len("Hello")


def test_generate_tokens_kv_cache_correctness() -> None:
    """KV cache should produce identical output to non-cached greedy generation."""
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

    idx = torch.randint(0, 100, (1, 5))

    out_no_cache = generate_tokens(
        model, idx, max_new_tokens=10, context_size=32, temperature=0.0, use_kv_cache=False
    )

    model.reset_kv_cache_gpt()
    out_with_cache = generate_tokens(
        model, idx, max_new_tokens=10, context_size=32, temperature=0.0, use_kv_cache=True
    )

    assert torch.equal(out_no_cache, out_with_cache)
