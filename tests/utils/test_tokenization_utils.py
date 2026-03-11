import tiktoken
import torch

from learning_llms_from_first_principles.utils.tokenization_utils import (
    text_to_token_ids,
    token_ids_to_text,
)


def test_text_to_token_ids() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, world!"
    token_ids = text_to_token_ids(text, tokenizer)

    # Expected shape: (batch_size, seq_len)
    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.dim() == 2
    assert token_ids.shape[0] == 1  # Batch dimension

    # Check if the content is correct (e.g. at least 1 token inside)
    assert token_ids.shape[1] > 0


def test_token_ids_to_text() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    original_text = "Testing tokenization backwards."

    # Forward pass
    token_ids = text_to_token_ids(original_text, tokenizer)

    # Backward pass
    decoded_text = token_ids_to_text(token_ids, tokenizer)

    # The decoded text should exactly match the original text
    assert decoded_text == original_text
