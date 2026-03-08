import os

import tiktoken
import torch

from data.dataloader import create_dataloader_v1


def test_e2e_data_pipeline() -> None:
    # Path to data
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "src", "data", "the-verdict.txt")

    assert os.path.exists(data_path), f"Data file not found at {data_path}"

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    batch_size = 4
    max_length = 8
    stride = 8

    dataloader = create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=False
    )

    # Check total samples
    assert len(dataloader.dataset) > 0  # type: ignore

    # Verify first batch
    x, y = next(iter(dataloader))

    assert x.shape == (batch_size, max_length)
    assert y.shape == (batch_size, max_length)

    # Shift check
    assert torch.equal(x[:, 1:], y[:, :-1])

    # Decoding check for first sample
    tokenizer = tiktoken.get_encoding("gpt2")
    input_text = tokenizer.decode(x[0].tolist())
    target_text = tokenizer.decode(y[0].tolist())

    assert len(input_text) > 0
    assert len(target_text) > 0
    # The target text usually starts with the second "token" part of input
    # but decoding might have different spacing. The logic is verified by IDs.
