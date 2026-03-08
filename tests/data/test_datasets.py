import tiktoken
import torch

from data.datasets import GPTDatasetV1


def test_gpt_dataset_v1_initialization() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    txt = "Hello world! This is a test string for the dataset."
    max_length = 4
    stride = 2

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Verify that the dataset returns tensors of the correct shape
    assert len(dataset) > 0
    x, y = dataset[0]
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert x.shape == (max_length,)
    assert y.shape == (max_length,)


def test_gpt_dataset_v1_alignment() -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    txt = "The quick brown fox jumps over the lazy dog."
    max_length = 3
    stride = 1

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    for i in range(len(dataset)):
        x, y = dataset[i]
        # Target should be input shifted by 1
        assert torch.equal(x[1:], y[:-1])
