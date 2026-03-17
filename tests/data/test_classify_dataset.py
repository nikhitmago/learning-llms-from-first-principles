from pathlib import Path

import pandas as pd
import pytest
import tiktoken
import torch

from learning_llms_from_first_principles.data.datasets import (
    SpamDataset,
    Split,
)


def _make_csv(tmp_path: Path) -> str:
    """Write a tiny combined CSV and return its path as a string."""
    rows = [
        {"Text": "Hello this is ham", "Label": 0, "split": Split.TRAIN},
        {"Text": "Win a free prize now", "Label": 1, "split": Split.TRAIN},
        {"Text": "How are you doing", "Label": 0, "split": Split.TRAIN},
        {"Text": "Congratulations you won", "Label": 1, "split": Split.VAL},
        {"Text": "Call me later", "Label": 0, "split": Split.TEST},
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "spam_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    return _make_csv(tmp_path)


def test_spam_dataset_train_split(csv_path: str, tokenizer: tiktoken.Encoding) -> None:
    dataset = SpamDataset(csv_file=csv_path, split_name=Split.TRAIN, tokenizer=tokenizer)

    assert len(dataset) == 3


def test_spam_dataset_returns_correct_types(csv_path: str, tokenizer: tiktoken.Encoding) -> None:
    dataset = SpamDataset(csv_file=csv_path, split_name=Split.TRAIN, tokenizer=tokenizer)
    x, y = dataset[0]

    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_spam_dataset_padding(csv_path: str, tokenizer: tiktoken.Encoding) -> None:
    dataset = SpamDataset(csv_file=csv_path, split_name=Split.TRAIN, tokenizer=tokenizer)

    lengths = {dataset[i][0].shape[0] for i in range(len(dataset))}
    assert len(lengths) == 1


def test_spam_dataset_truncation(csv_path: str, tokenizer: tiktoken.Encoding) -> None:
    dataset = SpamDataset(
        csv_file=csv_path, split_name=Split.TRAIN, tokenizer=tokenizer, max_length=3
    )
    x, _ = dataset[0]
    assert x.shape[0] == 3


def test_spam_dataset_val_and_test_splits(csv_path: str, tokenizer: tiktoken.Encoding) -> None:
    train_ds = SpamDataset(csv_file=csv_path, split_name=Split.TRAIN, tokenizer=tokenizer)
    val_ds = SpamDataset(
        csv_file=csv_path,
        split_name=Split.VAL,
        tokenizer=tokenizer,
        max_length=train_ds.max_length,
    )
    test_ds = SpamDataset(
        csv_file=csv_path,
        split_name=Split.TEST,
        tokenizer=tokenizer,
        max_length=train_ds.max_length,
    )

    assert len(val_ds) == 1
    assert len(test_ds) == 1

    assert val_ds[0][0].shape[0] == train_ds.max_length
    assert test_ds[0][0].shape[0] == train_ds.max_length


def test_spam_dataset_missing_split_raises(csv_path: str, tokenizer: tiktoken.Encoding) -> None:
    with pytest.raises(ValueError, match="No rows found for split"):
        SpamDataset(csv_file=csv_path, split_name="nonexistent", tokenizer=tokenizer)
