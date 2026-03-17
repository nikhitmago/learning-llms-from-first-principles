from typing import Any

import tiktoken
from torch.utils.data import DataLoader

from .datasets import GPTDatasetV1, SpamDataset


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: Any = None,
) -> DataLoader:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def create_classify_dataloader(
    csv_file: str,
    split_name: str,
    tokenizer: Any,
    max_length: int | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, SpamDataset]:
    """Create a DataLoader for the spam/ham classification task.

    Args:
        csv_file: Path to the combined CSV (with ``Text``, ``Label``, ``split`` columns).
        split_name: Which split to load — ``"train"``, ``"val"``, or ``"test"``.
        tokenizer: A tiktoken-compatible tokenizer.
        max_length: Sequence length cap. Pass ``None`` to auto-detect from the split.
            For val/test, pass ``train_dataset.max_length`` for consistent padding.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.
        drop_last: Drop the last incomplete batch.
        num_workers: Number of DataLoader worker processes.

    Returns:
        ``(dataloader, dataset)`` — the dataset is returned so callers can
        read ``dataset.max_length`` and pass it to subsequent splits.
    """
    dataset = SpamDataset(
        csv_file=csv_file,
        split_name=split_name,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader, dataset
