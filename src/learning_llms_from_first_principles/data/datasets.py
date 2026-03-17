import logging
from enum import Enum
from typing import Any, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class Split(str, Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int) -> None:
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


class SpamDataset(Dataset):
    """Dataset for spam/ham classification fine-tuning.

    Reads from a single CSV that has columns ``Text``, ``Label``, and ``split``.
    Only rows whose ``split`` column equals ``split_name`` are used.

    The ``Label`` column must be integer-encoded (0 = ham, 1 = spam).

    Args:
        csv_file: Path to the combined CSV file.
        split_name: One of ``"train"``, ``"val"``, or ``"test"``.
        tokenizer: A tiktoken-compatible tokenizer.
        max_length: Maximum token length. If ``None``, uses the longest sequence in this split.
            Pass ``train_dataset.max_length`` to val/test splits to ensure consistent padding.
        pad_token_id: Token ID used to pad sequences (default 50256 is the GPT-2 EOS token).
    """

    def __init__(
        self,
        csv_file: str,
        split_name: str,
        tokenizer: Any,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ) -> None:
        df = pd.read_csv(csv_file)
        self.data = df[df["split"] == split_name].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(
                f"No rows found for split '{split_name}' in {csv_file}. "
                f"Available splits: {df['split'].unique().tolist()}"
            )

        logger.info(f"Loaded {len(self.data)} rows for split='{split_name}' from {csv_file}")

        # Pre-tokenize all texts
        self.encoded_texts: List[List[int]] = [tokenizer.encode(text) for text in self.data["Text"]]

        # Determine max_length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences that exceed max_length
            self.encoded_texts = [enc[: self.max_length] for enc in self.encoded_texts]

        # Pad sequences to max_length
        self.encoded_texts = [
            enc + [pad_token_id] * (self.max_length - len(enc)) for enc in self.encoded_texts
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoded_texts[idx]
        label = int(self.data.iloc[idx]["Label"])
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def _longest_encoded_length(self) -> int:
        return max(len(enc) for enc in self.encoded_texts)
