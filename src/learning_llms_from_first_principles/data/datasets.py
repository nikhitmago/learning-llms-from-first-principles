import logging
from enum import Enum
from typing import Any, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from learning_llms_from_first_principles.utils.data_utils import format_instruct_prompt

logger = logging.getLogger(__name__)


class Split(str, Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class GPTDatasetV1(Dataset):
    """Dataset for next-token prediction pretraining.

    Example (1 word = 1 token, max_length=4, stride=2):

        Text: "The cat sat on the mat"
        Tokens: [1, 2, 3, 4, 5, 6]

        Chunk 1:  X = [1, 2, 3, 4]    Y = [2, 3, 4, 5]
        Chunk 2:  X = [3, 4, 5, 6]    Y = [4, 5, 6, EOS]

        At loss time for Chunk 1, model produces logits at EVERY position.
        vocab = {1:"The", 2:"cat", 3:"sat", 4:"on", 5:"the", 6:"mat"}

        model(X) produces (seq_len=4, vocab_size=6):
                    1     2     3     4     5     6
            pos 0: [0.05, 0.80, 0.05, 0.03, 0.04, 0.03]  ← predict after "The"       ✓ target=2("cat")
            pos 1: [0.02, 0.03, 0.85, 0.04, 0.03, 0.03]  ← predict after "The cat"   ✓ target=3("sat")
            pos 2: [0.03, 0.02, 0.04, 0.82, 0.05, 0.04]  ← predict after "...sat"    ✓ target=4("on")
            pos 3: [0.04, 0.03, 0.03, 0.02, 0.83, 0.05]  ← predict after "...on"     ✓ target=5("the")
        targets: [2, 3, 4, 5]

        ALL positions are used for loss — every token teaches the model.
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    """

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

    Example:

        Text: "Win a free prize now"  →  tokens: [12, 45, 78, 91, 23] (padded to max_length)
        Label: 1 (spam)

        At loss time, the model produces logits at every position (same as pretraining),
        but only the last row is used — it has attended to the full sequence:

        model(tokens) produces:
            pos 0: [0.1, 0.9]   ← "Win" context only        (discarded)
            pos 1: [0.3, 0.7]   ← "Win a" context            (discarded)
            pos 2: [0.2, 0.8]   ← "Win a free" context       (discarded)
            pos 3: [0.4, 0.6]   ← "Win a free prize" context (discarded)
            pos 4: [0.2, 0.8]   ← full sequence context      ✓ USED
                  ham↑  spam↑

        target: 1 (spam)
        loss = cross_entropy([0.2, 0.8], 1)

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


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning.

    Each entry is an Alpaca-style dict with ``instruction``, ``input``, and ``output``.
    Pre-tokenizes and prepares input/target pairs for each entry.

    Per-item logic (EOS append, input/target shift, -100 masking) lives here.
    Batch-level padding lives in instruct_collate_fn (data_utils.py) because
    padding to the global max length would waste compute on shorter batches —
    the collate_fn pads to the batch max instead.

    Example (1 word = 1 token for clarity):

    Entry: {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}

    Full text: "### Instruction: Translate to French ### Input: Hello ### Response: Bonjour"

    Tokenized: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ###  Ins  Tra  to  Fre  ###  Inp  Hel  ###  Res  Bon

    X (inputs):   [1,    2,    3,    4,    5,    6,    7,    8,    9,    10   ]
    Y (targets):  [2,    3,    4,    5,    6,    7,    8,    9,    10,   11   ]

    Y masked:     [-100, -100, -100, -100, -100, -100, -100, -100, -100, 11  ]
                   |_________________ prompt (ignored by loss) ________||loss|

    Why -100? PyTorch's cross_entropy has ignore_index=-100 by default.
    Any target token set to -100 is excluded from the loss calculation entirely.
    This means the model still processes the full prompt in the forward pass
    (it needs that context to generate the response), but gradients only flow
    from the response tokens. The prompt tokens contribute zero to the loss,
    so the model learns to generate good responses, not to memorize the prompt.

    Example proving -100 is ignored by cross_entropy:

        logits_2ex = torch.tensor([
            [-1.0, 1.0],
            [-0.5, 1.5]
        ])
        targets_2ex = torch.tensor([
            0,
            1
        ])
        loss_2ex = cross_entropy(logits_2ex, targets_2ex)  # 0.8619

        logits_3ex = torch.tensor([
            [-1.0, 1.0],
            [-0.5, 1.5],
            [-0.5, 1.5]
        ])
        targets_3ex = torch.tensor([
            0,
            1,
            -100  # 3rd example masked
        ])
        loss_3ex = cross_entropy(logits_3ex, targets_3ex)  # 0.8619 — same!

        The 3rd example with target=-100 contributes nothing to the loss.

    Args:
        data: List of instruction dicts.
        tokenizer: A tiktoken-compatible tokenizer.
        pad_token_id: Token ID used for end-of-sequence marker.
        ignore_index: Value used to mask positions the loss should ignore.
    """

    def __init__(
        self,
        data: List[dict[str, str]],
        tokenizer: Any,
        pad_token_id: int = 50256,
        ignore_index: int = -100,
    ) -> None:
        self.data = data

        self.input_ids: List[List[int]] = []
        self.target_ids: List[List[int]] = []

        for entry in data:
            instruction_plus_input = format_instruct_prompt(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Tokenize prompt and response separately to know the boundary
            prompt_tokens = tokenizer.encode(
                instruction_plus_input + response_text.split(entry["output"])[0]
            )
            response_tokens = tokenizer.encode(entry["output"])

            # Full sequence = prompt + response + EOS
            full_tokens = prompt_tokens + response_tokens + [pad_token_id]

            # Shift by 1: input is everything except last, target is everything except first
            input_ids = full_tokens[:-1]
            target_ids = full_tokens[1:]

            # Mask prompt positions with ignore_index — only response tokens contribute to loss
            prompt_len = len(prompt_tokens) - 1  # -1 because of the shift
            target_ids = [ignore_index] * prompt_len + target_ids[prompt_len:]

            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)

    def __getitem__(self, index: int) -> dict[str, List[int]]:
        return {"input_ids": self.input_ids[index], "target_ids": self.target_ids[index]}

    def __len__(self) -> int:
        return len(self.data)
