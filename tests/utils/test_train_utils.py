from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from learning_llms_from_first_principles.utils.train_utils import (
    calc_loss_batch,
    calc_loss_loader,
    train_model_v1,
)


class DummyDataset(Dataset):
    def __init__(self, seq_len: int = 4, vocab_size: int = 10) -> None:
        self.x = torch.randint(0, vocab_size, (20, seq_len))
        self.y = torch.randint(0, vocab_size, (20, seq_len))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 10) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.out_head = nn.Linear(16, vocab_size)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.out_head(self.embedding(x))


def test_calc_loss_batch() -> None:
    vocab_size = 10
    model = DummyModel(vocab_size)
    device = torch.device("cpu")
    input_batch = torch.randint(0, vocab_size, (2, 4))
    target_batch = torch.randint(0, vocab_size, (2, 4))

    loss = calc_loss_batch(input_batch, target_batch, model, device)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_calc_loss_loader() -> None:
    vocab_size = 10
    model = DummyModel(vocab_size)
    device = torch.device("cpu")
    dataset = DummyDataset(vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=2)

    loss = calc_loss_loader(loader, model, device, num_batches=2)
    assert isinstance(loss, float)
    assert loss > 0


def test_train_model_v1() -> None:
    vocab_size = 50257  # Match GPT_CONFIG_124M vocab size for tokenizer compatibility if needed
    model = DummyModel(vocab_size)
    device = torch.device("cpu")
    dataset = DummyDataset(vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    class DummyTokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1, 2, 3]

        def decode(self, ids: list[int]) -> str:
            return "dummy text"

    # Mocking token_ids_to_text if necessary, but we can just use a dummy tokenizer
    # that has the attribute expected by train_model_v1 calls to token_ids_to_text

    # Let's just run for 1 epoch, eval_freq 10 (so no eval output during loop)
    # The reporting section at the end might fail if tokenizer is not compatible with tiktoken
    # train_utils.py uses token_ids_to_text(prefill_ids, tokenizer)

    tokenizer = DummyTokenizer()

    model, train_losses, val_losses, lrs = train_model_v1(
        model, loader, loader, optimizer, device, num_epochs=1, eval_freq=6, tokenizer=tokenizer
    )

    assert isinstance(model, nn.Module)
    assert len(train_losses) == 1  # Evaluation happens at Step 6
    assert len(val_losses) == 1


def test_train_model_warmup() -> None:
    vocab_size = 10
    model = DummyModel(vocab_size)
    device = torch.device("cpu")
    dataset = DummyDataset(vocab_size=vocab_size)
    # 20 samples / 2 batch_size = 10 batches per epoch
    loader = DataLoader(dataset, batch_size=2)

    peak_lr = 0.01
    warmup_min_lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)

    class DummyTokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1]

        def decode(self, ids: list[int]) -> str:
            return ""

    # Total steps = 2 epochs * 10 batches/epoch = 20 steps
    # Warmup steps = 20 * 0.5 = 10 steps
    decay_floor_lr = 1e-4
    model, train_losses, val_losses, lrs = train_model_v1(
        model,
        loader,
        loader,
        optimizer,
        device,
        num_epochs=2,
        eval_freq=100,
        tokenizer=DummyTokenizer(),
        warmup_ratio=0.5,
        warmup_min_lr=warmup_min_lr,
        decay_floor_lr=decay_floor_lr,
        max_norm=1.0,
    )

    # Check first step (global_step 0)
    assert lrs[0] == pytest.approx(warmup_min_lr)

    # Check midpoint of warmup (step 5)
    # increment = (0.01 - 0.001) / 10 = 0.0009
    # step 5 = 0.001 + 5 * 0.0009 = 0.0055
    assert lrs[5] == pytest.approx(0.0055)

    # Check end of warmup / start of decay (step 10)
    # progress = (10-10)/(20-10) = 0 -> cos(0)=1 -> lr = peak_lr
    assert lrs[10] == pytest.approx(peak_lr)

    # Check midpoint of decay (step 15)
    # progress = (15-10)/(20-10) = 0.5 -> cos(pi/2)=0 -> lr = decay_floor + (peak-decay)*0.5
    expected_mid_decay = decay_floor_lr + (peak_lr - decay_floor_lr) * 0.5
    assert lrs[15] == pytest.approx(expected_mid_decay)

    # Check near end of decay (step 19)
    # Should be decreasing towards decay_floor_lr
    assert lrs[19] < lrs[18]
    assert lrs[19] > decay_floor_lr

    # Verify total number of recorded LRs
    assert len(lrs) == 20


@pytest.mark.parametrize("max_norm", [0.5, 1.0, 5.0])
def test_train_model_grad_clipping(max_norm: float) -> None:
    vocab_size = 10
    model = DummyModel(vocab_size)
    device = torch.device("cpu")
    dataset = DummyDataset(vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    class DummyTokenizer:
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1]

        def decode(self, ids: list[int]) -> str:
            return ""

    with pytest.importorskip("unittest.mock").patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        train_model_v1(
            model,
            loader,
            loader,
            optimizer,
            device,
            num_epochs=1,
            eval_freq=100,
            tokenizer=DummyTokenizer(),
            max_norm=max_norm,
        )

        # Ensure it was called for every batch (10 batches in DummyDataset with bs=2)
        assert mock_clip.call_count == 10
        # Check that it was called with the correct max_norm
        # The first argument is model.parameters(), second is max_norm
        args, kwargs = mock_clip.call_args
        assert kwargs["max_norm"] == max_norm
