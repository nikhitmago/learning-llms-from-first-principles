import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from learning_llms_from_first_principles.utils.classify_utils import (
    calc_accuracy_loader,
    calc_loss_batch_classify,
    calc_loss_loader_classify,
    train_classifier,
)


class ToyClassifier(nn.Module):
    """A tiny 2-class classifier for testing."""

    def __init__(self, vocab_size: int = 50257, emb_dim: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.out_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        return emb.unsqueeze(0).expand(x.shape[0], -1, -1) if False else self.out_head(emb)


def _make_loader(batch_size: int = 2, seq_len: int = 5, num_batches: int = 3) -> DataLoader:
    """Return a DataLoader with random integer inputs and binary labels."""
    inputs = torch.randint(0, 50257, (batch_size * num_batches, seq_len))
    labels = torch.randint(0, 2, (batch_size * num_batches,))
    dataset = list(zip(inputs, labels))
    return DataLoader(dataset, batch_size=batch_size)  # type: ignore[arg-type]


def test_calc_loss_batch_classify_returns_scalar() -> None:
    model = ToyClassifier()
    device = torch.device("cpu")
    inputs = torch.randint(0, 50257, (2, 5))
    labels = torch.randint(0, 2, (2,))

    loss = calc_loss_batch_classify(inputs, labels, model, device)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_calc_loss_batch_classify_gradients_flow() -> None:
    model = ToyClassifier()
    device = torch.device("cpu")
    inputs = torch.randint(0, 50257, (2, 5))
    labels = torch.randint(0, 2, (2,))

    loss = calc_loss_batch_classify(inputs, labels, model, device)
    loss.backward()

    assert any(p.grad is not None for p in model.parameters())


def test_calc_loss_loader_classify_empty_loader() -> None:
    model = ToyClassifier()
    device = torch.device("cpu")
    empty_loader: DataLoader = DataLoader([])  # type: ignore[arg-type]

    result = calc_loss_loader_classify(empty_loader, model, device)
    assert result != result


def test_calc_loss_loader_classify_num_batches() -> None:
    model = ToyClassifier()
    device = torch.device("cpu")
    loader = _make_loader()

    result = calc_loss_loader_classify(loader, model, device, num_batches=2)
    assert isinstance(result, float)
    assert result == result


def test_calc_accuracy_loader_range() -> None:
    model = ToyClassifier()
    device = torch.device("cpu")
    loader = _make_loader()

    acc = calc_accuracy_loader(loader, model, device)
    assert 0.0 <= acc <= 1.0


def test_train_classifier_returns_correct_types() -> None:
    model = ToyClassifier()
    device = torch.device("cpu")
    loader = _make_loader(batch_size=2, seq_len=5, num_batches=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trained_model, train_losses, val_losses, train_accs, val_accs = train_classifier(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        eval_freq=2,
        eval_iter=1,
    )

    assert isinstance(trained_model, ToyClassifier)
    assert isinstance(train_losses, list)
    assert isinstance(val_losses, list)
    assert isinstance(train_accs, list)
    assert isinstance(val_accs, list)
    assert len(train_accs) == 1
    assert len(val_accs) == 1
