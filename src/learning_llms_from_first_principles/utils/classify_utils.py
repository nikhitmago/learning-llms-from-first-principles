import logging
from typing import TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ModelT = TypeVar("ModelT", bound=nn.Module)

logger = logging.getLogger(__name__)


def calc_loss_batch_classify(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute cross-entropy loss for classification.

    Uses only the **last token's** logits, which is where the model
    accumulates context for the sequence-level label prediction.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)[
        :, -1, :
    ]  # take for the last token only, this is not the same as for next token prediction
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader_classify(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Average classification loss over ``num_batches`` batches."""
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    num_batches = (
        min(num_batches, len(data_loader)) if num_batches is not None else len(data_loader)
    )

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch_classify(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Compute classification accuracy over ``num_batches`` batches."""
    model.eval()
    correct_predictions = 0
    num_examples = 0

    num_batches = (
        min(num_batches, len(data_loader)) if num_batches is not None else len(data_loader)
    )

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)

        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


def train_classifier(
    model: ModelT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
) -> tuple[ModelT, list[float], list[float], list[float], list[float]]:
    """Fine-tune a GPT model for sequence classification.

    Only the last transformer block and the final layer norm have their
    gradients enabled (frozen elsewhere). The caller is responsible for
    replacing ``model.out_head`` with a classification head and freezing
    / unfreezing the appropriate layers before calling this function.

    Args:
        model: The GPT model with a classification ``out_head``.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        optimizer: Optimizer (e.g. AdamW).
        device: Target device.
        num_epochs: Total number of training epochs.
        eval_freq: Evaluate every ``eval_freq`` global steps.
        eval_iter: Number of batches to use for in-loop loss evaluation.

    Returns:
        ``(model, train_losses, val_losses, train_accs, val_accs)``
    """
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch_classify(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader_classify(
                        train_loader, model, device, num_batches=eval_iter
                    )
                    val_loss = calc_loss_loader_classify(
                        val_loader, model, device, num_batches=eval_iter
                    )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.info(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
                model.train()

        # --- Epoch-end accuracy ---
        model.eval()
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        logger.info(
            f"Epoch {epoch + 1} | "
            f"Train acc: {train_acc * 100:.2f}%  Val acc: {val_acc * 100:.2f}%"
        )
        logger.info("=" * 50)

    return model, train_losses, val_losses, train_accs, val_accs
