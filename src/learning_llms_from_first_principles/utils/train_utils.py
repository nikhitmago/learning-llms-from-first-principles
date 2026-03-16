from __future__ import annotations

from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.inference.generate import generate_text_simple
from learning_llms_from_first_principles.utils.tokenization_utils import token_ids_to_text

ModelT = TypeVar("ModelT", bound=nn.Module)


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: torch.device
) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)  # bs, seq_len, vocab_size
    bs, seq_len, vocab_size = logits.shape

    # Flatten (bs, seq_len, vocab_size ) => (bs * seq_len, vocab_size)
    logits_flattened = logits.view(-1, vocab_size)  # [bs * seq_len, vocab_size]
    targets_flattened = target_batch.view(-1)  # [bs * seq_len]

    # Use torch to calculate the CE loss
    loss_lib = nn.CrossEntropyLoss()(logits_flattened, targets_flattened)

    # Calculate CE loss from scratch
    probs_flattened = torch.softmax(logits_flattened, dim=-1)  # [bs * seq_len, vocab_size]
    probs_values = probs_flattened[torch.arange(probs_flattened.shape[0]), targets_flattened]
    loss_ce_scratch = -torch.log(probs_values).mean()

    assert torch.allclose(loss_lib, loss_ce_scratch)

    return loss_lib


def calc_loss_loader(
    data_loader: DataLoader, model: nn.Module, device: torch.device, num_batches: int | None = None
) -> float:
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def train_model_v1(
    model: ModelT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    tokenizer: Any,
) -> tuple[ModelT, list[float], list[float]]:
    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        for i, (input_batch, target_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            running_train_loss += loss.item()
            global_step += 1

            loss.backward()
            optimizer.step()

            # Print intermediate progress based on global steps
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = calc_loss_loader(val_loader, model, device)
                train_loss = running_train_loss / (i + 1)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # --- Epoch-end reporting ---
        print("\n" + "=" * 50 + "\n")

        model.eval()
        samples_printed = 0
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(val_loader):
                if samples_printed >= 2:
                    break

                input_ids = input_batch.to(device)

                # Generate 20 new tokens
                out_ids = generate_text_simple(
                    model=model,
                    idx=input_ids,
                    max_new_tokens=20,
                    context_size=int(GPT_CONFIG_124M["context_len"]),
                )

                # Process each sample in the batch until we hit our limit of 2
                for j in range(out_ids.shape[0]):
                    if samples_printed >= 2:
                        break

                    # Split input (prefill) from generated (decode) tokens
                    # Slicing the last 20 tokens as they are the ones generated
                    prefill_ids = out_ids[j, :-20]
                    decode_ids = out_ids[j, -20:]

                    prefill_text = token_ids_to_text(prefill_ids, tokenizer)
                    decode_text = token_ids_to_text(decode_ids, tokenizer)

                    print(f"--- Epoch {epoch+1} | Sample {samples_printed + 1} ---")
                    print(f"PREFILL: {prefill_text}")
                    print(f"DECODE:  {decode_text}")
                    print("-" * 30 + "\n")

                    samples_printed += 1

        print("=" * 50 + "\n")

    return model, train_losses, val_losses
