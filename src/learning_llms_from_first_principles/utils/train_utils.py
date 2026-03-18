import logging
import math
import time
from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from learning_llms_from_first_principles.config import GPT_CONFIG_124M
from learning_llms_from_first_principles.inference.generate import generate_tokens
from learning_llms_from_first_principles.utils.tokenization_utils import token_ids_to_text

ModelT = TypeVar("ModelT", bound=nn.Module)

logger = logging.getLogger(__name__)


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: torch.device
) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)  # bs, seq_len, vocab_size
    bs, seq_len, vocab_size = logits.shape

    # Flatten (bs, seq_len, vocab_size ) => (bs * seq_len, vocab_size)
    logits_flattened = logits.view(-1, vocab_size)  # [bs * seq_len, vocab_size]
    targets_flattened = target_batch.view(-1)  # [bs * seq_len]

    # Use torch to calculate the CE loss (Optimized and Fused for GPU/MPS)
    loss_lib = nn.CrossEntropyLoss()(logits_flattened, targets_flattened)

    # Note: Pedogogical CE loss from scratch (Disabled for Performance)
    # Intuition: Cross Entropy (CE) measures "Surprise" (Information Theory).
    # Each class has a probability P. We use -ln(P) for loss because as P -> 0, penalty explodes.
    # probs_flattened = torch.softmax(logits_flattened, dim=-1)
    # probs_values = probs_flattened[torch.arange(probs_flattened.shape[0]), targets_flattened]
    # loss_ce_scratch = -torch.log(probs_values).mean()
    # assert torch.allclose(loss_lib, loss_ce_scratch)

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
    warmup_ratio: float = 0.1,
    warmup_min_lr: float = 3e-05,
    decay_floor_lr: float = 1e-6,
    max_norm: float = 1.0,
) -> tuple[ModelT, list[float], list[float], list[float]]:
    train_losses, val_losses, lrs = [], [], []
    global_step = -1

    # Learning rate scheduling (warmup)
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_training_steps * warmup_ratio)
    warmup_lr_increment = (peak_lr - warmup_min_lr) / warmup_steps

    logger.info(f"LR Schedule: Warmup for {warmup_steps} steps ({warmup_ratio*100:.1f}%)")
    logger.info(f" Warm up LR Range: {warmup_min_lr} -> {peak_lr}")
    logger.info(
        f" Cosine Annealing: {peak_lr} -> {decay_floor_lr} over {total_training_steps - warmup_steps} steps"
    )
    logger.info(f"Total Training Steps: {total_training_steps}")

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        train_start_time = time.time()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            running_train_loss += loss.item()
            global_step += 1

            # Adjust the warmup lr
            if global_step < warmup_steps:
                lr = warmup_min_lr + global_step * warmup_lr_increment
            else:
                # Cosine annealing
                progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                lr = decay_floor_lr + (peak_lr - decay_floor_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            lrs.append(lr)

            loss.backward()

            # Gradient Clipping (Global Level)
            # This treats ALL model parameters (e.g., all 32B parameters in a 32B model)
            # as a single giant vector to ensure the total update doesn't explode.
            #
            # Pseudo-code logic:
            # grad_norm = torch.sqrt(torch.square(grads).sum())
            # if grad_norm > max_norm:
            #     scale = max_norm / grad_norm
            #     grads = grads * scale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()

            # Print intermediate progress based on global steps
            if global_step % eval_freq == 0 and global_step > 0:
                train_time = time.time() - train_start_time

                model.eval()
                val_start_time = time.time()
                with torch.no_grad():
                    val_loss = calc_loss_loader(val_loader, model, device)
                val_time = time.time() - val_start_time

                train_loss = running_train_loss / (i + 1)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.info(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} "
                    f"[Train: {train_time:.1f}s, Val: {val_time:.1f}s]"
                )

                # Reset training clock for next period
                train_start_time = time.time()

        # --- Epoch-end reporting ---
        logger.info("\n" + "=" * 50 + "\n")

        model.eval()
        samples_printed = 0
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(val_loader):
                if samples_printed >= 2:
                    break

                input_ids = input_batch.to(device)

                # Generate 20 new tokens
                out_ids = generate_tokens(
                    model=model,
                    idx=input_ids,
                    max_new_tokens=20,
                    context_size=int(GPT_CONFIG_124M["context_len"]),
                    temperature=0.0,
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

                    logger.info(f"--- Epoch {epoch+1} | Sample {samples_printed + 1} ---")
                    logger.info(f"PREFILL: {prefill_text}")
                    logger.info(f"DECODE:  {decode_text}")
                    logger.info("-" * 30 + "\n")

                    samples_printed += 1

        logger.info("=" * 50 + "\n")

    return model, train_losses, val_losses, lrs
