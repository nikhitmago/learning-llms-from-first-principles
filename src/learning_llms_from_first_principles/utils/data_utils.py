import torch


def split_data(data: str, train_ratio: float = 0.9, val_ratio: float = 0.1) -> tuple[str, str, str]:
    """
    Splits data (string, list, or tensor) into training, validation, and testing sets
    based on the provided ratios. The ratios must sum to 1.0.

    Args:
        data: The string data to be split.
        train_ratio: Float representing the proportion of data for training.
        val_ratio: Float representing the proportion of data for validation.

    Returns:
        tuple: (train_data, val_data, test_data)
    """

    total_len = len(data)

    # Calculate absolute split indices based on the ratios
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    # Slice the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def format_instruct_prompt(entry: dict[str, str]) -> str:
    """Format an Alpaca-style instruction entry into a prompt string."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def instruct_collate_fn(
    batch: list[dict[str, list[int]]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for instruction fine-tuning batches.

    Each item in batch is a dict with "input_ids" and "target_ids" (from InstructionDataset).
    Per-item prep (EOS, shift, -100 masking) is already done by the Dataset.
    This function handles batch-level padding to the longest sequence in the batch
    and stacking into tensors.

    Steps:
    1. Find the longest sequence in the batch
    2. Pad input_ids with pad_token_id and target_ids with ignore_index to batch max length
    3. Optionally truncate to allowed_max_length
    4. Return (inputs_tensor, targets_tensor) on the target device
    """
    batch_max_length = max(len(item["input_ids"]) for item in batch)

    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []

    for item in batch:
        input_ids = item["input_ids"][:batch_max_length]
        target_ids = item["target_ids"][:batch_max_length]

        # Pad to batch_max_length
        pad_len = batch_max_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * pad_len
        target_ids = target_ids + [ignore_index] * pad_len

        inputs_lst.append(torch.tensor(input_ids))
        targets_lst.append(torch.tensor(target_ids))

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
