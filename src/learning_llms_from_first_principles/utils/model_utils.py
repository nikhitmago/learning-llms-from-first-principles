import logging
import typing
from collections import defaultdict

import torch.nn as nn

logger = logging.getLogger(__name__)


def print_model_parameters(model: nn.Module) -> None:
    """
    Prints the parameter count for each top-level module in the network.

    Example Output:
    Layer Name           |      Parameters |      Trainable
    --------------------------------------------------------
    tok_emb              |      38,597,376 |      38,597,376
    pos_emb              |         786,432 |         786,432
    trf_blocks.0         |       7,085,568 |       7,085,568
    ...
    out_head             |      38,597,376 |      38,597,376
    --------------------------------------------------------
    Total                |     163,009,536 |     163,009,536
    """
    layer_params: typing.DefaultDict[str, int] = defaultdict(int)
    layer_trainable: typing.DefaultDict[str, int] = defaultdict(int)

    for name, param in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "trf_blocks":
            top_layer = f"{parts[0]}.{parts[1]}"
        else:
            top_layer = parts[0]

        layer_params[top_layer] += param.numel()
        if param.requires_grad:
            layer_trainable[top_layer] += param.numel()

    logger.info(f"{'Layer Name':<20} | {'Parameters':>15} | {'Trainable':>15} | {'Frozen':>15}")
    logger.info("-" * 74)
    for layer in layer_params:
        trainable = layer_trainable.get(layer, 0)
        frozen = layer_params[layer] - trainable
        logger.info(f"{layer:<20} | {layer_params[layer]:>15,} | {trainable:>15,} | {frozen:>15,}")

    total = sum(layer_params.values())
    total_trainable = sum(layer_trainable.values())
    total_frozen = total - total_trainable
    logger.info("-" * 74)
    logger.info(f"{'Total':<20} | {total:>15,} | {total_trainable:>15,} | {total_frozen:>15,}")


def print_transformer_block_parameters(model: nn.Module) -> None:
    """
    Prints a detailed parameter breakdown for a single transformer block.
    Assumes the model has an attribute `trf_blocks` containing at least one block.

    Example Output:
    Transformer Block Component    |      Parameters
    ------------------------------------------------
    ffn                            |       4,722,432
    layer_norm_1                   |           1,536
    layer_norm_2                   |           1,536
    multi_head_attention           |       2,360,064
    ------------------------------------------------
    Total Block Parameters         |       7,085,568
    """
    if not hasattr(model, "trf_blocks") or len(model.trf_blocks) == 0:  # type: ignore
        logger.info("Model does not contain 'trf_blocks'.")
        return

    block_params: typing.DefaultDict[str, int] = defaultdict(int)

    # Examine the first transformer block (trf_blocks.0)
    for name, param in model.trf_blocks[0].named_parameters():  # type: ignore
        # e.g. 'multi_head_attention.W_q.weight' -> 'multi_head_attention'
        parts = name.split(".")
        top_component = parts[0]
        block_params[top_component] += param.numel()

    logger.info(f"{'Transformer Block Component':<30} | {'Parameters':>15}")
    logger.info("-" * 48)
    for component, count in block_params.items():
        logger.info(f"{component:<30} | {count:>15,}")

    logger.info("-" * 48)
    logger.info(f"{'Total Block Parameters':<30} | {sum(block_params.values()):>15,}")
