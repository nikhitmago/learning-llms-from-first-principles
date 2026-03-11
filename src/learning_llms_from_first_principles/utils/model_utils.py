import typing
from collections import defaultdict

import torch.nn as nn


def print_model_parameters(model: nn.Module) -> None:
    """
    Prints the parameter count for each top-level module in the network.

    Example Output:
    Layer Name           |      Parameters
    --------------------------------------
    tok_emb              |      38,597,376
    pos_emb              |         786,432
    trf_blocks.0         |       7,085,568
    trf_blocks.1         |       7,085,568
    trf_blocks.2         |       7,085,568
    trf_blocks.3         |       7,085,568
    trf_blocks.4         |       7,085,568
    trf_blocks.5         |       7,085,568
    trf_blocks.6         |       7,085,568
    trf_blocks.7         |       7,085,568
    trf_blocks.8         |       7,085,568
    trf_blocks.9         |       7,085,568
    trf_blocks.10        |       7,085,568
    trf_blocks.11        |       7,085,568
    final_norm           |           1,536
    out_head             |      38,597,376
    --------------------------------------
    Total Parameters     |     163,009,536
    """
    layer_params: typing.DefaultDict[str, int] = defaultdict(int)

    # Group parameters by their top-level module name
    for name, param in model.named_parameters():
        # Split the name by '.' to group by the highest-level component
        # e.g., 'trf_blocks.0.layer_norm_1.scale' -> 'trf_blocks.0'
        parts = name.split(".")
        if parts[0] == "trf_blocks":
            # Group each transformer block separately
            top_layer = f"{parts[0]}.{parts[1]}"
        else:
            # For top-level embeddings and output head
            top_layer = parts[0]

        layer_params[top_layer] += param.numel()

    print(f"{'Layer Name':<20} | {'Parameters':>15}")
    print("-" * 38)
    for layer, count in layer_params.items():
        print(f"{layer:<20} | {count:>15,}")

    print("-" * 38)
    print(f"{'Total Parameters':<20} | {sum(layer_params.values()):>15,}")


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
        print("Model does not contain 'trf_blocks'.")
        return

    block_params: typing.DefaultDict[str, int] = defaultdict(int)

    # Examine the first transformer block (trf_blocks.0)
    for name, param in model.trf_blocks[0].named_parameters():  # type: ignore
        # e.g. 'multi_head_attention.W_q.weight' -> 'multi_head_attention'
        parts = name.split(".")
        top_component = parts[0]
        block_params[top_component] += param.numel()

    print(f"{'Transformer Block Component':<30} | {'Parameters':>15}")
    print("-" * 48)
    for component, count in block_params.items():
        print(f"{component:<30} | {count:>15,}")

    print("-" * 48)
    print(f"{'Total Block Parameters':<30} | {sum(block_params.values()):>15,}")
