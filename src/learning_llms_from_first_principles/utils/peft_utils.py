import torch
import torch.nn as nn

from learning_llms_from_first_principles.modules.peft import LinearLoRA


def replace_linear_with_lora(model: nn.Module, rank: int, alpha: float) -> None:
    """Recursively replace all ``nn.Linear`` layers with ``LinearLoRA``."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


def save_lora_adapters(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights (A and B matrices) to disk."""
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_layer" in k}
    torch.save(lora_state, path)


def load_lora_adapters(model: nn.Module, path: str) -> None:
    """Load LoRA adapter weights into a model that already has LinearLoRA layers."""
    lora_state = torch.load(path, weights_only=True)
    model.load_state_dict(lora_state, strict=False)


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA adapters into the base weights and replace LinearLoRA with nn.Linear.

    After merging, the model has no LoRA layers — just regular Linear layers
    with the adapted weights baked in. Useful for inference with no extra overhead.
    """
    for name, module in model.named_children():
        if isinstance(module, LinearLoRA):
            setattr(model, name, module.merge())
        else:
            merge_lora_weights(module)
