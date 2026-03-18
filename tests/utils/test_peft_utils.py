from pathlib import Path

import torch
import torch.nn as nn

from learning_llms_from_first_principles.modules.peft import LinearLoRA
from learning_llms_from_first_principles.utils.peft_utils import (
    load_lora_adapters,
    merge_lora_weights,
    replace_linear_with_lora,
    save_lora_adapters,
)


def test_replace_linear_with_lora() -> None:
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    replace_linear_with_lora(model, rank=4, alpha=1.0)
    assert isinstance(model[0], LinearLoRA)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], LinearLoRA)


def test_replace_linear_with_lora_param_count() -> None:
    wrapper = nn.Sequential(nn.Linear(64, 64))
    replace_linear_with_lora(wrapper, rank=4, alpha=1.0)
    lora_params = sum(p.numel() for n, p in wrapper.named_parameters() if "lora_layer" in n)
    assert lora_params == (64 * 4) + (4 * 64)


def test_save_and_load_lora_adapters(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    replace_linear_with_lora(model, rank=2, alpha=1.0)

    x = torch.randn(2, 8)
    out = model(x).sum()
    out.backward()

    path = str(tmp_path / "adapters.pth")
    save_lora_adapters(model, path)

    model2 = nn.Sequential(nn.Linear(8, 8))
    replace_linear_with_lora(model2, rank=2, alpha=1.0)
    load_lora_adapters(model2, path)

    for (k1, v1), (k2, v2) in zip(
        sorted((k, v) for k, v in model.state_dict().items() if "lora_layer" in k),
        sorted((k, v) for k, v in model2.state_dict().items() if "lora_layer" in k),
    ):
        assert torch.equal(v1, v2), f"Mismatch on {k1}"


def test_merge_lora_weights() -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    replace_linear_with_lora(model, rank=2, alpha=1.0)

    x = torch.randn(3, 8)
    model.eval()
    with torch.no_grad():
        before = model(x).clone()

    merge_lora_weights(model)

    assert isinstance(model[0], nn.Linear)
    with torch.no_grad():
        after = model(x)

    assert torch.allclose(before, after, atol=1e-5)


def test_merge_removes_lora_from_state_dict() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    replace_linear_with_lora(model, rank=2, alpha=1.0)

    assert any("lora_layer" in k for k in model.state_dict())

    merge_lora_weights(model)

    keys = list(model.state_dict().keys())
    assert not any("lora_layer" in k for k in keys)
    assert all(k.endswith((".weight", ".bias")) for k in keys)
