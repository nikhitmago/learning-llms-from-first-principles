import torch
import torch.nn as nn

from learning_llms_from_first_principles.modules.peft import (
    LinearLoRA,
    LoRALayer,
    replace_linear_with_lora,
)


def test_lora_layer_output_shape() -> None:
    layer = LoRALayer(dim_in=16, dim_out=32, rank=4, alpha=1.0)
    x = torch.randn(2, 16)
    out = layer(x)
    assert out.shape == (2, 32)


def test_lora_layer_zero_initialized_B() -> None:
    layer = LoRALayer(dim_in=8, dim_out=8, rank=2, alpha=1.0)
    x = torch.randn(1, 8)
    out = layer(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_linear_lora_output_shape() -> None:
    linear = nn.Linear(16, 32)
    lora = LinearLoRA(linear, rank=4, alpha=1.0)
    x = torch.randn(2, 16)
    out = lora(x)
    assert out.shape == (2, 32)


def test_linear_lora_matches_linear_at_init() -> None:
    linear = nn.Linear(8, 8)
    lora = LinearLoRA(linear, rank=2, alpha=1.0)
    x = torch.randn(3, 8)
    expected = linear(x)
    actual = lora(x)
    assert torch.allclose(expected, actual)


def test_linear_lora_gradients_flow() -> None:
    linear = nn.Linear(8, 4)
    lora = LinearLoRA(linear, rank=2, alpha=1.0)
    x = torch.randn(2, 8)
    out = lora(x).sum()
    out.backward()
    assert lora.lora_layer.A.grad is not None
    assert lora.lora_layer.B.grad is not None


def test_replace_linear_with_lora() -> None:
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    replace_linear_with_lora(model, rank=4, alpha=1.0)
    assert isinstance(model[0], LinearLoRA)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], LinearLoRA)


def test_replace_linear_with_lora_param_count() -> None:
    model = nn.Linear(64, 64)
    wrapper = nn.Sequential(model)
    replace_linear_with_lora(wrapper, rank=4, alpha=1.0)
    lora_params = sum(p.numel() for n, p in wrapper.named_parameters() if "lora_layer" in n)
    assert lora_params == (64 * 4) + (4 * 64)
