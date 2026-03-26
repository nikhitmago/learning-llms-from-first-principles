import torch

from learning_llms_from_first_principles.modules.loss import distillation_loss


def test_distillation_identical_logits_give_zero_loss() -> None:
    """If student and teacher produce the same logits, KL divergence should be 0."""
    torch.manual_seed(42)
    logits = torch.rand(4, 10)
    loss = distillation_loss(logits, logits, temperature=1.0)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_distillation_loss_is_non_negative() -> None:
    """KL divergence is always >= 0."""
    torch.manual_seed(7)
    student = torch.rand(4, 10)
    teacher = torch.rand(4, 10)
    loss = distillation_loss(student, teacher, temperature=2.0)
    assert loss >= -1e-6


def test_distillation_higher_temperature_smooths_distribution() -> None:
    """Higher temperature smooths distributions, reducing raw KL divergence."""
    torch.manual_seed(0)
    student = torch.tensor([[5.0, 1.0, 0.1]])
    teacher = torch.tensor([[1.0, 5.0, 0.1]])
    # Compare raw KL (before T^2 scaling) to verify smoothing effect
    loss_t1 = distillation_loss(student, teacher, temperature=1.0) / (1.0**2)
    loss_t5 = distillation_loss(student, teacher, temperature=5.0) / (5.0**2)
    assert loss_t5 < loss_t1
