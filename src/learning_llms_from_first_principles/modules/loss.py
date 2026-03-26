import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss using PyTorch.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Softmax temperature

    Returns:
        Distillation loss
    """
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student) = Σ teacher * log(teacher / student)
    #                        = Σ teacher * (log(teacher) - log(student))
    # Teacher probs act as weights: penalize mismatches proportionally to how
    # important the teacher thinks each token is. A big log-ratio on a token
    # the teacher assigns 0.001 probability barely matters, but a mismatch
    # on a token the teacher is 90% confident about gets heavily penalized.
    kl = (teacher_probs * (teacher_probs.log() - student_probs.log())).sum()
    kl_loss = (temperature**2) * kl

    return kl_loss
