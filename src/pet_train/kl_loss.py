"""KL distillation loss for knowledge distillation from teacher to student model.

Supports two modes:
- Full vocabulary KL: exact KL divergence with temperature softening
- Top-k approximate KL: KL computed only over teacher's top-k tokens,
  remaining probability mass consolidated into a rest bucket.
"""

import torch
import torch.nn.functional as F  # noqa: N812


def compute_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    lambda_kl: float = 0.1,
) -> torch.Tensor:
    """Compute full-vocabulary KL distillation loss with temperature softening.

    Args:
        student_logits: Student model output logits [batch, seq_len, vocab_size].
        teacher_logits: Teacher model output logits [batch, seq_len, vocab_size].
        temperature: Softening temperature. Higher values produce softer distributions.
        lambda_kl: Scaling factor for KL loss term.

    Returns:
        Scalar KL distillation loss.
    """
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    return lambda_kl * (temperature**2) * kl


def compute_topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    temperature: float = 2.0,
    lambda_kl: float = 0.1,
) -> torch.Tensor:
    """Compute top-k approximate KL distillation loss.

    When teacher logits are unavailable (API only returns top-k logprobs),
    compute KL only over the teacher's top-k tokens. Remaining probability
    mass is assigned to a single rest bucket.

    Note: teacher_logprobs are used as-is (at the teacher's original temperature).
    API providers return logprobs at T=1 and cannot be re-scaled without full logits.
    Temperature only softens the student distribution. Callers that have full teacher
    logits should use compute_kl_distillation_loss instead.

    Args:
        student_logits: Student logits [batch, seq_len, vocab_size].
        teacher_token_ids: Token IDs for teacher's top-k [batch, seq_len, k].
        teacher_logprobs: Teacher's log probabilities for top-k [batch, seq_len, k].
            These are at the teacher's original temperature (typically T=1 from APIs).
        temperature: Softening temperature applied to student logits only.
        lambda_kl: Scaling factor for KL loss term.

    Returns:
        Scalar approximate KL distillation loss.
    """
    # Teacher top-k probabilities (already from softmax, stored as logprobs)
    teacher_topk_probs = torch.exp(teacher_logprobs)  # [batch, seq, k]
    teacher_rest_prob = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(
        min=1e-8
    )

    # Student softmax at temperature — compute once, derive log-probs from it
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    student_log_prob = torch.log(student_probs.clamp(min=1e-8))

    # Gather student probabilities at teacher's top-k positions
    student_topk_logprobs = torch.gather(
        student_log_prob, dim=-1, index=teacher_token_ids
    )
    student_topk_probs = torch.gather(
        student_probs, dim=-1, index=teacher_token_ids
    )

    # Student rest bucket: log(1 - sum(top-k probs))
    student_rest_logprob = torch.log(
        (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    )

    # KL = sum(teacher_p * (log(teacher_p) - log(student_p)))
    # Top-k terms
    kl_topk = teacher_topk_probs * (teacher_logprobs - student_topk_logprobs)
    # Rest bucket term
    teacher_rest_logprob = torch.log(teacher_rest_prob)
    kl_rest = teacher_rest_prob * (teacher_rest_logprob - student_rest_logprob)

    # Use batchmean reduction (sum over all dims, divide by batch) to match F.kl_div
    batch_size = student_logits.size(0)
    kl = (kl_topk.sum(dim=-1) + kl_rest.squeeze(-1)).sum().clamp(min=0.0) / batch_size
    return lambda_kl * (temperature**2) * kl
