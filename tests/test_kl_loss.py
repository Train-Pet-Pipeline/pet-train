"""Tests for KL distillation loss computation."""

import torch

from pet_train.kl_loss import compute_kl_distillation_loss, compute_topk_kl_loss


class TestFullVocabKL:
    """Tests for full vocabulary KL distillation loss."""

    def test_identical_distributions_zero_loss(self):
        """KL loss is near-zero when student matches teacher exactly."""
        logits = torch.randn(2, 10, 100)  # [batch, seq_len, vocab]
        loss = compute_kl_distillation_loss(
            student_logits=logits,
            teacher_logits=logits,
            temperature=2.0,
            lambda_kl=0.1,
        )
        assert loss.item() < 1e-5

    def test_different_distributions_positive_loss(self):
        """KL loss is positive when distributions differ."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss = compute_kl_distillation_loss(
            student_logits=student,
            teacher_logits=teacher,
            temperature=2.0,
            lambda_kl=0.1,
        )
        assert loss.item() > 0

    def test_lambda_scales_loss(self):
        """Lambda parameter scales the loss proportionally."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss_01 = compute_kl_distillation_loss(
            student, teacher, temperature=2.0, lambda_kl=0.1
        )
        loss_02 = compute_kl_distillation_loss(
            student, teacher, temperature=2.0, lambda_kl=0.2
        )
        assert abs(loss_02.item() / loss_01.item() - 2.0) < 0.01

    def test_temperature_softens_distribution(self):
        """Higher temperature produces lower raw KL (softer distributions).

        The T^2 factor in the loss compensates gradient magnitude for training.
        The underlying KL divergence (before T^2 scaling) must decrease with
        higher temperature. Verified by dividing out the T^2 factor.
        """
        torch.manual_seed(42)
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        t1, t4 = 1.0, 4.0
        loss_t1 = compute_kl_distillation_loss(student, teacher, temperature=t1, lambda_kl=1.0)
        loss_t4 = compute_kl_distillation_loss(student, teacher, temperature=t4, lambda_kl=1.0)
        # Divide out T^2 to compare the raw KL divergences
        raw_kl_t1 = loss_t1.item() / (t1**2)
        raw_kl_t4 = loss_t4.item() / (t4**2)
        assert raw_kl_t4 < raw_kl_t1

    def test_output_is_scalar(self):
        """Loss output is a scalar tensor."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss = compute_kl_distillation_loss(student, teacher)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """Gradients flow through the loss to student logits."""
        student = torch.randn(2, 10, 100, requires_grad=True)
        teacher = torch.randn(2, 10, 100)
        loss = compute_kl_distillation_loss(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()


class TestTopKKL:
    """Tests for top-k approximate KL distillation loss."""

    def test_output_is_scalar(self):
        """Top-k KL returns a scalar loss."""
        student_logits = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        teacher_top_k_ids = torch.randint(0, 100, (2, 10, 5))  # top-5
        teacher_top_k_logprobs = torch.randn(2, 10, 5)
        loss = compute_topk_kl_loss(
            student_logits=student_logits,
            teacher_token_ids=teacher_top_k_ids,
            teacher_logprobs=teacher_top_k_logprobs,
            temperature=2.0,
            lambda_kl=0.1,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self):
        """Gradients flow through top-k KL loss."""
        student = torch.randn(2, 10, 100, requires_grad=True)
        ids = torch.randint(0, 100, (2, 10, 5))
        logprobs = torch.randn(2, 10, 5)
        loss = compute_topk_kl_loss(student, ids, logprobs)
        loss.backward()
        assert student.grad is not None

    def test_higher_k_closer_to_full(self):
        """With k approaching vocab_size, top-k KL approaches full KL."""
        vocab = 50
        student = torch.randn(2, 5, vocab)
        teacher = torch.randn(2, 5, vocab)

        full_loss = compute_kl_distillation_loss(
            student, teacher, temperature=2.0, lambda_kl=0.1
        )

        # Simulate top-k with k=vocab (all tokens)
        teacher_probs = torch.softmax(teacher / 2.0, dim=-1)
        top_k_logprobs = torch.log(teacher_probs)
        top_k_ids = torch.arange(vocab).unsqueeze(0).unsqueeze(0).expand(2, 5, vocab)

        topk_loss = compute_topk_kl_loss(
            student, top_k_ids, top_k_logprobs, temperature=2.0, lambda_kl=0.1
        )

        # Should be reasonably close
        assert abs(full_loss.item() - topk_loss.item()) < 0.5
