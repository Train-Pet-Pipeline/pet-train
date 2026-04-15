"""Abstract base class for teacher logits providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class LogitsResult:
    """Container for teacher model logits.

    Attributes:
        token_ids: Token IDs. Shape [seq_len, vocab_size] for full vocab,
            or [seq_len, k] for top-k.
        logprobs: Log probabilities corresponding to token_ids. Same shape.
        is_full_vocab: True if token_ids covers the full vocabulary,
            False if only top-k tokens.
    """

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    is_full_vocab: bool


class TeacherLogitsProvider(ABC):
    """Abstract interface for loading teacher model logits."""

    @abstractmethod
    def get_logits(self, sample_id: str) -> LogitsResult:
        """Retrieve teacher logits for a training sample.

        Args:
            sample_id: Unique identifier matching the training data ID field.

        Returns:
            LogitsResult containing teacher's token IDs and log probabilities.

        Raises:
            KeyError: If sample_id is not available.
        """
        ...

    @property
    @abstractmethod
    def available_samples(self) -> set[str]:
        """Return set of sample IDs that have cached logits."""
        ...
