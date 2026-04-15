"""Teacher logits provider implementations for KL distillation."""

from pet_train.logits_provider.api_provider import APILogitsProvider
from pet_train.logits_provider.base import LogitsResult, TeacherLogitsProvider
from pet_train.logits_provider.file_provider import FileLogitsProvider

__all__ = [
    "LogitsResult",
    "TeacherLogitsProvider",
    "FileLogitsProvider",
    "APILogitsProvider",
]
