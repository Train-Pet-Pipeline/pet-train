"""File-based teacher logits provider.

Reads pre-computed logits from .pt files on disk.
"""

import json
import logging
from pathlib import Path

import torch

from pet_train.logits_provider.base import LogitsResult, TeacherLogitsProvider

logger = logging.getLogger(__name__)


class FileLogitsProvider(TeacherLogitsProvider):
    """Load teacher logits from .pt files indexed by a manifest.

    Args:
        logits_dir: Directory containing .pt files and manifest.json.
    """

    def __init__(self, logits_dir: str):
        """Initialize FileLogitsProvider."""
        self.logits_dir = Path(logits_dir)
        self._manifest = self._load_manifest()
        self._is_full_vocab = "top_k" not in self._manifest

    def _load_manifest(self) -> dict:
        """Load and validate manifest.json."""
        manifest_path = self.logits_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json found in {self.logits_dir}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info(
            "Loaded logits manifest: model=%s, samples=%d",
            manifest.get("model", "unknown"),
            len(manifest.get("samples", {})),
        )
        return manifest

    @property
    def available_samples(self) -> set[str]:
        """Return set of sample IDs available in the manifest."""
        return set(self._manifest.get("samples", {}).keys())

    def get_logits(self, sample_id: str) -> LogitsResult:
        """Load logits for a specific sample from disk.

        Args:
            sample_id: Unique identifier for the training sample.

        Returns:
            LogitsResult with teacher token IDs and log probabilities.

        Raises:
            KeyError: If sample_id is not found in the manifest.
        """
        samples = self._manifest.get("samples", {})
        if sample_id not in samples:
            raise KeyError(f"Sample '{sample_id}' not found in logits manifest")

        pt_path = self.logits_dir / samples[sample_id]
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        return LogitsResult(
            token_ids=data["token_ids"],
            logprobs=data["logprobs"],
            is_full_vocab=self._is_full_vocab,
        )
