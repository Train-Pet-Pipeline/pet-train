"""API-based teacher logits provider with disk caching.

Fetches top-k logprobs from OpenAI-compatible APIs and caches results.
"""

import json
import logging
from pathlib import Path

import torch

from pet_train.logits_provider.base import LogitsResult, TeacherLogitsProvider

logger = logging.getLogger(__name__)


class APILogitsProvider(TeacherLogitsProvider):
    """Fetch teacher logprobs from API with transparent disk caching.

    Args:
        cache_dir: Directory for cached .pt files and manifest.json.
        base_url: API base URL (OpenAI-compatible).
        model_name: Model name for API requests.
        api_key_env: Environment variable name containing the API key.
        top_k: Number of top logprobs to request from API.
        timeout: API request timeout in seconds.
        max_retries: Maximum retry attempts for failed API calls.
    """

    def __init__(
        self,
        cache_dir: str,
        base_url: str,
        model_name: str,
        api_key_env: str,
        top_k: int = 20,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize APILogitsProvider."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.top_k = top_k
        self.timeout = timeout
        self.max_retries = max_retries
        self._manifest = self._load_or_create_manifest()
        self._is_full_vocab = "top_k" not in self._manifest

    def _load_or_create_manifest(self) -> dict:
        """Load existing manifest or create a new one."""
        manifest_path = self.cache_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return {
            "model": self.model_name,
            "top_k": self.top_k,
            "samples": {},
        }

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        manifest_path = self.cache_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    @property
    def available_samples(self) -> set[str]:
        """Return set of sample IDs available in the cache."""
        return set(self._manifest.get("samples", {}).keys())

    def get_logits(self, sample_id: str) -> LogitsResult:
        """Get teacher logits from cache. Raises if not cached.

        For batch collection, use collect_logits.sh which calls
        cache_sample() for each missing sample.

        Args:
            sample_id: Unique identifier for the training sample.

        Returns:
            LogitsResult with teacher token IDs and log probabilities.

        Raises:
            KeyError: If sample_id is not in the cache.
        """
        if sample_id not in self._manifest.get("samples", {}):
            raise KeyError(
                f"Sample '{sample_id}' not in cache. "
                f"Run collect_logits.sh to fetch missing logits."
            )

        pt_path = self.cache_dir / self._manifest["samples"][sample_id]
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

        return LogitsResult(
            token_ids=data["token_ids"],
            logprobs=data["logprobs"],
            is_full_vocab=self._is_full_vocab,
        )

    def cache_sample(
        self,
        sample_id: str,
        token_ids: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None:
        """Cache API-fetched logprobs to disk.

        Args:
            sample_id: Training sample identifier.
            token_ids: Top-k token IDs [seq_len, k].
            logprobs: Corresponding log probabilities [seq_len, k].
        """
        safe_id = sample_id.replace("/", "_").replace("..", "_")
        filename = f"{safe_id}.pt"
        torch.save(
            {"token_ids": token_ids, "logprobs": logprobs},
            self.cache_dir / filename,
        )
        self._manifest["samples"][sample_id] = filename
        self._save_manifest()
        logger.info("Cached logits for sample %s", sample_id)
