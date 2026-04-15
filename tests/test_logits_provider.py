"""Tests for teacher logits provider implementations."""

import json

import pytest
import torch

from pet_train.logits_provider.api_provider import APILogitsProvider
from pet_train.logits_provider.base import LogitsResult
from pet_train.logits_provider.file_provider import FileLogitsProvider


class TestLogitsResult:
    """Tests for LogitsResult dataclass."""

    def test_full_vocab_result(self):
        """Full vocab result has is_full_vocab=True."""
        result = LogitsResult(
            token_ids=torch.arange(100).unsqueeze(0),
            logprobs=torch.randn(1, 100),
            is_full_vocab=True,
        )
        assert result.is_full_vocab

    def test_topk_result(self):
        """Top-k result has is_full_vocab=False."""
        result = LogitsResult(
            token_ids=torch.randint(0, 100, (10, 5)),
            logprobs=torch.randn(10, 5),
            is_full_vocab=False,
        )
        assert not result.is_full_vocab


class TestFileProvider:
    """Tests for FileLogitsProvider."""

    @pytest.fixture
    def logits_dir(self, tmp_dir):
        """Create a directory with cached logits files and manifest."""
        ld = tmp_dir / "teacher_logits"
        ld.mkdir()

        for i in range(3):
            sample_id = f"sft_{i:05d}"
            data = {
                "token_ids": torch.randint(0, 1000, (20, 10)),
                "logprobs": torch.randn(20, 10),
            }
            torch.save(data, ld / f"{sample_id}.pt")

        manifest = {
            "model": "qwen2.5-vl-72b",
            "top_k": 10,
            "samples": {
                f"sft_{i:05d}": f"sft_{i:05d}.pt" for i in range(3)
            },
        }
        (ld / "manifest.json").write_text(json.dumps(manifest))
        return ld

    def test_load_manifest(self, logits_dir):
        """Provider loads manifest and discovers available samples."""
        provider = FileLogitsProvider(logits_dir=str(logits_dir))
        assert len(provider.available_samples) == 3
        assert "sft_00000" in provider.available_samples

    def test_get_logits_returns_result(self, logits_dir):
        """get_logits returns a valid LogitsResult for known sample."""
        provider = FileLogitsProvider(logits_dir=str(logits_dir))
        result = provider.get_logits("sft_00000")
        assert isinstance(result, LogitsResult)
        assert result.token_ids.shape[1] == 10
        assert not result.is_full_vocab

    def test_get_logits_unknown_sample_raises(self, logits_dir):
        """get_logits raises KeyError for unknown sample_id."""
        provider = FileLogitsProvider(logits_dir=str(logits_dir))
        with pytest.raises(KeyError, match="nonexistent"):
            provider.get_logits("nonexistent")

    def test_is_full_vocab_when_no_top_k(self, tmp_dir):
        """Provider marks as full_vocab when manifest has no top_k."""
        ld = tmp_dir / "full_logits"
        ld.mkdir()
        data = {
            "token_ids": torch.arange(50000).unsqueeze(0).expand(10, -1),
            "logprobs": torch.randn(10, 50000),
        }
        torch.save(data, ld / "sft_00000.pt")
        manifest = {
            "model": "local-72b",
            "samples": {"sft_00000": "sft_00000.pt"},
        }
        (ld / "manifest.json").write_text(json.dumps(manifest))

        provider = FileLogitsProvider(logits_dir=str(ld))
        result = provider.get_logits("sft_00000")
        assert result.is_full_vocab


class TestAPIProvider:
    """Tests for APILogitsProvider."""

    @pytest.fixture
    def cache_dir(self, tmp_dir):
        """Provide a temporary cache directory."""
        d = tmp_dir / "api_cache"
        d.mkdir()
        return d

    def test_cached_result_returned(self, cache_dir):
        """If logits are already cached on disk, return them without API call."""
        data = {
            "token_ids": torch.randint(0, 1000, (20, 20)),
            "logprobs": torch.randn(20, 20),
        }
        torch.save(data, cache_dir / "sft_00000.pt")
        manifest = {
            "model": "api-teacher",
            "top_k": 20,
            "samples": {"sft_00000": "sft_00000.pt"},
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        provider = APILogitsProvider(
            cache_dir=str(cache_dir),
            base_url="http://fake",
            model_name="fake",
            api_key_env="FAKE_KEY",
            top_k=20,
        )
        result = provider.get_logits("sft_00000")
        assert isinstance(result, LogitsResult)
        assert not result.is_full_vocab

    def test_missing_sample_raises_without_prompt(self, cache_dir):
        """get_logits raises KeyError when sample not cached and no prompt provided."""
        manifest = {"model": "api-teacher", "top_k": 20, "samples": {}}
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        provider = APILogitsProvider(
            cache_dir=str(cache_dir),
            base_url="http://fake",
            model_name="fake",
            api_key_env="FAKE_KEY",
            top_k=20,
        )
        with pytest.raises(KeyError):
            provider.get_logits("sft_00000")
