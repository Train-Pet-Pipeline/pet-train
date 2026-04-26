"""Tests for zero-shot audio inference module."""

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from pet_train.audio.inference import (
    AUDIOSET_CLASS_MAP,
    CLASSES,
    AudioInference,
    AudioPrediction,
)


def _create_wav(path: Path, sample_rate: int = 16000, duration_s: float = 1.0, channels: int = 1):
    """Helper to create a dummy WAV file using stdlib wave module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * duration_s)
    data = (np.random.randn(samples * channels) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return path


class TestAudioSetClassMap:
    """Tests for AudioSet → 5-class mapping."""

    def test_all_mapped_classes_valid(self):
        """All mapped classes are in our CLASSES list."""
        for cls in AUDIOSET_CLASS_MAP.values():
            assert cls in CLASSES

    def test_all_five_classes_covered(self):
        """Mapping covers eating, drinking, vomiting, ambient (other is fallback)."""
        mapped_classes = set(AUDIOSET_CLASS_MAP.values())
        assert "eating" in mapped_classes
        assert "drinking" in mapped_classes
        assert "vomiting" in mapped_classes
        assert "ambient" in mapped_classes

    def test_no_duplicate_indices(self):
        """Each AudioSet index maps to exactly one class."""
        indices = list(AUDIOSET_CLASS_MAP.keys())
        assert len(indices) == len(set(indices))


class TestAudioInference:
    """Tests for AudioInference zero-shot classifier."""

    def test_init_without_pretrained(self):
        """Initializes with random weights when no pretrained path."""
        model = AudioInference(pretrained_path=None)
        assert model.model is not None

    def test_predict_returns_prediction(self, tmp_dir):
        """predict() returns AudioPrediction with valid fields."""
        wav_path = _create_wav(tmp_dir / "test.wav")
        model = AudioInference(pretrained_path=None)
        result = model.predict(str(wav_path))
        assert isinstance(result, AudioPrediction)
        assert result.label in CLASSES
        assert 0.0 <= result.confidence <= 1.0
        assert set(result.class_scores.keys()) == set(CLASSES)

    def test_predict_resamples_non_16k(self, tmp_dir):
        """predict() handles audio at different sample rates."""
        wav_path = _create_wav(tmp_dir / "test_32k.wav", sample_rate=32000)
        model = AudioInference(pretrained_path=None)
        result = model.predict(str(wav_path))
        assert isinstance(result, AudioPrediction)

    def test_predict_handles_stereo(self, tmp_dir):
        """predict() handles stereo audio by converting to mono."""
        path = _create_wav(tmp_dir / "stereo.wav", channels=2)
        model = AudioInference(pretrained_path=None)
        result = model.predict(str(path))
        assert isinstance(result, AudioPrediction)

    def test_aggregate_scores_sums_correctly(self):
        """_aggregate_scores maps AudioSet probs to our 5 classes."""
        model = AudioInference(pretrained_path=None)
        fake_probs = torch.zeros(527)
        fake_probs[54] = 0.9   # Chewing → eating
        fake_probs[449] = 0.7  # Pour → drinking
        fake_probs[514] = 0.3  # Environmental → ambient
        scores = model._aggregate_scores(fake_probs)
        assert scores["eating"] == pytest.approx(0.9)
        assert scores["drinking"] == pytest.approx(0.7)
        assert scores["ambient"] == pytest.approx(0.3)

    def test_class_scores_all_present(self):
        """All 5 classes always present in class_scores."""
        model = AudioInference(pretrained_path=None)
        fake_probs = torch.zeros(527)
        scores = model._aggregate_scores(fake_probs)
        assert set(scores.keys()) == set(CLASSES)

    def test_pretrained_arch_drift_logs_warning(self, tmp_path, caplog):
        """F008 retro: when checkpoint shape doesn't match the local arch,
        load_state_dict(strict=False) silently drops keys. Make sure we at
        least log a warning so users know weights were not fully loaded."""
        import logging

        bogus = {"features.0.weight": torch.zeros(8, 8, 3, 3)}  # wrong everything
        ckpt_path = tmp_path / "bogus.pth"
        torch.save({"model": bogus}, ckpt_path)
        with caplog.at_level(logging.WARNING, logger="pet_train.audio.inference"):
            AudioInference(pretrained_path=str(ckpt_path))
        warns = [r for r in caplog.records if "partially loaded" in r.message]
        assert warns, "expected drift warning, got none"
