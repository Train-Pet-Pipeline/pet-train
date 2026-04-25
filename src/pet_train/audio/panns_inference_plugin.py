"""PANNs-based audio inference plugin (F008 fix — pluggable approach).

Wraps the official ``panns_inference`` package to produce 527-class
AudioSet logits, then reuses pet-train's existing AUDIOSET_CLASS_MAP +
_aggregate_scores logic to map to the 5-class pet feeder taxonomy
defined in DEV_GUIDE §2.5.

This is the **pluggable solution to F008**: rather than rewriting
pet-train's hand-rolled MobileNetV2AudioSet to match the official PANNs
checkpoint layout (which would pin pet-train to a single architecture),
we add a NEW plugin that wraps the official ``panns_inference``
project. Users pick which audio backend via params/recipe.

Selection (params.yaml):
    audio:
      inference_backend: "panns"   # vs "mobilenetv2_legacy"

Existing ``AudioInference`` class stays for backward-compat (it never
worked with official weights, but we don't want to break test fixtures).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pet_train.audio.inference import (
    AUDIOSET_CLASS_MAP,
    CLASSES,
    AudioPrediction,
)


class PANNsAudioInference:
    """Pluggable PANNs-based audio inference.

    Uses ``panns_inference.AudioTagging`` (official upstream wrapper) for
    the 527-class AudioSet head. Maps to pet-train's 5 pet feeder classes
    via AUDIOSET_CLASS_MAP + max-pool aggregate.

    Args:
        checkpoint_path: Optional path to PANNs MobileNetV2/Cnn14 checkpoint.
            If None, panns_inference auto-downloads to ~/panns_data/.
        device: 'cuda' or 'cpu'. Defaults to cuda when available.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        # Lazy import — panns_inference is an optional dep
        from panns_inference import AudioTagging

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tagger = AudioTagging(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self._device = device

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "PANNsAudioInference":
        """Build from params.yaml audio sub-dict.

        Reads optional ``checkpoint_path`` (panns_inference downloads default
        if absent) and ``device``.
        """
        return cls(
            checkpoint_path=params.get("checkpoint_path"),
            device=params.get("device"),
        )

    def _aggregate_scores(self, probs: torch.Tensor) -> dict[str, float]:
        """Aggregate AudioSet 527-class probs → 5-class pet feeder scores.

        Mirrors ``pet_train.audio.inference.AudioInference._aggregate_scores``
        verbatim so behaviour is identical regardless of inference backend.
        """
        scores: dict[str, float] = {cls: 0.0 for cls in CLASSES}
        for idx, cls_name in AUDIOSET_CLASS_MAP.items():
            prob = float(probs[idx])
            scores[cls_name] = max(scores[cls_name], prob)
        mapped_indices = set(AUDIOSET_CLASS_MAP.keys())
        other_probs = [
            float(probs[i]) for i in range(probs.shape[0])
            if i not in mapped_indices
        ]
        if other_probs:
            scores["other"] = max(other_probs)
        return scores

    @torch.no_grad()
    def predict(self, audio_path: str | Path) -> AudioPrediction:
        """Classify an audio file into one of 5 pet feeder classes.

        Args:
            audio_path: Path to a 16kHz mono wav (panns_inference handles
                resampling internally if rate differs).

        Returns:
            AudioPrediction with label = max-prob class + score.
        """
        # panns_inference expects (1, samples) numpy/torch array
        import librosa

        audio, _sr = librosa.load(str(audio_path), sr=32000, mono=True)
        # panns_inference auto-resamples internally; AudioTagging.inference
        # accepts shape (batch=1, samples)
        audio_tensor = audio.reshape(1, -1)
        clipwise_output, _embedding = self._tagger.inference(audio_tensor)
        # clipwise_output shape (1, 527), already softmaxed
        probs = torch.from_numpy(clipwise_output[0])
        scores = self._aggregate_scores(probs)
        label, score = max(scores.items(), key=lambda kv: kv[1])
        return AudioPrediction(label=label, confidence=score, class_scores=scores)
