"""Zero-shot audio classification using PANNs MobileNetV2 pretrained on AudioSet.

Maps AudioSet's 527 classes to 5 pet feeder classes:
eating, drinking, vomiting, ambient, other.

V1: zero-shot only (no fine-tuning). Weights handed to pet-quantize for INT8 conversion.
"""

import logging
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pet_infra.device import detect_device

logger = logging.getLogger(__name__)

# Our 5 target classes
CLASSES = ["eating", "drinking", "vomiting", "ambient", "other"]

# AudioSet index → our class mapping
AUDIOSET_CLASS_MAP: dict[int, str] = {
    # eating
    54: "eating",   # Chewing, mastication
    55: "eating",   # Biting
    # drinking
    445: "drinking",  # Splash, splatter
    448: "drinking",  # Drip
    449: "drinking",  # Pour
    450: "drinking",  # Trickle, dribble
    # vomiting (proxy classes)
    47: "vomiting",  # Cough
    56: "vomiting",  # Gargling
    57: "vomiting",  # Stomach rumble
    # ambient
    500: "ambient",  # Silence
    513: "ambient",  # Noise
    514: "ambient",  # Environmental noise
    515: "ambient",  # Static
    516: "ambient",  # Mains hum
    520: "ambient",  # White noise
    521: "ambient",  # Pink noise
}


@dataclass
class AudioPrediction:
    """Audio classification prediction result.

    Attributes:
        label: Predicted class name.
        confidence: Prediction confidence (0-1).
        class_scores: Confidence scores for all 5 classes.
    """

    label: str
    confidence: float
    class_scores: dict[str, float]


class AudioInference:
    """Zero-shot audio classifier using PANNs MobileNetV2.

    Loads pretrained AudioSet weights, aggregates predictions into
    our 5-class taxonomy. No fine-tuning in v1.

    Args:
        pretrained_path: Path to PANNs MobileNetV2 .pth checkpoint.
            If None, uses random weights (for testing only).
        device: Torch device string.
        sample_rate: Expected audio sample rate.
    """

    def __init__(
        self,
        pretrained_path: str | None = None,
        device: str | None = None,
        sample_rate: int = 16000,
    ):
        """Initialize AudioInference classifier.

        Args:
            pretrained_path: Path to PANNs MobileNetV2 .pth checkpoint.
                If None, uses random weights (for testing only).
            device: Torch device string. If None, auto-detected via detect_device().
            sample_rate: Expected audio sample rate.
        """
        if device is None:
            device = detect_device()
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self._pretrained_path = pretrained_path
        self.model = self._build_model(pretrained_path)
        self.model.eval()
        self.model.to(self.device)

    def _build_model(self, pretrained_path: str | None) -> torch.nn.Module:
        """Build MobileNetV2 backbone with AudioSet classification head.

        Uses the same architecture as PANNs for weight compatibility.
        This is a simplified version - loads full model for inference only.

        Args:
            pretrained_path: Path to .pth file, or None for random init.

        Returns:
            Loaded model in eval mode.
        """
        from pet_train.audio.arch import MobileNetV2AudioSet

        model = MobileNetV2AudioSet(num_classes=527, sample_rate=self.sample_rate)
        if pretrained_path is not None:
            checkpoint = torch.load(
                pretrained_path, map_location="cpu", weights_only=True
            )
            state_dict = checkpoint.get("model", checkpoint)
            result = model.load_state_dict(state_dict, strict=False)
            # F008 retro: log any architectural drift loudly so future maintainers
            # know if a checkpoint silently dropped layers (silent drop is what
            # let the F008 PANNs MobileNetV2 incompat slip past CI).
            if result.missing_keys or result.unexpected_keys:
                logger.warning(
                    "Pretrained weights partially loaded — missing=%d unexpected=%d "
                    "(likely architecture drift; consider PANNsAudioInference plugin)",
                    len(result.missing_keys),
                    len(result.unexpected_keys),
                )
            logger.info("Loaded pretrained weights from %s", pretrained_path)
        return model

    def _aggregate_scores(self, audioset_probs: torch.Tensor) -> dict[str, float]:
        """Aggregate AudioSet 527-class probabilities into our 5 classes.

        For each of our classes, take the max probability among its
        mapped AudioSet classes. "other" gets the max of all unmapped classes.

        Args:
            audioset_probs: Probability vector of shape [527].

        Returns:
            Dict mapping our 5 class names to confidence scores.
        """
        scores: dict[str, float] = {cls: 0.0 for cls in CLASSES}

        for idx, cls_name in AUDIOSET_CLASS_MAP.items():
            prob = audioset_probs[idx].item()
            scores[cls_name] = max(scores[cls_name], prob)

        # "other": max prob among all unmapped AudioSet classes
        mapped_indices = set(AUDIOSET_CLASS_MAP.keys())
        other_probs = [
            audioset_probs[i].item()
            for i in range(audioset_probs.shape[0])
            if i not in mapped_indices
        ]
        if other_probs:
            scores["other"] = max(other_probs)

        return scores

    @torch.no_grad()
    def predict(self, audio_path: str) -> AudioPrediction:
        """Classify an audio file into one of 5 pet feeder classes.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc).

        Returns:
            AudioPrediction with label, confidence, and per-class scores.
        """
        data, sr = sf.read(audio_path, dtype="float32")
        # soundfile returns (samples,) for mono, (samples, channels) for multi-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]  # [1, samples]
        else:
            data = data.T  # [channels, samples]
        waveform = torch.from_numpy(data)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(self.device)

        # Model expects [batch, samples]
        logits = self.model(waveform)  # [1, 527]
        probs = torch.sigmoid(logits).squeeze(0)  # [527] (multi-label)

        class_scores = self._aggregate_scores(probs)
        best_class = max(class_scores, key=lambda k: class_scores[k])

        return AudioPrediction(
            label=best_class,
            confidence=class_scores[best_class],
            class_scores=class_scores,
        )

    def get_weights_path(self) -> str | None:
        """Return the pretrained weights path for pet-quantize handoff.

        Returns:
            Path string or None if using random weights.
        """
        return self._pretrained_path
