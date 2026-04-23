"""Tests verifying audio transforms and arch respect params.yaml overrides (F6)."""

from __future__ import annotations

import torch

from pet_train.audio.arch import MobileNetV2AudioSet
from pet_train.audio.transforms import AudioTransform

SAMPLE_PARAMS = {
    "sample_rate": 22050,  # non-default value to prove override
    "n_mels": 32,  # non-default value
    "n_fft": 256,
    "hop_length": 80,
    "f_min": 80.0,
    "f_max": 4000.0,
}


def test_audio_transform_from_params_respects_overrides() -> None:
    """AudioTransform.from_params() must use params values, not hardcoded defaults."""
    transform = AudioTransform.from_params(SAMPLE_PARAMS)
    assert transform.sample_rate == 22050
    assert transform.n_mels == 32


def test_audio_transform_from_params_forward_shape() -> None:
    """AudioTransform built from params produces correct output shape."""
    transform = AudioTransform.from_params(SAMPLE_PARAMS)
    waveform = torch.randn(1, 22050)
    mel = transform(waveform)
    assert mel.dim() == 3
    assert mel.shape[1] == 32  # n_mels from params


def test_audio_transform_from_params_with_override() -> None:
    """from_params() kwargs override individual keys."""
    transform = AudioTransform.from_params(SAMPLE_PARAMS, spec_augment=True)
    assert transform.spec_augment is True


def test_mobilenetv2_from_params_num_classes_always_527() -> None:
    """MobileNetV2AudioSet.from_params() must always produce num_classes=527."""
    model = MobileNetV2AudioSet.from_params(SAMPLE_PARAMS)
    assert model.num_classes == 527


def test_mobilenetv2_from_params_mel_config() -> None:
    """MobileNetV2AudioSet.from_params() sets mel transform from params."""
    model = MobileNetV2AudioSet.from_params(SAMPLE_PARAMS)
    # Verify output shape — if n_mels or other params were wrong the conv would mismatch
    waveform = torch.randn(1, 1, 22050)
    with torch.no_grad():
        logits = model(waveform)
    assert logits.shape == (1, 527)
