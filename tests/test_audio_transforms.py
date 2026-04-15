"""Tests for audio feature extraction transforms."""


import torch

from pet_train.audio_transforms import AudioTransform


class TestAudioTransform:
    """Tests for log-mel spectrogram feature extraction."""

    def test_init_default_params(self):
        """Transform initializes with correct default parameters."""
        transform = AudioTransform(sample_rate=16000, n_mels=64)
        assert transform.sample_rate == 16000
        assert transform.n_mels == 64

    def test_output_shape(self):
        """Transform produces correct output shape for 1-second audio."""
        transform = AudioTransform(sample_rate=16000, n_mels=64)
        waveform = torch.randn(1, 16000)
        mel = transform(waveform)
        assert mel.dim() == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 64

    def test_resample_if_different_rate(self):
        """Transform resamples audio if input rate differs from target."""
        transform = AudioTransform(sample_rate=16000, n_mels=64)
        waveform = torch.randn(1, 32000)
        mel = transform(waveform, input_sample_rate=32000)
        assert mel.dim() == 3
        assert mel.shape[1] == 64

    def test_no_resample_if_same_rate(self):
        """Transform skips resampling when rates match."""
        transform = AudioTransform(sample_rate=16000, n_mels=64)
        waveform = torch.randn(1, 16000)
        mel = transform(waveform, input_sample_rate=16000)
        assert mel.dim() == 3

    def test_log_mel_values_finite(self):
        """Output values should all be finite (no NaN/Inf from log)."""
        transform = AudioTransform(sample_rate=16000, n_mels=64)
        waveform = torch.randn(1, 16000)
        mel = transform(waveform)
        assert torch.isfinite(mel).all()

    def test_spec_augment_training(self):
        """SpecAugment applies masking during training mode."""
        transform = AudioTransform(
            sample_rate=16000,
            n_mels=64,
            spec_augment=True,
            time_mask_param=20,
            freq_mask_param=10,
        )
        transform.train()
        waveform = torch.randn(1, 16000)
        mel = transform(waveform)
        assert mel.dim() == 3

    def test_spec_augment_disabled_eval(self):
        """SpecAugment does NOT apply during eval mode."""
        transform = AudioTransform(
            sample_rate=16000,
            n_mels=64,
            spec_augment=True,
            time_mask_param=20,
            freq_mask_param=10,
        )
        transform.eval()
        waveform = torch.randn(1, 16000)
        mel1 = transform(waveform)
        mel2 = transform(waveform)
        assert torch.allclose(mel1, mel2)
