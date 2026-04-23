"""Log-mel spectrogram feature extraction for audio classification.

Converts raw waveforms to log-mel spectrograms compatible with PANNs models.
Optionally applies SpecAugment during training.

params.yaml keys consumed (under audio:):
  sample_rate, n_mels, n_fft, hop_length, f_min, f_max
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as ta_transforms


class AudioTransform(nn.Module):
    """Extract log-mel spectrogram features from raw audio waveforms.

    Args:
        sample_rate: Target sample rate in Hz.
        n_mels: Number of mel filterbank bins.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency for mel filterbank.
        spec_augment: Whether to apply SpecAugment during training.
        time_mask_param: Maximum time mask length for SpecAugment.
        freq_mask_param: Maximum frequency mask length for SpecAugment.
        random_gain_db: Maximum random gain adjustment in dB. 0 to disable.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        spec_augment: bool = False,
        time_mask_param: int = 20,
        freq_mask_param: int = 10,
        random_gain_db: float = 0.0,
    ):
        """Initialize AudioTransform with mel spectrogram and optional SpecAugment."""
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.random_gain_db = random_gain_db

        self.mel_spectrogram = ta_transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        self.spec_augment = spec_augment
        if spec_augment:
            self.time_mask = ta_transforms.TimeMasking(time_mask_param=time_mask_param)
            self.freq_mask = ta_transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    @classmethod
    def from_params(cls, params: dict[str, Any], **override: Any) -> AudioTransform:
        """Construct from params.yaml audio sub-dict.

        Args:
            params: The ``audio:`` sub-dict from params.yaml.
            **override: Any explicit kwarg overrides (e.g. spec_augment=True).

        Returns:
            Configured AudioTransform instance.
        """
        kwargs = {
            "sample_rate": params["sample_rate"],
            "n_mels": params["n_mels"],
            "n_fft": params["n_fft"],
            "hop_length": params["hop_length"],
            "f_min": params["f_min"],
            "f_max": params["f_max"],
        }
        kwargs.update(override)
        return cls(**kwargs)

    def forward(
        self, waveform: torch.Tensor, input_sample_rate: int | None = None
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram.

        Args:
            waveform: Audio tensor of shape [channels, samples].
            input_sample_rate: Source sample rate. Resamples if different from target.

        Returns:
            Log-mel spectrogram of shape [channels, n_mels, time_frames].
        """
        if input_sample_rate is not None and input_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=input_sample_rate, new_freq=self.sample_rate
            )

        if self.training and self.random_gain_db > 0:
            gain = (torch.rand(1) * 2 - 1) * self.random_gain_db
            waveform = waveform * (10.0 ** (gain / 20.0))

        mel = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel + 1e-7)

        if self.training and self.spec_augment:
            log_mel = self.time_mask(log_mel)
            log_mel = self.freq_mask(log_mel)

        return log_mel
