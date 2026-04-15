"""PANNs MobileNetV2 architecture for AudioSet classification.

Minimal implementation compatible with PANNs pretrained weights.
Used by audio_inference.py for zero-shot inference.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as transforms


class _InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, inp: int, oup: int, expand_ratio: int, stride: int):
        """Initialize inverted residual block.

        Args:
            inp: Number of input channels.
            oup: Number of output channels.
            expand_ratio: Channel expansion ratio for hidden layer.
            stride: Convolution stride (1 or 2).
        """
        super().__init__()
        hidden = inp * expand_ratio
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.use_residual = stride == 1 and inp == oup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2AudioSet(nn.Module):
    """MobileNetV2 for AudioSet 527-class classification.

    Input: raw waveform [batch, 1, samples] or mel spectrogram.
    Output: logits [batch, 527].

    Architecture compatible with PANNs pretrained checkpoints.
    """

    def __init__(
        self,
        num_classes: int = 527,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ):
        """Initialize MobileNetV2AudioSet model.

        Args:
            num_classes: Number of output classes (527 for AudioSet).
            sample_rate: Audio sample rate (from params.yaml audio.sample_rate).
            n_mels: Number of mel bands (from params.yaml audio.n_mels).
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            f_min: Minimum frequency for mel filterbank.
            f_max: Maximum frequency for mel filterbank.
        """
        super().__init__()
        self.num_classes = num_classes

        # Mel spectrogram front-end — created ONCE here in __init__, not in forward()
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        # Mel spectrogram front-end convolutional stem
        self.mel_spec = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # MobileNetV2 backbone
        self.features = nn.Sequential(
            _InvertedResidual(32, 16, 1, 1),
            _InvertedResidual(16, 24, 6, 2),
            _InvertedResidual(24, 32, 6, 2),
            _InvertedResidual(32, 64, 6, 2),
            _InvertedResidual(64, 96, 6, 1),
            _InvertedResidual(96, 160, 6, 2),
            _InvertedResidual(160, 320, 6, 1),
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass from waveform to class logits.

        Args:
            waveform: Input tensor [batch, 1, samples] or [batch, channels, freq, time].

        Returns:
            Logits of shape [batch, num_classes].
        """
        # If input is 2D [batch, samples], add channel dim
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # If input is 3D [batch, 1, samples], compute mel spectrogram
        if waveform.dim() == 3:
            x = self.mel_transform(waveform)
            x = torch.log(x + 1e-7)
        else:
            x = waveform

        x = self.mel_spec(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
