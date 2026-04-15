# pet-train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete pet-train pipeline: SFT/DPO training via LLaMA-Factory, KL distillation with pluggable teacher logits, audio CNN transfer learning, and training-time schema compliance monitoring.

**Architecture:** Three independent training tracks (SFT, DPO, Audio CNN) sharing params.yaml and wandb project. VLM training driven by LLaMA-Factory v0.9.4 (vendor submodule). Audio training via PyTorch Lightning with PANNs MobileNetV2. KL distillation as optional SFT loss add-on with FileProvider and APIProvider backends.

**Tech Stack:** Python 3.11, PyTorch, LLaMA-Factory, PyTorch Lightning, PANNs, pet-schema v1.0.0, wandb, tenacity, httpx

**Spec:** `docs/superpowers/specs/2026-04-15-pet-train-design.md`

---

## File Structure

```
pet-train/
├── configs/
│   ├── base/
│   │   ├── sft_base.yaml
│   │   └── dpo_base.yaml
│   ├── experiments/
│   │   ├── sft_lora_r16_lr2e4_ep3.yaml
│   │   └── dpo_user_feedback_v1.yaml
│   └── audio/
│       └── mobilenetv2_transfer_v1.yaml
├── src/
│   ├── __init__.py
│   ├── kl_loss.py
│   ├── logits_provider/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── file_provider.py
│   │   └── api_provider.py
│   ├── schema_compliance_callback.py
│   ├── audio_model.py
│   ├── audio_datamodule.py
│   └── audio_transforms.py
├── scripts/
│   ├── train_sft.sh
│   ├── train_dpo.sh
│   ├── train_audio.sh
│   ├── merge_lora.sh
│   ├── collect_logits.sh
│   └── eval_after_train.sh
├── tests/
│   ├── conftest.py
│   ├── test_kl_loss.py
│   ├── test_logits_provider.py
│   ├── test_schema_compliance_callback.py
│   ├── test_audio_model.py
│   ├── test_audio_datamodule.py
│   └── test_audio_transforms.py
├── params.yaml
├── pyproject.toml
├── Makefile
└── .gitignore
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `params.yaml`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pet-train"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

dependencies = [
    "pet-schema @ git+https://github.com/Train-Pet-Pipeline/pet-schema.git@v1.0.0",
    "torch>=2.1",
    "torchaudio>=2.1",
    "pytorch-lightning>=2.1",
    "wandb",
    "pyyaml",
    "tenacity",
    "python-json-logger",
    "httpx",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
]

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.eggs/

# Training outputs
outputs/
data/

# IDE
.vscode/
.idea/

# Tools
.pytest_cache/
.mypy_cache/
.ruff_cache/

# OS
.DS_Store

# Wandb
wandb/
```

- [ ] **Step 3: Create params.yaml**

```yaml
# === SFT ===
sft:
  data_path: "data/pet_sft_train.jsonl"
  compliance_val_path: "data/compliance_val.jsonl"
  compliance_val_size: 50
  compliance_check_steps: 500
  compliance_sample_size: 20
  label_smoothing_factor: 0.1

# === KL Distillation ===
kl_distillation:
  enabled: false
  lambda_kl: 0.1
  temperature: 2.0
  top_k_approx: true
  top_k: 20
  provider: "file"
  logits_dir: "data/teacher_logits/"
  api:
    base_url: ""
    model_name: ""
    key_env: ""
    max_concurrent: 10
    timeout: 60
    max_retries: 3

# === DPO ===
dpo:
  min_pairs: 500
  pref_beta: 0.1
  sft_adapter_path: ""
  data_path: "data/pet_dpo_pairs.jsonl"
  learning_rate: 5.0e-5

# === Audio ===
audio:
  sample_rate: 16000
  n_mels: 64
  data_dir: "data/audio/"
  classes: ["eating", "drinking", "vomiting", "ambient", "other"]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

# === Common ===
wandb:
  project: "pet-train"
  entity: ""
```

- [ ] **Step 4: Create Makefile**

```makefile
.PHONY: setup test lint clean train-sft train-dpo train-audio merge collect-logits

setup:
	pip install -e ".[dev]"
	cd vendor/LLaMA-Factory && pip install -e ".[torch,metrics]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ && mypy src/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist/ *.egg-info

train-sft:
	bash scripts/train_sft.sh $(CONFIG)

train-dpo:
	bash scripts/train_dpo.sh $(CONFIG)

train-audio:
	bash scripts/train_audio.sh $(CONFIG)

merge:
	bash scripts/merge_lora.sh $(ADAPTER_PATH) $(OUTPUT_PATH)

collect-logits:
	bash scripts/collect_logits.sh
```

- [ ] **Step 5: Create src/__init__.py**

```python
"""pet-train: SFT/DPO training and audio CNN for smart pet feeder pipeline."""
```

- [ ] **Step 6: Create tests/conftest.py**

```python
"""Shared test fixtures for pet-train."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_params(tmp_dir):
    """Provide a minimal params.yaml for testing."""
    params = {
        "sft": {
            "data_path": str(tmp_dir / "train.jsonl"),
            "compliance_val_path": str(tmp_dir / "val.jsonl"),
            "compliance_val_size": 5,
            "compliance_check_steps": 10,
            "compliance_sample_size": 3,
            "label_smoothing_factor": 0.1,
        },
        "kl_distillation": {
            "enabled": False,
            "lambda_kl": 0.1,
            "temperature": 2.0,
            "top_k_approx": True,
            "top_k": 5,
            "provider": "file",
            "logits_dir": str(tmp_dir / "logits"),
        },
        "audio": {
            "sample_rate": 16000,
            "n_mels": 64,
            "data_dir": str(tmp_dir / "audio"),
            "classes": ["eating", "drinking", "vomiting", "ambient", "other"],
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
        },
    }
    params_path = tmp_dir / "params.yaml"
    params_path.write_text(yaml.dump(params))
    return params_path


@pytest.fixture
def sample_sharegpt_item():
    """Provide a single ShareGPT format training item."""
    return {
        "id": "sft_00001",
        "conversations": [
            {"from": "system", "value": "You are a pet feeder analyst."},
            {"from": "human", "value": "<image>\nAnalyze this feeding scene."},
            {
                "from": "gpt",
                "value": json.dumps(
                    {
                        "schema_version": "1.0",
                        "pet_present": True,
                        "pet_count": 1,
                        "pet": {
                            "species": "cat",
                            "action": {
                                "primary": "eating",
                                "distribution": {
                                    "eating": 0.85,
                                    "drinking": 0.05,
                                    "sniffing_only": 0.05,
                                    "leaving_bowl": 0.02,
                                    "sitting_idle": 0.02,
                                    "other": 0.01,
                                },
                            },
                            "eating_metrics": {
                                "duration_seconds": 120,
                                "speed": {
                                    "fast": 0.1,
                                    "normal": 0.7,
                                    "slow": 0.2,
                                },
                            },
                            "mood": {
                                "alertness": 0.6,
                                "anxiety": 0.1,
                                "engagement": 0.8,
                            },
                            "body_signals": {
                                "posture": "relaxed",
                                "ear_position": "forward",
                            },
                            "anomaly_signals": {
                                "vomit_gesture": 0.0,
                                "food_rejection": 0.0,
                                "excessive_sniffing": 0.05,
                                "lethargy": 0.0,
                                "aggression": 0.0,
                            },
                        },
                        "bowl": {
                            "food_fill_ratio": 0.6,
                            "water_fill_ratio": 0.8,
                            "food_type_visible": "dry_kibble",
                        },
                        "scene": {
                            "lighting": "normal",
                            "image_quality": "clear",
                            "confidence_overall": 0.92,
                        },
                        "narrative": "橘猫正常进食中，状态良好",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "images": ["data/frames/test_001.jpg"],
        "metadata": {
            "source": "selfshot",
            "schema_version": "1.0",
            "prompt_version": "1.0",
            "annotator": "qwen2.5-vl-72b",
            "review_status": "approved",
            "frame_id": "frame_test_001",
        },
    }
```

- [ ] **Step 7: Run lint to verify scaffolding**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-train && ruff check src/ tests/`
Expected: PASS (no errors)

- [ ] **Step 8: Commit scaffolding**

```bash
git add pyproject.toml Makefile params.yaml .gitignore src/__init__.py tests/conftest.py
git commit -m "feat(pet-train): add project scaffolding

pyproject.toml, Makefile, params.yaml, .gitignore, and test fixtures."
```

---

### Task 2: Audio Transforms

**Files:**
- Create: `src/audio_transforms.py`
- Create: `tests/test_audio_transforms.py`

- [ ] **Step 1: Write failing tests for audio_transforms**

```python
"""Tests for audio feature extraction transforms."""

import torch
import pytest
from unittest.mock import patch

from src.audio_transforms import AudioTransform


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
        # 1 second of 16kHz audio
        waveform = torch.randn(1, 16000)
        mel = transform(waveform)
        # Output should be [1, n_mels, time_frames]
        assert mel.dim() == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 64

    def test_resample_if_different_rate(self):
        """Transform resamples audio if input rate differs from target."""
        transform = AudioTransform(sample_rate=16000, n_mels=64)
        # 1 second of 32kHz audio
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
        # Eval mode should be deterministic (no random masking)
        assert torch.allclose(mel1, mel2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_transforms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.audio_transforms'`

- [ ] **Step 3: Implement audio_transforms.py**

```python
"""Log-mel spectrogram feature extraction for audio classification.

Converts raw waveforms to log-mel spectrograms compatible with PANNs models.
Optionally applies SpecAugment during training.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


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
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.random_gain_db = random_gain_db

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        self.spec_augment = spec_augment
        if spec_augment:
            self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_transforms.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/audio_transforms.py tests/test_audio_transforms.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/audio_transforms.py tests/test_audio_transforms.py
git commit -m "feat(pet-train): add audio transforms with log-mel spectrogram

SpecAugment support (time/freq masking), automatic resampling,
random gain augmentation. Compatible with PANNs 16kHz models."
```

---

### Task 3: Audio DataModule

**Files:**
- Create: `src/audio_datamodule.py`
- Create: `tests/test_audio_datamodule.py`

- [ ] **Step 1: Write failing tests for audio_datamodule**

```python
"""Tests for audio PyTorch Lightning DataModule."""

import json
from pathlib import Path

import pytest
import torch
import torchaudio

from src.audio_datamodule import AudioDataModule


def _create_wav(path: Path, sample_rate: int = 16000, duration_s: float = 1.0):
    """Helper to create a dummy WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * duration_s)
    waveform = torch.randn(1, samples)
    torchaudio.save(str(path), waveform, sample_rate)


class TestAudioDataModule:
    """Tests for AudioDataModule."""

    @pytest.fixture
    def audio_dir(self, tmp_dir):
        """Create a temporary audio dataset directory."""
        classes = ["eating", "drinking", "vomiting", "ambient", "other"]
        for cls in classes:
            cls_dir = tmp_dir / "audio" / cls
            for i in range(10):
                _create_wav(cls_dir / f"{cls}_{i:03d}.wav")
        return tmp_dir / "audio"

    def test_init(self, audio_dir):
        """DataModule initializes with correct parameters."""
        dm = AudioDataModule(
            data_dir=str(audio_dir),
            classes=["eating", "drinking", "vomiting", "ambient", "other"],
            sample_rate=16000,
            n_mels=64,
            batch_size=4,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        )
        assert dm.batch_size == 4
        assert dm.num_classes == 5

    def test_setup_creates_splits(self, audio_dir):
        """setup() creates train/val/test splits with correct proportions."""
        dm = AudioDataModule(
            data_dir=str(audio_dir),
            classes=["eating", "drinking", "vomiting", "ambient", "other"],
            sample_rate=16000,
            n_mels=64,
            batch_size=4,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        )
        dm.setup(stage="fit")
        total = len(dm.train_dataset) + len(dm.val_dataset)
        # 50 total files, ~40 train, ~5 val
        assert len(dm.train_dataset) > len(dm.val_dataset)
        assert total == 45  # 50 * 0.9 (test excluded in fit stage)

    def test_train_dataloader_returns_batches(self, audio_dir):
        """train_dataloader() yields (mel, label) batches."""
        dm = AudioDataModule(
            data_dir=str(audio_dir),
            classes=["eating", "drinking", "vomiting", "ambient", "other"],
            sample_rate=16000,
            n_mels=64,
            batch_size=4,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        )
        dm.setup(stage="fit")
        batch = next(iter(dm.train_dataloader()))
        mel, label = batch
        assert mel.dim() == 3  # [batch, n_mels, time]
        assert mel.shape[0] <= 4
        assert mel.shape[1] == 64
        assert label.dim() == 1
        assert label.dtype == torch.long

    def test_class_to_idx_mapping(self, audio_dir):
        """Classes are mapped to integer indices consistently."""
        dm = AudioDataModule(
            data_dir=str(audio_dir),
            classes=["eating", "drinking", "vomiting", "ambient", "other"],
            sample_rate=16000,
            n_mels=64,
            batch_size=4,
        )
        dm.setup(stage="fit")
        assert dm.class_to_idx == {
            "eating": 0,
            "drinking": 1,
            "vomiting": 2,
            "ambient": 3,
            "other": 4,
        }

    def test_empty_class_dir_skipped(self, audio_dir):
        """Empty class directories don't cause errors."""
        empty_dir = audio_dir / "empty_class"
        empty_dir.mkdir()
        dm = AudioDataModule(
            data_dir=str(audio_dir),
            classes=["eating", "drinking", "vomiting", "ambient", "other"],
            sample_rate=16000,
            n_mels=64,
            batch_size=4,
        )
        dm.setup(stage="fit")
        # Should still work, just with 5 classes from the known list
        assert dm.num_classes == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_datamodule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.audio_datamodule'`

- [ ] **Step 3: Implement audio_datamodule.py**

```python
"""PyTorch Lightning DataModule for audio classification.

Loads audio files from class-organized directories, applies transforms,
and creates train/val/test DataLoaders.
"""

from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split

from src.audio_transforms import AudioTransform


class AudioFileDataset(Dataset):
    """Dataset of audio files organized in class subdirectories.

    Args:
        file_paths: List of (file_path, label_idx) tuples.
        transform: AudioTransform instance for feature extraction.
        sample_rate: Expected sample rate of audio files.
    """

    def __init__(
        self,
        file_paths: list[tuple[Path, int]],
        transform: AudioTransform,
        sample_rate: int,
    ):
        self.file_paths = file_paths
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        """Return the number of audio samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and transform a single audio sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (log-mel spectrogram, label index).
        """
        path, label = self.file_paths[idx]
        waveform, sr = torchaudio.load(str(path))
        mel = self.transform(waveform, input_sample_rate=sr)
        # Remove channel dim: [1, n_mels, time] -> [n_mels, time]
        mel = mel.squeeze(0)
        return mel, label


class AudioDataModule(pl.LightningDataModule):
    """Lightning DataModule for audio classification training.

    Args:
        data_dir: Root directory containing class subdirectories.
        classes: Ordered list of class names (determines label indices).
        sample_rate: Target audio sample rate.
        n_mels: Number of mel filterbank bins.
        batch_size: Batch size for DataLoaders.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        num_workers: Number of DataLoader workers.
        spec_augment: Whether to apply SpecAugment during training.
        time_mask_param: SpecAugment time mask parameter.
        freq_mask_param: SpecAugment frequency mask parameter.
        random_gain_db: Random gain augmentation range in dB.
    """

    def __init__(
        self,
        data_dir: str,
        classes: list[str],
        sample_rate: int = 16000,
        n_mels: int = 64,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 4,
        spec_augment: bool = False,
        time_mask_param: int = 20,
        freq_mask_param: int = 10,
        random_gain_db: float = 0.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.num_classes = len(classes)

        self.train_transform = AudioTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            spec_augment=spec_augment,
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            random_gain_db=random_gain_db,
        )
        self.eval_transform = AudioTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            spec_augment=False,
        )

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _scan_files(self) -> list[tuple[Path, int]]:
        """Scan data directory for audio files in class subdirectories.

        Returns:
            List of (file_path, label_idx) tuples.
        """
        file_paths = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.data_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                    file_paths.append((f, cls_idx))
        return file_paths

    def setup(self, stage: str | None = None) -> None:
        """Create train/val/test splits from audio files.

        Args:
            stage: Lightning stage ('fit', 'test', or None for both).
        """
        all_files = self._scan_files()
        total = len(all_files)

        n_test = int(total * self.test_split)
        n_val = int(total * self.val_split)
        n_train = total - n_test - n_val

        generator = torch.Generator().manual_seed(42)
        train_files, val_files, test_files = random_split(
            all_files, [n_train, n_val, n_test], generator=generator
        )

        if stage in ("fit", None):
            self.train_transform.train()
            self.train_dataset = AudioFileDataset(
                list(train_files), self.train_transform, self.sample_rate
            )
            self.eval_transform.eval()
            self.val_dataset = AudioFileDataset(
                list(val_files), self.eval_transform, self.sample_rate
            )

        if stage in ("test", None):
            self.eval_transform.eval()
            self.test_dataset = AudioFileDataset(
                list(test_files), self.eval_transform, self.sample_rate
            )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with shuffle."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_datamodule.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/audio_datamodule.py tests/test_audio_datamodule.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/audio_datamodule.py tests/test_audio_datamodule.py
git commit -m "feat(pet-train): add audio DataModule with train/val/test splits

Lightning DataModule, class-organized directory scanning, configurable
splits with fixed seed for reproducibility."
```

---

### Task 4: Audio Model

**Files:**
- Create: `src/audio_model.py`
- Create: `tests/test_audio_model.py`

- [ ] **Step 1: Write failing tests for audio_model**

```python
"""Tests for PANNs MobileNetV2 audio classification model."""

import pytest
import torch

from src.audio_model import AudioClassifier


class TestAudioClassifier:
    """Tests for AudioClassifier Lightning module."""

    def test_init_default(self):
        """Model initializes with correct number of classes."""
        model = AudioClassifier(num_classes=5, pretrained_path=None)
        assert model.num_classes == 5

    def test_forward_shape(self):
        """Forward pass returns correct output shape."""
        model = AudioClassifier(num_classes=5, pretrained_path=None)
        # Simulate a log-mel spectrogram batch: [batch, n_mels, time]
        x = torch.randn(4, 64, 100)
        logits = model(x)
        assert logits.shape == (4, 5)

    def test_forward_no_nan(self):
        """Forward pass produces finite values."""
        model = AudioClassifier(num_classes=5, pretrained_path=None)
        x = torch.randn(2, 64, 100)
        logits = model(x)
        assert torch.isfinite(logits).all()

    def test_training_step_returns_loss(self):
        """training_step returns a scalar loss."""
        model = AudioClassifier(num_classes=5, pretrained_path=None)
        x = torch.randn(4, 64, 100)
        y = torch.tensor([0, 1, 2, 3])
        loss = model.training_step((x, y), batch_idx=0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_freeze_backbone(self):
        """When freeze_backbone=True, backbone params don't require grad."""
        model = AudioClassifier(
            num_classes=5, pretrained_path=None, freeze_backbone=True
        )
        for param in model.backbone.parameters():
            assert not param.requires_grad
        # Classification head should still require grad
        for param in model.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self):
        """When freeze_backbone=False, all params require grad."""
        model = AudioClassifier(
            num_classes=5, pretrained_path=None, freeze_backbone=False
        )
        for param in model.parameters():
            assert param.requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.audio_model'`

- [ ] **Step 3: Implement audio_model.py**

```python
"""PANNs MobileNetV2 audio classification with transfer learning.

Loads a pretrained MobileNetV2 backbone from PANNs (AudioSet) and replaces
the classification head for custom audio event classes.
"""

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix

logger = logging.getLogger(__name__)


class _MobileNetV2Backbone(nn.Module):
    """MobileNetV2-style backbone compatible with PANNs pretrained weights.

    Simplified architecture for audio classification. When pretrained_path is
    provided, loads PANNs weights. Otherwise initializes randomly.

    Args:
        pretrained_path: Path to PANNs MobileNetV2 checkpoint (.pth).
    """

    def __init__(self, pretrained_path: str | None = None):
        super().__init__()
        # MobileNetV2 feature extractor
        # Input: [batch, 1, n_mels, time] -> Output: [batch, 1280]
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # Inverted residual blocks (simplified)
            self._inverted_residual(32, 16, 1, 1),
            self._inverted_residual(16, 24, 6, 2),
            self._inverted_residual(24, 32, 6, 2),
            self._inverted_residual(32, 64, 6, 2),
            self._inverted_residual(64, 96, 6, 1),
            self._inverted_residual(96, 160, 6, 2),
            self._inverted_residual(160, 320, 6, 1),
            # Final conv
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 1280

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    @staticmethod
    def _inverted_residual(
        inp: int, oup: int, expand_ratio: int, stride: int
    ) -> nn.Sequential:
        """Create a single inverted residual block.

        Args:
            inp: Input channels.
            oup: Output channels.
            expand_ratio: Expansion factor for hidden dim.
            stride: Stride for depthwise conv.

        Returns:
            Sequential block.
        """
        hidden = inp * expand_ratio
        layers = []
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
        return nn.Sequential(*layers)

    def _load_pretrained(self, path: str) -> None:
        """Load PANNs pretrained weights, ignoring mismatched layers.

        Args:
            path: Path to .pth checkpoint file.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        # Filter to matching keys only
        model_dict = self.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(filtered)
        self.load_state_dict(model_dict)
        logger.info(
            "Loaded %d/%d pretrained parameters from %s",
            len(filtered),
            len(model_dict),
            path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from log-mel spectrogram.

        Args:
            x: Input tensor of shape [batch, n_mels, time].

        Returns:
            Feature vector of shape [batch, 1280].
        """
        # Add channel dim: [batch, n_mels, time] -> [batch, 1, n_mels, time]
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class AudioClassifier(pl.LightningModule):
    """Audio event classifier using MobileNetV2 backbone with transfer learning.

    Args:
        num_classes: Number of output classes.
        pretrained_path: Path to PANNs pretrained checkpoint. None for random init.
        freeze_backbone: Whether to freeze backbone parameters.
        learning_rate: Optimizer learning rate.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained_path: str | None = None,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.backbone = _MobileNetV2Backbone(pretrained_path=pretrained_path)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.output_dim, num_classes),
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_confusion = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier.

        Args:
            x: Log-mel spectrogram of shape [batch, n_mels, time].

        Returns:
            Class logits of shape [batch, num_classes].
        """
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute training loss and accuracy.

        Args:
            batch: Tuple of (mel_spectrogram, label).
            batch_idx: Batch index.

        Returns:
            Cross-entropy loss.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Compute validation loss and accuracy.

        Args:
            batch: Tuple of (mel_spectrogram, label).
            batch_idx: Batch index.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Compute test accuracy and confusion matrix.

        Args:
            batch: Tuple of (mel_spectrogram, label).
            batch_idx: Batch index.
        """
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.test_confusion(logits, y)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure Adam optimizer with cosine annealing schedule.

        Returns:
            Optimizer and scheduler configuration dict.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main():
    """CLI entry point for audio model training."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Train audio classifier")
    parser.add_argument("--config", required=True, help="Audio config YAML path")
    parser.add_argument("--params", required=True, help="params.yaml path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--run_name", required=True, help="Wandb run name")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.params) as f:
        params = yaml.safe_load(f)

    audio_cfg = params["audio"]
    wandb_cfg = params.get("wandb", {})

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    from src.audio_datamodule import AudioDataModule

    dm = AudioDataModule(
        data_dir=audio_cfg["data_dir"],
        classes=audio_cfg["classes"],
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["n_mels"],
        batch_size=config["training"]["batch_size"],
        train_split=audio_cfg["train_split"],
        val_split=audio_cfg["val_split"],
        test_split=audio_cfg["test_split"],
        spec_augment=config.get("augmentation", {}).get("spec_augment", False),
        time_mask_param=config.get("augmentation", {}).get("time_mask_param", 20),
        freq_mask_param=config.get("augmentation", {}).get("freq_mask_param", 10),
        random_gain_db=config.get("augmentation", {}).get("random_gain_db", 0.0),
    )

    model = AudioClassifier(
        num_classes=config["model"]["num_classes"],
        pretrained_path=config["model"].get("pretrained_path"),
        freeze_backbone=config["model"].get("freeze_backbone", True),
        learning_rate=config["training"]["learning_rate"],
    )

    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "pet-train"),
        name=args.run_name,
        group="audio",
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(
                monitor="val/loss",
                patience=config["training"]["early_stopping_patience"],
            ),
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="best",
                monitor="val/loss",
                save_top_k=1,
            ),
        ],
        default_root_dir=args.output_dir,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_model.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/audio_model.py tests/test_audio_model.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/audio_model.py tests/test_audio_model.py
git commit -m "feat(pet-train): add MobileNetV2 audio classifier

PANNs-compatible backbone with transfer learning, freeze/unfreeze support,
Lightning Trainer integration with accuracy and confusion matrix metrics."
```

---

### Task 5: KL Loss Module

**Files:**
- Create: `src/kl_loss.py`
- Create: `tests/test_kl_loss.py`

- [ ] **Step 1: Write failing tests for kl_loss**

```python
"""Tests for KL distillation loss computation."""

import pytest
import torch

from src.kl_loss import compute_kl_distillation_loss, compute_topk_kl_loss


class TestFullVocabKL:
    """Tests for full vocabulary KL distillation loss."""

    def test_identical_distributions_zero_loss(self):
        """KL loss is near-zero when student matches teacher exactly."""
        logits = torch.randn(2, 10, 100)  # [batch, seq_len, vocab]
        loss = compute_kl_distillation_loss(
            student_logits=logits,
            teacher_logits=logits,
            temperature=2.0,
            lambda_kl=0.1,
        )
        assert loss.item() < 1e-5

    def test_different_distributions_positive_loss(self):
        """KL loss is positive when distributions differ."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss = compute_kl_distillation_loss(
            student_logits=student,
            teacher_logits=teacher,
            temperature=2.0,
            lambda_kl=0.1,
        )
        assert loss.item() > 0

    def test_lambda_scales_loss(self):
        """Lambda parameter scales the loss proportionally."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss_01 = compute_kl_distillation_loss(
            student, teacher, temperature=2.0, lambda_kl=0.1
        )
        loss_02 = compute_kl_distillation_loss(
            student, teacher, temperature=2.0, lambda_kl=0.2
        )
        assert abs(loss_02.item() / loss_01.item() - 2.0) < 0.01

    def test_temperature_softens_distribution(self):
        """Higher temperature produces lower KL (softer distributions)."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss_t1 = compute_kl_distillation_loss(
            student, teacher, temperature=1.0, lambda_kl=1.0
        )
        loss_t4 = compute_kl_distillation_loss(
            student, teacher, temperature=4.0, lambda_kl=1.0
        )
        # T^2 scaling compensates, but softer distributions still differ less
        # At very high temperature, KL should be lower
        assert loss_t4.item() < loss_t1.item()

    def test_output_is_scalar(self):
        """Loss output is a scalar tensor."""
        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)
        loss = compute_kl_distillation_loss(student, teacher)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """Gradients flow through the loss to student logits."""
        student = torch.randn(2, 10, 100, requires_grad=True)
        teacher = torch.randn(2, 10, 100)
        loss = compute_kl_distillation_loss(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert torch.isfinite(student.grad).all()


class TestTopKKL:
    """Tests for top-k approximate KL distillation loss."""

    def test_output_is_scalar(self):
        """Top-k KL returns a scalar loss."""
        student_logits = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        teacher_top_k_ids = torch.randint(0, 100, (2, 10, 5))  # top-5
        teacher_top_k_logprobs = torch.randn(2, 10, 5)
        loss = compute_topk_kl_loss(
            student_logits=student_logits,
            teacher_token_ids=teacher_top_k_ids,
            teacher_logprobs=teacher_top_k_logprobs,
            temperature=2.0,
            lambda_kl=0.1,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self):
        """Gradients flow through top-k KL loss."""
        student = torch.randn(2, 10, 100, requires_grad=True)
        ids = torch.randint(0, 100, (2, 10, 5))
        logprobs = torch.randn(2, 10, 5)
        loss = compute_topk_kl_loss(student, ids, logprobs)
        loss.backward()
        assert student.grad is not None

    def test_higher_k_closer_to_full(self):
        """With k approaching vocab_size, top-k KL approaches full KL."""
        vocab = 50
        student = torch.randn(2, 5, vocab)
        teacher = torch.randn(2, 5, vocab)

        full_loss = compute_kl_distillation_loss(
            student, teacher, temperature=2.0, lambda_kl=0.1
        )

        # Simulate top-k with k=vocab (all tokens)
        teacher_probs = torch.softmax(teacher / 2.0, dim=-1)
        top_k_logprobs = torch.log(teacher_probs)
        top_k_ids = torch.arange(vocab).unsqueeze(0).unsqueeze(0).expand(2, 5, vocab)

        topk_loss = compute_topk_kl_loss(
            student, top_k_ids, top_k_logprobs, temperature=2.0, lambda_kl=0.1
        )

        # Should be reasonably close
        assert abs(full_loss.item() - topk_loss.item()) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kl_loss.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.kl_loss'`

- [ ] **Step 3: Implement kl_loss.py**

```python
"""KL distillation loss for knowledge distillation from teacher to student model.

Supports two modes:
- Full vocabulary KL: exact KL divergence with temperature softening
- Top-k approximate KL: KL computed only over teacher's top-k tokens,
  remaining probability mass consolidated into a rest bucket.
"""

import torch
import torch.nn.functional as F


def compute_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    lambda_kl: float = 0.1,
) -> torch.Tensor:
    """Compute full-vocabulary KL distillation loss with temperature softening.

    Args:
        student_logits: Student model output logits [batch, seq_len, vocab_size].
        teacher_logits: Teacher model output logits [batch, seq_len, vocab_size].
        temperature: Softening temperature. Higher values produce softer distributions.
        lambda_kl: Scaling factor for KL loss term.

    Returns:
        Scalar KL distillation loss.
    """
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    return lambda_kl * (temperature ** 2) * kl


def compute_topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    temperature: float = 2.0,
    lambda_kl: float = 0.1,
) -> torch.Tensor:
    """Compute top-k approximate KL distillation loss.

    When teacher logits are unavailable (API only returns top-k logprobs),
    compute KL only over the teacher's top-k tokens. Remaining probability
    mass is assigned to a single rest bucket.

    Args:
        student_logits: Student logits [batch, seq_len, vocab_size].
        teacher_token_ids: Token IDs for teacher's top-k [batch, seq_len, k].
        teacher_logprobs: Teacher's log probabilities for top-k [batch, seq_len, k].
        temperature: Softening temperature.
        lambda_kl: Scaling factor for KL loss term.

    Returns:
        Scalar approximate KL distillation loss.
    """
    # Teacher top-k probabilities (already from softmax, stored as logprobs)
    teacher_topk_probs = torch.exp(teacher_logprobs)  # [batch, seq, k]
    teacher_rest_prob = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(
        min=1e-8
    )

    # Student log-probs at teacher's top-k positions
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    student_topk_logprobs = torch.gather(
        student_log_prob, dim=-1, index=teacher_token_ids
    )

    # Student rest bucket: log(1 - sum(top-k probs))
    student_topk_probs = torch.gather(
        F.softmax(student_logits / temperature, dim=-1),
        dim=-1,
        index=teacher_token_ids,
    )
    student_rest_logprob = torch.log(
        (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    )

    # KL = sum(teacher_p * (log(teacher_p) - log(student_p)))
    # Top-k terms
    kl_topk = teacher_topk_probs * (teacher_logprobs - student_topk_logprobs)
    # Rest bucket term
    teacher_rest_logprob = torch.log(teacher_rest_prob)
    kl_rest = teacher_rest_prob * (teacher_rest_logprob - student_rest_logprob)

    kl = (kl_topk.sum(dim=-1) + kl_rest.squeeze(-1)).mean()
    return lambda_kl * (temperature ** 2) * kl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kl_loss.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/kl_loss.py tests/test_kl_loss.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/kl_loss.py tests/test_kl_loss.py
git commit -m "feat(pet-train): add KL distillation loss module

Full-vocab and top-k approximate KL modes. Temperature softening,
lambda scaling. Top-k uses rest-bucket consolidation for missing
probability mass."
```

---

### Task 6: Logits Provider

**Files:**
- Create: `src/logits_provider/__init__.py`
- Create: `src/logits_provider/base.py`
- Create: `src/logits_provider/file_provider.py`
- Create: `src/logits_provider/api_provider.py`
- Create: `tests/test_logits_provider.py`

- [ ] **Step 1: Write failing tests for logits providers**

```python
"""Tests for teacher logits provider implementations."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import torch

from src.logits_provider.base import LogitsResult, TeacherLogitsProvider
from src.logits_provider.file_provider import FileLogitsProvider
from src.logits_provider.api_provider import APILogitsProvider


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

        # Create sample logits files
        for i in range(3):
            sample_id = f"sft_{i:05d}"
            data = {
                "token_ids": torch.randint(0, 1000, (20, 10)),
                "logprobs": torch.randn(20, 10),
            }
            torch.save(data, ld / f"{sample_id}.pt")

        # Create manifest
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
        assert not result.is_full_vocab  # top_k=10 in manifest

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
        # Pre-cache a result
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_logits_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base.py**

```python
"""Abstract base class for teacher logits providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class LogitsResult:
    """Container for teacher model logits.

    Attributes:
        token_ids: Token IDs. Shape [seq_len, vocab_size] for full vocab,
            or [seq_len, k] for top-k.
        logprobs: Log probabilities corresponding to token_ids. Same shape.
        is_full_vocab: True if token_ids covers the full vocabulary,
            False if only top-k tokens.
    """

    token_ids: torch.Tensor
    logprobs: torch.Tensor
    is_full_vocab: bool


class TeacherLogitsProvider(ABC):
    """Abstract interface for loading teacher model logits.

    Implementations provide logits from different sources (disk files, API cache).
    """

    @abstractmethod
    def get_logits(self, sample_id: str) -> LogitsResult:
        """Retrieve teacher logits for a training sample.

        Args:
            sample_id: Unique identifier matching the training data ID field.

        Returns:
            LogitsResult containing teacher's token IDs and log probabilities.

        Raises:
            KeyError: If sample_id is not available.
        """
        ...

    @property
    @abstractmethod
    def available_samples(self) -> set[str]:
        """Return set of sample IDs that have cached logits."""
        ...
```

- [ ] **Step 4: Implement file_provider.py**

```python
"""File-based teacher logits provider.

Reads pre-computed logits from .pt files on disk. Supports logits from
any teacher model (offline local inference or pre-cached API results).
"""

import json
import logging
from pathlib import Path

import torch

from src.logits_provider.base import LogitsResult, TeacherLogitsProvider

logger = logging.getLogger(__name__)


class FileLogitsProvider(TeacherLogitsProvider):
    """Load teacher logits from .pt files indexed by a manifest.

    Args:
        logits_dir: Directory containing .pt files and manifest.json.
    """

    def __init__(self, logits_dir: str):
        self.logits_dir = Path(logits_dir)
        self._manifest = self._load_manifest()
        self._is_full_vocab = "top_k" not in self._manifest

    def _load_manifest(self) -> dict:
        """Load and validate manifest.json.

        Returns:
            Parsed manifest dictionary.

        Raises:
            FileNotFoundError: If manifest.json does not exist.
        """
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
            sample_id: Training sample identifier.

        Returns:
            LogitsResult with teacher logits.

        Raises:
            KeyError: If sample_id not in manifest.
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
```

- [ ] **Step 5: Implement api_provider.py**

```python
"""API-based teacher logits provider with disk caching.

Fetches top-k logprobs from OpenAI-compatible APIs and caches results
to disk for reuse across training runs.
"""

import json
import logging
import os
from pathlib import Path

import torch

from src.logits_provider.base import LogitsResult, TeacherLogitsProvider

logger = logging.getLogger(__name__)


class APILogitsProvider(TeacherLogitsProvider):
    """Fetch teacher logprobs from API with transparent disk caching.

    First call for a sample_id hits the API and caches the result.
    Subsequent calls read from cache. Uses the same .pt format as
    FileLogitsProvider for interoperability.

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
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.top_k = top_k
        self.timeout = timeout
        self.max_retries = max_retries
        self._manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> dict:
        """Load existing manifest or create a new one.

        Returns:
            Manifest dictionary.
        """
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
        collect_sample() for each missing sample.

        Args:
            sample_id: Training sample identifier.

        Returns:
            LogitsResult with teacher top-k logprobs.

        Raises:
            KeyError: If sample_id is not in cache.
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
            is_full_vocab=False,
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
        filename = f"{sample_id}.pt"
        torch.save(
            {"token_ids": token_ids, "logprobs": logprobs},
            self.cache_dir / filename,
        )
        self._manifest["samples"][sample_id] = filename
        self._save_manifest()
        logger.info("Cached logits for sample %s", sample_id)
```

- [ ] **Step 6: Create __init__.py**

```python
"""Teacher logits provider implementations for KL distillation."""

from src.logits_provider.base import LogitsResult, TeacherLogitsProvider
from src.logits_provider.file_provider import FileLogitsProvider
from src.logits_provider.api_provider import APILogitsProvider

__all__ = [
    "LogitsResult",
    "TeacherLogitsProvider",
    "FileLogitsProvider",
    "APILogitsProvider",
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_logits_provider.py -v`
Expected: All 7 tests PASS

- [ ] **Step 8: Run lint**

Run: `ruff check src/logits_provider/ tests/test_logits_provider.py`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/logits_provider/ tests/test_logits_provider.py
git commit -m "feat(pet-train): add teacher logits provider module

FileProvider for pre-computed .pt logits, APIProvider with disk caching
for OpenAI-compatible API logprobs. Manifest-indexed, pluggable interface."
```

---

### Task 7: Schema Compliance Callback

**Files:**
- Create: `src/schema_compliance_callback.py`
- Create: `tests/test_schema_compliance_callback.py`

- [ ] **Step 1: Write failing tests for schema_compliance_callback**

```python
"""Tests for training-time schema compliance monitoring callback."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.schema_compliance_callback import (
    SchemaComplianceCallback,
    load_compliance_val_set,
    evaluate_compliance,
)


class TestLoadComplianceValSet:
    """Tests for loading the compliance validation subset."""

    def test_loads_jsonl(self, tmp_dir, sample_sharegpt_item):
        """Loads items from a JSONL file."""
        path = tmp_dir / "val.jsonl"
        with open(path, "w") as f:
            for _ in range(10):
                f.write(json.dumps(sample_sharegpt_item) + "\n")
        items = load_compliance_val_set(str(path))
        assert len(items) == 10
        assert items[0]["id"] == "sft_00001"

    def test_empty_file_returns_empty(self, tmp_dir):
        """Empty file returns empty list."""
        path = tmp_dir / "empty.jsonl"
        path.write_text("")
        items = load_compliance_val_set(str(path))
        assert items == []


class TestEvaluateCompliance:
    """Tests for schema compliance evaluation logic."""

    def test_valid_output_passes(self):
        """Valid JSON output that passes schema validation counts as pass."""
        valid_output = json.dumps({
            "schema_version": "1.0",
            "pet_present": True,
            "pet_count": 1,
            "pet": {
                "species": "cat",
                "action": {
                    "primary": "eating",
                    "distribution": {
                        "eating": 0.85, "drinking": 0.05,
                        "sniffing_only": 0.05, "leaving_bowl": 0.02,
                        "sitting_idle": 0.02, "other": 0.01,
                    },
                },
                "eating_metrics": {
                    "duration_seconds": 120,
                    "speed": {"fast": 0.1, "normal": 0.7, "slow": 0.2},
                },
                "mood": {"alertness": 0.6, "anxiety": 0.1, "engagement": 0.8},
                "body_signals": {"posture": "relaxed", "ear_position": "forward"},
                "anomaly_signals": {
                    "vomit_gesture": 0.0, "food_rejection": 0.0,
                    "excessive_sniffing": 0.05, "lethargy": 0.0,
                    "aggression": 0.0,
                },
            },
            "bowl": {
                "food_fill_ratio": 0.6,
                "water_fill_ratio": 0.8,
                "food_type_visible": "dry_kibble",
            },
            "scene": {
                "lighting": "normal",
                "image_quality": "clear",
                "confidence_overall": 0.92,
            },
            "narrative": "橘猫正常进食中",
        })
        result = evaluate_compliance([valid_output])
        assert result["passed"] == 1
        assert result["total"] == 1
        assert result["compliance_rate"] == 1.0

    def test_invalid_json_fails(self):
        """Non-JSON output counts as failure."""
        result = evaluate_compliance(["not json at all"])
        assert result["passed"] == 0
        assert result["total"] == 1
        assert len(result["failed_reasons"]) == 1

    def test_mixed_results(self):
        """Compliance rate is correct for mixed valid/invalid outputs."""
        valid = json.dumps({"schema_version": "1.0", "pet_present": False,
                           "pet_count": 0, "pet": None,
                           "bowl": {"food_fill_ratio": 0.5,
                                    "water_fill_ratio": 0.5,
                                    "food_type_visible": "dry_kibble"},
                           "scene": {"lighting": "normal",
                                     "image_quality": "clear",
                                     "confidence_overall": 0.8},
                           "narrative": "无宠物"})
        invalid = "garbage output"
        result = evaluate_compliance([valid, invalid, valid])
        assert result["compliance_rate"] == pytest.approx(2 / 3)


class TestSchemaComplianceCallback:
    """Tests for the TrainerCallback wrapper."""

    def test_init_with_params(self, tmp_dir, sample_sharegpt_item):
        """Callback initializes with validation data path."""
        path = tmp_dir / "val.jsonl"
        with open(path, "w") as f:
            for _ in range(5):
                f.write(json.dumps(sample_sharegpt_item) + "\n")

        callback = SchemaComplianceCallback(
            val_path=str(path),
            check_steps=10,
            sample_size=3,
            output_dir=str(tmp_dir / "logs"),
        )
        assert len(callback.val_items) == 5
        assert callback.sample_size == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema_compliance_callback.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement schema_compliance_callback.py**

```python
"""Training-time schema compliance monitoring callback.

Periodically samples model outputs during training and validates them
against pet-schema to detect training collapse early.
"""

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from pet_schema import validate_output

logger = logging.getLogger(__name__)


def load_compliance_val_set(path: str) -> list[dict]:
    """Load compliance validation items from JSONL file.

    Args:
        path: Path to JSONL file in ShareGPT format.

    Returns:
        List of parsed JSON items.
    """
    items = []
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Compliance validation file not found: %s", path)
        return items
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d compliance validation items from %s", len(items), path)
    return items


def evaluate_compliance(model_outputs: list[str]) -> dict:
    """Evaluate schema compliance rate for a batch of model outputs.

    Args:
        model_outputs: List of raw model output strings (expected JSON).

    Returns:
        Dict with keys: total, passed, compliance_rate, failed_ids, failed_reasons.
    """
    total = len(model_outputs)
    passed = 0
    failed_reasons = []

    for i, output in enumerate(model_outputs):
        try:
            result = validate_output(output)
            if result.is_valid:
                passed += 1
            else:
                failed_reasons.append({
                    "index": i,
                    "errors": [str(e) for e in result.errors],
                })
        except Exception as e:
            failed_reasons.append({
                "index": i,
                "errors": [f"Validation exception: {e}"],
            })

    return {
        "total": total,
        "passed": passed,
        "compliance_rate": passed / total if total > 0 else 0.0,
        "failed_reasons": failed_reasons,
    }


class SchemaComplianceCallback:
    """LLaMA-Factory compatible callback for schema compliance monitoring.

    Triggers every `check_steps` steps, samples `sample_size` items from
    the validation set, generates model outputs, and validates them.

    Args:
        val_path: Path to compliance validation JSONL file.
        check_steps: Run compliance check every N training steps.
        sample_size: Number of items to sample per check.
        output_dir: Directory for compliance_log.jsonl output.
    """

    def __init__(
        self,
        val_path: str,
        check_steps: int = 500,
        sample_size: int = 20,
        output_dir: str = "outputs/compliance",
    ):
        self.val_items = load_compliance_val_set(val_path)
        self.check_steps = check_steps
        self.sample_size = min(sample_size, len(self.val_items))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "compliance_log.jsonl"

    def sample_items(self) -> list[dict]:
        """Randomly sample items from validation set for compliance check.

        Returns:
            List of sampled validation items.
        """
        if not self.val_items:
            return []
        return random.sample(self.val_items, k=self.sample_size)

    def log_result(self, step: int, result: dict) -> None:
        """Append compliance check result to log file.

        Args:
            step: Current training step.
            result: Compliance evaluation result dict.
        """
        entry = {
            "step": step,
            "compliance_rate": result["compliance_rate"],
            "total": result["total"],
            "passed": result["passed"],
            "failed_reasons": result["failed_reasons"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(
            "Step %d: schema compliance = %.1f%% (%d/%d)",
            step,
            result["compliance_rate"] * 100,
            result["passed"],
            result["total"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema_compliance_callback.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run lint**

Run: `ruff check src/schema_compliance_callback.py tests/test_schema_compliance_callback.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/schema_compliance_callback.py tests/test_schema_compliance_callback.py
git commit -m "feat(pet-train): add schema compliance monitoring callback

Periodic validation during training using pet_schema.validate_output().
Logs results to JSONL and structured logger. Configurable check interval
and sample size."
```

---

### Task 8: Training Configs

**Files:**
- Create: `configs/base/sft_base.yaml`
- Create: `configs/base/dpo_base.yaml`
- Create: `configs/experiments/sft_lora_r16_lr2e4_ep3.yaml`
- Create: `configs/experiments/dpo_user_feedback_v1.yaml`
- Create: `configs/audio/mobilenetv2_transfer_v1.yaml`

- [ ] **Step 1: Create sft_base.yaml**

```yaml
### LLaMA-Factory SFT Base Configuration
### Inheritable by experiment configs in configs/experiments/

# Model
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct
trust_remote_code: true
template: qwen2_vl

# LoRA
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,v_proj
freeze_vision_tower: true

# Training
cutoff_len: 4096
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
label_smoothing_factor: 0.1

# Data
dataset_dir: data/
dataset: pet_sft_train

# Logging
report_to: wandb
logging_steps: 10
save_steps: 500
```

- [ ] **Step 2: Create dpo_base.yaml**

```yaml
### LLaMA-Factory DPO Base Configuration

# Model
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct
trust_remote_code: true
template: qwen2_vl

# LoRA (loaded from SFT checkpoint)
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,v_proj
freeze_vision_tower: true

# DPO
stage: dpo
pref_beta: 0.1
pref_loss: sigmoid

# Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

# Data
dataset_dir: data/
dataset: pet_dpo_pairs

# Logging
report_to: wandb
logging_steps: 10
save_steps: 500
```

- [ ] **Step 3: Create experiment configs**

`configs/experiments/sft_lora_r16_lr2e4_ep3.yaml`:
```yaml
### Baseline SFT experiment: LoRA r=16, lr=2e-4, 3 epochs
### Inherits all other settings from configs/base/sft_base.yaml
```

`configs/experiments/dpo_user_feedback_v1.yaml`:
```yaml
### DPO v1 with user feedback data
### Inherits all other settings from configs/base/dpo_base.yaml
```

- [ ] **Step 4: Create audio config**

`configs/audio/mobilenetv2_transfer_v1.yaml`:
```yaml
### PANNs MobileNetV2 Transfer Learning Configuration

model:
  backbone: "MobileNetV2"
  pretrained_path: "data/pretrained/MobileNetV2_mAP=0.383.pth"
  num_classes: 5
  freeze_backbone: true
  unfreeze_last_n: 0

training:
  batch_size: 32
  learning_rate: 1.0e-3
  max_epochs: 50
  early_stopping_patience: 10
  optimizer: adam
  scheduler: cosine

augmentation:
  spec_augment: true
  time_mask_param: 20
  freq_mask_param: 10
  random_gain_db: 6.0
```

- [ ] **Step 5: Commit configs**

```bash
git add configs/
git commit -m "feat(pet-train): add training configs for SFT, DPO, and audio

Base configs for LLaMA-Factory SFT/DPO, baseline experiment configs,
and PANNs MobileNetV2 audio transfer learning config."
```

---

### Task 9: Shell Scripts

**Files:**
- Create: `scripts/train_sft.sh`
- Create: `scripts/train_dpo.sh`
- Create: `scripts/train_audio.sh`
- Create: `scripts/merge_lora.sh`
- Create: `scripts/collect_logits.sh`
- Create: `scripts/eval_after_train.sh`

- [ ] **Step 1: Create train_sft.sh**

```bash
#!/bin/bash
# SFT training wrapper for LLaMA-Factory.
# Extracts run_name from config filename, merges base+experiment config,
# and launches training.
#
# Usage: bash scripts/train_sft.sh configs/experiments/sft_lora_r16_lr2e4_ep3.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 <experiment_config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Extract run_name from filename (strip path and .yaml)
RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
BASE_CONFIG="$PROJECT_DIR/configs/base/sft_base.yaml"
OUTPUT_DIR="$PROJECT_DIR/outputs/sft/$RUN_NAME"

echo "=== pet-train SFT ==="
echo "Experiment: $RUN_NAME"
echo "Base config: $BASE_CONFIG"
echo "Experiment config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Merge base + experiment config (experiment overrides base)
MERGED_CONFIG="$OUTPUT_DIR/_merged_config.yaml"
python3 -c "
import yaml, sys
with open('$BASE_CONFIG') as f:
    base = yaml.safe_load(f) or {}
exp_path = '$CONFIG_PATH'
with open(exp_path) as f:
    exp = yaml.safe_load(f) or {}
base.update(exp)
base['output_dir'] = '$OUTPUT_DIR'
base['run_name'] = '$RUN_NAME'
with open('$MERGED_CONFIG', 'w') as f:
    yaml.dump(base, f, default_flow_style=False)
"

echo "Starting training..."
llamafactory-cli train "$MERGED_CONFIG"

echo "=== SFT training complete ==="
echo "Adapter saved to: $OUTPUT_DIR"

# Optional: trigger evaluation
if command -v pet-eval &>/dev/null; then
    echo "Running post-training evaluation..."
    bash "$SCRIPT_DIR/eval_after_train.sh" "$OUTPUT_DIR" "$RUN_NAME"
else
    echo "pet-eval not installed, skipping post-training evaluation."
    echo "Install pet-eval and run: bash scripts/eval_after_train.sh $OUTPUT_DIR $RUN_NAME"
fi
```

- [ ] **Step 2: Create train_dpo.sh**

```bash
#!/bin/bash
# DPO training wrapper for LLaMA-Factory.
# Checks data availability and SFT adapter existence before starting.
#
# Usage: bash scripts/train_dpo.sh configs/experiments/dpo_user_feedback_v1.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 <experiment_config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
BASE_CONFIG="$PROJECT_DIR/configs/base/dpo_base.yaml"
OUTPUT_DIR="$PROJECT_DIR/outputs/dpo/$RUN_NAME"

# Read DPO params from params.yaml
PARAMS_FILE="$PROJECT_DIR/params.yaml"
MIN_PAIRS=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['dpo']['min_pairs'])")
SFT_ADAPTER=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['dpo']['sft_adapter_path'])")
DPO_DATA=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['dpo']['data_path'])")

# Check SFT adapter exists
if [ -z "$SFT_ADAPTER" ] || [ ! -d "$SFT_ADAPTER" ]; then
    echo "ERROR: SFT adapter path not set or does not exist: $SFT_ADAPTER"
    echo "Set dpo.sft_adapter_path in params.yaml to a valid SFT output directory."
    exit 1
fi

# Check DPO data availability
if [ ! -f "$DPO_DATA" ]; then
    echo "ERROR: DPO data file not found: $DPO_DATA"
    exit 1
fi

PAIR_COUNT=$(wc -l < "$DPO_DATA" | tr -d ' ')
if [ "$PAIR_COUNT" -lt "$MIN_PAIRS" ]; then
    echo "ERROR: Insufficient DPO pairs. Found $PAIR_COUNT, need >= $MIN_PAIRS."
    echo "Collect more DPO data via pet-annotation before running DPO training."
    exit 1
fi

echo "=== pet-train DPO ==="
echo "Experiment: $RUN_NAME"
echo "SFT adapter: $SFT_ADAPTER"
echo "DPO pairs: $PAIR_COUNT (min: $MIN_PAIRS)"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

llamafactory-cli train \
    --config "$BASE_CONFIG" \
    --adapter_name_or_path "$SFT_ADAPTER" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME"

echo "=== DPO training complete ==="
echo "Adapter saved to: $OUTPUT_DIR"

if command -v pet-eval &>/dev/null; then
    bash "$SCRIPT_DIR/eval_after_train.sh" "$OUTPUT_DIR" "$RUN_NAME"
else
    echo "pet-eval not installed, skipping post-training evaluation."
fi
```

- [ ] **Step 3: Create train_audio.sh**

```bash
#!/bin/bash
# Audio CNN training via PyTorch Lightning.
#
# Usage: bash scripts/train_audio.sh configs/audio/mobilenetv2_transfer_v1.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 <audio_config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
OUTPUT_DIR="$PROJECT_DIR/outputs/audio/$RUN_NAME"

echo "=== pet-train Audio CNN ==="
echo "Experiment: $RUN_NAME"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python3 -m src.audio_model \
    --config "$CONFIG_PATH" \
    --params "$PROJECT_DIR/params.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME"

echo "=== Audio training complete ==="
echo "Checkpoint saved to: $OUTPUT_DIR"
```

- [ ] **Step 4: Create merge_lora.sh**

```bash
#!/bin/bash
# Merge LoRA adapter weights into base model for quantization.
#
# Usage: bash scripts/merge_lora.sh <adapter_path> <output_path>

set -euo pipefail

ADAPTER_PATH="${1:?Usage: $0 <adapter_path> <output_path>}"
OUTPUT_PATH="${2:?Usage: $0 <adapter_path> <output_path>}"

echo "=== pet-train LoRA Merge ==="
echo "Adapter: $ADAPTER_PATH"
echo "Output: $OUTPUT_PATH"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter path does not exist: $ADAPTER_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_dir "$OUTPUT_PATH" \
    --export_size 2 \
    --export_legacy_format false

echo "=== Merge complete ==="
echo "Full model saved to: $OUTPUT_PATH"
```

- [ ] **Step 5: Create collect_logits.sh**

```bash
#!/bin/bash
# Collect teacher logits for KL distillation.
# Reads configuration from params.yaml and fetches missing logits.
#
# Usage: bash scripts/collect_logits.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PARAMS_FILE="$PROJECT_DIR/params.yaml"

# Check if KL distillation is enabled
ENABLED=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['kl_distillation']['enabled'])")

if [ "$ENABLED" != "True" ]; then
    echo "KL distillation is disabled in params.yaml. Set kl_distillation.enabled: true to collect logits."
    exit 0
fi

echo "=== Collecting Teacher Logits ==="

python3 -c "
import yaml
import json
from pathlib import Path
from src.logits_provider import FileLogitsProvider, APILogitsProvider

params = yaml.safe_load(open('$PARAMS_FILE'))
kl_cfg = params['kl_distillation']
sft_cfg = params['sft']

# Load training sample IDs
data_path = sft_cfg['data_path']
sample_ids = []
with open(data_path) as f:
    for line in f:
        item = json.loads(line.strip())
        sample_ids.append(item['id'])

print(f'Total training samples: {len(sample_ids)}')

provider_type = kl_cfg['provider']
logits_dir = kl_cfg['logits_dir']

if provider_type == 'file':
    provider = FileLogitsProvider(logits_dir=logits_dir)
    cached = provider.available_samples
    missing = [s for s in sample_ids if s not in cached]
    print(f'Cached: {len(cached)}, Missing: {len(missing)}')
    if missing:
        print('ERROR: Missing logits for file provider. Generate them offline.')
        for s in missing[:10]:
            print(f'  - {s}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')
elif provider_type == 'api':
    print('API logits collection not yet implemented in this script.')
    print('Use the Python API directly: APILogitsProvider.cache_sample()')
"

echo "=== Collection complete ==="
```

- [ ] **Step 6: Create eval_after_train.sh**

```bash
#!/bin/bash
# Trigger pet-eval after training completes.
# Skips gracefully if pet-eval is not installed.
#
# Usage: bash scripts/eval_after_train.sh <model_path> <run_name>

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> <run_name>}"
RUN_NAME="${2:?Usage: $0 <model_path> <run_name>}"

echo "=== Post-Training Evaluation ==="

if ! python3 -c "import pet_eval" 2>/dev/null; then
    echo "WARNING: pet-eval is not installed. Skipping post-training evaluation."
    echo "Install pet-eval and re-run: bash scripts/eval_after_train.sh $MODEL_PATH $RUN_NAME"
    exit 0
fi

echo "Running pet-eval for: $RUN_NAME"
echo "Model path: $MODEL_PATH"

# pet-eval CLI will be implemented in the pet-eval repo
python3 -m pet_eval.runners.eval_trained \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME"

echo "=== Evaluation complete ==="
```

- [ ] **Step 7: Make scripts executable and commit**

```bash
chmod +x scripts/*.sh
git add scripts/
git commit -m "feat(pet-train): add training and utility shell scripts

train_sft.sh, train_dpo.sh, train_audio.sh with auto run_name extraction.
merge_lora.sh for quantization prep. collect_logits.sh for KL distillation.
eval_after_train.sh with graceful pet-eval skip."
```

---

### Task 10: Integration Test and Final Verification

**Files:**
- Modify: `src/__init__.py` (if needed for imports)
- All existing files (lint + test pass)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-train && pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run full lint**

Run: `ruff check src/ tests/ && mypy src/`
Expected: PASS

- [ ] **Step 3: Verify directory structure matches spec**

Run: `find . -not -path './vendor/*' -not -path './.git/*' -not -path './outputs/*' -not -path './data/*' -type f | sort`
Expected: All files from spec present

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test(pet-train): verify full test suite and lint pass"
```

---

### Task 11: Update DEVELOPMENT_GUIDE

**Files:**
- Modify: `/Users/bamboo/Githubs/Train-Pet-Pipeline/pet-infra/docs/DEVELOPMENT_GUIDE.md`

- [ ] **Step 1: Update pet-train directory structure in dev guide**

Update section 5.4 to match the actual implementation:
- Add `src/logits_provider/` module
- Change `audio_model.py` description to "PANNs MobileNetV2 transfer learning"
- Change `audio_transforms.py` description to "log-mel spectrogram (64 bins, 16kHz)"

- [ ] **Step 2: Update audio model section**

Change from "MobileNetV3 + MFCC 40-dim" to "PANNs MobileNetV2 + log-mel 64-bin transfer learning" with rationale.

- [ ] **Step 3: Update audio classes**

Change from 4 classes to 5 classes: eating, drinking, vomiting, ambient, other.

- [ ] **Step 4: Update KL distillation section**

Note that KL distillation is optional (enabled via params.yaml), with label_smoothing_factor=0.1 as default softening.

- [ ] **Step 5: Add logits_provider documentation**

Document the TeacherLogitsProvider interface, FileProvider and APIProvider, and collect_logits.sh workflow.

- [ ] **Step 6: Commit dev guide updates**

```bash
cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-infra
git add docs/DEVELOPMENT_GUIDE.md
git commit -m "docs(pet-infra): update DEVELOPMENT_GUIDE for pet-train implementation

Reflect actual implementation: PANNs MobileNetV2 audio model, 5-class
taxonomy, optional KL distillation with pluggable logits providers,
label_smoothing as default softening."
```
