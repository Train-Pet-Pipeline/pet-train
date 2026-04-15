# pet-train Design Spec

> Date: 2026-04-15
> Status: Approved
> Author: Claude + Bamboo

## 1. Overview

pet-train implements post-training for the Qwen2-VL-2B vision-language model and audio classification for the smart pet feeder pipeline. It consumes annotated data from pet-annotation and outputs LoRA weights for pet-quantize.

**Three training tracks, fully independent:**

- **SFT** (Supervised Fine-Tuning): LoRA on Qwen2-VL-2B, driven by LLaMA-Factory
- **DPO** (Direct Preference Optimization): preference alignment on top of SFT LoRA
- **Audio CNN**: PANNs MobileNetV2 transfer learning via PyTorch Lightning

## 2. Directory Structure

```
pet-train/
├── configs/
│   ├── base/
│   │   ├── sft_base.yaml                  # LLaMA-Factory SFT base config
│   │   └── dpo_base.yaml                  # LLaMA-Factory DPO base config
│   ├── experiments/                       # filename = experiment name = wandb run_name
│   │   ├── sft_lora_r16_lr2e4_ep3.yaml
│   │   └── dpo_user_feedback_v1.yaml
│   └── audio/
│       └── mobilenetv2_transfer_v1.yaml
├── src/
│   ├── __init__.py
│   ├── kl_loss.py                         # KL distillation loss (full vocab + top-k approx)
│   ├── logits_provider/
│   │   ├── __init__.py
│   │   ├── base.py                        # TeacherLogitsProvider abstract base
│   │   ├── file_provider.py               # Read .pt logits from disk
│   │   └── api_provider.py                # API top-k logprobs + disk cache
│   ├── schema_compliance_callback.py      # Every 500 steps, sample 20, validate
│   ├── audio_model.py                     # PANNs MobileNetV2 transfer learning
│   ├── audio_datamodule.py                # PyTorch Lightning DataModule
│   └── audio_transforms.py                # log-mel spectrogram feature extraction
├── scripts/
│   ├── train_sft.sh                       # llamafactory-cli wrapper, auto-inject run_name
│   ├── train_dpo.sh
│   ├── train_audio.sh
│   ├── merge_lora.sh                      # LoRA merge, output to outputs/merged/
│   ├── collect_logits.sh                  # Pre-training teacher logits collection
│   └── eval_after_train.sh                # Call pet-eval (skip if not installed)
├── tests/
│   ├── test_kl_loss.py
│   ├── test_logits_provider.py
│   ├── test_schema_compliance_callback.py
│   ├── test_audio_model.py
│   ├── test_audio_datamodule.py
│   └── test_audio_transforms.py
├── vendor/
│   └── LLaMA-Factory/                    # git submodule, locked to v0.9.4
├── outputs/                               # .gitignore
│   ├── sft/{experiment_name}/
│   ├── dpo/{experiment_name}/
│   ├── audio/{experiment_name}/
│   └── merged/{experiment_name}_merged/
├── data/                                  # .gitignore (training data, logits cache)
├── params.yaml
├── pyproject.toml
├── Makefile
└── .gitignore
```

## 3. SFT Training

### 3.1 Model Configuration

- Base model: Qwen/Qwen2-VL-2B-Instruct
- LoRA: rank=16, alpha=32, target=q_proj,v_proj
- Vision tower: frozen (mandatory)
- Template: qwen2_vl
- Cutoff length: 4096

### 3.2 Training Configuration

- Loss: Cross-entropy + label_smoothing_factor=0.1 + optional KL distillation
- Learning rate: 2e-4, cosine scheduler, warmup_ratio=0.1
- Precision: bf16
- Logging: wandb, every 10 steps
- Checkpoints: every 500 steps

### 3.3 Data Input

ShareGPT JSONL from pet-annotation:

```json
{
  "id": "sft_00001",
  "conversations": [
    {"from": "system", "value": "<system prompt>"},
    {"from": "human", "value": "<image>\n<user prompt>"},
    {"from": "gpt", "value": "{...complete JSON...}"}
  ],
  "images": ["path/to/frame.jpg"],
  "metadata": {...}
}
```

### 3.4 Experiment Naming Convention

- Format: `{task}_{variable}_{value}.yaml`
- `train_sft.sh` auto-extracts run_name from filename, no manual override
- Experiment configs inherit from base, only override changed params

### 3.5 train_sft.sh Flow

1. Accept experiment config path as argument
2. Extract run_name from filename (strip path and .yaml)
3. Merge base/sft_base.yaml + experiment config (experiment overrides base)
4. Inject run_name into wandb
5. Set output_dir to outputs/sft/{run_name}/
6. Call `llamafactory-cli train`
7. On completion, call eval_after_train.sh (optional)

## 4. KL Distillation Module

### 4.1 Overview

Optional additional loss term during SFT. Controlled by `params.yaml` `kl_distillation.enabled`.

Total loss = CE_loss(student, teacher_text) + label_smoothing + lambda * KL_loss(student_logits, teacher_logits)

### 4.2 kl_loss.py

Two modes:

- **Full vocab KL** (`top_k_approx: false`): Standard KL divergence with temperature softening. For offline local model logits.
- **Top-k approximate KL** (`top_k_approx: true`): Compute KL only on teacher's top-k tokens, remaining probability mass goes to a "rest" bucket. Industry-standard approach (DistilBERT, MiniLM). For API logprobs.

```python
def compute_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    lambda_kl: float = 0.1
) -> torch.Tensor:
    """Temperature-softened KL divergence."""
```

### 4.3 TeacherLogitsProvider Interface

```python
class TeacherLogitsProvider(ABC):
    @abstractmethod
    def get_logits(self, sample_id: str) -> LogitsResult: ...

@dataclass
class LogitsResult:
    token_ids: torch.Tensor      # [seq_len, k] or [seq_len, vocab_size]
    logprobs: torch.Tensor       # corresponding log probabilities
    is_full_vocab: bool          # True=full distribution, False=top-k
```

### 4.4 FileProvider

- Reads pre-computed .pt files from `data/teacher_logits/`
- `manifest.json` maps sample_id to file path, records model name, collection time, top-k value
- Supports logits from any model (Qwen-72B, DeepSeek, LLaMA-70B, etc.)

### 4.5 APIProvider

- Calls OpenAI-compatible APIs with `logprobs=true, top_logprobs=20`
- First call: fetch from API, cache to disk in same format as FileProvider
- Subsequent calls: read from cache, no redundant API calls
- tenacity retry, structured JSON logging
- Supports any compatible API (Dashscope, DeepSeek, OpenAI, etc.)

### 4.6 collect_logits.sh

Standalone pre-training script:

1. Read kl_distillation config from params.yaml
2. Read training JSONL, extract all sample_ids
3. Check existing cached logits, skip already collected
4. Collect missing logits via configured provider
5. Generate/update manifest.json
6. Summary report: total, cached, newly collected, failed

## 5. DPO Training

### 5.1 Prerequisites

- SFT LoRA weights exist at configured path
- DPO JSONL data count >= `dpo.min_pairs` (default 500)

### 5.2 Model Configuration

- Loads Qwen2-VL-2B + SFT LoRA adapter as starting point
- Continues LoRA training with DPO loss
- Same LoRA config as SFT (r=16, alpha=32, q_proj/v_proj)

### 5.3 DPO Configuration

- Stage: dpo
- Loss: sigmoid DPO (pref_beta=0.1)
- Learning rate: 5e-5 (one order of magnitude lower than SFT)
- Epochs: 1

### 5.4 Data Input

DPO JSONL from pet-annotation:

```json
{
  "id": "dpo_00001",
  "system": "<system prompt>",
  "prompt": "<image>\n<user prompt>",
  "images": ["path/to/frame.jpg"],
  "chosen": [
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "{...correct JSON...}"}
  ],
  "rejected": [
    {"role": "user", "content": "<prompt>"},
    {"role": "assistant", "content": "{...flawed JSON...}"}
  ],
  "metadata": {
    "rejection_reason": "narrative_anthropomorphism | distribution_sum_error | false_anomaly | mood_miscalibration | user_feedback"
  }
}
```

### 5.5 train_dpo.sh Flow

1. Extract run_name from filename
2. Check DPO data count >= min_pairs, exit with clear message if insufficient
3. Verify SFT LoRA weights path exists
4. Merge base/dpo_base.yaml + experiment config
5. Inject run_name, output_dir
6. Call `llamafactory-cli train`
7. On completion, call eval_after_train.sh

### 5.6 Training Pipeline

```
SFT LoRA (v1) → DPO training → LoRA (v2) → merge_lora.sh → full weights → pet-quantize
```

## 6. Audio CNN Training

### 6.1 Architecture

- Backbone: PANNs MobileNetV2 (~4.1M params), pretrained on AudioSet
- Pretrained weights: MobileNetV2_mAP=0.383.pth (16kHz variant)
- Classification head: replace final layer → 5 classes
- Framework: PyTorch Lightning

### 6.2 Classes

5-class coarse-grained classification:

| Class | Description |
|-------|-------------|
| eating | Chewing, crunching sounds |
| drinking | Lapping, water sounds |
| vomiting | Retching, vomit sounds |
| ambient | Background environmental noise |
| other | Unclassifiable / mixed sounds |

No fine-grained sub-categories in v1.

### 6.3 Feature Extraction (audio_transforms.py)

- Input: 16kHz WAV
- Resample to 16kHz if needed
- STFT: window=512, hop=160
- Mel filterbank: 64 bins, fmin=50, fmax=8000
- Log-mel: log(mel + 1e-7)
- Augmentation (configurable): SpecAugment (time/freq masking), random gain

### 6.4 Training Strategy

- Phase 1: Freeze backbone, train classification head only (fast convergence)
- Phase 2 (optional): Unfreeze last N layers for fine-tuning (controlled by params.yaml)
- Early stopping: patience=10 on validation loss
- Max epochs: 50

### 6.5 Data Organization

```
data/audio/
├── eating/
├── drinking/
├── vomiting/
├── ambient/
└── other/
```

Split: train 80% / val 10% / test 10%

## 7. Schema Compliance Callback

### 7.1 Purpose

Real-time health check during VLM training (SFT and DPO). Detects training collapse early.

### 7.2 Mechanism

- Trigger: every 500 steps (configurable via params.yaml)
- Sample: 20 random items from a fixed validation subset (50-100 items, separated from training data)
- Generate: greedy decode for reproducibility
- Validate: `pet_schema.validate_output()` on each output
- Record: compliance_rate to wandb + compliance_log.jsonl
- Action: log only, no auto-stop (v1 collects baseline data first)

### 7.3 Validation Subset

`data/compliance_val.jsonl`: pre-separated from training set, never used for training. Same ShareGPT format as SFT data.

## 8. pet-eval Integration

- pet-train does NOT implement evaluation logic
- `eval_after_train.sh` calls pet-eval CLI when installed
- When pet-eval is not installed: skip with clear log message
- schema_compliance_callback is pet-train's own responsibility (training-time health check, not post-training evaluation)

Responsibility boundary:
- pet-train: training-time schema compliance monitoring (lightweight, fast)
- pet-eval: post-training comprehensive evaluation (multi-metric, gold set, lm-eval-harness)

## 9. Project Configuration

### 9.1 params.yaml

```yaml
sft:
  data_path: "data/pet_sft_train.jsonl"
  compliance_val_path: "data/compliance_val.jsonl"
  compliance_val_size: 50
  compliance_check_steps: 500
  compliance_sample_size: 20
  label_smoothing_factor: 0.1

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

dpo:
  min_pairs: 500
  pref_beta: 0.1
  sft_adapter_path: ""
  data_path: "data/pet_dpo_pairs.jsonl"
  learning_rate: 5.0e-5

audio:
  sample_rate: 16000
  n_mels: 64
  data_dir: "data/audio/"
  classes: ["eating", "drinking", "vomiting", "ambient", "other"]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

wandb:
  project: "pet-train"
  entity: ""
```

### 9.2 pyproject.toml Dependencies

```toml
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
dev = ["pytest", "ruff", "mypy"]
```

LLaMA-Factory installed from vendor source via `make setup`, not in pyproject.toml.

### 9.3 Makefile Targets

| Target | Command |
|--------|---------|
| `setup` | pip install -e ".[dev]" + vendor/LLaMA-Factory |
| `test` | pytest tests/ -v |
| `lint` | ruff check + mypy |
| `clean` | remove caches and build artifacts |
| `train-sft` | scripts/train_sft.sh $(CONFIG) |
| `train-dpo` | scripts/train_dpo.sh $(CONFIG) |
| `train-audio` | scripts/train_audio.sh $(CONFIG) |
| `merge` | scripts/merge_lora.sh |
| `collect-logits` | scripts/collect_logits.sh |

### 9.4 Vendor Submodule

- LLaMA-Factory v0.9.4, commit 95ac3f23
- Locked via git submodule at vendor/LLaMA-Factory/
- Python 3.11 compatible

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| VLM training framework | LLaMA-Factory (vendor submodule) | Qwen2-VL native support, unstable PyPI so vendor lock |
| Audio model | PANNs MobileNetV2 (~4.1M) | Lightweight pretrained, AudioSet, MIT license |
| Audio framework | PyTorch Lightning | Clean DataModule/Trainer, independent from LLaMA-Factory |
| KL distillation | Optional, pluggable provider | v1 SFT works without logits, KL adds on when ready |
| Teacher logits source | FileProvider + APIProvider | Support both local offline and online API |
| Top-k approx KL | Industry standard (DistilBERT, MiniLM) | API returns top-k only, proven effective |
| SFT overfitting mitigation | label_smoothing=0.1 + LoRA + val monitoring + optional KL | Progressive softening: label_smoothing → KL top-k → KL full |
| Audio classes | 5-class coarse (eat/drink/vomit/ambient/other) | Avoid guessing with insufficient data, refine later |
| DPO timing | After SFT, optional in v1 | Data may be < 500 pairs initially |
| Evaluation | Delegate to pet-eval | pet-eval shared by pet-train + pet-quantize |
| Training-time monitoring | schema_compliance_callback | Lightweight, uses pet_schema directly, log only |

## 11. Deviations from DEVELOPMENT_GUIDE

| Original Spec | This Design | Reason | Action |
|---------------|-------------|--------|--------|
| MobileNetV3 + MFCC 40-dim | PANNs MobileNetV2 + log-mel 64-bin | Pretrained transfer learning is more practical than training from scratch | Update DEVELOPMENT_GUIDE |
| KL distillation always on | KL optional, label_smoothing as default softening | Teacher logits require separate collection step | Update DEVELOPMENT_GUIDE |
| CNN14 (discussed) | MobileNetV2 | CNN14 ~80M too heavy for lightweight audio module | N/A (CNN14 was discussion only) |

DEVELOPMENT_GUIDE will be updated to reflect these deviations after spec approval.
