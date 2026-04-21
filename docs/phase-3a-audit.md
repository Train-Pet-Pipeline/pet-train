# Phase 3A v1 Audit — pet-train

**Date:** 2026-04-21
**Spec:** `pet-infra/docs/superpowers/specs/2026-04-21-phase-3a-training-design.md` §2.1/§2.2
**Plan:** `pet-infra/docs/superpowers/plans/2026-04-21-phase-3a-training-plan.md` PR #P1-A

## git ls-files (38 files)

```
.github/workflows/ci.yml
.gitignore
.gitmodules
Makefile
configs/audio/mobilenetv2_transfer_v1.yaml
configs/base/dpo_base.yaml
configs/base/sft_base.yaml
configs/experiments/dpo_user_feedback_v1.yaml
configs/experiments/sft_lora_r16_lr2e4_ep3.yaml
docs/superpowers/plans/2026-04-15-pet-train-implementation.md
docs/superpowers/specs/2026-04-15-pet-train-design.md
docs/training-experiment-report.md
params.yaml
pyproject.toml
scripts/collect_logits.sh
scripts/eval_after_train.sh
scripts/merge_lora.sh
scripts/train_audio.sh
scripts/train_dpo.sh
scripts/train_sft.sh
src/pet_train/__init__.py
src/pet_train/audio_inference.py
src/pet_train/audio_model.py
src/pet_train/audio_model_arch.py
src/pet_train/audio_transforms.py
src/pet_train/kl_loss.py
src/pet_train/logits_provider/__init__.py
src/pet_train/logits_provider/api_provider.py
src/pet_train/logits_provider/base.py
src/pet_train/logits_provider/file_provider.py
src/pet_train/schema_compliance_callback.py
tests/conftest.py
tests/test_audio_inference.py
tests/test_audio_transforms.py
tests/test_kl_loss.py
tests/test_logits_provider.py
tests/test_schema_compliance_callback.py
vendor/LLaMA-Factory
```

## Classification Table

| Path | Action | Reason |
|------|--------|--------|
| `.github/workflows/ci.yml` | update | Makefile targets reference deleted scripts; CI itself is clean (no script refs), but Makefile targets need removal |
| `.gitignore` | keep | Infrastructure, no changes needed |
| `.gitmodules` | keep | LLaMA-Factory submodule definition |
| `Makefile` | update | `train-sft`, `train-dpo`, `train-audio`, `merge`, `collect-logits` targets reference deleted scripts |
| `configs/audio/mobilenetv2_transfer_v1.yaml` | delete | v1 config; all ratios migrate to params.yaml + Hydra defaults (P1-C/D) |
| `configs/base/dpo_base.yaml` | delete | v1 config; replaced by params.yaml + Hydra defaults |
| `configs/base/sft_base.yaml` | delete | v1 config; replaced by params.yaml + Hydra defaults |
| `configs/experiments/dpo_user_feedback_v1.yaml` | delete | v1 experiment config |
| `configs/experiments/sft_lora_r16_lr2e4_ep3.yaml` | delete | v1 experiment config |
| `docs/superpowers/plans/2026-04-15-pet-train-implementation.md` | keep | v1 implementation plan; historical reference |
| `docs/superpowers/specs/2026-04-15-pet-train-design.md` | keep | v1 design spec; historical reference |
| `docs/training-experiment-report.md` | keep | Experiment log; historical reference |
| `params.yaml` | keep | Central params file; values used by all trainers |
| `pyproject.toml` | update | Remove `wandb` dep; no `[project.scripts]` block present |
| `scripts/collect_logits.sh` | delete | CLI wrapper obsoleted by `pet run` orchestrator |
| `scripts/eval_after_train.sh` | delete | CLI wrapper obsoleted by `pet run` orchestrator |
| `scripts/merge_lora.sh` | delete | CLI wrapper obsoleted by `pet run` orchestrator |
| `scripts/train_audio.sh` | delete | CLI wrapper obsoleted by `pet run` orchestrator |
| `scripts/train_dpo.sh` | delete | CLI wrapper obsoleted by `pet run` orchestrator |
| `scripts/train_sft.sh` | delete | CLI wrapper obsoleted by `pet run` orchestrator |
| `src/pet_train/__init__.py` | keep | Package init; no v1 imports |
| `src/pet_train/audio_inference.py` | keep (rename in P1-B) | Zero-shot inference; keep for `audio/` namespace rename |
| `src/pet_train/audio_model.py` | delete | Confirmed v1 CLI (argparse entry-point calling `scripts/train_audio.sh`); not the arch lib |
| `src/pet_train/audio_model_arch.py` | keep (rename in P1-B) | MobileNetV2AudioSet architecture class; used by audio_inference.py |
| `src/pet_train/audio_transforms.py` | keep (rename in P1-B) | Mel spectrogram transforms; used by audio_inference.py |
| `src/pet_train/kl_loss.py` | delete | Replaced by LLaMA-Factory native DPO path |
| `src/pet_train/logits_provider/__init__.py` | delete | v1 logits provider subpackage |
| `src/pet_train/logits_provider/api_provider.py` | delete | v1 logits provider subpackage |
| `src/pet_train/logits_provider/base.py` | delete | v1 logits provider subpackage |
| `src/pet_train/logits_provider/file_provider.py` | delete | v1 logits provider subpackage |
| `src/pet_train/schema_compliance_callback.py` | delete | Replaced by LLaMA-Factory native callback system |
| `tests/conftest.py` | keep | Shared fixtures; no imports from deleted modules |
| `tests/test_audio_inference.py` | keep (update in P1-B) | Tests for audio_inference.py; imports update after rename |
| `tests/test_audio_transforms.py` | keep (update in P1-B) | Tests for audio_transforms.py; imports update after rename |
| `tests/test_kl_loss.py` | delete | Tests for deleted kl_loss.py |
| `tests/test_logits_provider.py` | delete | Tests for deleted logits_provider subpackage |
| `tests/test_schema_compliance_callback.py` | delete | Tests for deleted schema_compliance_callback.py |
| `vendor/LLaMA-Factory` | keep | Vendored submodule; required by all trainers |

## Cross-Import Safety Check

Verified no KEEP file imports from any DELETE target:
- `audio_inference.py` imports `audio_model_arch` (KEEP) and `pet_infra.device` (external)
- `audio_model_arch.py` has no internal cross-imports
- `audio_transforms.py` has no internal cross-imports
- `audio_model.py` (DELETE) imports `audio_model_arch` (KEEP) — safe to delete

## Makefile / CI Impact

- `Makefile`: `train-sft`, `train-dpo`, `train-audio`, `merge`, `collect-logits` targets reference deleted scripts — must remove
- `.github/workflows/ci.yml`: no direct script references — no changes needed

## Files Remaining After Purge (~19)

```
.github/workflows/ci.yml
.gitignore
.gitmodules
Makefile (updated)
docs/superpowers/plans/2026-04-15-pet-train-implementation.md
docs/superpowers/specs/2026-04-15-pet-train-design.md
docs/phase-3a-audit.md (this file)
docs/training-experiment-report.md
params.yaml
pyproject.toml (updated)
src/pet_train/__init__.py
src/pet_train/audio_inference.py
src/pet_train/audio_model_arch.py
src/pet_train/audio_transforms.py
tests/conftest.py
tests/test_audio_inference.py
tests/test_audio_transforms.py
vendor/LLaMA-Factory
```
