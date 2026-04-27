# pet-train Architecture

## §1 Repository Responsibility

**pet-train** is the model training engine in the smart pet feeder pipeline.

It provides three trainer plugins registered via `pet_infra.registry.TRAINERS`:

1. `llamafactory_sft` (`plugins/llamafactory_sft.py`) — SFT fine-tuning via LLaMA-Factory `run_sft`
2. `llamafactory_dpo` (`plugins/llamafactory_dpo.py`) — preference alignment via LLaMA-Factory `run_dpo`
3. `tiny_test` (`plugins/tiny_test.py`) — CPU-only smoke trainer for PR-gate validation (< 2 min)

Additionally, the `audio/` package provides PANNs-based zero-shot audio classification
(`MobileNetV2AudioSet` + `AudioInference`) that pet-eval imports at runtime (see §2).

**Pipeline position:**

```
pet-annotation → [pet-train] → pet-eval → pet-quantize → pet-ota
```

**Does:**
- Reads pet-annotation exported JSONL + validates schema against pet-schema models before training
- Calls LLaMA-Factory `run_sft` / `run_dpo` for VLM fine-tuning
- Exposes PANNs audio inference for cross-repo import by pet-eval (§5.3 runtime peer dep)
- Emits a `ModelCard` (pet-schema) after each training run
- Reads all numerics from `params.yaml` (sft / dpo / audio namespaces)

**Does not:**
- Annotate data (pet-annotation)
- Run evaluation gates or compute metrics (pet-eval)
- Convert or quantize checkpoints (pet-quantize)

---

## §2 I/O Contract

### Upstream dependencies

| Dependency | Mode | Locked version |
|---|---|---|
| pet-schema | β peer-dep (not in pyproject.dependencies) | v3.3.0 (compatibility_matrix 2026.11) |
| pet-infra | β peer-dep (not in pyproject.dependencies) | v2.9.5 (compatibility_matrix 2026.11) |
| pet-annotation | runtime data | exported JSONL files (paths from params.yaml) |

### Input types

| Format | Schema model | params.yaml key |
|---|---|---|
| ShareGPT SFT JSONL | `pet_schema.ShareGPTSFTSample` | `sft.data_path` |
| Alpaca DPO JSONL | `pet_schema.DPOSample` | `dpo.data_path` |
| Audio WAV files | N/A (file tree) | `audio.data_dir` |

**Consumer-side validation (F11):** Both SFT and DPO trainers call `validate_sft_jsonl` /
`validate_dpo_jsonl` in `plugins/data_validation.py` before passing data to LLaMA-Factory.
A missing data file raises `FileNotFoundError` immediately; a malformed row raises `ValueError`
with `file:line` context. This is defense-in-depth — pet-annotation is also expected to
validate on the producer side.

### Outputs

- `ModelCard` (pet-schema) — returned from `.run()`, consumed by pet-eval and pet-quantize; `metrics` is now populated from `all_results.json` + `trainer_state.json` (F022/F023 fix)
- Trained checkpoint directory — LLaMA-Factory LoRA adapter at `output_dir` (not `output_dir/adapter`; see F025)
- PANNs audio weights — `audio/inference.py:AudioInference.get_weights_path()` handed to pet-quantize for INT8 conversion

### Downstream consumers

- **pet-eval:** reads ModelCard + checkpoint; **also imports `pet_train.audio.inference`** at
  runtime for PANNs zero-shot audio classification (spec §5.3 cross-repo runtime peer dep)
- **pet-quantize:** converts LoRA adapter + PANNs weights to edge formats (ONNX / INT8)

---

## §3 Architecture Overview

### Directory tree

```
src/pet_train/
├── __init__.py                    ← __version__ = "2.2.5"
├── plugins/
│   ├── __init__.py
│   ├── _register.py               ← dual peer-dep guard (pet-schema first, then pet-infra)
│   ├── data_validation.py         ← F11 consumer-side JSONL validators
│   ├── llamafactory_sft.py        ← LlamaFactorySFTTrainer @TRAINERS.register_module
│   ├── llamafactory_dpo.py        ← LlamaFactoryDPOTrainer @TRAINERS.register_module
│   └── tiny_test.py               ← TinyTestTrainer (smoke only, no real data needed)
├── audio/
│   ├── __init__.py
│   ├── arch.py                    ← MobileNetV2AudioSet (num_classes=527 PANNs const)
│   ├── transforms.py              ← AudioTransform.from_params() (log-mel + SpecAugment)
│   └── inference.py               ← AudioInference (PANNs zero-shot, pet-eval imports this)
├── lineage/
│   └── collect_git_shas.py        ← collect_git_shas() walks sibling repos with hyphenated keys (F024)
vendor/
└── LLaMA-Factory/                 ← plain directory copy v0.9.4 (Apache 2.0, see NOTICE)
.github/workflows/
├── ci.yml                         ← 5-step install + version assert + lint + test
├── peer-dep-smoke.yml             ← mirrors ci.yml install sequence
└── no-wandb-residue.yml           ← scans src/ tests/ .github/ params.yaml; excludes vendor/
params.yaml                        ← sft / dpo / audio namespace numerics
tests/
├── conftest.py
├── test_version.py
├── test_audio_params.py
├── test_audio_transforms.py
├── test_audio_inference.py
└── plugins/
    ├── test_data_validation.py
    ├── test_llamafactory_sft.py
    ├── test_llamafactory_dpo.py
    ├── test_peer_dep_guards.py
    ├── test_register.py
    └── test_tiny_test.py
```

### Data flow

```
pet-annotation JSONL
        │
        ▼
validate_sft_jsonl / validate_dpo_jsonl   ← FileNotFoundError / ValueError fail-fast
        │
        ▼
run_sft / run_dpo (LLaMA-Factory vendored v0.9.4)
        │
        ▼
LoRA adapter checkpoint  ──────────────────────────────────────► pet-quantize (INT8)
        │
        ▼
ModelCard (pet-schema)  ───────────────────────────────────────► pet-eval (gate)

audio WAV files
        │
        ▼
AudioTransform.from_params(params["audio"])   ← log-mel, optional SpecAugment
        │
        ▼
MobileNetV2AudioSet (num_classes=527)
        │
        ▼
AudioInference.predict()  ─────────────────────────────────────► pet-eval (runtime import)
```

---

## §4 Core Modules

### 4.1 `plugins/llamafactory_sft.py` + `llamafactory_dpo.py`

**What:** `LlamaFactorySFTTrainer` and `LlamaFactoryDPOTrainer` are thin plugin wrappers
registered with `pet_infra.registry.TRAINERS` via `@TRAINERS.register_module`. Each maps
a hydra-style config dict to LLaMA-Factory kwargs (`_hydra_to_lf_args`), validates input data
(F11), then delegates to `run_sft` / `run_dpo`. After training it builds and returns a
`ModelCard` with SHA-pinned config (`hydra_config_sha`), git provenance, and adapter URI.

**F022/F023 fix:** Both wrappers now parse `all_results.json` and `trainer_state.json` from
the output directory to populate `card.metrics`. Previously `card.metrics` was always `{}`.
DPO-specific `rewards/{margins,chosen,rejected}` are extracted from the last per-step entry in
`trainer_state.json::log_history` (not from `all_results.json`, which does not contain them).

**F025 fix:** `card.checkpoint_uri = output_dir` (no `/adapter` suffix). The prior `/adapter`
suffix pointed to a non-existent path since LLaMA-Factory writes the adapter directly into
`output_dir`.

**Why LLaMA-Factory:** It is the SFT/DPO fine-tuning standard for open LLMs. Vendoring
avoids PyPI version drift and keeps the API surface stable — LLaMA-Factory's public API
changes frequently between minor releases.

**Tradeoff:** The vendor copy is a plain directory (not a git submodule). This simplifies
CI (no `git submodule update --init`) but means there is no commit hash pin embedded in
the repo. The version is v0.9.4 documented in `make setup` and `NOTICE`.

**Pitfall — lazy import:** `run_sft` / `run_dpo` are imported **inside `.run()`**, not at
module top level. LLaMA-Factory transitively requires `transformers` at a specific pin; if
a dev environment has a different version installed, top-level import would fail even when
running unit tests that never call `.run()`. The lazy import keeps the plugin importable in
all test environments.

**Pitfall — validate before LLaMA-Factory:** `validate_sft_jsonl` / `validate_dpo_jsonl`
run before `run_sft` / `run_dpo`. If the JSONL is malformed, training fails immediately with
a clear `ValueError` rather than after LLaMA-Factory has processed partial data.

### 4.2 `plugins/data_validation.py`

**What:** Two functions — `validate_sft_jsonl(path)` and `validate_dpo_jsonl(path)` — read a
JSONL file line-by-line and call `pet_schema.ShareGPTSFTSample.model_validate_json` /
`pet_schema.DPOSample.model_validate_json` on each non-empty row. Returns the count of valid
samples. Raises `ValueError` with `file:line` context on first bad row.

**Why:** Phase 4 E2E validation found that pet-annotation's flat JSONL export format could
drift silently from the schema LLaMA-Factory expects. F11 introduces consumer-side validation
in addition to producer-side validation in pet-annotation — if either side drifts, the
pipeline fails fast with a diagnostic message rather than producing a silently malformed model.

**Tradeoff:** Double validation (producer + consumer) adds a few seconds overhead on
million-row JSONL files — acceptable given the alternative (silent cross-repo contract drift).

**Pitfall — suffix check:** Validation only runs on `.jsonl` suffix files. Other formats
(parquet, arrow) are intentionally passed through to LLaMA-Factory without validation.
This is a design choice: pet-annotation's producer-side validation covers those formats,
and the silent-skip behaviour preserves extension points for future formats (§5).

**Pitfall — FileNotFoundError:** The trainers raise `FileNotFoundError` (not `ValueError`)
when `data_path` is set in config but the file does not exist. This gives a more actionable
error message pointing to the params.yaml key and the pet-annotation export step.

### 4.3 `audio/arch.py` + `audio/transforms.py`

**What:** `MobileNetV2AudioSet` (`arch.py`) is a MobileNetV2 CNN for AudioSet 527-class
classification, compatible with PANNs pretrained checkpoints. It includes a built-in
`MelSpectrogram` front-end initialized in `__init__` (not recreated per forward call).
`from_params(params)` constructs the model from the `audio:` sub-dict of `params.yaml`,
hardcoding `num_classes=527`.

`AudioTransform` (`transforms.py`) is an `nn.Module` that converts raw waveforms to
log-mel spectrograms with optional SpecAugment. Its `from_params(params, **override)`
classmethod reads all 6 required keys (`sample_rate`, `n_mels`, `n_fft`, `hop_length`,
`f_min`, `f_max`) from the `audio:` sub-dict and accepts keyword overrides (e.g.,
`spec_augment=True` for training).

**Why:** PANNs zero-shot audio classification maps AudioSet's 527 classes to 5 pet feeder
classes (eating / drinking / vomiting / ambient / other) without fine-tuning. This is
consumed by pet-eval via a cross-repo runtime import of `pet_train.audio.inference`.

**Tradeoff — params.yaml vs hardcode:** All 6 spectrogram hyperparameters are in
`params.yaml` (CLAUDE.md no-hardcode rule). `num_classes=527` is hardcoded because it is
an AudioSet taxonomy constant embedded in the network architecture — changing it would
invalidate PANNs pretrained weight loading. The hardcoding is documented with a docstring
warning in both `arch.py` and `from_params`.

**Pitfall — from_params KeyError:** `from_params` does direct key access (`params["n_fft"]`,
etc.) with no `.get()` fallback. A missing key raises `KeyError` immediately. This is
intentional: silent defaults would mask misconfigured params.yaml entries.

**Note — conftest gap:** The `sample_params` fixture in `tests/conftest.py` currently
provides only `sample_rate` and `n_mels` under `audio:` — it is missing `n_fft`,
`hop_length`, `f_min`, and `f_max`. Tests that call `AudioTransform.from_params` using the
fixture will raise `KeyError`. See §9 for the planned fix.

### 4.4 `plugins/_register.py`

**What:** Entry point callable (`register_all`) discovered by pet-infra's plugin loader via
`[project.entry-points."pet_infra.plugins"]` in `pyproject.toml`. When called, it first
checks that `pet_schema` is importable, then `pet_infra`, raising `RuntimeError` with
installation instructions on failure. It then imports the three trainer plugin modules to
trigger their `@TRAINERS.register_module` side-effects.

**Why — dual guard:** pet-schema and pet-infra are both β peer-deps (not listed in
`pyproject.dependencies`). Without the guard, a missing peer-dep would surface as a cryptic
`ImportError` deep in a plugin module. The ordered guard (pet-schema first as the upstream
contract, pet-infra second as the runtime framework) gives the operator a clear action message.

**Tradeoff — Mode B (delayed trigger):** The guard runs when `register_all()` is called by
pet-infra's plugin discovery — not at `import pet_train` time. This means tooling and unit
tests can import `pet_train` without having peer-deps installed. The cost is that peer-dep
absence is not caught until plugin registration.

**Pitfall — no version check:** The guard tests importability only, not version compatibility.
Cross-major-version breakage is caught by the CI version assert step and pytest integration,
not at registration time.

---

## §5 Extension Points

### 5.1 Add a new LLaMA-Factory training stage (e.g., RM, KTO)

1. Create `src/pet_train/plugins/llamafactory_<type>.py`
2. Implement a class with `__init__(self, **cfg)` and `run(self, input_card, recipe) -> ModelCard`
3. Decorate with `@TRAINERS.register_module(name="llamafactory_<type>")`
4. Add the import to `_register.py`'s `register_all()` alongside the existing plugin imports
5. Call `validate_sft_jsonl` or `validate_dpo_jsonl` before invoking the LLaMA-Factory run function
6. Use lazy import for the LLaMA-Factory run function (inside `.run()`, not at module top)

No changes needed to pet-schema, pet-infra, or the CI workflow.

### 5.2 Add a new JSONL format (e.g., Parquet)

1. Add a `validate_parquet_jsonl(path: Path) -> int` function in `plugins/data_validation.py`
   using the appropriate pet-schema Pydantic model
2. Update the suffix check in the trainer's `.run()` to branch on `.parquet` in addition to `.jsonl`
3. If the Pydantic model doesn't exist yet, add it to pet-schema first (upstream contract)

The silent-skip-for-unknown-suffixes design means no existing code breaks until the new
branch is explicitly added.

### 5.3 Add a new audio backbone (non-PANNs)

1. Create `src/pet_train/audio/<arch>.py` with a new `nn.Module`
2. If the new backbone uses a different output taxonomy than AudioSet 527, the `num_classes`
   constant in `arch.py` and `MobileNetV2AudioSet.from_params` will need updating (see §8.2)
3. Update `audio/inference.py:AudioInference._build_model` to construct the new backbone
4. Pet-eval's cross-repo import of `pet_train.audio.inference` is unaffected if `AudioInference`
   retains its public interface (`predict(audio_path) -> AudioPrediction`)

---

## §6 Dependency Management

### β peer-dep model

pet-schema and pet-infra are **not** listed in `[project.dependencies]` in `pyproject.toml`.
They are installed by the operator or CI before pet-train, per the compatibility matrix.
The `_register.py` dual guard surfaces clear `RuntimeError` if either is absent.

### CI 5-step install sequence (ci.yml)

```
1. pip install 'pet-schema @ git+https://...@v3.3.0'
2. pip install 'pet-infra @ git+https://...@v2.9.5'
3. pip install -e . --no-deps          # pet-train editable, no peer-dep re-resolution
4. pip install -e ".[dev]"             # dev extras (pytest, ruff, mypy, soundfile)
5. python -c "assert pet_schema.__version__ == '3.3.0' ..."  # version assert
```

This sequence is mirrored in `peer-dep-smoke.yml`. The `--no-deps` flag in step 3 prevents
pip from overwriting the pinned peer-dep versions resolved in steps 1-2.

### vendor/LLaMA-Factory

Apache 2.0 source copy, version v0.9.4 (plain directory, no git submodule). Attribution is
in `NOTICE`. `make setup` runs `cd vendor/LLaMA-Factory && pip install -e ".[metrics]"`.

### Third-party runtime deps (pyproject.toml)

`torch>=2.1`, `torchaudio>=2.1`, `pytorch-lightning>=2.1`, `pyyaml`, `tenacity`,
`python-json-logger`, `httpx`, `soundfile`.

---

## §7 Local Dev and Test

```bash
# Prerequisites: conda env pet-pipeline, peer-deps installed
conda activate pet-pipeline
pip install 'pet-schema @ git+https://github.com/Train-Pet-Pipeline/pet-schema@v3.3.0'
pip install 'pet-infra @ git+https://github.com/Train-Pet-Pipeline/pet-infra@v2.9.5'

make setup   # pip install -e ".[dev]" + vendor/LLaMA-Factory metrics extras
make lint    # ruff check src/ tests/ && mypy src/
make test    # pytest tests/ -v  (46 tests)
```

### Test coverage (46 tests)

| Test file | Coverage area |
|---|---|
| `test_version.py` | `__version__` matches pyproject.toml |
| `test_audio_params.py` | params.yaml audio namespace completeness |
| `test_audio_transforms.py` | `AudioTransform.from_params` + forward |
| `test_audio_inference.py` | `AudioInference` predict (random weights) |
| `plugins/test_data_validation.py` | SFT + DPO JSONL validation, FileNotFoundError |
| `plugins/test_llamafactory_sft.py` | SFTTrainer config mapping + ModelCard output |
| `plugins/test_llamafactory_dpo.py` | DPOTrainer config mapping + ModelCard output |
| `plugins/test_peer_dep_guards.py` | `_register.py` RuntimeError on missing peer-deps |
| `plugins/test_register.py` | `register_all()` happy path with mocked imports |
| `plugins/test_tiny_test.py` | `TinyTestTrainer` SGD loop + checkpoint save |

---

## §8 Known Complex Points (Preserved for Good Reasons)

### 8.1 LLaMA-Factory vendored as plain directory (v0.9.4, Apache 2.0)

**Why preserved:** Using a git submodule requires `git submodule update --init --recursive`
in every checkout and every CI step; operators and contributors routinely forget this step.
A plain directory copy is always present after `git clone` and `git pull` with no extra
commands. The `NOTICE` file records the source and license; attribution is compliant.

**What would be lost by removing:** The vendored copy pins a specific version. Moving to
PyPI `llamafactory` would expose the pipeline to upstream API changes between minor releases
(LLaMA-Factory's `run_sft` / `run_dpo` API is not semantically versioned stable).

**Condition to revisit:** LLaMA-Factory publishes a stable PyPI release with a committed
public API and semantic versioning, or a different version is needed simultaneously for
A/B experiments.

### 8.2 `num_classes=527` hardcoded in `arch.py` and `from_params`

**Why preserved:** AudioSet has exactly 527 classes in its fixed taxonomy. `MobileNetV2AudioSet`
is designed to load PANNs pretrained checkpoints that have a 527-output classification head.
Changing `num_classes` without replacing the backbone weights causes a shape mismatch on
`model.load_state_dict()`. The hardcoded value is documented with an explicit docstring
warning: "PANNs/AudioSet architecture constant — do not change".

**What would be lost by removing:** Any operator who passes `num_classes` via params.yaml
would load a randomly initialized head instead of PANNs pretrained weights, silently
degrading audio classification quality.

**Condition to revisit:** Switching to a non-PANNs audio backbone (e.g., AST, Wav2Vec2,
BEATs) where the output taxonomy differs from AudioSet 527. At that point `from_params`
should accept an optional `num_classes` override.

### 8.3 JSONL validator only runs on `.jsonl` suffix

**Why preserved:** Future data pipeline iterations may produce Parquet, Arrow, or other
columnar formats that LLaMA-Factory can read natively. A mandatory `.jsonl`-only validation
path would require every caller to convert first. Silent skip for non-.jsonl files is
intentional: pet-annotation's producer-side validation is the compensating control for
formats that bypass consumer-side validation in pet-train.

**What would be lost by removing the suffix check:** Every future format would need a
validator registered in `data_validation.py` before the pipeline could use it — that's
a reasonable gate to add when the format is actually adopted, not before.

**Condition to revisit:** A non-JSONL format enters production without a corresponding
producer-side validator in pet-annotation (i.e., both sides skip validation for a new format).

### 8.4 Lazy import of `run_sft` / `run_dpo` inside `.run()`

**Why preserved:** LLaMA-Factory depends on `transformers`, `peft`, `trl`, and related
libraries at specific version pins. In development and CI environments that run only unit
tests (no GPU, no LLaMA-Factory-compatible transformers version), a top-level import of
`llamafactory.train.sft.workflow` would fail at `import pet_train.plugins.llamafactory_sft`,
making the entire package unimportable. The lazy import defers this until `.run()` is
actually invoked.

**What would be lost by removing:** Dev environments that lack the LLaMA-Factory-compatible
transformers pin would be unable to `import pet_train` at all, blocking unit tests and
tooling like type checkers.

**Condition to revisit:** LLaMA-Factory becomes a lightweight package whose transitive
dependencies are always satisfied by the standard pet-pipeline conda environment, or when
torch becomes a universal baseline that all CI runners have.

---

## §9 Phase 5+ Follow-ups

1. **LLaMA-Factory vendor commit hash lock** — The vendor copy has no embedded git SHA
   or content hash. Add a `vendor/LLaMA-Factory/.commit` file recording the source
   commit SHA and tag (or migrate to a git submodule) so provenance is auditable
   without reading the NOTICE file.

2. **`data_validation.py` DRY** — `validate_sft_jsonl` and `validate_dpo_jsonl` share
   ~80% of their implementation (open file, iterate lines, strip, validate_json, count,
   raise ValueError with context). When a third format is added, extract a private
   `_validate_jsonl(path, schema_model)` helper to reduce duplication.

3. **`conftest.py` `sample_params` audio fixture incomplete** — The `sample_params` fixture
   provides `sample_rate` and `n_mels` under `audio:` but is missing `n_fft`, `hop_length`,
   `f_min`, and `f_max`. Any test that calls `AudioTransform.from_params(params["audio"])`
   with this fixture will raise `KeyError`. The fixture should be updated to include all
   6 keys matching `params.yaml` (n_fft=512, hop_length=160, f_min=50.0, f_max=8000.0).

4. **Prompt fidelity gap (cross-repo)** — pet-annotation's `rendered_prompt` does not inject
   the runtime `storage_uri` image token. Downstream training prompts may diverge from
   inference prompts if the image token path differs. Phase 5+ can address this with a
   pet-annotation migration 006 that stores `rendered_prompt` including the resolved image URI.

5. **DataRequirement contract (cross-repo, Phase 10)** — No trainer plugin currently declares
   `required_modalities`, `required_fields`, `min_samples`, or `allowed_provenance`. This
   cross-repo contract (pet-schema + pet-train + pet-infra + pet-annotation) has been raised
   four times across phases. Deferred to Phase 10 ecosystem closeout for unified planning
   across all four repos.

6. **`num_classes` override mechanism** — When switching to a non-PANNs backbone (§8.2
   condition), `MobileNetV2AudioSet.from_params` should accept an optional `num_classes`
   kwarg that overrides the AudioSet 527 default. The override should emit a warning log
   when used, since PANNs pretrained weights will not be loadable with a different value.

---

### Recent (2026-04-27)

F017-F027 续租验证周期的 pet-train 相关修复：

**F022** (`plugins/llamafactory_sft.py` + `llamafactory_dpo.py`) — LF wrappers 现在在 `run()` 返回前解析输出目录中的 `all_results.json` 和 `trainer_state.json`，并将关键指标（`train_loss`, `eval_loss`, `epoch`, 等）填入 `card.metrics`。修复前 `card.metrics` 始终为 `{}`。
- 参见：`pet-infra/docs/ecosystem-validation/2026-04-25-findings/F022-llamafactory-wrappers-not-capturing-train-metrics.md`

**F023** (`plugins/llamafactory_dpo.py`) — DPO wrapper 的 `rewards/{margins,chosen,rejected}` 指标从 `trainer_state.json::log_history` 最后一个 per-step 条目中提取（`all_results.json` 不包含 DPO rewards）。修复后 DPO cards 正确携带 reward metrics。
- 参见：`pet-infra/docs/ecosystem-validation/2026-04-25-findings/F023-dpo-rewards-not-in-all-results-need-trainer-state.md`

**F024** (`lineage/collect_git_shas.py` 新文件) — 新增 `pet_train.lineage.collect_git_shas()` 函数，使用 hyphenated key 命名约定（如 `pet-train`，非 `pet_train`）walk 所有 sibling 仓库收集 git SHA。与 `pet_infra.replay._current_git_shas` 共享 key 约定，解决了 underscore/hyphen 不匹配导致的 drift detection 失效。
- 参见：`pet-infra/docs/ecosystem-validation/2026-04-25-findings/F024-git-shas-key-mismatch-and-parents-off-by-one.md`

**F025** (`plugins/llamafactory_sft.py` + `llamafactory_dpo.py`) — `card.checkpoint_uri = output_dir`，去掉了原来错误的 `/adapter` 后缀。LLaMA-Factory 将 LoRA adapter 直接写入 `output_dir`，`output_dir/adapter` 子目录不存在。§4.1 中已相应更新。
- 参见：`pet-infra/docs/ecosystem-validation/2026-04-25-findings/F025-checkpoint-uri-points-to-nonexistent-adapter-subdir.md`
