"""LlamaFactoryDPOTrainer — thin wrapper over llamafactory.train.dpo.workflow.run_dpo."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pet_infra.registry import TRAINERS
from pet_schema.model_card import ModelCard

from pet_train.plugins.data_validation import validate_dpo_jsonl


@TRAINERS.register_module(name="llamafactory_dpo")
class LlamaFactoryDPOTrainer:
    """DPO training plugin wrapping LLaMA-Factory's `run_dpo` workflow.

    Expected config keys (passed via Registry.build kwargs):
      base_model, dataset, output_dir, lora_r, lora_alpha, lr, batch_size,
      grad_accum, max_steps, pref_beta. Optional: pref_loss, pref_ftx,
      version, trained_by, dataset_versions.
    """

    def __init__(self, **cfg: Any) -> None:
        self._cfg: dict[str, Any] = dict(cfg)
        self._lf_args: dict[str, Any] = self._hydra_to_lf_args(self._cfg)
        self._output_dir: str = str(cfg["output_dir"])
        self._adapter_uri: str | None = None
        self._metrics: dict[str, float] = {}

    @staticmethod
    def _hydra_to_lf_args(cfg: dict[str, Any]) -> dict[str, Any]:
        """Map hydra recipe config keys to LLaMA-Factory run_exp args dict."""
        # F013 fix companion: honor cfg["finetuning_type"] (default lora).
        ft_type = cfg.get("finetuning_type", "lora")
        args: dict[str, Any] = {
            "model_name_or_path": cfg["base_model"],
            "dataset": cfg["dataset"],
            "learning_rate": cfg["lr"],
            "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["grad_accum"],
            "max_steps": cfg["max_steps"],
            "output_dir": cfg["output_dir"],
            "finetuning_type": ft_type,
            "stage": "dpo",
            "pref_beta": cfg["pref_beta"],
            "pref_loss": cfg.get("pref_loss", "sigmoid"),
            "pref_ftx": cfg.get("pref_ftx", 0.0),
            "do_train": True,
        }
        if ft_type == "lora":
            args["lora_rank"] = cfg["lora_r"]
            args["lora_alpha"] = cfg["lora_alpha"]
        # F011 follow-on: pass through optional LF-native config keys when present
        for opt in ("dataset_dir", "template", "cutoff_len", "logging_steps",
                    "save_steps", "lr_scheduler_type", "warmup_ratio",
                    "num_train_epochs", "preprocessing_num_workers",
                    "trust_remote_code", "bf16", "fp16", "report_to"):
            if opt in cfg:
                args[opt] = cfg[opt]
        precision = cfg.get("precision")
        if precision == "bf16":
            args["bf16"] = True
        elif precision == "fp16":
            args["fp16"] = True
        return args

    def run(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Execute DPO training and return a populated ModelCard.

        Validates the JSONL data file against pet-schema DPOSample before
        passing it to LLaMA-Factory (F11 consumer-side defense).

        Lazy-imports run_dpo so module import doesn't fail when LLaMA-Factory's
        transformers pin is mismatched in dev/CI envs that only run unit tests.
        """
        data_path = self._cfg.get("data_path")
        if data_path:
            dp = Path(data_path)
            if not dp.exists():
                raise FileNotFoundError(
                    f"DPO training data file not found: {dp}. "
                    f"Check dpo.data_path in params.yaml or run "
                    f"pet-annotation export --format=dpo first."
                )
            if dp.suffix == ".jsonl":
                validate_dpo_jsonl(dp)

        # F011 fix: use run_exp(args=dict) public entry, not run_dpo(**kwargs) low-level
        from llamafactory.train.tuner import run_exp

        run_exp(args=self._lf_args)
        # F025 fix: LF saves adapter_model.safetensors directly to output_dir
        # (not output_dir/adapter). Previous code wrote a non-existent path so
        # downstream eval / vlm_inference loaded base model only — silent
        # finetune-disabled bug.
        self._adapter_uri = f"file://{Path(self._output_dir).resolve()}"
        self._metrics = self._collect_train_metrics()
        return self._build_model_card(input_card, recipe)

    def _collect_train_metrics(self) -> dict[str, float]:
        """Read LF's all_results.json + trainer_state.json log_history (F022 + F023).

        F022: ``all_results.json`` carries aggregate train_loss / train_runtime /
        epoch / total_flos / samples_per_second / steps_per_second.

        F023: DPO's ``rewards/{margins,chosen,rejected,accuracies}`` and
        ``logps/{chosen,rejected}`` / ``logits/{chosen,rejected}`` only appear in
        per-step entries inside ``trainer_state.json::log_history``. We pull the
        last entry whose keys include any of those families so card.metrics
        carries the FINAL training-step view of those signals (the most useful
        single point for downstream eval gates / ClearML reporting).
        """
        out_dir = Path(self._output_dir)
        metrics: dict[str, float] = {}
        for fname in ("all_results.json", "train_results.json"):
            p = out_dir / fname
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                break
            metrics.update({
                k: float(v) for k, v in payload.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            })
            break
        state_path = out_dir / "trainer_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError):
                state = {}
            for entry in reversed(state.get("log_history", []) or []):
                step_metrics = {
                    k: float(v) for k, v in entry.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and (
                        k.startswith("rewards/")
                        or k.startswith("logps/")
                        or k.startswith("logits/")
                    )
                }
                if step_metrics:
                    metrics.update(step_metrics)
                    break
        return metrics

    def _build_model_card(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Construct a ModelCard from training config and recipe metadata."""
        recipe_id = getattr(recipe, "recipe_id", "unknown")
        version = self._cfg.get("version") or getattr(recipe, "schema_version", "0.0.0")
        parent_models = [input_card.id] if input_card is not None else []

        return ModelCard(
            id="",  # orchestrator overwrites via card.id = card_id
            version=str(version),
            modality="vision",
            task="dpo",
            arch=self._derive_arch(),
            training_recipe=recipe_id,
            recipe_id=recipe_id,
            hydra_config_sha=self._hash_lf_args(),
            git_shas=self._collect_git_shas(),
            dataset_versions=self._cfg.get("dataset_versions") or {},
            checkpoint_uri=self._adapter_uri or "",
            parent_models=parent_models,
            lineage_role="dpo_output",
            metrics=dict(self._metrics),
            gate_status="pending",
            trained_at=datetime.now(UTC),
            trained_by=self._cfg.get("trained_by") or os.environ.get("USER", "ci"),
        )

    def _derive_arch(self) -> str:
        """Derive a short architecture string from base model + LoRA config + DPO suffix."""
        base = str(self._cfg["base_model"]).split("/")[-1].lower()
        r = self._lf_args["lora_rank"]
        a = self._lf_args["lora_alpha"]
        return f"{base}_lora_r{r}_a{a}_dpo"

    def _hash_lf_args(self) -> str:
        """Compute SHA-256 of the serialized LLaMA-Factory args dict."""
        payload = json.dumps(self._lf_args, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _collect_git_shas() -> dict[str, str]:
        """Return ``{<sibling-repo-name>: <HEAD sha>}`` for provenance (F024 fix)."""
        from pet_train.lineage import collect_git_shas
        return collect_git_shas()
