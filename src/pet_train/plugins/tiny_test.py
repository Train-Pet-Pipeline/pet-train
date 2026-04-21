"""TinyTestTrainer — tiny transformer for smoke_tiny recipe (PR gate <2min)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pet_infra.registry import TRAINERS
from pet_schema.model_card import ModelCard


@TRAINERS.register_module(name="tiny_test")
class TinyTestTrainer:
    """Tiny CPU-only transformer (~100K params) for smoke_tiny recipe.

    Runs a handful of SGD steps on synthetic data to exercise the training
    pipeline end-to-end without requiring a real dataset or GPU. Intended
    solely for PR-gate smoke (total wall-time < 2 minutes).
    """

    def __init__(self, **cfg: Any) -> None:
        self._cfg: dict[str, Any] = dict(cfg)
        self._output_dir: str = str(cfg.get("output_dir", "/tmp/tiny"))
        self._steps: int = int(cfg.get("max_steps", 10))
        self._hidden_dim: int = int(cfg.get("hidden_dim", 128))
        self._input_dim: int = int(cfg.get("input_dim", 64))
        self._batch_size: int = int(cfg.get("batch_size", 4))
        self._lr: float = float(cfg.get("lr", 1e-3))
        self._last_loss: float | None = None
        self._checkpoint_path: Path | None = None

    def _build_model(self) -> nn.Module:
        """Build the tiny transformer used for smoke training."""
        return nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._input_dim),
        )

    def run(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Execute a real (tiny) SGD training loop and return a populated ModelCard."""
        model = self._build_model()
        opt = torch.optim.SGD(model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        for _ in range(self._steps):
            x = torch.randn(self._batch_size, self._input_dim)
            y = model(x)
            loss = loss_fn(y, x)  # identity reconstruction target
            opt.zero_grad()
            loss.backward()
            opt.step()

        self._last_loss = float(loss.detach().item())

        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "tiny_model.pt"
        torch.save(model.state_dict(), ckpt)
        self._checkpoint_path = ckpt

        return self._build_model_card(input_card, recipe)

    def _build_model_card(self, input_card: ModelCard | None, recipe: Any) -> ModelCard:
        """Construct a ModelCard from training config and recipe metadata."""
        recipe_id = getattr(recipe, "recipe_id", "smoke_tiny")
        version = self._cfg.get("version") or getattr(recipe, "schema_version", "0.0.0")
        parent_models = [input_card.id] if input_card is not None else []

        return ModelCard(
            id="",  # orchestrator overwrites via card.id = card_id
            version=str(version),
            modality="vision",
            task="test",
            arch="tiny_test_transformer",
            training_recipe=recipe_id,
            recipe_id=recipe_id,
            hydra_config_sha=self._hash_cfg(),
            git_shas=self._collect_git_shas(),
            dataset_versions=self._cfg.get("dataset_versions") or {},
            checkpoint_uri=(
                f"file://{self._checkpoint_path.resolve()}" if self._checkpoint_path else ""
            ),
            parent_models=parent_models,
            metrics={"train_loss": self._last_loss if self._last_loss is not None else 0.0},
            gate_status="pending",
            trained_at=datetime.now(UTC),
            trained_by=self._cfg.get("trained_by") or os.environ.get("USER", "ci"),
        )

    def _hash_cfg(self) -> str:
        """Compute SHA-256 of the serialized config dict."""
        payload = json.dumps(self._cfg, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _collect_git_shas() -> dict[str, str]:
        """Return dict of repo->SHA for provenance; returns empty dict on failure."""
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return {"pet_train": sha}
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}
