from __future__ import annotations

from types import SimpleNamespace

import pytest

from pet_train.plugins.tiny_test import TinyTestTrainer


@pytest.fixture
def sample_cfg(tmp_path) -> dict:
    return {
        "output_dir": str(tmp_path / "tiny_run"),
        "max_steps": 3,
        "hidden_dim": 32,
        "input_dim": 16,
        "batch_size": 2,
        "lr": 1e-3,
    }


def test_init_stores_config(sample_cfg: dict) -> None:
    trainer = TinyTestTrainer(**sample_cfg)
    assert trainer._steps == 3
    assert trainer._hidden_dim == 32
    assert trainer._input_dim == 16


def test_build_model_has_expected_shape(sample_cfg: dict) -> None:
    trainer = TinyTestTrainer(**sample_cfg)
    model = trainer._build_model()
    param_count = sum(p.numel() for p in model.parameters())
    # 16*32 + 32 + 32*16 + 16 = 1104 for the small config; upper-bound for default ~100K
    assert param_count < 200_000


def test_run_produces_card_with_checkpoint(sample_cfg: dict) -> None:
    trainer = TinyTestTrainer(**sample_cfg)
    recipe = SimpleNamespace(recipe_id="smoke_tiny", schema_version="1.0.0")
    card = trainer.run(input_card=None, recipe=recipe)

    assert card.task == "test"
    assert card.arch == "tiny_test_transformer"
    assert card.modality == "vision"
    assert card.gate_status == "pending"
    assert card.checkpoint_uri.startswith("file://")
    assert card.checkpoint_uri.endswith("tiny_model.pt")
    assert "train_loss" in card.metrics
    assert isinstance(card.metrics["train_loss"], float)
    assert card.training_recipe == "smoke_tiny"
    assert card.recipe_id == "smoke_tiny"
    # hydra_config_sha should be a 64-char hex
    assert len(card.hydra_config_sha) == 64


def test_run_sets_parent_models_from_input_card(sample_cfg: dict) -> None:
    from datetime import UTC, datetime

    from pet_schema.model_card import ModelCard

    prev = ModelCard(
        id="prev-card-id",
        version="1.0.0",
        modality="vision",
        task="sft",
        arch="qwen2vl",
        training_recipe="sft_recipe",
        hydra_config_sha="a" * 64,
        git_shas={},
        dataset_versions={},
        checkpoint_uri="file:///tmp/prev",
        metrics={},
        gate_status="passed",
        trained_at=datetime.now(UTC),
        trained_by="ci",
    )

    trainer = TinyTestTrainer(**sample_cfg)
    recipe = SimpleNamespace(recipe_id="chain", schema_version="1.0.0")
    card = trainer.run(input_card=prev, recipe=recipe)

    assert card.parent_models == ["prev-card-id"]


def test_registers_to_trainers() -> None:
    from pet_train.plugins._register import register_all

    register_all()
    from pet_infra.registry import TRAINERS

    assert "tiny_test" in TRAINERS.module_dict


def test_registry_build_and_run(sample_cfg: dict) -> None:
    from pet_train.plugins._register import register_all

    register_all()
    from pet_infra.registry import TRAINERS

    trainer = TRAINERS.build({"type": "tiny_test", **sample_cfg})
    assert isinstance(trainer, TinyTestTrainer)
    recipe = SimpleNamespace(recipe_id="smoke_tiny", schema_version="1.0.0")
    card = trainer.run(input_card=None, recipe=recipe)
    assert card.task == "test"
