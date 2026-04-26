from __future__ import annotations

import pytest

from pet_train.plugins.llamafactory_sft import LlamaFactorySFTTrainer


@pytest.fixture
def sample_cfg() -> dict:
    return {
        "lora_r": 16,
        "lora_alpha": 32,
        "lr": 1e-4,
        "batch_size": 4,
        "grad_accum": 4,
        "max_steps": 1000,
        "base_model": "Qwen/Qwen2-VL-2B-Instruct",
        "dataset": "pet_annotation.vision_annotations",
        "output_dir": "/tmp/run",
    }


def test_hydra_to_lf_args_maps_lora_params(sample_cfg: dict) -> None:
    trainer = LlamaFactorySFTTrainer(**sample_cfg)
    args = trainer._lf_args
    assert args["lora_rank"] == 16
    assert args["lora_alpha"] == 32
    assert args["learning_rate"] == 1e-4
    assert args["per_device_train_batch_size"] == 4
    assert args["gradient_accumulation_steps"] == 4
    assert args["max_steps"] == 1000
    assert args["model_name_or_path"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert args["dataset"] == "pet_annotation.vision_annotations"
    assert args["output_dir"] == "/tmp/run"
    assert args["finetuning_type"] == "lora"
    assert args["stage"] == "sft"


def test_registers_to_trainers() -> None:
    from pet_train.plugins._register import register_all

    register_all()
    from pet_infra.registry import TRAINERS

    assert "llamafactory_sft" in TRAINERS.module_dict


def test_registry_build_produces_trainer(sample_cfg: dict) -> None:
    from pet_train.plugins._register import register_all

    register_all()
    from pet_infra.registry import TRAINERS

    trainer = TRAINERS.build({"type": "llamafactory_sft", **sample_cfg})
    assert isinstance(trainer, LlamaFactorySFTTrainer)
    assert trainer._lf_args["lora_rank"] == 16


def test_collect_train_metrics_reads_all_results_json(tmp_path, sample_cfg: dict) -> None:
    """F022 fix: _collect_train_metrics() must parse LF's all_results.json into card.metrics."""
    import json
    sample_cfg["output_dir"] = str(tmp_path)
    (tmp_path / "all_results.json").write_text(json.dumps({
        "epoch": 0.53,
        "total_flos": 730252.0,
        "train_loss": 0.5181,
        "train_runtime": 8.2567,
        "train_samples_per_second": 3.876,
    }))
    trainer = LlamaFactorySFTTrainer(**sample_cfg)
    metrics = trainer._collect_train_metrics()
    assert metrics["train_loss"] == 0.5181
    assert metrics["epoch"] == 0.53
    assert metrics["train_runtime"] == 8.2567
    # bool / non-numeric keys filtered
    assert all(isinstance(v, float) for v in metrics.values())


def test_collect_train_metrics_no_results_file(tmp_path, sample_cfg: dict) -> None:
    """F022 fix: missing all_results.json must return {} (not crash)."""
    sample_cfg["output_dir"] = str(tmp_path)
    trainer = LlamaFactorySFTTrainer(**sample_cfg)
    assert trainer._collect_train_metrics() == {}


def test_collect_train_metrics_falls_back_to_train_results(tmp_path, sample_cfg: dict) -> None:
    """F022 fix: if all_results.json missing but train_results.json present, use it."""
    import json
    sample_cfg["output_dir"] = str(tmp_path)
    (tmp_path / "train_results.json").write_text(json.dumps({"train_loss": 0.42}))
    trainer = LlamaFactorySFTTrainer(**sample_cfg)
    metrics = trainer._collect_train_metrics()
    assert metrics == {"train_loss": 0.42}


def test_collect_train_metrics_no_rewards_for_sft(tmp_path, sample_cfg: dict) -> None:
    """F023: SFT trainer_state has no rewards/* — _collect_train_metrics returns aggregate-only."""
    import json
    sample_cfg["output_dir"] = str(tmp_path)
    (tmp_path / "all_results.json").write_text(json.dumps({
        "epoch": 0.53, "train_loss": 0.518,
    }))
    (tmp_path / "trainer_state.json").write_text(json.dumps({
        "log_history": [
            {"step": 1, "loss": 0.59, "epoch": 0.07},  # SFT step, no rewards/*
            {"step": 8, "loss": 0.37, "epoch": 0.53},
            {"step": 8, "train_loss": 0.518},
        ],
    }))
    trainer = LlamaFactorySFTTrainer(**sample_cfg)
    metrics = trainer._collect_train_metrics()
    assert metrics["train_loss"] == 0.518
    assert metrics["epoch"] == 0.53
    # No rewards/* keys (SFT log_history has none)
    assert all(not k.startswith("rewards/") for k in metrics)
