from __future__ import annotations

import pytest

from pet_train.plugins.llamafactory_dpo import LlamaFactoryDPOTrainer


@pytest.fixture
def sample_cfg() -> dict:
    return {
        "lora_r": 8,
        "lora_alpha": 16,
        "lr": 5e-5,
        "batch_size": 2,
        "grad_accum": 8,
        "max_steps": 500,
        "base_model": "Qwen/Qwen2-VL-2B-Instruct",
        "dataset": "pet_annotation.dpo_pairs",
        "output_dir": "/tmp/dpo_run",
        "pref_beta": 0.1,
    }


def test_hydra_to_lf_args_maps_dpo_params(sample_cfg: dict) -> None:
    trainer = LlamaFactoryDPOTrainer(**sample_cfg)
    args = trainer._lf_args
    assert args["lora_rank"] == 8
    assert args["lora_alpha"] == 16
    assert args["learning_rate"] == 5e-5
    assert args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 8
    assert args["max_steps"] == 500
    assert args["model_name_or_path"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert args["dataset"] == "pet_annotation.dpo_pairs"
    assert args["output_dir"] == "/tmp/dpo_run"
    assert args["finetuning_type"] == "lora"
    assert args["stage"] == "dpo"
    assert args["pref_beta"] == 0.1
    assert args["pref_loss"] == "sigmoid"  # default
    assert args["pref_ftx"] == 0.0  # default


def test_pref_loss_and_ftx_configurable(sample_cfg: dict) -> None:
    cfg = {**sample_cfg, "pref_loss": "hinge", "pref_ftx": 0.5}
    trainer = LlamaFactoryDPOTrainer(**cfg)
    assert trainer._lf_args["pref_loss"] == "hinge"
    assert trainer._lf_args["pref_ftx"] == 0.5


def test_registers_to_trainers() -> None:
    from pet_train.plugins._register import register_all

    register_all()
    from pet_infra.registry import TRAINERS

    assert "llamafactory_dpo" in TRAINERS.module_dict


def test_registry_build_produces_trainer(sample_cfg: dict) -> None:
    from pet_train.plugins._register import register_all

    register_all()
    from pet_infra.registry import TRAINERS

    trainer = TRAINERS.build({"type": "llamafactory_dpo", **sample_cfg})
    assert isinstance(trainer, LlamaFactoryDPOTrainer)
    assert trainer._lf_args["stage"] == "dpo"
    assert trainer._lf_args["pref_beta"] == 0.1


def test_collect_train_metrics_reads_all_results_json(tmp_path, sample_cfg: dict) -> None:
    """F022 fix companion: DPO wrapper _collect_train_metrics must parse all_results.json."""
    import json
    sample_cfg["output_dir"] = str(tmp_path)
    (tmp_path / "all_results.json").write_text(json.dumps({
        "epoch": 0.5,
        "train_loss": 0.62,
        "rewards/margins": 0.34,
        "rewards/chosen": 0.18,
        "rewards/rejected": -0.16,
    }))
    trainer = LlamaFactoryDPOTrainer(**sample_cfg)
    metrics = trainer._collect_train_metrics()
    assert metrics["train_loss"] == 0.62
    assert metrics["rewards/margins"] == 0.34
    assert metrics["rewards/chosen"] == 0.18
    assert metrics["rewards/rejected"] == -0.16


def test_collect_train_metrics_no_results_file(tmp_path, sample_cfg: dict) -> None:
    """F022 fix: missing results file → {}, no crash."""
    sample_cfg["output_dir"] = str(tmp_path)
    trainer = LlamaFactoryDPOTrainer(**sample_cfg)
    assert trainer._collect_train_metrics() == {}
