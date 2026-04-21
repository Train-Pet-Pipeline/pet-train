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
