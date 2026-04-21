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
