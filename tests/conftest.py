"""Shared test fixtures for pet-train."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_params(tmp_dir):
    """Provide a minimal params.yaml for testing."""
    params = {
        "sft": {
            "data_path": str(tmp_dir / "train.jsonl"),
            "compliance_val_path": str(tmp_dir / "val.jsonl"),
            "compliance_val_size": 5,
            "compliance_check_steps": 10,
            "compliance_sample_size": 3,
            "label_smoothing_factor": 0.1,
        },
        "kl_distillation": {
            "enabled": False,
            "lambda_kl": 0.1,
            "temperature": 2.0,
            "top_k_approx": True,
            "top_k": 5,
            "provider": "file",
            "logits_dir": str(tmp_dir / "logits"),
        },
        "audio": {
            "sample_rate": 16000,
            "n_mels": 64,
            "data_dir": str(tmp_dir / "audio"),
            "classes": ["eating", "drinking", "vomiting", "ambient", "other"],
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
        },
    }
    params_path = tmp_dir / "params.yaml"
    params_path.write_text(yaml.dump(params))
    return params_path


@pytest.fixture
def sample_sharegpt_item():
    """Provide a single ShareGPT format training item."""
    return {
        "id": "sft_00001",
        "conversations": [
            {"from": "system", "value": "You are a pet feeder analyst."},
            {"from": "human", "value": "<image>\nAnalyze this feeding scene."},
            {
                "from": "gpt",
                "value": json.dumps(
                    {
                        "schema_version": "1.0",
                        "pet_present": True,
                        "pet_count": 1,
                        "pet": {
                            "species": "cat",
                            "action": {
                                "primary": "eating",
                                "distribution": {
                                    "eating": 0.85,
                                    "drinking": 0.05,
                                    "sniffing_only": 0.05,
                                    "leaving_bowl": 0.02,
                                    "sitting_idle": 0.02,
                                    "other": 0.01,
                                },
                            },
                            "eating_metrics": {
                                "duration_seconds": 120,
                                "speed": {
                                    "fast": 0.1,
                                    "normal": 0.7,
                                    "slow": 0.2,
                                },
                            },
                            "mood": {
                                "alertness": 0.6,
                                "anxiety": 0.1,
                                "engagement": 0.8,
                            },
                            "body_signals": {
                                "posture": "relaxed",
                                "ear_position": "forward",
                            },
                            "anomaly_signals": {
                                "vomit_gesture": 0.0,
                                "food_rejection": 0.0,
                                "excessive_sniffing": 0.05,
                                "lethargy": 0.0,
                                "aggression": 0.0,
                            },
                        },
                        "bowl": {
                            "food_fill_ratio": 0.6,
                            "water_fill_ratio": 0.8,
                            "food_type_visible": "dry_kibble",
                        },
                        "scene": {
                            "lighting": "normal",
                            "image_quality": "clear",
                            "confidence_overall": 0.92,
                        },
                        "narrative": "橘猫正常进食中，状态良好",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "images": ["data/frames/test_001.jpg"],
        "metadata": {
            "source": "selfshot",
            "schema_version": "1.0",
            "prompt_version": "1.0",
            "annotator": "qwen2.5-vl-72b",
            "review_status": "approved",
            "frame_id": "frame_test_001",
        },
    }
