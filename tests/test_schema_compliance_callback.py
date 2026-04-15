"""Tests for training-time schema compliance monitoring callback."""

import json

import pytest

from pet_train.schema_compliance_callback import (
    SchemaComplianceCallback,
    evaluate_compliance,
    load_compliance_val_set,
)

# Schema-compliant valid eating output (pet present)
_VALID_PET_OUTPUT = json.dumps({
    "schema_version": "1.0",
    "pet_present": True,
    "pet_count": 1,
    "pet": {
        "species": "cat",
        "breed_estimate": "british_shorthair",
        "id_tag": "grey_shorthair_medium",
        "id_confidence": 0.83,
        "action": {
            "primary": "eating",
            "distribution": {
                "eating": 0.76, "drinking": 0.00,
                "sniffing_only": 0.14, "leaving_bowl": 0.05,
                "sitting_idle": 0.03, "other": 0.02,
            },
        },
        "eating_metrics": {
            "speed": {"fast": 0.08, "normal": 0.71, "slow": 0.21},
            "engagement": 0.74,
            "abandoned_midway": 0.12,
        },
        "mood": {"alertness": 0.28, "anxiety": 0.09, "engagement": 0.76},
        "body_signals": {"posture": "relaxed", "ear_position": "forward"},
        "anomaly_signals": {
            "vomit_gesture": 0.02, "food_rejection": 0.09,
            "excessive_sniffing": 0.16, "lethargy": 0.04, "aggression": 0.01,
        },
    },
    "bowl": {
        "food_fill_ratio": 0.42,
        "water_fill_ratio": None,
        "food_type_visible": "dry",
    },
    "scene": {
        "lighting": "bright",
        "image_quality": "clear",
        "confidence_overall": 0.85,
    },
    "narrative": "灰色英短以正常速度进食干粮，碗内余粮约42%，状态放松。",
})

# Schema-compliant valid no-pet output
_VALID_NO_PET_OUTPUT = json.dumps({
    "schema_version": "1.0",
    "pet_present": False,
    "pet_count": 0,
    "pet": None,
    "bowl": {
        "food_fill_ratio": 0.5,
        "water_fill_ratio": 0.5,
        "food_type_visible": "dry",
    },
    "scene": {
        "lighting": "bright",
        "image_quality": "clear",
        "confidence_overall": 0.8,
    },
    "narrative": "无宠物",
})


class TestLoadComplianceValSet:
    """Tests for loading the compliance validation subset."""

    def test_loads_jsonl(self, tmp_dir, sample_sharegpt_item):
        """Loads items from a JSONL file."""
        path = tmp_dir / "val.jsonl"
        with open(path, "w") as f:
            for _ in range(10):
                f.write(json.dumps(sample_sharegpt_item) + "\n")
        items = load_compliance_val_set(str(path))
        assert len(items) == 10
        assert items[0]["id"] == "sft_00001"

    def test_empty_file_returns_empty(self, tmp_dir):
        """Empty file returns empty list."""
        path = tmp_dir / "empty.jsonl"
        path.write_text("")
        items = load_compliance_val_set(str(path))
        assert items == []


class TestEvaluateCompliance:
    """Tests for schema compliance evaluation logic."""

    def test_valid_output_passes(self):
        """Valid JSON output that passes schema validation counts as pass."""
        result = evaluate_compliance([_VALID_PET_OUTPUT])
        assert result["passed"] == 1
        assert result["total"] == 1
        assert result["compliance_rate"] == 1.0

    def test_invalid_json_fails(self):
        """Non-JSON output counts as failure."""
        result = evaluate_compliance(["not json at all"])
        assert result["passed"] == 0
        assert result["total"] == 1
        assert len(result["failed_reasons"]) == 1

    def test_mixed_results(self):
        """Compliance rate is correct for mixed valid/invalid outputs."""
        invalid = "garbage output"
        result = evaluate_compliance([_VALID_NO_PET_OUTPUT, invalid, _VALID_NO_PET_OUTPUT])
        assert result["compliance_rate"] == pytest.approx(2 / 3)


class TestSchemaComplianceCallback:
    """Tests for the TrainerCallback wrapper."""

    def test_init_with_params(self, tmp_dir, sample_sharegpt_item):
        """Callback initializes with validation data path."""
        path = tmp_dir / "val.jsonl"
        with open(path, "w") as f:
            for _ in range(5):
                f.write(json.dumps(sample_sharegpt_item) + "\n")

        callback = SchemaComplianceCallback(
            val_path=str(path),
            check_steps=10,
            sample_size=3,
            output_dir=str(tmp_dir / "logs"),
        )
        assert len(callback.val_items) == 5
        assert callback.sample_size == 3
