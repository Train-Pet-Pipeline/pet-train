"""Tests for consumer-side JSONL validation (F11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pet_train.plugins.data_validation import validate_dpo_jsonl, validate_sft_jsonl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_SFT_ROW = json.dumps(
    {
        "conversations": [
            {"from_": "human", "value": "Analyze this scene."},
            {"from_": "gpt", "value": "The cat is eating."},
        ],
        "system": None,
        "tools": None,
        "sample_id": "sft_001",
        "source_target_id": None,
        "annotator_id": "qwen2.5-vl-72b",
    }
)

VALID_DPO_ROW = json.dumps(
    {
        "prompt": "Describe the pet's behavior.",
        "chosen": "The cat is eating calmly.",
        "rejected": "idk",
        "sample_id": "dpo_001",
        "chosen_annotator_id": "vlm_72b",
        "rejected_annotator_id": "vlm_7b",
        "storage_uri": None,
    }
)


# ---------------------------------------------------------------------------
# SFT validator
# ---------------------------------------------------------------------------


def test_validate_sft_jsonl_valid_file(tmp_path: Path) -> None:
    """validate_sft_jsonl returns correct count for a valid file."""
    f = tmp_path / "train.jsonl"
    f.write_text(VALID_SFT_ROW + "\n" + VALID_SFT_ROW + "\n")
    count = validate_sft_jsonl(f)
    assert count == 2


def test_validate_sft_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    """validate_sft_jsonl ignores blank lines."""
    f = tmp_path / "train.jsonl"
    f.write_text("\n" + VALID_SFT_ROW + "\n\n" + VALID_SFT_ROW + "\n")
    count = validate_sft_jsonl(f)
    assert count == 2


def test_validate_sft_jsonl_malformed_raises_with_line_number(tmp_path: Path) -> None:
    """validate_sft_jsonl raises ValueError with file:line on malformed row."""
    bad_row = json.dumps({"not_conversations": "oops"})
    f = tmp_path / "train.jsonl"
    f.write_text(VALID_SFT_ROW + "\n" + bad_row + "\n")
    with pytest.raises(ValueError, match=r"train\.jsonl:2:"):
        validate_sft_jsonl(f)


def test_validate_sft_jsonl_invalid_json_raises(tmp_path: Path) -> None:
    """validate_sft_jsonl raises ValueError on unparseable JSON."""
    f = tmp_path / "train.jsonl"
    f.write_text(VALID_SFT_ROW + "\n" + "NOT_JSON\n")
    with pytest.raises(ValueError, match="train.jsonl:2:"):
        validate_sft_jsonl(f)


# ---------------------------------------------------------------------------
# DPO validator
# ---------------------------------------------------------------------------


def test_validate_dpo_jsonl_valid_file(tmp_path: Path) -> None:
    """validate_dpo_jsonl returns correct count for a valid file."""
    f = tmp_path / "dpo.jsonl"
    f.write_text(VALID_DPO_ROW + "\n")
    count = validate_dpo_jsonl(f)
    assert count == 1


def test_validate_dpo_jsonl_malformed_raises_with_line_number(tmp_path: Path) -> None:
    """validate_dpo_jsonl raises ValueError with file:line on malformed row."""
    bad_row = json.dumps({"prompt": "ok"})  # missing required fields
    f = tmp_path / "dpo.jsonl"
    f.write_text(VALID_DPO_ROW + "\n" + bad_row + "\n")
    with pytest.raises(ValueError, match=r"dpo\.jsonl:2:"):
        validate_dpo_jsonl(f)
