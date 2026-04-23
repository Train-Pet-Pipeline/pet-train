"""Pre-training JSONL validation — consumer-side F11 defense.

Both pet-annotation (producer) and pet-train (consumer) validate exported
JSONL against pet-schema ShareGPTSFTSample / DPOSample models. If the two
sides ever drift, validation at either end fails fast with a clear error
rather than silently feeding malformed training data to LLaMA-Factory.
"""

from __future__ import annotations

from pathlib import Path

from pet_schema import DPOSample, ShareGPTSFTSample


def validate_sft_jsonl(path: Path) -> int:
    """Validate every line of a ShareGPT SFT JSONL file.

    Args:
        path: Path to the JSONL file to validate.

    Returns:
        Number of valid (non-empty) samples.

    Raises:
        ValueError: On first malformed row with file:line number context.
    """
    count = 0
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ShareGPTSFTSample.model_validate_json(line)
            except Exception as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid ShareGPT SFT row — {e}"
                ) from e
            count += 1
    return count


def validate_dpo_jsonl(path: Path) -> int:
    """Validate every line of an Alpaca DPO JSONL file.

    Args:
        path: Path to the JSONL file to validate.

    Returns:
        Number of valid (non-empty) samples.

    Raises:
        ValueError: On first malformed row with file:line number context.
    """
    count = 0
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                DPOSample.model_validate_json(line)
            except Exception as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid Alpaca DPO row — {e}"
                ) from e
            count += 1
    return count
