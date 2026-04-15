"""Training-time schema compliance monitoring callback.

Periodically samples model outputs during training and validates them
against pet-schema to detect training collapse early.
"""

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

from pet_schema import validate_output

logger = logging.getLogger(__name__)


def load_compliance_val_set(path: str) -> list[dict]:
    """Load compliance validation items from JSONL file.

    Args:
        path: Path to JSONL file in ShareGPT format.

    Returns:
        List of parsed JSON items.
    """
    items: list[dict] = []
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Compliance validation file not found: %s", path)
        return items
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d compliance validation items from %s", len(items), path)
    return items


def evaluate_compliance(model_outputs: list[str]) -> dict:
    """Evaluate schema compliance rate for a batch of model outputs.

    Args:
        model_outputs: List of raw model output strings (expected JSON).

    Returns:
        Dict with keys: total, passed, compliance_rate, failed_ids, failed_reasons.
    """
    total = len(model_outputs)
    passed = 0
    failed_reasons = []

    for i, output in enumerate(model_outputs):
        try:
            result = validate_output(output)
            if result.valid:
                passed += 1
            else:
                failed_reasons.append({
                    "index": i,
                    "errors": [str(e) for e in result.errors],
                })
        except Exception as e:
            failed_reasons.append({
                "index": i,
                "errors": [f"Validation exception: {e}"],
            })

    return {
        "total": total,
        "passed": passed,
        "compliance_rate": passed / total if total > 0 else 0.0,
        "failed_reasons": failed_reasons,
    }


class SchemaComplianceCallback:
    """LLaMA-Factory compatible callback for schema compliance monitoring.

    Triggers every `check_steps` steps, samples `sample_size` items from
    the validation set, generates model outputs, and validates them.

    Args:
        val_path: Path to compliance validation JSONL file.
        check_steps: Run compliance check every N training steps.
        sample_size: Number of items to sample per check.
        output_dir: Directory for compliance_log.jsonl output.
    """

    def __init__(
        self,
        val_path: str,
        check_steps: int = 500,
        sample_size: int = 20,
        output_dir: str = "outputs/compliance",
    ):
        """Initialize SchemaComplianceCallback."""
        self.val_items = load_compliance_val_set(val_path)
        self.check_steps = check_steps
        self.sample_size = min(sample_size, len(self.val_items))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "compliance_log.jsonl"

    def sample_items(self) -> list[dict]:
        """Randomly sample items from validation set for compliance check.

        Returns:
            List of sampled validation items.
        """
        if not self.val_items:
            return []
        return random.sample(self.val_items, k=self.sample_size)

    def log_result(self, step: int, result: dict) -> None:
        """Append compliance check result to log file.

        Args:
            step: Current training step.
            result: Compliance evaluation result dict.
        """
        entry = {
            "step": step,
            "compliance_rate": result["compliance_rate"],
            "total": result["total"],
            "passed": result["passed"],
            "failed_reasons": result["failed_reasons"],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(
            "Step %d: schema compliance = %.1f%% (%d/%d)",
            step,
            result["compliance_rate"] * 100,
            result["passed"],
            result["total"],
        )
