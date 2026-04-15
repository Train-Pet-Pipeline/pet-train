#!/bin/bash
# DPO training wrapper for LLaMA-Factory.
# Usage: bash scripts/train_dpo.sh configs/experiments/dpo_user_feedback_v1.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 <experiment_config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
BASE_CONFIG="$PROJECT_DIR/configs/base/dpo_base.yaml"
OUTPUT_DIR="$PROJECT_DIR/outputs/dpo/$RUN_NAME"

PARAMS_FILE="$PROJECT_DIR/params.yaml"
MIN_PAIRS=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['dpo']['min_pairs'])")
SFT_ADAPTER=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['dpo']['sft_adapter_path'])")
DPO_DATA=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['dpo']['data_path'])")

if [ -z "$SFT_ADAPTER" ] || [ ! -d "$SFT_ADAPTER" ]; then
    echo "ERROR: SFT adapter path not set or does not exist: $SFT_ADAPTER"
    echo "Set dpo.sft_adapter_path in params.yaml to a valid SFT output directory."
    exit 1
fi

if [ ! -f "$DPO_DATA" ]; then
    echo "ERROR: DPO data file not found: $DPO_DATA"
    exit 1
fi

PAIR_COUNT=$(wc -l < "$DPO_DATA" | tr -d ' ')
if [ "$PAIR_COUNT" -lt "$MIN_PAIRS" ]; then
    echo "ERROR: Insufficient DPO pairs. Found $PAIR_COUNT, need >= $MIN_PAIRS."
    echo "Collect more DPO data via pet-annotation before running DPO training."
    exit 1
fi

echo "=== pet-train DPO ==="
echo "Experiment: $RUN_NAME"
echo "SFT adapter: $SFT_ADAPTER"
echo "DPO pairs: $PAIR_COUNT (min: $MIN_PAIRS)"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

llamafactory-cli train \
    --config "$BASE_CONFIG" \
    --adapter_name_or_path "$SFT_ADAPTER" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME"

echo "=== DPO training complete ==="
echo "Adapter saved to: $OUTPUT_DIR"

if command -v pet-eval &>/dev/null; then
    bash "$SCRIPT_DIR/eval_after_train.sh" "$OUTPUT_DIR" "$RUN_NAME"
else
    echo "pet-eval not installed, skipping post-training evaluation."
fi
