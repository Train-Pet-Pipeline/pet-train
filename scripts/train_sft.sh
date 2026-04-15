#!/bin/bash
# SFT training wrapper for LLaMA-Factory.
# Usage: bash scripts/train_sft.sh configs/experiments/sft_lora_r16_lr2e4_ep3.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 <experiment_config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
BASE_CONFIG="$PROJECT_DIR/configs/base/sft_base.yaml"
OUTPUT_DIR="$PROJECT_DIR/outputs/sft/$RUN_NAME"

echo "=== pet-train SFT ==="
echo "Experiment: $RUN_NAME"
echo "Base config: $BASE_CONFIG"
echo "Experiment config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

MERGED_CONFIG="$OUTPUT_DIR/_merged_config.yaml"
python3 -c "
import yaml, sys
with open('$BASE_CONFIG') as f:
    base = yaml.safe_load(f) or {}
exp_path = '$CONFIG_PATH'
with open(exp_path) as f:
    exp = yaml.safe_load(f) or {}
base.update(exp)
base['output_dir'] = '$OUTPUT_DIR'
base['run_name'] = '$RUN_NAME'
with open('$MERGED_CONFIG', 'w') as f:
    yaml.dump(base, f, default_flow_style=False)
"

echo "Starting training..."
llamafactory-cli train "$MERGED_CONFIG"

echo "=== SFT training complete ==="
echo "Adapter saved to: $OUTPUT_DIR"

if command -v pet-eval &>/dev/null; then
    echo "Running post-training evaluation..."
    bash "$SCRIPT_DIR/eval_after_train.sh" "$OUTPUT_DIR" "$RUN_NAME"
else
    echo "pet-eval not installed, skipping post-training evaluation."
    echo "Install pet-eval and run: bash scripts/eval_after_train.sh $OUTPUT_DIR $RUN_NAME"
fi
