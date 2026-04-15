#!/bin/bash
# Audio CNN training via PyTorch Lightning.
# Usage: bash scripts/train_audio.sh configs/audio/mobilenetv2_transfer_v1.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 <audio_config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
OUTPUT_DIR="$PROJECT_DIR/outputs/audio/$RUN_NAME"

echo "=== pet-train Audio CNN ==="
echo "Experiment: $RUN_NAME"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python3 -m pet_train.audio_model \
    --config "$CONFIG_PATH" \
    --params "$PROJECT_DIR/params.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME"

echo "=== Audio training complete ==="
echo "Checkpoint saved to: $OUTPUT_DIR"
