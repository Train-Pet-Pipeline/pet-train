#!/bin/bash
# Merge LoRA adapter weights into base model for quantization.
# Usage: bash scripts/merge_lora.sh <adapter_path> <output_path>

set -euo pipefail

ADAPTER_PATH="${1:?Usage: $0 <adapter_path> <output_path>}"
OUTPUT_PATH="${2:?Usage: $0 <adapter_path> <output_path>}"

echo "=== pet-train LoRA Merge ==="
echo "Adapter: $ADAPTER_PATH"
echo "Output: $OUTPUT_PATH"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter path does not exist: $ADAPTER_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template qwen2_vl \
    --finetuning_type lora \
    --export_dir "$OUTPUT_PATH" \
    --export_size 2 \
    --export_legacy_format false

echo "=== Merge complete ==="
echo "Full model saved to: $OUTPUT_PATH"
