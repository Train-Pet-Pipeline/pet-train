#!/bin/bash
# DPO training wrapper for LLaMA-Factory.
# Usage: bash scripts/train_dpo.sh configs/experiments/dpo_user_feedback_v1.yaml

set -euo pipefail

# Disable Python output buffering so training progress is visible in real-time
export PYTHONUNBUFFERED=1

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

MERGED_CONFIG="$OUTPUT_DIR/_merged_config.yaml"
python3 -c "
import yaml, sys, torch

# --- Merge configs ---
with open('$BASE_CONFIG') as f:
    base = yaml.safe_load(f) or {}
exp_path = '$CONFIG_PATH'
with open(exp_path) as f:
    exp = yaml.safe_load(f) or {}
base.update(exp)
base['output_dir'] = '$OUTPUT_DIR'
base['run_name'] = '$RUN_NAME'
base['adapter_name_or_path'] = '$SFT_ADAPTER'

# --- Device-aware defaults ---
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Detected device: {device}')

if device in ('mps', 'cpu'):
    for key in ('bf16', 'fp16', 'pure_bf16'):
        if base.get(key):
            print(f'WARNING: {key} disabled — {device} does not fully support mixed precision')
            base[key] = False

# --- VLM cutoff_len guard ---
template = base.get('template', '')
if 'vl' in template.lower():
    image_max_pixels = base.get('image_max_pixels', 768 * 768)
    patch_size, merge_size = 14, 2
    max_image_tokens = image_max_pixels // (patch_size ** 2 * merge_size ** 2)
    text_headroom = 256
    min_cutoff = max_image_tokens + text_headroom
    cutoff_len = base.get('cutoff_len', 4096)
    if cutoff_len < min_cutoff:
        print(f'ERROR: cutoff_len={cutoff_len} too short for VLM. Min={min_cutoff}.')
        print(f'Fix: increase cutoff_len or reduce image_max_pixels.')
        sys.exit(1)
    print(f'VLM guard: max_img_tokens={max_image_tokens}, cutoff_len={cutoff_len} OK')

with open('$MERGED_CONFIG', 'w') as f:
    yaml.dump(base, f, default_flow_style=False)
"

llamafactory-cli train "$MERGED_CONFIG"

echo "=== DPO training complete ==="
echo "Adapter saved to: $OUTPUT_DIR"

if python3 -c "import pet_eval" 2>/dev/null; then
    bash "$SCRIPT_DIR/eval_after_train.sh" "$OUTPUT_DIR" "$RUN_NAME"
else
    echo "pet-eval not installed, skipping post-training evaluation."
fi
