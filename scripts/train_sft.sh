#!/bin/bash
# SFT training wrapper for LLaMA-Factory.
# Usage: bash scripts/train_sft.sh configs/experiments/sft_lora_r16_lr2e4_ep3.yaml

set -euo pipefail

# Disable Python output buffering so training progress is visible in real-time
export PYTHONUNBUFFERED=1

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

# --- Device-aware defaults ---
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Detected device: {device}')

if device == 'mps':
    # MPS does not reliably support bf16/fp16 training
    base.setdefault('bf16', False)
    base.setdefault('fp16', False)
    base.setdefault('pure_bf16', False)
    if base.get('bf16') or base.get('fp16') or base.get('pure_bf16'):
        print('WARNING: Mixed precision disabled — MPS does not fully support bf16/fp16 training')
        base['bf16'] = False
        base['fp16'] = False
        base['pure_bf16'] = False
elif device == 'cpu':
    base.setdefault('bf16', False)
    base.setdefault('fp16', False)
    base.setdefault('pure_bf16', False)
# GPU: respect config as-is (bf16 recommended)

# --- VLM cutoff_len guard ---
# For VLM models, image tokens = image_pixels / (patch_size^2 * merge_size^2).
# cutoff_len must exceed max_image_tokens + text_tokens, otherwise truncation
# splits image tokens from features causing shape mismatch at training time.
template = base.get('template', '')
if 'vl' in template.lower():
    image_max_pixels = base.get('image_max_pixels', 768 * 768)
    # Qwen2-VL: patch_size=14, spatial_merge_size=2
    patch_size, merge_size = 14, 2
    max_image_tokens = image_max_pixels // (patch_size ** 2 * merge_size ** 2)
    text_headroom = 256  # prompt + response tokens
    min_cutoff = max_image_tokens + text_headroom
    cutoff_len = base.get('cutoff_len', 4096)
    if cutoff_len < min_cutoff:
        print(f'ERROR: cutoff_len={cutoff_len} too short for VLM template \"{template}\".')
        print(f'image_max_pixels={image_max_pixels} -> up to {max_image_tokens} image tokens.')
        print(f'Minimum cutoff_len = {max_image_tokens} + {text_headroom} (headroom) = {min_cutoff}.')
        print(f'Fix: increase cutoff_len or reduce image_max_pixels in your config.')
        sys.exit(1)
    print(f'VLM guard: image_max_pixels={image_max_pixels}, max_img_tokens={max_image_tokens}, cutoff_len={cutoff_len} (min={min_cutoff}) OK')

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
