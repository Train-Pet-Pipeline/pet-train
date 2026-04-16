#!/bin/bash
# Trigger pet-eval after training completes.
# Usage: bash scripts/eval_after_train.sh <model_path> <run_name>

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path> <run_name>}"
RUN_NAME="${2:?Usage: $0 <model_path> <run_name>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Post-Training Evaluation ==="

if ! python3 -c "import pet_eval" 2>/dev/null; then
    echo "WARNING: pet-eval is not installed. Skipping post-training evaluation."
    echo "Install pet-eval and re-run: bash scripts/eval_after_train.sh $MODEL_PATH $RUN_NAME"
    exit 0
fi

# Locate pet-eval params.yaml: sibling repo or fall back to current directory
EVAL_PARAMS=""
for candidate in "$PROJECT_DIR/../pet-eval/params.yaml" "params.yaml"; do
    if [ -f "$candidate" ]; then
        EVAL_PARAMS="$candidate"
        break
    fi
done

if [ -z "$EVAL_PARAMS" ]; then
    echo "WARNING: pet-eval params.yaml not found. Skipping evaluation."
    exit 0
fi

echo "Running pet-eval for: $RUN_NAME"
echo "Model path: $MODEL_PATH"
echo "Eval params: $EVAL_PARAMS"

python3 -m pet_eval.runners.eval_trained \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --params "$EVAL_PARAMS"

echo "=== Evaluation complete ==="
