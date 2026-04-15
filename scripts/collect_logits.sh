#!/bin/bash
# Collect teacher logits for KL distillation.
# Usage: bash scripts/collect_logits.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PARAMS_FILE="$PROJECT_DIR/params.yaml"

ENABLED=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS_FILE'))['kl_distillation']['enabled'])")

if [ "$ENABLED" != "True" ]; then
    echo "KL distillation is disabled in params.yaml. Set kl_distillation.enabled: true to collect logits."
    exit 0
fi

echo "=== Collecting Teacher Logits ==="

python3 -c "
import yaml
import json
from pathlib import Path
from pet_train.logits_provider import FileLogitsProvider, APILogitsProvider

params = yaml.safe_load(open('$PARAMS_FILE'))
kl_cfg = params['kl_distillation']
sft_cfg = params['sft']

data_path = sft_cfg['data_path']
sample_ids = []
with open(data_path) as f:
    for line in f:
        item = json.loads(line.strip())
        sample_ids.append(item['id'])

print(f'Total training samples: {len(sample_ids)}')

provider_type = kl_cfg['provider']
logits_dir = kl_cfg['logits_dir']

if provider_type == 'file':
    provider = FileLogitsProvider(logits_dir=logits_dir)
    cached = provider.available_samples
    missing = [s for s in sample_ids if s not in cached]
    print(f'Cached: {len(cached)}, Missing: {len(missing)}')
    if missing:
        print('ERROR: Missing logits for file provider. Generate them offline.')
        for s in missing[:10]:
            print(f'  - {s}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')
elif provider_type == 'api':
    print('API logits collection not yet implemented in this script.')
    print('Use the Python API directly: APILogitsProvider.cache_sample()')
"

echo "=== Collection complete ==="
