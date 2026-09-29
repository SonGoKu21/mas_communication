#!/usr/bin/env bash
set -euo pipefail
cd "${MAS_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
source ./server-env.sh
source ./qwen-local-env.sh
exec "${MAS_PYTHON:-python3}" run_shopping_mitigation.py \
  --base-url http://127.0.0.1:17770 \
  --task-manifest "$MAS_DATA_ROOT/datasets/shopping_mitigation_tasks_20260910_v1.json" \
  --output-dir "$MAS_DATA_ROOT/results/shopping_mitigation_qwen38_27b_135r_20260910_v2" \
  --required-model Qwen/Qwen3.8-27B --repetitions 1 "$@"
