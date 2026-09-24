#!/usr/bin/env bash
set -euo pipefail
cd '/home/hqn/zjh code/mas'
source ./server-env.sh
source ./qwen-local-env.sh
exec /opt/miniconda3/envs/mcal/bin/python run_shopping_mitigation.py \
  --base-url http://127.0.0.1:17770 \
  --task-manifest /data3/hqn/mas/datasets/shopping_mitigation_tasks_20260910_v1.json \
  --output-dir /data3/hqn/mas/results/shopping_mitigation_qwen38_27b_135r_20260910_v2 \
  --required-model Qwen/Qwen3.8-27B --repetitions 1 "$@"
