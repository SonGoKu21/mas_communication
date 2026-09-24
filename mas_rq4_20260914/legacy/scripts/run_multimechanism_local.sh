#!/usr/bin/env bash
set -euo pipefail
cd '/home/hqn/zjh code/mas'
source ./server-env.sh
source ./qwen-local-env.sh
exec /opt/miniconda3/envs/mcal/bin/python run_shopping_multimechanism.py "$@"
