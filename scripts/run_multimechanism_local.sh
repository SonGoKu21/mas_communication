#!/usr/bin/env bash
set -euo pipefail
cd "${MAS_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
source ./server-env.sh
source ./qwen-local-env.sh
exec "${MAS_PYTHON:-python3}" run_shopping_multimechanism.py "$@"
