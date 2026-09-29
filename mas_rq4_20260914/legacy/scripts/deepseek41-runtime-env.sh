#!/usr/bin/env bash
# Source this file without shell tracing. Credentials stay outside the code tree.
export MAS_CODE_ROOT="${MAS_CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MAS_DATA_ROOT="${MAS_DATA_ROOT:?Set MAS_DATA_ROOT to your storage directory}"
export PYTHONPATH="$MAS_CODE_ROOT/src:$MAS_CODE_ROOT"
export TMPDIR="$MAS_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$MAS_DATA_ROOT/cache"
export PYTHONDONTWRITEBYTECODE='1'
source "${MAS_CREDENTIALS_FILE:?Set MAS_CREDENTIALS_FILE to your private environment file}"
export LLM_MODEL='deepseek-flash'
export LLM_DISABLE_THINKING='1'
export LLM_MAX_TOKENS='2048'
export LLM_REQUEST_TIMEOUT_SECONDS='90'
export LLM_TOTAL_REQUEST_TIMEOUT_SECONDS='120'
export MAS_DEEPSEEK_OFFPEAK_ONLY='1'
