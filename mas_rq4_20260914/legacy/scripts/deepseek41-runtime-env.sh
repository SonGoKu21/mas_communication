#!/usr/bin/env bash
# Source this file without shell tracing. Credentials stay outside the code tree.
export MAS_CODE_ROOT='/data3/hqn/mas/code/deepseek41_multimechanism_20260912_v2'
export MAS_DATA_ROOT='/data3/hqn/mas'
export PYTHONPATH="$MAS_CODE_ROOT/src:$MAS_CODE_ROOT"
export TMPDIR="$MAS_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$MAS_DATA_ROOT/cache"
export PYTHONDONTWRITEBYTECODE='1'
source /data3/hqn/mas/private_config/deepseek-flash-env.sh
export LLM_MODEL='deepseek-flash'
export LLM_DISABLE_THINKING='1'
export LLM_MAX_TOKENS='2048'
export LLM_REQUEST_TIMEOUT_SECONDS='90'
export LLM_TOTAL_REQUEST_TIMEOUT_SECONDS='120'
export MAS_DEEPSEEK_OFFPEAK_ONLY='1'
