#!/usr/bin/env bash
set -euo pipefail
cd '/home/hqn/zjh code/mas'
source ./server-env.sh
source /opt/miniconda3/envs/mcal/vllm-runtime/bin/activate
export VLLM_USE_MODELSCOPE=true
export CUDA_VISIBLE_DEVICES="${MAS_CUDA_DEVICES:-4,5,6,7}"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export FLASHINFER_WORKSPACE_BASE="$MAS_DATA_ROOT/cache"
export TRITON_CACHE_DIR="$MAS_DATA_ROOT/cache/triton"
export CUDA_CACHE_PATH="$MAS_DATA_ROOT/cache/cuda"
# Conda SQLite/ICU requires a newer C++ ABI than the host's system library.
export LD_PRELOAD="/opt/miniconda3/envs/mcal/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
test -f /data3/hqn/mas/artifacts/qwen38_27b_download_manifest_20260910_verified.json
exec /opt/miniconda3/envs/mcal/vllm-runtime/bin/vllm serve \
  /data3/hqn/mas/models/Qwen3.8-27B \
  --served-model-name Qwen/Qwen3.8-27B \
  --host 127.0.0.1 --port 18001 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.72 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --generation-config vllm \
  --enforce-eager \
  --disable-custom-all-reduce
