# Source this file after activating /opt/miniconda3/envs/mcal.
export MAS_CODE_ROOT='/home/hqn/zjh code/mas'
export MAS_DATA_ROOT='/data3/hqn/mas'
export MODELSCOPE_CACHE="$MAS_DATA_ROOT/cache/modelscope"
export PIP_CACHE_DIR="$MAS_DATA_ROOT/cache/pip"
export PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple'
export CONDA_PKGS_DIRS="$MAS_DATA_ROOT/cache/conda/pkgs"
export PLAYWRIGHT_BROWSERS_PATH="$MAS_DATA_ROOT/browser"
export XDG_CACHE_HOME="$MAS_DATA_ROOT/cache/xdg"
export TMPDIR="$MAS_DATA_ROOT/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$MAS_CODE_ROOT/src:$MAS_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Fail closed until a real local model endpoint is explicitly selected.
export LLM_PROVIDER='modelscope_local'
export LLM_BASE_URL='http://127.0.0.1:18001/v1'
export LLM_MODEL='UNCONFIGURED'
export LLM_API_KEY='local-no-secret'
