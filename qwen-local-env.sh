# Source after server-env.sh to explicitly select the verified local model.
export LLM_PROVIDER='modelscope_local'
export LLM_BASE_URL='http://127.0.0.1:18001/v1'
export LLM_MODEL='Qwen/Qwen3.8-27B'
export LLM_API_KEY='local-no-secret'
export LLM_MAX_TOKENS=2048
export LLM_DISABLE_THINKING=1
export LLM_REQUEST_TIMEOUT_SECONDS=90
export LLM_TOTAL_REQUEST_TIMEOUT_SECONDS=120
