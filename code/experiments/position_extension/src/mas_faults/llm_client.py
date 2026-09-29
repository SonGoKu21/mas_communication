from __future__ import annotations

import json
import hashlib
import os
import signal
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


DEFAULT_PROVIDER = "deepseek"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY = "sk-9d1670f917fc4687afc7bc430fdb7ef4"

class ChatClient(Protocol):
    @property
    def call_count(self) -> int:
        ...

    @property
    def prompt_tokens(self) -> int:
        ...

    @property
    def completion_tokens(self) -> int:
        ...

    @property
    def model_info(self) -> "ModelInfo":
        ...

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        ...


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    base_url: str
    model: str
    client_type: str


class MissingLLMAPIKey(RuntimeError):
    pass


@contextmanager
def request_deadline(seconds: int):
    """Enforce a wall-clock deadline for a synchronous API call on Unix hosts."""
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"LLM request exceeded {seconds}s total deadline")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def extract_completion_content(data: dict[str, Any]) -> str:
    message = data["choices"][0]["message"]
    content = message.get("content")
    if content:
        return str(content)
    return str(message.get("reasoning_content") or "")


def build_completion_payload(
    *, provider: str, model: str, prompt: str, json_mode: bool = False, disable_thinking: bool | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if disable_thinking is None:
        disable_thinking = os.environ.get("LLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes"}
    if disable_thinking:
        if provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    max_tokens = os.environ.get("LLM_MAX_TOKENS")
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


class MockLLMClient:
    def __init__(self) -> None:
        self._info = get_model_info(mock_llm=True)

    @property
    def call_count(self) -> int:
        return 0

    @property
    def prompt_tokens(self) -> int:
        return 0

    @property
    def completion_tokens(self) -> int:
        return 0

    @property
    def model_info(self) -> ModelInfo:
        return self._info

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        return "MOCK_RESPONSE"


class OpenAICompatibleHTTPClient:
    def __init__(self, *, api_key: str, base_url: str, model: str, provider: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.disable_thinking = os.environ.get("LLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes"}
        self._call_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_log: list[dict[str, Any]] = []
        self._info = ModelInfo(provider=provider, base_url=self.base_url, model=model, client_type="http_openai_compatible")

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def model_info(self) -> ModelInfo:
        return self._info

    @property
    def request_log(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._request_log]

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        return self.complete_with_metadata(prompt, json_mode=json_mode, metadata={})

    def complete_with_metadata(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        self._call_count += 1
        payload = build_completion_payload(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            json_mode=json_mode,
            disable_thinking=self.disable_thinking,
        )
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        total_timeout_seconds = int(os.environ.get("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", self.timeout_seconds))
        try:
            with request_deadline(total_timeout_seconds):
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    usage = data.get("usage") or {}
                    self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self._completion_tokens += int(usage.get("completion_tokens") or 0)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider} HTTP error {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.provider} request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"{self.provider} request failed: {exc}") from exc
        content = extract_completion_content(data)
        usage = data.get("usage") or {}
        audit = {
            "request_index": self._call_count,
            "provider_request_id": str(data.get("id") or ""),
            "provider": self.provider,
            "model": self.model,
            "json_mode": json_mode,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_cache_hit_tokens": int(
                usage.get("prompt_cache_hit_tokens") or 0
            ),
            "prompt_cache_miss_tokens": int(
                usage.get("prompt_cache_miss_tokens") or 0
            ),
            **dict(metadata or {}),
        }
        self._request_log.append(audit)
        return content


class AutoGenOpenAIAdapter:
    def __init__(self, client: Any) -> None:
        self.client = client

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        raise RuntimeError(
            "AutoGen OpenAI-compatible client was constructed, but direct synchronous completion is not "
            "used by this runner. Use the HTTP-compatible client or --mock-llm."
        )


def get_provider() -> str:
    return os.environ.get("LLM_PROVIDER") or os.environ.get("DEEPSEEK_PROVIDER") or DEFAULT_PROVIDER


def get_api_key() -> str | None:
    return os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or DEFAULT_API_KEY


def get_base_url() -> str:
    return os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL


def get_model() -> str:
    return os.environ.get("LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL


def get_model_info(*, mock_llm: bool = False) -> ModelInfo:
    provider = get_provider()
    base_url = get_base_url()
    model = get_model()
    client_type = "mock" if mock_llm else "openai_compatible_http_or_autogen"
    return ModelInfo(provider=provider, base_url=base_url, model=model, client_type=client_type)


def print_model_info(info: ModelInfo) -> None:
    print(f"provider={info.provider}")
    print(f"base_url={info.base_url}")
    print(f"model={info.model}")
    print(f"client_type={info.client_type}")


def maybe_get_mock_client(mock_llm: bool) -> ChatClient | None:
    if mock_llm:
        info = get_model_info(mock_llm=True)
        print_model_info(info)
        return MockLLMClient()
    return None


def get_llm_client(*, mock_llm: bool = False) -> ChatClient:
    mock_client = maybe_get_mock_client(mock_llm)
    if mock_client is not None:
        return mock_client

    provider = get_provider()
    api_key = get_api_key()
    base_url = get_base_url()
    model = get_model()
    if not api_key:
        raise MissingLLMAPIKey(
            "LLM_API_KEY is required unless --mock-llm is used. "
            "Set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL for ModelScope, DashScope, DeepSeek, or any "
            "OpenAI-compatible endpoint. Legacy DEEPSEEK_API_KEY variables are still supported."
        )

    timeout_seconds = int(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    client = OpenAICompatibleHTTPClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        provider=provider,
        timeout_seconds=timeout_seconds,
    )
    print_model_info(client.model_info)
    return client


def get_deepseek_client(*, mock_llm: bool = False) -> ChatClient:
    return get_llm_client(mock_llm=mock_llm)


MissingDeepSeekAPIKey = MissingLLMAPIKey
DeepSeekHTTPClient = OpenAICompatibleHTTPClient
