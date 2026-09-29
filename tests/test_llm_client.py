import os
import json
import unittest
import io
import urllib.error
from contextlib import nullcontext
import pytest
from unittest.mock import patch

from mas_faults.llm_client import OpenAICompatibleHTTPClient, build_completion_payload, extract_completion_content, get_llm_client


class CompletionContentTests(unittest.TestCase):
    def test_prefers_content(self) -> None:
        self.assertEqual(extract_completion_content({"choices": [{"message": {"content": "patch", "reasoning_content": "thought"}}]}), "patch")

    def test_uses_reasoning_content_when_content_is_empty(self) -> None:
        self.assertEqual(extract_completion_content({"choices": [{"message": {"content": "", "reasoning_content": "patch"}}]}), "patch")

    def test_disables_thinking_with_deepseek_native_parameter(self) -> None:
        with patch.dict(os.environ, {"LLM_DISABLE_THINKING": "1"}, clear=False):
            payload = build_completion_payload(provider="deepseek", model="deepseek-v4-pro", prompt="fix")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("chat_template_kwargs", payload)

    def test_disables_thinking_with_modelscope_parameter(self) -> None:
        with patch.dict(os.environ, {"LLM_DISABLE_THINKING": "1"}, clear=False):
            payload = build_completion_payload(provider="modelscope_local", model="Qwen/Qwen3-8B", prompt="fix")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_json_mode_uses_openai_compatible_response_format(self) -> None:
        payload = build_completion_payload(provider="deepseek", model="deepseek-v4-pro", prompt="fix", json_mode=True)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_total_request_timeout_is_reported_as_an_api_error(self) -> None:
        client = OpenAICompatibleHTTPClient(
            api_key="test-key", base_url="https://example.invalid", model="test", provider="test", timeout_seconds=1
        )
        with patch("urllib.request.urlopen", side_effect=TimeoutError("deadline")):
            with self.assertRaisesRegex(RuntimeError, "request failed: TimeoutError"):
                client.complete("prompt")

    def test_client_uses_configured_socket_timeout(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://example.invalid", "LLM_MODEL": "test", "LLM_REQUEST_TIMEOUT_SECONDS": "123"},
            clear=False,
        ):
            client = get_llm_client()
        self.assertIsInstance(client, OpenAICompatibleHTTPClient)
        self.assertEqual(client.timeout_seconds, 123)

    def test_role_aware_completion_records_provider_request_audit_without_prompt_text(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "id": "request-123",
                        "choices": [{"message": {"content": "result"}}],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 3,
                            "prompt_cache_hit_tokens": 5,
                            "prompt_cache_miss_tokens": 6,
                        },
                    }
                ).encode("utf-8")

        client = OpenAICompatibleHTTPClient(
            api_key="test-key",
            base_url="https://example.invalid",
            model="test-model",
            provider="test-provider",
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = client.complete_with_metadata(
                "secret prompt text",
                json_mode=True,
                metadata={"agent_role": "Planner"},
            )

        self.assertEqual(result, "result")
        self.assertEqual(len(client.request_log), 1)
        audit = client.request_log[0]
        self.assertEqual(audit["provider_request_id"], "request-123")
        self.assertEqual(audit["agent_role"], "Planner")
        self.assertEqual(audit["prompt_tokens"], 11)
        self.assertEqual(audit["completion_tokens"], 3)
        self.assertEqual(audit["prompt_cache_hit_tokens"], 5)
        self.assertEqual(audit["prompt_cache_miss_tokens"], 6)
        self.assertIn("prompt_sha256", audit)
        self.assertIn("response_sha256", audit)
        self.assertNotIn("prompt", audit)
        self.assertNotIn("response", audit)


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")


def http_client():
    return OpenAICompatibleHTTPClient(api_key="secret-key", base_url="https://example.invalid",
                                      model="test", provider="test", timeout_seconds=0)


@pytest.mark.parametrize("failure", [
    urllib.error.HTTPError("https://example.invalid", 500, "secret-body", {}, io.BytesIO(b"secret-body")),
    urllib.error.URLError("secret-key"), TimeoutError("secret-prompt"),
])
def test_failed_transport_has_one_private_unknown_usage_audit(failure):
    client = http_client()
    with patch("urllib.request.urlopen", side_effect=failure) as send:
        with pytest.raises(RuntimeError) as caught:
            client.complete("secret-prompt")
    assert send.call_count == client.call_count == len(client.request_log) == 1
    audit = client.request_log[0]
    assert audit["status"] == "error"
    assert audit["error_type"] == type(failure).__name__
    assert audit["exception_chain"] == [type(failure).__name__]
    assert all(audit[key] is None for key in TOKEN_FIELDS)
    assert audit["response_sha256"] is None
    for secret in ("secret-body", "secret-key", "secret-prompt"):
        assert secret not in json.dumps(audit) + str(caught.value)


@pytest.mark.parametrize("body, error", [(b"secret-body", json.JSONDecodeError), (b"\xff", UnicodeDecodeError),
                                         (b"[]", AttributeError), (b"{}", KeyError)])
def test_parse_and_extraction_failures_are_audited(body, error):
    client = http_client()
    with patch("urllib.request.urlopen", return_value=nullcontext(io.BytesIO(body))):
        with pytest.raises(error):
            client.complete("secret-prompt")
    assert client.call_count == len(client.request_log) == 1
    assert client.request_log[0]["error_type"] == error.__name__
    assert all(client.request_log[0][key] is None for key in TOKEN_FIELDS)


@pytest.mark.parametrize("value, expected", [(7, 7), (0, 0), (None, None), (-1, None),
                                            (True, None), (1.5, None), ("7", None), ({}, None)])
def test_extraction_failure_preserves_only_valid_received_usage(value, expected):
    client = http_client()
    body = json.dumps({"usage": {"prompt_tokens": value, "completion_tokens": 3}}).encode()
    with patch("urllib.request.urlopen", return_value=nullcontext(io.BytesIO(body))):
        with pytest.raises(KeyError):
            client.complete("prompt")
    audit = client.request_log[0]
    assert audit["prompt_tokens"] == expected
    assert audit["completion_tokens"] == 3
    assert client.prompt_tokens == (expected or 0)
    assert client.completion_tokens == 3


def test_metadata_cannot_override_core_fields_on_success_or_failure():
    client = http_client()
    body = json.dumps({"choices": [{"message": {"content": "ok"}}],
                       "usage": {"prompt_tokens": 2, "completion_tokens": 1}}).encode()
    overrides = {"request_index": 999, "status": "forged", "error_type": "forged",
                 "exception_chain": ["forged"], "prompt_tokens": 999, "response_sha256": "forged",
                 "prompt_sha256": "forged", "agent_role": "Planner"}
    with patch("urllib.request.urlopen", return_value=nullcontext(io.BytesIO(body))):
        assert client.complete_with_metadata("prompt", metadata=overrides) == "ok"
    with patch("urllib.request.urlopen", side_effect=TimeoutError("secret")):
        with pytest.raises(RuntimeError):
            client.complete_with_metadata("prompt", metadata=overrides)
    first, second = client.request_log
    assert [first["request_index"], second["request_index"]] == [1, 2]
    assert first["status"] == "success" and first["error_type"] is None
    assert first["exception_chain"] == [] and first["prompt_tokens"] == 2
    assert second["status"] == "error" and second["prompt_tokens"] is None
    assert second["response_sha256"] is None
    assert all(r["agent_role"] == "Planner" and len(r["prompt_sha256"]) == 64 for r in client.request_log)


def test_exception_type_chain_is_bounded_and_cycle_safe():
    client = http_client()
    errors = [ValueError("secret") for _ in range(20)]
    for index, error in enumerate(errors):
        error.__cause__ = errors[(index + 1) % len(errors)]
    with patch("urllib.request.urlopen", side_effect=errors[0]):
        with pytest.raises(ValueError):
            client.complete("prompt")
    assert client.request_log[0]["exception_chain"] == ["ValueError"] * 8


@pytest.mark.parametrize("setting", ["LLM_MAX_TOKENS", "LLM_TOTAL_REQUEST_TIMEOUT_SECONDS"])
def test_request_setup_failure_still_has_one_audit(setting):
    client = http_client()
    with patch.dict(os.environ, {setting: "invalid"}):
        with pytest.raises(ValueError):
            client.complete("prompt")
    assert client.call_count == len(client.request_log) == 1
    assert client.request_log[0]["prompt_tokens"] is None


def test_response_cleanup_failure_retains_usage():
    class Response(io.BytesIO):
        def __exit__(self, *args):
            raise OSError("secret-body")

    client = http_client()
    response = Response(b'{"usage": {"prompt_tokens": 4, "completion_tokens": 0}}')
    with patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(OSError):
            client.complete("prompt")
    assert client.call_count == len(client.request_log) == 1
    assert client.request_log[0]["prompt_tokens"] == client.prompt_tokens == 4
    assert client.request_log[0]["completion_tokens"] == 0


@pytest.mark.parametrize("suppress", [False, True])
def test_exception_chain_records_types_not_messages_and_honors_suppression(suppress):
    client = http_client()
    error = ValueError("secret-body")
    error.__context__ = KeyError("secret-key")
    error.__context__.__context__ = error
    error.__suppress_context__ = suppress
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ValueError):
            client.complete("prompt")
    assert client.request_log[0]["exception_chain"] == (["ValueError"] if suppress else ["ValueError", "KeyError"])


def test_success_keeps_legacy_missing_usage_defaults_and_core_metadata():
    client = http_client()
    body = b'{"choices": [{"message": {"content": "ok"}}]}'
    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: nullcontext(io.BytesIO(body))):
        client.complete("prompt")
        metadata = {key: "forged" for key in client.request_log[0]}
        assert client.complete_with_metadata("prompt", metadata=metadata) == "ok"
    audit = client.request_log[1]
    assert all(value != "forged" for value in audit.values())
    assert all(audit[key] == 0 for key in TOKEN_FIELDS)


def test_urlerror_reason_exception_is_in_type_chain():
    client = http_client()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError(TimeoutError("secret"))):
        with pytest.raises(RuntimeError):
            client.complete("prompt")
    assert client.request_log[0]["exception_chain"] == ["URLError", "TimeoutError"]


def test_request_log_snapshot_cannot_mutate_stored_exception_chain():
    client = http_client()
    with patch("urllib.request.urlopen", side_effect=TimeoutError("secret")):
        with pytest.raises(RuntimeError):
            client.complete("prompt")
    client.request_log[0]["exception_chain"].append("forged")
    assert client.request_log[0]["exception_chain"] == ["TimeoutError"]


if __name__ == "__main__":
    unittest.main()
