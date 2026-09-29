"""Offline Flash entry tests: real client/scheduler, substituted wire and clock only."""
import asyncio
import io
import json
from datetime import datetime

import pytest

from mas_faults import deepseek_schedule as schedule
from mas_faults import llm_client as llm
from test_multimechanism_runner import OfflineClient, jobs, local_env, runner, tasks, offpeak_clock
from test_multimechanism_parallel_runner import OfflinePool, parallel


def set_time(monkeypatch, hour, minute=0):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, hour, minute, tzinfo=schedule.BEIJING).astimezone(tz)
    monkeypatch.setattr(schedule, "datetime", Clock)


@pytest.mark.parametrize("change", [
    {"LLM_MODEL": "deepseek-v4-flash"}, {"LLM_MODEL": "deepseek-v4-pro"},
    {"LLM_BASE_URL": "http://api.deepseek.com"},
    {"LLM_BASE_URL": "https://api.deepseek.com.evil"},
    {"LLM_BASE_URL": "https://secret@api.deepseek.com"},
    {"LLM_BASE_URL": "https://api.deepseek.com:444"},
    {"LLM_BASE_URL": "https://api.deepseek.com/v1?secret=x"},
    {"LLM_BASE_URL": "https://api.deepseek.com/other"},
    {"LLM_API_KEY": ""}, {"LLM_API_KEY": "  "}, {"LLM_DISABLE_THINKING": "0"},
    {"LLM_MAX_TOKENS": "512"}, {"LLM_REQUEST_TIMEOUT_SECONDS": "60"},
    {"LLM_TOTAL_REQUEST_TIMEOUT_SECONDS": "90"},
])
def test_flash_settings_refuse_wrong_identity_endpoint_key_or_budget(change):
    with pytest.raises(ValueError):
        runner().inference_settings({**local_env(), **change})


def test_flash_settings_keep_confirmed_qwen_budget_and_version():
    settings = runner().inference_settings(local_env())
    assert settings == {"provider": "deepseek", "model": "deepseek-flash",
        "model_version": "DeepSeek-V4.1-Flash", "api_base_url": "https://api.deepseek.com/v1",
        "temperature": 0, "max_tokens": 2048, "disable_thinking": True,
        "socket_timeout_seconds": 90, "total_timeout_seconds": 120}


def flash_client(monkeypatch):
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    return llm.get_llm_client()


def test_flash_wire_budget_raw_response_model_and_metadata_protection(monkeypatch):
    client = flash_client(monkeypatch)
    def send(request, *, timeout):
        assert request.full_url == "https://api.deepseek.com/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-secret"
        assert timeout == 90
        payload = json.loads(request.data)
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["temperature"] == 0 and payload["max_tokens"] == 2048
        return io.BytesIO(b'{"id":"req1","model":"deepseek-flash","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", send)
    assert client.complete_with_metadata("prompt", metadata={"model": "forged",
        "response_model": "forged", "request_sent": False, "provider": "forged",
        "model_version": "forged"}) == "ok"
    trace = client.request_log[0]
    assert trace["model"] == trace["response_model"] == "deepseek-flash"
    assert trace["provider"] == "deepseek" and trace["request_sent"] is True
    assert trace["model_version"] == "DeepSeek-V4.1-Flash"
    assert trace["provider_request_id"] == "req1"
    assert "test-secret" not in json.dumps(trace)
    kept = runner().usage_since(client, {"log_length": 0, "call_count": 0,
        "prompt_tokens": 0, "completion_tokens": 0}, complete=True)["model_requests"][0]
    assert kept["response_model"] == "deepseek-flash"
    assert kept["model_version"] == "DeepSeek-V4.1-Flash"


def test_existing_client_checks_every_request_and_keeps_crossing_audit(monkeypatch):
    client = flash_client(monkeypatch)
    set_time(monkeypatch, 8, 59)
    monkeypatch.setenv("MAS_DEEPSEEK_OFFPEAK_ONLY", "0")
    sends = []
    def send(*args, **kwargs):
        sends.append(1)
        set_time(monkeypatch, 9)
        return io.BytesIO(b'{"id":"req1","model":"deepseek-flash","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", send)
    client.complete("first")
    with pytest.raises(schedule.DeepSeekPeakWindowError):
        client.complete("second")
    assert sends == [1]
    blocked = client.request_log[-1]
    assert blocked["error_type"] == "DeepSeekPeakWindowError"
    assert blocked["request_sent"] is False and blocked["response_model"] is None
    assert blocked["model_version"] == "DeepSeek-V4.1-Flash"


@pytest.mark.parametrize("change", [
    {"model": "deepseek-v4-flash"}, {"model": "deepseek-v4-pro"}, {"model": "deepseek-chat"},
    {"provider": "modelscope_local"}, {"base_url": "http://api.deepseek.com"},
    {"api_key": " "}, {"timeout_seconds": 60},
])
def test_direct_deepseek_client_refuses_unsafe_settings_before_wire(monkeypatch, change):
    client = flash_client(monkeypatch)
    for key, value in change.items():
        setattr(client, key, value)
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unsafe paid send"))
    with pytest.raises(ValueError):
        client.complete("prompt")


def test_direct_flash_client_forbids_redirects_and_proxy_egress(monkeypatch):
    client = flash_client(monkeypatch)
    before = llm.urllib.request._opener
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    def send(*args, **kwargs):
        opener = llm.urllib.request._opener
        assert opener is not None and opener is not before
        assert not any(isinstance(h, llm.urllib.request.ProxyHandler) and h.proxies for h in opener.handlers)
        redirect = next(h for h in opener.handlers if isinstance(h, llm.urllib.request.HTTPRedirectHandler))
        redirect.redirect_request(None, None, 307, "redirect", {}, "https://other.invalid")
        pytest.fail("redirect was not rejected")
    monkeypatch.setattr(llm.urllib.request, "urlopen", send)
    with pytest.raises(ValueError, match="redirect"):
        client.complete("prompt")
    assert llm.urllib.request._opener is before


def test_flash_success_without_usage_remains_cost_unknown(monkeypatch):
    client = flash_client(monkeypatch)
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(
        b'{"model":"deepseek-flash","choices":[{"message":{"content":"ok"}}]}'))
    client.complete("prompt")
    assert client.request_log[0]["prompt_tokens"] is None
    assert client.request_log[0]["completion_tokens"] is None


@pytest.mark.parametrize("response_model", [None, "deepseek-v4-flash", "deepseek-v4-pro", "", 1])
def test_flash_response_model_must_match_exact_raw_id(monkeypatch, response_model):
    client = flash_client(monkeypatch)
    body = {"model": response_model, "id": "received-id", "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(body).encode()))
    with pytest.raises(ValueError, match="model"):
        client.complete("prompt")
    trace = client.request_log[0]
    assert trace["response_model"] == response_model
    assert trace["model"] == "deepseek-flash" and trace["status"] == "error"
    assert trace["provider_request_id"] == "received-id" and trace["prompt_tokens"] == 2


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_peak_job_guard_stops_before_durable_attempt(monkeypatch, tmp_path, mode):
    set_time(monkeypatch, 10)
    js = jobs()[:2]
    if mode == "serial":
        result = asyncio.run(runner().execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
            "http://127.0.0.1:17770", trial=lambda *a, **k: pytest.fail("busy runtime"),
            executor_factory=lambda _: object()))
    else:
        pool = OfflinePool(tmp_path)
        result = parallel().execute_jobs(tasks(), js, tmp_path, "digest", {},
            "http://127.0.0.1:17770", pool=pool)
        assert pool.submitted == []
    assert runner().read_jsonl(tmp_path / "run_attempts.jsonl") == []
    assert result["status"] == "incomplete"
    assert result["scheduler_stop"]["error_type"] == "DeepSeekPeakWindowError"
    assert list((tmp_path / "scheduler_stops").glob("*.json"))


def test_serial_crossing_stops_without_retry_and_preserves_version(monkeypatch, tmp_path):
    set_time(monkeypatch, 8, 59)
    async def trial(*args, **kwargs):
        set_time(monkeypatch, 9)
        raise schedule.DeepSeekPeakWindowError("blocked")
    summary = asyncio.run(runner().execute_jobs(tasks(), jobs()[:2], tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    starts = runner().read_jsonl(tmp_path / "run_attempts.jsonl")
    assert len(starts) == 1 and starts[0]["attempt"] == 1
    assert starts[0]["model_version"] == "DeepSeek-V4.1-Flash"
    assert len(runner().read_jsonl(tmp_path / "run_errors.jsonl")) == 1
    assert summary["scheduler_stop"]["error_type"] == "DeepSeekPeakWindowError"


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_transport_failure_crossing_window_does_not_spend_retry(monkeypatch, tmp_path, mode):
    set_time(monkeypatch, 8, 59)
    js = jobs()[:1]
    if mode == "serial":
        async def trial(*args, **kwargs):
            set_time(monkeypatch, 9)
            raise TimeoutError("offline wire timeout")
        summary = asyncio.run(runner().execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
            "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    else:
        def outcome(spec):
            set_time(monkeypatch, 9)
            return {"kind": "error", "record": {**spec["start"], "error_type": "TimeoutError"}}
        summary = parallel().execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770",
            pool=OfflinePool(tmp_path, outcome))
    starts = runner().read_jsonl(tmp_path / "run_attempts.jsonl")
    assert len(starts) == 1 and starts[0]["attempt"] == 1
    assert summary["scheduler_stop"]["phase"] == "job"


def test_worker_peak_guard_precedes_client_and_environment(monkeypatch, tmp_path):
    set_time(monkeypatch, 10)
    spec = {"start": {**jobs()[0], "attempt": 1, "attempt_id": "attempt",
        "model_version": "DeepSeek-V4.1-Flash"}, "settings": {}}
    def forbidden(*a, **k):
        pytest.fail("busy worker performed external work")
    message = parallel().execute_attempt(spec, client_factory=forbidden, executor_factory=forbidden)
    assert message["kind"] == "error" and message["record"]["offpeak_blocked"] is True
    assert message["record"]["error_type"] == "DeepSeekPeakWindowError"


def test_real_autogen_runtime_propagates_peak_stop_with_partial_trace(monkeypatch, tmp_path):
    client = flash_client(monkeypatch)
    set_time(monkeypatch, 8, 59)
    def send(request, **kwargs):
        set_time(monkeypatch, 9)
        return io.BytesIO(b'{"model":"deepseek-flash","id":"first","usage":{"prompt_tokens":2,"completion_tokens":1},"choices":[{"message":{"content":"{}"}}]}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", send)
    class Environment:
        http_receipts = []
        def reobserve_cart(self, task):
            return {"task_id": task["task_id"], "status": "no executed cart"}
    summary = asyncio.run(runner().execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", client,
        "http://127.0.0.1:17770", executor_factory=lambda _: Environment()))
    assert summary["completed_runs"] == 0
    assert summary["scheduler_stop"]["error_type"] == "DeepSeekPeakWindowError"
    errors = runner().read_jsonl(tmp_path / "run_errors.jsonl")
    assert len(errors) == 1 and errors[0]["offpeak_blocked"] is True
    assert len(errors[0]["partial_trial"]["events"]) == 1
    assert [r["request_sent"] for r in errors[0]["model_requests"]] == [True, False]
    assert errors[0]["known_total_tokens"] == 3
    assert errors[0]["usage_complete"] is False


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_startup_peak_guard_precedes_preflight_and_attempts(monkeypatch, tmp_path, mode):
    r = runner()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    set_time(monkeypatch, 10)
    monkeypatch.setattr(r.sys, "platform", "linux")
    ts = [{**tasks()[0], "task_id": f"t{i}"} for i in range(10)]
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(ts))
    output = tmp_path / "out"
    monkeypatch.setattr(r, "preflight_client", lambda *_: pytest.fail("busy startup reached preflight"))
    main = r.main if mode == "serial" else parallel().main
    assert main(["--task-manifest", str(manifest), "--output", str(output), "--max-jobs", "1"]) == 1
    assert r.read_jsonl(output / "run_attempts.jsonl") == []
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scheduler_stop"]["phase"] == "startup"
    frozen = json.loads((output / "matrix_manifest.json").read_text())["config"]
    assert frozen["model_version"] == "DeepSeek-V4.1-Flash"


def test_parallel_crossing_latches_stop_and_drains_other_lane(monkeypatch, tmp_path):
    set_time(monkeypatch, 8, 59)
    js = [next(j for j in jobs() if parallel().lane_for(j) == lane) for lane in range(2)]
    def outcome(spec):
        if spec["start"]["lane"] == 0:
            set_time(monkeypatch, 9)
            return {"kind": "error", "record": {**spec["start"],
                "error_type": "DeepSeekPeakWindowError", "offpeak_blocked": True}}
        return {"kind": "row", "record": {**spec["start"], "run_id": "other-lane"}}
    pool = OfflinePool(tmp_path, outcome)
    summary = parallel().execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770", pool=pool)
    assert len(pool.submitted) == 2 and pool.live == {}
    assert len(runner().read_jsonl(tmp_path / "run_errors.jsonl")) == 1
    rows = runner().read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(rows) == 1 and rows[0]["model_version"] == "DeepSeek-V4.1-Flash"
    assert summary["scheduler_stop"]["error_type"] == "DeepSeekPeakWindowError"


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("failure", ["timeout", "decode", "wrong_model", "extraction", "peak"])
def test_swallowed_api_error_cannot_become_completed_trial(monkeypatch, tmp_path, mode, failure):
    client = flash_client(monkeypatch)
    set_time(monkeypatch, 8, 59)
    def send(request, **kwargs):
        if failure == "timeout":
            raise TimeoutError("private response")
        if failure == "decode":
            return io.BytesIO(b"invalid json")
        if failure == "extraction":
            return io.BytesIO(b'{"model":"deepseek-flash","id":"received","usage":{"prompt_tokens":5,"completion_tokens":2}}')
        return io.BytesIO(b'{"model":"deepseek-v4-flash","id":"received","usage":{"prompt_tokens":5,"completion_tokens":2},"choices":[{"message":{"content":"ok"}}]}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", send)
    async def swallowing_trial(task, job, executor, received_client, **kwargs):
        if failure == "peak":
            set_time(monkeypatch, 9)
        try:
            received_client.complete("prompt")
        except Exception:
            pass
        # Adversarial trial return, not a replacement for the scientific runtime.
        return {"final_task_success": True, "llm_request_log": [],
                "events": [{"role": "Worker", "output": {"decision": "accept"}}]}
    if mode == "serial":
        summary = asyncio.run(runner().execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", client,
            "http://127.0.0.1:17770", trial=swallowing_trial, executor_factory=lambda _: object()))
        assert runner().read_jsonl(tmp_path / "main_runs.jsonl") == []
        errors = runner().read_jsonl(tmp_path / "run_errors.jsonl")
        assert len(errors) == (1 if failure == "peak" else 2)
        if failure == "peak":
            assert summary["scheduler_stop"]["error_type"] == "DeepSeekPeakWindowError"
        record = errors[0]
    else:
        start = {**jobs()[0], "attempt": 1, "attempt_id": "attempt", "config_digest": "digest",
                 "ledger_path": "action_ledgers/attempt.sqlite3", "model": "deepseek-flash",
                 "provider": "deepseek", "model_version": "DeepSeek-V4.1-Flash"}
        message = parallel().execute_attempt({"start": start, "task": tasks()[0], "settings": {},
            "output": str(tmp_path), "base_url": "http://127.0.0.1:17770"},
            client_factory=lambda _: client, executor_factory=lambda _: object(), trial=swallowing_trial)
        assert message["kind"] == "error"
        record = message["record"]
    assert record["status"] == "infra_error" and record["usage_complete"] is False
    assert record["model_requests"][0]["status"] == "error"
    assert record["known_total_tokens"] == (7 if failure in {"wrong_model", "extraction"} else 0)
    assert record["partial_trial"]["events"][0]["role"] == "Worker"
    assert record["offpeak_blocked"] is (failure == "peak")


def test_earlier_swallowed_error_is_not_hidden_by_later_success(monkeypatch, tmp_path):
    client = flash_client(monkeypatch)
    responses = iter([b'{"model":"deepseek-flash","usage":{"prompt_tokens":5,"completion_tokens":2}}',
        b'{"model":"deepseek-flash","usage":{"prompt_tokens":3,"completion_tokens":1},"choices":[{"message":{"content":"ok"}}]}'])
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    async def trial(task, job, executor, received_client, **kwargs):
        try:
            received_client.complete("first")
        except KeyError:
            pass
        received_client.complete("second")
        return {"final_task_success": True, "llm_request_log": received_client.request_log}
    start = {**jobs()[0], "attempt": 1, "attempt_id": "attempt", "ledger_path": "attempt.sqlite3"}
    message = parallel().execute_attempt({"start": start, "task": tasks()[0], "settings": {},
        "output": str(tmp_path), "base_url": "http://127.0.0.1:17770"},
        client_factory=lambda _: client, executor_factory=lambda _: object(), trial=trial)
    assert message["kind"] == "error"
    assert message["record"]["known_total_tokens"] == 11
    assert [r["status"] for r in message["record"]["model_requests"]] == ["error", "success"]


@pytest.mark.parametrize("change", [
    {"provider": "modelscope_local"}, {"model": "deepseek-v4-flash"},
    {"response_model": None}, {"model_version": None}, {"request_sent": False}, {"status": "error"},
])
def test_completed_boundary_requires_successful_sent_flash_identity(change):
    client = OfflineClient()
    before = runner().usage_snapshot(client)
    client.record()
    client.request_log[-1].update(change)
    with pytest.raises(runner().ModelRequestAuditError):
        runner().completed_usage(client, before, {"final_task_success": True})


def test_completed_boundary_refuses_missing_client_trace():
    client = OfflineClient()
    before = runner().usage_snapshot(client)
    client.record()
    client.request_log.clear()
    with pytest.raises(runner().ModelRequestAuditError):
        runner().completed_usage(client, before, {"final_task_success": True})


@pytest.mark.parametrize("phase", ["job", "request"])
def test_checkpoint_peak_stop_is_nonzero(monkeypatch, tmp_path, phase):
    from test_multimechanism_runner import prepare_main
    from mas_faults import shopping_multimechanism, shopping_action_protocol
    client = flash_client(monkeypatch)
    set_time(monkeypatch, 8, 59)
    r, argv, output, _ = prepare_main(tmp_path, monkeypatch, jobs()[:1])
    def preflight(settings):
        if phase == "job":
            set_time(monkeypatch, 9)
        return client, {}
    monkeypatch.setattr(r, "preflight_client", preflight)
    async def trial(task, job, executor, received_client, **kwargs):
        set_time(monkeypatch, 9)
        received_client.complete("blocked")
        pytest.fail("busy request did not stop")
    monkeypatch.setattr(shopping_multimechanism, "run_trial", trial)
    monkeypatch.setattr(shopping_action_protocol, "MultiStateShoppingExecutor", lambda _: object())
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("busy paid request"))
    assert r.main(argv + ["--max-jobs", "28"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete" and summary["scheduler_stop"]["phase"] == phase
    assert len(r.read_jsonl(output / "run_attempts.jsonl")) == (0 if phase == "job" else 1)


@pytest.mark.parametrize("failure", ["peak", "client_config"])
def test_definite_preclient_failure_has_zero_usage_but_is_not_complete(monkeypatch, failure):
    if failure == "peak":
        set_time(monkeypatch, 10)
    def client_factory(settings):
        if failure == "peak":
            pytest.fail("busy worker constructed a client")
        raise ValueError("invalid local configuration")
    result = parallel().execute_attempt({"start": {"attempt": 1}, "settings": {}},
        client_factory=client_factory, executor_factory=lambda _: pytest.fail("unexpected executor"))
    record = result["record"]
    assert result["kind"] == "error" and record["usage_complete"] is False
    assert record["model_calls"] == 0 and record["model_requests"] == []
    assert record["prompt_tokens"] == record["completion_tokens"] == record["total_tokens"] == 0
    assert record["request_sent"] is False


@pytest.mark.parametrize("setting,value", [("LLM_MAX_TOKENS", "1024"),
    ("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", "90")])
def test_definite_client_setup_rejection_has_zero_tokens(monkeypatch, setting, value):
    client = flash_client(monkeypatch)
    monkeypatch.setenv(setting, value)
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("preflight sent a request"))
    with pytest.raises(ValueError):
        client.complete("prompt")
    record = client.request_log[-1]
    assert record["status"] == "error" and record["request_sent"] is False
    assert record["provider_request_id"] == "" and record["response_model"] is None
    assert record["prompt_tokens"] == record["completion_tokens"] == 0


def test_sent_timeout_cost_remains_unknown(monkeypatch):
    client = flash_client(monkeypatch)
    def fail(*a, **k):
        raise TimeoutError("wire timeout")
    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError):
        client.complete("prompt")
    record = client.request_log[-1]
    assert record["status"] == "error" and record["request_sent"] is True
    assert record["prompt_tokens"] is None and record["completion_tokens"] is None
