"""Offline runner safety tests; doubles replace only real external calls."""
import asyncio
import copy
import hashlib
import importlib
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas_faults import multimechanism_matrix as matrix


def runner():
    return importlib.import_module("run_shopping_multimechanism")


def tasks():
    return [{"task_id": f"t{i}", "product_title": f"Product {i}",
             "product_url": f"http://127.0.0.1:17770/p{i}.html",
             "initial_quantity": 1, "quantity": 2} for i in range(2)]


def local_env():
    return {"LLM_PROVIDER": "modelscope_local", "LLM_MODEL": "Qwen/Qwen3.8-27B",
            "LLM_BASE_URL": "http://127.0.0.1:18001/v1", "LLM_API_KEY": "test-secret",
            "LLM_MAX_TOKENS": "2048", "LLM_DISABLE_THINKING": "true"}


def jobs():
    return matrix.build_jobs(tasks(), 1)


def clean_row(task_index=1, **changes):
    t = tasks()[task_index]
    j = next(j for j in jobs() if j["task_id"] == t["task_id"]
             and j["condition"] == "clean" and j["arm"] == "baseline")
    p = {"task_id": t["task_id"], "product_title": t["product_title"],
         "product_id": str(10 + task_index), "sku": f"SKU{task_index}",
         "requested_quantity": 2, "observed_quantity": 2, "cart_verified": True}
    p["evidence"] = json.dumps(p)
    envelope = {"task_id": t["task_id"], "session_id": "session-hash",
                "entity_id": "cart", "version": 2, "action_id": "action-hash",
                "evidence_id": f"e{task_index}", "source": "Worker", "payload": p}
    return {**j, "config_digest": "digest", "run_id": f"run{task_index}", "task": t,
            "source_evidence": envelope, "environment_state": copy.deepcopy(p),
            "environment_task_success": True, "final_task_success": True,
            "decision_correct": True, **changes}


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_manifest_preserves_multistate_tasks(tmp_path, wrapped):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": tasks()} if wrapped else tasks()))
    assert runner().load_tasks(path) == tasks()


@pytest.mark.parametrize("data", [{}, {"tasks": {}}, [1], [], tasks() * 2])
def test_malformed_manifests_are_rejected(tmp_path, data):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner().load_tasks(path)


@pytest.mark.parametrize("url", ["https://example.com", "http://127.0.0.1.evil/",
    "http://user:secret@localhost", "file:///tmp/x", "http://localhost?key=secret",
    "http://localhost/#secret", "http://0.0.0.0", "http://localhost:99999"])
def test_nonlocal_or_credential_bearing_urls_rejected(url):
    with pytest.raises(ValueError):
        runner().validate_loopback_url(url)


@pytest.mark.parametrize("url", ["http://localhost:17770/", "http://127.0.0.1:18001/v1",
                                 "http://[::1]:17770"])
def test_valid_loopback_endpoints(url):
    assert runner().validate_loopback_url(url) == url.rstrip("/")


@pytest.mark.parametrize("change", [{"LLM_PROVIDER": "deepseek"}, {"LLM_MODEL": "deepseek-chat"},
    {"LLM_BASE_URL": "https://api.deepseek.com"}, {"LLM_PROVIDER": ""},
    {"LLM_REQUEST_TIMEOUT_SECONDS": "0"}, {"LLM_MAX_TOKENS": "-1"}])
def test_inference_fails_closed_before_any_client_call(change):
    with pytest.raises(ValueError):
        runner().inference_settings({**local_env(), **change})


def test_inference_has_no_paid_defaults_or_secrets():
    with pytest.raises(ValueError):
        runner().inference_settings({"DEEPSEEK_API_KEY": "secret"})
    settings = runner().inference_settings(local_env())
    assert settings["max_tokens"] == 2048
    assert settings["disable_thinking"] is True
    assert "secret" not in json.dumps(settings)


def test_cli_defaults_and_smoke_do_not_change_frozen_config(tmp_path, monkeypatch):
    r = runner()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(tasks()))
    argv = ["--task-manifest", str(manifest), "--output-dir", str(tmp_path / "out")]
    args = r.parse_args(argv)
    assert args.repetitions == 3
    config = r.build_config(args, tasks())
    assert config["planned_runs"] == 1260
    assert config["source_hashes"]["run_shopping_multimechanism.py"]
    assert config == r.build_config(r.parse_args(argv + ["--resume", "--max-jobs", "1"]), tasks())
    monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
    assert matrix.config_digest(config) != matrix.config_digest(r.build_config(args, tasks()))


@pytest.mark.parametrize("extra", [["--max-jobs", "0"], ["--shard-count", "0"],
                                  ["--shard-index", "2", "--shard-count", "2"]])
def test_bad_cli_limits_rejected(extra):
    with pytest.raises(SystemExit):
        runner().parse_args(["--task-manifest", "tasks.json", "--output", "out"] + extra)


def test_manifest_is_frozen_under_lock_and_resume_rejects_changed_sources(tmp_path):
    r = runner()
    config = {"source_hashes": {"runtime.py": "old"}, "tasks": tasks()}
    with r.locked_output(tmp_path):
        digest = r.freeze_manifest(tmp_path, config, resume=False)
        assert digest == matrix.config_digest(config)
        assert r.freeze_manifest(tmp_path, config, resume=True) == digest
        with pytest.raises(ValueError):
            r.freeze_manifest(tmp_path, config, resume=False)
        with pytest.raises(ValueError, match="config"):
            r.freeze_manifest(tmp_path, {**config, "source_hashes": {}}, resume=True)


def test_output_and_host_locks_exclude_duplicate_runners(tmp_path):
    r = runner()
    with r.locked_output(tmp_path):
        with pytest.raises(RuntimeError, match="runner"):
            with r.locked_output(tmp_path):
                pytest.fail("output lock was not exclusive")
    path = tmp_path / "host.lock"
    with r.locked_workflow(path):
        with pytest.raises(RuntimeError, match="runner"):
            with r.locked_workflow(path):
                pytest.fail("host lock was not exclusive")
    with r.locked_workflow(path):
        pass


def attempt(job, number=1):
    return {**job, "attempt": number, "attempt_id": f"attempt-{number}", "config_digest": "digest"}


def test_interrupted_attempts_reconciled_once_and_exhaust_after_two(tmp_path):
    r = runner()
    j = jobs()[0]
    r.append_jsonl(tmp_path / "run_attempts.jsonl", attempt(j))
    state = r.reconcile_attempts(tmp_path, [j], "digest")
    assert state["errors"][0]["error_type"] == "InterruptedAttempt"
    assert state["errors"][0]["usage_complete"] is False
    assert state["pending"] == [j]
    assert len(r.reconcile_attempts(tmp_path, [j], "digest")["errors"]) == 1
    r.append_jsonl(tmp_path / "run_attempts.jsonl", attempt(j, 2))
    state = r.reconcile_attempts(tmp_path, [j], "digest")
    assert state["pending"] == []
    assert state["exhausted"] == [j["job_key"]]
    assert len(state["errors"]) == 2


def test_completed_attempt_not_retried_or_reconciled_as_error(tmp_path):
    r = runner()
    j = jobs()[0]
    r.append_jsonl(tmp_path / "run_attempts.jsonl", attempt(j))
    r.append_jsonl(tmp_path / "main_runs.jsonl", {**attempt(j), "run_id": "real-run"})
    state = r.reconcile_attempts(tmp_path, [j], "digest")
    assert state["pending"] == []
    assert state["errors"] == []


@pytest.mark.parametrize("bad", ["wrong_digest", "unknown_job", "duplicate_start", "no_start", "third_attempt"])
def test_resume_rejects_ambiguous_or_incompatible_journals(tmp_path, bad):
    r = runner()
    j = jobs()[0]
    start = attempt(j)
    if bad == "wrong_digest":
        start["config_digest"] = "other"
    if bad == "unknown_job":
        start["job_key"] = "unknown"
    if bad == "third_attempt":
        start["attempt"] = 3
    if bad == "no_start":
        r.append_jsonl(tmp_path / "main_runs.jsonl", start)
    else:
        r.append_jsonl(tmp_path / "run_attempts.jsonl", start)
    if bad == "duplicate_start":
        r.append_jsonl(tmp_path / "run_attempts.jsonl", start)
    with pytest.raises(ValueError):
        r.reconcile_attempts(tmp_path, [j], "digest")


def test_torn_terminal_record_keeps_original_and_consumes_interrupted_attempt(tmp_path):
    r = runner()
    j = jobs()[0]
    r.append_jsonl(tmp_path / "run_attempts.jsonl", attempt(j))
    (tmp_path / "main_runs.jsonl").write_bytes(b'{"job_key":')
    r.recover_torn_journals(tmp_path)
    assert next(tmp_path.glob("main_runs.jsonl.torn-*")).read_bytes() == b'{"job_key":'
    assert len(r.reconcile_attempts(tmp_path, [j], "digest")["errors"]) == 1


def test_middle_journal_corruption_never_silently_dropped(tmp_path):
    (tmp_path / "main_runs.jsonl").write_bytes(b'broken\n{}\n')
    with pytest.raises(ValueError):
        runner().recover_torn_journals(tmp_path)


def cross_job(condition="cross_task_replay", arm="baseline"):
    return next(j for j in jobs() if j["task_id"] == "t0" and j["condition"] == condition
                and j["arm"] == arm)


def test_source_frozen_actual_envelope_identical_across_arms_and_conditions(tmp_path):
    r = runner()
    rows = [clean_row(0), clean_row(1)]
    source = r.freeze_cross_task_source(rows, cross_job(), tasks(), tmp_path, "digest")
    assert source["envelope"] == rows[1]["source_evidence"]
    assert source["source_task_id"] == "t1"
    assert len(source["file_sha256"]) == 64
    again = r.freeze_cross_task_source(rows, cross_job("contract_consistent_identity_corruption", "combined"),
                                      tasks(), tmp_path, "digest")
    assert source == again


@pytest.mark.parametrize("fault", ["missing", "same_task", "same_product", "not_baseline", "not_clean",
                                  "failed_clean", "wrong_binding", "fabricated_payload"])
def test_invalid_sources_block_without_inventing_evidence(tmp_path, fault):
    r = runner()
    row = clean_row(1)
    ts = tasks()
    if fault == "missing":
        row["final_evidence"] = row.pop("source_evidence")
    elif fault == "same_task":
        row = clean_row(0)
    elif fault == "same_product":
        ts[1].update(product_title=ts[0]["product_title"], product_url=ts[0]["product_url"])
    elif fault == "not_baseline":
        row["arm"] = "combined"
    elif fault == "not_clean":
        row["condition"] = "valid_partial"
    elif fault == "failed_clean":
        row["final_task_success"] = False
    elif fault == "wrong_binding":
        row["source_evidence"]["task_id"] = "invented"
    else:
        row["source_evidence"]["payload"]["sku"] = "fabricated"
    with pytest.raises(r.SourceUnavailable):
        r.freeze_cross_task_source([row], cross_job(), ts, tmp_path, "digest")
    assert not list(tmp_path.rglob("*.json"))


def test_tampered_frozen_source_rejected(tmp_path):
    r = runner()
    rows = [clean_row(0), clean_row(1)]
    source = r.freeze_cross_task_source(rows, cross_job(), tasks(), tmp_path, "digest")
    path = tmp_path / source["file"]
    data = json.loads(path.read_text())
    data["envelope"]["payload"]["sku"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source"):
        r.freeze_cross_task_source(rows, cross_job(), tasks(), tmp_path, "digest")


class OfflineClient:
    """Only used in unit tests; no operational simulation option exists."""
    def __init__(self):
        self.model_info = SimpleNamespace(provider="modelscope_local", model="Qwen/Qwen3.8-27B",
            base_url="http://127.0.0.1:18001/v1", client_type="http_openai_compatible")
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        self.request_log = []

    def record(self):
        self.call_count += 1
        self.prompt_tokens += 5
        self.completion_tokens += 2
        self.request_log.append({"request_index": self.call_count, "prompt_tokens": 5,
                                 "completion_tokens": 2, "api_key": "test-secret"})


@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize("usage", [{"prompt_tokens": None, "completion_tokens": 2},
    {"completion_tokens": 2}, {"prompt_tokens": True, "completion_tokens": 2},
    {"prompt_tokens": -1, "completion_tokens": 2}, {"prompt_tokens": "5", "completion_tokens": 2}])
def test_usage_since_unknown_counts_keep_known_subtotals(complete, usage):
    r, client = runner(), OfflineClient()
    client.record()
    before = r.usage_snapshot(client)
    client.call_count += 1
    client.completion_tokens += 2
    client.request_log.append({"request_index": 2, **usage})
    result = r.usage_since(client, before, complete=complete)
    assert result["known_prompt_tokens"] == 0
    assert result["known_completion_tokens"] == result["known_total_tokens"] == 2
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] == 2
    assert result["total_tokens"] is None and result["usage_complete"] is False


def test_failed_attempt_with_known_usage_keeps_numeric_totals_but_not_completeness():
    r, client = runner(), OfflineClient()
    before = r.usage_snapshot(client)
    client.record()
    client.request_log[-1].update(status="error", error_type="KeyError", exception_chain=["KeyError"])
    result = r.usage_since(client, before, complete=False)
    assert result["total_tokens"] == result["known_total_tokens"] == 7
    assert result["usage_complete"] is False
    assert result["model_requests"][0]["status"] == "error"
    assert result["model_requests"][0]["error_type"] == "KeyError"
    assert result["model_requests"][0]["exception_chain"] == ["KeyError"]
    assert "api_key" not in result["model_requests"][0]
    result["model_requests"][0]["exception_chain"].append("mutated")
    assert client.request_log[0]["exception_chain"] == ["KeyError"]


def test_missing_request_usage_does_not_turn_counter_delta_into_complete_total():
    r, client = runner(), OfflineClient()
    before = r.usage_snapshot(client)
    client.record()
    client.request_log.clear()
    result = r.usage_since(client, before, complete=True)
    assert result["known_total_tokens"] == 7
    assert result["total_tokens"] is None and result["usage_complete"] is False


def test_completed_nullable_usage_persists_and_reports_without_retries(tmp_path):
    r, client = runner(), OfflineClient()
    j = jobs()[0]

    async def trial(task, job, executor, received_client, **kwargs):
        received_client.call_count += 1
        received_client.completion_tokens += 2
        received_client.request_log.append({"request_index": 1, "prompt_tokens": None,
                                            "completion_tokens": 2, "status": "success"})
        return {"run_id": "nullable-run", "final_task_success": True}

    summary = asyncio.run(r.execute_jobs(tasks(), [j], tmp_path, "digest", client,
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    rows = r.read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(rows) == 1 and rows[0]["total_tokens"] is None
    assert rows[0]["known_total_tokens"] == 2
    assert rows[0]["usage_complete"] is False
    assert r.read_jsonl(tmp_path / "run_errors.jsonl") == []
    assert summary["status"] == "complete"
    assert summary["total_tokens_completed"] is None
    assert summary["known_tokens_completed"] == 2
    assert summary["by_arm"][j["arm"]]["total_tokens"] is None
    assert summary["by_arm"][j["arm"]]["known_total_tokens"] == 2


def test_attempt_is_durable_before_external_work_and_usage_survives_errors(tmp_path):
    r = runner()
    j = jobs()[0]
    client = OfflineClient()

    def executor_factory(base_url):
        starts = r.read_jsonl(tmp_path / "run_attempts.jsonl")
        assert len(starts) >= 1
        return object()

    async def trial(task, job, executor, received_client, *, cross_task_evidence=None, ledger_path):
        assert isinstance(ledger_path, Path)
        starts = r.read_jsonl(tmp_path / "run_attempts.jsonl")
        assert starts[-1]["attempt"] == job["attempt"]
        received_client.record()
        raise RuntimeError("Bearer test-secret guest-cart-token")

    asyncio.run(r.execute_jobs(tasks(), [j], tmp_path, "digest", client,
                              "http://127.0.0.1:17770", trial=trial, executor_factory=executor_factory))
    starts = r.read_jsonl(tmp_path / "run_attempts.jsonl")
    errors = r.read_jsonl(tmp_path / "run_errors.jsonl")
    assert [s["attempt"] for s in starts] == [1, 2]
    assert len({s["ledger_path"] for s in starts}) == 2
    assert [e["known_total_tokens"] for e in errors] == [7, 7]
    assert all(len(e["model_requests"]) == 1 for e in errors)
    assert all(e["usage_complete"] is False for e in errors)
    assert "test-secret" not in json.dumps(errors)
    assert "guest-cart-token" not in json.dumps(errors)
    asyncio.run(r.execute_jobs(tasks(), [j], tmp_path, "digest", client,
                              "http://127.0.0.1:17770", trial=trial, executor_factory=executor_factory))
    assert len(r.read_jsonl(tmp_path / "run_attempts.jsonl")) == 2


def test_success_usage_and_smoke_limit_persist_without_marking_matrix_complete(tmp_path):
    r = runner()
    client = OfflineClient()
    js = jobs()[:2]

    async def trial(task, job, executor, received_client, **kwargs):
        received_client.record()
        return {"run_id": "runtime-run", "final_task_success": True,
                "environment_task_success": True, "decision_correct": True}

    asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", client,
        "http://127.0.0.1:17770", max_jobs=1, trial=trial, executor_factory=lambda _: object()))
    rows = r.read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(rows) == 1
    assert rows[0]["total_tokens"] == 7
    assert rows[0]["config_digest"] == "digest"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["planned_runs"] == 2
    assert summary["completed_runs"] == 1
    assert (tmp_path / "main_runs.csv").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_missing_source_blocks_cell_without_spending_attempts(tmp_path):
    r = runner()
    js = [cross_job(arm=arm) for arm in matrix.ARMS]

    async def forbidden(*args, **kwargs):
        pytest.fail("missing source reached runtime")

    asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=forbidden, executor_factory=lambda _: object()))
    assert r.read_jsonl(tmp_path / "run_attempts.jsonl") == []
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["source_blocked_job_keys"]) == 7
    assert summary["completed_runs"] == 0
    assert summary["status"] == "incomplete"


def test_preflight_checks_real_model_listing_without_completion_calls(monkeypatch):
    r = runner()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    client = OfflineClient()
    monkeypatch.setattr(r, "get_llm_client", lambda: client)

    def models(url, *, timeout):
        assert url == "http://127.0.0.1:18001/v1/models"
        assert timeout == 10
        return io.BytesIO(json.dumps({"data": [{"id": "Qwen/Qwen3.8-27B"}]}).encode())

    monkeypatch.setattr(r.urllib.request, "urlopen", models)
    actual, receipt = r.preflight_client(r.inference_settings())
    assert actual is client
    assert receipt["model_calls"] == 0
    assert len(receipt["served_models_sha256"]) == 64
    assert "secret" not in json.dumps(receipt)


def test_preflight_rejects_changed_env_before_client_creation(monkeypatch):
    r = runner()
    settings = r.inference_settings(local_env())
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(r, "get_llm_client", lambda: pytest.fail("unsafe client constructed"))
    with pytest.raises(ValueError):
        r.preflight_client(settings)


def test_preflight_rejects_missing_model(monkeypatch):
    r = runner()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(r, "get_llm_client", OfflineClient)
    monkeypatch.setattr(r.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b'{"data":[]}'))
    with pytest.raises(ValueError, match="absent"):
        r.preflight_client(r.inference_settings())


def test_local_transport_forbids_redirects_and_restores_opener():
    r = runner()
    previous = r.urllib.request._opener
    with r.local_inference_transport():
        with pytest.raises(ValueError, match="redirect"):
            r._NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://api.deepseek.com")
        assert not any(isinstance(h, r.urllib.request.ProxyHandler)
                       and h.proxies for h in r.urllib.request._opener.handlers)
    assert r.urllib.request._opener is previous


def test_failure_keeps_available_sanitized_http_receipts(tmp_path):
    r = runner()
    receipt = {"receipt_index": 0, "request_method": "POST", "status_code": 200,
               "request_url": "http://127.0.0.1:17770/rest/guest-carts/[redacted]/items"}
    executor = SimpleNamespace(http_receipts=[receipt], guest_cart_id="must-not-be-persisted")

    async def trial(*args, **kwargs):
        raise RuntimeError("raw-error-secret")

    asyncio.run(r.execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: executor))
    errors = r.read_jsonl(tmp_path / "run_errors.jsonl")
    assert errors[0]["http_receipts"] == [receipt]
    assert "must-not-be-persisted" not in json.dumps(errors)
    assert "raw-error-secret" not in json.dumps(errors)


def test_frozen_envelope_passed_to_runtime_not_wrapper_metadata(tmp_path):
    r = runner()
    clean = clean_row(1)
    j = cross_job()
    clean_job = next(job for job in jobs() if job["job_key"] == clean["job_key"])
    start = attempt(clean_job)
    r.append_jsonl(tmp_path / "run_attempts.jsonl", start)
    r.append_jsonl(tmp_path / "main_runs.jsonl", {**clean, **start})

    async def trial(task, job, executor, client, *, cross_task_evidence, ledger_path):
        assert cross_task_evidence == clean["source_evidence"]
        cross_task_evidence["payload"]["sku"] = "runtime-mutated-copy"
        return {"run_id": "fault-run", "final_task_success": False}

    asyncio.run(r.execute_jobs(tasks(), [clean_job, j], tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    rows = r.read_jsonl(tmp_path / "main_runs.jsonl")
    assert rows[-1]["cross_task_source"]["envelope"]["payload"]["sku"] == "SKU1"
    assert rows[-1]["final_task_success"] is False
    assert r.reconcile_attempts(tmp_path, [clean_job, j], "digest")["pending"] == []


def test_null_target_clean_evidence_does_not_crash_source_selection(tmp_path):
    rows = [clean_row(0, source_evidence=None, final_task_success=False), clean_row(1)]
    source = runner().freeze_cross_task_source(rows, cross_job(), tasks(), tmp_path, "digest")
    assert source["source_task_id"] == "t1"


def test_manifest_changed_between_load_and_freeze_is_rejected(tmp_path, monkeypatch):
    r = runner()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks()))
    loaded = r.load_tasks(path)
    changed = tasks()
    changed[0]["quantity"] = 3
    path.write_text(json.dumps(changed))
    args = r.parse_args(["--task-manifest", str(path), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="manifest"):
        r.build_config(args, loaded)


def test_duplicate_completed_jobs_rejected_before_journal_mutation(tmp_path):
    r = runner()
    j, interrupted = jobs()[:2]
    for n in (1, 2):
        r.append_jsonl(tmp_path / "run_attempts.jsonl", attempt(j, n))
        r.append_jsonl(tmp_path / "main_runs.jsonl", {**attempt(j, n), "run_id": f"run{n}"})
    r.append_jsonl(tmp_path / "run_attempts.jsonl", {**attempt(interrupted), "attempt_id": "other"})
    with pytest.raises(ValueError, match="duplicate"):
        r.reconcile_attempts(tmp_path, [j, interrupted], "digest")
    assert r.read_jsonl(tmp_path / "run_errors.jsonl") == []


def test_circuit_breaker_stops_third_error_across_jobs_and_survives_resume(tmp_path):
    r = runner()
    js = jobs()[:4]

    async def broken(*args, **kwargs):
        raise RuntimeError("systematic-bug")

    kwargs = {"trial": broken, "executor_factory": lambda _: object()}
    summary = asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
                                        "http://127.0.0.1:17770", **kwargs))
    starts = r.read_jsonl(tmp_path / "run_attempts.jsonl")
    assert [(s["job_key"], s["attempt"]) for s in starts] == [
        (js[0]["job_key"], 1), (js[0]["job_key"], 2), (js[1]["job_key"], 1)]
    assert summary["circuit_breaker"] == {"threshold": 3, "consecutive_error_attempts": 3, "tripped": True}
    assert summary["status"] == "incomplete"
    assert summary["exhausted_job_keys"] == [js[0]["job_key"]]
    asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
                              "http://127.0.0.1:17770", **kwargs))
    assert len(r.read_jsonl(tmp_path / "run_attempts.jsonl")) == 3


def test_completed_scientific_failure_resets_error_streak(tmp_path):
    r = runner()
    js = jobs()[:4]

    async def trial(task, job, executor, client, **kwargs):
        if job["job_key"] in {js[0]["job_key"], js[2]["job_key"]}:
            raise RuntimeError("error")
        return {"run_id": job["attempt_id"], "final_task_success": False}

    summary = asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    assert summary["completed_runs"] == 2
    assert summary["error_attempts"] == 4
    assert summary["circuit_breaker"]["tripped"] is False
    assert summary["circuit_breaker"]["consecutive_error_attempts"] == 0


def test_interrupted_attempts_count_towards_cross_job_breaker(tmp_path):
    r = runner()
    js = jobs()[:2]
    for j, n in ((js[0], 1), (js[0], 2), (js[1], 1)):
        r.append_jsonl(tmp_path / "run_attempts.jsonl", {**attempt(j, n), "attempt_id": f"{j['arm']}-{n}"})
    state = r.reconcile_attempts(tmp_path, js, "digest")
    assert state["consecutive_error_attempts"] == 3
    assert len(state["pending"]) == 1


def prepare_main(tmp_path, monkeypatch, js):
    r = runner()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(tasks()))
    output = tmp_path / "out"
    argv = ["--task-manifest", str(manifest), "--output-dir", str(output)]
    config = r.build_config(r.parse_args(argv), tasks())
    config.update(jobs=js, planned_runs=len(js), shard_runs=len(js))
    monkeypatch.setattr(r, "build_config", lambda *args: config)
    monkeypatch.setattr(r.sys, "platform", "linux")
    original_lock = r.locked_workflow
    monkeypatch.setattr(r, "locked_workflow", lambda: original_lock(tmp_path / "host.lock"))
    return r, argv, output, config


@pytest.mark.parametrize("completed,expected_code", [(True, 0), (False, 1)])
def test_main_no_pending_is_success_only_for_completed_matrix(tmp_path, monkeypatch, completed, expected_code):
    js = jobs()[:1]
    r, argv, output, config = prepare_main(tmp_path, monkeypatch, js)
    with r.locked_output(output):
        digest = r.freeze_manifest(output, config, resume=False)
        r.freeze_source_snapshot(output, config["source_hashes"], root=Path(r.__file__).parent, resume=False)
    for n in range(1, 2 if completed else 3):
        start = {**attempt(js[0], n), "config_digest": digest}
        r.append_jsonl(output / "run_attempts.jsonl", start)
        filename = "main_runs.jsonl" if completed else "run_errors.jsonl"
        r.append_jsonl(output / filename, {**start, "run_id": f"run{n}"})
    monkeypatch.setattr(r, "preflight_client", lambda *_: pytest.fail("no pending work reached preflight"))
    assert r.main(argv + ["--resume"]) == expected_code
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == ("complete" if completed else "incomplete")


@pytest.mark.parametrize("kind", ["source_blocked", "breaker", "exhausted"])
def test_main_returns_nonzero_after_blocked_or_error_run(tmp_path, monkeypatch, kind):
    js = [cross_job()] if kind == "source_blocked" else jobs()[:3 if kind == "breaker" else 1]
    r, argv, output, _ = prepare_main(tmp_path, monkeypatch, js)
    monkeypatch.setattr(r, "preflight_client", lambda *_: (OfflineClient(), {}))
    original_execute = r.execute_jobs

    async def broken(*args, **kwargs):
        raise RuntimeError("failure")

    async def offline_execute(*args, **kwargs):
        return await original_execute(*args, **kwargs, trial=broken, executor_factory=lambda _: object())

    monkeypatch.setattr(r, "execute_jobs", offline_execute)
    assert r.main(argv) != 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    if kind == "breaker":
        monkeypatch.setattr(r, "preflight_client", lambda *_: pytest.fail("tripped breaker reached preflight"))
        assert r.main(argv + ["--resume"]) != 0


def test_summary_distinguishes_exposure_from_plan_and_prior_contract_failure(tmp_path):
    r = runner()
    j = next(j for j in jobs() if j["condition"] == "request_non_delivery")
    event = {"condition": j["condition"], "boundary": j["boundary"], "original_sha256": "a" * 64,
             "delivered_count": 0, "delivered_sha256": []}
    variants = [
        {"fault_events": [event], "action_contract_valid": True},
        {"fault_events": [], "action_contract_valid": True},
        {"fault_events": [event], "action_contract_valid": False},
        {"fault_events": [{"condition": j["condition"]}], "action_contract_valid": True},
    ]
    js = [{**j, "job_key": f"fault-{i}", "pair_key": f"pair-{i}"} for i in range(4)] + [jobs()[0]]
    rows = [{**job, **variant, "observed_A_symptom": [j["condition"]]}
            for job, variant in zip(js, variants)] + [{**js[-1], "fault_events": []}]
    original = copy.deepcopy(rows)
    state = {"rows": rows, "errors": [], "exhausted": [], "consecutive_error_attempts": 0}
    summary = r.write_reports(tmp_path, js, state)
    assert summary["exposure"] == {"clean_runs": 1, "planned_fault_runs": 4,
        "injection_recorded_runs": 2, "exposed_fault_runs": 1, "unreached_fault_runs": 1,
        "prior_contract_failure_runs": 1, "unknown_fault_exposure_runs": 1,
        "error_attempts_exposure_unknown": 0}
    assert rows == original
    assert "暴露" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("invalid_source", ["partial", "self_consistent_wrong_identity", "readback"])
def test_recovered_clean_does_not_qualify_invalid_current_worker_source(tmp_path, invalid_source):
    r = runner()
    row = clean_row(1)
    row["final_evidence"] = copy.deepcopy(row["environment_state"])
    row["recovery_detected"] = True
    if invalid_source == "partial":
        row["source_evidence"]["payload"] = {"task_id": "t1", "cart_verified": True}
    elif invalid_source == "self_consistent_wrong_identity":
        p = row["source_evidence"]["payload"]
        p.update(sku="wrong", product_id="wrong")
        p["evidence"] = json.dumps({k: v for k, v in p.items() if k != "evidence"})
    else:
        row["source_evidence"]["source"] = "EnvironmentReadback"
    with pytest.raises(r.SourceUnavailable):
        r.freeze_cross_task_source([row], cross_job(), tasks(), tmp_path, "digest")


def test_two_error_streak_carries_across_smoke_invocations(tmp_path):
    r = runner()
    js = jobs()[:3]

    async def broken(*args, **kwargs):
        raise RuntimeError("error")

    kwargs = {"trial": broken, "executor_factory": lambda _: object(), "max_jobs": 1}
    first = asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
                                      "http://127.0.0.1:17770", **kwargs))
    assert first["circuit_breaker"]["consecutive_error_attempts"] == 2
    assert first["circuit_breaker"]["tripped"] is False
    second = asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
                                       "http://127.0.0.1:17770", **kwargs))
    assert second["circuit_breaker"]["tripped"] is True
    assert len(r.read_jsonl(tmp_path / "run_attempts.jsonl")) == 3


def test_main_successful_explicit_smoke_exits_zero_but_reports_incomplete(tmp_path, monkeypatch):
    r, argv, output, config = prepare_main(tmp_path, monkeypatch, jobs()[:2])
    assert config["max_consecutive_error_attempts"] == 3
    monkeypatch.setattr(r, "preflight_client", lambda *_: (OfflineClient(), {}))
    original_execute = r.execute_jobs

    async def trial(task, job, executor, client, **kwargs):
        return {"run_id": job["attempt_id"], "final_task_success": True}

    async def offline_execute(*args, **kwargs):
        return await original_execute(*args, **kwargs, trial=trial, executor_factory=lambda _: object())

    monkeypatch.setattr(r, "execute_jobs", offline_execute)
    assert r.main(argv + ["--max-jobs", "1"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["completed_runs"] == 1


def test_model_timeout_preserves_allowlisted_partial_trial_without_exception_text(tmp_path, capsys):
    r = runner()
    context = {"events": [{"role": "Worker", "input": {"stage": "current"}}],
               "judgments": [{"judgment_id": "j1", "verdict": {"decision": "reject"}}],
               "detection_events": [{"kind": "contract_check"}],
               "recovery_events": [{"kind": "common_recovery", "receipt_indices": [0]}],
               "budget": {"get": 2, "model_call": 1},
               "graph_errors": [{"error_type": "invalid_dependency"}]}

    async def trial(task, job, executor, client, **kwargs):
        client.record()
        error = TimeoutError("raw-exception-secret")
        error.partial_trial = {**context, "api_key": "unexpected-secret",
                               "error_message": "raw-exception-secret", "executor": executor}
        error.other_context = "arbitrary-secret"
        raise error

    asyncio.run(r.execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    errors = r.read_jsonl(tmp_path / "run_errors.jsonl")
    assert len(errors) == 2
    assert all(e["partial_trial"] == context for e in errors)
    assert all(e["status"] == "timeout" and e["known_total_tokens"] == 7 for e in errors)
    context["events"].append({"role": "later-mutation"})
    assert len(errors[0]["partial_trial"]["events"]) == 1
    captured = capsys.readouterr()
    for secret in ("raw-exception-secret", "unexpected-secret", "arbitrary-secret"):
        assert secret not in json.dumps(errors) + captured.out + captured.err


@pytest.mark.parametrize("partial", [None, "raw-exception-secret", ["not-an-object"],
    {"events": [object()], "budget": {"get": float("nan")}, "graph_errors": []}])
def test_malformed_partial_context_cannot_prevent_error_journaling(tmp_path, partial):
    r = runner()

    async def trial(*args, **kwargs):
        error = RuntimeError("raw-exception-secret")
        error.partial_trial = partial
        raise error

    asyncio.run(r.execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    errors = r.read_jsonl(tmp_path / "run_errors.jsonl")
    assert len(errors) == 2
    expected = {"graph_errors": []} if isinstance(partial, dict) else {}
    assert all(e["partial_trial"] == expected for e in errors)
    assert "raw-exception-secret" not in json.dumps(errors)


def test_postwrite_unknown_completed_row_is_not_retried_on_resume(tmp_path):
    r = runner()
    js = jobs()[:1]

    async def trial(task, job, executor, client, **kwargs):
        return {"run_id": job["attempt_id"], "action_outcome_unknown": True,
                "final_task_success": False, "environment_task_success": False,
                "events": [{"role": "ActionExecutor", "output": []}]}

    kwargs = {"trial": trial, "executor_factory": lambda _: object()}
    for _ in range(2):
        asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
                                  "http://127.0.0.1:17770", **kwargs))
    assert len(r.read_jsonl(tmp_path / "run_attempts.jsonl")) == 1
    assert r.read_jsonl(tmp_path / "run_errors.jsonl") == []
    rows = r.read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(rows) == 1 and rows[0]["action_outcome_unknown"] is True


def snapshot_fixture(tmp_path):
    root, output = tmp_path / "root", tmp_path / "out"
    (root / "src" / "package").mkdir(parents=True)
    output.mkdir()
    contents = {"runner.py": b'print("frozen")\r\n', "src/package/core.py": b"VALUE = 1\n"}
    for name, content in contents.items():
        (root / name).write_bytes(content)
    (root / "qwen-local-env.sh").write_text("private-env")
    (root / "logs.jsonl").write_text("private-log")
    (root / "unlisted.py").write_text("unlisted")
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    return root, output, contents, hashes


def test_source_snapshot_preserves_exact_listed_python_bytes_and_resume_never_writes(tmp_path):
    r = runner()
    root, output, contents, hashes = snapshot_fixture(tmp_path)
    r.freeze_source_snapshot(output, hashes, root=root, resume=False)
    snapshot = output / "source_snapshot"
    files = {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob("*") if p.is_file()}
    assert files == contents
    before = {name: (snapshot / name).stat().st_mtime_ns for name in contents}
    r.freeze_source_snapshot(output, hashes, root=root, resume=True)
    assert before == {name: (snapshot / name).stat().st_mtime_ns for name in contents}
    with pytest.raises(ValueError):
        r.freeze_source_snapshot(output, hashes, root=root, resume=False)


@pytest.mark.parametrize("damage", ["missing_original", "changed_original", "missing_snapshot",
    "changed_snapshot", "missing_directory", "unlisted_file"])
def test_missing_or_tampered_source_snapshot_fails_closed(tmp_path, damage):
    r = runner()
    root, output, contents, hashes = snapshot_fixture(tmp_path)
    resume = damage not in {"missing_original", "changed_original", "missing_directory"}
    if resume:
        r.freeze_source_snapshot(output, hashes, root=root, resume=False)
    source, frozen = root / "runner.py", output / "source_snapshot" / "runner.py"
    if damage == "missing_original":
        source.unlink()
    elif damage == "changed_original":
        source.write_text("changed")
    elif damage == "missing_snapshot":
        frozen.unlink()
    elif damage == "changed_snapshot":
        frozen.write_text("tampered")
    elif damage == "unlisted_file":
        (output / "source_snapshot" / "logs.jsonl").write_text("private-log")
    elif damage == "missing_directory":
        resume = True
    with pytest.raises(ValueError, match="source|snapshot"):
        r.freeze_source_snapshot(output, hashes, root=root, resume=resume)
    if damage == "missing_snapshot":
        assert not frozen.exists()
    elif damage == "changed_snapshot":
        assert frozen.read_text() == "tampered"
    elif not resume or damage == "missing_directory":
        assert not (output / "source_snapshot").exists()


@pytest.mark.parametrize("name", ["../outside.py", "/tmp/outside.py", "src/../../outside.py",
    "src//package/core.py", "./runner.py", "src/../runner.py", "qwen-local-env.sh", "logs.jsonl",
    "src\\outside.py"])
def test_snapshot_manifest_paths_are_relative_contained_python_only(tmp_path, name):
    root, output, _, hashes = snapshot_fixture(tmp_path)
    with pytest.raises(ValueError):
        runner().freeze_source_snapshot(output, {name: next(iter(hashes.values()))}, root=root, resume=False)
    assert not (output / "source_snapshot").exists()


@pytest.mark.parametrize("kind", ["source_file", "source_directory", "snapshot_root", "snapshot_directory", "snapshot_file"])
def test_snapshot_refuses_source_and_destination_symlinks(tmp_path, kind):
    r = runner()
    root, output, contents, hashes = snapshot_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "runner.py").write_bytes(contents["runner.py"])
    snapshot = output / "source_snapshot"
    resume = kind.startswith("snapshot")
    if kind == "source_file":
        (root / "runner.py").unlink()
        (root / "runner.py").symlink_to(outside / "runner.py")
    elif kind == "source_directory":
        (root / "linked").symlink_to(outside, target_is_directory=True)
        hashes = {"linked/runner.py": hashes["runner.py"]}
    elif kind == "snapshot_root":
        snapshot.symlink_to(outside, target_is_directory=True)
    else:
        r.freeze_source_snapshot(output, hashes, root=root, resume=False)
        if kind == "snapshot_file":
            (snapshot / "runner.py").unlink()
            (snapshot / "runner.py").symlink_to(outside / "runner.py")
        else:
            (snapshot / "src" / "package" / "core.py").unlink()
            (snapshot / "src" / "package").rmdir()
            (snapshot / "src" / "package").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        r.freeze_source_snapshot(output, hashes, root=root, resume=resume)
    assert (outside / "runner.py").read_bytes() == contents["runner.py"]
    assert not (outside / "core.py").exists()


def test_main_freezes_source_bytes_before_preflight_and_refuses_missing_copy_on_resume(tmp_path, monkeypatch):
    r, argv, output, config = prepare_main(tmp_path, monkeypatch, jobs()[:1])

    def preflight(settings):
        snapshot = output / "source_snapshot"
        assert (output / "matrix_manifest.json").is_file()
        assert snapshot.is_dir()
        assert {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob("*.py")} == config["source_hashes"]
        raise RuntimeError("end offline preflight")

    monkeypatch.setattr(r, "preflight_client", preflight)
    assert r.main(argv) == 1
    (output / "source_snapshot" / "run_shopping_multimechanism.py").unlink()
    monkeypatch.setattr(r, "preflight_client", lambda *_: pytest.fail("missing snapshot reached preflight"))
    assert r.main(argv + ["--resume"]) == 1
    assert not (output / "source_snapshot" / "run_shopping_multimechanism.py").exists()


def test_snapshot_verifies_written_bytes_and_never_repairs_corrupt_copy(tmp_path, monkeypatch):
    r = runner()
    root, output, _, hashes = snapshot_fixture(tmp_path)
    frozen = output / "source_snapshot" / "runner.py"
    original_sync = r._sync_directory
    corrupted = False

    def corrupt_after_write(path):
        nonlocal corrupted
        original_sync(path)
        if Path(path) == frozen.parent and frozen.exists() and not corrupted:
            frozen.write_bytes(b"corrupt-copy")
            corrupted = True

    monkeypatch.setattr(r, "_sync_directory", corrupt_after_write)
    with pytest.raises(ValueError, match="hash"):
        r.freeze_source_snapshot(output, hashes, root=root, resume=False)
    assert corrupted and frozen.read_bytes() == b"corrupt-copy"
    with pytest.raises(ValueError):
        r.freeze_source_snapshot(output, hashes, root=root, resume=True)
    assert frozen.read_bytes() == b"corrupt-copy"


def test_partial_action_accounting_persists_on_resume_without_overriding_http(tmp_path):
    r = runner()
    context = {"action_protocol_events": [{"kind": "unknown", "action_id": "a"}],
               "action_ledger_state": {"state": "unknown", "execution_count": 1},
               "action_ledger_events": [{"event": "execute_entered"}]}
    receipts = [{"receipt_index": 0, "request_method": "POST", "status_code": 200}]

    async def trial(*args, **kwargs):
        error = RuntimeError("not-for-logs")
        error.partial_trial = {**context, "http_receipts": [{"secret": "do-not-copy"}]}
        raise error

    js = jobs()[:1]
    asyncio.run(r.execute_jobs(tasks(), js, tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial,
        executor_factory=lambda _: SimpleNamespace(http_receipts=receipts)))
    state = r.reconcile_attempts(tmp_path, js, "digest")
    assert len(state["errors"]) == 2
    assert all(e["partial_trial"] == context and e["http_receipts"] == receipts for e in state["errors"])
    assert "do-not-copy" not in json.dumps(state["errors"])


@pytest.mark.parametrize("context,expected", [
    ({"action_ledger_state": None}, {"action_ledger_state": None}),
    ({"action_protocol_events": {}, "action_ledger_state": [], "action_ledger_events": "secret"}, {}),
    ({}, {}),
])
def test_partial_action_fields_are_typed_and_preserve_explicit_absent_state(context, expected):
    error = RuntimeError("secret")
    error.partial_trial = context
    assert runner().partial_trial_context(error) == expected


@pytest.mark.parametrize("audit_exists", [False, True])
def test_config_explicitly_freezes_existing_audit_script(tmp_path, monkeypatch, audit_exists):
    r = runner()
    root, output, _, _ = snapshot_fixture(tmp_path)
    script = root / "scripts" / "audit_shopping_multimechanism.py"
    script.parent.mkdir()
    if audit_exists:
        script.write_bytes(b"AUDIT_VERSION = 1\n")
    (script.parent / "unrelated.py").write_bytes(b"UNLISTED = True\n")
    monkeypatch.setattr(r, "__file__", str(root / "runner.py"))
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(tasks()))
    args = r.parse_args(["--task-manifest", str(manifest), "--output-dir", str(output)])
    config = r.build_config(args, tasks())
    name = "scripts/audit_shopping_multimechanism.py"
    assert (name in config["source_hashes"]) is audit_exists
    assert "scripts/unrelated.py" not in config["source_hashes"]
    r.freeze_source_snapshot(output, config["source_hashes"], root=root, resume=False)
    if audit_exists:
        assert (output / "source_snapshot" / name).read_bytes() == b"AUDIT_VERSION = 1\n"
        script.write_bytes(b"AUDIT_VERSION = 2\n")
        assert matrix.config_digest(r.build_config(args, tasks())) != matrix.config_digest(config)
        with pytest.raises(ValueError, match="hash"):
            r.freeze_source_snapshot(output, config["source_hashes"], root=root, resume=True)


def test_timeout_chain_handles_cause_context_urlreason_cycles_and_depth_without_messages():
    r = runner()

    class MessageMustNotBeRead(RuntimeError):
        def __str__(self):
            pytest.fail("exception message was read")

    caused = MessageMustNotBeRead()
    caused.__cause__ = TimeoutError()
    contextual = MessageMustNotBeRead()
    contextual.__context__ = TimeoutError()
    suppressed = MessageMustNotBeRead()
    suppressed.__context__ = TimeoutError()
    suppressed.__suppress_context__ = True
    cyclic = MessageMustNotBeRead()
    cyclic.__cause__ = cyclic
    deep = TimeoutError()
    for _ in range(64):
        outer = MessageMustNotBeRead()
        outer.__cause__ = deep
        deep = outer
    assert r.is_timeout_exception(caused)
    assert r.is_timeout_exception(contextual)
    assert r.is_timeout_exception(urllib.error.URLError(TimeoutError()))
    assert not r.is_timeout_exception(urllib.error.URLError("timed out"))
    assert not r.is_timeout_exception(suppressed)
    assert not r.is_timeout_exception(cyclic)
    assert not r.is_timeout_exception(deep)


def test_wrapped_model_timeout_is_journaled_as_timeout_not_infrastructure_error(tmp_path, capsys):
    r = runner()

    async def trial(*args, **kwargs):
        raise RuntimeError("outer-secret") from TimeoutError("inner-secret")

    asyncio.run(r.execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", OfflineClient(),
        "http://127.0.0.1:17770", trial=trial, executor_factory=lambda _: object()))
    errors = r.read_jsonl(tmp_path / "run_errors.jsonl")
    assert all(e["status"] == "timeout" and e["error_type"] == "RuntimeError" for e in errors)
    captured = capsys.readouterr()
    assert "secret" not in json.dumps(errors) + captured.out + captured.err
