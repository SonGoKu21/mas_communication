"""Offline unit tests only: no service launches, real inference, or performance proof."""
import hashlib
import importlib
import io
import json
import urllib.request
import urllib.response
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    name = "scripts.probe_local_inference_config"
    assert importlib.util.find_spec(name), "inference config probe is not implemented"
    return importlib.import_module(name)


@pytest.fixture
def settings(probe, monkeypatch):
    for key, value in {"LLM_PROVIDER": "modelscope_local", "LLM_MODEL": "Qwen/Qwen3.8-27B",
                       "LLM_BASE_URL": "http://127.0.0.1:18001/v1", "LLM_API_KEY": "unit-secret",
                       "LLM_MAX_TOKENS": "128", "LLM_DISABLE_THINKING": "1",
                       "LLM_REQUEST_TIMEOUT_SECONDS": "2", "LLM_TOTAL_REQUEST_TIMEOUT_SECONDS": "2"}.items():
        monkeypatch.setenv(key, value)
    return probe.checked_settings()


@pytest.fixture
def service(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    artifacts = {"config.json": {}, "tokenizer_config.json": {}, "tokenizer.json": {},
                 "generation_config.json": {}, "chat_template.jinja": "unit template",
                 "model.safetensors.index.json": {"weight_map": {"unit": "model-00001.safetensors"}},
                 "model-00001.safetensors": "unit weight bytes"}
    files = []
    for name, value in artifacts.items():
        data = (value if isinstance(value, str) else json.dumps(value)).encode()
        (model / name).write_bytes(data)
        files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(json.dumps({"model_id": "Qwen/Qwen3.8-27B", "file_hashes": files}))
    proc = tmp_path / "proc"
    process = proc / "42"
    (process / "fd").mkdir(parents=True)
    (process / "fd" / "3").symlink_to("socket:[12345]")
    (process / "stat").write_text("42 (vllm worker) " + " ".join(["S"] + ["0"] * 18 + ["123"] + ["0"] * 5))
    argv = ["/runtime/bin/python", "/runtime/bin/vllm", "serve", str(model),
            "--served-model-name", "Qwen/Qwen3.8-27B", "--host", "127.0.0.1", "--port", "18001",
            "--tensor-parallel-size", "4", "--dtype", "bfloat16", "--max-model-len", "16384",
            "--max-num-seqs", "4", "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.72",
            "--language-model-only", "--reasoning-parser", "qwen3", "--default-chat-template-kwargs",
            '{"enable_thinking":false}', "--generation-config", "vllm", "--enforce-eager", "--disable-custom-all-reduce"]
    (process / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    (process / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=4,5,6,7\0LLM_API_KEY=never-record\0AWS_SECRET_ACCESS_KEY=also-secret\0")
    (proc / "net").mkdir()
    (proc / "net" / "tcp").write_text("header\n0: 0100007F:4651 00000000:0000 0A 0:0 0:0 0 1000 0 12345\n")
    return SimpleNamespace(proc=proc, process=process, manifest=manifest, model=model, argv=argv)


def test_busy_refuses_before_network_output_or_configuration_read(probe, tmp_path):
    @contextmanager
    def busy():
        raise RuntimeError("busy")
        yield
    def forbidden(*a, **kw):
        pytest.fail("busy must do no work")
    args = SimpleNamespace(output_dir=tmp_path / "out", service_pid=42, model_manifest=tmp_path / "absent")
    with pytest.raises(RuntimeError, match="busy"):
        probe.run_probe(args, workflow_lock=busy, platform_name="linux", capture=forbidden,
                        preflight=forbidden, benchmark=forbidden)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("field,value", [("LLM_PROVIDER", "deepseek"), ("LLM_MODEL", "other"),
    ("LLM_BASE_URL", "https://paid.invalid/v1"), ("LLM_BASE_URL", "http://127.0.0.1:18001/elsewhere"),
    ("LLM_MAX_TOKENS", ""), ("LLM_DISABLE_THINKING", "0")])
def test_explicit_local_generation_settings_required(probe, settings, monkeypatch, field, value):
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError):
        probe.checked_settings()


def test_service_config_is_process_and_socket_bound_without_secrets(probe, settings, service):
    result = probe.capture_service(42, settings, service.manifest, proc_root=service.proc)
    assert result["pid"] == 42 and result["start_ticks"] == "123"
    assert result["flags"]["dtype"] == "bfloat16" and result["flags"]["tensor-parallel-size"] == "4"
    assert result["flags"]["enforce-eager"] is True
    assert result["flags"]["disable-custom-all-reduce"] is True
    assert result["engine_activation_verified"] is False
    assert result["weight_content_rehashed"] is False
    assert "chat_template.jinja" in result["artifact_hashes"]
    assert "secret" not in json.dumps(result).lower()


@pytest.mark.parametrize("change", ["socket", "dtype", "unknown_flag", "template", "model_file"])
def test_config_mismatch_or_unbound_label_refused(probe, settings, service, change):
    if change == "socket":
        (service.process / "fd" / "3").unlink()
    elif change == "template":
        (service.model / "chat_template.jinja").write_text("modified")
    elif change == "model_file":
        (service.model / "model-00001.safetensors").write_text("modified")
    else:
        argv = service.argv.copy()
        if change == "dtype":
            argv[argv.index("bfloat16")] = "float16"
        else:
            argv += ["--api-key", "do-not-record"]
        (service.process / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    with pytest.raises(ValueError):
        probe.capture_service(42, settings, service.manifest, proc_root=service.proc)


def test_request_manifest_deterministic_exact_payload_bytes(probe, settings):
    first = probe.build_request_manifest(settings)
    assert first == probe.build_request_manifest(settings) and len(first) == 3
    assert {r["expected_decision"] for r in first} == {"accept", "reject"}
    for row in first:
        assert row["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
        assert row["payload_sha256"] == hashlib.sha256(row["payload_utf8"].encode()).hexdigest()
        payload = json.loads(row["payload_utf8"])
        assert payload["messages"] == [{"role": "user", "content": row["prompt"]}]
        assert payload["temperature"] == 0 and payload["max_tokens"] == 128


class UnitHTTP(urllib.request.HTTPHandler):
    """Inject only HTTP bytes; exercise the real client and capture/deadline path."""
    def __init__(self, reply, seen):
        super().__init__()
        self.reply, self.seen = reply, seen
    def http_open(self, request):
        self.seen.append(request)
        body = json.dumps(self.reply).encode()
        return urllib.response.addinfourl(io.BytesIO(body), {"Content-Type": "application/json"}, request.full_url, 200)


def response_for(case, usage=True):
    result = {"model": "Qwen/Qwen3.8-27B", "id": "unit-response", "choices": [{"finish_reason": "stop",
        "message": {"content": json.dumps({"task_id": case["case_id"], "decision": case["expected_decision"]})}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    return result


@pytest.mark.parametrize("change", [None, "missing_usage", "wrong_semantics", "length", "wrong_model"])
def test_real_client_raw_capture_and_fail_closed_usage(probe, settings, change):
    case = probe.build_request_manifest(settings)[0]
    reply, seen = response_for(case, usage=change != "missing_usage"), []
    if change == "wrong_semantics":
        reply["choices"][0]["message"]["content"] = '{"task_id":"wrong","decision":"accept"}'
    if change == "length":
        reply["choices"][0]["finish_reason"] = "length"
    if change == "wrong_model":
        reply["model"] = "other"
    def transport(capture):
        return probe.recording_transport(capture, handlers=[UnitHTTP(reply, seen)])
    row = probe.execute_request({"case": case, "settings": settings, "request_id": "unit", "phase": "measured",
                                 "round": 1, "mode": "serial"}, transport_factory=transport)
    assert len(seen) == row["model_calls"] == 1
    assert seen[0].data == case["payload_utf8"].encode()
    assert row["passed"] is (change is None)
    assert row["response_sha256"] == hashlib.sha256(row["response_text"].encode()).hexdigest()
    assert row["total_tokens"] == (None if change == "missing_usage" else 30)
    assert row["usage_complete"] is (change != "missing_usage")


def test_redirect_and_unexpected_endpoint_are_refused_offline(probe, settings):
    capture = []
    with probe.recording_transport(capture):
        with pytest.raises(ValueError):
            urllib.request.urlopen("https://paid.invalid/v1/chat/completions")
        opener = urllib.request._opener
        redirect = next(h for h in opener.handlers if isinstance(h, probe.runner._NoRedirect))
        with pytest.raises(ValueError):
            redirect.redirect_request(None, None, 302, "redirect", {}, "http://127.0.0.1:18002/v1")
    assert capture == []


def unit_request(spec):
    case = spec["case"]
    content = json.dumps({"task_id": case["case_id"], "decision": case["expected_decision"]})
    return {"request_id": spec["request_id"], "case_id": case["case_id"], "phase": spec["phase"],
            "round": spec["round"], "mode": spec["mode"], "prompt_sha256": case["prompt_sha256"],
            "payload_sha256": case["payload_sha256"], "passed": True, "status": "completed",
            "usage_complete": True, "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
            "model_calls": 1, "latency_seconds": 100, "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "response_text": content, "model": "Qwen/Qwen3.8-27B", "served_model": "Qwen/Qwen3.8-27B",
            "provider": "modelscope_local", "finish_reason": "stop", "wire_response_sha256": "a" * 64,
            "raw_usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "http_captures": [{"status_code": 200, "payload_sha256": case["payload_sha256"], "wire_response_sha256": "a" * 64}],
            "semantic_valid": True, "parsed": {"task_id": case["case_id"], "decision": case["expected_decision"]}}


def test_crossover_exact_requests_warmup_excluded_cost_included(probe, settings, tmp_path):
    seen, widths = [], []
    def request(spec):
        seen.append(spec)
        return unit_request(spec)
    def pool(max_workers):
        widths.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)
    times = iter([0, 1, 2, 12, 13, 18, 19, 23, 24, 32])
    result = probe.run_benchmark(probe.build_request_manifest(settings), settings, tmp_path,
                                 request_runner=request, pool_factory=pool, clock=lambda: next(times))
    assert widths == [1, 1, 2, 2, 1]
    assert len(seen) == 14 and len({s["request_id"] for s in seen}) == 14
    measured = [s for s in seen if s["phase"] == "measured"]
    assert len(measured) == 12 and result["cost_total_tokens"] == 420
    assert result["measured_total_tokens"] == 360 and result["warmup_total_tokens"] == 60
    for round_number in (1, 2):
        by_mode = {mode: [s["case"] for s in measured if s["round"] == round_number and s["mode"] == mode]
                   for mode in ("serial", "parallel")}
        assert by_mode["serial"] == by_mode["parallel"]
    assert [r["raw_wall_ratio"] for r in result["rounds"]] == [2, 2]
    assert result["valid_speed_claim"] is True


@pytest.mark.parametrize("failure", ["error", "unknown", "warmup"])
def test_errors_unknown_and_warmup_failure_never_faster(probe, settings, tmp_path, failure):
    seen = []
    def request(spec):
        seen.append(spec["request_id"])
        row = unit_request(spec)
        if spec["mode"] == "parallel" or failure == "warmup" and spec["phase"] == "warmup":
            if failure == "error":
                raise TimeoutError("private error")
            row.update(passed=False, usage_complete=False, total_tokens=None)
        return row
    result = probe.run_benchmark(probe.build_request_manifest(settings), settings, tmp_path,
                                 request_runner=request, pool_factory=ThreadPoolExecutor)
    assert len(seen) == len(set(seen)) <= 14
    assert result["valid_speed_claim"] is False and result["passed"] is False
    assert result["cost_total_tokens"] is None


def test_probe_newdir_lock_provenance_and_no_service_management(probe, settings, service, tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / "out", service_pid=42, model_manifest=service.manifest)
    held = []
    @contextmanager
    def lock():
        held.append(True)
        try:
            yield
        finally:
            held.clear()
    def capture(*a):
        assert held
        return probe.capture_service(*a, proc_root=service.proc)
    def preflight(settings):
        assert held
        return None, {"model_calls": 0, "unit_only": True}
    def benchmark(*a):
        assert held
        return probe.run_benchmark(*a, request_runner=unit_request, pool_factory=ThreadPoolExecutor)
    result = probe.run_probe(args, workflow_lock=lock, platform_name="linux", capture=capture,
                             preflight=preflight, benchmark=benchmark)
    assert result["service_unchanged"] is True and not held
    assert args.output_dir.stat().st_mode & 0o777 == 0o700
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert len(manifest["requests"]) == 3 and manifest["config_sha256"]
    assert "unit-secret" not in json.dumps(manifest)
    assert len((args.output_dir / "requests.jsonl").read_text().splitlines()) == 14
    assert (args.output_dir / "requests.csv").exists() and (args.output_dir / "summary_zh.md").exists()
    with pytest.raises(FileExistsError):
        probe.run_probe(args, workflow_lock=lock, platform_name="linux", capture=capture)


@pytest.mark.parametrize("change", ["false_usage", "bad_hash", "wrong_model", "http_error", "raw_text", "wrong_count", "case_id"])
def test_parent_rejects_false_worker_pass_flags(probe, settings, tmp_path, change):
    def request(spec):
        row = unit_request(spec)
        if change == "false_usage":
            row["raw_usage"] = {}
        elif change == "bad_hash":
            row["response_sha256"] = "b" * 64
        elif change == "wrong_model":
            row["served_model"] = "different-model"
        elif change == "http_error":
            row["http_captures"][0]["status_code"] = 500
        elif change == "raw_text":
            row["response_text"] = "not JSON"
            row["response_sha256"] = hashlib.sha256(row["response_text"].encode()).hexdigest()
        elif change == "wrong_count":
            row["model_calls"] = 2
        else:
            row["case_id"] = "not-the-requested-case"
        return row
    result = probe.run_benchmark(probe.build_request_manifest(settings), settings, tmp_path,
                                 request_runner=request, pool_factory=ThreadPoolExecutor)
    assert result["passed"] is False and result["valid_speed_claim"] is False
    if change == "false_usage":
        assert result["cost_total_tokens"] is None


def test_benchmark_refuses_changed_or_unbounded_case_set(probe, settings, tmp_path):
    cases = probe.build_request_manifest(settings)
    with pytest.raises(ValueError):
        probe.run_benchmark(cases * 2, settings, tmp_path,
                             request_runner=lambda _: pytest.fail("no calls for changed manifests"))
    assert list(tmp_path.iterdir()) == []


def test_same_payload_different_text_hashes_reported_not_hidden(probe, settings, tmp_path):
    def request(spec):
        row = unit_request(spec)
        if spec["mode"] == "parallel":
            row["response_text"] = json.dumps(row["parsed"], indent=2)
            row["response_sha256"] = hashlib.sha256(row["response_text"].encode()).hexdigest()
        return row
    result = probe.run_benchmark(probe.build_request_manifest(settings), settings, tmp_path,
                                 request_runner=request, pool_factory=ThreadPoolExecutor)
    assert result["passed"] is True
    assert all(r["semantics_consistent"] and not r["text_identical"] for r in result["response_comparisons"])


def offline_actual_client_worker(spec, expire):
    import os
    import threading
    import time
    from scripts import probe_local_inference_config as probe
    class Reply(UnitHTTP):
        def http_open(self, request):
            if expire:
                time.sleep(3)
            return super().http_open(request)
    def transport(capture):
        return probe.recording_transport(capture, handlers=[Reply(response_for(spec["case"]), [])])
    row = probe.execute_request(spec, transport_factory=transport)
    return row, os.getpid(), threading.current_thread() is threading.main_thread()


def test_spawn_actual_client_deadline_expires_then_worker_reused(probe, settings, monkeypatch):
    monkeypatch.setenv("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", "1")
    settings = probe.checked_settings()
    case = probe.build_request_manifest(settings)[0]
    spec = {"request_id": "unit-spawn", "phase": "measured", "round": 1, "mode": "serial", "case": case, "settings": settings}
    with probe._pool(max_workers=1) as pool:
        failed, pid, main = pool.submit(offline_actual_client_worker, spec, True).result(timeout=15)
        passed, same_pid, next_main = pool.submit(offline_actual_client_worker, spec, False).result(timeout=15)
    assert pid == same_pid and main and next_main
    assert failed["passed"] is False and failed["model_calls"] == 1 and failed["usage_complete"] is False
    assert failed["latency_seconds"] < 3
    assert passed["passed"] is True and passed["model_calls"] == 1


def test_nonlinux_and_actual_busy_lock_make_no_output(probe, settings, service, tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / "out", service_pid=42, model_manifest=service.manifest)
    with pytest.raises(ValueError, match="Linux"):
        probe.run_probe(args, platform_name="darwin")
    lock = lambda: probe.runner.locked_workflow(tmp_path / "unit.lock")
    with lock(), pytest.raises(RuntimeError):
        probe.run_probe(args, platform_name="linux", workflow_lock=lock)
    assert not args.output_dir.exists()


def test_changed_service_cannot_keep_speed_claim(probe, settings, service, tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / "out", service_pid=42, model_manifest=service.manifest)
    def capture(*a):
        return probe.capture_service(*a, proc_root=service.proc)
    def benchmark(*a):
        result = probe.run_benchmark(*a, request_runner=unit_request, pool_factory=ThreadPoolExecutor)
        result["valid_speed_claim"] = True
        (service.process / "stat").write_text((service.process / "stat").read_text().replace("123", "456"))
        return result
    result = probe.run_probe(args, platform_name="linux", workflow_lock=nullcontext, capture=capture,
                             preflight=lambda _: (None, {}), benchmark=benchmark)
    assert result["service_unchanged"] is False and result["valid_speed_claim"] is False


def test_actual_redirect_response_is_not_followed(probe, settings):
    case, seen = probe.build_request_manifest(settings)[0], []
    class Redirect(urllib.request.HTTPHandler):
        def http_open(self, request):
            seen.append(request.full_url)
            return urllib.response.addinfourl(io.BytesIO(b"{}"), {"Location": "https://paid.invalid/v1"}, request.full_url, 302)
    def transport(capture):
        return probe.recording_transport(capture, handlers=[Redirect()])
    row = probe.execute_request({"case": case, "settings": settings, "request_id": "redirect", "phase": "warmup",
                                 "round": 0, "mode": "serial"}, transport_factory=transport)
    assert seen == [settings["api_base_url"] + "/chat/completions"]
    assert row["passed"] is False and row["model_calls"] == 1 and row["usage_complete"] is False


def test_batch_service_check_stops_remaining_calls_and_retains_cost(probe, settings, tmp_path):
    checks = []
    def check():
        checks.append(1)
        if len(checks) == 2:
            raise ValueError("service changed")
    result = probe.run_benchmark(probe.build_request_manifest(settings), settings, tmp_path,
                                 request_runner=unit_request, pool_factory=ThreadPoolExecutor, before_batch=check)
    assert result["passed"] is False and result["valid_speed_claim"] is False
    assert len(result["rows"]) == 2 and result["cost_total_tokens"] == 60
    assert result["phase_errors"] == [{"phase": "measured", "round": 1, "mode": "serial", "error_type": "ValueError"}]


@pytest.mark.parametrize("remove", [["--enforce-eager"], ["--enforce-eager", "--disable-custom-all-reduce"]])
def test_candidate_flags_observed_not_claimed_activated(probe, settings, service, remove):
    before = probe.capture_service(42, settings, service.manifest, proc_root=service.proc)
    argv = [a for a in service.argv if a not in remove]
    (service.process / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    after = probe.capture_service(42, settings, service.manifest, proc_root=service.proc)
    assert before != after and probe.object_hash(before) != probe.object_hash(after)
    assert after["engine_activation_verified"] is False


@pytest.mark.parametrize("also_owned", [False, True])
def test_multiple_listen_inodes_at_endpoint_are_ambiguous(probe, settings, service, also_owned):
    table = service.proc / "net" / "tcp"
    table.write_text(table.read_text() + "1: 0100007F:4651 00000000:0000 0A 0:0 0:0 0 1000 0 67890\n")
    if also_owned:
        (service.process / "fd" / "4").symlink_to("socket:[67890]")
    with pytest.raises(ValueError, match="unambiguous"):
        probe.capture_service(42, settings, service.manifest, proc_root=service.proc)


def test_repeated_fd_for_one_listener_is_not_a_second_listener(probe, settings, service):
    (service.process / "fd" / "4").symlink_to("socket:[12345]")
    result = probe.capture_service(42, settings, service.manifest, proc_root=service.proc)
    assert result["listener_inode"] == "12345"


def duplicate_decision_content(case, variant):
    expected = case["expected_decision"]
    first = expected if variant == "identical" else ("reject" if expected == "accept" else "accept")
    last_key = '"deci\\u0073ion"' if variant == "escaped" else '"decision"'
    return ('{"task_id":' + json.dumps(case["case_id"]) + ',"decision":' + json.dumps(first)
            + ',' + last_key + ':' + json.dumps(expected) + '}')


@pytest.mark.parametrize("variant", ["contradictory", "identical", "escaped"])
def test_worker_rejects_duplicate_decision_keys(probe, settings, variant):
    case, seen = probe.build_request_manifest(settings)[0], []
    reply = response_for(case)
    content = duplicate_decision_content(case, variant)
    reply["choices"][0]["message"]["content"] = content
    def transport(capture):
        return probe.recording_transport(capture, handlers=[UnitHTTP(reply, seen)])
    row = probe.execute_request({"case": case, "settings": settings, "request_id": "duplicates", "phase": "warmup",
                                 "round": 0, "mode": "serial"}, transport_factory=transport)
    assert len(seen) == row["model_calls"] == 1
    assert row["passed"] is False and row["semantic_valid"] is False
    assert row["response_text"] == content and row["total_tokens"] == 30


@pytest.mark.parametrize("variant", ["contradictory", "identical", "escaped"])
def test_parent_rejects_duplicate_keys_despite_last_wins_parsed_value(probe, settings, tmp_path, variant):
    def request(spec):
        row = unit_request(spec)
        row["response_text"] = duplicate_decision_content(spec["case"], variant)
        row["response_sha256"] = hashlib.sha256(row["response_text"].encode()).hexdigest()
        assert json.loads(row["response_text"]) == row["parsed"]
        return row
    result = probe.run_benchmark(probe.build_request_manifest(settings), settings, tmp_path,
                                 request_runner=request, pool_factory=ThreadPoolExecutor)
    assert result["passed"] is False and result["valid_speed_claim"] is False
    assert all(row["semantic_valid"] is False for row in result["rows"])
    assert result["cost_total_tokens"] == 60


def test_worker_rejects_duplicate_keys_in_raw_api_envelope(probe, settings):
    case = probe.build_request_manifest(settings)[0]
    raw = json.dumps(response_for(case)).replace('"model":', '"model":"wrong","model":', 1).encode()
    class Reply(urllib.request.HTTPHandler):
        def http_open(self, request):
            return urllib.response.addinfourl(io.BytesIO(raw), {}, request.full_url, 200)
    def transport(capture):
        return probe.recording_transport(capture, handlers=[Reply()])
    row = probe.execute_request({"case": case, "settings": settings, "request_id": "duplicate-envelope", "phase": "warmup",
                                 "round": 0, "mode": "serial"}, transport_factory=transport)
    assert row["passed"] is False and row["model_calls"] == 1
