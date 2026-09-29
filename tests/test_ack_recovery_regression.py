"""Offline tests for the bounded real launcher; never experiment evidence."""
import hashlib
import importlib.util
import itertools
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    path = ROOT / "scripts/run_ack_recovery_regression.py"
    assert path.is_file(), "targeted launcher is missing"
    spec = importlib.util.spec_from_file_location("ack_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for key, value in {"LLM_PROVIDER": "modelscope_local", "LLM_MODEL": "Qwen/Qwen3.8-27B",
                       "LLM_BASE_URL": "http://127.0.0.1:18001/v1", "LLM_API_KEY": "unit-secret",
                       "LLM_MAX_TOKENS": "2048", "LLM_DISABLE_THINKING": "true"}.items():
        monkeypatch.setenv(key, value)
    tasks = [{"task_id": f"shopping-{i:03d}-q1to2", "product_title": f"Product {i}",
              "product_url": f"http://127.0.0.1:17770/product-{i}.html",
              "initial_quantity": 1, "quantity": 2} for i in range(1, 11)]
    manifest = tmp_path / "original-gate10.json"
    manifest.write_text(json.dumps({"tasks": tasks}))
    args = SimpleNamespace(task_manifest=manifest, task_id=tasks[0]["task_id"],
        output_dir=tmp_path / "ack-regression-new", base_url="http://127.0.0.1:17770",
        repetitions=1, shard_index=0, shard_count=1)
    return args, tasks


def test_exact_18_cells_freeze_original_input_then_select(setup, monkeypatch):
    m = launcher()
    args, tasks = setup
    original = m.runner.build_config
    seen = []

    def build(args, received):
        seen.append(received)
        return original(args, received)

    monkeypatch.setattr(m.runner, "build_config", build)
    config = m.build_config(args, tasks)
    assert seen == [tasks] and len(seen[0]) == 10
    assert config["tasks"] == [tasks[0]]
    assert config["planned_runs"] == config["shard_runs"] == 18
    jobs = config["jobs"]
    expected = set(itertools.product(("sequential", "flat", "hierarchical"),
        ("clean", "acknowledgement_loss"), ("baseline", "dependency", "combined")))
    assert {(j["topology"], j["condition"], j["arm"]) for j in jobs} == expected
    assert len({j["job_key"] for j in jobs}) == 18
    assert all(j["task_id"] == args.task_id and j["repeat_index"] == 1 for j in jobs)
    assert all(j["condition"] not in m.runner.SOURCE_CONDITIONS for j in jobs)
    assert config["max_attempts_per_job"] == 2
    assert config["max_consecutive_error_attempts"] == 3
    assert config["task_manifest_sha256"] == hashlib.sha256(args.task_manifest.read_bytes()).hexdigest()
    assert config["scope"]["selected_task_ids"] == [args.task_id]
    assert config["scope"]["input_task_count"] == 10
    assert config["source_hashes"]["scripts/run_ack_recovery_regression.py"] == hashlib.sha256(
        (ROOT / "scripts/run_ack_recovery_regression.py").read_bytes()).hexdigest()
    assert config["source_policy"] == "not_applicable_no_cross_task_source_conditions"


@pytest.mark.parametrize("damage", ["missing_id", "wrong_quantity", "single_task", "nonlocal_product", "changed_manifest"])
def test_invalid_selection_fails_before_output_or_network(setup, monkeypatch, damage):
    m = launcher()
    args, tasks = setup
    if damage == "missing_id":
        args.task_id = "shopping-001"  # No prefix or substring selection.
    elif damage == "wrong_quantity":
        tasks[0]["quantity"] = 3
    elif damage == "single_task":
        tasks = tasks[:1]
    elif damage == "nonlocal_product":
        tasks[0]["product_url"] = "https://example.com/product"
    if damage != "changed_manifest":
        args.task_manifest.write_text(json.dumps(tasks))
    else:
        args.task_manifest.write_text(json.dumps(tasks[1:]))
    monkeypatch.setattr(m.runner, "preflight_client", lambda *_: pytest.fail("network called"))
    with pytest.raises(ValueError):
        m.build_config(args, tasks)
    assert not args.output_dir.exists()


def test_host_lock_is_first_and_denial_has_no_output_or_network(setup, monkeypatch):
    m = launcher()
    args, _ = setup
    monkeypatch.setattr(m, "sys", SimpleNamespace(platform="linux"))

    @contextmanager
    def locked():
        raise RuntimeError("busy")
        yield

    monkeypatch.setattr(m.runner, "locked_workflow", locked)
    monkeypatch.setattr(m.runner, "load_tasks", lambda *_: pytest.fail("read before host lock"))
    monkeypatch.setattr(m.runner, "preflight_client", lambda *_: pytest.fail("network before lock"))
    with pytest.raises(RuntimeError, match="busy"):
        m.run_regression(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("unsafe", ["existing", "symlink", "nested_result", "pilot", "formal"])
def test_new_only_output_guard(setup, tmp_path, monkeypatch, unsafe):
    m = launcher()
    args, _ = setup
    monkeypatch.setattr(m, "sys", SimpleNamespace(platform="linux"))
    if unsafe == "existing":
        args.output_dir.mkdir()
    elif unsafe == "symlink":
        args.output_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    elif unsafe == "nested_result":
        parent = tmp_path / "old-run"
        parent.mkdir()
        (parent / "matrix_manifest.json").write_text("{}")
        args.output_dir = parent / "new"
    else:
        args.output_dir = tmp_path / f"{unsafe}-results" / "new"
    monkeypatch.setattr(m.runner, "preflight_client", lambda *_: pytest.fail("network called"))
    with pytest.raises((ValueError, FileExistsError)):
        m.run_regression(args)


def test_nonlinux_cli_has_no_side_effects(setup, monkeypatch):
    m = launcher()
    args, _ = setup
    monkeypatch.setattr(m, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(m.runner, "locked_workflow", lambda: pytest.fail("lock on non-Linux"))
    with pytest.raises(ValueError):
        m.run_regression(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("option", ["--resume", "--mock", "--repetitions", "--max-jobs"])
def test_cli_has_no_resume_mock_or_matrix_expansion(setup, option):
    args, _ = setup
    with pytest.raises(SystemExit):
        launcher().parse_args(["--task-manifest", str(args.task_manifest), "--task-id", args.task_id,
                              "--output-dir", str(args.output_dir), option])


@pytest.mark.parametrize("fail", [False, True])
def test_actual_execute_jobs_runs_only_selected_cells_under_locks(setup, monkeypatch, fail):
    m = launcher()
    args, tasks = setup
    monkeypatch.setattr(m, "sys", SimpleNamespace(platform="linux"))
    active = set()

    def lock(name):
        @contextmanager
        def context(*_):
            active.add(name)
            try:
                yield
            finally:
                active.remove(name)
        return context

    monkeypatch.setattr(m.runner, "locked_workflow", lock("host"))
    monkeypatch.setattr(m.runner, "locked_output", lock("output"))
    monkeypatch.setattr(m.runner, "local_inference_transport", lock("transport"))
    client = SimpleNamespace(prompt_tokens=0, completion_tokens=0, call_count=0, request_log=[])

    def preflight(settings):
        assert active == {"host", "output", "transport"}
        wrapped = json.loads((args.output_dir / "matrix_manifest.json").read_text())
        assert wrapped["config"]["planned_runs"] == 18
        assert (args.output_dir / "input_task_manifest.json").read_bytes() == args.task_manifest.read_bytes()
        assert (args.output_dir / "source_snapshot/scripts/run_ack_recovery_regression.py").is_file()
        return client, {"model_calls": 0}

    monkeypatch.setattr(m.runner, "preflight_client", preflight)
    from mas_faults import shopping_multimechanism as runtime
    from mas_faults import shopping_action_protocol as action
    seen = []

    async def trial(task, job, executor, received_client, **kwargs):
        assert active == {"host", "output", "transport"}
        assert task == tasks[0] and kwargs["cross_task_evidence"] is None
        seen.append(job)
        if fail:
            raise TimeoutError("unit-secret")
        return {"run_id": job["job_key"], "final_task_success": True, "final_commit_allowed": True,
                "environment_task_success": True, "common_recovery_enabled": True}

    monkeypatch.setattr(runtime, "run_trial", trial)
    monkeypatch.setattr(action, "MultiStateShoppingExecutor", lambda *_: SimpleNamespace(http_receipts=[]))
    result = m.run_regression(args)
    if fail:
        assert len(seen) == 3 and [j["attempt"] for j in seen] == [1, 2, 1]
        assert result["completed_runs"] == 0 and result["error_attempts"] == 3
    else:
        assert len(seen) == 18 and len({j["job_key"] for j in seen}) == 18
        assert all(j["attempt"] == 1 for j in seen)
        assert result["completed_runs"] == 18
    assert result["regression_passed"] is False  # No genuine model usage in these unit rows.
    assert result["formal_matrix_started"] is False
    assert result["strict_audit_passed"] is False
    assert active == set()


def test_existing_auditor_is_incompatible_without_weakening_gate(setup):
    m = launcher()
    args, tasks = setup
    config = m.build_config(args, tasks)
    args.output_dir.mkdir()
    m.runner.freeze_manifest(args.output_dir, config, resume=False)
    from scripts.audit_shopping_multimechanism import Auditor
    audit = Auditor(args.output_dir, ROOT)
    audit.manifest()
    codes = {finding["code"] for finding in audit.findings}
    assert {"manifest_jobs_mismatch", "planned_count_mismatch"} <= codes
    assert config["audit_compatibility"]["compatible"] is False


@pytest.mark.parametrize("damage", [None, "graph", "usage", "outcome", "missing", "errors", "fault", "budget"])
def test_targeted_report_requires_outcomes_usage_graph_and_actual_exposure(setup, damage):
    m = launcher()
    args, tasks = setup
    config = m.build_config(args, tasks)
    rows = [{**job, "environment_task_success": True, "final_task_success": True,
             "final_commit_allowed": True, "common_recovery_enabled": True, "graph_errors": [],
             "usage_complete": True, "total_tokens": 10, "model_calls": 1,
             "budget": {"limits": {"get": 4, "model_call": 3, "replay": 1},
                        "used": {"get": 0, "model_call": 0, "replay": 0}},
             "fault_events": [] if job["condition"] == "clean" else [{"condition": "acknowledgement_loss",
                 "boundary": "action_ack", "delivered_count": 0}]} for job in config["jobs"]]
    errors = []
    if damage == "graph":
        rows[0]["graph_errors"] = [{"error_type": "invalid_dependency"}]
    elif damage == "usage":
        rows[0]["total_tokens"] = None
    elif damage == "outcome":
        rows[0]["final_commit_allowed"] = False
    elif damage == "missing":
        rows.pop()
    elif damage == "errors":
        errors.append({"status": "timeout"})
    elif damage == "fault":
        next(r for r in rows if r["condition"] == "acknowledgement_loss")["fault_events"] = []
    elif damage == "budget":
        rows[0]["budget"]["used"]["get"] = 5
    summary = {"status": "complete", "exhausted_job_keys": [], "source_blocked_job_keys": [],
               "circuit_breaker": {"tripped": False}}
    result = m.gate_report(config, summary, rows, errors)
    assert result["regression_passed"] is (damage is None)
    assert result["strict_audit_passed"] is False and result["formal_matrix_authorized"] is False


def test_output_lock_failure_never_reaches_preflight(setup, monkeypatch):
    m = launcher()
    args, _ = setup
    monkeypatch.setattr(m, "sys", SimpleNamespace(platform="linux"))

    @contextmanager
    def denied(*_):
        raise RuntimeError("output locked")
        yield

    monkeypatch.setattr(m.runner, "locked_output", denied)
    monkeypatch.setattr(m.runner, "preflight_client", lambda *_: pytest.fail("network called"))
    with pytest.raises(RuntimeError, match="output locked"):
        m.run_regression(args)
    assert not (args.output_dir / "matrix_manifest.json").exists()


@pytest.mark.parametrize("change", [{"LLM_PROVIDER": "deepseek"}, {"LLM_MODEL": "other"},
                                   {"LLM_BASE_URL": "https://api.example.com"}])
def test_inference_guard_stops_before_output_or_preflight(setup, monkeypatch, change):
    m = launcher()
    args, _ = setup
    monkeypatch.setattr(m, "sys", SimpleNamespace(platform="linux"))
    for key, value in change.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(m.runner, "preflight_client", lambda *_: pytest.fail("network called"))
    with pytest.raises(ValueError):
        m.run_regression(args)
    assert not args.output_dir.exists()
