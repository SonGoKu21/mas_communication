"""Offline continuation contracts for a four-lane operational epoch."""
import hashlib
import importlib
import json
import os
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_multimechanism_parallel_runner as legacy
from test_multimechanism_runner import jobs, local_env, tasks, offpeak_clock

EPOCH = "e" * 64


def driver():
    name = "scripts.run_multimechanism_parallel4"
    assert importlib.util.find_spec(name) is not None, "four-lane epoch driver is not implemented"
    return importlib.import_module(name)


def execute(output, js, outcome=None, max_jobs=None, pool=None):
    q = driver()
    pool = pool or legacy.OfflinePool(output, outcome)
    summary = q.execute_jobs(tasks(), js, output, "digest", {}, "http://127.0.0.1:17770",
        pool=pool, max_jobs=max_jobs, epoch_digest=EPOCH)
    return summary, pool


def mixed_jobs():
    q = driver()
    candidates = [j for j in q.runner.matrix.build_jobs(tasks(), 3) if j["condition"] == "clean"]
    keys = {j["job_key"] for lane in range(4)
            for j in [j for j in candidates if q.lane_for(j) == lane][:8]}
    return [j for j in candidates if j["job_key"] in keys]


def test_four_lanes_share_pair_assignment_preserve_order_barrier_and_epoch(tmp_path):
    q = driver()
    js = [j for j in q.runner.matrix.build_jobs(tasks(), 3) if j["condition"] in {"clean", "valid_partial"}]
    summary, pool = execute(tmp_path, js)
    assert summary["status"] == "complete" and pool.peak == 4
    starts = [s["start"] for s in pool.submitted]
    assert len({s["attempt_id"] for s in starts}) == len(js)
    assert len({s["ledger_path"] for s in starts}) == len(js)
    for lane in range(4):
        assert [s["job_key"] for s in starts if s["lane"] == lane] == [
            j["job_key"] for j in js if q.lane_for(j) == lane]
    for start in starts:
        assert start["lane"] == int(hashlib.sha256(start["pair_key"].encode()).hexdigest(), 16) % 4
        assert start["execution_epoch"] == EPOCH and start["config_digest"] == "digest"
    assert next(i for i, s in enumerate(starts) if s["condition"] != "clean") == sum(
        j["condition"] == "clean" for j in js)
    assert not execute(tmp_path, js)[1].submitted


def seed_old(output, job, *, attempt=1, kind="row"):
    p = legacy.parallel()
    start = {**job, "attempt": attempt, "attempt_id": f"old-{job['job_key']}-{attempt}",
        "config_digest": "digest", "lane": p.lane_for(job),
        "ledger_path": f"action_ledgers/old-{job['job_key']}-{attempt}.sqlite3", "cross_task_source": None}
    p.runner.append_jsonl(output / "run_attempts.jsonl", start)
    record = {**start, "run_id": start["attempt_id"], "known_total_tokens": 7,
              "usage_complete": False, "status": "infra_error" if kind == "error" else "completed"}
    p.runner.append_jsonl(output / ("run_errors.jsonl" if kind == "error" else "main_runs.jsonl"), record)
    return start


def test_mixed_history_keeps_exact_prefix_and_global_remaining_attempt_budget(tmp_path):
    q = driver()
    js = mixed_jobs()
    seed_old(tmp_path, js[0])
    seed_old(tmp_path, js[1], kind="error")
    seed_old(tmp_path, js[2])
    prefixes = {name: (tmp_path / name).read_bytes() for name in
                ("run_attempts.jsonl", "main_runs.jsonl", "run_errors.jsonl")}
    summary, pool = execute(tmp_path, js)
    assert summary["status"] == "complete"
    submitted = {s["start"]["job_key"]: s["start"] for s in pool.submitted}
    assert js[0]["job_key"] not in submitted and js[2]["job_key"] not in submitted
    assert submitted[js[1]["job_key"]]["attempt"] == 2
    for name, prefix in prefixes.items():
        assert (tmp_path / name).read_bytes().startswith(prefix)
    starts = q.runner.read_jsonl(tmp_path / "run_attempts.jsonl")
    assert all("execution_epoch" not in s for s in starts[:3])
    assert all(s["execution_epoch"] == EPOCH for s in starts[3:])
    assert not execute(tmp_path, js)[1].submitted


@pytest.mark.parametrize("damage", ["old_lane", "new_lane", "unknown_epoch", "terminal_epoch"])
def test_mixed_lane_and_epoch_binding_fail_closed_before_dispatch(tmp_path, damage):
    q = driver()
    js = mixed_jobs()
    seed_old(tmp_path, js[0])
    execute(tmp_path, js, max_jobs=1)
    path = tmp_path / ("main_runs.jsonl" if damage == "terminal_epoch" else "run_attempts.jsonl")
    records = q.runner.read_jsonl(path)
    if damage == "old_lane":
        records[0]["lane"] = 1 - records[0]["lane"]
    elif damage == "new_lane":
        records[-1]["lane"] = (records[-1]["lane"] + 1) % 4
    else:
        records[-1]["execution_epoch"] = "f" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    before = path.read_bytes()
    pool = legacy.OfflinePool(tmp_path)
    with pytest.raises(ValueError):
        execute(tmp_path, js, pool=pool)
    assert not pool.submitted and path.read_bytes() == before


def test_three_error_circuit_drains_four_lanes_and_stays_latched(tmp_path):
    q = driver()
    count = 0
    def outcome(spec):
        nonlocal count
        count += 1
        return {"kind": "error" if count <= 3 else "row", "record": {
            **spec["start"], "run_id": spec["start"]["attempt_id"], "known_total_tokens": 0}}
    js = mixed_jobs()
    summary, pool = execute(tmp_path, js, outcome)
    assert summary["circuit_breaker"]["tripped"] and len(pool.submitted) <= 6
    assert pool.peak == 4 and not pool.live
    starts = q.runner.read_jsonl(tmp_path / "run_attempts.jsonl")
    terminals = q.runner.read_jsonl(tmp_path / "main_runs.jsonl") + q.runner.read_jsonl(tmp_path / "run_errors.jsonl")
    assert {s["attempt_id"] for s in starts} == {s["attempt_id"] for s in terminals}
    assert not execute(tmp_path, js)[1].submitted


def test_interrupt_drains_four_and_never_duplicates_durable_terminal(tmp_path):
    q = driver()
    submitted = []
    class InterruptPool:
        def submit(self, lane, spec):
            submitted.append(spec)
            class Once:
                interrupted = False
                def result(self):
                    if spec is submitted[0] and not self.interrupted:
                        self.interrupted = True
                        raise KeyboardInterrupt()
                    return {"kind": "row", "record": {**spec["start"], "run_id": spec["start"]["attempt_id"]}}
            return Once()
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, mixed_jobs(), pool=InterruptPool())
    rows = q.runner.read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(submitted) == len(rows) == len({r["attempt_id"] for r in rows}) == 4
    assert not q.runner.read_jsonl(tmp_path / "run_errors.jsonl")


@pytest.mark.parametrize("lane", range(4))
@pytest.mark.parametrize("phase", ["submit", "result"])
def test_broken_pool_stops_and_drains_without_spending_queued_budgets(tmp_path, lane, phase):
    q = driver()
    def outcome(spec):
        if spec["start"]["lane"] == lane:
            raise BrokenProcessPool("private")
        return {"kind": "row", "record": {**spec["start"], "run_id": spec["start"]["attempt_id"]}}
    class Pool(legacy.OfflinePool):
        def submit(self, number, spec):
            if phase == "submit" and number == lane:
                self.submitted.append(spec)
                raise BrokenProcessPool("private")
            return super().submit(number, spec)
    js = mixed_jobs()
    pool = Pool(tmp_path, outcome)
    summary, _ = execute(tmp_path, js, pool=pool)
    assert summary["scheduler_stop"]["error_type"] == "BrokenProcessPool"
    assert summary["scheduler_stop"]["execution_epoch"] == EPOCH
    assert not pool.live
    starts = q.runner.read_jsonl(tmp_path / "run_attempts.jsonl")
    errors = q.runner.read_jsonl(tmp_path / "run_errors.jsonl")
    assert len([s for s in starts if s["lane"] == lane]) == len(errors) == 1
    assert errors[0]["total_tokens"] is None and not errors[0]["usage_complete"]
    assert len(starts) == lane + 1 if phase == "submit" else len(starts) <= 4 + lane
    assert execute(tmp_path, js)[0]["status"] == "complete"
    state = q.p.reconcile(tmp_path, js, "digest")
    assert state["attempts"][errors[0]["job_key"]] == 2
    assert all(n <= 2 for n in state["attempts"].values())


@pytest.mark.parametrize("phase", ["job", "request"])
def test_peak_stop_retains_usage_and_drains_without_retry(tmp_path, monkeypatch, phase):
    q = driver()
    original = q.runner.guard_paid_work
    calls = 0
    def guard(output, digest, *, phase):
        nonlocal calls
        calls += 1
        if calls == 3:
            return q.runner.record_peak_stop(output, digest, phase=phase)
        return original(output, digest, phase=phase)
    if phase == "job":
        monkeypatch.setattr(q.runner, "guard_paid_work", guard)
    def outcome(spec):
        return {"kind": "error", "record": {**spec["start"], "offpeak_blocked": True,
            "error_type": "DeepSeekPeakWindowError", "known_total_tokens": 11, "usage_complete": False}}
    summary, pool = execute(tmp_path, mixed_jobs(), outcome if phase == "request" else None)
    assert summary["scheduler_stop"] is not None and not pool.live
    assert len(pool.submitted) == (2 if phase == "job" else 4)
    assert all(s["start"]["attempt"] == 1 for s in pool.submitted)
    if phase == "request":
        assert summary["known_tokens_error_attempts"] == 44


@pytest.mark.parametrize("name", [
    "test_two_lifetime_attempts_and_no_infinite_retry",
    "test_checkpoint_does_not_enter_fault_before_remaining_clean",
    "test_actual_source_frozen_by_parent_same_across_all_arms",
    "test_missing_actual_sources_block_without_attempts",
    "test_submit_failure_records_start_and_unknown_terminal",
])
def test_reused_unmodified_scheduler_safety_contracts(tmp_path, monkeypatch, name):
    q = driver()
    real = q.execute_jobs
    adapter = SimpleNamespace(runner=q.runner, lane_for=q.lane_for,
        execute_jobs=lambda *a, **kw: real(*a, **kw, epoch_digest=EPOCH))
    monkeypatch.setattr(legacy, "parallel", lambda: adapter)
    getattr(legacy, name)(tmp_path)


def test_real_four_spawn_processes_use_original_attempt_and_parent_lifetime(monkeypatch):
    q = driver()
    assert q.execute_attempt is q.p.execute_attempt
    monkeypatch.setattr(q, "execute_attempt", legacy._offline_process_identity)
    with q.SpawnLanes() as pool:
        first = [pool.submit(i, {"lane": i}) for i in range(4)]
        rows = [f.result(timeout=15) for f in first]
        again = [pool.submit(i, {}).result(timeout=15) for i in range(4)]
        assert all(p._mp_context.get_start_method() == "spawn" for p in pool.pools)
        assert all(p._initializer is q.p._bind_parent_lifetime for p in pool.pools)
    assert len({r["pid"] for r in rows} | {os.getpid()}) == 5
    assert [r["pid"] for r in rows] == [r["pid"] for r in again]
    assert all(r["main_thread"] for r in rows + again)


@pytest.mark.parametrize("flags", [[], ["--max-jobs", "0"], ["--simulate"], ["--shard-count", "2"]])
def test_cli_cannot_create_reset_or_shard_matrix(flags):
    argv = ["--output-dir", "unused", *flags]
    if flags:
        argv.append("--resume")
    with pytest.raises((ValueError, SystemExit)):
        driver().parse_args(argv)


def formal_output(tmp_path, monkeypatch):
    p = legacy.parallel()
    for key, value in local_env().items():
        monkeypatch.setenv(key, value)
    manifest = tmp_path / "tasks.json"
    ts = [{**tasks()[0], "task_id": f"formal{i}"} for i in range(10)]
    manifest.write_text(json.dumps(ts))
    output = tmp_path / "out"
    output.mkdir()
    args = p.parse_args(["--task-manifest", str(manifest), "--output-dir", str(output)])
    config = p.build_config(args, ts)
    p.runner.freeze_manifest(output, config, resume=False)
    p.runner.freeze_source_snapshot(output, config["source_hashes"], root=p.ROOT, resume=False)
    return output, config


def test_cli_verifies_epoch_and_locks_before_preflight_and_preserves_base(tmp_path, monkeypatch):
    q = driver()
    output, config = formal_output(tmp_path, monkeypatch)
    frozen = (output / "matrix_manifest.json").read_bytes()
    calls = []
    lock = tmp_path / "workflow.lock"
    original = q.runner.locked_workflow
    monkeypatch.setattr(q.runner, "locked_workflow", lambda: original(lock))
    monkeypatch.setattr(q.sys, "platform", "linux")
    def prepare(out, cfg, root, isolation):
        with pytest.raises(RuntimeError):
            with original(lock):
                pass
        with pytest.raises(RuntimeError):
            with q.runner.locked_output(output):
                pass
        assert cfg == config and out == output and root == q.ROOT
        assert isolation == tmp_path / "isolation.json"
        calls.append("prepare")
        return {"epoch_digest": EPOCH}
    def verify(*_):
        calls.append("verify")
        return {"valid": True, "epoch_digest": EPOCH}
    monkeypatch.setattr(q, "epoch_helper", lambda: SimpleNamespace(prepare_epoch=prepare, verify_epoch=verify))
    def preflight(settings):
        assert calls == ["prepare", "verify"] and settings == config["inference_settings"]
        raise RuntimeError("offline-stop")
    monkeypatch.setattr(q.runner, "preflight_client", preflight)
    assert q.main(["--output-dir", str(output), "--resume", "--isolation-report", str(tmp_path / "isolation.json")]) == 1
    assert (output / "matrix_manifest.json").read_bytes() == frozen
    assert not (output / "run_attempts.jsonl").exists()


@pytest.mark.parametrize("damage", ["invalid_audit", "epoch_mismatch", "source_changed", "base_digest"])
def test_cli_refuses_unverified_epoch_or_base_before_network(tmp_path, monkeypatch, damage):
    q = driver()
    output, _ = formal_output(tmp_path, monkeypatch)
    monkeypatch.setattr(q.sys, "platform", "linux")
    lock = q.runner.locked_workflow
    monkeypatch.setattr(q.runner, "locked_workflow", lambda: lock(tmp_path / "lock"))
    if damage == "source_changed":
        (output / "source_snapshot/scripts/run_multimechanism_parallel.py").write_text("tampered")
    if damage == "base_digest":
        path = output / "matrix_manifest.json"
        value = json.loads(path.read_text())
        value["config_digest"] = "wrong"
        path.write_text(json.dumps(value))
    monkeypatch.setattr(q, "epoch_helper", lambda: SimpleNamespace(
        prepare_epoch=lambda *_: {"epoch_digest": EPOCH},
        verify_epoch=lambda *_: {"valid": damage != "invalid_audit",
            "epoch_digest": "f" * 64 if damage == "epoch_mismatch" else EPOCH}))
    monkeypatch.setattr(q.runner, "preflight_client", lambda *_: pytest.fail("unverified epoch reached preflight"))
    assert q.main(["--output-dir", str(output), "--resume"]) == 1
    assert not (output / "run_attempts.jsonl").exists()


def isolation_report(tmp_path):
    path = tmp_path / "isolation.json"
    path.write_text(json.dumps({"gate_type": "four_cart_http_isolation", "passed": True,
        "max_workers": 4, "model_calls": 0, "flows": [{"passed": True,
        "http_receipts_verified": True, "cart_id_sha256": hashlib.sha256(str(i).encode()).hexdigest()}
        for i in range(4)]}))
    return path


def main_offline(tmp_path, monkeypatch):
    q = driver()
    output, config = formal_output(tmp_path, monkeypatch)
    monkeypatch.setattr(q.sys, "platform", "linux")
    lock = q.runner.locked_workflow
    monkeypatch.setattr(q.runner, "locked_workflow", lambda: lock(tmp_path / "workflow.lock"))
    monkeypatch.setattr(q.runner, "preflight_client", lambda settings: (None, {
        "model": q.runner.MODEL, "model_calls": 0}))
    class Pool(legacy.OfflinePool):
        def __init__(self):
            super().__init__(output)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            assert not self.live
    monkeypatch.setattr(q, "SpawnLanes", Pool)
    return q, output, config


def test_real_epoch_helper_checkpoint28_and_resume_preserve_6300_and_prefix(tmp_path, monkeypatch):
    q, output, config = main_offline(tmp_path, monkeypatch)
    digest = q.runner.matrix.config_digest(config)
    job = config["jobs"][0]
    start = {**job, "attempt": 1, "attempt_id": "old-completed", "config_digest": digest,
        "lane": q.p.lane_for(job), "ledger_path": "action_ledgers/old.sqlite3", "cross_task_source": None}
    q.runner.append_jsonl(output / "run_attempts.jsonl", start)
    q.runner.append_jsonl(output / "main_runs.jsonl", {**start, "run_id": "old-completed"})
    original = {n: (output / n).read_bytes() for n in ("matrix_manifest.json", "run_attempts.jsonl", "main_runs.jsonl")}
    argv = ["--output-dir", str(output), "--resume", "--max-jobs", "28"]
    assert q.main(argv + ["--isolation-report", str(isolation_report(tmp_path))]) == 1
    helper = q.epoch_helper()
    epoch_path = output / "execution_epochs/parallel4_v1/manifest.json"
    epoch_bytes = epoch_path.read_bytes()
    epoch = json.loads(epoch_bytes)
    audit = helper.verify_epoch(output, config, q.ROOT)
    assert audit["valid"] and audit["legacy_attempts"] == 1 and audit["epoch_attempts"] == 28
    assert audit["unfinished_attempts"] == 0 and audit["completed_runs"] == 29
    assert q.main(argv) == 1
    assert epoch_path.read_bytes() == epoch_bytes
    starts = q.runner.read_jsonl(output / "run_attempts.jsonl")
    assert len(starts) == len({s["job_key"] for s in starts}) == 57
    assert all(s["attempt"] == 1 and s["config_digest"] == digest for s in starts)
    assert all(s["execution_epoch"] == epoch["epoch_digest"] for s in starts[1:])
    assert all((output / n).read_bytes().startswith(data) for n, data in original.items())
    assert (output / "matrix_manifest.json").read_bytes() == original["matrix_manifest.json"]
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["planned_runs"] == 6300 and summary["completed_runs"] == 57


def test_first_epoch_refuses_dangling_old_start_without_reconciling_or_paid_work(tmp_path, monkeypatch):
    q, output, config = main_offline(tmp_path, monkeypatch)
    start = {**config["jobs"][0], "attempt": 1, "attempt_id": "old-unfinished",
        "config_digest": q.runner.matrix.config_digest(config), "lane": q.p.lane_for(config["jobs"][0])}
    q.runner.append_jsonl(output / "run_attempts.jsonl", start)
    before = (output / "run_attempts.jsonl").read_bytes()
    monkeypatch.setattr(q.runner, "preflight_client", lambda *_: pytest.fail("dangling transition reached network"))
    assert q.main(["--output-dir", str(output), "--resume", "--isolation-report", str(isolation_report(tmp_path))]) == 1
    assert (output / "run_attempts.jsonl").read_bytes() == before
    assert not (output / "run_errors.jsonl").exists()
    assert not (output / "execution_epochs").exists()


def test_existing_epoch_interrupted_start_uses_only_second_attempt(tmp_path, monkeypatch):
    q, output, config = main_offline(tmp_path, monkeypatch)
    epoch = q.epoch_helper().prepare_epoch(output, config, q.ROOT, isolation_report(tmp_path))
    job = config["jobs"][0]
    start = {**job, "attempt": 1, "attempt_id": "epoch-unfinished",
        "config_digest": q.runner.matrix.config_digest(config), "execution_epoch": epoch["epoch_digest"],
        "lane": q.lane_for(job), "ledger_path": "action_ledgers/unfinished.sqlite3", "cross_task_source": None}
    q.runner.append_jsonl(output / "run_attempts.jsonl", start)
    assert q.main(["--output-dir", str(output), "--resume", "--max-jobs", "1"]) == 1
    starts = q.runner.read_jsonl(output / "run_attempts.jsonl")
    assert [s["attempt"] for s in starts] == [1, 2]
    assert {s["job_key"] for s in starts} == {job["job_key"]}
    errors = q.runner.read_jsonl(output / "run_errors.jsonl")
    assert len(errors) == 1 and errors[0]["error_type"] == "InterruptedAttempt"
    assert errors[0]["usage_complete"] is False
    assert q.epoch_helper().verify_epoch(output, config, q.ROOT)["unfinished_attempts"] == 0


def test_parent_failure_after_fsync_drains_without_duplicate_terminal(tmp_path, monkeypatch):
    q = driver()
    original = q.runner.append_jsonl
    failed = False
    def append(path, record):
        nonlocal failed
        original(path, record)
        if path.name == "main_runs.jsonl" and not failed:
            failed = True
            raise RuntimeError("after-fsync")
    monkeypatch.setattr(q.runner, "append_jsonl", append)
    with pytest.raises(RuntimeError, match="after-fsync"):
        execute(tmp_path, mixed_jobs())
    rows = q.runner.read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(rows) == len({r["attempt_id"] for r in rows}) == 4
    assert not q.runner.read_jsonl(tmp_path / "run_errors.jsonl")


def test_actual_four_lane_child_death_drains_and_resumes_remaining_budget(tmp_path, monkeypatch):
    q = driver()
    js = mixed_jobs()
    monkeypatch.setattr(q, "execute_attempt", legacy._offline_child_death)
    with q.SpawnLanes() as pool:
        summary, _ = execute(tmp_path, js, pool=pool)
        assert pool.pools[1]._broken
    errors = q.runner.read_jsonl(tmp_path / "run_errors.jsonl")
    assert len(errors) == 1 and errors[0]["total_tokens"] is None
    assert summary["scheduler_stop"]["error_type"] == "BrokenProcessPool"
    assert len(q.runner.read_jsonl(tmp_path / "main_runs.jsonl")) >= 3
    monkeypatch.setattr(q, "execute_attempt", legacy._offline_worker_row)
    with q.SpawnLanes() as pool:
        resumed, _ = execute(tmp_path, js, pool=pool)
    assert resumed["status"] == "complete"
    state = q.p.reconcile(tmp_path, js, "digest")
    assert state["attempts"][errors[0]["job_key"]] == 2
    assert all(n == 1 for key, n in state["attempts"].items() if key != errors[0]["job_key"])


def test_main_startup_peak_refuses_before_preflight_or_attempt(tmp_path, monkeypatch):
    q, output, config = main_offline(tmp_path, monkeypatch)
    from datetime import datetime
    from mas_faults import deepseek_schedule
    class Peak(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 14, 10, tzinfo=deepseek_schedule.BEIJING)
    monkeypatch.setattr(deepseek_schedule, "datetime", Peak)
    monkeypatch.setattr(q.runner, "preflight_client", lambda *_: pytest.fail("peak startup reached network"))
    assert q.main(["--output-dir", str(output), "--resume", "--isolation-report", str(isolation_report(tmp_path))]) == 1
    assert not (output / "run_attempts.jsonl").exists()
    assert json.loads((output / "summary.json").read_bytes())["scheduler_stop"]["phase"] == "startup"
    assert q.epoch_helper().verify_epoch(output, config, q.ROOT)["epoch_attempts"] == 0


def test_existing_epoch_torn_terminal_recovers_only_after_prefix_verification(tmp_path, monkeypatch):
    q, output, config = main_offline(tmp_path, monkeypatch)
    epoch = q.epoch_helper().prepare_epoch(output, config, q.ROOT, isolation_report(tmp_path))
    frozen = (output / "execution_epochs/parallel4_v1/manifest.json").read_bytes()
    job = config["jobs"][0]
    start = {**job, "attempt": 1, "attempt_id": "torn-terminal",
        "config_digest": q.runner.matrix.config_digest(config), "execution_epoch": epoch["epoch_digest"],
        "lane": q.lane_for(job), "ledger_path": "action_ledgers/torn.sqlite3", "cross_task_source": None}
    q.runner.append_jsonl(output / "run_attempts.jsonl", start)
    tail = b'{"attempt_id":'
    (output / "run_errors.jsonl").write_bytes(tail)
    assert q.main(["--output-dir", str(output), "--resume", "--max-jobs", "1"]) == 1
    starts = q.runner.read_jsonl(output / "run_attempts.jsonl")
    assert [s["attempt"] for s in starts] == [1, 2]
    assert len(q.runner.read_jsonl(output / "main_runs.jsonl")) == 1
    errors = q.runner.read_jsonl(output / "run_errors.jsonl")
    assert len(errors) == 1 and errors[0]["error_type"] == "InterruptedAttempt"
    assert errors[0]["usage_complete"] is False
    assert [p.read_bytes() for p in output.glob("run_errors.jsonl.torn-*")] == [tail]
    assert (output / "execution_epochs/parallel4_v1/manifest.json").read_bytes() == frozen
    assert q.epoch_helper().verify_epoch(output, config, q.ROOT)["unfinished_attempts"] == 0


def test_legacy_prefix_truncation_refuses_without_repair_or_preflight(tmp_path, monkeypatch):
    q, output, config = main_offline(tmp_path, monkeypatch)
    job = config["jobs"][0]
    start = {**job, "attempt": 1, "attempt_id": "old-prefix",
        "config_digest": q.runner.matrix.config_digest(config), "lane": q.p.lane_for(job)}
    q.runner.append_jsonl(output / "run_attempts.jsonl", start)
    q.runner.append_jsonl(output / "main_runs.jsonl", {**start, "run_id": "old-prefix"})
    q.epoch_helper().prepare_epoch(output, config, q.ROOT, isolation_report(tmp_path))
    path = output / "main_runs.jsonl"
    damaged = path.read_bytes()[:-5]
    path.write_bytes(damaged)
    monkeypatch.setattr(q.runner, "preflight_client", lambda *_: pytest.fail("damaged legacy prefix reached network"))
    assert q.main(["--output-dir", str(output), "--resume", "--max-jobs", "1"]) == 1
    assert path.read_bytes() == damaged and not list(output.glob("*.torn-*"))
