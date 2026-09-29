"""Offline safety contracts for the formal two-lane parent scheduler."""
import hashlib
import importlib
import json
import os
import threading
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_multimechanism_runner import OfflineClient, clean_row, jobs, local_env, tasks


def parallel():
    name = "scripts.run_multimechanism_parallel"
    assert importlib.util.find_spec(name) is not None, "parallel runner is not implemented"
    return importlib.import_module(name)


class OfflinePool:
    def __init__(self, output, outcome=None):
        self.output, self.outcome = output, outcome
        self.submitted = []
        self.live = {}
        self.peak = 0

    def submit(self, lane, spec):
        p = parallel()
        starts = p.runner.read_jsonl(self.output / "run_attempts.jsonl")
        assert starts[-1] == spec["start"]
        if spec["start"]["condition"] != "clean":
            terminals = p.runner.read_jsonl(self.output / "main_runs.jsonl") + p.runner.read_jsonl(self.output / "run_errors.jsonl")
            ended = {r["attempt_id"] for r in terminals}
            assert all(r["attempt_id"] in ended for r in starts if r["condition"] == "clean")
        assert lane not in self.live
        assert spec["start"]["job_key"] not in [s["start"]["job_key"] for s in self.live.values()]
        self.live[lane] = spec
        self.peak = max(self.peak, len(self.live))
        self.submitted.append(spec)
        owner = self

        class Deferred(Future):
            def result(self, timeout=None):
                del owner.live[lane]
                if owner.outcome:
                    return owner.outcome(spec)
                return {"kind": "row", "record": {**spec["start"], "run_id": spec["start"]["attempt_id"],
                    "total_tokens": None, "known_total_tokens": 0, "usage_complete": False}}

        return Deferred()


def execute(tmp_path, js, outcome=None, max_jobs=None):
    p = parallel()
    pool = OfflinePool(tmp_path, outcome)
    summary = p.execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770",
                             pool=pool, max_jobs=max_jobs)
    return summary, pool


def test_lane_is_stable_pair_sha256_and_full_formal_config(tmp_path, monkeypatch):
    p = parallel()
    ts = [{**tasks()[0], "task_id": f"formal{i}"} for i in range(10)]
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(ts))
    for k, v in local_env().items():
        monkeypatch.setenv(k, v)
    args = p.parse_args(["--task-manifest", str(manifest), "--output", str(tmp_path / "out"), "--max-jobs", "2"])
    config = p.build_config(args, ts)
    assert len(config["jobs"]) == config["planned_runs"] == config["shard_runs"] == 6300
    assert config["shard_count"] == 1
    assert config["parallel_execution"]["logical_lanes"] == 2
    path = Path(p.__file__)
    assert config["source_hashes"]["scripts/run_multimechanism_parallel.py"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for job in config["jobs"]:
        assert p.lane_for(job) == int(hashlib.sha256(job["pair_key"].encode()).hexdigest(), 16) % 2
    args.max_jobs = None
    assert p.build_config(args, ts) == config
    with pytest.raises(ValueError):
        p.build_config(args, tasks())

    out = tmp_path / "audit"
    out.mkdir()
    digest = p.runner.freeze_manifest(out, config, resume=False)
    auditor = importlib.import_module("scripts.audit_shopping_multimechanism").Auditor(out)
    audited, audited_digest, indexed, planned = auditor.manifest()
    assert audited == config and audited_digest == digest
    assert len(indexed) == planned == 6300 and not auditor.findings


def test_scheduler_unique_two_inflight_order_clean_barrier_and_resume(tmp_path):
    p = parallel()
    js = [j for j in jobs() if j["condition"] in {"clean", "valid_partial"}]
    summary, pool = execute(tmp_path, js)
    assert summary["status"] == "complete"
    assert pool.peak == 2
    starts = [s["start"] for s in pool.submitted]
    assert len({s["attempt_id"] for s in starts}) == len(js)
    assert len({s["ledger_path"] for s in starts}) == len(js)
    for lane in range(2):
        assert [s["job_key"] for s in starts if s["lane"] == lane] == [j["job_key"] for j in js if p.lane_for(j) == lane]
    first_fault = next(i for i, s in enumerate(starts) if s["condition"] != "clean")
    assert first_fault == sum(j["condition"] == "clean" for j in js)
    assert not p.runner.read_jsonl(tmp_path / "run_errors.jsonl")
    assert execute(tmp_path, js)[1].submitted == []


def test_two_lifetime_attempts_and_no_infinite_retry(tmp_path):
    def fail(spec):
        return {"kind": "error", "record": {**spec["start"], "status": "infra_error",
            "error_type": "OfflineError", "known_total_tokens": 7, "usage_complete": False}}
    summary, pool = execute(tmp_path, jobs()[:1], fail)
    assert len(pool.submitted) == 2
    assert summary["known_tokens_error_attempts"] == 14
    assert summary["exhausted_job_keys"] == [jobs()[0]["job_key"]]
    assert not execute(tmp_path, jobs()[:1], fail)[1].submitted


def test_circuit_latches_and_drains_late_success(tmp_path):
    p = parallel()
    js = [j for j in jobs() if j["condition"] == "clean"]
    completed = 0

    def outcome(spec):
        nonlocal completed
        completed += 1
        kind = "error" if completed <= 3 else "row"
        return {"kind": kind, "record": {**spec["start"], "run_id": spec["start"]["attempt_id"],
            "error_type": "OfflineError", "status": "infra_error", "known_total_tokens": 0}}

    summary, pool = execute(tmp_path, js, outcome)
    assert summary["circuit_breaker"]["tripped"]
    assert len(pool.submitted) <= 4
    assert not pool.live
    starts = p.runner.read_jsonl(tmp_path / "run_attempts.jsonl")
    terminals = p.runner.read_jsonl(tmp_path / "main_runs.jsonl") + p.runner.read_jsonl(tmp_path / "run_errors.jsonl")
    assert {r["attempt_id"] for r in starts} == {r["attempt_id"] for r in terminals}
    assert not execute(tmp_path, js)[1].submitted


def test_interrupted_attempt_reconciles_unknown_and_spends_only_remainder(tmp_path):
    p = parallel()
    j = jobs()[0]
    p.runner.append_jsonl(tmp_path / "run_attempts.jsonl", {**j, "attempt": 1, "attempt_id": "old", "config_digest": "digest"})
    summary, pool = execute(tmp_path, [j])
    assert len(pool.submitted) == 1 and pool.submitted[0]["start"]["attempt"] == 2
    error = p.runner.read_jsonl(tmp_path / "run_errors.jsonl")[0]
    assert error["error_type"] == "InterruptedAttempt" and error["usage_complete"] is False
    assert summary["usage_incomplete_attempts"] == 2


def test_checkpoint_does_not_enter_fault_before_remaining_clean(tmp_path):
    js = [j for j in jobs() if j["condition"] in {"clean", "valid_partial"}]
    summary, pool = execute(tmp_path, js, max_jobs=2)
    assert summary["planned_runs"] == len(js) and summary["status"] == "incomplete"
    assert len(pool.submitted) == 2
    assert all(s["start"]["condition"] == "clean" for s in pool.submitted)


def test_worker_reuses_nullable_usage_partial_context_and_independent_clients(tmp_path):
    p = parallel()
    clients, ledgers = [], []

    def client_factory(settings):
        client = OfflineClient()
        clients.append(client)
        return client

    async def trial(task, job, executor, client, **kwargs):
        client.record()
        client.request_log[-1]["prompt_tokens"] = None
        ledgers.append(kwargs["ledger_path"])
        error = RuntimeError("secret")
        error.partial_trial = {"action_ledger_state": None, "events": []}
        raise error from TimeoutError("secret")

    for i in range(2):
        start = {**jobs()[0], "attempt": i + 1, "attempt_id": str(i), "config_digest": "digest",
                 "ledger_path": f"action_ledgers/{i}.sqlite3"}
        result = p.execute_attempt({"task": tasks()[0], "start": start, "settings": {},
            "base_url": "http://127.0.0.1:17770", "output": str(tmp_path)},
            client_factory=client_factory, executor_factory=lambda _: SimpleNamespace(http_receipts=[]), trial=trial)
        record = result["record"]
        assert result["kind"] == "error" and record["status"] == "timeout"
        assert record["total_tokens"] is None and record["known_total_tokens"] == 7
        assert record["partial_trial"]["action_ledger_state"] is None
        assert "secret" not in json.dumps(result)
    assert clients[0] is not clients[1] and ledgers[0] != ledgers[1]
    assert not list(tmp_path.glob("*.jsonl"))


def test_actual_source_frozen_by_parent_same_across_all_arms(tmp_path):
    p = parallel()
    js = [j for j in jobs() if j["condition"] in {"clean", *p.runner.SOURCE_CONDITIONS}]

    def outcome(spec):
        start = spec["start"]
        result = clean_row(int(start["task_id"][1:])) if start["condition"] == "clean" else {}
        if start["condition"] in p.runner.SOURCE_CONDITIONS:
            source = start["cross_task_source"]
            assert source["source_task_id"] != start["task_id"]
            assert hashlib.sha256((tmp_path / source["file"]).read_bytes()).hexdigest() == source["file_sha256"]
        return {"kind": "row", "record": {**result, **start, "run_id": start["attempt_id"]}}

    summary, pool = execute(tmp_path, js, outcome)
    assert summary["status"] == "complete"
    sources = {}
    for spec in pool.submitted:
        start = spec["start"]
        if start["cross_task_source"]:
            key = (start["task_id"], start["topology"], start["repeat_index"])
            assert sources.setdefault(key, start["cross_task_source"]) == start["cross_task_source"]
    starts_bytes = (tmp_path / "run_attempts.jsonl").read_bytes()
    assert not execute(tmp_path, js)[1].submitted
    assert (tmp_path / "run_attempts.jsonl").read_bytes() == starts_bytes
    first = next(iter(sources.values()))
    (tmp_path / first["file"]).unlink()
    with pytest.raises(ValueError, match="source"):
        execute(tmp_path, js)
    assert not (tmp_path / first["file"]).exists()


def test_missing_actual_sources_block_without_attempts(tmp_path):
    p = parallel()
    js = [j for j in jobs() if j["condition"] in p.runner.SOURCE_CONDITIONS]
    summary, pool = execute(tmp_path, js)
    assert not pool.submitted
    assert set(summary["source_blocked_job_keys"]) == {j["job_key"] for j in js}
    assert len(p.runner.read_jsonl(tmp_path / "blocked_cells.jsonl")) == len({j["pair_key"] for j in js})


@pytest.mark.parametrize("damage", ["wrong_digest", "duplicate_jobs", "wrong_terminal"])
def test_resume_refuses_unmodified_original_digest_and_metadata(tmp_path, damage):
    p = parallel()
    js = jobs()[:1]
    execute(tmp_path, js)
    path = tmp_path / "main_runs.jsonl"
    if damage == "duplicate_jobs":
        js = js * 2
    else:
        row = p.runner.read_jsonl(path)[0]
        row["config_digest" if damage == "wrong_digest" else "arm"] = "tampered"
        path.write_text(json.dumps(row) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        execute(tmp_path, js)
    assert path.read_bytes() == before


@pytest.mark.parametrize("outcome", ["raised", "malformed", "changed_metadata", "nan"])
def test_worker_loss_or_invalid_result_unknown_not_free_and_bounded(tmp_path, outcome):
    p = parallel()

    def result(spec):
        if outcome == "raised":
            raise RuntimeError("worker-secret")
        if outcome == "malformed":
            return None
        record = {**spec["start"]}
        if outcome == "changed_metadata":
            record["config_digest"] = "wrong"
        else:
            record["total_tokens"] = float("nan")
        return {"kind": "row", "record": record}

    summary, pool = execute(tmp_path, jobs()[:1], result)
    assert len(pool.submitted) == 2 and summary["status"] == "incomplete"
    errors = p.runner.read_jsonl(tmp_path / "run_errors.jsonl")
    assert all(r["total_tokens"] is None and not r["usage_complete"] for r in errors)
    assert "worker-secret" not in json.dumps(errors)


def test_submit_failure_records_start_and_unknown_terminal(tmp_path):
    p = parallel()

    class BrokenPool:
        def submit(self, lane, spec):
            raise RuntimeError("submit-secret")

    summary = p.execute_jobs(tasks(), jobs()[:1], tmp_path, "digest", {}, "http://127.0.0.1:17770", pool=BrokenPool())
    assert summary["error_attempts"] == 2
    assert len(p.runner.read_jsonl(tmp_path / "run_attempts.jsonl")) == 2


def _offline_process_identity(spec):
    return {"pid": os.getpid(), "main_thread": threading.current_thread() is threading.main_thread(),
            "spec": spec}


def test_real_spawn_lanes_are_distinct_stable_processes_and_main_threads(monkeypatch):
    p = parallel()
    monkeypatch.setattr(p, "execute_attempt", _offline_process_identity)
    with p.SpawnLanes() as pool:
        first = [pool.submit(i, {"lane": i}) for i in range(2)]
        rows = [f.result(timeout=15) for f in first]
        again = [pool.submit(i, {}).result(timeout=15) for i in range(2)]
        assert all(pool_._mp_context.get_start_method() == "spawn" for pool_ in pool.pools)
    assert len({r["pid"] for r in rows} | {os.getpid()}) == 3
    assert [r["pid"] for r in rows] == [r["pid"] for r in again]
    assert all(r["main_thread"] for r in rows + again)


@pytest.mark.parametrize("flag", [["--shard-count", "2"], ["--repetitions", "1"], ["--simulate"], ["--mock"]])
def test_no_shard_small_goal_or_simulation_cli(flag):
    with pytest.raises((ValueError, SystemExit)):
        parallel().parse_args(["--task-manifest", "tasks.json", "--output", "out", *flag])


def test_parent_holds_both_locks_and_freezes_all_sources_before_preflight(tmp_path, monkeypatch):
    p = parallel()
    for k, v in local_env().items():
        monkeypatch.setenv(k, v)
    ts = [{**tasks()[0], "task_id": f"formal{i}"} for i in range(10)]
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps(ts))
    out = tmp_path / "out"
    lock = tmp_path / "workflow.lock"
    original_lock = p.runner.locked_workflow
    monkeypatch.setattr(p.runner, "locked_workflow", lambda: original_lock(lock))
    monkeypatch.setattr(p.sys, "platform", "linux")

    def preflight(settings):
        with pytest.raises(RuntimeError):
            with original_lock(lock):
                pass
        with pytest.raises(RuntimeError):
            with p.runner.locked_output(out):
                pass
        frozen = json.loads((out / "matrix_manifest.json").read_text())
        assert len(frozen["config"]["jobs"]) == 6300
        assert (out / "source_snapshot/scripts/run_multimechanism_parallel.py").read_bytes() == Path(p.__file__).read_bytes()
        raise RuntimeError("offline-preflight-stop")

    monkeypatch.setattr(p.runner, "preflight_client", preflight)
    argv = ["--task-manifest", str(manifest), "--output", str(out)]
    assert p.main(argv) == 1
    frozen = (out / "matrix_manifest.json").read_bytes()
    snapshot = out / "source_snapshot/scripts/run_multimechanism_parallel.py"
    snapshot.unlink()
    monkeypatch.setattr(p.runner, "preflight_client", lambda *_: pytest.fail("tampered resume reached preflight"))
    assert p.main(argv + ["--resume"]) == 1
    assert not snapshot.exists() and (out / "matrix_manifest.json").read_bytes() == frozen


@pytest.mark.parametrize("failure", [None, "prctl", "orphan"])
def test_linux_workers_bind_to_parent_death_before_any_attempt(monkeypatch, failure):
    p = parallel()
    assert hasattr(p, "_bind_parent_lifetime"), "spawn workers must not outlive the workflow-lock owner"
    monkeypatch.setattr(p.sys, "platform", "linux")
    calls = []

    def prctl(*args):
        calls.append(args)
        return -1 if failure == "prctl" else 0

    monkeypatch.setattr(p.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=prctl))
    monkeypatch.setattr(p.os, "getppid", lambda: 98 if failure == "orphan" else 99)
    if failure:
        with pytest.raises((OSError, RuntimeError)):
            p._bind_parent_lifetime(99)
    else:
        p._bind_parent_lifetime(99)
    assert calls == [(1, p.signal.SIGKILL, 0, 0, 0)]


def test_parent_interrupt_drains_all_inflight_without_new_submissions(tmp_path):
    p = parallel()
    js = [j for j in jobs() if j["condition"] == "clean"]
    submitted = []

    class InterruptPool:
        def submit(self, lane, spec):
            submitted.append(spec)

            class InterruptOnce:
                interrupted = False

                def result(self):
                    if spec is submitted[0] and not self.interrupted:
                        self.interrupted = True
                        raise KeyboardInterrupt()
                    return {"kind": "row", "record": {**spec["start"], "run_id": spec["start"]["attempt_id"]}}

            return InterruptOnce()

    with pytest.raises(KeyboardInterrupt):
        p.execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770", pool=InterruptPool())
    assert len(submitted) == 2
    assert len(p.runner.read_jsonl(tmp_path / "main_runs.jsonl")) == 2
    assert not p.runner.read_jsonl(tmp_path / "run_errors.jsonl")


def test_parent_failure_after_durable_terminal_never_duplicates_it(tmp_path, monkeypatch):
    p = parallel()
    append = p.runner.append_jsonl
    failed = False

    def interrupted_append(path, record):
        nonlocal failed
        append(path, record)
        if path.name == "main_runs.jsonl" and not failed:
            failed = True
            raise RuntimeError("interrupted-after-fsync")

    monkeypatch.setattr(p.runner, "append_jsonl", interrupted_append)
    with pytest.raises(RuntimeError, match="interrupted-after-fsync"):
        execute(tmp_path, [j for j in jobs() if j["condition"] == "clean"])
    rows = p.runner.read_jsonl(tmp_path / "main_runs.jsonl")
    assert len(rows) == len({r["attempt_id"] for r in rows}) == 2


def mixed_lane_jobs():
    p = parallel()
    candidates = [j for j in p.runner.matrix.build_jobs(tasks(), 3) if j["condition"] == "clean"]
    selected = {j["job_key"] for lane in range(2)
                for j in [j for j in candidates if p.lane_for(j) == lane][:8]}
    return [j for j in candidates if j["job_key"] in selected]


def assert_broken_pool_stop(output, js, broken_lane, phase, summary):
    p = parallel()
    starts = p.runner.read_jsonl(output / "run_attempts.jsonl")
    errors = p.runner.read_jsonl(output / "run_errors.jsonl")
    rows = p.runner.read_jsonl(output / "main_runs.jsonl")
    broken_jobs = [j for j in js if p.lane_for(j) == broken_lane]
    assert len(broken_jobs) == 8
    broken_starts = [s for s in starts if s["lane"] == broken_lane]
    assert len(broken_starts) == 1, "a broken process pool must not consume queued job budgets"
    assert broken_starts[0]["job_key"] == broken_jobs[0]["job_key"]
    assert {j["job_key"] for j in broken_jobs[1:]}.isdisjoint(s["job_key"] for s in starts)
    assert len(errors) == 1 and errors[0]["attempt"] == 1
    assert errors[0]["error_type"] == "BrokenProcessPool"
    assert errors[0]["total_tokens"] is None and errors[0]["usage_complete"] is False
    assert {s["attempt_id"] for s in starts} == {r["attempt_id"] for r in rows + errors}
    assert all(r["lane"] != broken_lane for r in rows)
    assert not summary["exhausted_job_keys"]
    assert summary["status"] == "incomplete"
    stops = list((output / "scheduler_stops").glob("*.json"))
    assert len(stops) == 1
    stop = json.loads(stops[0].read_text())
    assert stop == summary["scheduler_stop"]
    assert stop["kind"] == "scheduler_stop" and stop["error_type"] == "BrokenProcessPool"
    assert stop["phase"] == phase and stop["lane"] == broken_lane
    assert stop["attempt_id"] == broken_starts[0]["attempt_id"] and stop["config_digest"] == "digest"
    assert stop["resume_policy"] == "fresh_pool_remaining_attempt_budget"
    assert json.loads((output / "summary.json").read_text())["scheduler_stop"] == stop
    return starts, rows, stops[0]


@pytest.mark.parametrize("broken_lane", [0, 1])
@pytest.mark.parametrize("phase", ["result", "submit"])
def test_broken_pool_stops_both_lanes_without_spending_seven_queued_jobs(tmp_path, broken_lane, phase):
    p = parallel()
    js = mixed_lane_jobs()

    def outcome(spec):
        if spec["start"]["lane"] == broken_lane:
            raise BrokenProcessPool("must-not-log-worker-secret")
        return {"kind": "row", "record": {**spec["start"], "run_id": spec["start"]["attempt_id"]}}

    class MixedPool(OfflinePool):
        def submit(self, lane, spec):
            if phase == "submit" and lane == broken_lane:
                self.submitted.append(spec)
                raise BrokenProcessPool("must-not-log-submit-secret")
            return super().submit(lane, spec)

    pool = MixedPool(tmp_path, outcome)
    summary = p.execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770", pool=pool)
    starts, rows, stop = assert_broken_pool_stop(tmp_path, js, broken_lane, phase, summary)
    assert not pool.live
    if phase == "result":
        assert rows, "the healthy in-flight lane must be drained"
        assert len(starts) <= 3
    else:
        assert len(starts) == broken_lane + 1, "submit failure must stop even the current dispatch loop"
    assert "secret" not in stop.read_text()
    before = stop.read_bytes()
    resumed, resumed_pool = execute(tmp_path, js)
    assert resumed["status"] == "complete" and resumed["scheduler_stop"] is None
    assert stop.read_bytes() == before
    assert len(list(stop.parent.glob("*.json"))) == 1
    counts = p.reconcile(tmp_path, js, "digest")["attempts"]
    failed_key = next(s["job_key"] for s in starts if s["lane"] == broken_lane)
    assert counts[failed_key] == 2
    assert all(counts[j["job_key"]] == 1 for j in js if j["job_key"] != failed_key)


def _offline_worker_row(spec):
    return {"kind": "row", "record": {**spec["start"], "run_id": spec["start"]["attempt_id"],
        "worker_pid": os.getpid(), "worker_main_thread": threading.current_thread() is threading.main_thread()}}


def _offline_child_death(spec):
    if spec["start"]["lane"] == 1:
        os._exit(23)
    return _offline_worker_row(spec)


def test_actual_spawn_child_death_stops_mixed_lanes_and_fresh_pool_resume(tmp_path, monkeypatch):
    p = parallel()
    js = mixed_lane_jobs()
    monkeypatch.setattr(p, "execute_attempt", _offline_child_death)
    with p.SpawnLanes() as pool:
        summary = p.execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770", pool=pool)
        assert pool.pools[1]._broken
    starts, rows, stop = assert_broken_pool_stop(tmp_path, js, 1, "result", summary)
    assert rows and all(r["worker_pid"] != os.getpid() and r["worker_main_thread"] for r in rows)
    before = stop.read_bytes()
    monkeypatch.setattr(p, "execute_attempt", _offline_worker_row)
    with p.SpawnLanes() as pool:
        resumed = p.execute_jobs(tasks(), js, tmp_path, "digest", {}, "http://127.0.0.1:17770", pool=pool)
    assert resumed["status"] == "complete" and resumed["scheduler_stop"] is None
    assert stop.read_bytes() == before
    state = p.reconcile(tmp_path, js, "digest")
    failed_key = next(s["job_key"] for s in starts if s["lane"] == 1)
    assert state["attempts"][failed_key] == 2 and len(state["errors"]) == 1
    assert all(state["attempts"][j["job_key"]] == 1 for j in js if j["job_key"] != failed_key)
