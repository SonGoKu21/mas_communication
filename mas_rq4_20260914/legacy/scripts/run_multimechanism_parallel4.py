"""Resume the frozen v2 matrix through a separately frozen four-lane epoch.

Only the parent appends global journals. Original scientific code, workers,
matrix digest, lifetime attempt budgets and source snapshots are unchanged.
The scheduling loop preserves v2 durable-start-order waiting and draining.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import multiprocessing
import os
import platform
import sys
import time
import uuid
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts import run_multimechanism_parallel as p

runner = p.runner
execute_attempt = p.execute_attempt
LANES = 4


def epoch_helper():
    return importlib.import_module("scripts.multimechanism_execution_epoch")


def lane_for(job):
    return int(hashlib.sha256(job["pair_key"].encode()).hexdigest(), 16) % LANES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--isolation-report", type=Path)
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("max-jobs must be positive")
    return args


def load_config(output):
    path = output / "matrix_manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("the original matrix manifest is required")
    manifest = json.loads(path.read_bytes())
    config = manifest["config"]
    digest = runner.matrix.config_digest(config)
    if manifest != {"config": config, "config_digest": digest}:
        raise ValueError("original matrix digest differs")
    if (config.get("model") != runner.MODEL or config.get("provider") != runner.PROVIDER
            or config.get("model_version") != runner.MODEL_VERSION
            or config.get("inference_settings") != runner.inference_settings()
            or config.get("max_attempts_per_job") != runner.MAX_ATTEMPTS
            or config.get("max_consecutive_error_attempts") != runner.MAX_CONSECUTIVE_ERRORS
            or config.get("repetitions") != 3 or config.get("shard_index") != 0
            or config.get("shard_count") != 1 or len(config.get("tasks", [])) != 10
            or config.get("planned_runs") != 6300 or config.get("shard_runs") != 6300
            or config.get("jobs") != runner.matrix.build_jobs(config["tasks"], 3)):
        raise ValueError("original full Flash matrix and inference settings required")
    policy = config.get("parallel_execution", {})
    if (policy.get("logical_lanes") != 2 or policy.get("max_inflight_attempts") != 2
            or policy.get("lane_assignment") != "sha256(pair_key_utf8)_integer_mod_2"):
        raise ValueError("original two-lane configuration required")
    if runner.validate_loopback_url(config["base_url"]) != config["base_url"]:
        raise ValueError("original Shopping endpoint differs")
    if "scripts/run_multimechanism_parallel.py" not in config.get("source_hashes", {}):
        raise ValueError("original scheduler source hash is required")
    return config, digest


class SpawnLanes:
    """Four single-process pools using the original attempt and lifetime binding."""
    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.pools = [self.stack.enter_context(ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context("spawn"), initializer=p._bind_parent_lifetime,
                initargs=(os.getpid(),))) for _ in range(LANES)]
        except BaseException:
            self.stack.close()
            raise
        return self

    def submit(self, lane, spec):
        return self.pools[lane].submit(execute_attempt, spec)

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def verify_used_sources(tasks, jobs, output, digest, epoch_digest):
    """Preserve v2 source checks, selecting each start's frozen lane policy."""
    starts = runner.read_jsonl(output / "run_attempts.jsonl")
    rows = runner.read_jsonl(output / "main_runs.jsonl")
    by_key = {j["job_key"]: j for j in jobs}
    by_id = {s.get("attempt_id"): s for s in starts}
    for record in rows + runner.read_jsonl(output / "run_errors.jsonl"):
        start = by_id.get(record.get("attempt_id"))
        if start is not None and record.get("execution_epoch") != start.get("execution_epoch"):
            raise ValueError("terminal execution epoch differs from durable start")
    checked = {}
    for start in starts:
        job = by_key.get(start.get("job_key"))
        if job is None or start.get("config_digest") != digest:
            raise ValueError("attempt differs from frozen configuration")
        epoch = start.get("execution_epoch")
        if epoch is not None and epoch != epoch_digest:
            raise ValueError("unknown attempt execution epoch")
        expected = p.lane_for(job) if epoch is None else lane_for(job)
        if type(start.get("lane")) is not int or start["lane"] != expected:
            raise ValueError("attempt lane differs from its execution epoch")
        source = start.get("cross_task_source")
        if job["condition"] not in runner.SOURCE_CONDITIONS:
            if source is not None:
                raise ValueError("unexpected cross-task source")
            continue
        target = (job["task_id"], job["topology"], job["repeat_index"])
        relative = "cross_task_sources/" + runner.matrix.config_digest(list(target)) + ".json"
        path = output / relative
        if not isinstance(source, dict) or source.get("file") != relative or not path.is_file() or path.is_symlink():
            raise ValueError("previously used frozen source missing or invalid")
        if target not in checked:
            checked[target] = runner.freeze_cross_task_source(rows, job, tasks, output, digest)
        if source != checked[target]:
            raise ValueError("attempt source differs from frozen actual source")


def execute_jobs(tasks, jobs, output, digest, settings, base_url, *, pool, epoch_digest, max_jobs=None):
    """Caller holds both locks and has verified the immutable epoch and prefix."""
    output = Path(output)
    if (not isinstance(epoch_digest, str) or len(epoch_digest) != 64
            or any(c not in "0123456789abcdef" for c in epoch_digest)):
        raise ValueError("verified execution epoch digest required")
    if len({j["job_key"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate global job_key")
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("max_jobs must be positive")
    verify_used_sources(tasks, jobs, output, digest, epoch_digest)
    state = p.reconcile(output, jobs, digest)
    threshold = runner.MAX_CONSECUTIVE_ERRORS
    if state["consecutive_error_attempts"] >= threshold:
        return runner.write_reports(output, jobs, state)
    task_map = {t["task_id"]: t for t in tasks}
    selected = state["pending"] if max_jobs is None else state["pending"][:max_jobs]
    blocked = []
    active = deque()
    busy = set()
    active_keys = set()
    consecutive = state["consecutive_error_attempts"]
    tripped = False
    scheduler_stop = None
    (output / "action_ledgers").mkdir(exist_ok=True)

    def stop_broken_pool(start, phase):
        nonlocal scheduler_stop
        if scheduler_stop is not None:
            return
        scheduler_stop = {"kind": "scheduler_stop", "error_type": "BrokenProcessPool",
            "config_digest": digest, "execution_epoch": epoch_digest,
            "job_key": start["job_key"], "attempt_id": start["attempt_id"],
            "attempt": start["attempt"], "lane": start["lane"], "phase": phase,
            "timestamp_unix": time.time(), "resume_policy": "fresh_pool_remaining_attempt_budget"}
        directory = output / "scheduler_stops"
        directory.mkdir(exist_ok=True)
        runner._sync_directory(output)
        runner._write_json(directory / (start["attempt_id"] + ".json"), scheduler_stop)

    def finish(entry):
        nonlocal consecutive, tripped, scheduler_stop
        lane, job, start, future = entry
        try:
            message = future.result()
            if not isinstance(message, dict) or message.get("kind") not in {"row", "error"}:
                raise ValueError("invalid worker terminal message")
            record = message["record"]
            if not isinstance(record, dict) or any(record.get(k) != v for k, v in start.items()):
                raise ValueError("worker changed parent attempt metadata")
            json.dumps(record, allow_nan=False)
            kind = message["kind"]
        except Exception as exc:
            if isinstance(exc, BrokenProcessPool):
                stop_broken_pool(start, "result")
            kind = "error"
            record = {**start, **p._unknown_usage(), "status": "infra_error",
                      "error_type": type(exc).__name__, "worker_result_unavailable": True,
                      "timestamp_unix": time.time()}
        runner.append_jsonl(output / ("main_runs.jsonl" if kind == "row" else "run_errors.jsonl"), record)
        state["rows" if kind == "row" else "errors"].append(record)
        if kind == "error" and (record.get("offpeak_blocked") is True
                or record.get("error_type") == "DeepSeekPeakWindowError") and scheduler_stop is None:
            scheduler_stop = runner.record_peak_stop(output, digest, phase="request", start=start)
        consecutive = consecutive + 1 if kind == "error" else 0
        tripped = tripped or consecutive >= threshold
        busy.remove(lane)
        active_keys.remove(job["job_key"])
        return kind

    try:
        for clean_phase in (True, False):
            if not clean_phase:
                done = {r["job_key"] for r in state["rows"]}
                if any(j["condition"] == "clean" and j["job_key"] not in done
                       and state["attempts"][j["job_key"]] < runner.MAX_ATTEMPTS for j in jobs):
                    break
            queues = [deque(j for j in selected if (j["condition"] == "clean") == clean_phase
                            and lane_for(j) == lane) for lane in range(LANES)]
            while any(queues) or active:
                if not tripped and scheduler_stop is None:
                    for lane in range(LANES):
                        if scheduler_stop is not None:
                            break
                        if lane in busy or not queues[lane]:
                            continue
                        scheduler_stop = runner.guard_paid_work(output, digest, phase="job")
                        if scheduler_stop is not None:
                            break
                        job = queues[lane].popleft()
                        key = job["job_key"]
                        if key in active_keys:
                            raise ValueError("duplicate active job_key")
                        source = None
                        if job["condition"] in runner.SOURCE_CONDITIONS:
                            try:
                                source = runner.freeze_cross_task_source(state["rows"], job, tasks, output, digest)
                            except runner.SourceUnavailable:
                                blocked.append(key)
                                known = runner.read_jsonl(output / "blocked_cells.jsonl")
                                if not any(r.get("pair_key") == job["pair_key"] for r in known):
                                    runner.append_jsonl(output / "blocked_cells.jsonl", {
                                        "pair_key": job["pair_key"], "config_digest": digest,
                                        "execution_epoch": epoch_digest,
                                        "reason": "missing_actual_baseline_clean_source", "timestamp_unix": time.time()})
                                continue
                        number = state["attempts"][key] + 1
                        if number > runner.MAX_ATTEMPTS:
                            raise ValueError("lifetime attempt budget exceeded")
                        attempt_id = uuid.uuid4().hex
                        start = {**job, "attempt": number, "attempt_id": attempt_id,
                            "config_digest": digest, "execution_epoch": epoch_digest,
                            "model": runner.MODEL, "provider": runner.PROVIDER, "model_version": runner.MODEL_VERSION,
                            "lane": lane, "ledger_path": f"action_ledgers/{attempt_id}.sqlite3",
                            "cross_task_source": source, "timestamp_unix": time.time()}
                        runner.append_jsonl(output / "run_attempts.jsonl", start)
                        state["attempts"][key] = number
                        spec = {"start": copy.deepcopy(start), "task": copy.deepcopy(task_map[job["task_id"]]),
                                "output": str(output.resolve()), "base_url": base_url, "settings": copy.deepcopy(settings)}
                        pool_broken = False
                        try:
                            future = pool.submit(lane, spec)
                        except Exception as exc:
                            pool_broken = isinstance(exc, BrokenProcessPool)
                            future = Future()
                            future.set_exception(exc)
                        active.append((lane, job, start, future))
                        busy.add(lane)
                        active_keys.add(key)
                        if pool_broken:
                            stop_broken_pool(start, "submit")
                if active:
                    entry = active[0]
                    kind = finish(entry)
                    active.popleft()
                    lane, job, start, _ = entry
                    if (kind == "error" and start["attempt"] < runner.MAX_ATTEMPTS
                            and not tripped and scheduler_stop is None):
                        queues[lane].appendleft(job)
                elif tripped or scheduler_stop is not None:
                    break
            if tripped or scheduler_stop is not None:
                break
    finally:
        durable = {r["attempt_id"] for name in ("main_runs.jsonl", "run_errors.jsonl")
                   for r in runner.read_jsonl(output / name)}
        while active:
            if active[0][2]["attempt_id"] not in durable:
                finish(active[0])
            active.popleft()
        state = p.reconcile(output, jobs, digest)
        summary = runner.write_reports(output, jobs, state, blocked, scheduler_stop=scheduler_stop)
        summary["execution_epoch"] = epoch_digest
        runner._write_json(output / "summary.json", summary)
    return summary


def checked_epoch(helper, output, config, expected):
    audit = helper.verify_epoch(output, config, ROOT)
    if not isinstance(audit, dict) or audit.get("valid") is not True or audit.get("epoch_digest") != expected:
        raise ValueError("execution epoch verification failed")
    return audit


def main(argv=None):
    try:
        args = parse_args(argv)
        if sys.platform != "linux":
            raise ValueError("formal results must be produced on the Linux server")
        if not args.output_dir.is_dir():
            raise ValueError("existing v2 output directory required")
        with runner.locked_workflow(), runner.locked_output(args.output_dir), runner.local_inference_transport():
            config, digest = load_config(args.output_dir)
            runner.freeze_source_snapshot(args.output_dir, config["source_hashes"], root=ROOT, resume=True)
            helper = epoch_helper()
            if (args.output_dir / "execution_epochs/parallel4_v1/manifest.json").exists():
                helper.recover_epoch_journals(args.output_dir, config, ROOT)
            epoch = helper.prepare_epoch(args.output_dir, config, ROOT, args.isolation_report)
            epoch_digest = epoch["epoch_digest"]
            checked_epoch(helper, args.output_dir, config, epoch_digest)
            tasks, jobs = config["tasks"], config["jobs"]
            verify_used_sources(tasks, jobs, args.output_dir, digest, epoch_digest)
            state = p.reconcile(args.output_dir, jobs, digest)
            summary = runner.write_reports(args.output_dir, jobs, state)
            if summary["circuit_breaker"]["tripped"]:
                return 1
            if state["pending"]:
                stop = runner.guard_paid_work(args.output_dir, digest, phase="startup")
                if stop is not None:
                    runner.write_reports(args.output_dir, jobs, state, scheduler_stop=stop)
                    return 1
                _, preflight = runner.preflight_client(config["inference_settings"])
                runner.append_jsonl(args.output_dir / "preflight.jsonl", {**preflight,
                    "config_digest": digest, "execution_epoch": epoch_digest,
                    "python": platform.python_version(), "platform": platform.system(),
                    "resume": True, "max_jobs": args.max_jobs,
                    "parallel_execution": epoch["config"]["parallel_execution"]})
                try:
                    with SpawnLanes() as pool:
                        summary = execute_jobs(tasks, jobs, args.output_dir, digest,
                            config["inference_settings"], config["base_url"], pool=pool,
                            epoch_digest=epoch_digest, max_jobs=args.max_jobs)
                finally:
                    checked_epoch(helper, args.output_dir, config, epoch_digest)
        return 0 if summary["status"] == "complete" else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Four-lane runner stopped ({type(exc).__name__}); inspect server-side journals. "
              "Exception text suppressed for credential safety.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
