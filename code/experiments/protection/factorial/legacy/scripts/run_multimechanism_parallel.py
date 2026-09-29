"""Formal 6,300-job Shopping runner with two isolated spawn-process lanes.

Only the parent writes global journals, frozen sources, manifests and reports.
Checkpoints limit jobs considered, never the frozen matrix or lifetime budget.
No simulation CLI, paid fallback, or worker-local retries are provided.
Broken pools stop the invocation; explicit resume uses fresh pools and the
remaining lifetime budget. Scheduler-stop records remain as audit history.
"""
from __future__ import annotations

import asyncio
import copy
import ctypes
import hashlib
import json
import multiprocessing
import os
import platform
import signal
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

import run_shopping_multimechanism as runner


def lane_for(job):
    return int(hashlib.sha256(job["pair_key"].encode()).hexdigest(), 16) % 2


def parse_args(argv=None):
    args = runner.parse_args(argv)
    if args.shard_index != 0 or args.shard_count != 1 or args.repetitions != 3:
        raise ValueError("formal parallel execution requires the full unsharded three-repeat matrix")
    return args


def build_config(args, tasks):
    if len(tasks) != 10 or args.repetitions != 3 or args.shard_index != 0 or args.shard_count != 1:
        raise ValueError("formal goal requires ten tasks, three repeats and no shards")
    config = runner.build_config(args, tasks)
    if len(config["jobs"]) != 6300:
        raise ValueError("formal goal must contain exactly 6300 jobs")
    config["parallel_execution"] = {
        "logical_lanes": 2, "max_inflight_attempts": 2, "start_method": "spawn",
        "lane_assignment": "sha256(pair_key_utf8)_integer_mod_2",
        "lane_order": "frozen_jobs_order_retry_before_next_job",
        "terminal_order": "durable_start_order", "clean_barrier": True,
        "circuit_policy": "latched_first_three_consecutive_errors_in_start_order",
        "worker_client_scope": "fresh_per_attempt", "global_writer": "parent_only",
        "worker_lifetime": "linux_parent_death_sigkill",
        "broken_pool_policy": "stop_all_submissions_drain_resume_with_fresh_pool",
        "timing_comparability": "parallel_not_serial_pilot",
    }
    script = Path(__file__).resolve()
    config["source_hashes"][str(script.relative_to(ROOT))] = hashlib.sha256(script.read_bytes()).hexdigest()
    return config


def _real_client(settings):
    if (sys.platform != "linux" or runner.inference_settings() != settings
            or not runner.os.environ.get("LLM_API_KEY")):
        raise ValueError("Linux and unchanged explicit Flash inference settings required")
    client = runner.get_llm_client()
    info = client.model_info
    if (info.provider != runner.PROVIDER or info.model != runner.MODEL
            or info.client_type != "http_openai_compatible"
            or runner.openai_api_base(runner.validate_deepseek_url(info.base_url)) != settings["api_base_url"]):
        raise ValueError("worker client differs from frozen Flash inference settings")
    return client


def _executor(base_url):
    from mas_faults.shopping_action_protocol import MultiStateShoppingExecutor
    executor = MultiStateShoppingExecutor(base_url)
    executor.session.trust_env = False
    return executor


def _unknown_usage():
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
            "known_prompt_tokens": 0, "known_completion_tokens": 0, "known_total_tokens": 0,
            "model_calls": None, "model_requests": [], "usage_complete": False}


def _unsent_usage():
    return {**_unknown_usage(), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "model_calls": 0, "request_sent": False}


def execute_attempt(spec, *, client_factory=None, executor_factory=None, trial=None):
    """One attempt on the spawned process main thread; never write global files."""
    started = time.perf_counter()
    start = spec["start"]
    client = before = executor = None
    try:
        runner.ensure_deepseek_offpeak(runner.MODEL)
        if trial is None:
            from mas_faults.shopping_multimechanism import run_trial
            trial = run_trial
        with runner.local_inference_transport():
            client = (client_factory or _real_client)(spec["settings"])
            before = runner.usage_snapshot(client)
            if any(before.values()):
                raise ValueError("attempt client must be unused")
            executor = (executor_factory or _executor)(spec["base_url"])
            source = start.get("cross_task_source")
            result = asyncio.run(trial(copy.deepcopy(spec["task"]),
                {**{k: start[k] for k in ("job_key", "pair_key", "task_id", "topology", "condition",
                    "boundary", "repeat_index", "arm")}, "attempt": start["attempt"],
                    "attempt_id": start["attempt_id"]}, executor, client,
                cross_task_evidence=copy.deepcopy(source["envelope"]) if source else None,
                ledger_path=Path(spec["output"]) / start["ledger_path"]))
            if not isinstance(result, dict):
                raise TypeError("run_trial must return a dict")
            record = {**result, **start, **runner.completed_usage(client, before, result),
                      "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
            if not record.get("run_id"):
                record["run_id"] = start["attempt_id"]
            json.dumps(record, allow_nan=False)
            return {"kind": "row", "record": record}
    except BaseException as exc:
        record = {**start, **(runner.usage_since(client, before, complete=False)
                  if client is not None and before is not None else _unsent_usage()),
                  "error_type": type(exc).__name__,
                  "offpeak_blocked": runner.is_peak_exception(exc),
                  "status": "timeout" if runner.is_timeout_exception(exc) else "infra_error",
                  "partial_trial": runner.partial_trial_context(exc),
                  "http_receipts": copy.deepcopy(getattr(executor, "http_receipts", [])),
                  "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        return {"kind": "error", "record": record}
    finally:
        session = getattr(executor, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def _bind_parent_lifetime(parent_pid):
    if sys.platform != "linux":
        return  # Portable offline tests only; the operational CLI requires Linux.
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG also protects ledger isolation after a parent SIGKILL.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot bind worker lifetime")
    if os.getppid() != parent_pid:
        raise RuntimeError("parent exited before worker initialization")


class SpawnLanes:
    """One single-process pool per logical lane; no shared clients or runtimes."""
    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.pools = [self.stack.enter_context(ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context("spawn"), initializer=_bind_parent_lifetime,
                initargs=(os.getpid(),))) for _ in range(2)]
        except BaseException:
            self.stack.close()
            raise
        return self

    def submit(self, lane, spec):
        return self.pools[lane].submit(execute_attempt, spec)

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def reconcile(output, jobs, digest):
    """Call only without active workers; preserve the serial recovery contract."""
    starts = runner.read_jsonl(output / "run_attempts.jsonl")
    ids = [s.get("attempt_id") for s in starts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global attempt_id")
    by_id = {s.get("attempt_id"): s for s in starts}
    for name in ("main_runs.jsonl", "run_errors.jsonl"):
        for record in runner.read_jsonl(output / name):
            start = by_id.get(record.get("attempt_id"))
            if start is not None and any(record.get(k) != start.get(k)
                    for k in ("cross_task_source", "ledger_path", "lane")):
                raise ValueError("terminal attempt binding differs from durable start")
    state = runner.reconcile_attempts(output, jobs, digest)
    errors = {r["attempt_id"] for r in state["errors"]}
    consecutive = 0
    for start in starts:
        consecutive = consecutive + 1 if start["attempt_id"] in errors else 0
        if consecutive >= runner.MAX_CONSECUTIVE_ERRORS:
            # A success drained after a trip cannot reset the frozen circuit.
            state["consecutive_error_attempts"] = max(state["consecutive_error_attempts"], consecutive)
    return state


def verify_used_sources(tasks, jobs, output, digest):
    """Resume must verify previously used sources, never recreate missing files."""
    starts = runner.read_jsonl(output / "run_attempts.jsonl")
    rows = runner.read_jsonl(output / "main_runs.jsonl")
    by_key = {j["job_key"]: j for j in jobs}
    checked = {}
    for start in starts:
        job = by_key.get(start.get("job_key"))
        if job is None or start.get("config_digest") != digest:
            raise ValueError("attempt differs from frozen configuration")
        if "lane" in start and start["lane"] != lane_for(job):
            raise ValueError("attempt lane differs from stable pair assignment")
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


def execute_jobs(tasks, jobs, output, digest, settings, base_url, *, pool, max_jobs=None):
    """Parent scheduler. Caller holds both locks; pool injection is test-only."""
    output = Path(output)
    if len({j["job_key"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate global job_key")
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("max_jobs must be positive")
    verify_used_sources(tasks, jobs, output, digest)
    state = reconcile(output, jobs, digest)
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
            "config_digest": digest, "job_key": start["job_key"], "attempt_id": start["attempt_id"],
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
            record = {**start, **_unknown_usage(), "status": "infra_error",
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
                # A checkpoint cannot use partial clean evidence as a final source pool.
                done = {r["job_key"] for r in state["rows"]}
                if any(j["condition"] == "clean" and j["job_key"] not in done
                       and state["attempts"][j["job_key"]] < runner.MAX_ATTEMPTS for j in jobs):
                    break
            queues = [deque(j for j in selected if (j["condition"] == "clean") == clean_phase
                            and lane_for(j) == lane) for lane in range(2)]
            while any(queues) or active:
                if not tripped and scheduler_stop is None:
                    for lane in range(2):
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
                                        "reason": "missing_actual_baseline_clean_source", "timestamp_unix": time.time()})
                                continue
                        number = state["attempts"][key] + 1
                        if number > runner.MAX_ATTEMPTS:
                            raise ValueError("lifetime attempt budget exceeded")
                        attempt_id = uuid.uuid4().hex
                        start = {**job, "attempt": number, "attempt_id": attempt_id,
                            "config_digest": digest, "model": runner.MODEL, "provider": runner.PROVIDER,
                            "model_version": runner.MODEL_VERSION,
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
                            # Track the durable start before a stop-write can fail.
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
        # Stop new submissions on a parent exception, but preserve in-flight outcomes.
        durable = {r["attempt_id"] for name in ("main_runs.jsonl", "run_errors.jsonl")
                   for r in runner.read_jsonl(output / name)}
        while active:
            # An interruption can land after fsync but before deque removal.
            if active[0][2]["attempt_id"] not in durable:
                finish(active[0])
            active.popleft()
        state = reconcile(output, jobs, digest)
        summary = runner.write_reports(output, jobs, state, blocked)
        summary["scheduler_stop"] = scheduler_stop
        runner._write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    try:
        args = parse_args(argv)
        if sys.platform != "linux":
            raise ValueError("formal results must be produced on the Linux server")
        tasks = runner.load_tasks(args.task_manifest)
        config = build_config(args, tasks)
        with runner.locked_workflow(), runner.locked_output(args.output_dir), runner.local_inference_transport():
            digest = runner.freeze_manifest(args.output_dir, config, resume=args.resume)
            runner.freeze_source_snapshot(args.output_dir, config["source_hashes"], root=ROOT, resume=args.resume)
            runner.recover_torn_journals(args.output_dir)
            verify_used_sources(tasks, config["jobs"], args.output_dir, digest)
            state = reconcile(args.output_dir, config["jobs"], digest)
            summary = runner.write_reports(args.output_dir, config["jobs"], state)
            if summary["circuit_breaker"]["tripped"]:
                return 1
            if state["pending"]:
                stop = runner.guard_paid_work(args.output_dir, digest, phase="startup")
                if stop is not None:
                    runner.write_reports(args.output_dir, config["jobs"], state, scheduler_stop=stop)
                    return 1
                _, preflight = runner.preflight_client(config["inference_settings"])
                runner.append_jsonl(args.output_dir / "preflight.jsonl", {**preflight,
                    "config_digest": digest, "python": platform.python_version(), "platform": platform.system(),
                    "resume": args.resume, "max_jobs": args.max_jobs, "parallel_execution": config["parallel_execution"]})
                with SpawnLanes() as pool:
                    summary = execute_jobs(tasks, config["jobs"], args.output_dir, digest,
                        config["inference_settings"], config["base_url"], pool=pool, max_jobs=args.max_jobs)
        return 0 if summary["status"] == "complete" else 1
    except Exception as exc:
        print(f"Parallel runner stopped ({type(exc).__name__}); inspect server-side journals. "
              "Exception text suppressed for credential safety.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
