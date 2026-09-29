"""Server-side real Qwen multi-state Shopping matrix; no mock mode or transfers.

Set LLM_PROVIDER=modelscope_local, LLM_MODEL=Qwen/Qwen3.8-27B,
LLM_BASE_URL to the loopback inference endpoint, and LLM_API_KEY explicitly.
--max-jobs limits jobs considered in this invocation, not the frozen matrix.
Resume can spend only the remainder of the two-attempt lifetime budget.
Three consecutive error attempts trip a journal-backed circuit breaker.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import fcntl
import hashlib
import json
import os
import platform
import stat
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from mas_faults import multimechanism_matrix as matrix
from mas_faults.llm_client import get_llm_client, openai_api_base
from mas_faults.shopping_mitigation import check_evidence

MODEL = "Qwen/Qwen3.8-27B"
PROVIDER = "modelscope_local"
MAX_ATTEMPTS = 2
MAX_CONSECUTIVE_ERRORS = 3
SOURCE_CONDITIONS = {"cross_task_replay", "contract_consistent_identity_corruption"}
JOURNALS = ("run_attempts.jsonl", "main_runs.jsonl", "run_errors.jsonl", "blocked_cells.jsonl")
REQUEST_FIELDS = ("request_index", "provider_request_id", "provider", "model", "json_mode",
                  "started_at", "completed_at", "latency_ms", "prompt_sha256", "response_sha256",
                  "prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens",
                  "prompt_cache_miss_tokens", "status", "error_type", "exception_chain")
PARTIAL_TRIAL_FIELDS = {"events": list, "judgments": list, "detection_events": list,
                        "recovery_events": list, "budget": dict, "graph_errors": list,
                        "action_protocol_events": list, "action_ledger_state": (dict, type(None)),
                        "action_ledger_events": list}


class SourceUnavailable(ValueError):
    """This cell cannot run until an eligible actual baseline source exists."""


def validate_loopback_url(value):
    try:
        parsed = urlsplit(value)
        valid = (isinstance(value, str) and not any(c.isspace() for c in value)
                 and parsed.scheme in {"http", "https"}
                 and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment
                 and (parsed.port is None or 1 <= parsed.port <= 65535))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("a credential-free HTTP(S) loopback URL is required")
    return value.rstrip("/")


def load_tasks(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks):
        raise ValueError("task manifest must be a JSON list or an object containing tasks")
    matrix.validate_tasks(tasks)
    return tasks


def inference_settings(environ=None):
    env = os.environ if environ is None else environ
    # Never let get_llm_client's legacy paid defaults choose an endpoint/model.
    if env.get("LLM_PROVIDER") != PROVIDER or env.get("LLM_MODEL") != MODEL:
        raise ValueError("explicit ModelScope-local Qwen/Qwen3.8-27B configuration required")
    base = validate_loopback_url(env.get("LLM_BASE_URL", ""))
    socket_timeout = int(env.get("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    total_timeout = int(env.get("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", str(socket_timeout)))
    max_tokens = int(env["LLM_MAX_TOKENS"]) if env.get("LLM_MAX_TOKENS") else None
    if min(socket_timeout, total_timeout) <= 0 or max_tokens is not None and max_tokens <= 0:
        raise ValueError("inference timeouts and token limit must be positive")
    return {"provider": PROVIDER, "model": MODEL, "api_base_url": openai_api_base(base),
            "temperature": 0, "max_tokens": max_tokens,
            "disable_thinking": env.get("LLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes"},
            "socket_timeout_seconds": socket_timeout, "total_timeout_seconds": total_timeout}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:17770")
    parser.add_argument("--required-model", choices=[MODEL], default=MODEL)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.max_jobs is not None and args.max_jobs < 1:
        parser.error("repetitions and max-jobs must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    return args


def build_config(args, tasks):
    manifest_bytes = args.task_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_tasks = manifest.get("tasks") if isinstance(manifest, dict) else manifest
    if manifest_tasks != tasks:
        raise ValueError("task manifest changed after loading")
    root = Path(__file__).resolve().parent
    sources = set(root.glob("*.py")) | set((root / "src").rglob("*.py"))
    audit_script = root / "scripts" / "audit_shopping_multimechanism.py"
    if audit_script.is_file():
        sources.add(audit_script)
    sources = sorted(sources)
    all_jobs = matrix.build_jobs(tasks, args.repetitions)
    selected = matrix.select_shard(all_jobs, args.shard_index, args.shard_count)
    return {"version": matrix.VERSION, "runner_schema": 1, "tasks": copy.deepcopy(tasks),
            "task_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "jobs": selected, "planned_runs": len(all_jobs), "shard_runs": len(selected),
            "base_url": validate_loopback_url(args.base_url), "model": MODEL, "provider": PROVIDER,
            "inference_settings": inference_settings(), "repetitions": args.repetitions,
            "shard_index": args.shard_index, "shard_count": args.shard_count,
            "max_attempts_per_job": MAX_ATTEMPTS,
            "max_consecutive_error_attempts": MAX_CONSECUTIVE_ERRORS,
            "source_policy": "frozen_actual_baseline_clean_different_task_and_product_v1",
            "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources}}


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value, *, exclusive=False):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path if exclusive else path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if not exclusive:
        temporary.replace(path)
    _sync_directory(path.parent)


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("journal records must be JSON objects")
    return rows


def append_jsonl(path, row):
    payload = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


@contextmanager
def _locked_file(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runner owns the workflow or output lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def locked_output(output):
    output.mkdir(parents=True, exist_ok=True)
    with _locked_file(output / ".runner.lock"):
        yield


@contextmanager
def locked_workflow(path=None):
    # One host-wide lock also covers localhost/IPv4/IPv6 aliases and new outputs.
    with _locked_file(Path(path) if path is not None else Path("/tmp/mas-shopping-multimechanism.lock")):
        yield


def freeze_manifest(output, config, *, resume):
    path = output / "matrix_manifest.json"
    digest = matrix.config_digest(config)
    expected = {"config": config, "config_digest": digest}
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("frozen configuration differs or --resume was not supplied")
    else:
        if resume or any(p.name != ".runner.lock" for p in output.iterdir()):
            raise ValueError("manifest missing: use a new empty output directory")
        _write_json(path, expected, exclusive=True)
    return digest


def freeze_source_snapshot(output, source_hashes, *, root, resume):
    """Freeze listed Python bytes once; resume only verifies, never repairs."""
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("nonempty frozen source hashes required")
    paths = {}
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("invalid source snapshot path")
        relative = Path(name)
        if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
                or relative.suffix != ".py" or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("source snapshot requires canonical relative Python paths and SHA-256 hashes")
        paths[name] = relative

    def contained_path(base, relative):
        current = base
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("source snapshot symlinks are forbidden")
        if not current.resolve().is_relative_to(base):
            raise ValueError("source snapshot path escapes its root")
        return current

    def checked_bytes(base, relative, digest):
        path = contained_path(base, relative)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("source snapshot entries must be regular files")
            content = handle.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("source snapshot hash mismatch")
        return content

    try:
        root = Path(root).resolve(strict=True)
        output = Path(output).resolve(strict=True)
        snapshot = contained_path(output, Path("source_snapshot"))
        if resume:
            if not snapshot.is_dir():
                raise ValueError("source snapshot missing on resume")
        elif snapshot.exists():
            raise ValueError("source snapshot already exists; refusing overwrite")

        # Validate every live source before creating any snapshot files.
        contents = {name: checked_bytes(root, relative, source_hashes[name])
                    for name, relative in paths.items()}
        if not resume:
            snapshot.mkdir()
            _sync_directory(output)
            for name, relative in paths.items():
                parent = snapshot
                for part in relative.parts[:-1]:
                    child = contained_path(snapshot, (parent / part).relative_to(snapshot))
                    if not child.exists():
                        child.mkdir()
                        _sync_directory(parent)
                    parent = child
                target = contained_path(snapshot, relative)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(contents[name])
                    handle.flush()
                    os.fsync(handle.fileno())
                _sync_directory(parent)

        actual = set()
        for path in snapshot.rglob("*"):
            relative = path.relative_to(snapshot)
            contained_path(snapshot, relative)
            if not path.is_dir():
                actual.add(relative.as_posix())
        if actual != set(paths):
            raise ValueError("source snapshot has missing or unlisted files")
        for name, relative in paths.items():
            checked_bytes(snapshot, relative, source_hashes[name])
        return snapshot
    except OSError as exc:
        raise ValueError("source snapshot could not be frozen or verified") from exc


def recover_torn_journals(output):
    """Under both locks, preserve partial final writes before repairing the tail."""
    for name in JOURNALS:
        path = output / name
        if not path.is_file():
            continue
        original = path.read_bytes()
        lines = original.splitlines(keepends=True)
        repaired = original
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError("journal corruption requires manual audit") from exc
                repaired = b"".join(lines[:index])
        if repaired and not repaired.endswith(b"\n"):
            repaired += b"\n"
        if repaired == original:
            continue
        suffix = uuid.uuid4().hex
        backup = path.with_name(name + ".torn-" + suffix)
        temporary = path.with_name(name + ".repair-" + suffix)
        for target, content in ((backup, original), (temporary, repaired)):
            with target.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(path)
        _sync_directory(output)


def reconcile_attempts(output, jobs, digest):
    starts = read_jsonl(output / "run_attempts.jsonl")
    rows = read_jsonl(output / "main_runs.jsonl")
    errors = read_jsonl(output / "run_errors.jsonl")
    expected = {j["job_key"]: j for j in jobs}
    indexed, finished, counts = {}, {}, Counter()
    for kind, records in (("start", starts), ("row", rows), ("error", errors)):
        for record in records:
            key, number = record.get("job_key"), record.get("attempt")
            if record.get("config_digest") != digest or key not in expected:
                raise ValueError("resume configuration or job does not match frozen matrix")
            if type(number) is not int or not 1 <= number <= MAX_ATTEMPTS or not record.get("attempt_id"):
                raise ValueError("invalid attempt identifier or lifetime attempt budget")
            if any(record.get(field) != value for field, value in expected[key].items()):
                raise ValueError("journal job metadata differs from frozen matrix")
            identity = (key, number)
            if kind == "start":
                if identity in indexed:
                    raise ValueError("duplicate attempt start")
                indexed[identity] = record
                counts[key] += 1
            else:
                if identity in finished or identity not in indexed:
                    raise ValueError("duplicate terminal record or missing attempt start")
                if record["attempt_id"] != indexed[identity]["attempt_id"]:
                    raise ValueError("terminal record bound to a different attempt")
                finished[identity] = record
    for key, count in counts.items():
        if {n for k, n in indexed if k == key} != set(range(1, count + 1)):
            raise ValueError("attempt numbers must be contiguous")
    run_ids = [row["run_id"] for row in rows if row.get("run_id")]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate runtime run IDs")
    completed = {row["job_key"]: row["attempt"] for row in rows}
    if len(completed) != len(rows):
        raise ValueError("duplicate completed job")
    if any(number > completed.get(key, MAX_ATTEMPTS) for key, number in indexed):
        raise ValueError("an attempt was started after successful completion")
    for identity, start in indexed.items():
        if identity not in finished:
            error = {**start, "error_type": "InterruptedAttempt", "status": "infra_error",
                     "known_prompt_tokens": 0, "known_completion_tokens": 0, "known_total_tokens": 0,
                     "model_requests": [], "usage_complete": False, "timestamp_unix": time.time()}
            append_jsonl(output / "run_errors.jsonl", error)
            errors.append(error)
    pending = matrix.pending_jobs(jobs, rows, errors, digest, max_attempts=MAX_ATTEMPTS)
    exhausted = [j["job_key"] for j in jobs if j["job_key"] not in completed
                 and counts[j["job_key"]] >= MAX_ATTEMPTS]
    error_attempts = {(r["job_key"], r["attempt"]) for r in errors}
    consecutive = 0
    # Start-journal order, not wall-clock timestamps, survives process interruption.
    for start in reversed(starts):
        if (start["job_key"], start["attempt"]) not in error_attempts:
            break
        consecutive += 1
    return {"rows": rows, "errors": errors, "attempts": counts, "pending": pending,
            "exhausted": exhausted, "consecutive_error_attempts": consecutive}


def _eligible_source(row, job, task_map, digest):
    source_task = task_map.get(row.get("task_id"))
    target_task = task_map[job["task_id"]]
    envelope = row.get("source_evidence")
    if (not source_task or row.get("config_digest") != digest or row.get("condition") != "clean"
            or row.get("arm") != "baseline" or row.get("final_task_success") is not True
            or row.get("environment_task_success") is not True or not row.get("run_id")
            or source_task["task_id"] == target_task["task_id"]
            or source_task["product_title"] == target_task["product_title"]
            or urlsplit(source_task["product_url"]).path == urlsplit(target_task["product_url"]).path
            or not isinstance(envelope, dict) or envelope.get("task_id") != source_task["task_id"]
            or envelope.get("source") != "Worker"
            or any(not envelope.get(k) for k in ("evidence_id", "session_id", "entity_id", "action_id", "source"))
            or type(envelope.get("version")) is not int):
        return False
    payload, truth = envelope.get("payload"), row.get("environment_state")
    return bool(isinstance(payload, dict) and isinstance(truth, dict)
                and check_evidence(source_task, payload).valid and truth.get("cart_verified") is True
                and all(payload.get(k) == truth.get(k) for k in
                        ("task_id", "product_title", "product_id", "sku", "observed_quantity")))


def freeze_cross_task_source(rows, job, tasks, output, digest):
    task_map = {t["task_id"]: t for t in tasks}
    target = [job["task_id"], job["topology"], job["repeat_index"]]
    path = output / "cross_task_sources" / (matrix.config_digest(target) + ".json")
    candidates = [r for r in rows if _eligible_source(r, job, task_map, digest)]
    target_clean = [r for r in rows if r.get("task_id") == job["task_id"]
                    and r.get("condition") == "clean" and r.get("arm") == "baseline"]
    target_payloads = [r["source_evidence"]["payload"] for r in target_clean
                       if isinstance(r.get("source_evidence"), dict)
                       and isinstance(r["source_evidence"].get("payload"), dict)]
    candidates = [r for r in candidates if not any(
        p.get("sku") == r["source_evidence"]["payload"].get("sku")
        or p.get("product_id") == r["source_evidence"]["payload"].get("product_id")
        for p in target_payloads)]

    def frozen_value(row):
        return {"config_digest": digest, "target": target, "source_task_id": row["task_id"],
                "source_run_id": row["run_id"], "source_job_key": row["job_key"],
                "envelope": copy.deepcopy(row["source_evidence"]),
                "envelope_sha256": matrix.config_digest(row["source_evidence"])}

    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if not any(result == frozen_value(row) for row in candidates):
            raise ValueError("frozen source no longer matches actual clean records")
    else:
        if not candidates:
            raise SourceUnavailable("missing successful baseline clean source from a different task and product")
        source = min(candidates, key=lambda r: (r["topology"] != job["topology"],
                     r["repeat_index"] != job["repeat_index"], r["task_id"], r["job_key"], r["run_id"]))
        result = frozen_value(source)
        path.parent.mkdir(exist_ok=True)
        _write_json(path, result, exclusive=True)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = str(path.relative_to(output))
    for record in read_jsonl(output / "run_attempts.jsonl"):
        previous = record.get("cross_task_source") or {}
        if previous.get("file") == relative and previous.get("file_sha256") != file_hash:
            raise ValueError("frozen source file hash differs from prior attempts")
    return {**result, "file": relative, "file_sha256": file_hash}


def usage_snapshot(client):
    return {"prompt_tokens": client.prompt_tokens, "completion_tokens": client.completion_tokens,
            "call_count": client.call_count, "log_length": len(client.request_log)}


def usage_since(client, before, *, complete):
    requests = [{k: copy.deepcopy(v) for k, v in row.items() if k in REQUEST_FIELDS}
                for row in client.request_log[before["log_length"]:]]
    prompt = client.prompt_tokens - before["prompt_tokens"]
    completion = client.completion_tokens - before["completion_tokens"]
    calls = client.call_count - before["call_count"]
    totals = {}
    for field, known in (("prompt_tokens", prompt), ("completion_tokens", completion)):
        values = [r.get(field) for r in requests]
        totals[field] = known if len(requests) == calls and all(
            type(v) is int and v >= 0 for v in values) and sum(values) == known else None
    usage_known = all(value is not None for value in totals.values())
    # A failed attempt may have fully known billing, but is not certified complete.
    return {**totals, "total_tokens": prompt + completion if usage_known else None,
            "known_prompt_tokens": prompt, "known_completion_tokens": completion,
            "known_total_tokens": prompt + completion, "model_calls": calls,
            "model_requests": requests, "usage_complete": bool(complete and usage_known)}


def partial_trial_context(exc):
    """Copy only runtime-owned trace fields, never the exception's message/attrs."""
    partial = getattr(exc, "partial_trial", None)
    if not isinstance(partial, dict):
        return {}
    context = {}
    for field, expected_type in PARTIAL_TRIAL_FIELDS.items():
        if field not in partial:
            continue
        value = partial.get(field)
        if not isinstance(value, expected_type):
            continue
        try:
            context[field] = json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError, RecursionError):
            # One malformed trace field must not suppress the error journal.
            continue
    return context


def is_timeout_exception(exc):
    """Inspect bounded exception links by type, never messages or string reasons."""
    seen = set()
    for _ in range(32):
        if not isinstance(exc, BaseException) or id(exc) in seen:
            return False
        seen.add(id(exc))
        if isinstance(exc, TimeoutError):
            return True
        if exc.__cause__ is not None:
            exc = exc.__cause__
        elif not exc.__suppress_context__ and exc.__context__ is not None:
            exc = exc.__context__
        elif isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
            exc = exc.reason
        else:
            return False
    return False


def exposure_counts(rows, errors):
    """Describe recorded boundary exposure, never infer an A symptom from a plan."""
    counts = dict.fromkeys(("clean_runs", "planned_fault_runs", "injection_recorded_runs",
                           "exposed_fault_runs", "unreached_fault_runs", "prior_contract_failure_runs",
                           "unknown_fault_exposure_runs", "error_attempts_exposure_unknown"), 0)
    for row in rows:
        if row["condition"] == "clean":
            counts["clean_runs"] += 1
            continue
        counts["planned_fault_runs"] += 1
        events = row.get("fault_events")
        recorded = False
        if isinstance(events, list) and len(events) == 1 and isinstance(events[0], dict):
            event = events[0]
            delivered = event.get("delivered_sha256")
            hashes = [event.get("original_sha256"), *(delivered if isinstance(delivered, list) else [])]
            recorded = (event.get("condition") == row["condition"]
                        and event.get("boundary") == matrix.CELLS.get(row["condition"])
                        and event.get("boundary") is not None
                        and event.get("applied") is not False and event.get("reached") is not False
                        and isinstance(delivered, list) and type(event.get("delivered_count")) is int
                        and event["delivered_count"] == len(delivered)
                        and all(isinstance(h, str) and len(h) == 64
                                and all(c in "0123456789abcdefABCDEF" for c in h) for h in hashes))
        counts["injection_recorded_runs"] += int(recorded)
        if row.get("action_contract_valid") is False:
            category = "prior_contract_failure_runs"
        elif events == []:
            category = "unreached_fault_runs"
        elif recorded and row.get("action_contract_valid") is True:
            category = "exposed_fault_runs"
        else:
            category = "unknown_fault_exposure_runs"
        counts[category] += 1
    counts["error_attempts_exposure_unknown"] = sum(r.get("condition") != "clean" for r in errors)
    return counts


def nullable_token_sum(rows, field):
    values = [row.get(field) for row in rows]
    return sum(values) if all(type(value) is int and value >= 0 for value in values) else None


def write_reports(output, jobs, state, source_blocked=()):
    rows, errors = state["rows"], state["errors"]
    done = {r["job_key"] for r in rows}
    groups, pairs = defaultdict(list), defaultdict(dict)
    for row in rows:
        groups[row["arm"]].append(row)
        pairs[row["pair_key"]][row["arm"]] = row
    summary = {"version": matrix.VERSION, "status": "complete" if len(done) == len(jobs) else "incomplete",
               "planned_runs": len(jobs), "completed_runs": len(rows), "error_attempts": len(errors),
               "unrun_job_keys": [j["job_key"] for j in jobs if j["job_key"] not in done],
               "exhausted_job_keys": state["exhausted"], "source_blocked_job_keys": list(source_blocked),
               "total_tokens_completed": nullable_token_sum(rows, "total_tokens"),
               "known_tokens_completed": sum(r.get("known_total_tokens", 0) for r in rows),
               "known_tokens_error_attempts": sum(r.get("known_total_tokens", 0) for r in errors),
               "usage_incomplete_attempts": sum(r.get("usage_complete") is not True for r in rows + errors),
               "complete_seven_arm_pairs": sum(set(matrix.ARMS).issubset(p) for p in pairs.values()),
               "circuit_breaker": {"threshold": MAX_CONSECUTIVE_ERRORS,
                   "consecutive_error_attempts": state["consecutive_error_attempts"],
                   "tripped": state["consecutive_error_attempts"] >= MAX_CONSECUTIVE_ERRORS},
               "exposure": exposure_counts(rows, errors),
               "by_arm": {}}
    for arm in matrix.ARMS:
        values = groups[arm]
        summary["by_arm"][arm] = {"runs": len(values), "final_task_success": sum(
            r.get("final_task_success") is True for r in values), "environment_task_success": sum(
            r.get("environment_task_success") is True for r in values), "decision_correct": sum(
            r.get("decision_correct") is True for r in values), "error_attempts": sum(
            r["arm"] == arm for r in errors), "total_tokens": nullable_token_sum(values, "total_tokens"),
            "known_total_tokens": sum(r.get("known_total_tokens", 0) for r in values),
            "evidence_acceptance_errors": sum(r.get("evidence_acceptance_errors", 0) for r in values),
            "exposure": exposure_counts(values, [r for r in errors if r["arm"] == arm])}
    _write_json(output / "summary.json", summary)
    fields = ("job_key", "pair_key", "task_id", "topology", "condition", "arm", "repeat_index",
              "run_id", "attempt_id", "attempt", "config_digest", "status", "environment_task_success",
              "decision_correct", "final_task_success", "evidence_acceptance_errors", "total_tokens",
              "known_total_tokens", "usage_complete", "model_calls", "latency_ms", "error_type")
    for name, records in (("main_runs.csv", rows), ("run_errors.csv", errors)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    lines = ["# 多机制 Shopping 实验汇总", "", f"状态：{summary['status']}。",
             f"计划 {len(jobs)} 条，完成 {len(rows)} 条，错误尝试 {len(errors)} 次。",
             f"缺少真实 clean 源阻塞 {len(source_blocked)} 条；尝试耗尽 {len(state['exhausted'])} 条。", "",
             "| 策略 | 完成 | 环境成功 | 决策正确 | 最终成功 | 错误尝试 | tokens |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, stats in summary["by_arm"].items():
        lines.append(f"| {arm} | {stats['runs']} | {stats['environment_task_success']} | "
                     f"{stats['decision_correct']} | {stats['final_task_success']} | "
                     f"{stats['error_attempts']} | {stats['total_tokens']} |")
    lines += ["", f"错误尝试已知 tokens：{summary['known_tokens_error_attempts']}。未知用量不按零消耗解释。",
              f"连续错误尝试：{state['consecutive_error_attempts']}；达到 {MAX_CONSECUTIVE_ERRORS} 次即熔断，续跑不重置。",
              f"计划故障行：{summary['exposure']['planned_fault_runs']}；记录到完整注入：{summary['exposure']['injection_recorded_runs']}；"
              f"契约有效且边界已暴露：{summary['exposure']['exposed_fault_runs']}。",
              f"未到达注入边界：{summary['exposure']['unreached_fault_runs']}；先前动作契约失败：{summary['exposure']['prior_contract_failure_runs']}；"
              f"暴露未知：{summary['exposure']['unknown_fault_exposure_runs']}。",
              "注入记录或计划条件不直接算 A 症状；暴露统计不改写运行时评价。",
              "重复编号不是可控模型 seed；错误尝试、未解决状态和安全拒绝不等同于任务成功。",
              "这是当前分片的基本进度汇总，正式配对与共同 clean 子集统计由独立审计完成。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


async def execute_jobs(tasks, jobs, output, digest, client, base_url, *, max_jobs=None,
                       trial=None, executor_factory=None):
    """Run under both locks. Dependency injection is for offline tests only."""
    state = reconcile_attempts(output, jobs, digest)
    if state["consecutive_error_attempts"] >= MAX_CONSECUTIVE_ERRORS:
        return write_reports(output, jobs, state)
    if trial is None:
        from mas_faults.shopping_multimechanism import run_trial
        trial = run_trial
    if executor_factory is None:
        from mas_faults.shopping_action_protocol import MultiStateShoppingExecutor
        executor_factory = MultiStateShoppingExecutor
    task_map = {t["task_id"]: t for t in tasks}
    pending = state["pending"] if max_jobs is None else state["pending"][:max_jobs]
    blocked = []
    for job in pending:
        source = None
        if job["condition"] in SOURCE_CONDITIONS:
            try:
                source = freeze_cross_task_source(state["rows"], job, tasks, output, digest)
            except SourceUnavailable:
                blocked.append(job["job_key"])
                known = read_jsonl(output / "blocked_cells.jsonl")
                if not any(r.get("pair_key") == job["pair_key"] for r in known):
                    append_jsonl(output / "blocked_cells.jsonl", {"pair_key": job["pair_key"],
                        "config_digest": digest, "reason": "missing_actual_baseline_clean_source",
                        "timestamp_unix": time.time()})
                continue
        for number in range(state["attempts"][job["job_key"]] + 1, MAX_ATTEMPTS + 1):
            attempt_id = uuid.uuid4().hex
            ledger_path = output / "action_ledgers" / (attempt_id + ".sqlite3")
            ledger_path.parent.mkdir(exist_ok=True)
            before = usage_snapshot(client)
            start = {**job, "attempt": number, "attempt_id": attempt_id, "config_digest": digest,
                     "model": MODEL, "provider": PROVIDER, "ledger_path": str(ledger_path.relative_to(output)),
                     "cross_task_source": source, "timestamp_unix": time.time()}
            # A durable start precedes constructor, model calls and environment writes.
            append_jsonl(output / "run_attempts.jsonl", start)
            started = time.perf_counter()
            executor = None
            try:
                executor = executor_factory(base_url)
                result = await trial(copy.deepcopy(task_map[job["task_id"]]),
                    {**job, "attempt": number, "attempt_id": attempt_id}, executor, client,
                    cross_task_evidence=copy.deepcopy(source["envelope"]) if source else None,
                    ledger_path=ledger_path)
                if not isinstance(result, dict):
                    raise TypeError("run_trial must return a dict")
                result = {**result, **start, **usage_since(client, before, complete=True),
                          "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
                if not result.get("run_id"):
                    result["run_id"] = attempt_id
                # Validate serialization while errors can still be journaled.
                json.dumps(result, allow_nan=False)
            except BaseException as exc:
                error = {**start, **usage_since(client, before, complete=False),
                         "error_type": type(exc).__name__, "status": "timeout" if is_timeout_exception(exc)
                         else "infra_error", "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                         "partial_trial": partial_trial_context(exc),
                         "http_receipts": copy.deepcopy(getattr(executor, "http_receipts", []))}
                append_jsonl(output / "run_errors.jsonl", error)
                state = reconcile_attempts(output, jobs, digest)
                write_reports(output, jobs, state, blocked)
                print(f"ERROR attempt={number} type={type(exc).__name__}", flush=True)
                if not isinstance(exc, Exception):
                    raise
                if state["consecutive_error_attempts"] >= MAX_CONSECUTIVE_ERRORS:
                    print("STOP circuit breaker: three consecutive error attempts", flush=True)
                    return write_reports(output, jobs, state, blocked)
            else:
                append_jsonl(output / "main_runs.jsonl", result)
                print(f"DONE attempt={number} success={result.get('final_task_success') is True}", flush=True)
                break
        state = reconcile_attempts(output, jobs, digest)
        write_reports(output, jobs, state, blocked)
    return write_reports(output, jobs, reconcile_attempts(output, jobs, digest), blocked)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("local inference redirects are forbidden")


@contextmanager
def local_inference_transport():
    # The shared client uses urllib.urlopen; prevent proxy or redirect egress.
    old = urllib.request._opener
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()))
    try:
        yield
    finally:
        urllib.request._opener = old


def preflight_client(settings):
    if inference_settings() != settings or not os.environ.get("LLM_API_KEY"):
        raise ValueError("explicit local inference settings and LLM_API_KEY required")
    client = get_llm_client()
    info = client.model_info
    if (info.provider != PROVIDER or info.model != MODEL or info.client_type != "http_openai_compatible"
            or openai_api_base(validate_loopback_url(info.base_url)) != settings["api_base_url"]):
        raise ValueError("client does not match the frozen real local model")
    started = time.time()
    with urllib.request.urlopen(settings["api_base_url"] + "/models", timeout=10) as response:
        served = json.load(response)
    if MODEL not in {item.get("id") for item in served.get("data", []) if isinstance(item, dict)}:
        raise ValueError("required model is absent from local inference server")
    return client, {"model": MODEL, "provider": PROVIDER, "timestamp_unix": started,
                    "served_models_sha256": matrix.config_digest(served), "model_calls": 0}


def main(argv=None):
    args = parse_args(argv)
    try:
        if sys.platform != "linux":
            raise ValueError("real results must be produced on the Linux server, not downloaded to this host")
        tasks = load_tasks(args.task_manifest)
        config = build_config(args, tasks)
        with locked_workflow(), locked_output(args.output_dir), local_inference_transport():
            digest = freeze_manifest(args.output_dir, config, resume=args.resume)
            freeze_source_snapshot(args.output_dir, config["source_hashes"],
                                   root=Path(__file__).resolve().parent, resume=args.resume)
            recover_torn_journals(args.output_dir)
            state = reconcile_attempts(args.output_dir, config["jobs"], digest)
            summary = write_reports(args.output_dir, config["jobs"], state)
            if summary["circuit_breaker"]["tripped"]:
                return 1
            if not state["pending"]:
                return 0 if summary["status"] == "complete" else 1
            client, preflight = preflight_client(config["inference_settings"])
            append_jsonl(args.output_dir / "preflight.jsonl", {**preflight, "config_digest": digest,
                         "python": platform.python_version(), "platform": platform.system(),
                         "resume": args.resume, "max_jobs": args.max_jobs})
            summary = asyncio.run(execute_jobs(tasks, config["jobs"], args.output_dir, digest, client,
                                              config["base_url"], max_jobs=args.max_jobs))
        if (summary["circuit_breaker"]["tripped"] or summary["exhausted_job_keys"]
                or summary["source_blocked_job_keys"]):
            return 1
        return 0 if summary["status"] == "complete" or args.max_jobs is not None else 1
    except Exception as exc:
        # HTTP exceptions may contain credentials or raw backend responses.
        print(f"Runner stopped ({type(exc).__name__}); inspect configuration and server-side journals. "
              "Exception text suppressed for credential safety.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
