"""Paired, single-injection receiver-mitigation pilot. No simulated experiment mode."""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import fcntl
import hashlib
import json
import os
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from mas_faults.llm_client import get_base_url, get_llm_client, openai_api_base
from mas_faults.shopping_mitigation import EvidencePolicy, MODES, check_evidence
from mas_faults.webarena_task_selection import load_task_manifest
from run_webarena_architecture_rq2 import run_one

VERSION = "shopping-receiver-mitigation-pilot-v2-shared-recovery"
TOPOLOGIES = ("sequential", "flat", "hierarchical")
FAULTS = ("none", "omission", "message_corruption", "valid_partial", "stale_replay")
TASK_FIELDS = ("task_id", "product_title", "product_url", "quantity")


def model_listing_url(base_url):
    return openai_api_base(base_url) + "/models"


def inference_settings():
    socket_timeout = int(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    return {"api_base_url": openai_api_base(get_base_url()), "temperature": 0,
            "max_tokens": int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else None,
            "disable_thinking": os.environ.get("LLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes"},
            "socket_timeout_seconds": socket_timeout,
            "total_timeout_seconds": int(os.environ.get("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", socket_timeout))}


def build_jobs(tasks, repetitions):
    if not tasks or repetitions < 1:
        raise ValueError("tasks and positive repetitions required")
    ids = [task["task_id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task identifiers")
    jobs = []
    for fi, fault in enumerate(FAULTS):
        for ti, task in enumerate(tasks):
            for ai, topology in enumerate(TOPOLOGIES):
                for repeat in range(1, repetitions + 1):
                    pair = json.dumps([task["task_id"], topology, fault, 4, repeat], separators=(",", ":"))
                    offset = (fi + ti + ai + repeat - 1) % len(MODES)
                    for mode in MODES[offset:] + MODES[:offset]:
                        jobs.append({"pair_key": pair, "job_key": pair + ":" + mode,
                                     "task_id": task["task_id"], "topology": topology,
                                     "fault": fault, "injection_step": 4,
                                     "repeat_index": repeat, "mitigation_mode": mode})
    return jobs


def freeze_stale_source(rows, job, output):
    key = [job["task_id"], job["topology"], job["repeat_index"]]
    name = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20] + ".json"
    directory = output / "stale_sources"
    directory.mkdir(exist_ok=True)
    path = directory / name
    candidates = [r for r in rows if r.get("fault_type") == "none" and r.get("mitigation_mode") == "baseline"
                  and r.get("final_task_success") and r.get("task_id") != job["task_id"]
                  and r.get("topology") == job["topology"] and r.get("repeat_index") == job["repeat_index"]]
    eligible = []
    for candidate in candidates:
        event = next((e for e in candidate["events"] if e["step_index"] == 4 and e["source_agent"] == "Shopping Worker"), None)
        truth = candidate.get("environment_state", {})
        source_task = {"task_id": candidate["task_id"], "product_title": truth.get("product_title"),
                       "quantity": truth.get("requested_quantity")}
        if event and truth.get("cart_verified") is True and _matches_environment(source_task, event.get("original_message"), truth):
            eligible.append((candidate, event))
    if not eligible:
        raise ValueError("stale replay requires a successful real clean from a different task")
    source, event = sorted(eligible, key=lambda pair: (pair[0]["task_id"], pair[0]["run_id"]))[0]
    payload = copy.deepcopy(event["original_message"])
    if not isinstance(payload, dict) or payload.get("task_id") != source["task_id"]:
        raise ValueError("clean source worker message is not task-bound")
    result = {"source_run_id": source["run_id"], "source_message_id": event["message_id"],
              "source_task_id": source["task_id"], "model": source["model"], "provider": source["provider"],
              "payload": payload, "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    if path.is_file():
        if json.loads(path.read_text()) != result:
            raise ValueError("frozen stale source differs from clean records")
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def _matches_environment(task, payload, environment):
    return bool(check_evidence(task, payload).valid and isinstance(environment, dict)
                and all(payload.get(key) == environment.get(key) for key in
                        ("task_id", "product_title", "product_id", "sku", "observed_quantity")))


def audit_outcome(row, task):
    """Offline evaluator; environment truth is never passed to EvidencePolicy."""
    result = copy.deepcopy(row)
    result["legacy_evaluation"] = copy.deepcopy({key: row.get(key) for key in
        ("final_answer", "final_task_success", "observed_M_consequence", "recovery_detected", "recovery_type", "propagation_class")})
    primary, decision = row.get("primary_evidence"), row.get("decision_evidence")
    truth = row.get("environment_state", {})
    final_verdict = row["final_answer"]["verdict"]
    verification = row.get("verification_verdict")
    events = row.get("mitigation_events", [])
    common_events = row.get("common_recovery_events", [])
    final_receiver = {"sequential": "Coordinator", "flat": "Coordinator",
                      "hierarchical": "Supervisor"}.get(row.get("topology"))

    def bound_decision(verdict, expected):
        return (isinstance(verdict, dict) and verdict.get("task_id") == task["task_id"]
                and verdict.get("decision") == expected)

    def raw_acceptance(verdict):
        return isinstance(verdict, dict) and verdict.get("decision") == "accept"

    def expected_for(payload):
        return "accept" if _matches_environment(task, payload, truth) and truth.get("cart_verified") is True else "reject"

    def recovery_requested(verdict):
        return (isinstance(verdict, dict) and verdict.get("decision") == "reject"
                or not isinstance(verdict, dict) or verdict.get("task_id") != task["task_id"])

    def linked_readback_receipts(event):
        indices, ledger = event.get("readback_receipt_indices"), row.get("http_receipts", [])
        if (not isinstance(indices, list) or len(indices) != 2 or not isinstance(ledger, list)
                or any(type(index) is not int or not 0 <= index < len(ledger) for index in indices)
                or indices[1] != indices[0] + 1):
            return []
        receipts = [ledger[index] for index in indices]
        for index, receipt, purpose in zip(indices, receipts, ("reobserve_cart.product_page", "reobserve_cart.items")):
            if not isinstance(receipt, dict):
                return []
            digest, status = receipt.get("response_sha256"), receipt.get("status_code")
            if (receipt.get("receipt_index") != index or receipt.get("purpose") != purpose
                    or receipt.get("request_method") != "GET" or type(status) is not int or not 200 <= status < 300
                    or not isinstance(receipt.get("request_url"), str) or not receipt["request_url"]
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdefABCDEF" for character in digest)):
                return []
        return receipts

    def forwarded_to_final(payload):
        for event in row.get("events", []):
            if (event.get("step_index") != 5 or event.get("source_agent") != row.get("primary_receiver")
                    or event.get("target_agent") != final_receiver):
                continue
            messages = event.get("delivered_message")
            if messages is None and event.get("fault_applied") is False and event.get("fault_type") == "none":
                messages = [event.get("original_message")]
            if isinstance(messages, list) and any(isinstance(message, dict) and message.get("evidence") == payload
                                                  for message in messages):
                return True
        return False

    m, semantic = set(), set()
    accepted_inputs = []
    # Audit historical acceptances independently of the eventual repaired state.
    stages = [(primary, decision, verification, final_verdict)]
    stages += [(event.get("before_primary_evidence"), event.get("before_decision_evidence"),
                event.get("before_verification_verdict"), event.get("before_verdict"))
               for event in common_events]
    for stage_primary, stage_decision, stage_verification, stage_verdict in stages:
        if raw_acceptance(stage_verification):
            accepted_inputs.append(stage_primary)
        if raw_acceptance(stage_verdict):
            accepted_inputs.append(stage_decision)
        if stage_verification is not None and not bound_decision(stage_verification, expected_for(stage_primary)):
            semantic.add("incorrect_verification")
    for payload in accepted_inputs:
        check = check_evidence(task, payload)
        if check.missing_fields:
            m.add("M14_partial_tool_or_message_result_acceptance")
            semantic.add("partial_result_as_complete")
        if isinstance(payload, dict) and payload.get("task_id") not in (None, task["task_id"]):
            m.add("M5_stale_context_acceptance")
            semantic.add("stale_belief_acceptance")
        inconsistent = any(issue.startswith("inner_outer:") for issue in check.issues)
        if isinstance(payload, dict) and payload.get("cart_verified") is True:
            inconsistent |= (type(payload.get("requested_quantity")) is int
                             and type(payload.get("observed_quantity")) is int
                             and payload["requested_quantity"] != payload["observed_quantity"])
        if inconsistent:
            m.add("M6_state_inconsistency")
            semantic.add("state_inconsistency")
        if any(issue.startswith("constraint:") for issue in check.issues):
            semantic.add("constraint_loss")
    expected_decision = expected_for(decision)
    final_correct = bound_decision(final_verdict, expected_decision)
    success = expected_decision == "accept" and final_correct
    if not final_correct:
        m.add("M4_incorrect_collective_decision")
    if not success and check_evidence(task, decision).missing_fields:
        m.add("M3_incomplete_information_aggregation")
        semantic.add("evidence_omission")
    if not success:
        m.add("M2_task_timeout_or_failure")
    receiver_acceptances = []
    if bound_decision(verification, "accept"):
        receiver_acceptances.append((row.get("primary_receiver"), primary))
    if bound_decision(final_verdict, "accept") and row.get("topology") != "sequential":
        receiver_acceptances.append((row.get("decision_evidence_receiver"), decision))

    # Sequential forwarding is a use of the repaired primary, not a flat bypass.
    if (row.get("topology") == "sequential" and bound_decision(final_verdict, "accept")
            and row.get("decision_evidence_receiver") == final_receiver
            and primary == decision and forwarded_to_final(decision)):
        receiver_acceptances.append((row.get("primary_receiver"), decision))

    recovered_events = []
    readbacks = []
    for index, event in enumerate(events):
        if event.get("readback_called") is True:
            readbacks.append(index)
            source_index = index
        else:
            source_index = event.get("cached_readback_event_index")
        source = (events[source_index] if type(source_index) is int and 0 <= source_index <= index
                  and (source_index < index or event.get("readback_called") is True) else {})
        fresh = (readbacks == [source_index] and source.get("readback_called") is True
                 and not source.get("readback_error") and linked_readback_receipts(source)
                 and source.get("readback_response") == event.get("after"))
        if (success and not common_events and fresh and event.get("before_issues")
                and event.get("replacement_used") is True and not event.get("after_issues")
                and any(event.get("receiver") == receiver and receiver is not None
                        and event.get("after") == accepted for receiver, accepted in receiver_acceptances)
                and _matches_environment(task, event.get("after"), truth)):
            recovered_events.append(index)
    if len(readbacks) > 1:
        recovered_events = []

    common_recovered_events = []
    common_attempted = any(recovery_requested(event.get("before_verdict"))
                           and event.get("readback_called") is True for event in common_events)
    for index, event in enumerate(common_events):
        receipts = linked_readback_receipts(event)
        prior_receipt_indices = [receipt_index for prior in events
                                 for receipt_index in (prior.get("readback_receipt_indices") or [])
                                 if type(receipt_index) is int]
        if (success and len(common_events) == 1 and recovery_requested(event.get("before_verdict"))
                and all(key in event for key in ("before_verdict", "before_decision_evidence",
                                                "before_primary_evidence", "before_verification_verdict"))
                and event.get("readback_called") is True and event.get("replacement_used") is True
                and receipts and event.get("readback_receipts") == receipts
                and (not prior_receipt_indices or event["readback_receipt_indices"][0] > max(prior_receipt_indices))
                and bound_decision(event.get("after_verdict"), "accept")
                and event.get("after_verdict") == final_verdict
                and event.get("readback") == decision and event.get("receiver") == final_receiver
                and row.get("decision_evidence_receiver") == final_receiver and final_receiver is not None
                and _matches_environment(task, event.get("readback"), truth)):
            common_recovered_events.append(index)

    mitigation_recovered = bool(recovered_events)
    common_recovered = bool(common_recovered_events)
    recovered = mitigation_recovered or common_recovered
    detected = any(event.get("triggered") for event in events)
    if recovered:
        propagation = "detected_and_recovered"
    elif (detected or common_attempted) and not success:
        propagation = "detected_but_unrecovered"
    elif m and not success:
        propagation = "propagated_to_M_final_failure"
    elif m:
        propagation = "silent_propagation_to_M"
    else:
        propagation = "exposed_at_A_only" if row.get("fault_type") != "none" else "clean"
    result.update(evaluator_version=VERSION, final_task_success=success, task_score=float(success),
                  final_decision_correct=final_correct, observed_M_consequence=sorted(m) or ["none"],
                  semantic_consequences=sorted(semantic) or ["none"],
                  system_consequences=["task_failure"] if not success else ["none"],
                  mitigation_detected=detected, recovery_detected=recovered,
                  mitigation_recovery_detected=mitigation_recovered,
                  common_recovery_detected=common_recovered, common_recovery_attempted=common_attempted,
                  recovery_type="common_live_readback" if common_recovered else "validated_live_readback" if mitigation_recovered else "none",
                  recovery_evidence={"mitigation_event_indices": recovered_events,
                                     "common_recovery_event_indices": common_recovered_events} if recovered else {},
                  propagation_class=propagation, m_propagated=bool(m))
    result["final_answer"]["cart_verified"] = success
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def recover_torn_journals(output):
    """Called under the output lock; preserve original bytes before tail repair."""
    for name in ("run_attempts.jsonl", "run_errors.jsonl", "main_runs.jsonl"):
        path = output / name
        if not path.is_file():
            continue
        original = path.read_bytes()
        lines = original.splitlines(keepends=True)
        repaired = original
        reason = None
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError(f"corrupt journal, manual audit required: {path}:{index + 1}") from exc
                repaired = b"".join(lines[:index])
                reason = "torn_final_record"
        if repaired and not repaired.endswith(b"\n"):
            repaired += b"\n"
            reason = reason or "missing_final_newline"
        if reason:
            suffix = str(time.time_ns())
            backup = path.with_name(name + ".torn-" + suffix)
            with backup.open("xb") as handle:
                handle.write(original)
                handle.flush()
                os.fsync(handle.fileno())
            temporary = path.with_name(name + ".repair-" + suffix)
            with temporary.open("xb") as handle:
                handle.write(repaired)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            append_jsonl(output / "journal_repairs.jsonl", {"file": name, "backup": backup.name,
                         "reason": reason, "timestamp_unix": time.time(), "usage_complete": False})


def pending_jobs(jobs, output):
    rows = read_jsonl(output / "main_runs.jsonl")
    keys = [row["job_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate completed job keys")
    run_ids = [row["run_id"] for row in rows if row.get("run_id")]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run IDs")
    expected = {job["job_key"] for job in jobs}
    if set(keys) - expected:
        raise ValueError("checkpoint contains jobs outside frozen matrix")
    attempts = attempt_counts(output)
    blocked = [job["job_key"] for job in jobs if attempts[job["job_key"]] >= 2 and job["job_key"] not in keys]
    return [job for job in jobs if job["job_key"] not in set(keys) | set(blocked)], blocked


def attempt_counts(output):
    counts = Counter(row["job_key"] for row in read_jsonl(output / "run_errors.jsonl"))
    for row in read_jsonl(output / "run_attempts.jsonl"):
        counts[row["job_key"]] = max(counts[row["job_key"]], row["attempt"])
    return counts


@contextmanager
def locked_output(output):
    with (output / ".runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runner is already running in this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def freeze_manifest(output, config, *, resume):
    path = output / "matrix_manifest.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != config:
            raise ValueError("existing manifest differs or --resume was not supplied")
        return
    if resume or (output.exists() and any(output.iterdir())):
        raise ValueError("manifest absent: use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def build_summary(rows, errors):
    grouped = defaultdict(list)
    pairs = defaultdict(dict)
    for row in rows:
        grouped[row["mitigation_mode"]].append(row)
        pairs[row["pair_key"]][row["mitigation_mode"]] = row
    paired = [pair for pair in pairs.values() if "baseline" in pair and "guarded_recheck" in pair]
    clean_success = {(r["task_id"], r["topology"], r["repeat_index"], r["mitigation_mode"])
                     for r in rows if r.get("fault_type") == "none" and r["final_task_success"]}
    matched = [p for p in paired if p["baseline"].get("fault_type") != "none"
               and all((p[mode]["task_id"], p[mode]["topology"], p[mode]["repeat_index"], mode) in clean_success
                       for mode in ("baseline", "guarded_recheck"))]
    def count(mode_rows):
        return {"runs": len(mode_rows), "success": sum(r["final_task_success"] for r in mode_rows),
                "M_propagation": sum(r.get("observed_M_consequence", ["none"]) != ["none"] for r in mode_rows),
                "confirmed_recovery": sum(r.get("recovery_detected", False) for r in mode_rows),
                "http_requests_added": sum(r.get("additional_http_requests", 0) for r in mode_rows),
                "mitigation_recovery_successes": sum(r.get("mitigation_recovery_detected", False) for r in mode_rows),
                "common_recovery_successes": sum(r.get("common_recovery_detected", False) for r in mode_rows),
                "common_recovery_http_requests": sum(r.get("common_recovery_http_requests", 0) for r in mode_rows),
                "total_tokens": sum(r.get("total_tokens", 0) for r in mode_rows),
                "mean_latency_ms": sum(r.get("latency_ms", 0) for r in mode_rows) / len(mode_rows)}
    return {"evaluator_version": VERSION, "completed_runs": len(rows), "error_attempts": len(errors),
            "total_tokens_completed": sum(r.get("total_tokens", 0) for r in rows),
            "known_tokens_error_attempts": sum(r.get("known_total_tokens", 0) for r in errors),
            "by_mode": {mode: count(values) for mode, values in grouped.items()},
            "complete_three_arm_pairs": sum(set(MODES).issubset(p) for p in pairs.values()),
            "baseline_guarded_pairs": len(paired), "clean_matched_fault_pairs": len(matched),
            "paired_improvements": sum(not p["baseline"]["final_task_success"] and p["guarded_recheck"]["final_task_success"] for p in matched),
            "paired_regressions": sum(p["baseline"]["final_task_success"] and not p["guarded_recheck"]["final_task_success"] for p in matched),
            "clean_success_regressions": sum(p["baseline"].get("fault_type") == "none" and p["baseline"]["final_task_success"]
                                             and not p["guarded_recheck"]["final_task_success"] for p in paired)}


def write_reports(output, blocked):
    rows, errors = read_jsonl(output / "main_runs.jsonl"), read_jsonl(output / "run_errors.jsonl")
    summary = {**build_summary(rows, errors), "blocked_job_keys": blocked}
    if (output / "matrix_manifest.json").is_file():
        manifest = json.loads((output / "matrix_manifest.json").read_text())
        expected = {job["job_key"] for job in manifest["jobs"]}
        unrun = sorted(expected - {row["job_key"] for row in rows})
        summary.update(planned_runs=len(expected), unrun_job_keys=unrun,
                       status="complete" if not unrun else "incomplete")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row if key not in ("events", "mitigation_events")})
    with (output / "main_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list))
                             else row.get(key) for key in fields})
    with (output / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for kind in ("events", "mitigation_events", "common_recovery_events", "http_receipts", "model_requests"):
                for index, event in enumerate(row.get(kind, [])):
                    handle.write(json.dumps({**event, "run_id": row["run_id"], "job_key": row["job_key"],
                                             "trace_group": kind, "group_event_index": index}, ensure_ascii=False) + "\n")
    lines = ["# WebArena 缓解试验", "", f"完成 {len(rows)} 条；错误尝试 {len(errors)} 次。", "",
             "原基线、固定一次复查、按需复查使用相同任务、拓扑和故障位置。模型重复编号不是可控模型 seed。", "",
             "各组均启用一次拒绝后的共有恢复；下面的新增 HTTP 请求仅指接收端缓解，共有恢复请求单独统计。", "",
             "| 对照 | Runs | 成功 | M 传播 | 缓解恢复 | 共有恢复 | 新增 HTTP | 共有恢复 HTTP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, values in summary["by_mode"].items():
        lines.append(f"| {mode} | {values['runs']} | {values['success']} | {values['M_propagation']} | {values['mitigation_recovery_successes']} | {values['common_recovery_successes']} | {values['http_requests_added']} | {values['common_recovery_http_requests']} |")
    lines += ["", f"两侧 clean 均成功的故障配对：{summary['clean_matched_fault_pairs']}；改善 {summary['paired_improvements']}，退化 {summary['paired_regressions']}。",
              f"clean 成功退化：{summary['clean_success_regressions']}。clean 上触发检查不直接算误报。",
              "", "结果仅适用于本次重建版本和冻结样本；不能与历史主矩阵直接拼接。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    representatives = {}
    for row in rows:
        representatives.setdefault((row["mitigation_mode"], row["propagation_class"]), row)
    trace_lines = ["# 代表性传播与缓解记录", ""]
    for key, row in representatives.items():
        trace_lines += [f"## {key[0]} / {key[1]}", f"Run: `{row['run_id']}`", "```json",
                        json.dumps({"received": row.get("delivered_message"), "mitigation": row.get("mitigation_events"),
                                    "common_recovery": row.get("common_recovery_events"),
                                    "http_receipts": row.get("http_receipts"),
                                    "verdict": row["final_answer"], "M": row["observed_M_consequence"]}, ensure_ascii=False, indent=2), "```", ""]
    (output / "representative_traces.md").write_text("\n".join(trace_lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://localhost:7770")
    parser.add_argument("--required-model", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    tasks = load_task_manifest(Path(args.task_manifest))
    tasks = [{key: task[key] for key in TASK_FIELDS} for task in tasks]
    jobs = build_jobs(tasks, args.repetitions)
    output = Path(args.output_dir)
    root = Path(__file__).resolve().parent
    source_files = sorted((root / "src" / "mas_faults").rglob("*.py")) + [Path(__file__), root / "run_webarena_architecture_rq2.py"]
    config = {"version": VERSION, "tasks": tasks, "jobs": jobs, "base_url": args.base_url,
              "model": args.required_model, "provider": "modelscope_local", "max_attempts_per_job": 2,
              "inference_settings": inference_settings(),
              "repetitions": args.repetitions, "max_readbacks_per_run": 1,
              "common_recovery": {"enabled": True, "trigger": "final_rejected_or_unbound", "max_readbacks": 1},
              "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    freeze_manifest(output, config, resume=args.resume)
    with locked_output(output):
        recover_torn_journals(output)
        pending, blocked = pending_jobs(jobs, output)
        print(f"planned={len(jobs)} pending={len(pending)} blocked={len(blocked)}", flush=True)
        if args.plan_only:
            return
        execute_matrix(args, tasks, jobs, output)


def execute_matrix(args, tasks, jobs, output):
    pending, blocked = pending_jobs(jobs, output)
    client = get_llm_client()
    info = client.model_info
    if info.provider != "modelscope_local" or info.model != args.required_model or urlparse(info.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("this pilot requires the exact real ModelScope-local model on a loopback endpoint")
    with urllib.request.urlopen(model_listing_url(info.base_url), timeout=10) as response:
        served = json.load(response)
    if args.required_model not in {item.get("id") for item in served.get("data", [])}:
        raise ValueError("required model not present in inference server /models")
    task_map = {task["task_id"]: task for task in tasks}
    attempts = attempt_counts(output)
    for job in pending:
        if job["fault"] != "none":
            cleans = read_jsonl(output / "main_runs.jsonl")
            admitted = any(r["task_id"] == job["task_id"] and r["topology"] == job["topology"]
                           and r["repeat_index"] == job["repeat_index"] and r["fault_type"] == "none"
                           and r["mitigation_mode"] == "baseline" and r["final_task_success"] for r in cleans)
            if not admitted:
                continue
        for attempt in range(attempts[job["job_key"]] + 1, 3):
            task = task_map[job["task_id"]]
            before = client.prompt_tokens + client.completion_tokens
            append_jsonl(output / "run_attempts.jsonl", {**job, "attempt": attempt,
                         "model": info.model, "provider": info.provider, "timestamp_unix": time.time()})
            try:
                policy = EvidencePolicy(job["mitigation_mode"], task)
                stale_source = freeze_stale_source(read_jsonl(output / "main_runs.jsonl"), job, output) if job["fault"] == "stale_replay" else None
                result = asyncio.run(run_one(client, task, job["topology"], job["fault"], 4,
                                             job["repeat_index"], args.base_url, receiver_policy=policy,
                                             common_recovery=True, stale_replay_source=stale_source))
                result.update(job)
                result = audit_outcome(result, task)
            except Exception as exc:
                append_jsonl(output / "run_errors.jsonl", {**job, "attempt": attempt, "error_type": type(exc).__name__,
                             "error_stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                                             for frame in traceback.extract_tb(exc.__traceback__)],
                             "known_total_tokens": client.prompt_tokens + client.completion_tokens - before,
                             "usage_complete": False, "timestamp_unix": time.time()})
                print(f"ERROR attempt={attempt} {job['job_key']} {type(exc).__name__}", flush=True)
                if attempt < 2:
                    time.sleep(2)
            else:
                append_jsonl(output / "main_runs.jsonl", result)
                print(f"DONE {job['job_key']} success={result['final_task_success']}", flush=True)
                break
        _, blocked = pending_jobs(jobs, output)
        write_reports(output, blocked)
    _, blocked = pending_jobs(jobs, output)
    write_reports(output, blocked)


if __name__ == "__main__":
    main()
