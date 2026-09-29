#!/usr/bin/env python3
"""运行 WebArena Shopping Admin controlled-tool 通信故障主矩阵。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import derive_propagation_class, normalize_run_record
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_admin_controlled import (
    ControlledRunError,
    EvaluatorWorkerClient,
    run_controlled_clean_task,
)
from mas_faults.webarena_admin_fault_matrix import (
    ADMIN_FAULT_CONDITIONS,
    AdminCommunicationInterceptor,
)
from mas_faults.webarena_admin_real import BrowserWorkerClient, load_admin_tasks


@dataclass(frozen=True)
class MatrixJob:
    task: dict[str, Any]
    condition: str
    repeat_index: int
    matrix_run_index: int


def build_matrix_jobs(
    tasks: Iterable[dict[str, Any]],
    *,
    runs_per_condition: int,
    conditions: Iterable[str] | None = None,
) -> list[MatrixJob]:
    selected = list(conditions or ADMIN_FAULT_CONDITIONS)
    unknown = [condition for condition in selected if condition not in ADMIN_FAULT_CONDITIONS]
    if unknown:
        raise ValueError(f"未知 fault condition: {unknown}")
    ordered = [condition for condition in ADMIN_FAULT_CONDITIONS if condition in selected]
    task_rows = list(tasks)
    jobs: list[MatrixJob] = []
    for condition in ordered:
        for repeat_index in range(1, runs_per_condition + 1):
            for task in task_rows:
                jobs.append(
                    MatrixJob(
                        task=task,
                        condition=condition,
                        repeat_index=repeat_index,
                        matrix_run_index=len(jobs) + 1,
                    )
                )
    return jobs


def select_stale_carrier(
    carriers: list[dict[str, Any]],
    *,
    current_task_id: str,
    repeat_index: int,
) -> dict[str, Any]:
    candidates = [
        carrier
        for carrier in carriers
        if str(carrier["task_id"]) != str(current_task_id)
    ]
    if not candidates:
        raise ValueError(f"task {current_task_id} 没有可用的跨任务 stale carrier")
    candidates.sort(
        key=lambda carrier: (
            carrier["repeat_index"] != repeat_index,
            abs(int(carrier["repeat_index"]) - repeat_index),
        )
    )
    return json.loads(json.dumps(candidates[0]["message"], ensure_ascii=False))


def _labels(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return ["none"]


def _observed(value: Any) -> bool:
    return any(label != "none" for label in _labels(value))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row.get(key) or 0) for row in rows) / len(rows), 3) if rows else 0.0


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row.get("final_task_success")) for row in rows)
    a_count = sum(_observed(row.get("observed_A_symptom")) for row in rows)
    m_count = sum(_observed(row.get("observed_M_consequence")) for row in rows)
    recovery_count = sum(bool(row.get("recovery_detected")) for row in rows)
    return {
        "runs": count,
        "fault_applied_count": sum(bool(row.get("fault_applied")) for row in rows),
        "A_exposure_count": a_count,
        "M_consequence_count": m_count,
        "recovery_count": recovery_count,
        "final_success_count": successes,
        "final_failure_count": count - successes,
        "final_task_success_rate": round(successes / count, 4) if count else 0.0,
        "A_layer_exposure_rate": round(a_count / count, 4) if count else 0.0,
        "M_layer_propagation_rate": round(m_count / count, 4) if count else 0.0,
        "recovery_rate": round(recovery_count / count, 4) if count else 0.0,
        "final_failure_rate": round((count - successes) / count, 4) if count else 0.0,
        "mean_latency_ms": _mean(rows, "latency_ms"),
        "mean_token_usage": _mean(rows, "total_tokens"),
        "mean_api_call_count": _mean(rows, "api_call_count"),
    }


TRANSITION_NAMES = {
    "masked": "fault_injected_to_masked",
    "exposed_at_A_only": "fault_injected_to_exposed_at_A_only",
    "detected_and_recovered": "fault_injected_to_propagated_to_M_recovered",
    "detected_but_unrecovered": "fault_injected_to_detected_but_unrecovered",
    "silent_propagation_to_M": "fault_injected_to_propagated_to_M_final_success_without_recovery",
    "propagated_to_M_final_failure": "fault_injected_to_propagated_to_M_final_failure",
}


def aggregate_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition", "unknown"))].append(row)
    transitions = Counter()
    for row in rows:
        if not row.get("fault_applied"):
            continue
        transition = TRANSITION_NAMES.get(str(row.get("propagation_class")))
        if transition:
            transitions[transition] += 1
    for name in TRANSITION_NAMES.values():
        transitions.setdefault(name, 0)
    return {
        "overall": _metrics(rows),
        "transition_counts": dict(transitions),
        "by_condition": {
            condition: _metrics(condition_rows)
            for condition, condition_rows in by_condition.items()
        },
        "A_symptom_counts": dict(
            Counter(label for row in rows for label in _labels(row.get("observed_A_symptom")) if label != "none")
        ),
        "M_consequence_counts": dict(
            Counter(label for row in rows for label in _labels(row.get("observed_M_consequence")) if label != "none")
        ),
        "system_consequence_counts": dict(
            Counter(label for row in rows for label in _labels(row.get("system_consequences")) if label != "none")
        ),
        "semantic_consequence_counts": dict(
            Counter(label for row in rows for label in _labels(row.get("semantic_consequences")) if label != "none")
        ),
        "propagation_class_counts": dict(Counter(str(row.get("propagation_class")) for row in rows)),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row if key != "events"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# WebArena Shopping Admin 通信故障主矩阵摘要",
        "",
        "## 总体指标",
        "",
        f"- Runs: {overall['runs']}",
        f"- 最终任务成功率: {overall['final_task_success_rate']:.2%}",
        f"- A 层暴露率: {overall['A_layer_exposure_rate']:.2%}",
        f"- M 层传播率: {overall['M_layer_propagation_rate']:.2%}",
        f"- Recovery 率: {overall['recovery_rate']:.2%}",
        f"- 最终失败率: {overall['final_failure_rate']:.2%}",
        f"- 平均延迟: {overall['mean_latency_ms']:.1f} ms",
        f"- 平均 token: {overall['mean_token_usage']:.1f}",
        "",
        "## 分条件结果",
        "",
        "| Condition | Runs | A exposed | M consequence | Recovery | Success | Failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in summary["by_condition"].items():
        lines.append(
            f"| {condition} | {metrics['runs']} | {metrics['A_exposure_count']} | "
            f"{metrics['M_consequence_count']} | {metrics['recovery_count']} | "
            f"{metrics['final_success_count']} | {metrics['final_failure_count']} |"
        )
    lines.extend(["", "## 传播转移", ""])
    for name, count in summary["transition_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本结果测量的是 controlled-tool AutoGen process trace；官方 evaluator 仅判定最终答案。最终成功不覆盖 A/M 层传播，recovery 只在 trace 提供直接证据时记录。",
        ]
    )
    return "\n".join(lines) + "\n"


def _representative_traces(rows: list[dict[str, Any]]) -> str:
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("propagation_class"))
        representatives.setdefault(key, row)
    lines = ["# 代表性因果传播 Trace", ""]
    for propagation_class, row in representatives.items():
        lines.extend(
            [
                f"## {row.get('trace_id')} - {propagation_class}",
                "",
                f"- Task: `{row.get('task_id')}`",
                f"- Condition: `{row.get('condition')}`",
                f"- A: `{','.join(_labels(row.get('observed_A_symptom')))}`",
                f"- M: `{','.join(_labels(row.get('observed_M_consequence')))}`",
                f"- Recovery: `{row.get('recovery_type', 'none')}`",
                f"- Final success: `{row.get('final_task_success')}`",
                "",
                "| Step | Source -> Target | Effect | Fault |",
                "|---|---|---|---|",
            ]
        )
        for event in row.get("events", []):
            lines.append(
                f"| {event.get('step_id', '')} | {event.get('source_agent', '')} -> "
                f"{event.get('target_agent', '')} | {event.get('observed_runtime_effect', '')} | "
                f"{event.get('fault_type', 'clean')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_matrix_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_matrix(rows)
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (output_dir / "llm_communication_traces.jsonl").write_text(serialized, encoding="utf-8")
    (output_dir / "llm_communication_runs.jsonl").write_text(serialized, encoding="utf-8")
    _write_csv(output_dir / "llm_communication_runs.csv", rows)
    (output_dir / "llm_communication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_rows = [
        {"condition": condition, **metrics}
        for condition, metrics in summary["by_condition"].items()
    ]
    _write_csv(output_dir / "llm_communication_summary.csv", summary_rows)
    (output_dir / "llm_communication_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    (output_dir / "representative_causal_traces.md").write_text(
        _representative_traces(rows), encoding="utf-8"
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def browser_environment(shopping_admin_url: str) -> dict[str, str]:
    return {
        "PLAYWRIGHT_BROWSERS_PATH": "/data2/system5/mas/playwright-browsers-1.32",
        "SHOPPING": "http://10.102.35.120:7770",
        "SHOPPING_ADMIN": shopping_admin_url,
        "REDDIT": "http://10.102.35.120:7771",
        "GITLAB": "http://10.102.35.120:8929",
        "MAP": "http://10.102.35.120:7772",
        "WIKIPEDIA": "http://10.102.35.120:8888",
        "HOMEPAGE": "http://10.102.35.120:4399",
    }


def _error_record(
    *,
    client: Any,
    task: dict[str, Any],
    job: MatrixJob,
    exc: Exception,
    before: tuple[int, int, int],
    request_log_before: int,
    latency_ms: float,
) -> dict[str, Any]:
    events = list(getattr(exc, "events", []))
    fault_event = next((event for event in events if event.get("fault_applied")), None)
    spec = ADMIN_FAULT_CONDITIONS[job.condition]
    fault_applied = fault_event is not None
    a_symptom = [fault_event.get("observed_A_symptom", spec.observed_a_symptom)] if fault_event else ["none"]
    observed_m = ["M2_task_timeout_or_failure"]
    if fault_event and fault_event.get("delivered_message") is None:
        observed_m.append("M3_incomplete_information_aggregation")
    propagation = derive_propagation_class(
        fault_applied=fault_applied,
        observed_a_symptom=a_symptom,
        observed_m_consequence=observed_m,
        recovery_detected=False,
        final_task_success=False,
    )
    prompt_tokens = client.prompt_tokens - before[1]
    completion_tokens = client.completion_tokens - before[2]
    run_id = next((event.get("run_id") for event in events if event.get("run_id")), f"admin-matrix-error-{uuid.uuid4().hex[:12]}")
    trace_id = next((event.get("trace_id") for event in events if event.get("trace_id")), f"trace-{uuid.uuid4()}")
    return normalize_run_record(
        {
            "run_id": run_id,
            "trace_id": trace_id,
            "scenario": "webarena_shopping_admin_controlled_fault_matrix",
            "dataset": "WebArena",
            "benchmark": "WebArena Shopping Admin",
            "framework": "AutoGen",
            "topology": "sequential",
            "task_id": str(task["task_id"]),
            "task_stratum": task.get("task_stratum", ""),
            "intent": task.get("intent", ""),
            "condition": job.condition,
            "repeat_index": job.repeat_index,
            "matrix_run_index": job.matrix_run_index,
            "model": client.model_info.model,
            "provider": client.model_info.provider,
            "seed_or_run_index": job.repeat_index,
            "fault_id": spec.fault_id,
            "fault_type": spec.fault_type,
            "fault_severity": spec.severity,
            "fault_parameters": spec.parameters,
            "fault_applied": fault_applied,
            "source_agent": "Evidence Worker" if fault_event else "",
            "target_agent": "Coordinator" if fault_event else "",
            "original_message": fault_event.get("original_message") if fault_event else {},
            "delivered_message": fault_event.get("delivered_message") if fault_event else {},
            "first_divergence": f"A:evidence_handoff:{job.condition}" if fault_applied else "none",
            "observed_runtime_effect": getattr(exc, "termination_reason", "runtime_error"),
            "observed_A_symptom": a_symptom,
            "observed_M_consequence": observed_m,
            "system_consequences": ["task_failure"],
            "semantic_consequences": ["evidence_omission"] if len(observed_m) > 1 else ["none"],
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": [],
            "expected_answer": task.get("eval", {}).get("reference_answers", {}),
            "final_answer": {},
            "task_score": 0.0,
            "final_task_success": False,
            "propagation_class": propagation,
            "latency_ms": latency_ms,
            "api_call_count": client.call_count - before[0],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "llm_calls": list(getattr(client, "request_log", []))[request_log_before:],
            "error": f"{type(exc).__name__}: {exc}",
            "events": events,
            "standard_webarena_action_protocol": False,
            "process_trajectory_type": "controlled_tool_trace",
        }
    )


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default="187,199")
    parser.add_argument("--runs-per-condition", type=int, default=3)
    parser.add_argument("--conditions", default=",".join(ADMIN_FAULT_CONDITIONS))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-evidence-chars", type=int, default=24000)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--strict-gate", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data2/system5/mas/task_manifests/webarena_shopping_admin_readonly_30_20260813.json"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data2/system5/mas/task_configs/webarena_shopping_admin_verified_20260814"),
    )
    parser.add_argument("--webarena-root", type=Path, default=Path("/data2/system5/mas/third_party/webarena"))
    parser.add_argument("--shopping-admin-url", default="http://10.102.35.120:7780/admin")
    return parser.parse_args()


def _gate_errors(rows: list[dict[str, Any]], jobs: list[MatrixJob]) -> list[str]:
    errors: list[str] = []
    if len(rows) != len(jobs):
        errors.append(f"run_count:{len(rows)}!={len(jobs)}")
    for row in rows:
        condition = str(row.get("condition"))
        if condition == "clean" and not row.get("final_task_success"):
            errors.append(f"clean_failure:{row.get('run_id')}")
        if condition != "clean" and not row.get("fault_applied"):
            errors.append(f"fault_not_applied:{row.get('run_id')}:{condition}")
        if not row.get("trace_id") or not row.get("events"):
            errors.append(f"missing_trace:{row.get('run_id')}")
        errors.extend(f"axis:{row.get('run_id')}:{error}" for error in row.get("consequence_axis_validation_errors", []))
        errors.extend(f"strict:{row.get('run_id')}:{error}" for error in row.get("strict_derivation_validation_errors", []))
    return errors


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"结果目录已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    load_dotenv(Path(__file__).with_name(".env.local"), override=False)
    requested_ids = {int(value) for value in args.task_ids.split(",") if value.strip()}
    tasks = [task for task in load_admin_tasks(args.manifest) if int(task["task_id"]) in requested_ids]
    missing = requested_ids - {int(task["task_id"]) for task in tasks}
    if missing:
        raise ValueError(f"manifest 中没有 task IDs: {sorted(missing)}")
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    jobs = build_matrix_jobs(tasks, runs_per_condition=args.runs_per_condition, conditions=conditions)
    client = get_llm_client(mock_llm=args.mock_llm)
    environment = browser_environment(args.shopping_admin_url)
    evaluator = EvaluatorWorkerClient(webarena_root=str(args.webarena_root), env=environment)
    rows: list[dict[str, Any]] = []
    carriers: list[dict[str, Any]] = []
    checkpoint = args.output_dir / "llm_communication_runs.checkpoint.jsonl"

    try:
        with tempfile.TemporaryDirectory(prefix="webarena-admin-matrix-") as temp:
            sanitized_root = Path(temp)
            for job in jobs:
                stale_message = None
                if job.condition in {"a6_inner_evidence_poisoning", "a12_stale_replay"}:
                    stale_message = select_stale_carrier(
                        carriers,
                        current_task_id=str(job.task["task_id"]),
                        repeat_index=job.repeat_index,
                    )
                interceptor = AdminCommunicationInterceptor(job.condition, stale_message=stale_message)
                before = (client.call_count, client.prompt_tokens, client.completion_tokens)
                request_log_before = len(getattr(client, "request_log", []))
                started = time.perf_counter()
                browser = BrowserWorkerClient(
                    webarena_root=str(args.webarena_root), env=environment, browser_only=True
                )
                try:
                    row = asyncio.run(
                        run_controlled_clean_task(
                            client,
                            browser,
                            evaluator,
                            job.task,
                            original_config_file=args.config_dir / f"{job.task['task_id']}.json",
                            sanitized_config_dir=sanitized_root / f"run-{job.matrix_run_index}",
                            run_index=job.matrix_run_index,
                            max_steps=args.max_steps,
                            max_evidence_chars=args.max_evidence_chars,
                            condition=job.condition,
                            interceptor=interceptor,
                        )
                    )
                except Exception as exc:
                    row = _error_record(
                        client=client,
                        task=job.task,
                        job=job,
                        exc=exc,
                        before=before,
                        request_log_before=request_log_before,
                        latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    )
                finally:
                    try:
                        browser.close()
                    except Exception:
                        pass
                row["repeat_index"] = job.repeat_index
                row["seed_or_run_index"] = job.repeat_index
                row["matrix_run_index"] = job.matrix_run_index
                rows.append(row)
                if job.condition == "clean" and isinstance(row.get("original_message"), dict) and row["original_message"].get("payload"):
                    carriers.append(
                        {
                            "task_id": str(job.task["task_id"]),
                            "repeat_index": job.repeat_index,
                            "message": row["original_message"],
                        }
                    )
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"完成 {len(rows)}/{len(jobs)} task={job.task['task_id']} "
                    f"condition={job.condition} repeat={job.repeat_index} "
                    f"class={row.get('propagation_class')} success={row.get('final_task_success')} "
                    f"error={row.get('error')}",
                    flush=True,
                )
    finally:
        evaluator.close()

    root = Path(__file__).resolve().parent
    source_paths = [
        root / "src/mas_faults/webarena_admin_controlled.py",
        root / "src/mas_faults/webarena_admin_fault_matrix.py",
        root / "run_webarena_admin_controlled_fault_matrix.py",
    ]
    experiment_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "WebArena Shopping Admin",
        "framework": "AutoGen",
        "topology": "sequential",
        "protocol": "controlled_read_only_tools",
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "task_ids": [task["task_id"] for task in tasks],
        "conditions": conditions,
        "runs_per_condition_per_task": args.runs_per_condition,
        "total_target_runs": len(jobs),
        "fault_injection_edge": "Evidence Worker -> Coordinator",
        "recovery_policy": "same_no_retry_no_fallback_for_all_conditions",
        "fault_parameters": {condition: ADMIN_FAULT_CONDITIONS[condition].parameters for condition in conditions},
        "max_steps": args.max_steps,
        "max_evidence_chars": args.max_evidence_chars,
        "answer_isolation": "browser sanitized config + separate evaluator process",
        "official_final_answer_evaluator": True,
        "standard_webarena_action_protocol": False,
        "process_trajectory_type": "controlled_tool_trace",
        "source_sha256": {str(path.relative_to(root)): _source_hash(path) for path in source_paths},
    }
    summary = write_matrix_outputs(rows, args.output_dir, experiment_config=experiment_config)
    gate_errors = _gate_errors(rows, jobs)
    (args.output_dir / "matrix_gate.json").write_text(
        json.dumps({"passed": not gate_errors, "errors": gate_errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output_dir), "summary": summary, "gate_errors": gate_errors}, ensure_ascii=False), flush=True)
    if args.strict_gate and gate_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
