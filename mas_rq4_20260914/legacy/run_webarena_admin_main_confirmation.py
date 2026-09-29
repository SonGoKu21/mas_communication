#!/usr/bin/env python3
"""Run the frozen WebArena Shopping Admin main confirmation matrix."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.deepseek_schedule import ensure_deepseek_offpeak
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_admin_confirmation import run_admin_confirmation_task
from mas_faults.webarena_admin_controlled import (
    ControlledRunError,
    EvaluatorWorkerClient,
)
from mas_faults.webarena_admin_main_evaluator import evaluate_main_outcome
from mas_faults.webarena_admin_main_matrix import (
    CONDITION_BY_NAME,
    MAIN_CONDITION_CELLS,
    MAIN_TASK_IDS,
    MAIN_TOPOLOGIES,
    MainMatrixJob,
    build_main_matrix_jobs,
)
from mas_faults.webarena_admin_real import BrowserWorkerClient, load_admin_tasks
from run_webarena_admin_controlled_fault_matrix import browser_environment


DEFAULT_SCHEDULE_SEED = 20260815


def build_phase_jobs(
    phase: str,
    tasks: list[dict[str, Any]],
    *,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    repetitions: int | None = None,
    task_ids: Iterable[int] = MAIN_TASK_IDS,
    condition_cells: Iterable[Any] = MAIN_CONDITION_CELLS,
) -> list[MainMatrixJob]:
    if phase not in {"admission", "smoke", "formal"}:
        raise ValueError(f"unknown phase: {phase}")
    selected_repetitions = (
        repetitions if repetitions is not None else (3 if phase == "formal" else 1)
    )
    selected_cells = tuple(condition_cells)
    jobs = build_main_matrix_jobs(
        tasks,
        repetitions=selected_repetitions,
        task_ids=task_ids,
        condition_cells=selected_cells,
    )
    if phase == "admission":
        jobs = [job for job in jobs if job.condition_cell.condition == "clean"]
    elif phase == "smoke":
        jobs = [job for job in jobs if int(job.task["task_id"]) == 199]
    grouped: dict[tuple[str, int, int], list[MainMatrixJob]] = defaultdict(list)
    for job in jobs:
        grouped[
            (
                job.condition_cell.condition,
                int(job.task["task_id"]),
                job.repeat_index,
            )
        ].append(job)

    rng = random.Random(schedule_seed)
    clean_keys = [key for key in grouped if key[0] == "clean"]
    fault_keys = [key for key in grouped if key[0] != "clean"]
    rng.shuffle(clean_keys)
    rng.shuffle(fault_keys)
    scheduled: list[MainMatrixJob] = []
    for key in (*clean_keys, *fault_keys):
        block = list(grouped[key])
        rng.shuffle(block)
        scheduled.extend(block)
    return scheduled


def select_fault_shard(
    jobs: list[MainMatrixJob],
    *,
    shard_count: int,
    shard_index: int,
) -> list[MainMatrixJob]:
    """Select whole condition/task/repeat blocks for a parallel fault shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least one")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    blocks: list[list[MainMatrixJob]] = []
    current_key: tuple[str, int, int] | None = None
    for job in jobs:
        if job.condition_cell.condition == "clean":
            continue
        key = (
            job.condition_cell.condition,
            int(job.task["task_id"]),
            job.repeat_index,
        )
        if key != current_key:
            blocks.append([])
            current_key = key
        blocks[-1].append(job)
    return [
        job
        for block_index, block in enumerate(blocks)
        if block_index % shard_count == shard_index
        for job in block
    ]


def select_topology_jobs(
    jobs: list[MainMatrixJob], topologies: Iterable[str]
) -> list[MainMatrixJob]:
    selected = tuple(dict.fromkeys(topologies))
    if not selected:
        raise ValueError("at least one topology must be selected")
    unknown = [name for name in selected if name not in MAIN_TOPOLOGIES]
    if unknown:
        raise ValueError(f"unknown main topology: {unknown}")
    return [job for job in jobs if job.topology in selected]


def prepare_execution_jobs(
    all_jobs: list[MainMatrixJob],
    *,
    topologies: Iterable[str],
    clean_only: bool,
    fault_shard_count: int,
    fault_shard_index: int,
) -> list[MainMatrixJob]:
    jobs = select_topology_jobs(all_jobs, topologies)
    if clean_only:
        if fault_shard_count > 1:
            raise ValueError("clean-only cannot be combined with fault sharding")
        return [
            job for job in jobs if job.condition_cell.condition == "clean"
        ]
    if fault_shard_count > 1:
        return select_fault_shard(
            jobs,
            shard_count=fault_shard_count,
            shard_index=fault_shard_index,
        )
    return jobs


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_completed_job_keys(path: Path) -> set[str]:
    return {
        str(row["job_key"])
        for row in _load_jsonl(path)
        if row.get("job_key")
    }


def _labels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return ["none"]


def _observed(value: Any) -> bool:
    return any(label != "none" for label in _labels(value))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(
        sum(float(row.get(key) or 0) for row in rows) / len(rows), 3
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row.get("final_task_success")) for row in rows)
    a_count = sum(_observed(row.get("observed_A_symptom")) for row in rows)
    m_count = sum(_observed(row.get("observed_M_consequence")) for row in rows)
    recovery_count = sum(bool(row.get("recovery_detected")) for row in rows)
    return {
        "runs": count,
        "fault_applied_count": sum(bool(row.get("fault_applied")) for row in rows),
        "pre_injection_model_failure_count": sum(
            bool(row.get("pre_injection_model_failure")) for row in rows
        ),
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


def _group_metrics(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "none"))].append(row)
    return {name: _metrics(group) for name, group in sorted(groups.items())}


def aggregate_main_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(
        str(row.get("propagation_class", "unknown"))
        for row in rows
        if row.get("fault_applied")
    )
    topology_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        topology_condition[
            f"{row.get('topology')}|{row.get('condition')}"
        ].append(row)
    return {
        "overall": _metrics(rows),
        "by_topology": _group_metrics(rows, "topology"),
        "by_condition": _group_metrics(rows, "condition"),
        "by_fault_family": _group_metrics(rows, "fault_family"),
        "by_task_stratum": _group_metrics(rows, "task_stratum"),
        "by_injection_step": _group_metrics(rows, "injection_step"),
        "by_topology_condition": {
            key: _metrics(group) for key, group in sorted(topology_condition.items())
        },
        "propagation_class_counts": dict(transitions),
        "A_symptom_counts": dict(
            Counter(
                label
                for row in rows
                for label in _labels(row.get("observed_A_symptom"))
                if label != "none"
            )
        ),
        "M_consequence_counts": dict(
            Counter(
                label
                for row in rows
                for label in _labels(row.get("observed_M_consequence"))
                if label != "none"
            )
        ),
        "system_consequence_counts": dict(
            Counter(
                label
                for row in rows
                for label in _labels(row.get("system_consequences"))
                if label != "none"
            )
        ),
        "semantic_consequence_counts": dict(
            Counter(
                label
                for row in rows
                for label in _labels(row.get("semantic_consequences"))
                if label != "none"
            )
        ),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row if key != "events"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# WebArena Shopping Admin 主确认矩阵摘要",
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
        "## 拓扑结果",
        "",
        "| Topology | Runs | A | M | Recovery | Success | Failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["by_topology"].items():
        lines.append(
            f"| {name} | {metrics['runs']} | {metrics['A_exposure_count']} | "
            f"{metrics['M_consequence_count']} | {metrics['recovery_count']} | "
            f"{metrics['final_success_count']} | {metrics['final_failure_count']} |"
        )
    lines.extend(["", "## Fault condition 结果", ""])
    lines.extend(
        [
            "| Condition | Runs | A | M | Recovery | Success | Failure |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in summary["by_condition"].items():
        lines.append(
            f"| {name} | {metrics['runs']} | {metrics['A_exposure_count']} | "
            f"{metrics['M_consequence_count']} | {metrics['recovery_count']} | "
            f"{metrics['final_success_count']} | {metrics['final_failure_count']} |"
        )
    lines.extend(["", "## 传播分类", ""])
    for name, count in summary["propagation_class_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "最终成功不覆盖 A/M 传播；M14、M5、M4、M6 由实际接受状态和官方结果推导；recovery 只在 trace 中有直接机制证据时记录。",
        ]
    )
    return "\n".join(lines) + "\n"


def _representative_traces(rows: list[dict[str, Any]]) -> str:
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        representatives.setdefault(str(row.get("propagation_class")), row)
    lines = ["# 代表性因果传播 Trace", ""]
    for propagation_class, row in representatives.items():
        lines.extend(
            [
                f"## {row.get('trace_id')} - {propagation_class}",
                "",
                f"- Topology: `{row.get('topology')}`",
                f"- Task: `{row.get('task_id')}` / `{row.get('task_stratum')}`",
                f"- Condition: `{row.get('condition')}` / Step `{row.get('injection_step')}`",
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
                f"| {event.get('abstract_step', '')} | {event.get('source_agent', '')} -> "
                f"{event.get('target_agent', '')} | {str(event.get('observed_runtime_effect', '')).replace('|', '/')} | "
                f"{event.get('fault_type', 'clean')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_main_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = experiment_config.get("model")
    provider = experiment_config.get("provider")
    for row in rows:
        fault_event = next(
            (
                event
                for event in row.get("events", [])
                if event.get("fault_applied")
            ),
            None,
        )
        if not row.get("model") and model:
            row["model"] = model
        if not row.get("provider") and provider:
            row["provider"] = provider
        if "original_message" not in row:
            row["original_message"] = (
                fault_event.get("original_message") if fault_event else None
            )
        if "delivered_message" not in row:
            row["delivered_message"] = (
                fault_event.get("delivered_message") if fault_event else None
            )
    summary = aggregate_main_results(rows)
    canonical = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    (output_dir / "llm_communication_traces.jsonl").write_text(
        canonical, encoding="utf-8"
    )
    _write_csv(output_dir / "llm_communication_runs.csv", rows)
    (output_dir / "llm_communication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def _extract_carrier(row: dict[str, Any]) -> dict[str, Any] | None:
    for event in row.get("events", []):
        if (
            event.get("abstract_step") == 4
            and event.get("source_agent") == "Evidence Worker"
            and isinstance(event.get("original_message"), dict)
        ):
            return event["original_message"]
    return None


def carriers_from_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    carriers: list[dict[str, Any]] = []
    for row in rows:
        if row.get("condition") != "clean":
            continue
        carrier = _extract_carrier(row)
        if carrier is None:
            continue
        carriers.append(
            {
                "topology": str(row.get("topology")),
                "task_id": str(row.get("task_id")),
                "repeat_index": int(row.get("repeat_index", 1)),
                "message": carrier,
            }
        )
    return carriers


def _select_stale_carrier(
    carriers: list[dict[str, Any]],
    job: MainMatrixJob,
    *,
    task_ids: Iterable[int] = MAIN_TASK_IDS,
) -> dict[str, Any]:
    selected_task_ids = tuple(dict.fromkeys(int(task_id) for task_id in task_ids))
    target_id = int(job.task["task_id"])
    target_index = selected_task_ids.index(target_id)
    candidates: list[dict[str, Any]] = []
    desired_task_id = None
    for offset in range(job.repeat_index, job.repeat_index + len(selected_task_ids)):
        candidate_task_id = selected_task_ids[
            (target_index + offset) % len(selected_task_ids)
        ]
        if candidate_task_id == target_id:
            continue
        candidates = [
            item
            for item in carriers
            if item["topology"] == job.topology
            and int(item["task_id"]) == candidate_task_id
        ]
        if candidates:
            desired_task_id = candidate_task_id
            break
    if not candidates:
        raise ValueError(
            "no real cross-task stale carrier for "
            f"{job.topology}/{job.task['task_id']}"
        )
    candidates.sort(
        key=lambda item: (
            item.get("repeat_index") != job.repeat_index,
            abs(int(item.get("repeat_index", 1)) - job.repeat_index),
        )
    )
    return json.loads(json.dumps(candidates[0]["message"], ensure_ascii=False))


def _usage(client: Any) -> tuple[int, int, int]:
    return (
        int(getattr(client, "call_count", 0)),
        int(getattr(client, "prompt_tokens", 0)),
        int(getattr(client, "completion_tokens", 0)),
    )


def _error_record(
    client: Any,
    job: MainMatrixJob,
    exc: Exception,
    *,
    before: tuple[int, int, int],
    request_log_before: int,
    latency_ms: float,
) -> dict[str, Any]:
    events = list(getattr(exc, "events", []))
    fault_event = next(
        (event for event in events if event.get("fault_applied")), None
    )
    after = _usage(client)
    record = {
        "run_id": next(
            (event.get("run_id") for event in events if event.get("run_id")),
            f"admin-main-error-{uuid.uuid4().hex[:12]}",
        ),
        "trace_id": next(
            (event.get("trace_id") for event in events if event.get("trace_id")),
            f"trace-{uuid.uuid4()}",
        ),
        "scenario": "webarena_shopping_admin_main_confirmation",
        "dataset": "WebArena",
        "benchmark": "WebArena Shopping Admin",
        "framework": "AutoGen",
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "topology": job.topology,
        "task_id": str(job.task["task_id"]),
        "task_stratum": job.task.get("task_stratum", ""),
        "intent": job.task.get("intent", ""),
        "condition": job.condition_cell.condition,
        "fault_family": job.condition_cell.fault_family,
        "injection_step": job.condition_cell.injection_step,
        "fault_id": fault_event.get("fault_id", "none") if fault_event else "none",
        "fault_type": fault_event.get("fault_type", "clean") if fault_event else "clean",
        "fault_cause": fault_event.get("fault_cause", "none") if fault_event else "none",
        "fault_severity": job.condition_cell.severity,
        "fault_parameters": job.condition_cell.parameters,
        "fault_applied": bool(fault_event),
        "injection_valid": job.condition_cell.condition == "clean" or bool(fault_event),
        "source_agent": fault_event.get("source_agent", "") if fault_event else "none",
        "target_agent": fault_event.get("target_agent", "") if fault_event else "none",
        "original_message": fault_event.get("original_message") if fault_event else None,
        "delivered_message": fault_event.get("delivered_message") if fault_event else None,
        "first_divergence": (
            f"step_{fault_event.get('abstract_step')}:{job.condition_cell.condition}"
            if fault_event
            else "none"
        ),
        "observed_runtime_effect": (
            fault_event.get("observed_runtime_effect", "none")
            if fault_event
            else "none"
        ),
        "observed_A_symptom": [
            fault_event.get("observed_A_symptom", "none") if fault_event else "none"
        ],
        "observed_M_consequence": ["M2_task_timeout_or_failure"],
        "system_consequences": ["task_failure"],
        "semantic_consequences": ["none"],
        "verification": {"decision": "reject", "answer": "N/A", "reason": str(exc)},
        "final_answer": {"decision": "reject", "answer": "N/A", "reason": str(exc)},
        "final_task_success": False,
        "final_decision_correct": None,
        "task_score": 0.0,
        "recovery_detected": False,
        "recovery_type": "none",
        "recovery_evidence": [],
        "topology_recovery_evidence": [],
        "propagation_class": (
            "propagated_to_M_final_failure" if fault_event else "clean_task_failure"
        ),
        "latency_ms": latency_ms,
        "api_call_count": after[0] - before[0],
        "prompt_tokens": after[1] - before[1],
        "completion_tokens": after[2] - before[2],
        "total_tokens": (after[1] - before[1]) + (after[2] - before[2]),
        "llm_calls": list(getattr(client, "request_log", []))[request_log_before:],
        "error": f"{type(exc).__name__}: {exc}",
        "events": events,
        "termination_reason": str(getattr(exc, "termination_reason", "")),
        "browser_state": getattr(exc, "browser_state", {}),
        "official_final_answer_evaluator": False,
        "final_evaluator_mode": "not_executed",
        "official_evaluator_input": "none",
        "axis_evaluation_mode": "strict_evidence_v1",
    }
    record = evaluate_main_outcome(record)
    return normalize_run_record(
        record, axis_evaluation_mode=str(record["axis_evaluation_mode"])
    )


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("admission", "smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--carrier-source",
        type=Path,
        help="Optional canonical/checkpoint JSONL containing real clean Step 4 carriers.",
    )
    parser.add_argument("--max-attempts-per-job", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-evidence-chars", type=int, default=64000)
    parser.add_argument("--schedule-seed", type=int, default=DEFAULT_SCHEDULE_SEED)
    parser.add_argument(
        "--task-id",
        action="append",
        type=int,
        help="Restrict the matrix to selected task IDs; repeat as needed.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=tuple(CONDITION_BY_NAME),
        help="Restrict the matrix to selected condition cells; repeat as needed.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Override the phase repetition count.",
    )
    parser.add_argument("--fault-shard-count", type=int, default=1)
    parser.add_argument("--fault-shard-index", type=int, default=0)
    parser.add_argument(
        "--topology",
        action="append",
        choices=MAIN_TOPOLOGIES,
        help="Restrict execution to one or more topologies.",
    )
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--strict-gate", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data2/system5/mas/task_manifests/"
            "webarena_shopping_admin_readonly_30_20260813.json"
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(
            "/data2/system5/mas/task_configs/"
            "webarena_shopping_admin_verified_20260814"
        ),
    )
    parser.add_argument(
        "--webarena-root",
        type=Path,
        default=Path("/data2/system5/mas/third_party/webarena"),
    )
    parser.add_argument(
        "--shopping-admin-url", default="http://10.102.35.120:7780/admin"
    )
    return parser.parse_args()


def _is_infrastructure_error(exc: Exception) -> bool:
    cause = getattr(exc, "cause", exc)
    text = f"{type(cause).__name__}: {cause}".lower()
    fault_applied = any(
        bool(event.get("fault_applied"))
        for event in getattr(exc, "events", [])
    )
    browser_timeout = "browser worker" in text and (
        "timeout" in text or "exceeded" in text
    )
    if browser_timeout:
        return not fault_applied
    missing_browser_input = (
        "browser worker" in text
        and "expected one visible input" in text
        and "found 0" in text
    )
    if missing_browser_input:
        return True
    markers = (
        "browser worker exited",
        "returned no response",
        "request failed",
        "http error",
        "connection reset",
        "connection refused",
        "timed out",
    )
    return any(marker in text for marker in markers)


def _is_clean_matched_pre_injection_model_failure(
    accepted_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    reference_rows: Iterable[dict[str, Any]] = (),
) -> bool:
    """Identify a scheduled fault job blocked by a reproduced clean model failure."""
    if (
        candidate.get("condition") == "clean"
        or candidate.get("fault_applied")
        or candidate.get("termination_reason")
        not in {
            "tool_request_parse_failure",
            "tool_not_available_in_delivered_state",
        }
    ):
        return False
    return any(
        row.get("condition") == "clean"
        and str(row.get("task_id")) == str(candidate.get("task_id"))
        and row.get("topology") == candidate.get("topology")
        and row.get("final_task_success") is False
        and row.get("termination_reason") == candidate.get("termination_reason")
        and row.get("error") == candidate.get("error")
        for row in (*accepted_rows, *reference_rows)
    )


def _post_fault_failure_signature(row: dict[str, Any]) -> tuple[Any, ...] | None:
    error = str(row.get("error") or "")
    lowered = error.lower()
    if (
        not row.get("fault_applied")
        or "request failed" not in lowered
        or "deadline" not in lowered
    ):
        return None
    events = list(row.get("events") or [])
    fault_index = next(
        (index for index, event in enumerate(events) if event.get("fault_applied")),
        None,
    )
    if fault_index is None or fault_index >= len(events) - 1:
        return None
    last = events[-1]
    normalized_error = re.sub(r"\d+s total deadline", "Ns total deadline", lowered)
    return (
        normalized_error,
        row.get("termination_reason"),
        events[fault_index].get("abstract_step"),
        last.get("abstract_step"),
        last.get("source_agent"),
        last.get("target_agent"),
        last.get("observed_runtime_effect"),
    )


def _is_reproducible_post_fault_failure(
    attempts: list[dict[str, Any]],
    *,
    required_repetitions: int = 2,
) -> bool:
    if len(attempts) < required_repetitions:
        return False
    signatures = [
        _post_fault_failure_signature(row)
        for row in attempts[-required_repetitions:]
    ]
    return signatures[0] is not None and len(set(signatures)) == 1


def _gate_errors(
    rows: list[dict[str, Any]],
    jobs: list[MainMatrixJob],
    phase: str,
    *,
    expected_run_count: int | None = None,
) -> list[str]:
    errors = []
    expected = len(jobs) if expected_run_count is None else expected_run_count
    if len(rows) != expected:
        errors.append(f"run_count:{len(rows)}!={expected}")
    if len({row.get("job_key") for row in rows}) != len(rows):
        errors.append("duplicate_job_keys")
    for row in rows:
        labels = _labels(row.get("observed_M_consequence"))
        if (
            row.get("condition") != "clean"
            and not row.get("fault_applied")
            and not row.get("pre_injection_model_failure")
        ):
            errors.append(f"fault_not_applied:{row.get('job_key')}")
        if row.get("infrastructure_invalid"):
            errors.append(f"infrastructure_invalid:{row.get('job_key')}")
        if not row.get("trace_id") or not row.get("events"):
            errors.append(f"missing_trace:{row.get('job_key')}")
        if (
            "M4_incorrect_collective_decision" in labels
            and row.get("official_final_answer_evaluator") is not True
        ):
            errors.append(f"m4_without_official_evaluator:{row.get('job_key')}")
        if row.get("recovery_detected") and not row.get("recovery_evidence"):
            errors.append(f"recovery_without_trace_evidence:{row.get('job_key')}")
        if row.get("recovery_type") == "framework_recovery":
            errors.append(f"implicit_framework_recovery:{row.get('job_key')}")
        if (
            row.get("topology") == "flat"
            and row.get("official_final_answer_evaluator") is True
            and not any(
                event.get("source_agent") == "Evidence Worker"
                and event.get("target_agent") == "Coordinator"
                and event.get("observed_runtime_effect") == "clean_delivery"
                for event in row.get("events", [])
            )
        ):
            errors.append(f"flat_direct_delivery_missing:{row.get('job_key')}")
        errors.extend(
            f"axis:{row.get('job_key')}:{value}"
            for value in row.get("axis_evidence_validation_errors", [])
        )
        errors.extend(
            f"strict:{row.get('job_key')}:{value}"
            for value in row.get("strict_derivation_validation_errors", [])
        )
        if phase == "admission" and not row.get("final_task_success"):
            errors.append(f"admission_clean_failure:{row.get('job_key')}")
        if (
            row.get("condition") == "clean"
            and row.get("final_task_success") is True
            and _observed(row.get("observed_M_consequence"))
        ):
            errors.append(
                f"clean_success_has_M_consequence:{row.get('job_key')}"
            )
    return errors


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"结果目录已存在: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(__file__).with_name(".env.local"), override=False)
    selected_task_ids = tuple(dict.fromkeys(args.task_id or MAIN_TASK_IDS))
    selected_condition_cells = tuple(
        CONDITION_BY_NAME[name]
        for name in (args.condition or tuple(CONDITION_BY_NAME))
    )
    tasks = [
        task
        for task in load_admin_tasks(args.manifest)
        if int(task["task_id"]) in selected_task_ids
    ]
    if {int(task["task_id"]) for task in tasks} != set(selected_task_ids):
        raise ValueError("manifest does not contain all selected matrix tasks")
    all_jobs = build_phase_jobs(
        args.phase,
        tasks,
        schedule_seed=args.schedule_seed,
        repetitions=args.repetitions,
        task_ids=selected_task_ids,
        condition_cells=selected_condition_cells,
    )
    schedule_positions = {
        job.job_key: index for index, job in enumerate(all_jobs, start=1)
    }
    selected_topologies = tuple(args.topology or MAIN_TOPOLOGIES)
    if args.fault_shard_count > 1:
        if args.phase != "formal":
            raise ValueError("fault sharding is supported only for formal phase")
    jobs = prepare_execution_jobs(
        all_jobs,
        topologies=selected_topologies,
        clean_only=args.clean_only,
        fault_shard_count=args.fault_shard_count,
        fault_shard_index=args.fault_shard_index,
    )
    checkpoint = args.output_dir / "valid_runs.checkpoint.jsonl"
    invalid_checkpoint = args.output_dir / "invalid_runs.jsonl"
    rows = _load_jsonl(checkpoint) if args.resume else []
    completed = {str(row["job_key"]) for row in rows}
    carriers = carriers_from_rows(rows)
    carrier_rows = _load_jsonl(args.carrier_source) if args.carrier_source else []
    carriers.extend(carriers_from_rows(carrier_rows))

    client = get_llm_client(mock_llm=False)
    environment = browser_environment(args.shopping_admin_url)
    evaluator = EvaluatorWorkerClient(
        webarena_root=str(args.webarena_root), env=environment
    )
    try:
        with tempfile.TemporaryDirectory(prefix="webarena-admin-main-") as temp:
            sanitized_root = Path(temp)
            for job in jobs:
                if job.job_key in completed:
                    continue
                ensure_deepseek_offpeak(client.model_info.model)
                stale_message = None
                if job.condition_cell.condition in {
                    "semantic_corruption_step4",
                    "stale_replay_step4",
                }:
                    stale_message = _select_stale_carrier(
                        carriers,
                        job,
                        task_ids=selected_task_ids,
                    )

                accepted_row = None
                attempt_history: list[dict[str, Any]] = []
                for attempt in range(1, args.max_attempts_per_job + 1):
                    before = _usage(client)
                    request_log_before = len(getattr(client, "request_log", []))
                    started = time.perf_counter()
                    browser = BrowserWorkerClient(
                        webarena_root=str(args.webarena_root),
                        env=environment,
                        browser_only=True,
                    )
                    try:
                        row = asyncio.run(
                            run_admin_confirmation_task(
                                client,
                                browser,
                                evaluator,
                                job.task,
                                original_config_file=(
                                    args.config_dir / f"{job.task['task_id']}.json"
                                ),
                                sanitized_config_dir=(
                                    sanitized_root
                                    / f"{job.matrix_run_index}-attempt-{attempt}"
                                ),
                                topology=job.topology,
                                condition_cell=job.condition_cell,
                                run_index=job.matrix_run_index,
                                stale_message=stale_message,
                                max_steps=args.max_steps,
                                max_evidence_chars=args.max_evidence_chars,
                            )
                        )
                    except Exception as exc:
                        row = _error_record(
                            client,
                            job,
                            exc,
                            before=before,
                            request_log_before=request_log_before,
                            latency_ms=round(
                                (time.perf_counter() - started) * 1000, 3
                            ),
                        )
                        invalid = bool(
                            _is_infrastructure_error(exc)
                            or (
                                job.condition_cell.condition != "clean"
                                and not row.get("fault_applied")
                            )
                        )
                    else:
                        invalid = not bool(row.get("injection_valid", True))
                    finally:
                        try:
                            browser.close()
                        except Exception:
                            pass

                    row.update(
                        {
                            "job_key": job.job_key,
                            "repeat_index": job.repeat_index,
                            "seed_or_run_index": job.repeat_index,
                            "matrix_run_index": job.matrix_run_index,
                            "schedule_position": schedule_positions[job.job_key],
                            "schedule_seed": args.schedule_seed,
                            "attempt_index": attempt,
                            "infrastructure_invalid": invalid,
                        }
                    )
                    attempt_history.append(row)
                    if invalid and _is_clean_matched_pre_injection_model_failure(
                        rows, row, reference_rows=carrier_rows
                    ):
                        invalid = False
                        row.update(
                            {
                                "infrastructure_invalid": False,
                                "pre_injection_model_failure": True,
                                "fault_effect_attributable": False,
                                "injection_valid": False,
                                "propagation_class": "pre_injection_model_failure",
                            }
                        )
                    if invalid and _is_reproducible_post_fault_failure(
                        attempt_history
                    ):
                        invalid = False
                        row.update(
                            {
                                "infrastructure_invalid": False,
                                "reproducible_post_fault_runtime_failure": True,
                                "runtime_failure_repetitions": 2,
                                "runtime_failure_signature": list(
                                    _post_fault_failure_signature(row) or ()
                                ),
                            }
                        )
                    if invalid:
                        append_jsonl_record(invalid_checkpoint, row)
                        print(
                            f"无效重试 job={job.job_key} attempt={attempt} "
                            f"error={row.get('error')}",
                            flush=True,
                        )
                        continue
                    accepted_row = row
                    break

                if accepted_row is None:
                    raise RuntimeError(
                        f"job exhausted invalid attempts: {job.job_key}"
                    )
                rows.append(accepted_row)
                append_jsonl_record(checkpoint, accepted_row)
                completed.add(job.job_key)
                if accepted_row.get("condition") == "clean":
                    carrier = _extract_carrier(accepted_row)
                    if carrier:
                        carriers.append(
                            {
                                "topology": job.topology,
                                "task_id": str(job.task["task_id"]),
                                "repeat_index": job.repeat_index,
                                "message": carrier,
                            }
                        )
                print(
                    f"完成 {len(rows)}/{len(jobs)} job={job.job_key} "
                    f"class={accepted_row.get('propagation_class')} "
                    f"success={accepted_row.get('final_task_success')}",
                    flush=True,
                )
    finally:
        evaluator.close()

    root = Path(__file__).resolve().parent
    source_paths = [
        root / "src/mas_faults/webarena_admin_confirmation.py",
        root / "src/mas_faults/webarena_admin_main_matrix.py",
        root / "src/mas_faults/webarena_admin_main_evaluator.py",
        root / "src/mas_faults/webarena_admin_topologies.py",
        root / "run_webarena_admin_main_confirmation.py",
    ]
    experiment_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "benchmark": "WebArena Shopping Admin",
        "framework": "AutoGen",
        "topologies": list(selected_topologies),
        "task_ids": list(selected_task_ids),
        "condition_cells": [
            {
                "condition": cell.condition,
                "fault_family": cell.fault_family,
                "injection_step": cell.injection_step,
                "fault_id": cell.fault_id,
                "fault_type": cell.fault_type,
                "fault_cause": cell.fault_cause,
                "parameters": cell.parameters,
            }
            for cell in selected_condition_cells
        ],
        "target_valid_runs": len(jobs),
        "repetitions": (
            args.repetitions
            if args.repetitions is not None
            else (3 if args.phase == "formal" else 1)
        ),
        "schedule_seed": args.schedule_seed,
        "fault_shard_count": args.fault_shard_count,
        "fault_shard_index": args.fault_shard_index,
        "matrix_scope": (
            "clean_only"
            if args.clean_only
            else (
                "fault_shard"
                if args.fault_shard_count > 1
                else "complete_phase"
            )
        ),
        "execution_schedule": (
            "clean_carrier_blocks_first_then_randomized_condition_task_repeat_"
            "blocks_with_all_three_topologies_interleaved"
        ),
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "llm_disable_thinking": True,
        "recovery_policy": "same_natural_behavior_for_all_conditions",
        "post_fault_runtime_failure_rule": (
            "accept_after_three_matching_fault_applied_trace_position_timeouts"
        ),
        "official_final_answer_evaluator": True,
        "deterministic_fuzzy_adapter": (
            "structured month-count tasks only; unsupported semantic fuzzy tasks are rejected"
        ),
        "process_trajectory_type": "controlled_tool_trace",
        "source_sha256": {
            str(path.relative_to(root)): _source_hash(path) for path in source_paths
        },
    }
    summary = write_main_outputs(
        rows, args.output_dir, experiment_config=experiment_config
    )
    errors = _gate_errors(rows, jobs, args.phase)
    (args.output_dir / "matrix_gate.json").write_text(
        json.dumps({"passed": not errors, "errors": errors}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "summary": summary["overall"],
                "gate_errors": errors,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.strict_gate and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
