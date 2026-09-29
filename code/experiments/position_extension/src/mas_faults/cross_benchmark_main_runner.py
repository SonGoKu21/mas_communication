"""CLI runner for the shared SWE/TAC Flash main confirmation matrix."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mas_faults.causal_trace_report import build_report
from mas_faults.cross_benchmark_main_confirmation import run_matrix_job
from mas_faults.cross_benchmark_main_matrix import (
    FLASH_MAIN_CONDITIONS,
    MAIN_TOPOLOGIES,
    BenchmarkCarrier,
    MatrixJob,
    build_execution_schedule,
    load_admitted_carriers,
    select_stale_carrier,
)
from mas_faults.deepseek_schedule import ensure_deepseek_offpeak
from mas_faults.llm_client import get_llm_client


DEFAULT_REQUIRED_MODEL = "deepseek-v4-flash"


def validate_required_model(client: Any, required_model: str) -> None:
    actual_model = client.model_info.model
    if actual_model != required_model:
        raise SystemExit(f"expected {required_model}, got {actual_model}")


def order_schedule(
    schedule: Sequence[MatrixJob], *, seed: int
) -> tuple[MatrixJob, ...]:
    clean = [job for job in schedule if job.condition == "clean"]
    faults = [job for job in schedule if job.condition != "clean"]
    random.Random(seed).shuffle(faults)
    return tuple(clean + faults)


def partition_schedule(
    schedule: Sequence[MatrixJob], shard_count: int, shard_index: int
) -> tuple[MatrixJob, ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard configuration")
    return tuple(
        job for index, job in enumerate(schedule) if index % shard_count == shard_index
    )


def select_repeat_indices(
    schedule: Sequence[MatrixJob], repeat_indices: Iterable[int]
) -> tuple[MatrixJob, ...]:
    selected = tuple(dict.fromkeys(int(index) for index in repeat_indices))
    if not selected or any(index < 1 for index in selected):
        raise ValueError("repeat indices must be positive integers")
    available = {job.repeat_index for job in schedule}
    missing = set(selected) - available
    if missing:
        raise ValueError(f"repeat indices are outside the schedule: {sorted(missing)}")
    return tuple(job for job in schedule if job.repeat_index in selected)


def _exposed(value: Any) -> bool:
    return value not in (None, "none", ["none"], [])


def _transition_path(row: dict[str, Any]) -> str:
    if not bool(row.get("fault_applied")):
        return "clean"
    a_exposed = _exposed(row.get("observed_A_symptom"))
    m_exposed = _exposed(row.get("observed_M_consequence"))
    recovered = bool(row.get("recovery_detected"))
    final_success = bool(row.get("final_task_success"))
    if not a_exposed and not m_exposed:
        return "fault_injected_to_masked"
    if m_exposed:
        if recovered and final_success:
            return "fault_injected_to_propagated_to_M_to_recovered"
        if not final_success:
            return "fault_injected_to_propagated_to_M_to_final_failure"
        return "fault_injected_to_propagated_to_M"
    if recovered and final_success:
        return "fault_injected_to_exposed_at_A_only_to_recovered"
    if not final_success:
        return "fault_injected_to_exposed_at_A_only_to_final_failure"
    return "fault_injected_to_exposed_at_A_only"


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "runs": 0,
            "A_exposure_count": 0,
            "M_consequence_count": 0,
            "recovery_count": 0,
            "final_success_count": 0,
            "final_failure_count": 0,
        }
    )
    by_topology: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "runs": 0,
            "A_exposure_count": 0,
            "M_consequence_count": 0,
            "recovery_count": 0,
            "final_success_count": 0,
            "final_failure_count": 0,
        }
    )
    for row in rows:
        for bucket in (by_condition[str(row["condition"])], by_topology[str(row["topology"])]):
            bucket["runs"] += 1
            bucket["A_exposure_count"] += int(_exposed(row.get("observed_A_symptom")))
            bucket["M_consequence_count"] += int(
                _exposed(row.get("observed_M_consequence"))
            )
            bucket["recovery_count"] += int(bool(row.get("recovery_detected")))
            bucket["final_success_count"] += int(bool(row.get("final_task_success")))
            bucket["final_failure_count"] += int(not bool(row.get("final_task_success")))
    count = len(rows)
    fault_rows = [row for row in rows if bool(row.get("fault_applied"))]
    fault_count = len(fault_rows)
    a_exposure_count = sum(_exposed(row.get("observed_A_symptom")) for row in rows)
    m_consequence_count = sum(
        _exposed(row.get("observed_M_consequence")) for row in rows
    )
    recovery_count = sum(bool(row.get("recovery_detected")) for row in rows)
    final_success_count = sum(bool(row.get("final_task_success")) for row in rows)
    final_failure_count = count - final_success_count

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "runs": count,
        "fault_run_count": fault_count,
        "fault_applied_count": fault_count,
        "A_exposure_count": a_exposure_count,
        "M_consequence_count": m_consequence_count,
        "recovery_count": recovery_count,
        "final_success_count": final_success_count,
        "final_failure_count": final_failure_count,
        "final_task_success_rate_all_runs": rate(final_success_count, count),
        "A_layer_exposure_rate_fault_runs": rate(
            sum(_exposed(row.get("observed_A_symptom")) for row in fault_rows),
            fault_count,
        ),
        "M_layer_propagation_rate_fault_runs": rate(
            sum(
                _exposed(row.get("observed_M_consequence"))
                for row in fault_rows
            ),
            fault_count,
        ),
        "recovery_rate_fault_runs": rate(
            sum(bool(row.get("recovery_detected")) for row in fault_rows),
            fault_count,
        ),
        "final_failure_rate_fault_runs": rate(
            sum(not bool(row.get("final_task_success")) for row in fault_rows),
            fault_count,
        ),
        "mean_latency_ms": statistics.mean(
            float(row.get("latency_ms") or 0) for row in rows
        ) if rows else 0.0,
        "mean_total_tokens": statistics.mean(
            int(row.get("total_tokens") or 0) for row in rows
        ) if rows else 0.0,
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "api_call_count": sum(int(row.get("api_call_count") or 0) for row in rows),
        "propagation_class_counts": dict(
            Counter(str(row.get("propagation_class")) for row in rows)
        ),
        "transition_counts": dict(
            sorted(Counter(_transition_path(row) for row in rows).items())
        ),
        "by_condition": dict(sorted(by_condition.items())),
        "by_topology": dict(sorted(by_topology.items())),
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-benchmark Flash main confirmation",
        "",
        f"- Runs: {summary['runs']}",
        f"- Fault applied: {summary['fault_applied_count']}",
        f"- A exposure: {summary['A_exposure_count']}",
        f"- M propagation: {summary['M_consequence_count']}",
        f"- Recovery: {summary['recovery_count']}",
        f"- Final success: {summary['final_success_count']}",
        f"- Final failure: {summary['final_failure_count']}",
        f"- Fault-only A exposure rate: {summary['A_layer_exposure_rate_fault_runs']:.2%}",
        f"- Fault-only M propagation rate: {summary['M_layer_propagation_rate_fault_runs']:.2%}",
        f"- Fault-only recovery rate: {summary['recovery_rate_fault_runs']:.2%}",
        f"- Fault-only final failure rate: {summary['final_failure_rate_fault_runs']:.2%}",
        "",
        "| Condition | Runs | A | M | Recovery | Success | Failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in summary["by_condition"].items():
        lines.append(
            f"| {condition} | {values['runs']} | {values['A_exposure_count']} | "
            f"{values['M_consequence_count']} | {values['recovery_count']} | "
            f"{values['final_success_count']} | {values['final_failure_count']} |"
        )
    return "\n".join(lines) + "\n"


def canonical_causal_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert run-local events to the strict project-wide causal schema."""
    trace_id = str(row["trace_id"])
    task_id = str(row["task_id"])
    pair_id = f"{row['benchmark']}:{task_id}:r{row.get('repeat_index', 1)}"
    trace_variant = "fault" if row.get("fault_applied") else "clean"
    canonical: list[dict[str, Any]] = []
    parent_span_id = None
    for index, raw in enumerate(row.get("events", []), start=1):
        span_id = f"{trace_id}:span-{index:03d}"
        fault_code = raw.get("fault_id") if raw.get("fault_applied") else None
        timestamp = raw.get("send_timestamp") or datetime.now(timezone.utc).isoformat()
        logical_message_id = raw.get("message_id") or f"{row['run_id']}:message-{index}"
        canonical.append(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "fault_id": fault_code or "none",
                "carrier_id": f"carrier-{row['run_id']}",
                "carrier_instance_id": f"{logical_message_id}#{index}",
                "duplicate_index": 0,
                "logical_message_id": logical_message_id,
                "pair_id": pair_id,
                "trace_variant": trace_variant,
                "timestamp": timestamp,
                "event_layer": "A",
                "component": raw.get("target_agent") or "communication_interceptor",
                "event_type": raw.get("event_type") or "communication_intercepted",
                "event_status": "fault_applied" if fault_code else "delivered",
                "source": raw.get("source_agent"),
                "target": raw.get("target_agent"),
                "injection_operator_code": fault_code,
                "injection_point_kind": "communication_interceptor" if fault_code else None,
                "injected_fault_code": fault_code,
                "injected_fault_layer": fault_code[0] if fault_code else None,
                "expected_manifest_code": fault_code,
                "expected_manifest_layer": fault_code[0] if fault_code else None,
                "observed_effect": raw.get("observed_runtime_effect"),
                "propagation_label": "injected" if fault_code else "pre_injection",
                "evidence": {
                    "benchmark": row.get("benchmark"),
                    "task_id": task_id,
                    "condition": row.get("condition"),
                    "topology": row.get("topology"),
                    "abstract_step": raw.get("abstract_step"),
                    "delivery_count": raw.get("delivery_count"),
                    "observed_A_symptom": raw.get("observed_A_symptom"),
                    "original_message": raw.get("original_message"),
                    "delivered_message": raw.get("delivered_message"),
                },
            }
        )
        parent_span_id = span_id

    m_values = row.get("observed_M_consequence") or ["none"]
    m_exposed = _exposed(m_values)
    fault_code = str(row.get("fault_id") or "") if row.get("fault_applied") else None
    final_span = f"{trace_id}:span-{len(canonical) + 1:03d}"
    canonical.append(
        {
            "trace_id": trace_id,
            "span_id": final_span,
            "parent_span_id": parent_span_id,
            "fault_id": fault_code or "none",
            "carrier_id": f"carrier-{row['run_id']}",
            "carrier_instance_id": f"{row['run_id']}:final#0",
            "duplicate_index": 0,
            "logical_message_id": f"{row['run_id']}:final",
            "pair_id": pair_id,
            "trace_variant": trace_variant,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_layer": "M",
            "component": "task_evaluator",
            "event_type": "final_consequence",
            "event_status": "success" if row.get("final_task_success") else "failure",
            "source": "task_evaluator",
            "target": "MAS",
            "injection_operator_code": fault_code,
            "injection_point_kind": "communication_interceptor" if fault_code else None,
            "injected_fault_code": fault_code,
            "injected_fault_layer": fault_code[0] if fault_code else None,
            "expected_manifest_code": fault_code,
            "expected_manifest_layer": fault_code[0] if fault_code else None,
            "observed_effect": ",".join(str(value) for value in m_values),
            "propagation_label": (
                "propagated"
                if m_exposed
                else "masked"
                if row.get("fault_applied")
                else "pre_injection"
            ),
            "evidence": {
                "benchmark": row.get("benchmark"),
                "task_id": task_id,
                "condition": row.get("condition"),
                "topology": row.get("topology"),
                "observed_M_consequence": m_values,
                "recovery_detected": row.get("recovery_detected"),
                "final_task_success": row.get("final_task_success"),
            },
        }
    )
    return canonical


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize_rows(rows)
    (output_dir / "main_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "main_summary.md").write_text(
        _render_summary(summary), encoding="utf-8"
    )
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_dir / "main_runs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
    events_path = output_dir / "causal_events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for enriched in canonical_causal_events(row):
                handle.write(
                    json.dumps(enriched, ensure_ascii=False, sort_keys=True) + "\n"
                )
    if events_path.stat().st_size:
        build_report(events_path, output_dir / "causal_trace_report")


def _select_carriers(
    carriers: Sequence[BenchmarkCarrier], task_ids: Sequence[str], target_tasks: int
) -> tuple[BenchmarkCarrier, ...]:
    selected = [carrier for carrier in carriers if not task_ids or carrier.task_id in task_ids]
    if task_ids:
        missing = set(task_ids) - {carrier.task_id for carrier in selected}
        if missing:
            raise ValueError(f"requested tasks are not admitted: {sorted(missing)}")
    selected = selected[:target_tasks]
    if len(selected) != target_tasks:
        raise ValueError(
            f"need exactly {target_tasks} admitted tasks, found {len(selected)}"
        )
    return tuple(selected)


def resolve_stale_carrier_pool(
    source_carriers: Sequence[BenchmarkCarrier],
    scheduled_carriers: Sequence[BenchmarkCarrier],
) -> tuple[BenchmarkCarrier, ...]:
    """Keep admitted unselected tasks available only as stale replay sources."""
    source_ids = {carrier.task_id for carrier in source_carriers}
    scheduled_ids = {carrier.task_id for carrier in scheduled_carriers}
    missing = scheduled_ids - source_ids
    if missing:
        raise ValueError(f"scheduled carriers missing from source pool: {sorted(missing)}")
    return tuple(source_carriers)


async def _execute(
    client: Any,
    *,
    schedule: Sequence[MatrixJob],
    carriers: Sequence[BenchmarkCarrier],
    runs_path: Path,
    errors_path: Path,
    completed: set[str],
    max_attempts: int,
) -> None:
    by_task = {carrier.task_id: carrier for carrier in carriers}
    for job in schedule:
        if job.run_id in completed:
            continue
        ensure_deepseek_offpeak(client.model_info.model)
        carrier = by_task[job.task_id]
        stale = select_stale_carrier(carrier, carriers)
        for attempt in range(1, max_attempts + 1):
            try:
                row = await run_matrix_job(
                    client, carrier=carrier, stale_carrier=stale, job=job
                )
                _append_jsonl(runs_path, row)
                completed.add(job.run_id)
                break
            except Exception as exc:
                _append_jsonl(
                    errors_path,
                    {
                        "run_id": job.run_id,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the shared SWE/TAC Flash main confirmation matrix."
    )
    parser.add_argument(
        "--benchmark",
        choices=["SWE-bench Verified", "TheAgentCompany", "WebArena Reddit"],
        required=True,
    )
    parser.add_argument("--source-jsonl", action="append", type=Path, required=True)
    parser.add_argument(
        "--source-model",
        action="append",
        default=[],
        help=(
            "Clean-admission model accepted in source JSONL; repeat for multiple "
            "models. Defaults to deepseek-v4-flash for backward compatibility."
        ),
    )
    parser.add_argument("--task-ids", nargs="*", default=[])
    parser.add_argument("--target-tasks", type=int, default=12)
    parser.add_argument("--topologies", nargs="+", default=list(MAIN_TOPOLOGIES))
    parser.add_argument("--conditions", nargs="+", default=list(FLASH_MAIN_CONDITIONS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-indices", nargs="+", type=int)
    parser.add_argument("--schedule-seed", type=int, default=20260817)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--required-model", default=DEFAULT_REQUIRED_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_models = tuple(args.source_model or ("deepseek-v4-flash",))
    source_carriers = load_admitted_carriers(
        args.source_jsonl,
        benchmark=args.benchmark,
        allowed_source_models=source_models,
    )
    carriers = _select_carriers(
        source_carriers,
        args.task_ids,
        args.target_tasks,
    )
    stale_carriers = resolve_stale_carrier_pool(source_carriers, carriers)
    unfiltered_schedule = order_schedule(
        build_execution_schedule(
            carriers,
            topologies=args.topologies,
            conditions=args.conditions,
            repeats=args.repeats,
        ),
        seed=args.schedule_seed,
    )
    repeat_indices = tuple(
        args.repeat_indices or range(1, args.repeats + 1)
    )
    full_schedule = select_repeat_indices(
        unfiltered_schedule, repeat_indices
    )
    schedule = partition_schedule(
        full_schedule, args.shard_count, args.shard_index
    )
    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "model": args.required_model,
        "provider": "pending_client_initialization",
        "topologies": args.topologies,
        "task_ids": [carrier.task_id for carrier in carriers],
        "conditions": args.conditions,
        "repeats": args.repeats,
        "repeat_indices": list(repeat_indices),
        "target_unfiltered_runs": len(unfiltered_schedule),
        "target_full_runs": len(full_schedule),
        "target_shard_runs": len(schedule),
        "schedule_seed": args.schedule_seed,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "source_jsonl": [str(path) for path in args.source_jsonl],
        "source_models": list(source_models),
        "stale_source_task_ids": [carrier.task_id for carrier in stale_carriers],
        "matrix_scope": "real clean-admitted benchmark evidence with repeated AutoGen communication topology suffix",
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    runs_path = args.output_dir / "main_runs.jsonl"
    errors_path = args.output_dir / "run_errors.jsonl"
    existing = _read_jsonl(runs_path)
    completed = {str(row["run_id"]) for row in existing}
    client = get_llm_client()
    validate_required_model(client, args.required_model)
    config["model"] = client.model_info.model
    config["provider"] = client.model_info.provider
    config["base_url"] = client.model_info.base_url
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    asyncio.run(
        _execute(
            client,
            schedule=schedule,
            carriers=stale_carriers,
            runs_path=runs_path,
            errors_path=errors_path,
            completed=completed,
            max_attempts=args.max_attempts,
        )
    )
    rows = _read_jsonl(runs_path)
    _write_outputs(args.output_dir, rows)
    summary = summarize_rows(rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if len(rows) != len(schedule):
        raise SystemExit(
            f"incomplete shard: expected {len(schedule)} successful rows, got {len(rows)}"
        )


if __name__ == "__main__":
    main()
