#!/usr/bin/env python3
"""Seed a repeated WebArena Admin matrix from audited repetition-one rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from mas_faults.webarena_admin_main_evaluator import evaluate_main_outcome
from mas_faults.webarena_admin_main_matrix import CONDITION_BY_NAME, MainMatrixJob
from mas_faults.webarena_admin_real import load_admin_tasks
from run_webarena_admin_main_confirmation import (
    _load_jsonl,
    build_phase_jobs,
    select_fault_shard,
)


DEFAULT_TASK_IDS = (
    0,
    4,
    193,
    194,
    198,
    199,
    208,
    211,
    288,
    292,
    185,
    187,
)
DEFAULT_CONDITIONS = (
    "clean",
    "timeliness_deadline_step4",
    "non_delivery_step2",
    "non_delivery_step4",
    "semantic_corruption_step4",
    "valid_partial_message_step4",
    "stale_replay_step4",
)


def matrix_protocol_counts(
    jobs: list[MainMatrixJob], *, fault_shard_count: int
) -> dict[str, Any]:
    clean = [job for job in jobs if job.condition_cell.condition == "clean"]
    reused = [job for job in jobs if job.repeat_index == 1]
    shard_counts = {
        str(shard): len(
            select_fault_shard(
                jobs, shard_count=fault_shard_count, shard_index=shard
            )
        )
        for shard in range(fault_shard_count)
    }
    return {
        "expected_final_rows": len(jobs),
        "reused_repetition_one_rows": len(reused),
        "new_api_runs": len(jobs) - len(reused),
        "clean_rows": len(clean),
        "clean_rows_per_topology": len(clean) // 3,
        "fault_rows": len(jobs) - len(clean),
        "fault_rows_per_shard": shard_counts,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"seed checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def rebase_repetition_one_rows(
    source_rows: list[dict[str, Any]],
    jobs: list[MainMatrixJob],
    *,
    source_result_dir: str,
    schedule_seed: int,
    expected_model: str = "deepseek-v4-pro",
    expected_provider: str = "deepseek",
) -> list[dict[str, Any]]:
    """Select exact rep-1 rows, reevaluate them, and map them to a new schedule."""
    source_keys = [str(row.get("job_key")) for row in source_rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("duplicate source job keys")
    source_by_key = dict(zip(source_keys, source_rows))
    rep_one_jobs = [job for job in jobs if job.repeat_index == 1]
    expected_keys = {job.job_key for job in rep_one_jobs}
    missing = expected_keys - set(source_by_key)
    if missing:
        raise ValueError(
            f"missing repetition 1 job keys: {sorted(missing)[:5]}"
        )
    positions = {job.job_key: index for index, job in enumerate(jobs, start=1)}
    job_by_key = {job.job_key: job for job in rep_one_jobs}
    rebased = []
    for job in rep_one_jobs:
        source = source_by_key[job.job_key]
        row = evaluate_main_outcome(source)
        row.update(
            {
                "job_key": job.job_key,
                "repeat_index": 1,
                "seed_or_run_index": 1,
                "matrix_run_index": job.matrix_run_index,
                "schedule_position": positions[job.job_key],
                "schedule_seed": schedule_seed,
                "reused_from_previous_matrix": True,
                "reused_row_reevaluated": True,
                "source_result_dir": source_result_dir,
                "source_run_id": source.get("run_id"),
                "source_trace_id": source.get("trace_id"),
            }
        )
        if (
            row.get("model") != expected_model
            or row.get("provider") != expected_provider
        ):
            raise ValueError(f"unexpected source model/provider: {job.job_key}")
        if str(row.get("task_id")) != str(job.task["task_id"]):
            raise ValueError(f"source task mismatch: {job.job_key}")
        rebased.append(row)
    if len(job_by_key) != len(rebased):
        raise ValueError("duplicate repetition 1 jobs")
    return sorted(rebased, key=lambda row: int(row["schedule_position"]))


def repartition_fault_rows(
    input_rows: list[dict[str, Any]],
    jobs: list[MainMatrixJob],
    *,
    schedule_seed: int,
    fault_shard_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rebase and repartition valid fault rows after a shard-schedule mismatch."""
    fault_jobs = [
        job for job in jobs if job.condition_cell.condition != "clean"
    ]
    job_by_key = {job.job_key: job for job in fault_jobs}
    positions = {job.job_key: index for index, job in enumerate(jobs, start=1)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    unexpected = []
    for row in input_rows:
        key = str(row.get("job_key"))
        if key not in job_by_key:
            unexpected.append(key)
            continue
        grouped.setdefault(key, []).append(row)
    if unexpected:
        raise ValueError(f"unexpected fault job keys: {sorted(set(unexpected))[:5]}")

    selected: dict[str, dict[str, Any]] = {}
    duplicate_rows_excluded = 0
    for key, candidates in grouped.items():
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            reused = [
                row
                for row in candidates
                if bool(row.get("reused_from_previous_matrix"))
            ]
            if len(reused) != 1:
                raise ValueError(f"ambiguous duplicate fault job key: {key}")
            chosen = reused[0]
            duplicate_rows_excluded += len(candidates) - 1
        job = job_by_key[key]
        row = evaluate_main_outcome(chosen)
        row.update(
            {
                "job_key": key,
                "repeat_index": job.repeat_index,
                "seed_or_run_index": job.repeat_index,
                "matrix_run_index": job.matrix_run_index,
                "schedule_position": positions[key],
                "schedule_seed": schedule_seed,
                "repartitioned_after_schedule_alignment": True,
            }
        )
        selected[key] = row

    partitions: dict[str, list[dict[str, Any]]] = {}
    assigned: set[str] = set()
    for shard in range(fault_shard_count):
        shard_keys = {
            job.job_key
            for job in select_fault_shard(
                jobs, shard_count=fault_shard_count, shard_index=shard
            )
        }
        rows = sorted(
            (selected[key] for key in shard_keys if key in selected),
            key=lambda row: int(row["schedule_position"]),
        )
        partitions[str(shard)] = rows
        assigned.update(str(row["job_key"]) for row in rows)
    if assigned != set(selected):
        raise ValueError("repartitioned fault rows are incomplete")
    audit = {
        "passed": True,
        "input_rows": len(input_rows),
        "unique_rows_preserved": len(selected),
        "duplicate_rows_excluded": duplicate_rows_excluded,
        "reused_rows_preserved": sum(
            bool(row.get("reused_from_previous_matrix"))
            for row in selected.values()
        ),
        "new_api_rows_preserved": sum(
            not bool(row.get("reused_from_previous_matrix"))
            for row in selected.values()
        ),
        "fault_shard_counts": {
            shard: len(rows) for shard, rows in partitions.items()
        },
        "schedule_seed": schedule_seed,
    }
    return partitions, audit


def prepare_seed_checkpoints(
    source_rows: list[dict[str, Any]],
    jobs: list[MainMatrixJob],
    *,
    output_root: Path,
    source_result_dir: str,
    schedule_seed: int,
    fault_shard_count: int,
    expected_model: str = "deepseek-v4-pro",
    expected_provider: str = "deepseek",
) -> dict[str, Any]:
    """Write restart-safe checkpoints that let the existing runner skip rep 1."""
    seed_rows = rebase_repetition_one_rows(
        source_rows,
        jobs,
        source_result_dir=source_result_dir,
        schedule_seed=schedule_seed,
        expected_model=expected_model,
        expected_provider=expected_provider,
    )
    row_by_key = {str(row["job_key"]): row for row in seed_rows}
    clean_counts: dict[str, int] = {}
    for topology in ("sequential", "flat", "hierarchical"):
        rows = [
            row
            for row in seed_rows
            if row.get("condition") == "clean" and row.get("topology") == topology
        ]
        clean_counts[topology] = len(rows)
        _write_jsonl(
            output_root
            / "clean_parallel"
            / topology
            / "valid_runs.checkpoint.jsonl",
            rows,
        )

    fault_counts: dict[str, int] = {}
    fault_keys_seen: set[str] = set()
    for shard in range(fault_shard_count):
        shard_jobs = select_fault_shard(
            jobs, shard_count=fault_shard_count, shard_index=shard
        )
        keys = {
            job.job_key for job in shard_jobs if job.repeat_index == 1
        }
        rows = sorted(
            (row_by_key[key] for key in keys),
            key=lambda row: int(row["schedule_position"]),
        )
        if fault_keys_seen.intersection(keys):
            raise ValueError("fault seed shards overlap")
        fault_keys_seen.update(keys)
        fault_counts[str(shard)] = len(rows)
        _write_jsonl(
            output_root / f"fault_shard_{shard}" / "valid_runs.checkpoint.jsonl",
            rows,
        )

    expected_fault_seed_keys = {
        job.job_key
        for job in jobs
        if job.repeat_index == 1 and job.condition_cell.condition != "clean"
    }
    if fault_keys_seen != expected_fault_seed_keys:
        raise ValueError("fault seed shards are incomplete")
    audit = {
        "passed": True,
        "source_result_dir": source_result_dir,
        "source_repetition_one_rows": len(seed_rows),
        "expected_final_rows": len(jobs),
        "new_api_runs": len(jobs) - len(seed_rows),
        "clean_seed_counts": clean_counts,
        "fault_seed_counts": fault_counts,
        "schedule_seed": schedule_seed,
        "fault_shard_count": fault_shard_count,
        "expected_model": expected_model,
        "expected_provider": expected_provider,
        "task_ids": sorted({int(job.task["task_id"]) for job in jobs}),
        "conditions": sorted({job.condition_cell.condition for job in jobs}),
        "repetitions": sorted({job.repeat_index for job in jobs}),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "seed_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-final", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule-seed", type=int, default=20260817)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--fault-shard-count", type=int, default=3)
    parser.add_argument("--expected-model", default="deepseek-v4-pro")
    parser.add_argument("--expected-provider", default="deepseek")
    parser.add_argument("--task-id", action="append", type=int)
    parser.add_argument("--condition", action="append", choices=tuple(CONDITION_BY_NAME))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = tuple(dict.fromkeys(args.task_id or DEFAULT_TASK_IDS))
    condition_names = tuple(dict.fromkeys(args.condition or DEFAULT_CONDITIONS))
    tasks = [
        task
        for task in load_admin_tasks(args.manifest)
        if int(task["task_id"]) in task_ids
    ]
    if {int(task["task_id"]) for task in tasks} != set(task_ids):
        raise ValueError("manifest does not contain all selected tasks")
    jobs = build_phase_jobs(
        "formal",
        tasks,
        schedule_seed=args.schedule_seed,
        repetitions=args.repetitions,
        task_ids=task_ids,
        condition_cells=tuple(CONDITION_BY_NAME[name] for name in condition_names),
    )
    source_trace = args.source_final / "llm_communication_traces.jsonl"
    audit = prepare_seed_checkpoints(
        _load_jsonl(source_trace),
        jobs,
        output_root=args.output_root,
        source_result_dir=str(args.source_final),
        schedule_seed=args.schedule_seed,
        fault_shard_count=args.fault_shard_count,
        expected_model=args.expected_model,
        expected_provider=args.expected_provider,
    )
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
