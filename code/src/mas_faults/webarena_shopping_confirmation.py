"""Freeze and execute the targeted WebArena Shopping cross-repetition matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from mas_faults.webarena_task_selection import load_task_manifest


@dataclass(frozen=True)
class ConfirmationBatch:
    fault_code: str
    fault_type: str
    steps: tuple[int, ...]

    @property
    def slug(self) -> str:
        return f"{self.fault_code}_{self.fault_type}_step{'-'.join(str(step) for step in self.steps)}"


CONFIRMATION_BATCHES = (
    ConfirmationBatch("none", "none", (2,)),
    ConfirmationBatch("A4", "endpoint_unavailable", (4,)),
    ConfirmationBatch("A5", "omission", (4,)),
    ConfirmationBatch("A9", "duplicate_request", (2,)),
    ConfirmationBatch("A11", "schema_mismatch", (4,)),
    ConfirmationBatch("A12", "stale_replay", (4,)),
    ConfirmationBatch("A15", "contract_violation", (3, 4)),
)


def build_execution_environment(
    *,
    model: str,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Freeze model behavior for repetitions that must match the discovery matrix."""
    environment = dict(os.environ if base_environment is None else base_environment)
    environment["LLM_MODEL"] = model
    environment["LLM_DISABLE_THINKING"] = "1"
    return environment


def build_confirmation_manifest(
    *,
    task_ids: Sequence[str],
    topologies: Sequence[str],
    repeat_start: int,
    repeat_count: int,
    model: str,
) -> dict[str, Any]:
    units = [
        {
            "task_id": task_id,
            "topology": topology,
            "fault_code": batch.fault_code,
            "fault_type": batch.fault_type,
            "injection_step": step,
            "repeat_index": repeat_index,
        }
        for repeat_index in range(repeat_start, repeat_start + repeat_count)
        for topology in topologies
        for task_id in task_ids
        for batch in CONFIRMATION_BATCHES
        for step in batch.steps
    ]
    return {
        "experiment": "webarena_shopping_targeted_cross_repetition_confirmation",
        "model": model,
        "provider": "deepseek",
        "framework": "AutoGen",
        "disable_thinking": True,
        "repeat_semantics": "independent API repetition; model seed is not controllable",
        "repeat_start": repeat_start,
        "repeat_count": repeat_count,
        "topologies": list(topologies),
        "task_ids": list(task_ids),
        "fault_cells": [
            {"fault_code": batch.fault_code, "fault_type": batch.fault_type, "steps": list(batch.steps)}
            for batch in CONFIRMATION_BATCHES
        ],
        "unit_count": len(units),
        "run_count": len(units),
        "units": units,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_runner_commands(
    *,
    repo_root: Path,
    output_root: Path,
    task_manifest: Path,
    topologies: Sequence[str],
    task_count: int,
    repeat_start: int,
    repeat_count: int,
    model: str,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for batch in CONFIRMATION_BATCHES:
        commands.append([
            sys.executable,
            str(repo_root / "run_webarena_architecture_rq2.py"),
            "--base-url", "http://localhost:7770",
            "--tasks", str(task_count),
            "--runs-per-condition", str(repeat_count),
            "--run-index-start", str(repeat_start),
            "--task-manifest", str(task_manifest),
            "--topologies", ",".join(topologies),
            "--faults", batch.fault_type,
            "--steps", ",".join(str(step) for step in batch.steps),
            "--fault-step-registry",
            "--required-model", model,
            "--resume",
            "--output-dir", str(output_root / batch.slug),
        ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结并执行 WebArena Shopping 目标多重复确认矩阵。")
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--task-count", type=int, default=5)
    parser.add_argument("--topologies", default="flat,hierarchical,team,hybrid")
    parser.add_argument("--repeat-start", type=int, default=2)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    topologies = tuple(value for value in args.topologies.split(",") if value)
    tasks = load_task_manifest(args.task_manifest)[: args.task_count]
    if len(tasks) != args.task_count:
        raise SystemExit(f"task manifest only provided {len(tasks)} tasks; expected {args.task_count}")
    manifest = build_confirmation_manifest(
        task_ids=[str(task["task_id"]) for task in tasks],
        topologies=topologies,
        repeat_start=args.repeat_start,
        repeat_count=args.repeat_count,
        model=args.model,
    )
    commands = build_runner_commands(
        repo_root=Path(__file__).resolve().parent,
        output_root=args.output_root,
        task_manifest=args.task_manifest,
        topologies=topologies,
        task_count=args.task_count,
        repeat_start=args.repeat_start,
        repeat_count=args.repeat_count,
        model=args.model,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "confirmation_matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "driver_commands.json").write_text(
        json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"FROZEN units={manifest['unit_count']} output={args.output_root}", flush=True)
    if not args.execute:
        return

    environment = build_execution_environment(model=args.model)
    for index, command in enumerate(commands, start=1):
        print(f"BATCH {index}/{len(commands)} {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=Path(__file__).resolve().parent, env=environment, check=True)
    print(f"COMPLETE output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
