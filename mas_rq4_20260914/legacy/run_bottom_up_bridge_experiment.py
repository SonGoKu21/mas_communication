from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from mas_faults.bottom_up_bridge import (
    BRIDGE_CONDITIONS,
    BridgeRunSpec,
    build_bridge_specs,
    enrich_run_record,
    probe_lower_layer,
    write_bridge_outputs,
)
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_task_selection import load_task_manifest
from run_webarena_architecture_rq2 import run_one, run_with_retries


DEFAULT_TASK_IDS = (
    "shopping-001-q1",
    "shopping-005-q2",
    "shopping-009-q3",
)
DEFAULT_TOPOLOGIES = ("sequential", "flat")
DEFAULT_TASK_MANIFEST = Path(
    "/data2/system5/mas/manifests/webarena_shopping_rq1_30_tasks_v2_preflight.json"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_frozen_tasks(
    available: Sequence[dict[str, Any]],
    requested_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id = {str(task["task_id"]): task for task in available}
    missing = [task_id for task_id in requested_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"missing frozen task(s): {', '.join(missing)}")
    return [dict(by_id[task_id]) for task_id in requested_ids]


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_ids = [str(row["run_id"]) for row in rows]
    job_keys = [str(row["job_key"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("bridge checkpoint contains duplicate run_id values")
    if len(job_keys) != len(set(job_keys)):
        raise ValueError("bridge checkpoint contains duplicate job_key values")
    return rows


def pending_specs(
    specs: Sequence[BridgeRunSpec],
    rows: Sequence[dict[str, Any]],
) -> list[BridgeRunSpec]:
    completed = {str(row["job_key"]) for row in rows}
    return [spec for spec in specs if spec.job_key not in completed]


def validate_local_model(
    client: Any,
    *,
    required_model: str,
    required_provider: str,
) -> None:
    provider = str(client.model_info.provider)
    model = str(client.model_info.model)
    if provider != required_provider:
        raise ValueError(
            f"required provider {required_provider!r}, configured provider is {provider!r}; "
            "the bridge experiment must not use a paid API"
        )
    if model != required_model:
        raise ValueError(f"required model {required_model!r}, configured model is {model!r}")


async def execute_bridge_spec(
    *,
    client: Any,
    task: dict[str, Any],
    spec: BridgeRunSpec,
    base_url: str,
) -> dict[str, Any]:
    probe_payload = {
        "task_id": task["task_id"],
        "product_title": task["product_title"],
        "requested_quantity": task["quantity"],
        "cart_verified": True,
    }
    exposure = await probe_lower_layer(
        spec.condition,
        probe_payload,
        seed=spec.repeat_index,
    )
    base_row = await run_one(
        client,
        task,
        spec.topology,
        exposure.a_operator,
        spec.injection_step,
        spec.repeat_index,
        base_url,
    )
    return enrich_run_record(base_row, exposure, spec)


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_manifest(
    *,
    specs: Sequence[BridgeRunSpec],
    tasks: Sequence[dict[str, Any]],
    model: str,
    provider: str,
    task_manifest: Path,
) -> dict[str, Any]:
    return {
        "experiment": "bottom_up_webarena_shopping_bridge",
        "framework": "AutoGen",
        "model": model,
        "provider": provider,
        "paid_api_calls_allowed": False,
        "task_manifest": str(task_manifest),
        "tasks": list(tasks),
        "conditions": [condition.name for condition in BRIDGE_CONDITIONS],
        "unit_count": len(specs),
        "units": [
            {
                "job_key": spec.job_key,
                "task_id": spec.task_id,
                "topology": spec.topology,
                "condition": spec.condition,
                "injection_step": spec.injection_step,
                "repeat_index": spec.repeat_index,
            }
            for spec in specs
        ],
        "created_at": now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local P/D/N/T-to-A-to-M WebArena bridge experiment."
    )
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--topologies", default=",".join(DEFAULT_TOPOLOGIES))
    parser.add_argument(
        "--conditions",
        default=",".join(condition.name for condition in BRIDGE_CONDITIONS),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--base-url", default="http://localhost:7770")
    parser.add_argument("--required-model", default="Qwen3.8-27B")
    parser.add_argument("--required-provider", default="modelscope_local")
    parser.add_argument("--llm-run-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    load_dotenv(".env.local")

    task_ids = tuple(args.task_ids or DEFAULT_TASK_IDS)
    topologies = tuple(value for value in args.topologies.split(",") if value)
    conditions = tuple(value for value in args.conditions.split(",") if value)
    available = load_task_manifest(args.task_manifest)
    tasks = select_frozen_tasks(available, task_ids)
    specs = build_bridge_specs(
        tasks=tasks,
        topologies=topologies,
        repetitions=args.repetitions,
        conditions=conditions,
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "bridge_full.checkpoint.jsonl"
    rows = load_checkpoint_rows(checkpoint) if args.resume else []
    remaining = pending_specs(specs, rows)

    client = get_llm_client()
    validate_local_model(
        client,
        required_model=args.required_model,
        required_provider=args.required_provider,
    )
    manifest = build_manifest(
        specs=specs,
        tasks=tasks,
        model=client.model_info.model,
        provider=client.model_info.provider,
        task_manifest=args.task_manifest,
    )
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    for index, spec in enumerate(remaining, start=1):
        row = asyncio.run(
            run_with_retries(
                lambda spec=spec: execute_bridge_spec(
                    client=client,
                    task=tasks_by_id[spec.task_id],
                    spec=spec,
                    base_url=args.base_url,
                ),
                attempts=args.llm_run_attempts,
                delay_seconds=args.retry_delay_seconds,
            )
        )
        append_checkpoint(checkpoint, row)
        rows.append(row)
        print(
            f"DONE {len(rows)}/{len(specs)} pending_index={index}/{len(remaining)} "
            f"topology={spec.topology} task={spec.task_id} condition={spec.condition} "
            f"repeat={spec.repeat_index} A={row['observed_A_symptom']} "
            f"M={row['observed_M_consequence']} success={row['final_task_success']}",
            flush=True,
        )

    experiment_config = {
        "experiment": "bottom_up_webarena_shopping_bridge",
        "framework": "AutoGen",
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "paid_api_calls_allowed": False,
        "task_manifest": str(args.task_manifest),
        "task_ids": list(task_ids),
        "topologies": list(topologies),
        "repetitions": args.repetitions,
        "expected_runs": len(specs),
        "base_url": args.base_url,
        "llm_run_attempts": args.llm_run_attempts,
        "timestamp": now(),
    }
    write_bridge_outputs(rows, args.output_dir, experiment_config=experiment_config)
    gate = json.loads((args.output_dir / "matrix_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise SystemExit(f"matrix gate failed: {json.dumps(gate, ensure_ascii=False)}")
    print(
        f"COMPLETE runs={len(rows)} total_tokens={gate['total_tokens']} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
