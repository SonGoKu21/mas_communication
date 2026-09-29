from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from mas_faults.bottom_up_bridge import write_bridge_outputs
from mas_faults.llm_client import get_llm_client
from mas_faults.netem_task_bridge import (
    NETEM_TASK_CONDITIONS,
    DockerNetemController,
    NetemRunSpec,
    RelayDeliveryAdapter,
    build_netem_specs,
    enrich_netem_record,
    parse_tc_dropped_packets,
)
from mas_faults.webarena_task_selection import load_task_manifest
from run_bottom_up_bridge_experiment import (
    append_checkpoint,
    load_checkpoint_rows,
    select_frozen_tasks,
    validate_local_model,
)
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


def pending_specs(
    specs: Sequence[NetemRunSpec],
    rows: Sequence[dict[str, Any]],
) -> list[NetemRunSpec]:
    completed = {str(row["job_key"]) for row in rows}
    return [spec for spec in specs if spec.job_key not in completed]


async def execute_netem_spec(
    *,
    client: Any,
    task: dict[str, Any],
    spec: NetemRunSpec,
    base_url: str,
    controller: DockerNetemController,
) -> dict[str, Any]:
    condition = NETEM_TASK_CONDITIONS[spec.condition]
    controller.apply(condition)
    qdisc_before = controller.stats()
    adapter = RelayDeliveryAdapter(condition, controller.relay_url)
    try:
        base_row = await run_one(
            client,
            task,
            spec.topology,
            adapter.fault_type,
            spec.injection_step,
            spec.repeat_index,
            base_url,
            delivery_adapter=adapter,
        )
        qdisc_after = controller.stats()
    finally:
        controller.clear()
    if adapter.last_observation is None:
        raise RuntimeError("netem validation failed: selected relay edge was not exercised")
    dropped_delta = max(
        0,
        parse_tc_dropped_packets(qdisc_after)
        - parse_tc_dropped_packets(qdisc_before),
    )
    row = enrich_netem_record(
        base_row,
        spec,
        observation=adapter.last_observation,
        qdisc_before=qdisc_before,
        qdisc_after=qdisc_after,
        dropped_delta=dropped_delta,
    )
    validate_netem_row(row)
    return row


def validate_netem_row(row: dict[str, Any]) -> None:
    condition = str(row["condition"])
    observation = row["relay_observation"]
    if not row.get("netem_real_network_path"):
        raise RuntimeError("netem validation failed: row lacks real network evidence")
    if condition == "clean":
        if row.get("fault_applied") or not observation.get("payload_preserved"):
            raise RuntimeError("netem validation failed: clean relay did not preserve payload")
        if not row.get("final_task_success"):
            raise RuntimeError("netem validation failed: clean task failed")
    elif condition == "packet_loss_retransmission":
        if int(row.get("netem_dropped_packets_delta", 0)) < 1:
            raise RuntimeError("netem validation failed: packet-loss qdisc dropped no packets")
        if not observation.get("payload_preserved"):
            raise RuntimeError("netem validation failed: packet-loss relay did not recover payload")
    else:
        symptoms = set(row.get("observed_A_symptom") or [])
        if not row.get("fault_applied") or symptoms == {"none"}:
            raise RuntimeError(f"netem validation failed: {condition} did not expose at A")


def build_manifest(
    *,
    specs: Sequence[NetemRunSpec],
    tasks: Sequence[dict[str, Any]],
    model: str,
    provider: str,
    task_manifest: Path,
    controller: DockerNetemController,
) -> dict[str, Any]:
    return {
        "experiment": "docker_tc_netem_webarena_bridge",
        "framework": "AutoGen",
        "model": model,
        "provider": provider,
        "paid_api_calls_allowed": False,
        "validation_method": "docker_tc_netem",
        "relay_container": controller.container_name,
        "relay_url": controller.relay_url,
        "task_manifest": str(task_manifest),
        "tasks": list(tasks),
        "conditions": [asdict(condition) for condition in NETEM_TASK_CONDITIONS.values()],
        "unit_count": len(specs),
        "units": [asdict(spec) | {"job_key": spec.job_key} for spec in specs],
        "created_at": now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Docker/tc-netem-to-A-to-M WebArena bridge experiment."
    )
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--topologies", default=",".join(DEFAULT_TOPOLOGIES))
    parser.add_argument("--conditions", default=",".join(NETEM_TASK_CONDITIONS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--base-url", default="http://localhost:7770")
    parser.add_argument("--required-model", default="Qwen3.8-27B")
    parser.add_argument("--required-provider", default="modelscope_local")
    parser.add_argument("--llm-run-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--relay-container", default="mas-netem-task-bridge-20260825")
    parser.add_argument("--relay-port", type=int, default=18090)
    parser.add_argument("--keep-relay", action="store_true")
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
    tasks = select_frozen_tasks(load_task_manifest(args.task_manifest), task_ids)
    specs = build_netem_specs(
        tasks=tasks,
        topologies=topologies,
        repetitions=args.repetitions,
        conditions=conditions,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "netem_bridge_full.checkpoint.jsonl"
    rows = load_checkpoint_rows(checkpoint) if args.resume else []
    remaining = pending_specs(specs, rows)

    client = get_llm_client()
    validate_local_model(
        client,
        required_model=args.required_model,
        required_provider=args.required_provider,
    )
    controller = DockerNetemController(
        container_name=args.relay_container,
        host_port=args.relay_port,
    )
    manifest = build_manifest(
        specs=specs,
        tasks=tasks,
        model=client.model_info.model,
        provider=client.model_info.provider,
        task_manifest=args.task_manifest,
        controller=controller,
    )
    (args.output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    controller.start()
    try:
        for index, spec in enumerate(remaining, start=1):
            row = asyncio.run(
                run_with_retries(
                    lambda spec=spec: execute_netem_spec(
                        client=client,
                        task=tasks_by_id[spec.task_id],
                        spec=spec,
                        base_url=args.base_url,
                        controller=controller,
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
                f"repeat={spec.repeat_index} dropped={row['netem_dropped_packets_delta']} "
                f"A={row['observed_A_symptom']} M={row['observed_M_consequence']} "
                f"success={row['final_task_success']}",
                flush=True,
            )
    finally:
        controller.clear()
        if not args.keep_relay:
            controller.stop()

    experiment_config = {
        "experiment": "docker_tc_netem_webarena_bridge",
        "framework": "AutoGen",
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "paid_api_calls_allowed": False,
        "validation_method": "docker_tc_netem",
        "task_manifest": str(args.task_manifest),
        "task_ids": list(task_ids),
        "topologies": list(topologies),
        "repetitions": args.repetitions,
        "expected_runs": len(specs),
        "base_url": args.base_url,
        "llm_run_attempts": args.llm_run_attempts,
        "relay_container": args.relay_container,
        "relay_port": args.relay_port,
        "timestamp": now(),
    }
    write_bridge_outputs(
        rows,
        args.output_dir,
        experiment_config=experiment_config,
        condition_definitions=tuple(NETEM_TASK_CONDITIONS.values()),
    )
    matrix_gate = json.loads((args.output_dir / "matrix_gate.json").read_text())
    netem_gate = {
        "passed": bool(
            matrix_gate["passed"]
            and len(rows) == len(specs)
            and all(row.get("netem_real_network_path") for row in rows)
            and all(
                row.get("fault_applied")
                for row in rows
                if row.get("condition") != "clean"
            )
        ),
        "expected_runs": len(specs),
        "run_count": len(rows),
        "real_network_rows": sum(bool(row.get("netem_real_network_path")) for row in rows),
        "fault_rows": sum(row.get("condition") != "clean" for row in rows),
        "fault_applied_rows": sum(
            bool(row.get("fault_applied"))
            for row in rows
            if row.get("condition") != "clean"
        ),
        "packet_loss_rows_with_observed_drop": sum(
            int(row.get("netem_dropped_packets_delta", 0)) > 0
            for row in rows
            if row.get("condition") == "packet_loss_retransmission"
        ),
        "models": sorted({str(row.get("model")) for row in rows}),
        "providers": sorted({str(row.get("provider")) for row in rows}),
    }
    (args.output_dir / "netem_gate.json").write_text(
        json.dumps(netem_gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not netem_gate["passed"]:
        raise SystemExit(f"netem gate failed: {json.dumps(netem_gate, ensure_ascii=False)}")
    print(
        f"COMPLETE runs={len(rows)} total_tokens={matrix_gate['total_tokens']} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
