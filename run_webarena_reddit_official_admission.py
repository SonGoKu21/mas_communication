#!/usr/bin/env python3
"""Run clean admission for official WebArena Reddit tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.deepseek_schedule import DeepSeekPeakWindowError, ensure_deepseek_offpeak
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_admin_controlled import (
    EvaluatorWorkerClient,
    write_sanitized_browser_config,
)
from mas_faults.webarena_admin_real import BrowserWorkerClient
from mas_faults.webarena_reddit_official import (
    load_reddit_tasks,
    routed_start_url,
    run_clean_reddit_task,
    write_clean_outputs,
)
from run_webarena_admin_controlled_fault_matrix import browser_environment


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _error_record(client: Any, task: dict[str, Any], exc: Exception, run_index: int) -> dict[str, Any]:
    return normalize_run_record(
        {
            "run_id": f"reddit-official-{task['task_id']}-clean-error-r{run_index}-{uuid.uuid4().hex[:8]}",
            "trace_id": f"trace-{uuid.uuid4()}",
            "scenario": "webarena_reddit_official_clean",
            "dataset": "WebArena",
            "benchmark": "WebArena Reddit",
            "framework": "AutoGen",
            "topology": "sequential",
            "task_id": str(task["task_id"]),
            "task_stratum": task["task_stratum"],
            "intent": task["intent"],
            "condition": "clean",
            "model": client.model_info.model,
            "provider": client.model_info.provider,
            "seed_or_run_index": run_index,
            "fault_id": "none",
            "fault_type": "clean",
            "fault_applied": False,
            "observed_A_symptom": ["none"],
            "observed_M_consequence": ["none"],
            "system_consequences": ["none"],
            "semantic_consequences": ["none"],
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": [],
            "expected_answer": task["eval"]["reference_answers"],
            "final_answer": {},
            "task_score": 0.0,
            "final_task_success": False,
            "propagation_class": "clean_task_failure",
            "termination_reason": "runtime_error",
            "latency_ms": 0.0,
            "api_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "events": [],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", action="append", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=Path("/data2/system5/mas/third_party/webarena/config_files"))
    parser.add_argument("--webarena-root", type=Path, default=Path("/data2/system5/mas/third_party/webarena"))
    parser.add_argument("--reddit-url", default="http://10.102.35.120:7771")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-observation-chars", type=int, default=12000)
    parser.add_argument("--run-index", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    load_dotenv(".env.local")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"result directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "clean_runs.checkpoint.jsonl"
    rows = _load_jsonl(checkpoint) if args.resume else []
    completed = {str(row["task_id"]) for row in rows}
    tasks = load_reddit_tasks(args.config_dir, args.task_id)
    client = get_llm_client(mock_llm=False)
    env = browser_environment("http://10.102.35.120:7780")
    env["REDDIT"] = args.reddit_url
    evaluator = EvaluatorWorkerClient(webarena_root=str(args.webarena_root), env=env)
    paused_for_peak = False
    try:
        with tempfile.TemporaryDirectory(prefix="webarena-reddit-official-") as temp:
            for task in tasks:
                if str(task["task_id"]) in completed:
                    continue
                try:
                    ensure_deepseek_offpeak(client.model_info.model)
                except DeepSeekPeakWindowError:
                    paused_for_peak = True
                    break
                original = args.config_dir / f"{task['task_id']}.json"
                sanitized = write_sanitized_browser_config(original, Path(temp) / str(task["task_id"]))
                browser_config = json.loads(sanitized.path.read_text(encoding="utf-8"))
                browser_config["start_url"] = routed_start_url(task, args.reddit_url)
                sanitized.path.write_text(
                    json.dumps(browser_config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                browser = BrowserWorkerClient(webarena_root=str(args.webarena_root), env=env, browser_only=True)
                try:
                    row = asyncio.run(
                        run_clean_reddit_task(
                            client,
                            browser,
                            evaluator,
                            task,
                            browser_config_file=str(sanitized.path),
                            evaluator_config_file=str(original),
                            run_index=args.run_index,
                            max_steps=args.max_steps,
                            max_observation_chars=args.max_observation_chars,
                        )
                    )
                except Exception as exc:
                    row = _error_record(client, task, exc, args.run_index)
                finally:
                    browser.close()
                rows.append(row)
                completed.add(str(task["task_id"]))
                _append_jsonl(checkpoint, row)
                print(
                    f"DONE task={task['task_id']} success={row['final_task_success']} "
                    f"error={row.get('error')}",
                    flush=True,
                )
    finally:
        evaluator.close()

    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "WebArena Reddit",
        "framework": "AutoGen",
        "topology": "sequential",
        "task_ids": args.task_id,
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "offpeak_only": True,
        "paused_for_peak": paused_for_peak,
        "completed_tasks": sorted(completed),
    }
    if rows:
        write_clean_outputs(rows, args.output_dir, config)
    else:
        (args.output_dir / "experiment_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
