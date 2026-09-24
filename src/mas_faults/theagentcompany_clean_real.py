from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mas_faults.llm_client import get_llm_client
from mas_faults.theagentcompany_real import (
    build_agent_prompt,
    build_evaluation_command,
    build_initialization_command,
    task_from_workspace,
    task_image,
)


def execute(command: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行真实 TheAgentCompany clean baseline")
    parser.add_argument("--workspace-root", default="/data2/system5/mas/benchmarks/TheAgentCompany-gitcode")
    parser.add_argument("--task", default="sde-install-go")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    task = task_from_workspace(Path(args.workspace_root), args.task)
    image = task_image(task)
    if execute(["docker", "image", "inspect", image], timeout=60).returncode != 0:
        raise SystemExit(f"task image unavailable: {image}")
    run_id, trace_id = f"tac-{args.task}-{uuid.uuid4().hex[:8]}", f"trace-{uuid.uuid4()}"
    container = f"{run_id}-container"
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    client = get_llm_client()
    try:
        execute(["docker", "run", "-d", "--name", container, "--network", "host", image, "sleep", "infinity"], timeout=120)
        init = execute(build_initialization_command(task, container, args.base_url, args.model), timeout=900)
        instruction = execute(["docker", "exec", container, "cat", "/instruction/task.md"], timeout=60).stdout
        command = client.complete(build_agent_prompt(instruction)).strip()
        action = execute(["docker", "exec", container, "bash", "-lc", command], timeout=900)
        trajectory = {"trace_id": trace_id, "run_id": run_id, "task": args.task, "agent": args.model, "instruction": instruction, "command": command, "exit_code": action.returncode, "stdout": action.stdout[-4000:], "stderr": action.stderr[-4000:]}
        (output / "trajectory.jsonl").write_text(json.dumps(trajectory, ensure_ascii=False) + "\n", encoding="utf-8")
        execute(["docker", "cp", str(output / "trajectory.jsonl"), f"{container}:/tmp/trajectory.jsonl"], timeout=60)
        execute(["docker", "exec", container, "python_default", "-m", "pip", "install", "-q", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "setuptools<81"], timeout=300)
        evaluation = execute(build_evaluation_command(container, "/tmp/trajectory.jsonl", "/tmp/evaluation.json", base_url=args.base_url, model=args.model), timeout=900)
        copied = execute(["docker", "cp", f"{container}:/tmp/evaluation.json", str(output / "evaluation.json")], timeout=60)
        evaluation_result: dict[str, Any] = {}
        if copied.returncode == 0:
            evaluation_result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        final_success = bool(evaluation_result.get("final_score", {}).get("result"))
        events.append({"trace_id": trace_id, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "benchmark": "TheAgentCompany", "task": args.task, "dependencies": task.dependencies, "model": args.model, "provider": client.model_info.provider, "instruction": instruction, "agent_command": command, "initialization_exit_code": init.returncode, "action_exit_code": action.returncode, "evaluator_exit_code": evaluation.returncode, "evaluation": evaluation_result, "final_task_success": final_success, "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    finally:
        execute(["docker", "rm", "-f", container], timeout=120)
    (output / "theagentcompany_runs.jsonl").write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n", encoding="utf-8")
    (output / "theagentcompany_summary.json").write_text(json.dumps(events[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(events[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
