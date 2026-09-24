from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mas_faults.causal_trace_report import build_report
from mas_faults.llm_client import ChatClient, get_llm_client
from mas_faults.theagentcompany_real import (
    InterceptedTaskEvidence,
    TaskEvidence,
    build_evaluation_command,
    build_initialization_command,
    evaluate_task_evidence,
    intercept_task_evidence,
    task_from_workspace,
    task_image,
)
from mas_faults.theagentcompany_autogen_runtime import (
    TACContainerTools,
    create_autogen_model_client,
    run_coordinator_agent,
    run_operator_agent,
    wait_for_required_services,
)


def execute(command: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        timeout_message = f"command timed out after {timeout} seconds"
        return subprocess.CompletedProcess(command, 124, stdout, "\n".join(part for part in (stderr, timeout_message) if part))


def git_repository_ready() -> bool:
    """GitLab health can precede its HTTP repository service after task reset."""
    probe = execute(
        ["git", "ls-remote", "http://the-agent-company.com:8929/root/doc.git"],
        timeout=20,
    )
    return probe.returncode == 0 and bool(probe.stdout.strip())


def gitlab_wiki_ready() -> bool:
    """The repository service can recover before the GitLab Wiki application."""
    try:
        with urllib.request.urlopen(
            "http://the-agent-company.com:8929/root/doc/-/wikis",
            timeout=15,
        ) as response:
            body = response.read(65536).decode("utf-8", errors="replace")
            return response.status == 200 and "Something went wrong" not in body
    except (urllib.error.URLError, TimeoutError):
        return False


def tac_service_ready(service: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:2999/api/healthcheck/{service}", timeout=8) as response:
            healthy = response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False
    return healthy and (service != "gitlab" or (git_repository_ready() and gitlab_wiki_ready()))


def build_coordinator_prompt(task_instruction: str) -> str:
    return (
        "You are the Coordinator in a TheAgentCompany MAS. You have no access to the task container. "
        "Produce a concise execution handoff for the Operator using imperative steps. Never state that the task is already complete, "
        "installed, verified, or otherwise executed. Do not write shell commands.\n\n"
        f"Task:\n{task_instruction.strip()}\n"
    )


def build_operator_prompt(task_instruction: str, coordinator_handoff: str) -> str:
    return (
        "You are the Operator in a sandboxed TheAgentCompany task container. Complete the task. "
        "Return exactly one single shell command, with no markdown, explanation, or shell prompt. "
        "The command will be executed with /bin/sh -lc. Do not treat the handoff as evidence that the task is complete; "
        "you must execute the task yourself and may only use the handoff as a suggestion. You already run as root. Do not use sudo.\n\n"
        f"Task:\n{task_instruction.strip()}\n\nCoordinator handoff:\n{coordinator_handoff.strip()}\n"
    )


def normalize_operator_command(raw: str) -> str:
    fenced = re.findall(r"```(?:bash|shell|sh)?\s*\n(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    return (fenced[-1] if fenced else raw).strip()


def build_task_verifier_prompt(task_slug: str, evidence: TaskEvidence | None) -> str:
    serialized = "missing" if evidence is None else json.dumps(
        {
            "task_slug": evidence.task_slug,
            "passed": evidence.passed,
            "evaluation": evidence.evaluation,
            "log": evidence.log[:1200],
        },
        ensure_ascii=False,
    )
    return (
        "You are the Verifier in a TheAgentCompany MAS. Accept only when the evaluation evidence belongs to the "
        "current task and reports a passed official score. Return strict JSON: "
        '{"decision": "accept"|"reject", "reason": string}.\n'
        f"Current task: {task_slug}\nEvaluation evidence: {serialized}\n"
    )


def parse_verifier_output(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return {"decision": "reject", "reason": "verifier_non_json"}
    decision = str(value.get("decision", "reject")).lower()
    return {"decision": "accept" if decision == "accept" else "reject", "reason": str(value.get("reason", ""))}


def load_stale_task_evidence(path: Path) -> TaskEvidence:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        message = record.get("original_message")
        if not isinstance(message, dict) or not message.get("task_slug"):
            continue
        evaluation = message.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        return TaskEvidence(
            task_slug=str(message["task_slug"]),
            passed=bool(message.get("passed")),
            evaluation=evaluation,
            log=str(message.get("log", "")),
        )
    raise ValueError(f"no reusable task evidence in {path}")


def write_event(
    path: Path,
    *,
    trace_id: str,
    run_id: str,
    task_slug: str,
    condition: str,
    layer: str,
    event_type: str,
    component: str,
    status: str,
    effect: str,
    label: str,
    evidence: dict[str, Any],
) -> None:
    root = f"span-{uuid.uuid5(uuid.NAMESPACE_URL, trace_id)}"
    fault_code = {
        "a1_moderate_delay": "A1", "a1_deadline_delay": "A1", "a5_omission": "A5",
        "a8_truncation": "A8", "a12_stale_replay": "A12",
    }.get(condition)
    event = {
        "trace_id": trace_id,
        "span_id": root if event_type == "workflow_started" else f"span-{uuid.uuid4()}",
        "parent_span_id": None if event_type == "workflow_started" else root,
        "fault_id": None if condition == "clean" else f"fault-{run_id}",
        "carrier_id": f"carrier-{run_id}",
        "carrier_instance_id": f"carrier-{run_id}#1",
        "duplicate_index": 1,
        "logical_message_id": run_id,
        "pair_id": f"theagentcompany-{task_slug}",
        "trace_variant": "clean" if condition == "clean" else "fault",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_unix": time.time(),
        "event_layer": layer,
        "component": component,
        "event_type": event_type,
        "event_status": status,
        "source": None,
        "target": None,
        "injection_operator_code": fault_code,
        "injection_point_kind": "operator_to_verifier_interceptor" if fault_code else None,
        "injected_fault_code": fault_code,
        "injected_fault_layer": "A" if fault_code else None,
        "expected_manifest_code": "A2" if condition == "a1_deadline_delay" else fault_code,
        "expected_manifest_layer": "A" if fault_code else None,
        "observed_effect": effect,
        "propagation_label": label,
        "evidence": {"run_id": run_id, "task_id": task_slug, **evidence},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def write_interception_events(
    path: Path,
    trace_id: str,
    run_id: str,
    task_slug: str,
    condition: str,
    intercepted: InterceptedTaskEvidence,
) -> None:
    write_event(
        path, trace_id=trace_id, run_id=run_id, task_slug=task_slug, condition=condition, layer="A",
        event_type="fault_applied" if intercepted.fault_applied else "message_delivered",
        component="communication_interceptor", status="applied" if intercepted.fault_applied else "delivered",
        effect=intercepted.observed_a_symptom, label="injected" if intercepted.fault_applied else "pre_injection", evidence={},
    )
    if intercepted.fault_applied:
        write_event(
            path, trace_id=trace_id, run_id=run_id, task_slug=task_slug, condition=condition, layer="A",
            event_type="runtime_effect_observed", component="communication_interceptor", status="effect_observed",
            effect=intercepted.observed_a_symptom, label="propagated", evidence={},
        )
        write_event(
            path, trace_id=trace_id, run_id=run_id, task_slug=task_slug, condition=condition, layer="A",
            event_type="message_dropped" if intercepted.delivered is None else "message_delivered", component="Verifier",
            status="dropped" if intercepted.delivered is None else "delivered", effect=intercepted.observed_a_symptom,
            label="propagated", evidence={},
        )


def task_success_from_evaluation(evaluation: dict[str, Any]) -> bool:
    score = evaluation.get("final_score", {})
    if not isinstance(score, dict):
        return False
    result, total = score.get("result"), score.get("total")
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)) and isinstance(total, (int, float)):
        return total > 0 and result == total
    return False


def repeat_indices(repeats: int, *, start: int = 1) -> range:
    return range(start, start + repeats)


def run_instance(
    client: ChatClient,
    *,
    workspace_root: Path,
    task_slug: str,
    condition: str,
    output_dir: Path,
    base_url: str,
    model: str,
    stale: TaskEvidence | None,
    repeat_index: int,
) -> tuple[dict[str, Any], TaskEvidence]:
    task = task_from_workspace(workspace_root, task_slug)
    run_id, trace_id = f"tac-{task_slug}-{condition}-r{repeat_index}", f"trace-{uuid.uuid4()}"
    container = f"{run_id}-{uuid.uuid4().hex[:8]}"
    events = output_dir / "theagentcompany_causal_events.jsonl"
    started = time.perf_counter()
    calls_before, prompt_before, completion_before = client.call_count, client.prompt_tokens, client.completion_tokens
    write_event(events, trace_id=trace_id, run_id=run_id, task_slug=task_slug, condition=condition, layer="A", event_type="workflow_started", component="Coordinator", status="started", effect="TheAgentCompany MAS started", label="pre_injection", evidence={})
    image = task_image(task)
    if execute(["docker", "image", "inspect", image], timeout=60).returncode != 0:
        raise RuntimeError(f"task image unavailable: {image}")
    initialization, action, evaluation_process = None, None, None
    instruction = coordinator_handoff = operator_command = operator_raw_output = ""
    agent_events: list[dict[str, Any]] = []
    tool_observations: list[dict[str, Any]] = []
    evaluation: dict[str, Any] = {}
    try:
        execute(["docker", "run", "-d", "--name", container, "--network", "host", image, "sleep", "infinity"], timeout=120)
        initialization = execute(build_initialization_command(task, container, base_url, model), timeout=900)
        instruction = execute(["docker", "exec", container, "cat", "/instruction/task.md"], timeout=60).stdout.strip()
        # TAC init scripts can schedule an asynchronous GitLab reset after they return.
        # Require a stable readiness window before giving the environment to the Agent.
        readiness = wait_for_required_services(
            task.dependencies,
            probe=tac_service_ready,
            attempts=48 if "gitlab" in task.dependencies else 24,
            consecutive_successes=3,
        )
        if not readiness.ready:
            evaluation = {
                "environment_error": readiness.error_kind,
                "unavailable_services": list(readiness.unavailable_services),
            }
            action = subprocess.CompletedProcess(["autogen"], 124, "", "required TAC service did not become healthy")
        else:
            async def run_agents() -> tuple[Any, Any]:
                model_client = create_autogen_model_client()
                try:
                    coordinator = await run_coordinator_agent(model_client=model_client, task_instruction=instruction)
                    operator = await run_operator_agent(
                        model_client=model_client,
                        task_instruction=instruction,
                        coordinator_handoff=coordinator.handoff,
                        tools=TACContainerTools(container),
                    )
                    return coordinator, operator
                finally:
                    await model_client.close()

            coordinator, operator = asyncio.run(run_agents())
            coordinator_handoff = coordinator.handoff
            operator_raw_output = operator.final_message
            operator_command = "\n\n".join(item.command for item in operator.tool_observations)
            agent_events = [
                {"agent": "Coordinator", **event} for event in coordinator.events
            ] + [
                {"agent": "Operator", **event} for event in operator.events
            ]
            tool_observations = [item.__dict__ for item in operator.tool_observations]
            last_exit_code = operator.tool_observations[-1].exit_code if operator.tool_observations else 1
            action = subprocess.CompletedProcess(
                ["autogen", "operator"],
                last_exit_code,
                operator.final_message,
                "",
            )
            for event in agent_events:
                write_event(
                    events,
                    trace_id=trace_id,
                    run_id=run_id,
                    task_slug=task_slug,
                    condition=condition,
                    layer="A",
                    event_type="agent_message",
                    component=event["agent"],
                    status="observed",
                    effect=event["event_type"],
                    label="pre_injection",
                    evidence=event,
                )
        trajectory = {
            "trace_id": trace_id, "run_id": run_id, "task": task_slug, "agent": client.model_info.model,
            "instruction": instruction, "coordinator_handoff": coordinator_handoff,
            "operator_raw_output": operator_raw_output, "command": operator_command,
            "exit_code": action.returncode, "stdout": action.stdout[-4000:], "stderr": action.stderr[-4000:],
            "agent_events": agent_events, "tool_observations": tool_observations,
        }
        trajectory_path = output_dir / f"{run_id}-trajectory.jsonl"
        trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False) + "\n", encoding="utf-8")
        execute(["docker", "cp", str(trajectory_path), f"{container}:/tmp/trajectory.jsonl"], timeout=60)
        execute(["docker", "exec", container, "python_default", "-m", "pip", "install", "-q", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "setuptools<81"], timeout=300)
        evaluation_process = execute(build_evaluation_command(container, "/tmp/trajectory.jsonl", "/tmp/evaluation.json", base_url=base_url, model=model), timeout=900)
        copied = execute(["docker", "cp", f"{container}:/tmp/evaluation.json", str(output_dir / f"{run_id}-evaluation.json")], timeout=60)
        if copied.returncode == 0:
            evaluation = json.loads((output_dir / f"{run_id}-evaluation.json").read_text(encoding="utf-8"))
    finally:
        execute(["docker", "rm", "-f", container], timeout=120)
    original = TaskEvidence(task_slug, task_success_from_evaluation(evaluation), evaluation, json.dumps(evaluation, ensure_ascii=False))
    intercepted = intercept_task_evidence(original, condition, stale)
    write_interception_events(events, trace_id, run_id, task_slug, condition, intercepted)
    verifier = parse_verifier_output(client.complete(build_task_verifier_prompt(task_slug, intercepted.delivered)))
    outcome = evaluate_task_evidence(condition, intercepted.delivered, verifier, original)
    for consequence in outcome["observed_M_consequence"]:
        if consequence != "none":
            write_event(events, trace_id=trace_id, run_id=run_id, task_slug=task_slug, condition=condition, layer="M", event_type="m_consequence_observed", component="task_evaluator", status="observed", effect=consequence, label="propagated", evidence={"mas_consequence": consequence})
    write_event(events, trace_id=trace_id, run_id=run_id, task_slug=task_slug, condition=condition, layer="M", event_type="final_consequence", component="task_evaluator", status="preserved" if outcome["final_task_success"] else "failed", effect=outcome["benchmark_consequence"], label="pre_injection" if condition == "clean" else "propagated", evidence={"task_success": outcome["final_task_success"]})
    row = {
        "run_id": run_id, "trace_id": trace_id, "benchmark": "TheAgentCompany", "task_slug": task_slug,
        "condition": condition, "model": client.model_info.model, "provider": client.model_info.provider,
        "fault_id": "none" if condition == "clean" else f"fault-{run_id}", "fault_applied": intercepted.fault_applied,
        "source_agent": "Operator", "target_agent": "Verifier", "original_message": original.__dict__,
        "delivered_message": None if intercepted.delivered is None else intercepted.delivered.__dict__,
        "instruction": instruction, "coordinator_handoff": coordinator_handoff,
        "operator_raw_output": operator_raw_output, "operator_command": operator_command,
        "tool_observations": tool_observations, "agent_events": agent_events,
        "initialization_exit_code": None if initialization is None else initialization.returncode,
        "action_exit_code": None if action is None else action.returncode,
        "evaluator_exit_code": None if evaluation_process is None else evaluation_process.returncode,
        "official_task_success": original.passed, "official_evaluation": evaluation, "verifier_output": verifier,
        **outcome,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "api_call_count": client.call_count - calls_before,
        "prompt_tokens": client.prompt_tokens - prompt_before,
        "completion_tokens": client.completion_tokens - completion_before,
    }
    row["total_tokens"] = row["prompt_tokens"] + row["completion_tokens"]
    return row, original


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_summary_markdown(summary: dict[str, Any]) -> str:
    counts = summary.get("propagation_class_counts", {})
    rows = "\n".join(f"| `{label}` | {count} |" for label, count in sorted(counts.items())) or "| 无 | 0 |"
    return (
        "# TheAgentCompany 通信故障实验汇总\n\n"
        f"- 运行数：{summary.get('runs', 0)}\n"
        f"- 官方任务成功：{summary.get('official_task_success', 0)}\n"
        f"- MAS 最终成功：{summary.get('final_success', 0)}\n\n"
        "| 传播类别 | 次数 |\n|---|---:|\n"
        f"{rows}\n"
    )


def build_experiment_config(
    tasks: list[str],
    conditions: list[str],
    repeats: int,
    model: str,
    provider: str,
    *,
    stale_evidence_jsonl: str | None = None,
) -> dict[str, Any]:
    return {
        "benchmark": "TheAgentCompany",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
        "tasks": tasks,
        "conditions": conditions,
        "runs_per_condition": repeats,
        "stale_evidence_jsonl": stale_evidence_jsonl,
        "workflow": "Coordinator -> Operator -> official evaluator -> communication interceptor -> Verifier",
    }


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({str(row["condition"]) for row in rows})
    by_condition: dict[str, dict[str, int]] = {}
    for condition in conditions:
        group = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "runs": len(group),
            "a_exposure_count": sum(row.get("observed_A_symptom") != ["none"] for row in group),
            "m_consequence_count": sum(row.get("observed_M_consequence") != ["none"] for row in group),
            "recovery_count": sum(bool(row.get("recovery_detected")) for row in group),
            "final_success_count": sum(bool(row.get("final_task_success")) for row in group),
            "final_failure_count": sum(not bool(row.get("final_task_success")) for row in group),
        }
    return {
        "runs": len(rows),
        "official_task_success": sum(bool(row["official_task_success"]) for row in rows),
        "final_success": sum(bool(row["final_task_success"]) for row in rows),
        "propagation_class_counts": {
            label: sum(row["propagation_class"] == label for row in rows)
            for label in sorted({row["propagation_class"] for row in rows})
        },
        "by_condition": by_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real TheAgentCompany communication-fault experiments.")
    parser.add_argument("--workspace-root", default="/data2/system5/mas/benchmarks/TheAgentCompany-gitcode")
    parser.add_argument("--tasks", nargs="+", default=["sde-install-openjdk"])
    parser.add_argument("--conditions", nargs="+", default=["clean", "a1_moderate_delay", "a1_deadline_delay", "a5_omission", "a8_truncation"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-start", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--stale-evidence-jsonl")
    args = parser.parse_args()
    if args.repeats < 1 or args.repeat_start < 1:
        raise SystemExit("--repeats and --repeat-start must be at least 1")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True)
    client = get_llm_client()
    config = build_experiment_config(
        args.tasks, args.conditions, args.repeats, client.model_info.model, client.model_info.provider,
        stale_evidence_jsonl=args.stale_evidence_jsonl,
    )
    (output_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stale_reference = load_stale_task_evidence(Path(args.stale_evidence_jsonl)) if args.stale_evidence_jsonl else None
    rows: list[dict[str, Any]] = []
    stale: TaskEvidence | None = None
    runs_path = output_dir / "theagentcompany_runs.jsonl"
    for task_slug in args.tasks:
        for repeat_index in repeat_indices(args.repeats, start=args.repeat_start):
            for condition in args.conditions:
                stale_for_run = stale_reference if condition == "a12_stale_replay" and stale_reference is not None else stale
                row, evidence = run_instance(client, workspace_root=Path(args.workspace_root), task_slug=task_slug, condition=condition, output_dir=output_dir, base_url=args.base_url, model=args.model, stale=stale_for_run, repeat_index=repeat_index)
                row["repeat_index"] = repeat_index
                rows.append(row)
                append_jsonl(runs_path, row)
                stale = evidence
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "theagentcompany_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    summary = summarize_runs(rows)
    (output_dir / "theagentcompany_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "theagentcompany_summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    build_report(output_dir / "theagentcompany_causal_events.jsonl", output_dir / "causal_trace_report")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
