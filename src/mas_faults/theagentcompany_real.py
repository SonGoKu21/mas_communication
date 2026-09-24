from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path


RELEASE = "1.0.0"
EVALUATOR_KEY = "theagentcompany is all you need"


@dataclass(frozen=True)
class TheAgentCompanyTask:
    slug: str
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class TaskEvidence:
    task_slug: str
    passed: bool
    evaluation: dict[str, Any]
    log: str


@dataclass(frozen=True)
class InterceptedTaskEvidence:
    original: TaskEvidence
    delivered: TaskEvidence | None
    observed_a_symptom: str
    fault_applied: bool


def intercept_task_evidence(
    evidence: TaskEvidence,
    condition: str,
    stale: TaskEvidence | None = None,
) -> InterceptedTaskEvidence:
    if condition == "clean":
        return InterceptedTaskEvidence(evidence, evidence, "none", False)
    if condition == "a1_moderate_delay":
        return InterceptedTaskEvidence(evidence, evidence, "A1_message_latency", True)
    if condition in {"a1_deadline_delay", "a5_omission"}:
        symptom = "A2_message_timeout" if condition == "a1_deadline_delay" else "A5_message_omission"
        return InterceptedTaskEvidence(evidence, None, symptom, True)
    if condition == "a8_truncation":
        partial = TaskEvidence(evidence.task_slug, evidence.passed, {}, evidence.log[:80])
        return InterceptedTaskEvidence(evidence, partial, "A8_message_truncation", True)
    if condition == "a12_stale_replay":
        if stale is None:
            raise ValueError("a12_stale_replay requires prior task evidence")
        if stale.task_slug == evidence.task_slug:
            raise ValueError("a12_stale_replay requires evidence from a different task")
        return InterceptedTaskEvidence(evidence, stale, "A12_timing_or_session_mismatch", True)
    raise ValueError(f"unsupported condition={condition!r}")


def evaluate_task_evidence(
    condition: str,
    delivered_evidence: TaskEvidence | None,
    verifier_output: dict[str, Any],
    original_evidence: TaskEvidence,
) -> dict[str, Any]:
    a_symptom = {
        "clean": "none", "a1_moderate_delay": "A1_message_latency", "a1_deadline_delay": "A2_message_timeout",
        "a5_omission": "A5_message_omission", "a8_truncation": "A8_message_truncation",
        "a12_stale_replay": "A12_timing_or_session_mismatch",
    }[condition]
    accepted = str(verifier_output.get("decision", "")).lower() == "accept"
    consequences: list[str] = []
    detected = False
    semantic = "none"
    if delivered_evidence is None:
        consequences.append("M3_incomplete_information_aggregation")
        semantic = "theagentcompany.missing_evaluation_evidence"
    elif condition == "a8_truncation":
        consequences.append("M3_incomplete_information_aggregation")
        if accepted:
            consequences.append("M14_partial_tool_or_message_result_acceptance")
            semantic = "theagentcompany.verifier_accepted_partial_evidence"
        else:
            detected = True
            semantic = "theagentcompany.verifier_rejected_partial_evidence"
    elif condition == "a12_stale_replay":
        if delivered_evidence.task_slug != original_evidence.task_slug and accepted:
            consequences.extend(["M5_stale_context_acceptance", "M6_state_inconsistency"])
            semantic = "theagentcompany.verifier_accepted_stale_evidence"
        elif delivered_evidence.task_slug != original_evidence.task_slug:
            detected = True
            semantic = "theagentcompany.verifier_rejected_stale_evidence"
    partial_or_stale = condition in {"a8_truncation", "a12_stale_replay"}
    successful = bool(original_evidence.passed and delivered_evidence and delivered_evidence.passed and accepted and not consequences and not partial_or_stale)
    if not successful:
        if accepted and condition in {"a8_truncation", "a12_stale_replay"}:
            consequences.append("M4_incorrect_collective_decision")
        consequences.append("M2_task_timeout_or_failure")
    consequences = list(dict.fromkeys(consequences))
    if condition == "clean":
        propagation_class = "clean" if successful else "clean_task_failure"
    elif detected:
        propagation_class = "detected_but_unrecovered"
    elif consequences:
        propagation_class = "propagated_to_M_final_failure"
    elif successful:
        propagation_class = "exposed_at_A_only"
    else:
        propagation_class = "fault_not_observed"
    return {
        "observed_A_symptom": [a_symptom],
        "observed_M_consequence": consequences or ["none"],
        "benchmark_consequence": semantic,
        "final_task_success": successful,
        "recovery_detected": False,
        "recovery_type": "none",
        "recovery_evidence": "",
        "propagation_class": propagation_class,
    }


def task_image(task: TheAgentCompanyTask) -> str:
    return f"ghcr.io/theagentcompany/{task.slug}-image:{RELEASE}"


def build_agent_prompt(task_instruction: str) -> str:
    return (
        "You are an autonomous operator inside a sandboxed TheAgentCompany task container. "
        "Complete the following task. Return exactly one single shell command, with no markdown, explanation, "
        "or shell prompt. The command will be executed with /bin/sh -lc.\n\n"
        f"Task:\n{task_instruction.strip()}\n"
    )


def build_initialization_command(
    task: TheAgentCompanyTask,
    container_name: str,
    base_url: str,
    model: str,
) -> list[str]:
    return [
        "docker", "exec", container_name, "env",
        "SERVER_HOSTNAME=localhost",
        "LITELLM_API_KEY=local",
        f"LITELLM_BASE_URL={base_url}",
        f"LITELLM_MODEL={model}",
        "bash", "/utils/init.sh",
    ]


def build_evaluation_command(
    container_name: str,
    trajectory_path: str,
    output_path: str,
    *,
    base_url: str = "http://127.0.0.1:8004",
    model: str = "Qwen/Qwen3-8B",
) -> list[str]:
    return [
        "docker", "exec", container_name, "env",
        "LITELLM_API_KEY=local",
        f"LITELLM_BASE_URL={base_url}",
        f"LITELLM_MODEL={model}",
        f"DECRYPTION_KEY={EVALUATOR_KEY}",
        "python_default", "/utils/eval.py",
        "--trajectory_path", trajectory_path,
        "--result_path", output_path,
    ]


def task_from_workspace(workspace_root: Path, slug: str) -> TheAgentCompanyTask:
    dependency_file = workspace_root / "workspaces" / "tasks" / slug / "dependencies.yml"
    if not dependency_file.exists():
        raise FileNotFoundError(f"task dependency definition not found: {dependency_file}")
    dependencies = tuple(
        line.removeprefix("-").strip()
        for line in dependency_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("-")
    )
    return TheAgentCompanyTask(slug=slug, dependencies=dependencies)
