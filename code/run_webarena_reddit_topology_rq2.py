"""Real stateful WebArena Reddit topology comparison with trace-backed fault consequences."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.llm.consequence_axes import (
    SemanticStateEvidence,
    SystemRuntimeEvidence,
    evaluate_semantic_consequences,
    evaluate_system_consequences,
    m_consequences_from_axes,
)
from mas_faults.llm.topology_experiment_controls import (
    TopologyExperimentControls,
    resolve_topology_experiment_controls,
)
from mas_faults.llm.webarena_reddit_topologies import REDDIT_TOPOLOGIES, get_reddit_topology
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_reddit_stateful_real import RedditHTTPExecutor, build_edit_request, choose_edit_target
from mas_faults.webarena_shopping_real import parse_verdict
from run_webarena_reddit_stateful_rq1 import CONDITIONS, TASKS, inject_state_fault, task_dict


STATE_KEYS = ("task_id", "forum", "submission_id", "slug", "body")


@dataclass
class RedditMessage:
    kind: str
    payload: dict[str, Any]


def is_complete_reddit_state(task: dict[str, Any], state: dict[str, Any] | None) -> bool:
    return bool(
        state and all(key in state for key in STATE_KEYS)
        and state.get("task_id") == task["task_id"]
        and state.get("forum") == task["forum"]
        and state.get("submission_id") == task["submission_id"]
        and state.get("slug") == task["slug"]
        and isinstance(state.get("body"), str)
        and not state.get("truncated")
    )


def is_structurally_complete_reddit_state(state: dict[str, Any] | None) -> bool:
    return bool(
        state
        and all(key in state for key in STATE_KEYS)
        and isinstance(state.get("body"), str)
        and isinstance(state.get("evidence"), str)
        and not state.get("truncated")
    )


def reddit_state_is_internally_inconsistent(state: dict[str, Any] | None) -> bool:
    if not state or not isinstance(state.get("evidence"), str):
        return False
    try:
        nested = json.loads(state["evidence"])
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(nested, dict):
        return False
    comparable = ("forum", "submission_id", "slug", "body")
    return any(key in nested and nested.get(key) != state.get(key) for key in comparable)


def is_valid_current_reddit_state(
    task: dict[str, Any],
    state: dict[str, Any] | None,
    original_state: dict[str, Any],
) -> bool:
    """Validate task binding, the pre-edit snapshot, and nested evidence together."""
    if not is_complete_reddit_state(task, state) or reddit_state_is_internally_inconsistent(state):
        return False
    assert state is not None
    comparable = ("forum", "submission_id", "slug", "title", "body", "url")
    return all(
        key not in original_state or state.get(key) == original_state.get(key)
        for key in comparable
    )


def decision_state_for_topology(topology: str, primary_state: dict[str, Any] | None, direct_state: dict[str, Any] | None) -> dict[str, Any] | None:
    return direct_state if topology in {"flat", "hybrid"} else primary_state


def topology_has_direct_state_redundancy(topology: str) -> bool:
    """A decision node can reconcile inputs only when the topology has a direct Reader edge."""
    return topology in {"flat", "hybrid"}


def topology_uses_verifier_gate(topology: str) -> bool:
    """Match the runtime agents to the declared topology rather than adding a shared hidden gate."""
    return topology != "hierarchical"


def decision_inputs_for_controls(
    controls: TopologyExperimentControls,
    primary_state: dict[str, Any] | None,
    direct_state: dict[str, Any] | None,
    verification: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    state = direct_state if controls.direct_state_visibility == "delivered" else primary_state
    visible_verification = verification if controls.verifier_visibility == "delivered" else None
    return state, visible_verification


def role_labels_for_controls(
    topology: str,
    controls: TopologyExperimentControls,
) -> tuple[str, str]:
    if controls.comparison_mode != "native":
        return "Verifier", "Decision Agent"
    verifier_role = "Verification Team" if topology == "team" else "Verifier"
    decision_role = {
        "flat": "Coordinator",
        "hierarchical": "Supervisor",
        "team": "Team Lead",
        "hybrid": "Supervisor",
    }[topology]
    return verifier_role, decision_role


def realized_message_path_for_controls(
    controls: TopologyExperimentControls,
    verifier_role: str,
    decision_role: str,
) -> tuple[tuple[str, str], ...]:
    path: list[tuple[str, str]] = []
    if controls.verifier_visibility == "delivered":
        path.extend((("Reader", verifier_role), (verifier_role, decision_role)))
    else:
        path.append(("Reader", decision_role))
    if controls.direct_state_visibility == "delivered":
        path.append(("Reader", decision_role))
    return tuple(dict.fromkeys(path))


def validate_native_message_path(
    topology_name: str,
    controls: TopologyExperimentControls,
    realized_path: tuple[tuple[str, str], ...],
) -> None:
    if controls.comparison_mode != "native":
        return
    declared = set(get_reddit_topology(topology_name).message_path)
    if set(realized_path) != declared:
        raise ValueError(
            f"native message path mismatch for {topology_name}: "
            f"declared={sorted(declared)!r} realized={sorted(set(realized_path))!r}"
        )


def claim_scope_for_controls(controls: TopologyExperimentControls) -> str:
    return (
        "architecture_bundle_only"
        if controls.comparison_mode == "native"
        else "controlled_mechanism_ablation"
    )


def repeat_indices(repeats: int) -> range:
    if repeats < 1:
        raise ValueError("runs_per_condition must be at least one")
    return range(1, repeats + 1)


def direct_state_recovery_evidence(
    *,
    topology: str,
    task: dict[str, Any],
    decision_state: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    final_verdict: dict[str, Any],
    fault_applied: bool = True,
    direct_state_visible: bool | None = None,
) -> dict[str, Any]:
    """Record recovery only when two delivered inputs visibly disagree and the direct input wins."""
    if not (
        fault_applied
        and
        (topology_has_direct_state_redundancy(topology) if direct_state_visible is None else direct_state_visible)
        and is_complete_reddit_state(task, decision_state)
        and (verification or {}).get("decision") == "reject"
        and final_verdict.get("decision") == "accept"
    ):
        return {}
    return {
        "mechanism": "redundancy_recovery",
        "reconciled_direct_state": True,
        "rejected_secondary_verification": True,
        "verification_reason": str((verification or {}).get("reason", "")),
        "decision_reason": str(final_verdict.get("reason", "")),
    }


def append_checkpoint(row: dict[str, Any], event_path: Path, run_path: Path) -> None:
    """Persist a completed call immediately so an API outage cannot erase earlier traces."""
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row["event"], ensure_ascii=False) + "\n")
    with run_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: value for key, value in row.items() if key != "event"}, ensure_ascii=False) + "\n")


def stale_seed_task(tasks: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if len(tasks) < 2:
        raise ValueError("stateful stale replay requires at least two tasks")
    return tasks[1] if index == 0 else tasks[index - 1]


def final_decision_prompt(
    role: str,
    task: dict[str, Any],
    state: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    *,
    allow_direct_state_recovery: bool = False,
    decision_policy: str | None = None,
) -> str:
    legacy_direct_prompt = decision_policy is None and allow_direct_state_recovery
    policy = decision_policy or ("conflict_aware" if allow_direct_state_recovery else "strict")
    if policy not in {"strict", "conflict_aware"}:
        raise ValueError(f"unsupported decision policy: {policy}")
    if legacy_direct_prompt:
        verification_rule = (
            "This state arrived through a direct Reader channel, while the verifier verdict came from a secondary "
            "communication branch. A reject verdict alone is not a hard gate: when the direct state itself is complete "
            "and matches the task, and the reject identifies the secondary branch as stale, truncated, missing, or "
            "mismatched, accept the direct state. State this conflict reconciliation in your reason. Otherwise reject."
        )
    elif policy == "conflict_aware":
        verification_rule = (
            "The current state evidence and verifier verdict may arrive through different communication paths. "
            "A reject verdict alone is not a hard gate: when the current state evidence itself is complete and matches "
            "the task, and the reject identifies another observed state as stale, truncated, missing, or mismatched, "
            "accept the current state evidence. State this conflict reconciliation in your reason. Otherwise reject."
        )
    else:
        verification_rule = "Reject missing, stale, truncated, mismatched state, or a rejecting verifier."
    return (
        f"You are the {role} for a real WebArena Reddit edit task. Return JSON only: "
        '{"decision":"accept"|"reject","task_id":string,"reason":string}. '
        "The delivered state is a pre-edit snapshot and need not yet contain the required append; an empty body is a valid pre-edit state. "
        "the append is applied only after your accept decision. Accept only when the pre-edit state is complete "
        "and current. "
        f"{verification_rule}\n"
        f"Task: {json.dumps(task, ensure_ascii=False)}\nState: {json.dumps(state, ensure_ascii=False)}\n"
        f"Verifier verdict: {json.dumps(verification, ensure_ascii=False)}"
    )


class StateVerifierAgent(RoutedAgent):
    def __init__(self, client: Any, task: dict[str, Any], role: str) -> None:
        super().__init__(role)
        self.client, self.task, self.role = client, task, role

    @message_handler
    async def handle(self, message: RedditMessage, ctx: Any) -> RedditMessage:
        state = message.payload.get("state")
        raw = self.client.complete(
            f"You are the {self.role} in a real WebArena Reddit edit workflow. Return JSON only: "
            '{"decision":"accept"|"reject","task_id":string,"reason":string}. '
            "Accept only a complete current pre-edit state matching task_id, forum, submission_id and slug. "
            "Reject missing, stale, truncated, or mismatched state.\n"
            f"Task: {json.dumps(self.task, ensure_ascii=False)}\nState: {json.dumps(state, ensure_ascii=False)}"
        )
        return RedditMessage("verification", {"verdict": parse_verdict(raw), "state": state})


class FinalDecisionAgent(RoutedAgent):
    def __init__(
        self,
        client: Any,
        task: dict[str, Any],
        role: str,
        *,
        decision_policy: str = "strict",
    ) -> None:
        super().__init__(role)
        self.client, self.task, self.role = client, task, role
        self.decision_policy = decision_policy

    @message_handler
    async def handle(self, message: RedditMessage, ctx: Any) -> RedditMessage:
        state, verification = message.payload.get("state"), message.payload.get("verification")
        raw = self.client.complete(
            final_decision_prompt(
                self.role,
                self.task,
                state,
                verification,
                decision_policy=self.decision_policy,
            )
        )
        return RedditMessage("verdict", {"verdict": parse_verdict(raw)})


async def ask(runtime: SingleThreadedAgentRuntime, message: RedditMessage, recipient: AgentId) -> RedditMessage:
    response = await runtime.send_message(message, recipient)
    if not isinstance(response, RedditMessage):
        raise TypeError(f"unexpected AutoGen response: {response!r}")
    return response


def evaluate_reddit_run_consequences(
    *,
    task: dict[str, Any],
    original_state: dict[str, Any],
    primary_state: dict[str, Any] | None,
    decision_state: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    final_verdict: dict[str, Any],
    final_task_success: bool,
) -> dict[str, Any]:
    system_evidence = SystemRuntimeEvidence(
        workflow_completed=True,
        final_task_success=final_task_success,
        timeout_observed=False,
        execution_count=1 if final_verdict.get("decision") == "accept" else 0,
        expected_execution_count=1,
    )
    deterministic_verifier_decision = None
    if verification is not None:
        deterministic_verifier_decision = (
            "accept" if is_valid_current_reddit_state(task, primary_state, original_state) else "reject"
        )
    expected_final_decision = (
        "accept" if is_valid_current_reddit_state(task, decision_state, original_state) else "reject"
    )
    semantic_evidence = SemanticStateEvidence(
        expected_task_id=task["task_id"],
        accepted_evidence=decision_state,
        final_decision=str(final_verdict.get("decision", "reject")),
        required_fields=STATE_KEYS + ("evidence",),
        accepted_complete=is_structurally_complete_reddit_state(decision_state),
        expected_constraints={
            "forum": task["forum"],
            "submission_id": task["submission_id"],
            "slug": task["slug"],
        },
        state_inconsistent=reddit_state_is_internally_inconsistent(decision_state),
        verifier_decision=(verification or {}).get("decision"),
        verifier_expected_decision=deterministic_verifier_decision,
        authority_conflict=bool(verification and verification.get("decision") != expected_final_decision),
    )
    system = evaluate_system_consequences(system_evidence)
    semantic = evaluate_semantic_consequences(semantic_evidence)
    final_decision_correct = final_verdict.get("decision") == expected_final_decision
    observed_m = m_consequences_from_axes(
        system_consequences=system,
        semantic_consequences=semantic,
        final_decision_correct=final_decision_correct,
        final_task_success=final_task_success,
    )
    return {
        "axis_evaluation_mode": "strict_evidence_v1",
        "system_evaluator_evidence": asdict(system_evidence),
        "semantic_evaluator_evidence": asdict(semantic_evidence),
        "system_consequences": system or ["none"],
        "semantic_consequences": semantic or ["none"],
        "observed_M_consequence": observed_m or ["none"],
        "final_decision_correct": final_decision_correct,
        "consequence_derivation_path": "runtime/state evidence -> axes -> M -> propagation",
    }


async def run_one(
    client: Any,
    task: dict[str, Any],
    topology_name: str,
    condition: str,
    stale: dict[str, Any],
    *,
    repeat_index: int = 1,
    controls: TopologyExperimentControls | None = None,
) -> dict[str, Any]:
    topology = get_reddit_topology(topology_name)
    controls = controls or resolve_topology_experiment_controls(topology_name, comparison_mode="native")
    run_id, trace_id = f"reddit-rq2-{topology_name}-{task['task_id']}-{condition}-r{repeat_index}-{uuid.uuid4().hex[:8]}", f"trace-{uuid.uuid4()}"
    before, started = (client.call_count, client.prompt_tokens, client.completion_tokens), time.perf_counter()
    executor = RedditHTTPExecutor(); executor.login()
    runtime = SingleThreadedAgentRuntime()
    verifier_id, decision_id = AgentId("reddit_verifier", "default"), AgentId("reddit_decider", "default")
    verifier_role, decision_role = role_labels_for_controls(topology_name, controls)
    uses_verifier_gate = controls.verifier_visibility == "delivered"
    if uses_verifier_gate:
        await runtime.register_agent_instance(
            StateVerifierAgent(client, task, verifier_role),
            verifier_id,
        )
    await runtime.register_agent_instance(
        FinalDecisionAgent(
            client,
            task,
            decision_role,
            decision_policy=controls.decision_policy,
        ),
        decision_id,
    )
    runtime.start()
    restore, foreign_original = [], None
    try:
        original = executor.read_submission(task["forum"], task["submission_id"], task["slug"])
        original.update({"task_id": task["task_id"], "message_id": f"message-{uuid.uuid4()}", "evidence": json.dumps(original, ensure_ascii=False)})
        primary, symptom, applied = inject_state_fault(original, condition, stale)
        direct = copy.deepcopy(original) if controls.direct_state_visibility == "delivered" else None
        verification_response = (
            await ask(runtime, RedditMessage("state", {"state": primary}), verifier_id)
            if uses_verifier_gate
            else None
        )
        verification = verification_response.payload["verdict"] if verification_response else None
        relayed_state = verification_response.payload["state"] if verification_response else primary
        decision_state, visible_verification = decision_inputs_for_controls(controls, relayed_state, direct, verification)
        final = (await ask(runtime, RedditMessage("decision", {"state": decision_state, "verification": visible_verification}), decision_id)).payload["verdict"]
        wrote, verified = False, None
        if final["decision"] == "accept" and decision_state is not None:
            forum, submission_id, slug = choose_edit_target(task, decision_state)
            if (forum, submission_id, slug) != (task["forum"], task["submission_id"], task["slug"]):
                foreign_original = executor.read_submission(forum, submission_id, slug)
            executor.update_submission(forum, submission_id, slug, build_edit_request(task, decision_state)["body"])
            wrote = True
        verified = executor.read_submission(task["forum"], task["submission_id"], task["slug"])
        success = bool(wrote and task["required_append"] in verified["body"] and original["body"] in verified["body"])
    finally:
        if "original" in locals():
            executor.update_submission(task["forum"], task["submission_id"], task["slug"], original["body"]); restore.append("current_state_restored")
        if foreign_original is not None:
            executor.update_submission(foreign_original["forum"], foreign_original["submission_id"], foreign_original["slug"], foreign_original["body"]); restore.append("foreign_state_restored")
        await runtime.stop()
    consequence_result = evaluate_reddit_run_consequences(
        task=task,
        original_state=original,
        primary_state=primary,
        decision_state=decision_state,
        verification=visible_verification,
        final_verdict=final,
        final_task_success=success,
    )
    recovery_evidence = direct_state_recovery_evidence(
        topology=topology_name,
        task=task,
        decision_state=decision_state,
        verification=verification,
        final_verdict=final,
        fault_applied=applied,
        direct_state_visible=controls.direct_state_visibility == "delivered",
    )
    recovery_detected = bool(recovery_evidence)
    realized_path = realized_message_path_for_controls(controls, verifier_role, decision_role)
    validate_native_message_path(topology_name, controls, realized_path)
    runtime_dispatch_path = []
    if uses_verifier_gate:
        runtime_dispatch_path.extend((
            ("ExperimentOrchestrator", verifier_id.type),
            (verifier_id.type, "ExperimentOrchestrator"),
        ))
    runtime_dispatch_path.extend((
        ("ExperimentOrchestrator", decision_id.type),
        (decision_id.type, "ExperimentOrchestrator"),
    ))
    event = {
        "run_id": run_id, "trace_id": trace_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": "reddit_stateful_topology_edit", "benchmark": "WebArena-Verified-Reddit", "framework": "AutoGen",
        "topology": topology.name, "declared_message_path": topology.message_path,
        "realized_message_path": realized_path, "runtime_dispatch_path": runtime_dispatch_path,
        "experiment_controls": controls.to_dict(), "claim_scope": claim_scope_for_controls(controls),
        "task_id": task["task_id"], "source_agent": "Reader",
        "target_agent": verifier_role if uses_verifier_gate else decision_role,
        "original_message": original, "delivered_message": primary, "direct_message": direct,
        "verification": verification, "visible_verification": visible_verification, "final_verdict": final,
        "fault_type": condition, "fault_applied": applied, "observed_A_symptom": symptom,
        **consequence_result, "recovery_evidence": recovery_evidence,
    }
    propagation_class = (
        "clean" if not applied else
        "detected_and_recovered" if recovery_detected and success else
        "propagated_to_M_final_failure" if consequence_result["system_consequences"] != ["none"] and not success else
        "silent_propagation_to_M" if consequence_result["observed_M_consequence"] != ["none"] else
        "exposed_at_A_only" if success else
        "detected_but_unrecovered"
    )
    row = {**event, "condition": condition, "repeat_index": repeat_index, "seed_or_run_index": repeat_index, "model_seed": None, "fault_id": "none" if condition == "clean" else f"fault-{run_id}", "model": client.model_info.model, "provider": client.model_info.provider, "final_task_success": success, "task_score": float(success), "propagation_class": propagation_class, "recovery_detected": recovery_detected, "recovery_type": recovery_evidence.get("mechanism", "none"), "recovery_evidence": recovery_evidence, "environment_restore_evidence": restore, "latency_ms": round((time.perf_counter()-started)*1000,3), "api_call_count": client.call_count-before[0], "prompt_tokens": client.prompt_tokens-before[1], "completion_tokens": client.completion_tokens-before[2], "total_tokens": client.prompt_tokens-before[1]+client.completion_tokens-before[2], "event": event}
    return normalize_run_record(row, axis_evaluation_mode="strict_evidence_v1")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", default=",".join(REDDIT_TOPOLOGIES))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--runs-per-condition", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--comparison-mode",
        choices=("native", "matched_control", "mechanism_ablation"),
        default="native",
    )
    parser.add_argument("--decision-policy", choices=("strict", "conflict_aware"))
    parser.add_argument("--verifier-visibility", choices=("none", "delivered"))
    parser.add_argument("--direct-state-visibility", choices=("none", "delivered"))
    parser.add_argument("--prompt-profile", choices=("common_v1",), default="common_v1")
    return parser


def controls_from_args(args: argparse.Namespace, topology: str) -> TopologyExperimentControls:
    return resolve_topology_experiment_controls(
        topology,
        comparison_mode=args.comparison_mode,
        decision_policy=args.decision_policy,
        verifier_visibility=args.verifier_visibility,
        direct_state_visibility=args.direct_state_visibility,
        prompt_profile=args.prompt_profile,
    )


def main() -> None:
    args = build_argument_parser().parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    event_path, run_path = output / "llm_communication_traces.jsonl", output / "llm_communication_runs.jsonl"
    requested_topologies = tuple(name for name in args.topologies.split(",") if name)
    unknown_topologies = [name for name in requested_topologies if name not in REDDIT_TOPOLOGIES]
    if unknown_topologies:
        raise SystemExit(f"unsupported topologies: {','.join(unknown_topologies)}")
    topologies = requested_topologies
    controls_by_topology = {name: controls_from_args(args, name) for name in topologies}
    tasks = [task_dict(value) for value in TASKS[:args.tasks]]
    if len(tasks) < 2: raise SystemExit("stateful topology experiment needs at least two tasks")
    client, rows = get_llm_client(), []
    for index, task in enumerate(tasks):
        stale_task = stale_seed_task(tasks, index); seed = RedditHTTPExecutor(); seed.login(); stale = seed.read_submission(stale_task["forum"], stale_task["submission_id"], stale_task["slug"]); stale.update({"task_id":"previous-task","message_id":"seed","evidence":json.dumps(stale)})
        for topology in topologies:
            for condition in args.conditions.split(","):
                for repeat_index in repeat_indices(args.runs_per_condition):
                    row = asyncio.run(
                        run_one(
                            client,
                            task,
                            topology,
                            condition,
                            stale,
                            repeat_index=repeat_index,
                            controls=controls_by_topology[topology],
                        )
                    ); rows.append(row)
                    append_checkpoint(row, event_path, run_path)
                    print(f"DONE {len(rows)} topology={topology} task={task['task_id']} condition={condition} success={row['final_task_success']}", flush=True)
    summary={"runs":len(rows),"success":sum(r["final_task_success"] for r in rows),"system":sum(r["system_consequences"] != ["none"] for r in rows),"semantic":sum(r["semantic_consequences"] != ["none"] for r in rows)}
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"experiment_config.json").write_text(json.dumps({
        "framework": "AutoGen",
        "topologies": topologies,
        "conditions": args.conditions.split(","),
        "tasks": tasks,
        "runs_per_condition": args.runs_per_condition,
        "repeat_semantics": "independent API repetition; no model seed supplied",
        "model_seed": None,
        "comparison_mode": args.comparison_mode,
        "resolved_controls": {name: value.to_dict() for name, value in controls_by_topology.items()},
        "axis_evaluation_mode": "strict_evidence_v1",
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
