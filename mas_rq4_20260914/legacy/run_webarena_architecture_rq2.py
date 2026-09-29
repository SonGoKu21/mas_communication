from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler

from mas_faults.llm.communication_interceptor import CommunicationInterceptor, MessageEnvelope
from mas_faults.llm.architecture_manifestations import classify_manifestation
from mas_faults.llm.application_fault_matrix import APPLICATION_FAULTS, fault_names_for_steps, steps_for_fault
from mas_faults.llm.consequence_axes import classify_consequence_axes
from mas_faults.llm.fault_scenarios import FaultScenario
from mas_faults.llm.propagation_evaluator import evaluate_propagation
from mas_faults.deepseek_schedule import ensure_deepseek_offpeak
from mas_faults.llm.webarena_topologies import SUPPORTED_TOPOLOGIES, TopologyDefinition, get_topology
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_shopping_real import ShoppingHTTPExecutor, parse_verdict
from mas_faults.webarena_task_selection import load_task_manifest
from summarize_rq2_cases import build_case_summary, write_representative_cases


FAULTS = ("none",) + fault_names_for_steps((2, 3, 4))
INJECTION_STEPS = {2: "action_request", 3: "environment_observation", 4: "final_evidence"}
TRACE_SCHEMA_VERSION = "unified-mas-trace-v1"
EVALUATOR_VERSION = "webarena-propagation-evaluator-v1"


@dataclass
class WorkflowMessage:
    kind: str
    payload: dict[str, Any]


class ShoppingWorkerAgent(RoutedAgent):
    def __init__(self, client: Any, task: dict[str, Any]) -> None:
        super().__init__("Real Qwen Shopping Worker")
        self.client = client
        self.task = task

    @message_handler
    async def handle(self, message: WorkflowMessage, ctx: Any) -> WorkflowMessage:
        if message.kind == "instruction":
            prompt = (
                "You are the Shopping Worker in a real WebArena Shopping workflow. "
                "Return JSON only with keys task_id, action, product_title, product_url, quantity. "
                "This is a one-action workflow: action MUST be exactly add_to_cart. "
                "Do not return navigate, search, or a plan. Use task values exactly. Task: "
                + json.dumps(self.task, ensure_ascii=False)
            )
            return WorkflowMessage("action", parse_object(self.client.complete(prompt), {"task_id": self.task["task_id"], "worker_llm_error": "invalid action JSON"}))
        if message.kind == "observation":
            prompt = (
                "You are the Shopping Worker. Convert the delivered WebArena observation into evidence. "
                "Return JSON only with keys task_id, product_title, product_id, sku, requested_quantity, "
                "observed_quantity, cart_verified, evidence, status. Do not invent missing values. "
                "Expected task: " + json.dumps(self.task, ensure_ascii=False) +
                " Delivered observation: " + json.dumps(message.payload, ensure_ascii=False)
            )
            return WorkflowMessage("evidence", parse_object(self.client.complete(prompt), {"task_id": self.task["task_id"], "cart_verified": False, "status": "invalid evidence JSON"}))
        return WorkflowMessage("error", {"error": f"unsupported worker message {message.kind}"})


class VerifierAgent(RoutedAgent):
    def __init__(self, client: Any, task: dict[str, Any]) -> None:
        super().__init__("Real Qwen Verifier")
        self.client = client
        self.task = task

    @message_handler
    async def handle(self, message: WorkflowMessage, ctx: Any) -> WorkflowMessage:
        evidence = message.payload.get("evidence")
        prompt = (
            "You are an independent verifier for a real Shopping task. Return strict JSON only: "
            '{"decision":"accept"|"reject","task_id":string,"reason":string}. '
            "Accept only complete evidence for the exact current task, title, quantity, and cart_verified=true. "
            "Reject missing, stale, malformed, or truncated evidence.\n"
            "Current task: " + json.dumps(self.task, ensure_ascii=False) +
            "\nEvidence: " + json.dumps(evidence, ensure_ascii=False)
        )
        return WorkflowMessage("verification", {"evidence": evidence, "verdict": parse_verdict(self.client.complete(prompt))})


class DecisionAgent(RoutedAgent):
    def __init__(self, client: Any, task: dict[str, Any], role: str) -> None:
        super().__init__(f"Real Qwen {role}")
        self.client = client
        self.task = task
        self.role = role

    @message_handler
    async def handle(self, message: WorkflowMessage, ctx: Any) -> WorkflowMessage:
        evidence = message.payload.get("evidence")
        verification = message.payload.get("verification")
        prompt = (
            f"You are the {self.role} for a real Shopping task. Return strict JSON only: "
            '{"decision":"accept"|"reject","task_id":string,"reason":string}. '
            "Accept only complete evidence for the exact current task, title, quantity, and cart_verified=true. "
            "Reject missing, stale, malformed, or truncated evidence.\n"
            "Current task: " + json.dumps(self.task, ensure_ascii=False) +
            "\nEvidence: " + json.dumps(evidence, ensure_ascii=False) +
            "\nVerifier report: " + json.dumps(verification, ensure_ascii=False)
        )
        return WorkflowMessage("verdict", parse_verdict(self.client.complete(prompt)))


def parse_object(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    value = raw.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return fallback
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return fallback
    return parsed if isinstance(parsed, dict) else fallback


def normalize_single_action_request(worker_action: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed WebArena Shopping tool contract before Step 2 delivery.

    The benchmark task has one legitimate environment operation.  This keeps
    incidental plan/navigation wording from becoming a clean-baseline failure;
    the worker still produces real LLM evidence and all later LLM decisions.
    """
    return {
        "task_id": task["task_id"],
        "action": "add_to_cart",
        "product_title": task["product_title"],
        "product_url": task["product_url"],
        "quantity": task["quantity"],
    }


async def ask(runtime: SingleThreadedAgentRuntime, message: WorkflowMessage, recipient: AgentId) -> WorkflowMessage:
    response = await runtime.send_message(message, recipient)
    if not isinstance(response, WorkflowMessage):
        raise TypeError(f"unexpected AutoGen response: {response!r}")
    return response


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repeat_metadata(run_index: int) -> dict[str, int | None]:
    """Expose independent repetition identity without claiming a controllable model seed."""
    return {
        "seed_or_run_index": run_index,
        "repeat_index": run_index,
        "model_seed": None,
    }


def decision_role_for_topology(topology_name: str) -> str:
    """Return the final decision role declared by a topology definition."""
    return {
        "sequential": "Coordinator",
        "flat": "Coordinator",
        "hierarchical": "Supervisor",
        "team": "Team Lead",
        "hybrid": "Supervisor",
    }[topology_name]


def resolve_tasks(task_manifest: str | None, *, task_count: int, base_url: str) -> list[dict[str, Any]]:
    """Use a frozen manifest when supplied, otherwise discover the current live pool."""
    if task_manifest:
        frozen_tasks = load_task_manifest(Path(task_manifest))
        if len(frozen_tasks) < task_count:
            raise ValueError(f"task manifest contains {len(frozen_tasks)} tasks, fewer than requested {task_count}")
        return frozen_tasks[:task_count]
    return ShoppingHTTPExecutor(base_url).discover_tasks(max(task_count, 1))[:task_count]


async def run_with_retries(operation: Any, *, attempts: int, delay_seconds: float) -> dict[str, Any]:
    """Retry a whole run only for transient provider request failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except RuntimeError as exc:
            if "request failed:" not in str(exc) or attempt == attempts:
                raise
            print(f"RETRY attempt={attempt + 1}/{attempts} reason={exc}", flush=True)
            await asyncio.sleep(delay_seconds)
    raise AssertionError("retry loop exited without a result")


def stale_seed(target: str, task: dict[str, Any]) -> dict[str, Any]:
    if target == "WebArena Shopping":
        return {"task_id": "previous-task", "action": "add_to_cart", "product_title": "previous product", "product_url": task["product_url"], "quantity": 1}
    return {"task_id": "previous-task", "product_title": "previous product", "product_id": "old-product", "sku": "OLD-SKU", "requested_quantity": 1, "observed_quantity": 1, "cart_verified": True, "evidence": json.dumps({"task_id": "previous-task", "product_id": "old-product", "observed_quantity": 1})}


def decision_evidence_for_topology(
    topology_name: str,
    primary_evidence: dict[str, Any] | None,
    direct_evidence: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the evidence payload that the final decision agent can actually use."""
    if topology_name in {"flat", "hybrid"}:
        return direct_evidence
    return primary_evidence


def delivery_payloads(envelopes: list[MessageEnvelope]) -> list[Any]:
    """Retain every delivered payload in the causal trace, including malformed text."""
    return [envelope.payload for envelope in envelopes]


def execute_delivered_actions(executor: ShoppingHTTPExecutor, action_list: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    """Execute every delivered action so A9 is a real duplicate side effect, not trace-only noise."""
    actions = [action for action in action_list if action.get("action") == "add_to_cart"]
    if not actions:
        return {"task_id": task["task_id"], "cart_verified": False, "status": "action not delivered"}
    environment: dict[str, Any] = {}
    for action in actions:
        environment = executor.add_to_cart(action)
    if len(actions) > 1:
        environment.update(executor.verify_cart(task))
        environment["duplicate_execution_count"] = len(actions)
    return environment


def reordering_seed(target: str, task: dict[str, Any], logical_step: int) -> dict[str, Any]:
    """An earlier state from the current session, distinct from A12 cross-session replay."""
    if logical_step == 3:
        return {"task_id": task["task_id"], "cart_verified": False, "status": "earlier action state", "state_version": 0}
    return {
        "task_id": task["task_id"], "product_title": task["product_title"], "cart_verified": False,
        "requested_quantity": task["quantity"], "observed_quantity": 0, "evidence": "earlier same-session cart state", "state_version": 0,
    }


async def run_one(client: Any, task: dict[str, Any], topology_name: str, fault: str, inject_step: int, run_index: int, base_url: str, *, receiver_policy: Any = None, common_recovery: bool = False, stale_replay_source: dict[str, Any] | None = None) -> dict[str, Any]:
    if receiver_policy is not None and fault == "stale_replay":
        if not stale_replay_source or not stale_replay_source.get("source_run_id") or not stale_replay_source.get("source_message_id"):
            raise ValueError("pilot stale replay requires a real clean source")
        payload = stale_replay_source.get("payload")
        if not isinstance(payload, dict) or payload.get("task_id") in (None, task["task_id"]):
            raise ValueError("pilot stale replay requires a real clean source from a different task")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if stale_replay_source.get("payload_sha256") != digest:
            raise ValueError("stale replay source hash mismatch")
    topology = get_topology(topology_name)
    run_id = f"rq2-{topology_name}-{task['task_id']}-{fault}-step{inject_step}-r{run_index}-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = (client.call_count, client.prompt_tokens, client.completion_tokens)
    request_start = len(getattr(client, "request_log", []))
    executor = ShoppingHTTPExecutor(base_url)
    interceptor = CommunicationInterceptor(FaultScenario(fault="none", delay_ms=450, timeout_ms=800))
    runtime = SingleThreadedAgentRuntime()
    worker_id = AgentId("shopping_worker", "default")
    verifier_id = AgentId("verifier", "default")
    decision_id = AgentId("decision_agent", "default")
    await runtime.register_agent_instance(ShoppingWorkerAgent(client, task), worker_id)
    await runtime.register_agent_instance(VerifierAgent(client, task), verifier_id)
    decision_role = decision_role_for_topology(topology_name)
    await runtime.register_agent_instance(DecisionAgent(client, task, decision_role), decision_id)
    runtime.start()
    events: list[dict[str, Any]] = []
    common_recovery_events: list[dict[str, Any]] = []
    mitigation_http_requests = 0
    common_recovery_http_requests = 0

    def receive_with_policy(payload: Any, receiver: str) -> Any:
        nonlocal mitigation_http_requests
        receipt_start = len(getattr(executor, "http_receipts", []))

        def readback() -> dict[str, Any]:
            nonlocal mitigation_http_requests
            try:
                return executor.reobserve_cart(task)
            finally:
                mitigation_http_requests += getattr(executor, "readback_http_request_count", 0)

        received = receiver_policy.receive(payload, readback, receiver=receiver)
        event = receiver_policy.events[-1]
        receipts = getattr(executor, "http_receipts", [])
        event["readback_receipt_indices"] = list(range(receipt_start, len(receipts)))
        if event.get("replacement_used") and not event.get("readback_called"):
            event["cached_readback_event_index"] = next((i for i, prior in enumerate(receiver_policy.events[:-1])
                                                        if prior.get("readback_called") and prior.get("replacement_used")), None)
        return received

    def deliver(logical_step: int, source: str, target: str, payload: dict[str, Any], *, faultable: bool) -> list[dict[str, Any]]:
        active = fault if faultable and logical_step == inject_step else "none"
        interceptor.scenario = FaultScenario(fault=active, delay_ms=450, timeout_ms=800)
        if active == "stale_replay" and target not in interceptor._last_by_target:
            replay = stale_replay_source["payload"] if stale_replay_source else stale_seed(target, task)
            replay_id = stale_replay_source["source_message_id"] if stale_replay_source else f"{run_id}:stale-seed"
            interceptor._last_by_target[target] = MessageEnvelope("Previous Task Worker", target, copy.deepcopy(replay), logical_message_id=replay_id)
        if active == "reordering" and target not in interceptor._last_by_target:
            interceptor._last_by_target[target] = MessageEnvelope("Earlier Same-session State", target, reordering_seed(target, task, logical_step), logical_message_id=f"{run_id}:reordering-seed")
        result = interceptor.transmit(MessageEnvelope(source, target, copy.deepcopy(payload), logical_message_id=f"{run_id}:step{logical_step}:{len(events)+1}"))
        delivered_payload = delivery_payloads(result.delivered)
        delivered = [item for item in delivered_payload if isinstance(item, dict)]
        events.append({
            "trace_schema_version": TRACE_SCHEMA_VERSION, "run_id": run_id, "trace_id": trace_id,
            "dataset": "WebArena", "benchmark": "WebArena-Verified-Shopping", "framework": "AutoGen",
            **repeat_metadata(run_index),
            "topology": topology.name, "architecture_taxonomy": topology.architecture_taxonomy,
            "agent_roles": list(topology.agent_roles), "task_id": task["task_id"],
            "step_id": f"step-{logical_step:03d}", "step_index": logical_step, "timestamp": now(),
            "source_agent": source, "target_agent": target, "message_id": f"{run_id}:step{logical_step}:{len(events)+1}",
            "original_message": payload, "delivered_message": delivered_payload, "fault_id": fault if active != "none" else "none",
            "fault_type": active, "fault_applied": result.fault_injected,
            "fault_parameters": {"injection_step": inject_step, "logical_step": logical_step},
            "latency_ms": result.latency_ms, "first_divergence": f"A:fault_applied:step-{logical_step:03d}" if result.fault_injected else "none",
            "observed_runtime_effect": result.notes, "observed_A_symptom": result.a_layer_symptom,
            "observed_M_consequence": ["none"], "propagation_path": [result.a_layer_symptom] if result.a_layer_symptom != "none" else ["clean"],
            "recovery_detected": False, "recovery_type": "none", "recovery_evidence": "", "propagation_class": "none", "notes": result.notes,
        })
        return delivered

    try:
        instruction_source = decision_role_for_topology(topology_name)
        worker_instruction = {"task_id": task["task_id"], "request": "Add the selected product to a fresh Guest Cart."}
        deliver(1, instruction_source, "Shopping Worker" if topology_name != "team" else "Execution Team", worker_instruction, faultable=False)
        action_message = await ask(runtime, WorkflowMessage("instruction", worker_instruction), worker_id)
        action_source = "Execution Team" if topology_name == "team" else "Shopping Worker"
        action_request = normalize_single_action_request(action_message.payload, task)
        action_list = deliver(2, action_source, "WebArena Shopping", action_request, faultable=True)
        environment = execute_delivered_actions(executor, action_list, task)
        observation_target = "Execution Team" if topology_name == "team" else "Shopping Worker"
        observation_list = deliver(3, "WebArena Shopping", observation_target, environment, faultable=True)
        observation = observation_list[-1] if observation_list else {"task_id": task["task_id"], "cart_verified": False, "status": "observation omitted"}
        evidence_message = await ask(runtime, WorkflowMessage("observation", observation), worker_id)
        step4_target = {"sequential": "Verifier", "flat": "Verifier", "hierarchical": "Supervisor", "team": "Verification Team", "hybrid": "Verifier"}[topology_name]
        evidence_list = deliver(4, action_source, step4_target, evidence_message.payload, faultable=True)
        evidence = evidence_list[-1] if evidence_list else None
        if receiver_policy is not None:
            raw_delivered = events[-1]["delivered_message"]
            evidence = receive_with_policy(raw_delivered[-1] if raw_delivered else None, step4_target)
        direct_evidence = None
        if topology_name in {"flat", "hybrid"}:
            direct_target = "Coordinator" if topology_name == "flat" else "Supervisor"
            direct_list = deliver(4, action_source, direct_target, evidence_message.payload, faultable=False)
            direct_evidence = direct_list[-1] if direct_list else None
            if receiver_policy is not None:
                raw_direct = events[-1]["delivered_message"]
                direct_evidence = receive_with_policy(raw_direct[-1] if raw_direct else None, direct_target)
        decision_evidence = decision_evidence_for_topology(topology_name, evidence, direct_evidence)
        if topology_name == "hierarchical":
            verdict = (await ask(runtime, WorkflowMessage("decision_input", {"evidence": decision_evidence, "verification": None}), decision_id)).payload
        else:
            verification = (await ask(runtime, WorkflowMessage("evidence", {"evidence": evidence}), verifier_id)).payload
            verification_source = "Verification Team" if topology_name == "team" else "Verifier"
            decision_target = {"sequential": "Coordinator", "flat": "Coordinator", "team": "Team Lead", "hybrid": "Supervisor"}[topology_name]
            verification_list = deliver(5, verification_source, decision_target, verification, faultable=False)
            verification_for_decision = verification_list[-1] if verification_list else {"verdict": {"decision": "reject", "task_id": task["task_id"], "reason": "verification omitted"}, "evidence": None}
            verdict = (await ask(runtime, WorkflowMessage("decision_input", {"evidence": decision_evidence, "verification": verification_for_decision.get("verdict")}), decision_id)).payload
        verification_verdict = verification_for_decision.get("verdict") if topology_name != "hierarchical" else None
        if common_recovery and (verdict.get("decision") != "accept" or verdict.get("task_id") != task["task_id"]):
            recovery_event = {"trigger": "final_rejected_or_unbound", "readback_called": True,
                              "before_decision_evidence": copy.deepcopy(decision_evidence),
                              "before_primary_evidence": copy.deepcopy(evidence),
                              "before_verification_verdict": copy.deepcopy(verification_verdict),
                              "before_verdict": copy.deepcopy(verdict), "receiver": decision_role,
                              "started_at": now()}
            receipt_start = len(getattr(executor, "http_receipts", []))
            fresh = executor.reobserve_cart(task)
            common_recovery_http_requests = getattr(executor, "readback_http_request_count", 0)
            fresh_list = deliver(6, "WebArena Shopping", decision_role, fresh, faultable=False)
            decision_evidence = fresh_list[-1] if fresh_list else None
            verdict = (await ask(runtime, WorkflowMessage("decision_input", {"evidence": decision_evidence,
                                                                                        "verification": None}), decision_id)).payload
            recovery_event.update(readback=copy.deepcopy(fresh), replacement_used=bool(fresh_list),
                                  readback_receipts=copy.deepcopy(getattr(executor, "http_receipts", [])[receipt_start:]),
                                  readback_receipt_indices=list(range(receipt_start, len(getattr(executor, "http_receipts", [])))),
                                  after_verdict=copy.deepcopy(verdict), completed_at=now())
            common_recovery_events.append(recovery_event)
        accepted = verdict.get("decision") == "accept"
        complete = bool(decision_evidence and decision_evidence.get("task_id") == task["task_id"] and decision_evidence.get("product_title") == task["product_title"] and decision_evidence.get("cart_verified") is True and decision_evidence.get("requested_quantity") == task["quantity"] and decision_evidence.get("observed_quantity") == task["quantity"] and decision_evidence.get("product_id") and decision_evidence.get("sku") and decision_evidence.get("evidence"))
        success = bool(environment.get("cart_verified")) and accepted and complete
        evaluation = evaluate_propagation(task=task, evidence=decision_evidence, decision=str(verdict.get("decision", "unknown")), final_task_success=success)
        manifestation = classify_manifestation(topology_name, evidence, direct_evidence, verification_verdict)
        axes = classify_consequence_axes(
            task=task,
            primary_evidence=evidence,
            decision_evidence=decision_evidence,
            verification=verification_verdict,
            final_decision=str(verdict.get("decision", "unknown")),
            final_task_success=success,
            duplicate_execution_count=int(environment.get("duplicate_execution_count", 0)),
            observed_quantity=environment.get("observed_quantity"),
        )
        for event in events:
            event["architecture_manifestation"] = manifestation.label
            event["architecture_manifestation_evidence"] = manifestation.evidence
        if axes.system or axes.semantic:
            events[-1]["system_consequences"] = axes.system or ["none"]
            events[-1]["semantic_consequences"] = axes.semantic or ["none"]
        if evaluation.consequences:
            events[-1]["observed_M_consequence"] = evaluation.consequences
            events[-1]["propagation_path"].extend(evaluation.consequences)
            events[-1]["propagation_class"] = evaluation.propagation_class
        a_symptoms = sorted({event["observed_A_symptom"] for event in events if event["observed_A_symptom"] != "none"}) or ["none"]
        injection_event = next((event for event in events if event["fault_applied"]), events[-1])
        return {
            "run_id": run_id, "trace_id": trace_id, "scenario": "webarena_shopping", "dataset": "WebArena", "benchmark": "WebArena-Verified-Shopping",
            "framework": "AutoGen", "topology": topology.name, "architecture_taxonomy": topology.architecture_taxonomy,
            **repeat_metadata(run_index),
            "agent_roles": list(topology.agent_roles), "message_path": [[event["source_agent"], event["target_agent"]] for event in events], "architecture_message_count": len(events),
            "task_id": task["task_id"], "condition": "clean" if fault == "none" else fault, "injection_step": inject_step,
            "model": client.model_info.model, "provider": client.model_info.provider, "fault_id": "none" if fault == "none" else f"fault-{run_id}", "fault_type": fault,
            "fault_severity": "default", "fault_parameters": {"step": inject_step, "step_name": INJECTION_STEPS[inject_step]}, "fault_applied": any(event["fault_applied"] for event in events),
            "source_agent": injection_event["source_agent"], "target_agent": injection_event["target_agent"], "original_message": injection_event["original_message"], "delivered_message": injection_event["delivered_message"],
            "first_divergence": next((event["first_divergence"] for event in events if event["first_divergence"] != "none"), "none"),
            "observed_runtime_effect": [note for event in events for note in event["observed_runtime_effect"]], "observed_A_symptom": a_symptoms,
            "observed_M_consequence": evaluation.consequences or ["none"], "recovery_detected": evaluation.recovery_detected, "recovery_type": evaluation.recovery_type, "recovery_evidence": "",
            "system_consequences": axes.system or ["none"], "semantic_consequences": axes.semantic or ["none"],
            "propagation_class": evaluation.propagation_class if evaluation.consequences else ("exposed_at_A_only" if fault != "none" else "masked"),
            "architecture_manifestation": manifestation.label, "architecture_manifestation_evidence": manifestation.evidence,
            "expected_answer": {"task_id": task["task_id"], "cart_verified": True}, "final_answer": {"verdict": verdict, "cart_verified": success}, "task_score": 1.0 if success else 0.0,
            "final_task_success": success, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "api_call_count": client.call_count - before[0],
            "prompt_tokens": client.prompt_tokens - before[1], "completion_tokens": client.completion_tokens - before[2], "total_tokens": (client.prompt_tokens - before[1]) + (client.completion_tokens - before[2]), "error": None, "events": events,
            "model_requests": copy.deepcopy(getattr(client, "request_log", [])[request_start:]),
            "http_receipts": copy.deepcopy(getattr(executor, "http_receipts", [])),
            "common_recovery_enabled": common_recovery, "common_recovery_events": common_recovery_events,
            "common_recovery_http_requests": common_recovery_http_requests,
            "stale_replay_source": copy.deepcopy(stale_replay_source) if fault == "stale_replay" else None,
            **({"mitigation_mode": receiver_policy.mode, "mitigation_events": receiver_policy.events,
                "primary_evidence": evidence, "decision_evidence": decision_evidence,
                "primary_receiver": step4_target,
                "decision_evidence_receiver": decision_role,
                "verification_verdict": verification_verdict, "environment_state": environment,
                "additional_http_requests": mitigation_http_requests}
               if receiver_policy is not None else {}),
        }
    finally:
        await runtime.stop()


def write_outputs(rows: list[dict[str, Any]], output: Path, args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "llm_communication_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for event in row["events"]:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    flat = [{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value) for key, value in row.items() if key != "events"} for row in rows]
    with (output / "llm_communication_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(flat[0])); writer.writeheader(); writer.writerows(flat)
    summary = {"total_runs": len(rows), "topologies": list(args.topologies.split(",")), "evaluator_version": EVALUATOR_VERSION, "by_topology": {name: {"runs": sum(row["topology"] == name for row in rows), "final_success": sum(row["topology"] == name and row["final_task_success"] for row in rows), "A_exposure": sum(row["topology"] == name and row["observed_A_symptom"] != ["none"] for row in rows), "M_consequence": sum(row["topology"] == name and row["observed_M_consequence"] != ["none"] for row in rows), "mean_latency_ms": round(sum(row["latency_ms"] for row in rows if row["topology"] == name) / max(1, sum(row["topology"] == name for row in rows)), 3), "mean_total_tokens": round(sum(row["total_tokens"] for row in rows if row["topology"] == name) / max(1, sum(row["topology"] == name for row in rows)), 3)} for name in args.topologies.split(",")}}
    summary.update(build_case_summary(rows))
    (output / "llm_communication_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "llm_communication_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topology", "runs", "final_success", "A_exposure", "M_consequence", "mean_latency_ms", "mean_total_tokens"]); writer.writeheader()
        writer.writerows({"topology": key, **value} for key, value in summary["by_topology"].items())
    (output / "llm_communication_summary.md").write_text("# WebArena RQ2 Architecture Comparison\n\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "rq2_consequence_summary.json").write_text(json.dumps(build_case_summary(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_representative_cases(rows, output)
    (output / "experiment_config.json").write_text(json.dumps({"framework": "AutoGen", "topologies": args.topologies.split(","), "evaluator_version": EVALUATOR_VERSION, "model": rows[0]["model"] if rows else os.getenv("LLM_MODEL"), "provider": rows[0]["provider"] if rows else os.getenv("LLM_PROVIDER"), "task_manifest": args.task_manifest, "faults": args.faults.split(","), "injection_steps": args.steps.split(","), "runs_per_condition": args.runs_per_condition, "llm_run_attempts": args.llm_run_attempts, "retry_delay_seconds": args.retry_delay_seconds, "tasks": tasks, "timestamp": now()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_checkpoint(row: dict[str, Any], output: Path) -> None:
    """Persist a completed run immediately so provider failures do not erase progress."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "llm_communication_full.checkpoint.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    run = {key: value for key, value in row.items() if key != "events"}
    with (output / "llm_communication_runs.checkpoint.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False) + "\n")
    with (output / "llm_communication_traces.checkpoint.jsonl").open("a", encoding="utf-8") as handle:
        for event in row["events"]:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _repeat_index_from_row(row: dict[str, Any]) -> int:
    if row.get("repeat_index") not in (None, ""):
        return int(row["repeat_index"])
    match = re.search(r"-r(\d+)-", str(row.get("run_id", "")))
    if not match:
        raise ValueError(f"cannot recover repeat index from run_id={row.get('run_id')}")
    return int(match.group(1))


def _row_spec_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["topology"]),
        str(row["fault_type"]),
        int(row["injection_step"]),
        _repeat_index_from_row(row),
        str(row["task_id"]),
    )


def _spec_key(spec: tuple[str, str, int, int, dict[str, Any]]) -> tuple[str, str, int, int, str]:
    topology, fault, step, run_index, task = spec
    return topology, fault, step, run_index, str(task["task_id"])


def load_checkpoint_rows(output: Path) -> list[dict[str, Any]]:
    """Load durable full-row checkpoints, falling back to legacy split files."""
    full_path = output / "llm_communication_full.checkpoint.jsonl"
    if full_path.is_file():
        rows = [json.loads(line) for line in full_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        run_path = output / "llm_communication_runs.checkpoint.jsonl"
        trace_path = output / "llm_communication_traces.checkpoint.jsonl"
        if not run_path.is_file():
            return []
        events_by_run: dict[str, list[dict[str, Any]]] = {}
        if trace_path.is_file():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event = json.loads(line)
                    events_by_run.setdefault(str(event["run_id"]), []).append(event)
        rows = []
        for line in run_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            run_id = str(row["run_id"])
            if run_id not in events_by_run:
                raise ValueError(f"checkpoint row has no trace events: {run_id}")
            rows.append({**row, "events": events_by_run[run_id]})
    keys = [_row_spec_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("checkpoint contains duplicate experiment specs")
    return rows


def pending_run_specs(
    specs: list[tuple[str, str, int, int, dict[str, Any]]],
    completed_rows: list[dict[str, Any]],
) -> list[tuple[str, str, int, int, dict[str, Any]]]:
    completed = {_row_spec_key(row) for row in completed_rows}
    return [spec for spec in specs if _spec_key(spec) not in completed]


def require_model(client: Any, expected_model: str | None) -> None:
    if expected_model and client.model_info.model != expected_model:
        raise ValueError(
            f"required model {expected_model!r}, configured model is {client.model_info.model!r}"
        )


def guard_deepseek_schedule(client: Any) -> None:
    """Refuse each new paid run when the shared off-peak guard is closed."""
    ensure_deepseek_offpeak(client.model_info.model)


def iter_run_specs(
    topologies: tuple[str, ...],
    faults: tuple[str, ...],
    steps: tuple[int, ...],
    runs_per_condition: int,
    tasks: list[dict[str, Any]],
    *,
    application_layer_full: bool = False,
    run_index_start: int = 1,
):
    """Yield a frozen matrix, limiting full-A faults to meaningful workflow edges."""
    for topology in topologies:
        for task in tasks:
            for fault in faults:
                fault_steps = (steps[0],) if fault == "none" else steps
                if application_layer_full and fault != "none":
                    fault_steps = tuple(step for step in steps if step in steps_for_fault(fault))
                for step in fault_steps:
                    for run_index in range(run_index_start, run_index_start + runs_per_condition):
                        yield topology, fault, step, run_index, copy.deepcopy(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("WEBARENA_BASE_URL", "http://localhost:7770"))
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--runs-per-condition", type=int, default=1)
    parser.add_argument("--run-index-start", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip completed experiment specs from durable checkpoints.")
    parser.add_argument("--required-model", help="Abort before any API call unless the configured model matches exactly.")
    parser.add_argument("--task-manifest", help="JSON manifest used to freeze the exact live Shopping tasks.")
    parser.add_argument("--llm-run-attempts", type=int, default=3, help="Attempts for transient provider request failures.")
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--topologies", default=",".join(SUPPORTED_TOPOLOGIES))
    parser.add_argument("--faults", default=",".join(FAULTS))
    parser.add_argument("--steps", default="2,3,4")
    parser.add_argument("--application-layer-full", action="store_true", help="Run all A1-A15 faults only at each fault's valid workflow steps.")
    parser.add_argument("--fault-step-registry", action="store_true", help="Apply the valid-step registry to an explicitly selected fault subset.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    topologies = tuple(value for value in args.topologies.split(",") if value in SUPPORTED_TOPOLOGIES)
    if not topologies:
        raise SystemExit("no supported topology selected")
    faults = ("none",) + tuple(spec.name for spec in APPLICATION_FAULTS) if args.application_layer_full else tuple(value for value in args.faults.split(",") if value in FAULTS)
    steps = tuple(int(value) for value in args.steps.split(",") if int(value) in INJECTION_STEPS)
    client = get_llm_client()
    require_model(client, args.required_model)
    tasks = resolve_tasks(args.task_manifest, task_count=args.tasks, base_url=args.base_url)
    output = Path(args.output_dir)
    rows = load_checkpoint_rows(output) if args.resume else []
    specs = list(iter_run_specs(
        topologies,
        faults,
        steps,
        args.runs_per_condition,
        tasks,
        application_layer_full=args.application_layer_full or args.fault_step_registry,
        run_index_start=args.run_index_start,
    ))
    remaining = pending_run_specs(specs, rows)
    for topology, fault, step, run_index, task in remaining:
        guard_deepseek_schedule(client)
        row = asyncio.run(run_with_retries(
            lambda: run_one(client, task, topology, fault, step, run_index, args.base_url),
            attempts=args.llm_run_attempts,
            delay_seconds=args.retry_delay_seconds,
        ))
        append_checkpoint(row, output)
        rows.append(row)
        print(f"DONE {len(rows)} topology={topology} task={task['task_id']} fault={fault} step={step} success={rows[-1]['final_task_success']}", flush=True)
    write_outputs(rows, output, args, tasks)
    print(f"WROTE {args.output_dir}")


if __name__ == "__main__":
    main()
