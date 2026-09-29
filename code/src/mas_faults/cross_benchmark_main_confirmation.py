"""Run one frozen-evidence cross-benchmark topology job."""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from typing import Any, Callable

from mas_faults.benchmark_trace_contract import derive_propagation_class
from mas_faults.cross_benchmark_main_matrix import (
    BenchmarkCarrier,
    MatrixJob,
)
from mas_faults.webarena_admin_main_evaluator import evaluate_main_outcome
from mas_faults.webarena_admin_main_matrix import (
    CONDITION_BY_NAME,
    MainCommunicationInterceptor,
    MainDeliveryBatch,
)
from mas_faults.webarena_admin_topologies import run_decision_topology


def _empty_delivery(
    job: MatrixJob, carrier: BenchmarkCarrier, *, effect: str
) -> MainDeliveryBatch:
    return MainDeliveryBatch(
        condition=job.condition,
        injection_step=None,
        fault_id="none",
        fault_type="clean",
        fault_family="clean",
        fault_cause="none",
        fault_parameters={},
        fault_applied=False,
        original_message=deepcopy(carrier.envelope),
        delivered_messages=(),
        send_timestamp="",
        delivery_timestamps=(),
        observed_runtime_effect=effect,
        observed_a_symptom="none",
        parseable=True,
    )


def _event(
    *,
    trace_id: str,
    run_id: str,
    step: int,
    source: str,
    target: str,
    delivery: MainDeliveryBatch,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": run_id,
        "abstract_step": step,
        "event_type": "communication_intercepted",
        "layer": "A",
        "source_agent": source,
        "target_agent": target,
        "message_id": (
            delivery.original_message.get("message_id")
            if isinstance(delivery.original_message, dict)
            else None
        ),
        "original_message": delivery.original_message,
        "delivered_message": (
            delivery.delivered_messages[0]
            if delivery.delivered_messages
            else None
        ),
        "delivered_messages": list(delivery.delivered_messages),
        "delivery_count": delivery.delivery_count,
        "fault_id": delivery.fault_id,
        "fault_type": delivery.fault_type,
        "fault_family": delivery.fault_family,
        "fault_cause": delivery.fault_cause,
        "fault_parameters": delivery.fault_parameters,
        "fault_applied": delivery.fault_applied,
        "observed_A_symptom": delivery.observed_a_symptom,
        "observed_runtime_effect": delivery.observed_runtime_effect,
        "send_timestamp": delivery.send_timestamp,
        "delivery_timestamps": list(delivery.delivery_timestamps),
        "parseable": delivery.parseable,
    }


def _usage(client: Any) -> dict[str, int]:
    return {
        "api_call_count": int(getattr(client, "call_count", 0)),
        "prompt_tokens": int(getattr(client, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(client, "completion_tokens", 0)),
    }


def _primary_evidence_supports_task(
    delivery: MainDeliveryBatch, task_id: str
) -> bool:
    if not delivery.delivered_messages:
        return False
    envelope = delivery.delivered_messages[0]
    if not isinstance(envelope, dict) or envelope.get("task_id") != task_id:
        return False
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return False
    evidence = payload.get("evidence_result")
    structured = payload.get("structured_task_evidence")
    if not isinstance(evidence, dict) or not isinstance(structured, dict):
        return False
    return bool(
        evidence.get("task_id") == task_id
        and isinstance(evidence.get("candidate_answer"), str)
        and bool(evidence.get("candidate_answer").strip())
        and isinstance(evidence.get("evidence_summary"), str)
        and isinstance(evidence.get("evidence_row_indices"), list)
        and structured.get("task_id") == task_id
        and structured.get("official_success") is True
        and isinstance(payload.get("visible_evidence"), list)
    )


def answer_matches_expected(answer: str, expected: Any) -> bool:
    """Match a coordinator answer against a frozen deterministic contract."""
    if isinstance(expected, str):
        return answer.strip().casefold() == expected.strip().casefold()
    if not isinstance(expected, dict) or not expected:
        return False
    normalized = answer.strip().casefold()
    if set(expected) == {"must_include"}:
        required = expected["must_include"]
        return bool(
            isinstance(required, list)
            and required
            and all(str(value).casefold() in normalized for value in required)
        )
    if set(expected) == {"exact_match"}:
        values = expected["exact_match"]
        values = values if isinstance(values, list) else [values]
        return any(normalized == str(value).strip().casefold() for value in values)
    if set(expected) == {"fuzzy_match"}:
        reference = str(expected["fuzzy_match"]).strip().upper()
        observed = "".join(character for character in normalized if character.isalpha())
        return reference == "N/A" and observed in {"na", "notapplicable"}
    if set(expected) == {"state_assertion"}:
        assertion = expected["state_assertion"]
        if not isinstance(assertion, dict) or not assertion:
            return False
        try:
            observed_state = json.loads(answer)
        except (TypeError, json.JSONDecodeError):
            return False
        return observed_state == assertion
    return False


async def run_matrix_job(
    client: Any,
    *,
    carrier: BenchmarkCarrier,
    stale_carrier: BenchmarkCarrier,
    job: MatrixJob,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Apply one communication condition and execute the selected AutoGen graph."""
    if carrier.task_id != job.task_id:
        raise ValueError("matrix job and carrier task do not match")
    cell = CONDITION_BY_NAME[job.condition]
    trace_id = f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = _usage(client)
    interceptor = MainCommunicationInterceptor(
        cell, stale_message=stale_carrier.envelope, sleep_fn=sleep_fn
    )
    request = {
        "message_id": f"{job.run_id}:task-request",
        "task_id": carrier.task_id,
        "action": "execute_admitted_task",
        "instruction": carrier.instruction,
    }
    def corrupt_task_request(message: dict[str, Any]) -> dict[str, Any]:
        message["task_id"] = stale_carrier.task_id
        message["instruction"] = stale_carrier.instruction
        message["task_binding_corrupted"] = True
        return message

    step2 = interceptor.intercept(
        2,
        request,
        context={"eligible": True, "semantic_corruptor": corrupt_task_request},
    )
    events = [
        _event(
            trace_id=trace_id,
            run_id=job.run_id,
            step=2,
            source="Planner",
            target="Evidence Worker",
            delivery=step2,
        )
    ]

    evidence_message: dict[str, Any] | None = None
    if step2.delivery_count == 0:
        step4 = _empty_delivery(
            job, carrier, effect="no_evidence_after_task_request_non_delivery"
        )
    else:
        delivered_request = step2.delivered_messages[0]
        execution_carrier = (
            stale_carrier
            if isinstance(delivered_request, dict)
            and delivered_request.get("task_id") == stale_carrier.task_id
            and stale_carrier.task_id != carrier.task_id
            else carrier
        )
        current_state = deepcopy(execution_carrier.envelope)
        current_state["state_version"] = 2
        current_state["payload"]["structured_task_evidence"]["state_version"] = 2
        older_state = deepcopy(current_state)
        older_state["state_version"] = 1
        older_state["payload"]["structured_task_evidence"]["state_version"] = 1
        step3 = interceptor.intercept(
            3,
            current_state,
            context={"eligible": True, "older_message": older_state},
        )
        events.append(
            _event(
                trace_id=trace_id,
                run_id=job.run_id,
                step=3,
                source="Task Environment",
                target="Evidence Worker",
                delivery=step3,
            )
        )
        evidence_message = (
            deepcopy(step3.delivered_messages[-1])
            if step3.delivered_messages
            else None
        )
        if evidence_message is None:
            step4 = _empty_delivery(
                job, carrier, effect="no_evidence_after_state_delivery"
            )
        else:
            step4 = interceptor.intercept(
                4, evidence_message, context={"eligible": True}
            )
    step4_target = {
        "sequential": "Verifier",
        "flat": "Verifier",
        "hierarchical": "Supervisor",
    }[job.topology]
    events.append(
        _event(
            trace_id=trace_id,
            run_id=job.run_id,
            step=4,
            source="Evidence Worker",
            target=step4_target,
            delivery=step4,
        )
    )

    direct_delivery = None
    if job.topology == "flat":
        if step2.delivery_count == 0:
            direct_delivery = _empty_delivery(
                job, carrier, effect="no_direct_evidence_after_task_request_non_delivery"
            )
        elif evidence_message is None:
            direct_delivery = _empty_delivery(
                job, carrier, effect="no_direct_evidence_after_state_delivery"
            )
        else:
            direct_delivery = interceptor.intercept(
                4,
                evidence_message,
                context={"eligible": False, "branch": "direct"},
            )
        events.append(
            _event(
                trace_id=trace_id,
                run_id=job.run_id,
                step=4,
                source="Evidence Worker",
                target="Coordinator",
                delivery=direct_delivery,
            )
        )

    task_view = {
        "task_id": carrier.task_id,
        "intent": carrier.instruction,
        "benchmark": carrier.benchmark,
    }
    topology = await run_decision_topology(
        client,
        topology=job.topology,
        task=task_view,
        delivery=step4,
        direct_delivery=direct_delivery,
        domain_label=carrier.benchmark,
    )
    final_decision = topology.final_decision
    answer_accepted = bool(
        final_decision.get("decision") == "accept"
        and answer_matches_expected(
            str(final_decision.get("answer", "")), carrier.expected_answer
        )
    )
    primary_supports_task = _primary_evidence_supports_task(
        step4, carrier.task_id
    )
    direct_recovery = bool(
        topology.recovery_evidence
        and direct_delivery is not None
        and _primary_evidence_supports_task(direct_delivery, carrier.task_id)
    )
    final_success = bool(
        answer_accepted and (primary_supports_task or direct_recovery)
    )
    after = _usage(client)
    fault_event = next((event for event in events if event["fault_applied"]), None)
    row: dict[str, Any] = {
        "run_id": job.run_id,
        "trace_id": trace_id,
        "scenario": "cross_benchmark_frozen_evidence_main_confirmation",
        "benchmark": carrier.benchmark,
        "task_id": carrier.task_id,
        "condition": job.condition,
        "topology": job.topology,
        "repeat_index": job.repeat_index,
        "seed_or_run_index": job.repeat_index,
        "model": client.model_info.model,
        "provider": client.model_info.provider,
        "source_agent": fault_event.get("source_agent") if fault_event else None,
        "target_agent": fault_event.get("target_agent") if fault_event else None,
        "fault_id": fault_event.get("fault_id") if fault_event else "none",
        "fault_type": fault_event.get("fault_type") if fault_event else "clean",
        "fault_family": fault_event.get("fault_family") if fault_event else "clean",
        "fault_severity": cell.severity if fault_event else "none",
        "fault_parameters": fault_event.get("fault_parameters") if fault_event else {},
        "first_divergence": (
            f"A:communication_interceptor:step{fault_event['abstract_step']}:fault_applied"
            if fault_event
            else None
        ),
        "observed_runtime_effect": (
            fault_event.get("observed_runtime_effect") if fault_event else None
        ),
        "original_message": fault_event.get("original_message") if fault_event else carrier.envelope,
        "delivered_message": fault_event.get("delivered_message") if fault_event else carrier.envelope,
        "events": events,
        "verification": topology.verification,
        "final_answer": final_decision,
        "expected_answer": carrier.expected_answer,
        "task_score": 1.0 if final_success else 0.0,
        "final_task_success": final_success,
        "official_final_answer_evaluator": True,
        "official_source_task_success": True,
        "current_state_version": 2,
        "source_run_id": carrier.source_row.get("run_id"),
        "source_trace_id": carrier.source_row.get("trace_id"),
        "source_jsonl": carrier.source_jsonl,
        "topology_declared_edges": list(topology.declared_edges),
        "topology_used_edges": list(topology.used_edges),
        "topology_branch_inputs": topology.branch_inputs,
        "topology_recovery_evidence": list(topology.recovery_evidence),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "api_call_count": after["api_call_count"] - before["api_call_count"],
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens": (
            after["prompt_tokens"]
            + after["completion_tokens"]
            - before["prompt_tokens"]
            - before["completion_tokens"]
        ),
        "error": None,
    }
    result = evaluate_main_outcome(row)
    primary = step4.delivered_messages[0] if step4.delivered_messages else None
    nested_task_id = None
    if isinstance(primary, dict):
        payload = primary.get("payload")
        evidence = payload.get("evidence_result") if isinstance(payload, dict) else None
        if isinstance(evidence, dict):
            nested_task_id = evidence.get("task_id")
    accepted_inconsistent = bool(
        topology.verification.get("decision") == "accept"
        and isinstance(nested_task_id, str)
        and nested_task_id != carrier.task_id
        and not direct_recovery
    )
    if accepted_inconsistent:
        observed_m = list(result.get("observed_M_consequence") or [])
        if observed_m == ["none"]:
            observed_m = []
        for consequence in (
            "M6_state_inconsistency",
            "M4_incorrect_collective_decision",
        ):
            if consequence not in observed_m:
                observed_m.append(consequence)
        semantic = list(result.get("semantic_consequences") or [])
        if semantic == ["none"]:
            semantic = []
        if "state_inconsistency" not in semantic:
            semantic.append("state_inconsistency")
        result["observed_M_consequence"] = observed_m
        result["semantic_consequences"] = semantic
        result["propagation_class"] = derive_propagation_class(
            fault_applied=bool(result.get("fault_applied")),
            observed_a_symptom=result.get("observed_A_symptom") or ["none"],
            observed_m_consequence=observed_m,
            recovery_detected=bool(result.get("recovery_detected")),
            final_task_success=bool(result.get("final_task_success")),
        )
    return result
