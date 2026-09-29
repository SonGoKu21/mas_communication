"""Trace-backed communication faults for controlled WebArena Admin runs."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from mas_faults.benchmark_trace_contract import derive_propagation_class
from mas_faults.llm.consequence_axes import (
    SemanticStateEvidence,
    SystemRuntimeEvidence,
    evaluate_semantic_consequences,
    evaluate_system_consequences,
    m_consequences_from_axes,
)


@dataclass(frozen=True)
class AdminFaultSpec:
    condition: str
    fault_id: str
    fault_type: str
    severity: str
    observed_a_symptom: str
    parameters: dict[str, Any]


ADMIN_FAULT_CONDITIONS: dict[str, AdminFaultSpec] = {
    "clean": AdminFaultSpec("clean", "none", "clean", "none", "none", {}),
    "a1_moderate_delay": AdminFaultSpec(
        "a1_moderate_delay",
        "A1",
        "delay",
        "moderate_delay",
        "A1_message_latency",
        {"delay_ms": 250},
    ),
    "a1_deadline_delay": AdminFaultSpec(
        "a1_deadline_delay",
        "A1",
        "delay",
        "deadline_exceeding_delay",
        "A2_message_timeout",
        {"delay_ms": 1500, "deadline_ms": 500},
    ),
    "a5_omission": AdminFaultSpec(
        "a5_omission",
        "A5",
        "omission",
        "drop_once",
        "A5_message_omission",
        {"drop_count": 1},
    ),
    "a6_inner_evidence_poisoning": AdminFaultSpec(
        "a6_inner_evidence_poisoning",
        "A6",
        "semantic_corruption",
        "inner_evidence_poisoning",
        "A6_message_semantic_corruption",
        {"preserve_outer_binding": True},
    ),
    "a8_truncation": AdminFaultSpec(
        "a8_truncation",
        "A8",
        "truncation",
        "syntactically_valid_partial",
        "A8_message_truncation",
        {"preserve_json_syntax": True},
    ),
    "a12_stale_replay": AdminFaultSpec(
        "a12_stale_replay",
        "A12",
        "stale_replay",
        "cross_task_replay",
        "A12_timing_or_session_mismatch",
        {"replace_entire_envelope": True},
    ),
}


@dataclass(frozen=True)
class AdminDelivery:
    condition: str
    fault_id: str
    fault_type: str
    fault_severity: str
    fault_parameters: dict[str, Any]
    fault_applied: bool
    original_message: dict[str, Any]
    delivered_message: dict[str, Any] | None
    send_timestamp: str
    delivery_timestamp: str | None
    observed_runtime_effect: str
    observed_a_symptom: str
    delivery_count: int

    def as_trace_fields(self) -> dict[str, Any]:
        return asdict(self)


def make_evidence_envelope(
    *,
    message_id: str,
    task_id: str,
    source_session: str,
    state_version: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "message_id": str(message_id),
        "task_id": str(task_id),
        "source_session": str(source_session),
        "state_version": state_version,
        "source_agent": "Evidence Worker",
        "target_agent": "Coordinator",
        "payload": deepcopy(payload),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AdminCommunicationInterceptor:
    """Apply one deterministic local fault at an inter-agent handoff."""

    def __init__(
        self,
        condition: str,
        *,
        stale_message: dict[str, Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], str] = _utc_now,
    ) -> None:
        try:
            self.spec = ADMIN_FAULT_CONDITIONS[condition]
        except KeyError as exc:
            raise ValueError(f"unknown Admin fault condition: {condition}") from exc
        self.stale_message = deepcopy(stale_message)
        self.sleep_fn = sleep_fn
        self.now_fn = now_fn

    def intercept(self, message: dict[str, Any]) -> AdminDelivery:
        original = deepcopy(message)
        delivered: dict[str, Any] | None = deepcopy(message)
        effect = "clean_delivery"
        delivery_count = 1
        send_timestamp = self.now_fn()
        condition = self.spec.condition

        if condition == "a1_moderate_delay":
            self.sleep_fn(self.spec.parameters["delay_ms"] / 1000)
            effect = "delayed_then_delivered"
        elif condition == "a1_deadline_delay":
            self.sleep_fn(self.spec.parameters["deadline_ms"] / 1000)
            delivered = None
            delivery_count = 0
            effect = "deadline_exceeded_before_delivery"
        elif condition == "a5_omission":
            delivered = None
            delivery_count = 0
            effect = "message_dropped_before_delivery"
        elif condition == "a6_inner_evidence_poisoning":
            stale = self._required_stale_message()
            delivered_payload = deepcopy(delivered["payload"])
            delivered_payload["evidence_result"] = deepcopy(
                stale["payload"]["evidence_result"]
            )
            delivered["payload"] = delivered_payload
            effect = "nested_evidence_replaced_with_prior_task_evidence"
        elif condition == "a8_truncation":
            payload = deepcopy(delivered["payload"])
            evidence = payload.get("evidence_result") or {}
            payload["evidence_result"] = {
                "candidate_answer": evidence.get("candidate_answer", ""),
                "evidence_summary": str(evidence.get("evidence_summary", ""))[:48],
            }
            visible = payload.get("visible_evidence") or []
            payload["visible_evidence"] = visible[:2]
            payload["structured_numeric_evidence"] = {}
            delivered["payload"] = payload
            effect = "syntactically_valid_partial_message_delivered"
        elif condition == "a12_stale_replay":
            delivered = self._required_stale_message()
            effect = "prior_task_envelope_replayed"

        delivery_timestamp = self.now_fn() if delivered is not None else None
        return AdminDelivery(
            condition=condition,
            fault_id=self.spec.fault_id,
            fault_type=self.spec.fault_type,
            fault_severity=self.spec.severity,
            fault_parameters=deepcopy(self.spec.parameters),
            fault_applied=condition != "clean",
            original_message=original,
            delivered_message=delivered,
            send_timestamp=send_timestamp,
            delivery_timestamp=delivery_timestamp,
            observed_runtime_effect=effect,
            observed_a_symptom=self.spec.observed_a_symptom,
            delivery_count=delivery_count,
        )

    def _required_stale_message(self) -> dict[str, Any]:
        stale = deepcopy(self.stale_message)
        if not isinstance(stale, dict) or not isinstance(stale.get("payload"), dict):
            raise ValueError(
                f"condition {self.spec.condition} requires a valid stale_message"
            )
        if not isinstance(stale["payload"].get("evidence_result"), dict):
            raise ValueError(
                f"condition {self.spec.condition} requires stale_message evidence_result"
            )
        return stale


def _flatten_envelope(message: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {}
    payload = message.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    evidence = payload.get("evidence_result")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "task_id": message.get("task_id"),
        "message_id": message.get("message_id"),
        "source_session": message.get("source_session"),
        "state_version": message.get("state_version"),
        "candidate_answer": evidence.get("candidate_answer"),
        "evidence_summary": evidence.get("evidence_summary"),
        "evidence_row_indices": evidence.get("evidence_row_indices"),
        "visible_evidence": payload.get("visible_evidence"),
        "structured_numeric_evidence": payload.get("structured_numeric_evidence"),
    }


def _contract_valid(state: dict[str, Any]) -> bool:
    indices = state.get("evidence_row_indices")
    return bool(
        isinstance(state.get("task_id"), str)
        and isinstance(state.get("candidate_answer"), str)
        and isinstance(state.get("evidence_summary"), str)
        and isinstance(indices, list)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in indices)
        and isinstance(state.get("visible_evidence"), list)
        and isinstance(state.get("structured_numeric_evidence"), dict)
    )


def _candidate(message: dict[str, Any] | None) -> str | None:
    state = _flatten_envelope(message)
    value = state.get("candidate_answer")
    return value if isinstance(value, str) else None


def build_admin_outcome(
    *,
    expected_task_id: str,
    delivery: AdminDelivery,
    decision: dict[str, Any],
    final_task_success: bool,
    workflow_completed: bool,
) -> dict[str, Any]:
    """Derive A/M/recovery only from message state, decision, and task score."""
    delivered_state = _flatten_envelope(delivery.delivered_message)
    required_fields = (
        "task_id",
        "candidate_answer",
        "evidence_summary",
        "evidence_row_indices",
        "visible_evidence",
        "structured_numeric_evidence",
    )
    missing = tuple(key for key in required_fields if delivered_state.get(key) is None)
    contract_valid = _contract_valid(delivered_state)
    accepted = decision.get("decision") == "accept"
    original_candidate = _candidate(delivery.original_message)
    delivered_candidate = _candidate(delivery.delivered_message)
    final_answer = decision.get("answer")

    stale_binding = bool(delivered_state) and str(delivered_state.get("task_id")) != str(
        expected_task_id
    )
    poisoned_inner_state = bool(
        delivery.condition == "a6_inner_evidence_poisoning"
        and delivered_candidate != original_candidate
    )
    state_inconsistent = bool(accepted and (stale_binding or poisoned_inner_state))

    system_evidence = SystemRuntimeEvidence(
        workflow_completed=workflow_completed,
        final_task_success=final_task_success,
        timeout_observed=delivery.observed_a_symptom == "A2_message_timeout",
        expected_delegations=1,
        completed_delegations=delivery.delivery_count,
        delivery_count=delivery.delivery_count,
        expected_delivery_count=1,
        execution_count=1 if delivery.delivery_count else 0,
        expected_execution_count=1,
    )
    semantic_evidence = SemanticStateEvidence(
        expected_task_id=str(expected_task_id),
        accepted_evidence=delivered_state,
        final_decision=str(decision.get("decision", "reject")),
        required_fields=required_fields,
        accepted_complete=not missing and contract_valid,
        available_complete_evidence=not missing and contract_valid,
        missing_required_fields=missing,
        contract_type_valid=contract_valid,
        current_state_version=3,
        accepted_state_version=delivered_state.get("state_version"),
        state_inconsistent=state_inconsistent,
    )
    system = evaluate_system_consequences(system_evidence)
    semantic = evaluate_semantic_consequences(semantic_evidence)
    observed_m = m_consequences_from_axes(
        system_consequences=system,
        semantic_consequences=semantic,
        final_decision_correct=final_task_success,
        final_task_success=final_task_success,
    )

    recovery_evidence: list[str] = []
    recovery_detected = bool(
        delivery.condition == "a6_inner_evidence_poisoning"
        and final_task_success
        and accepted
        and original_candidate
        and final_answer == original_candidate
        and delivered_candidate != original_candidate
        and delivery.delivered_message is not None
        and delivery.delivered_message.get("payload", {}).get("visible_evidence")
        == delivery.original_message.get("payload", {}).get("visible_evidence")
    )
    if recovery_detected:
        recovery_evidence.append(
            "Coordinator returned the original current-evidence candidate instead of the delivered poisoned candidate while current visible rows remained available."
        )
    recovery_type = "reasoning_recovery" if recovery_detected else "none"
    observed_a = [delivery.observed_a_symptom]
    propagation_class = derive_propagation_class(
        fault_applied=delivery.fault_applied,
        observed_a_symptom=observed_a,
        observed_m_consequence=observed_m,
        recovery_detected=recovery_detected,
        final_task_success=final_task_success,
    )
    return {
        "observed_A_symptom": observed_a,
        "observed_M_consequence": observed_m or ["none"],
        "system_consequences": system or ["none"],
        "semantic_consequences": semantic or ["none"],
        "system_evaluator_evidence": asdict(system_evidence),
        "semantic_evaluator_evidence": asdict(semantic_evidence),
        "final_decision_correct": final_task_success,
        "recovery_detected": recovery_detected,
        "recovery_type": recovery_type,
        "recovery_evidence": recovery_evidence,
        "propagation_class": propagation_class,
        "fault_detection_evidence": (
            [str(decision.get("reason", ""))]
            if delivery.fault_applied and decision.get("decision") == "reject"
            else []
        ),
    }
