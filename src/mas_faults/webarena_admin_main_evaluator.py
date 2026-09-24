"""Strict trace/state evaluator for the WebArena Admin main matrix."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mas_faults.benchmark_trace_contract import derive_propagation_class
from mas_faults.llm.consequence_axes import (
    SemanticStateEvidence,
    SystemRuntimeEvidence,
    evaluate_semantic_consequences,
    evaluate_system_consequences,
    m_consequences_from_axes,
)


REQUIRED_EVIDENCE_FIELDS = (
    "task_id",
    "candidate_answer",
    "evidence_summary",
    "evidence_row_indices",
    "visible_evidence",
    "structured_task_evidence",
)


def _fault_event(record: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (event for event in record.get("events", []) if event.get("fault_applied")),
        None,
    )


def _flatten_envelope(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        return {}
    payload = envelope.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    evidence = payload.get("evidence_result")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "task_id": envelope.get("task_id"),
        "message_id": envelope.get("message_id"),
        "source_session": envelope.get("source_session"),
        "state_version": envelope.get("state_version"),
        "candidate_answer": evidence.get("candidate_answer"),
        "evidence_summary": evidence.get("evidence_summary"),
        "evidence_row_indices": evidence.get("evidence_row_indices"),
        "visible_evidence": payload.get("visible_evidence"),
        "structured_task_evidence": payload.get("structured_task_evidence"),
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
        and isinstance(state.get("structured_task_evidence"), dict)
        and isinstance(state.get("state_version"), int)
        and not isinstance(state.get("state_version"), bool)
    )


def _redelivery_recovery(
    record: dict[str, Any], fault_event: dict[str, Any]
) -> bool:
    if fault_event.get("abstract_step") != 2 or fault_event.get("delivery_count") != 0:
        return False
    original = fault_event.get("original_message")
    fault_seen = False
    for event in record.get("events", []):
        if event is fault_event:
            fault_seen = True
            continue
        if (
            fault_seen
            and event.get("abstract_step") == 2
            and event.get("original_message") == original
            and int(event.get("delivery_count") or 0) > 0
        ):
            return True
    return False


def evaluate_main_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one completed run without deriving consequences from fault names."""
    result = dict(record)
    fault_event = _fault_event(record)
    fault_applied = fault_event is not None
    handoff_event = next(
        (
            event
            for event in record.get("events", [])
            if event.get("abstract_step") == 4
            and event.get("source_agent") == "Evidence Worker"
        ),
        fault_event if fault_event and fault_event.get("abstract_step") == 4 else None,
    )
    expected_task_id = str(record.get("task_id", ""))
    final_success = bool(record.get("final_task_success"))
    official_evaluator_executed = bool(
        record.get("official_final_answer_evaluator") is True
    )
    final_decision_correct = final_success if official_evaluator_executed else None
    verification = record.get("verification") or {}
    final_answer = record.get("final_answer") or {}
    branch_accepted = verification.get("decision") == "accept"

    original_envelope = (
        handoff_event.get("original_message")
        if handoff_event
        else record.get("original_message")
    )
    delivered_envelope = (
        handoff_event.get("delivered_message")
        if handoff_event
        else record.get("delivered_message")
    )
    original_state = _flatten_envelope(original_envelope)
    delivered_state = _flatten_envelope(delivered_envelope)
    original_complete = _contract_valid(original_state)
    delivered_complete = _contract_valid(delivered_state)
    missing = tuple(
        field for field in REQUIRED_EVIDENCE_FIELDS if delivered_state.get(field) is None
    )

    original_candidate = original_state.get("candidate_answer")
    delivered_candidate = delivered_state.get("candidate_answer")
    candidate_changed = bool(
        isinstance(original_candidate, str)
        and isinstance(delivered_candidate, str)
        and original_candidate.strip() != delivered_candidate.strip()
    )
    visible_rows_unchanged = bool(
        original_state
        and delivered_state
        and original_state.get("visible_evidence")
        == delivered_state.get("visible_evidence")
    )
    state_inconsistent = bool(
        branch_accepted and candidate_changed and visible_rows_unchanged
    )
    expected_verifier_decision = (
        "accept"
        if delivered_complete
        and str(delivered_state.get("task_id")) == expected_task_id
        and not state_inconsistent
        else "reject"
    )

    timeout_observed = bool(
        fault_event
        and fault_event.get("observed_A_symptom") == "A2_message_timeout"
    ) or "timeout" in str(record.get("error") or "").lower()
    delivery_count = int(fault_event.get("delivery_count") or 0) if fault_event else 1
    expected_delegations = 1 if fault_event and delivery_count == 0 else 0
    completed_delegations = 0 if expected_delegations else 0
    execution_count = delivery_count if fault_event and fault_event.get("abstract_step") == 2 else 1
    system_evidence = SystemRuntimeEvidence(
        workflow_completed=record.get("error") in {None, ""},
        final_task_success=final_success,
        timeout_observed=timeout_observed,
        expected_delegations=expected_delegations,
        completed_delegations=completed_delegations,
        delivery_count=delivery_count,
        expected_delivery_count=1,
        execution_count=execution_count,
        expected_execution_count=1,
    )
    semantic_evidence = SemanticStateEvidence(
        expected_task_id=expected_task_id,
        accepted_evidence=delivered_state or None,
        final_decision=str(verification.get("decision", "reject")),
        required_fields=REQUIRED_EVIDENCE_FIELDS,
        accepted_complete=delivered_complete,
        available_complete_evidence=delivered_complete,
        missing_required_fields=missing,
        contract_type_valid=delivered_complete if delivered_state else None,
        current_state_version=original_state.get("state_version"),
        accepted_state_version=delivered_state.get("state_version"),
        state_inconsistent=state_inconsistent,
        verifier_decision=str(verification.get("decision", "reject")),
        verifier_expected_decision=expected_verifier_decision,
        authority_conflict=bool(
            record.get("topology") == "flat"
            and verification.get("decision") == "accept"
            and final_answer.get("decision") == "accept"
            and verification.get("answer") != final_answer.get("answer")
        ),
    )
    system = evaluate_system_consequences(system_evidence)
    semantic = evaluate_semantic_consequences(semantic_evidence)
    observed_m = m_consequences_from_axes(
        system_consequences=system,
        semantic_consequences=semantic,
        final_decision_correct=final_decision_correct,
        final_task_success=final_success,
    )

    recovery_detected = False
    recovery_type = "none"
    recovery_evidence: list[str] = []
    topology_recovery = list(record.get("topology_recovery_evidence") or [])
    if final_success and topology_recovery:
        recovery_detected = True
        recovery_type = "redundancy_recovery"
        recovery_evidence = topology_recovery
    elif final_success and fault_event and _redelivery_recovery(record, fault_event):
        recovery_detected = True
        recovery_type = "timeout_or_retry_recovery"
        recovery_evidence = [
            "task_critical_message_redelivered_after_initial_non_delivery"
        ]
    detection_evidence: list[str] = []
    if fault_event and verification.get("decision") == "reject":
        detection_evidence.append(
            f"verifier_rejected_faulted_branch:{verification.get('reason', '')}"
        )
    if recovery_type == "timeout_or_retry_recovery":
        detection_evidence.extend(recovery_evidence)

    observed_a = (
        [str(fault_event.get("observed_A_symptom", "none"))]
        if fault_event
        else ["none"]
    )
    propagation_class = derive_propagation_class(
        fault_applied=fault_applied,
        observed_a_symptom=observed_a,
        observed_m_consequence=observed_m,
        recovery_detected=recovery_detected,
        final_task_success=final_success,
    )
    result.update(
        {
            "fault_applied": fault_applied,
            "observed_A_symptom": observed_a,
            "observed_M_consequence": observed_m or ["none"],
            "system_consequences": system or ["none"],
            "semantic_consequences": semantic or ["none"],
            "system_evaluator_evidence": asdict(system_evidence),
            "semantic_evaluator_evidence": asdict(semantic_evidence),
            "final_decision_correct": final_decision_correct,
            "recovery_detected": recovery_detected,
            "recovery_type": recovery_type,
            "recovery_evidence": recovery_evidence,
            "fault_detection_evidence": detection_evidence,
            "propagation_class": propagation_class,
            "original_evidence_complete": original_complete,
            "delivered_evidence_complete": delivered_complete,
        }
    )
    return result
