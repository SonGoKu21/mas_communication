from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PropagationEvaluation:
    consequences: list[str]
    recovery_detected: bool
    recovery_type: str
    propagation_class: str


def evaluate_propagation(
    *,
    task: dict[str, Any],
    evidence: dict[str, Any] | None,
    decision: str,
    final_task_success: bool,
    recovery_evidence: dict[str, Any] | None = None,
) -> PropagationEvaluation:
    complete = _is_complete(task, evidence)
    accepted = decision == "accept"
    stale = bool(evidence and evidence.get("task_id") != task["task_id"])
    consequences: list[str] = []

    if accepted and stale:
        consequences.append("M5_stale_context_acceptance")
        if _is_internally_inconsistent(task, evidence):
            consequences.append("M6_state_inconsistency")
    elif accepted and evidence and evidence.get("cart_verified") is True and not complete:
        consequences.append("M14_partial_tool_or_message_result_acceptance")

    if not complete and not accepted:
        consequences.extend(["M2_task_timeout_or_failure", "M3_incomplete_information_aggregation"])
    elif not final_task_success and accepted and complete:
        consequences.append("M4_incorrect_collective_decision")
    elif not final_task_success and accepted:
        consequences.append("M2_task_timeout_or_failure")
    elif not final_task_success and not accepted:
        consequences.append("M2_task_timeout_or_failure")

    recovery_detected = bool(recovery_evidence and recovery_evidence.get("accepted"))
    recovery_type = str(recovery_evidence.get("mechanism", "none")) if recovery_detected else "none"
    if consequences and recovery_detected and final_task_success:
        propagation_class = "detected_and_recovered"
    elif consequences and not final_task_success:
        propagation_class = "propagated_to_M_final_failure"
    elif consequences:
        propagation_class = "silent_propagation_to_M"
    else:
        propagation_class = "none"
    return PropagationEvaluation(consequences, recovery_detected, recovery_type, propagation_class)


def _is_complete(task: dict[str, Any], evidence: dict[str, Any] | None) -> bool:
    required = {"task_id", "product_title", "product_id", "sku", "requested_quantity", "observed_quantity", "cart_verified", "evidence"}
    return bool(
        evidence
        and required.issubset(evidence)
        and evidence.get("task_id") == task["task_id"]
        and evidence.get("product_title") == task["product_title"]
        and evidence.get("requested_quantity") == task["quantity"]
        and evidence.get("observed_quantity") == task["quantity"]
        and evidence.get("cart_verified") is True
    )


def _is_internally_inconsistent(task: dict[str, Any], evidence: dict[str, Any]) -> bool:
    return (
        evidence.get("task_id") != task["task_id"]
        or evidence.get("product_title") != task["product_title"]
        or evidence.get("requested_quantity") != task["quantity"]
        or evidence.get("observed_quantity") != task["quantity"]
    )
