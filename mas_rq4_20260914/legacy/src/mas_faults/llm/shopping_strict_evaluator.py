from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from mas_faults.llm.consequence_axes import (
    SemanticStateEvidence,
    SystemRuntimeEvidence,
    evaluate_semantic_consequences,
    evaluate_system_consequences,
    m_consequences_from_axes,
)


SHOPPING_REQUIRED_FIELDS = (
    "task_id",
    "product_title",
    "product_id",
    "sku",
    "requested_quantity",
    "observed_quantity",
    "cart_verified",
    "evidence",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def shopping_evidence_contract_types_valid(evidence: dict[str, Any] | None) -> bool:
    if not isinstance(evidence, dict):
        return False
    validators = {
        "task_id": lambda value: isinstance(value, str),
        "product_title": lambda value: isinstance(value, str),
        "product_id": lambda value: isinstance(value, str) and bool(value),
        "sku": lambda value: isinstance(value, str) and bool(value),
        "requested_quantity": _is_int,
        "observed_quantity": _is_int,
        "cart_verified": lambda value: isinstance(value, bool),
        "evidence": lambda value: isinstance(value, str) and bool(value),
    }
    return all(validator(evidence[key]) for key, validator in validators.items() if key in evidence)


def is_structurally_complete_shopping_evidence(evidence: dict[str, Any] | None) -> bool:
    if not isinstance(evidence, dict) or not all(key in evidence for key in SHOPPING_REQUIRED_FIELDS):
        return False
    return shopping_evidence_contract_types_valid(evidence)


def shopping_evidence_is_internally_inconsistent(evidence: dict[str, Any] | None) -> bool:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("evidence"), str):
        return False
    try:
        nested = json.loads(evidence["evidence"])
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(nested, dict):
        return False
    comparable = (
        "task_id",
        "product_title",
        "product_id",
        "sku",
        "requested_quantity",
        "observed_quantity",
        "cart_verified",
    )
    return any(key in nested and nested.get(key) != evidence.get(key) for key in comparable)


def is_valid_shopping_evidence(task: dict[str, Any], evidence: dict[str, Any] | None) -> bool:
    return bool(
        is_structurally_complete_shopping_evidence(evidence)
        and evidence is not None
        and evidence.get("task_id") == task["task_id"]
        and evidence.get("product_title") == task["product_title"]
        and evidence.get("requested_quantity") == task["quantity"]
        and evidence.get("observed_quantity") == task["quantity"]
        and evidence.get("cart_verified") is True
        and not shopping_evidence_is_internally_inconsistent(evidence)
    )


def evaluate_shopping_run_consequences(
    *,
    task: dict[str, Any],
    primary_evidence: dict[str, Any] | None,
    decision_evidence: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    final_verdict: dict[str, Any],
    final_task_success: bool,
    runtime_evidence: SystemRuntimeEvidence,
) -> dict[str, Any]:
    deterministic_verifier_decision = None
    if verification is not None:
        deterministic_verifier_decision = (
            "accept" if is_valid_shopping_evidence(task, primary_evidence) else "reject"
        )
    expected_final_decision = (
        "accept" if is_valid_shopping_evidence(task, decision_evidence) else "reject"
    )
    missing_required_fields = tuple(
        key for key in SHOPPING_REQUIRED_FIELDS
        if not isinstance(decision_evidence, dict) or key not in decision_evidence
    )
    contract_type_valid = (
        shopping_evidence_contract_types_valid(decision_evidence)
        if isinstance(decision_evidence, dict)
        else None
    )
    semantic_evidence = SemanticStateEvidence(
        expected_task_id=task["task_id"],
        accepted_evidence=decision_evidence,
        final_decision=str(final_verdict.get("decision", "reject")),
        required_fields=SHOPPING_REQUIRED_FIELDS,
        accepted_complete=is_structurally_complete_shopping_evidence(decision_evidence),
        missing_required_fields=missing_required_fields,
        contract_type_valid=contract_type_valid,
        current_state_version=task.get("state_version"),
        accepted_state_version=(decision_evidence or {}).get("state_version"),
        expected_constraints={
            "task_id": task["task_id"],
            "product_title": task["product_title"],
            "requested_quantity": task["quantity"],
            "observed_quantity": task["quantity"],
            "cart_verified": True,
        },
        state_inconsistent=shopping_evidence_is_internally_inconsistent(decision_evidence),
        verifier_decision=(verification or {}).get("decision"),
        verifier_expected_decision=deterministic_verifier_decision,
        authority_conflict=bool(
            verification and verification.get("decision") != expected_final_decision
        ),
    )
    system = evaluate_system_consequences(runtime_evidence)
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
        "system_evaluator_evidence": asdict(runtime_evidence),
        "semantic_evaluator_evidence": asdict(semantic_evidence),
        "system_consequences": system or ["none"],
        "semantic_consequences": semantic or ["none"],
        "observed_M_consequence": observed_m or ["none"],
        "final_decision_correct": final_decision_correct,
        "consequence_derivation_path": "runtime/state evidence -> axes -> M -> propagation",
    }
