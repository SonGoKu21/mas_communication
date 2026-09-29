from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConsequenceAxes:
    system: list[str]
    semantic: list[str]


@dataclass(frozen=True)
class SystemRuntimeEvidence:
    """Runtime-only evidence for system consequences.

    Deliberately excludes fault labels, M-layer labels, and semantic state so
    system consequences can be reproduced independently from runtime events.
    """

    workflow_completed: bool
    final_task_success: bool
    timeout_observed: bool = False
    expected_delegations: int = 0
    completed_delegations: int = 0
    expected_quorum: int = 0
    received_votes: int = 0
    delivery_count: int = 0
    expected_delivery_count: int = 1
    execution_count: int = 0
    expected_execution_count: int = 1
    authoritative_state_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticStateEvidence:
    """Task-state-only evidence for semantic consequences.

    Deliberately excludes fault labels and final task success. Acceptance and
    deterministic state invariants are sufficient to classify semantic drift.
    """

    expected_task_id: str
    accepted_evidence: Mapping[str, Any] | None
    final_decision: str
    required_fields: tuple[str, ...] = ()
    accepted_complete: bool | None = None
    available_complete_evidence: bool | None = None
    missing_required_fields: tuple[str, ...] = ()
    contract_type_valid: bool | None = None
    current_state_version: int | str | None = None
    accepted_state_version: int | str | None = None
    expected_constraints: Mapping[str, Any] = field(default_factory=dict)
    expected_plan_id: str | None = None
    actual_plan_id: str | None = None
    expected_role_bindings: Mapping[str, str] = field(default_factory=dict)
    actual_role_bindings: Mapping[str, str] = field(default_factory=dict)
    state_inconsistent: bool = False
    verifier_decision: str | None = None
    verifier_expected_decision: str | None = None
    consensus_required: int = 0
    supporting_votes: int = 0
    authority_conflict: bool = False


def evaluate_system_consequences(evidence: SystemRuntimeEvidence) -> list[str]:
    """Derive system labels only from deterministic runtime evidence."""
    labels: list[str] = []
    if evidence.timeout_observed:
        labels.append("task_timeout")
    if evidence.expected_delegations > evidence.completed_delegations:
        labels.append("failed_delegation")
    if evidence.expected_quorum > 0 and evidence.received_votes < evidence.expected_quorum:
        labels.append("quorum_failure")
    if evidence.delivery_count > evidence.expected_delivery_count:
        labels.append("duplicate_delivery")
    if evidence.execution_count > evidence.expected_execution_count:
        labels.append("duplicate_execution")
    state_hashes = {value for value in evidence.authoritative_state_hashes if value}
    if len(state_hashes) > 1:
        labels.append("split_brain")
    return labels


def _normalized_state_version(value: Any) -> tuple[str, Any] | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return ("int", int(stripped))
        except ValueError:
            return ("str", stripped)
    return None


def evaluate_semantic_consequences(evidence: SemanticStateEvidence) -> list[str]:
    """Derive semantic labels only from accepted state and task invariants."""
    labels: list[str] = []
    accepted = evidence.final_decision == "accept"
    state = evidence.accepted_evidence or {}
    missing_fields = list(dict.fromkeys([
        *evidence.missing_required_fields,
        *(key for key in evidence.required_fields if key not in state),
    ]))
    raw_partial_evidence = bool(missing_fields) or (
        evidence.accepted_complete is False and evidence.contract_type_valid is not False
    )
    partial_evidence = raw_partial_evidence and not (
        not accepted and evidence.available_complete_evidence is True
    )

    if accepted and any(state.get(key) != value for key, value in evidence.expected_constraints.items()):
        labels.append("constraint_loss")
    if partial_evidence:
        labels.append("evidence_omission")
    if (
        evidence.expected_plan_id is not None
        and evidence.actual_plan_id is not None
        and evidence.expected_plan_id != evidence.actual_plan_id
    ):
        labels.append("plan_drift")
    if evidence.expected_role_bindings and dict(evidence.expected_role_bindings) != dict(evidence.actual_role_bindings):
        labels.append("role_desynchronization")
    accepted_task_id = state.get("task_id")
    stale_task = bool(
        isinstance(accepted_task_id, (str, int))
        and not isinstance(accepted_task_id, bool)
        and str(accepted_task_id).strip()
        and str(accepted_task_id).strip() != evidence.expected_task_id
    )
    current_version = _normalized_state_version(evidence.current_state_version)
    accepted_version = _normalized_state_version(evidence.accepted_state_version)
    stale_version = bool(
        current_version is not None
        and accepted_version is not None
        and accepted_version != current_version
    )
    if accepted and (stale_task or stale_version):
        labels.append("stale_belief_acceptance")
    if accepted and evidence.consensus_required > evidence.supporting_votes:
        labels.append("false_consensus")
    if accepted and partial_evidence:
        labels.append("partial_result_acceptance")
    if accepted and evidence.contract_type_valid is False:
        labels.append("contract_semantic_drift")
    if (
        evidence.verifier_expected_decision is not None
        and evidence.verifier_decision is not None
        and evidence.verifier_expected_decision != evidence.verifier_decision
    ):
        labels.append("incorrect_verification")
    if accepted and evidence.state_inconsistent:
        labels.append("state_inconsistency")
    if evidence.authority_conflict:
        labels.append("authority_state_conflict")
    return labels


def m_consequences_from_axes(
    *,
    system_consequences: Sequence[str],
    semantic_consequences: Sequence[str],
    final_decision_correct: bool | None,
    final_task_success: bool | None = None,
) -> list[str]:
    """Map independently evaluated axes to M labels in one direction only."""
    labels: list[str] = []
    system = set(system_consequences)
    semantic = set(semantic_consequences)
    if final_task_success is False or system.intersection({"task_failure", "task_timeout"}):
        labels.append("M2_task_timeout_or_failure")
    if "evidence_omission" in semantic:
        labels.append("M3_incomplete_information_aggregation")
    if final_decision_correct is False:
        labels.append("M4_incorrect_collective_decision")
    if "stale_belief_acceptance" in semantic:
        labels.append("M5_stale_context_acceptance")
    if "state_inconsistency" in semantic:
        labels.append("M6_state_inconsistency")
    if "partial_result_acceptance" in semantic:
        labels.append("M14_partial_tool_or_message_result_acceptance")
    return labels


def classify_consequence_axes(
    *,
    task: dict[str, Any],
    primary_evidence: dict[str, Any] | None,
    decision_evidence: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    final_decision: str,
    final_task_success: bool,
    duplicate_execution_count: int,
    observed_quantity: int | None = None,
    primary_complete: bool | None = None,
    decision_complete: bool | None = None,
    state_inconsistent: bool = False,
) -> ConsequenceAxes:
    """Classify trace-backed system and semantic effects without inferring from fault names."""
    system: list[str] = []
    semantic: list[str] = []
    primary_complete = _is_complete(task, primary_evidence) if primary_complete is None else primary_complete
    decision_complete = _is_complete(task, decision_evidence) if decision_complete is None else decision_complete
    verification_rejected = (verification or {}).get("decision") == "reject"

    if not final_task_success:
        system.append("task_failure")
    if duplicate_execution_count > 1 and observed_quantity is not None and observed_quantity != task["quantity"]:
        system.append("duplicate_execution")

    if not decision_complete and final_decision == "reject":
        semantic.append("evidence_omission")
    if decision_evidence and decision_evidence.get("task_id") != task["task_id"] and final_decision == "accept":
        semantic.append("stale_belief_acceptance")
    if state_inconsistent:
        semantic.append("state_inconsistency")
    if decision_complete and not primary_complete and verification_rejected and final_decision == "reject":
        semantic.extend(["incorrect_verification", "authority_state_conflict"])
    return ConsequenceAxes(system, semantic)


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
