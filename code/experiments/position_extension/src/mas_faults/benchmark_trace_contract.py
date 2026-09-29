"""Canonical run-level trace contract shared by real benchmark adapters.

The contract deliberately treats communication propagation and final task
outcome as independent observations.  It does not assign an M-layer label;
adapters must supply those labels from actual task state and trace evidence.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from mas_faults.llm.consequence_axes import (
    SemanticStateEvidence,
    SystemRuntimeEvidence,
    evaluate_semantic_consequences,
    evaluate_system_consequences,
    m_consequences_from_axes,
)


CANONICAL_PROPAGATION_CLASSES = frozenset(
    {
        "clean",
        "clean_task_failure",
        "masked",
        "exposed_at_A_only",
        "detected_and_recovered",
        "detected_but_unrecovered",
        "silent_propagation_to_M",
        "propagated_to_M_final_failure",
    }
)


# Study-level axes are intentionally broader than the M-layer taxonomy.  A
# benchmark adapter may record an axis directly when its runtime trace proves
# it (for example, duplicate execution), while common M labels remain mapped
# below for adapters that only emit M-layer observations.
SYSTEM_CONSEQUENCE_VOCABULARY = frozenset(
    {
        "none",
        "task_failure",
        "task_timeout",
        "failed_delegation",
        "quorum_failure",
        "duplicate_delivery",
        "duplicate_execution",
        "split_brain",
    }
)
SEMANTIC_CONSEQUENCE_VOCABULARY = frozenset(
    {
        "none",
        "constraint_loss",
        "evidence_omission",
        "plan_drift",
        "role_desynchronization",
        "stale_belief_acceptance",
        "false_consensus",
        "partial_result_acceptance",
        "contract_semantic_drift",
        "incorrect_verification",
        # Existing benchmark-specific, trace-backed labels retained for
        # backwards-compatible result aggregation.
        "incorrect_collective_decision",
        "state_inconsistency",
        "authority_state_conflict",
    }
)


SYSTEM_CONSEQUENCE_FROM_M = {
    "M2_task_timeout_or_failure": "task_failure",
}
SEMANTIC_CONSEQUENCE_FROM_M = {
    "M3_incomplete_information_aggregation": "evidence_omission",
    "M4_incorrect_collective_decision": "incorrect_collective_decision",
    "M5_stale_context_acceptance": "stale_belief_acceptance",
    "M6_state_inconsistency": "state_inconsistency",
    "M14_partial_tool_or_message_result_acceptance": "partial_result_acceptance",
}

AXIS_EVALUATION_MODES = frozenset({"legacy_m_compat", "strict_evidence_v1"})


def _labels(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return ["none"]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return _labels(parsed)
        return [value] if value else ["none"]
    labels = list(dict.fromkeys(str(item) for item in value if str(item)))
    non_empty = [label for label in labels if label != "none"]
    return non_empty or ["none"]


def _merge_labels(*values: str | Iterable[str] | None) -> list[str]:
    labels: list[str] = []
    for value in values:
        labels.extend(_labels(value))
    return _labels(labels)


def consequence_axis_validation_errors(
    *,
    system_consequences: str | Iterable[str] | None,
    semantic_consequences: str | Iterable[str] | None,
) -> list[str]:
    """Return explicit audit errors for axis labels outside the frozen vocabularies."""
    errors = [
        f"unknown_system_consequence:{label}"
        for label in _labels(system_consequences)
        if label not in SYSTEM_CONSEQUENCE_VOCABULARY
    ]
    errors.extend(
        f"unknown_semantic_consequence:{label}"
        for label in _labels(semantic_consequences)
        if label not in SEMANTIC_CONSEQUENCE_VOCABULARY
    )
    return errors


def has_observation(labels: str | Iterable[str] | None) -> bool:
    return any(label != "none" for label in _labels(labels))


def consequence_axes_from_m(
    observed_m_consequence: str | Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    """Derive only trace-backed system and semantic axes from M observations."""
    observed = _labels(observed_m_consequence)
    system = [SYSTEM_CONSEQUENCE_FROM_M[label] for label in observed if label in SYSTEM_CONSEQUENCE_FROM_M]
    semantic = [SEMANTIC_CONSEQUENCE_FROM_M[label] for label in observed if label in SEMANTIC_CONSEQUENCE_FROM_M]
    return list(dict.fromkeys(system)) or ["none"], list(dict.fromkeys(semantic)) or ["none"]


def derive_propagation_class(
    *,
    fault_applied: bool,
    observed_a_symptom: str | Iterable[str] | None,
    observed_m_consequence: str | Iterable[str] | None,
    recovery_detected: bool,
    final_task_success: bool,
) -> str:
    """Classify from evidence without allowing final success to erase M state."""
    a_exposed = has_observation(observed_a_symptom)
    propagated = has_observation(observed_m_consequence)
    if not fault_applied:
        return "clean" if final_task_success else "clean_task_failure"
    if propagated and not final_task_success:
        return "propagated_to_M_final_failure"
    if propagated and recovery_detected and final_task_success:
        return "detected_and_recovered"
    if propagated and final_task_success:
        return "silent_propagation_to_M"
    if recovery_detected and final_task_success:
        return "detected_and_recovered"
    if a_exposed and final_task_success:
        return "exposed_at_A_only"
    if a_exposed:
        return "detected_but_unrecovered"
    return "masked"


def _display_labels(labels: str | Iterable[str] | None) -> str:
    return ",".join(_labels(labels))


def _strict_derivation(
    record: dict[str, Any],
) -> tuple[list[str] | None, list[str] | None, list[str] | None, list[str]]:
    """Recompute strict labels; direct evidence is authoritative when valid."""
    errors: list[str] = []
    try:
        system_evidence = SystemRuntimeEvidence(**record["system_evaluator_evidence"])
        derived_system = _labels(evaluate_system_consequences(system_evidence))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid_system_evaluator_evidence:{type(exc).__name__}:{exc}")
        derived_system = None
    try:
        semantic_evidence = SemanticStateEvidence(**record["semantic_evaluator_evidence"])
        derived_semantic = _labels(evaluate_semantic_consequences(semantic_evidence))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid_semantic_evaluator_evidence:{type(exc).__name__}:{exc}")
        derived_semantic = None
    derived_m: list[str] | None = None
    if derived_system is not None and derived_semantic is not None:
        derived_m = _labels(m_consequences_from_axes(
            system_consequences=derived_system,
            semantic_consequences=derived_semantic,
            final_decision_correct=record.get("final_decision_correct"),
            final_task_success=bool(record.get("final_task_success", False)),
        ))
        comparisons = (
            ("system_axis", record.get("system_consequences"), derived_system),
            ("semantic_axis", record.get("semantic_consequences"), derived_semantic),
            ("M_consequence", record.get("observed_M_consequence"), derived_m),
        )
        for name, recorded, derived in comparisons:
            if _labels(recorded) != derived:
                errors.append(
                    f"{name}_mismatch:{_display_labels(recorded)}!={_display_labels(derived)}"
                )
    return derived_system, derived_semantic, derived_m, errors


def normalize_run_record(
    record: dict[str, Any],
    *,
    axis_evaluation_mode: str | None = None,
) -> dict[str, Any]:
    """Normalize adapter output into the canonical run-level fields.

    The function is intentionally permissive about benchmark-specific fields so
    old result readers remain compatible while the two adapters converge.
    """
    normalized = dict(record)
    evaluation_mode = axis_evaluation_mode or str(record.get("axis_evaluation_mode", "legacy_m_compat"))
    if evaluation_mode not in AXIS_EVALUATION_MODES:
        raise ValueError(f"unknown axis evaluation mode: {evaluation_mode}")
    normalized["axis_evaluation_mode"] = evaluation_mode
    normalized.setdefault("task_id", str(record.get("instance_id", record.get("task_slug", ""))))
    normalized.setdefault("scenario", str(record.get("benchmark", "")))
    normalized.setdefault("fault_type", str(record.get("condition", "clean")))
    normalized.setdefault("fault_severity", "none" if normalized["fault_type"] == "clean" else "default")
    normalized.setdefault("fault_parameters", {})
    normalized.setdefault("seed_or_run_index", record.get("repeat_index", 1))
    normalized.setdefault("source_agent", "")
    normalized.setdefault("target_agent", "")
    normalized.setdefault(
        "first_divergence",
        "A:fault_applied" if bool(record.get("fault_applied", False)) else "none",
    )
    normalized.setdefault("observed_runtime_effect", record.get("observed_A_symptom", "none"))
    normalized.setdefault("expected_answer", {})
    normalized.setdefault("final_answer", {})
    normalized.setdefault("error", record.get("official_runner_error"))
    normalized["observed_A_symptom"] = _labels(record.get("observed_A_symptom"))
    normalized["observed_M_consequence"] = _labels(record.get("observed_M_consequence"))
    if evaluation_mode == "strict_evidence_v1":
        normalized["system_consequences"] = _labels(record.get("system_consequences"))
        normalized["semantic_consequences"] = _labels(record.get("semantic_consequences"))
        system_evidence = record.get("system_evaluator_evidence")
        semantic_evidence = record.get("semantic_evaluator_evidence")
        normalized["system_evaluator_evidence"] = system_evidence or {}
        normalized["semantic_evaluator_evidence"] = semantic_evidence or {}
        evidence_errors: list[str] = []
        if not isinstance(system_evidence, dict) or not system_evidence:
            evidence_errors.append("missing_system_evaluator_evidence")
        if not isinstance(semantic_evidence, dict) or not semantic_evidence:
            evidence_errors.append("missing_semantic_evaluator_evidence")
        normalized["axis_evidence_validation_errors"] = evidence_errors
        derived_system, derived_semantic, derived_m, derivation_errors = _strict_derivation(record)
        normalized["strict_derivation_validation_errors"] = derivation_errors
        if derived_system is not None and derived_semantic is not None and derived_m is not None:
            normalized["system_consequences"] = derived_system
            normalized["semantic_consequences"] = derived_semantic
            normalized["observed_M_consequence"] = derived_m
        normalized.setdefault(
            "consequence_derivation_path",
            "runtime/state/final-outcome evidence -> axes/M -> propagation",
        )
    else:
        derived_system, derived_semantic = consequence_axes_from_m(normalized["observed_M_consequence"])
        normalized["system_consequences"] = _merge_labels(
            record.get("system_consequences"),
            derived_system,
        )
        normalized["semantic_consequences"] = _merge_labels(
            record.get("semantic_consequences"),
            derived_semantic,
        )
        normalized.setdefault("axis_evidence_validation_errors", [])
        normalized.setdefault("strict_derivation_validation_errors", [])
        normalized.setdefault("consequence_derivation_path", "legacy adapter/M compatibility normalization")
    normalized["consequence_axis_validation_errors"] = consequence_axis_validation_errors(
        system_consequences=normalized["system_consequences"],
        semantic_consequences=normalized["semantic_consequences"],
    )
    normalized["fault_applied"] = bool(record.get("fault_applied", False))
    normalized["recovery_detected"] = bool(record.get("recovery_detected", False))
    normalized["final_task_success"] = bool(record.get("final_task_success", False))
    normalized.setdefault("task_score", 1.0 if normalized["final_task_success"] else 0.0)
    adapter_class = str(record.get("propagation_class", ""))
    derived_class = derive_propagation_class(
        fault_applied=normalized["fault_applied"],
        observed_a_symptom=normalized["observed_A_symptom"],
        observed_m_consequence=normalized["observed_M_consequence"],
        recovery_detected=normalized["recovery_detected"],
        final_task_success=normalized["final_task_success"],
    )
    if evaluation_mode == "strict_evidence_v1":
        normalized["propagation_class"] = derived_class
        normalized["propagation_class_validation_errors"] = (
            [f"adapter_class_mismatch:{adapter_class}!={derived_class}"]
            if adapter_class in CANONICAL_PROPAGATION_CLASSES and adapter_class != derived_class
            else []
        )
    else:
        normalized["propagation_class"] = (
            adapter_class
            if normalized["fault_applied"] and adapter_class in CANONICAL_PROPAGATION_CLASSES
            else derived_class
        )
        normalized.setdefault("propagation_class_validation_errors", [])
    return normalized
