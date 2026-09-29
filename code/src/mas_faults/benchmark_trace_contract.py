"""Canonical run-level trace contract shared by real benchmark adapters.

The contract deliberately treats communication propagation and final task
outcome as independent observations.  It does not assign an M-layer label;
adapters must supply those labels from actual task state and trace evidence.
"""

from __future__ import annotations

from typing import Any, Iterable


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


def _labels(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return ["none"]
    if isinstance(value, str):
        return [value] if value else ["none"]
    labels = [str(item) for item in value if str(item)]
    return list(dict.fromkeys(labels)) or ["none"]


def has_observation(labels: str | Iterable[str] | None) -> bool:
    return any(label != "none" for label in _labels(labels))


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


def normalize_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize adapter output into the canonical run-level fields.

    The function is intentionally permissive about benchmark-specific fields so
    old result readers remain compatible while the two adapters converge.
    """
    normalized = dict(record)
    normalized.setdefault("task_id", str(record.get("instance_id", "")))
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
    normalized["fault_applied"] = bool(record.get("fault_applied", False))
    normalized["recovery_detected"] = bool(record.get("recovery_detected", False))
    normalized["final_task_success"] = bool(record.get("final_task_success", False))
    normalized.setdefault("task_score", 1.0 if normalized["final_task_success"] else 0.0)
    normalized["propagation_class"] = derive_propagation_class(
        fault_applied=normalized["fault_applied"],
        observed_a_symptom=normalized["observed_A_symptom"],
        observed_m_consequence=normalized["observed_M_consequence"],
        recovery_detected=normalized["recovery_detected"],
        final_task_success=normalized["final_task_success"],
    )
    return normalized
