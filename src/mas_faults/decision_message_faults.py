"""Semantic faults for final MAS control and decision messages."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionMessageInterception:
    original: str
    delivered: str
    observed_a_symptom: str
    fault_parameters: dict[str, object]


def replay_stale_guidance(current_guidance: str, stale_guidance: str, *, stale_task_id: str) -> DecisionMessageInterception:
    if not stale_guidance:
        raise ValueError("stale guidance must come from a prior clean task")
    return DecisionMessageInterception(
        original=current_guidance,
        delivered=stale_guidance,
        observed_a_symptom="A12_timing_or_session_mismatch",
        fault_parameters={"semantic_operator": "cross_task_guidance_replay", "stale_task_id": stale_task_id},
    )


def drift_reviewer_contract(review_message: str, *, foreign_task_id: str) -> DecisionMessageInterception:
    try:
        payload = json.loads(review_message)
    except json.JSONDecodeError:
        payload = {"review": review_message}
    if not isinstance(payload, dict):
        payload = {"review": review_message}
    payload["task_id"] = foreign_task_id
    if "evidence_assessment" in payload:
        payload["evidence_for_task"] = payload.pop("evidence_assessment")
    if "review_assessment" in payload:
        payload["review_for_task"] = payload.pop("review_assessment")
    return DecisionMessageInterception(
        original=review_message,
        delivered=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        observed_a_symptom="A14_task_binding_mismatch",
        fault_parameters={"semantic_operator": "review_contract_task_binding_drift", "foreign_task_id": foreign_task_id},
    )
