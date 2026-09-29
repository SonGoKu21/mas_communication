from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Manifestation:
    """A structure-specific trace observation, separate from generic M consequences."""

    label: str
    evidence: dict[str, object]


def _summary(payload: dict[str, Any] | None) -> dict[str, object]:
    if not payload:
        return {"present": False}
    return {
        "present": True,
        "task_id": payload.get("task_id"),
        "product_id": payload.get("product_id"),
        "sku": payload.get("sku"),
        "requested_quantity": payload.get("requested_quantity"),
        "observed_quantity": payload.get("observed_quantity"),
        "cart_verified": payload.get("cart_verified"),
        "has_nested_evidence": bool(payload.get("evidence")),
    }


def _disagrees(primary: dict[str, Any] | None, direct: dict[str, Any] | None) -> bool:
    if not primary or not direct:
        return False
    keys = ("task_id", "product_id", "sku", "requested_quantity", "observed_quantity", "cart_verified", "evidence")
    return any(primary.get(key) != direct.get(key) for key in keys)


def classify_manifestation(
    topology: str,
    primary_evidence: dict[str, Any] | None,
    direct_evidence: dict[str, Any] | None,
    verification: dict[str, Any] | None,
) -> Manifestation:
    """Classify only visible message-path conditions; it never consults fault metadata."""

    evidence: dict[str, object] = {
        "primary": _summary(primary_evidence),
        "direct": _summary(direct_evidence),
        "verification_decision": verification.get("decision") if verification else None,
        "payload_disagreement": _disagrees(primary_evidence, direct_evidence),
    }
    if topology == "hierarchical" and primary_evidence is None:
        return Manifestation("hierarchical_control_bottleneck", evidence)
    if topology == "team" and primary_evidence is None:
        return Manifestation("team_cross_boundary_handoff_failure", evidence)
    if topology == "flat" and _disagrees(primary_evidence, direct_evidence):
        return Manifestation("flat_peer_state_divergence", evidence)
    if topology == "hybrid" and _disagrees(primary_evidence, direct_evidence):
        return Manifestation("hybrid_authority_conflict", evidence)
    return Manifestation("none", evidence)


def manifestation_as_dict(value: Manifestation) -> dict[str, object]:
    return asdict(value)
