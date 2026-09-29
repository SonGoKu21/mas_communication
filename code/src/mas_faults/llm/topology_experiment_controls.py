from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_TOPOLOGIES = frozenset({"flat", "hierarchical", "team", "hybrid"})
_COMPARISON_MODES = frozenset({"native", "matched_control", "mechanism_ablation"})
_DECISION_POLICIES = frozenset({"strict", "conflict_aware"})
_VISIBILITY_VALUES = frozenset({"none", "delivered"})
_PROMPT_PROFILES = frozenset({"common_v1"})


@dataclass(frozen=True)
class TopologyExperimentControls:
    comparison_mode: str
    decision_policy: str
    verifier_visibility: str
    direct_state_visibility: str
    prompt_profile: str = "common_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_topology_experiment_controls(
    topology: str,
    *,
    comparison_mode: str = "native",
    decision_policy: str | None = None,
    verifier_visibility: str | None = None,
    direct_state_visibility: str | None = None,
    prompt_profile: str = "common_v1",
) -> TopologyExperimentControls:
    """Resolve topology-independent controls into concrete traceable values."""
    if topology not in _TOPOLOGIES:
        raise ValueError(f"unsupported Reddit topology: {topology}")
    if comparison_mode not in _COMPARISON_MODES:
        raise ValueError(f"unsupported comparison mode: {comparison_mode}")
    if prompt_profile not in _PROMPT_PROFILES:
        raise ValueError(f"unsupported prompt profile: {prompt_profile}")

    if comparison_mode == "native":
        if any(value is not None for value in (decision_policy, verifier_visibility, direct_state_visibility)):
            raise ValueError("native controls cannot be overridden")
        return TopologyExperimentControls(
            comparison_mode="native",
            decision_policy="conflict_aware" if topology in {"flat", "hybrid"} else "strict",
            verifier_visibility="none" if topology == "hierarchical" else "delivered",
            direct_state_visibility="delivered" if topology in {"flat", "hybrid"} else "none",
            prompt_profile=prompt_profile,
        )

    default_policy = "strict" if comparison_mode == "matched_control" else "conflict_aware"
    resolved_policy = decision_policy or default_policy
    resolved_verifier = verifier_visibility or "delivered"
    resolved_direct = direct_state_visibility or "delivered"
    if resolved_policy not in _DECISION_POLICIES:
        raise ValueError(f"unsupported decision policy: {resolved_policy}")
    if resolved_verifier not in _VISIBILITY_VALUES:
        raise ValueError(f"unsupported verifier visibility: {resolved_verifier}")
    if resolved_direct not in _VISIBILITY_VALUES:
        raise ValueError(f"unsupported direct-state visibility: {resolved_direct}")
    return TopologyExperimentControls(
        comparison_mode=comparison_mode,
        decision_policy=resolved_policy,
        verifier_visibility=resolved_verifier,
        direct_state_visibility=resolved_direct,
        prompt_profile=prompt_profile,
    )
