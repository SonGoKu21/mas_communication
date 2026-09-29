"""Topology-specific, communication-realizable fault profiles for SWE-bench.

Each profile selects an existing interceptor edge and an existing A-layer fault
operator.  A profile is an experimental hypothesis, not a synthetic outcome:
the evaluator still derives M-layer consequences from the run trace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class DesignedFaultProfile:
    fault_id: str
    topology: str
    condition: str
    injection_slot: str
    semantic_slot: str
    structural_mechanism: str
    expected_a_symptom: str
    candidate_m_consequences: tuple[str, ...]


DESIGNED_FAULTS: tuple[DesignedFaultProfile, ...] = (
    DesignedFaultProfile(
        fault_id="sequential_critical_verdict_omission",
        topology="sequential",
        condition="a5_omission",
        injection_slot="tester_to_verifier",
        semantic_slot="S3_evidence_observation",
        structural_mechanism="single_critical_path",
        expected_a_symptom="A5_message_omission",
        candidate_m_consequences=("M3_incomplete_information_aggregation", "M2_task_timeout_or_failure"),
    ),
    DesignedFaultProfile(
        fault_id="flat_peer_state_divergence",
        topology="flat",
        condition="a12_stale_replay",
        injection_slot="tester_to_coder",
        semantic_slot="S3_evidence_observation",
        structural_mechanism="asymmetric_peer_state",
        expected_a_symptom="A12_timing_or_session_mismatch",
        candidate_m_consequences=("M5_stale_context_acceptance", "M6_state_inconsistency"),
    ),
    DesignedFaultProfile(
        fault_id="hierarchical_manager_stale_aggregation",
        topology="hierarchical",
        condition="a12_stale_replay",
        injection_slot="tester_to_manager",
        semantic_slot="S3_evidence_observation",
        structural_mechanism="manager_aggregation_bottleneck",
        expected_a_symptom="A12_timing_or_session_mismatch",
        candidate_m_consequences=("M5_stale_context_acceptance", "M6_state_inconsistency", "M4_incorrect_collective_decision"),
    ),
    DesignedFaultProfile(
        fault_id="team_cross_boundary_evidence_omission",
        topology="team",
        condition="a5_omission",
        injection_slot="tester_to_reviewer",
        semantic_slot="S3_evidence_observation",
        structural_mechanism="cross_team_handoff",
        expected_a_symptom="A5_message_omission",
        candidate_m_consequences=("M3_incomplete_information_aggregation", "M2_task_timeout_or_failure"),
    ),
    DesignedFaultProfile(
        fault_id="hybrid_bridge_feedback_truncation",
        topology="hybrid",
        condition="a8_truncation",
        injection_slot="reviewer_to_manager",
        semantic_slot="S4_feedback_decision",
        structural_mechanism="manager_team_bridge",
        expected_a_symptom="A8_message_truncation",
        candidate_m_consequences=("M3_incomplete_information_aggregation", "M14_partial_tool_or_message_result_acceptance"),
    ),
)


RUNNER_BY_TOPOLOGY = {
    "sequential": "mas_faults.swe_bench_verified_real",
    "flat": "mas_faults.swe_bench_flat_real",
    "hierarchical": "mas_faults.swe_bench_hierarchical_real",
    "team": "mas_faults.swe_bench_team_real",
    "hybrid": "mas_faults.swe_bench_hybrid_real",
}


def get_designed_fault(fault_id: str) -> DesignedFaultProfile:
    for profile in DESIGNED_FAULTS:
        if profile.fault_id == fault_id:
            return profile
    supported = ", ".join(profile.fault_id for profile in DESIGNED_FAULTS)
    raise ValueError(f"unsupported designed fault {fault_id!r}; supported: {supported}")


def profiles_for_topology(topology: str) -> tuple[DesignedFaultProfile, ...]:
    return tuple(profile for profile in DESIGNED_FAULTS if profile.topology == topology)


def profile_execution_spec(profile: DesignedFaultProfile) -> dict[str, object]:
    """Return the existing runner arguments that realize this profile."""
    return {
        "designed_fault_id": profile.fault_id,
        "runner_module": RUNNER_BY_TOPOLOGY[profile.topology],
        "topology": profile.topology,
        "condition": profile.condition,
        "injection_slot": profile.injection_slot,
        "semantic_slot": profile.semantic_slot,
        "structural_mechanism": profile.structural_mechanism,
        "expected_A_symptom": profile.expected_a_symptom,
        "candidate_M_consequences": list(profile.candidate_m_consequences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print topology-specific SWE fault profiles.")
    parser.add_argument("--fault-id", choices=[profile.fault_id for profile in DESIGNED_FAULTS])
    args = parser.parse_args()
    profiles = (get_designed_fault(args.fault_id),) if args.fault_id else DESIGNED_FAULTS
    for profile in profiles:
        print(json.dumps(profile_execution_spec(profile), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
