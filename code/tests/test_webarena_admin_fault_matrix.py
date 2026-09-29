from __future__ import annotations

from copy import deepcopy

import pytest

from mas_faults.webarena_admin_fault_matrix import (
    ADMIN_FAULT_CONDITIONS,
    AdminCommunicationInterceptor,
    build_admin_outcome,
    make_evidence_envelope,
)


def _envelope(task_id: str, answer: str = "CURRENT") -> dict:
    return make_evidence_envelope(
        message_id=f"msg-{task_id}",
        task_id=task_id,
        source_session=f"session-{task_id}",
        state_version=3,
        payload={
            "evidence_result": {
                "candidate_answer": answer,
                "evidence_summary": f"Evidence for {answer}",
                "evidence_row_indices": [1],
            },
            "visible_evidence": [
                ["ID", "Value"],
                ["1", answer],
            ],
            "structured_numeric_evidence": {},
        },
    )


def test_reduced_main_matrix_is_frozen():
    assert list(ADMIN_FAULT_CONDITIONS) == [
        "clean",
        "a1_moderate_delay",
        "a1_deadline_delay",
        "a5_omission",
        "a6_inner_evidence_poisoning",
        "a8_truncation",
        "a12_stale_replay",
    ]


def test_clean_uses_interceptor_path_without_changing_message():
    original = _envelope("187")
    delivery = AdminCommunicationInterceptor("clean").intercept(original)

    assert delivery.original_message == original
    assert delivery.delivered_message == original
    assert delivery.delivered_message is not original
    assert delivery.fault_applied is False
    assert delivery.observed_a_symptom == "none"
    assert delivery.send_timestamp
    assert delivery.delivery_timestamp


def test_delay_conditions_record_real_wait_and_deadline_drop():
    sleeps = []
    original = _envelope("187")

    moderate = AdminCommunicationInterceptor(
        "a1_moderate_delay", sleep_fn=sleeps.append
    ).intercept(original)
    deadline = AdminCommunicationInterceptor(
        "a1_deadline_delay", sleep_fn=sleeps.append
    ).intercept(original)

    assert sleeps == [0.25, 0.5]
    assert moderate.delivered_message == original
    assert moderate.observed_a_symptom == "A1_message_latency"
    assert deadline.delivered_message is None
    assert deadline.observed_a_symptom == "A2_message_timeout"
    assert deadline.fault_parameters == {"delay_ms": 1500, "deadline_ms": 500}


def test_semantic_faults_transform_only_the_intended_message_state():
    current = _envelope("187", "CURRENT")
    stale = _envelope("199", "STALE")

    omitted = AdminCommunicationInterceptor("a5_omission").intercept(current)
    poisoned = AdminCommunicationInterceptor(
        "a6_inner_evidence_poisoning", stale_message=stale
    ).intercept(current)
    truncated = AdminCommunicationInterceptor("a8_truncation").intercept(current)
    replayed = AdminCommunicationInterceptor(
        "a12_stale_replay", stale_message=stale
    ).intercept(current)

    assert omitted.delivered_message is None
    assert poisoned.delivered_message["task_id"] == "187"
    assert poisoned.delivered_message["message_id"] == "msg-187"
    assert poisoned.delivered_message["payload"]["visible_evidence"] == current["payload"]["visible_evidence"]
    assert poisoned.delivered_message["payload"]["evidence_result"]["candidate_answer"] == "STALE"
    assert "evidence_row_indices" not in truncated.delivered_message["payload"]["evidence_result"]
    assert isinstance(truncated.delivered_message, dict)
    assert replayed.delivered_message == stale
    assert replayed.delivered_message["task_id"] == "199"


def test_stale_dependent_fault_requires_a_valid_prior_message():
    with pytest.raises(ValueError, match="stale_message"):
        AdminCommunicationInterceptor("a12_stale_replay").intercept(_envelope("187"))


def test_m14_only_when_partial_message_is_accepted_as_complete():
    original = _envelope("187")
    delivery = AdminCommunicationInterceptor("a8_truncation").intercept(original)

    accepted = build_admin_outcome(
        expected_task_id="187",
        delivery=delivery,
        decision={"decision": "accept", "answer": "CURRENT", "reason": "Enough"},
        final_task_success=True,
        workflow_completed=True,
    )
    rejected = build_admin_outcome(
        expected_task_id="187",
        delivery=delivery,
        decision={"decision": "reject", "answer": "N/A", "reason": "Partial"},
        final_task_success=False,
        workflow_completed=True,
    )

    assert "M14_partial_tool_or_message_result_acceptance" in accepted["observed_M_consequence"]
    assert "M14_partial_tool_or_message_result_acceptance" not in rejected["observed_M_consequence"]
    assert accepted["propagation_class"] == "silent_propagation_to_M"


def test_m5_m6_and_m4_require_accepted_stale_wrong_state():
    original = _envelope("187", "CURRENT")
    stale = _envelope("199", "STALE")
    delivery = AdminCommunicationInterceptor(
        "a12_stale_replay", stale_message=stale
    ).intercept(original)

    accepted_wrong = build_admin_outcome(
        expected_task_id="187",
        delivery=delivery,
        decision={"decision": "accept", "answer": "STALE", "reason": "Accepted"},
        final_task_success=False,
        workflow_completed=True,
    )
    rejected = build_admin_outcome(
        expected_task_id="187",
        delivery=delivery,
        decision={"decision": "reject", "answer": "N/A", "reason": "Stale task"},
        final_task_success=False,
        workflow_completed=True,
    )

    assert set(accepted_wrong["observed_M_consequence"]) >= {
        "M2_task_timeout_or_failure",
        "M4_incorrect_collective_decision",
        "M5_stale_context_acceptance",
        "M6_state_inconsistency",
    }
    assert "M5_stale_context_acceptance" not in rejected["observed_M_consequence"]
    assert "M6_state_inconsistency" not in rejected["observed_M_consequence"]
    assert "M4_incorrect_collective_decision" in rejected["observed_M_consequence"]


def test_reasoning_recovery_requires_trace_proof_not_only_final_success():
    original = _envelope("187", "CURRENT")
    stale = _envelope("199", "STALE")
    poisoned = AdminCommunicationInterceptor(
        "a6_inner_evidence_poisoning", stale_message=stale
    ).intercept(original)
    delayed = AdminCommunicationInterceptor(
        "a1_moderate_delay", sleep_fn=lambda _seconds: None
    ).intercept(original)

    recovered = build_admin_outcome(
        expected_task_id="187",
        delivery=poisoned,
        decision={"decision": "accept", "answer": "CURRENT", "reason": "Visible row supports CURRENT"},
        final_task_success=True,
        workflow_completed=True,
    )
    latency_only = build_admin_outcome(
        expected_task_id="187",
        delivery=delayed,
        decision={"decision": "accept", "answer": "CURRENT", "reason": "Supported"},
        final_task_success=True,
        workflow_completed=True,
    )

    assert recovered["recovery_detected"] is True
    assert recovered["recovery_type"] == "reasoning_recovery"
    assert recovered["recovery_evidence"]
    assert latency_only["recovery_detected"] is False
    assert latency_only["propagation_class"] == "exposed_at_A_only"


def test_inputs_are_never_mutated():
    current = _envelope("187")
    stale = _envelope("199", "STALE")
    before = deepcopy((current, stale))

    AdminCommunicationInterceptor(
        "a6_inner_evidence_poisoning", stale_message=stale
    ).intercept(current)

    assert (current, stale) == before
