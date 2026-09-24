from copy import deepcopy

from mas_faults.webarena_admin_main_evaluator import evaluate_main_outcome


def _envelope(task_id="199", answer="299", *, complete=True, state_version=3):
    evidence = {
        "candidate_answer": answer,
        "evidence_summary": "visible evidence",
    }
    if complete:
        evidence["evidence_row_indices"] = [1]
    payload = {
        "evidence_result": evidence,
        "visible_evidence": [["ID", "Status"], [answer, "Pending"]],
    }
    if complete:
        payload["structured_task_evidence"] = {"row_count": 1}
    return {
        "message_id": f"m-{task_id}",
        "task_id": task_id,
        "source_session": f"session-{task_id}",
        "state_version": state_version,
        "payload": payload,
    }


def _record(*, condition, original=None, delivered=None, verification=None, success=True):
    original = original or _envelope()
    verification = verification or {
        "decision": "accept",
        "answer": "299",
        "reason": "complete",
    }
    delivered_messages = [] if delivered is None else [delivered]
    event = {
        "abstract_step": 4,
        "fault_applied": True,
        "original_message": original,
        "delivered_message": delivered,
        "delivered_messages": delivered_messages,
        "delivery_count": len(delivered_messages),
        "observed_A_symptom": "A8_message_truncation",
        "observed_runtime_effect": "faulted",
    }
    return {
        "condition": condition,
        "fault_applied": True,
        "task_id": "199",
        "topology": "sequential",
        "events": [event],
        "original_message": original,
        "delivered_message": delivered,
        "observed_A_symptom": [event["observed_A_symptom"]],
        "verification": verification,
        "final_answer": {
            "decision": "accept" if verification["decision"] == "accept" else "reject",
            "answer": verification["answer"],
            "reason": verification["reason"],
        },
        "final_task_success": success,
        "official_final_answer_evaluator": True,
        "task_score": 1.0 if success else 0.0,
        "topology_recovery_evidence": [],
        "error": None,
    }


def test_m14_only_when_valid_partial_evidence_is_accepted_as_complete() -> None:
    partial = _envelope(complete=False)
    accepted = evaluate_main_outcome(
        _record(
            condition="valid_partial_message_step4",
            delivered=partial,
            success=True,
        )
    )
    rejected_record = _record(
        condition="valid_partial_message_step4",
        delivered=partial,
        verification={"decision": "reject", "answer": "N/A", "reason": "partial"},
        success=False,
    )
    rejected_record["final_answer"] = {
        "decision": "reject",
        "answer": "N/A",
        "reason": "partial",
    }
    rejected = evaluate_main_outcome(rejected_record)

    assert "M14_partial_tool_or_message_result_acceptance" in accepted["observed_M_consequence"]
    assert "M14_partial_tool_or_message_result_acceptance" not in rejected["observed_M_consequence"]


def test_m5_only_when_cross_session_stale_evidence_is_accepted() -> None:
    stale = _envelope(task_id="288", answer="Samantha Jones")
    accepted = _record(
        condition="stale_replay_step4",
        delivered=stale,
        verification={
            "decision": "accept",
            "answer": "Samantha Jones",
            "reason": "accepted",
        },
        success=False,
    )
    accepted["final_answer"] = {
        "decision": "accept",
        "answer": "Samantha Jones",
        "reason": "accepted",
    }
    rejected = deepcopy(accepted)
    rejected["verification"] = {
        "decision": "reject",
        "answer": "N/A",
        "reason": "stale task",
    }
    rejected["final_answer"] = rejected["verification"]

    assert "M5_stale_context_acceptance" in evaluate_main_outcome(accepted)["observed_M_consequence"]
    assert "M5_stale_context_acceptance" not in evaluate_main_outcome(rejected)["observed_M_consequence"]


def test_m4_is_controlled_only_by_the_official_final_decision_result() -> None:
    correct = evaluate_main_outcome(
        _record(condition="semantic_corruption_step4", delivered=_envelope(), success=True)
    )
    wrong = evaluate_main_outcome(
        _record(condition="semantic_corruption_step4", delivered=_envelope(), success=False)
    )

    assert "M4_incorrect_collective_decision" not in correct["observed_M_consequence"]
    assert "M4_incorrect_collective_decision" in wrong["observed_M_consequence"]


def test_m4_requires_the_official_evaluator_to_have_executed() -> None:
    record = _record(
        condition="non_delivery_step4",
        delivered=None,
        verification={"decision": "reject", "answer": "N/A", "reason": "missing"},
        success=False,
    )
    record["official_final_answer_evaluator"] = False
    record["task_score"] = 0.0
    record["error"] = "ControlledRunError: coordinator output was invalid"

    outcome = evaluate_main_outcome(record)

    assert "M2_task_timeout_or_failure" in outcome["observed_M_consequence"]
    assert "M4_incorrect_collective_decision" not in outcome["observed_M_consequence"]
    assert outcome["final_decision_correct"] is None


def test_m6_requires_an_accepted_internal_inconsistency() -> None:
    original = _envelope()
    poisoned = deepcopy(original)
    poisoned["payload"]["evidence_result"]["candidate_answer"] = "Samantha Jones"
    accepted = _record(
        condition="semantic_corruption_step4",
        original=original,
        delivered=poisoned,
        verification={
            "decision": "accept",
            "answer": "Samantha Jones",
            "reason": "accepted",
        },
        success=False,
    )
    accepted["final_answer"] = accepted["verification"]
    rejected = deepcopy(accepted)
    rejected["verification"] = {
        "decision": "reject",
        "answer": "N/A",
        "reason": "inconsistent",
    }
    rejected["final_answer"] = rejected["verification"]

    assert "M6_state_inconsistency" in evaluate_main_outcome(accepted)["observed_M_consequence"]
    assert "M6_state_inconsistency" not in evaluate_main_outcome(rejected)["observed_M_consequence"]


def test_m6_excludes_missing_or_type_only_candidate_drift() -> None:
    original = _envelope()
    missing_candidate = deepcopy(original)
    missing_candidate["payload"]["evidence_result"].pop("candidate_answer")
    typed_candidate = deepcopy(original)
    typed_candidate["payload"]["evidence_result"]["candidate_answer"] = ["299"]
    typed_candidate["state_version"] = "3"

    missing_outcome = evaluate_main_outcome(
        _record(
            condition="contract_key_drift_step4",
            original=original,
            delivered=missing_candidate,
            success=True,
        )
    )
    typed_outcome = evaluate_main_outcome(
        _record(
            condition="contract_type_drift_step4",
            original=original,
            delivered=typed_candidate,
            success=True,
        )
    )

    assert "M6_state_inconsistency" not in missing_outcome["observed_M_consequence"]
    assert "M6_state_inconsistency" not in typed_outcome["observed_M_consequence"]


def test_final_success_does_not_erase_propagation_and_does_not_imply_recovery() -> None:
    partial = _envelope(complete=False)
    outcome = evaluate_main_outcome(
        _record(
            condition="valid_partial_message_step4",
            delivered=partial,
            success=True,
        )
    )

    assert outcome["final_task_success"] is True
    assert outcome["recovery_detected"] is False
    assert outcome["propagation_class"] == "silent_propagation_to_M"


def test_redundancy_recovery_requires_topology_trace_evidence_and_final_success() -> None:
    record = _record(
        condition="non_delivery_step4",
        delivered=None,
        success=True,
    )
    no_evidence = evaluate_main_outcome(record)
    record["topology"] = "flat"
    record["topology_recovery_evidence"] = [
        "flat_direct_evidence_branch_used_after_verifier_branch_fault"
    ]
    recovered = evaluate_main_outcome(record)

    assert no_evidence["recovery_detected"] is False
    assert recovered["recovery_detected"] is True
    assert recovered["recovery_type"] == "redundancy_recovery"


def test_step2_redelivery_is_trace_backed_retry_recovery() -> None:
    record = _record(
        condition="non_delivery_step2",
        delivered=None,
        success=True,
    )
    record["events"][0].update(
        {
            "abstract_step": 2,
            "original_message": {"tool": "set_range_filter", "arguments": {"field": "ID"}},
        }
    )
    record["events"].append(
        {
            "abstract_step": 2,
            "fault_applied": False,
            "original_message": record["events"][0]["original_message"],
            "delivered_messages": [record["events"][0]["original_message"]],
            "delivery_count": 1,
        }
    )

    outcome = evaluate_main_outcome(record)

    assert outcome["recovery_detected"] is True
    assert outcome["recovery_type"] == "timeout_or_retry_recovery"
    assert outcome["recovery_evidence"] == [
        "task_critical_message_redelivered_after_initial_non_delivery"
    ]


def test_later_newer_state_without_detection_is_not_framework_recovery() -> None:
    record = _record(
        condition="same_session_reordering_step3",
        delivered=_envelope(),
        success=True,
    )
    older = {"state_version": 2, "visible_evidence": [["old"]]}
    newer = {"state_version": 3, "visible_evidence": [["new"]]}
    record["events"] = [
        {
            "abstract_step": 3,
            "fault_applied": True,
            "original_message": newer,
            "delivered_message": older,
            "delivered_messages": [newer, older],
            "delivery_count": 2,
            "observed_A_symptom": "A10_message_reordering",
        },
        {
            "abstract_step": 3,
            "fault_applied": False,
            "original_message": newer,
            "delivered_message": newer,
            "delivered_messages": [newer],
            "delivery_count": 1,
            "observed_A_symptom": "none",
        },
    ]

    outcome = evaluate_main_outcome(record)

    assert outcome["recovery_detected"] is False
    assert outcome["recovery_type"] == "none"
    assert outcome["recovery_evidence"] == []


def test_clean_complete_handoff_has_no_a_or_m_consequence() -> None:
    envelope = _envelope()
    record = _record(
        condition="clean",
        delivered=envelope,
        success=True,
    )
    record["fault_applied"] = False
    record["events"][0]["fault_applied"] = False
    record["events"][0]["observed_A_symptom"] = "none"
    record["events"][0]["source_agent"] = "Evidence Worker"
    record.pop("original_message")
    record.pop("delivered_message")

    outcome = evaluate_main_outcome(record)

    assert outcome["observed_A_symptom"] == ["none"]
    assert outcome["observed_M_consequence"] == ["none"]
    assert outcome["propagation_class"] == "clean"


def test_clean_workflow_failure_has_trace_backed_m2_but_not_m4() -> None:
    record = _record(
        condition="clean",
        delivered=None,
        verification={"decision": "reject", "answer": "N/A", "reason": "invalid"},
        success=False,
    )
    record["fault_applied"] = False
    record["events"][0]["fault_applied"] = False
    record["events"][0]["observed_A_symptom"] = "none"
    record["official_final_answer_evaluator"] = False
    record["error"] = "ControlledRunError: invalid coordinator output"

    outcome = evaluate_main_outcome(record)

    assert outcome["observed_A_symptom"] == ["none"]
    assert "M2_task_timeout_or_failure" in outcome["observed_M_consequence"]
    assert "M4_incorrect_collective_decision" not in outcome["observed_M_consequence"]
    assert outcome["final_decision_correct"] is None
    assert outcome["propagation_class"] == "clean_task_failure"
    assert "system_evaluator_evidence" in outcome
    assert "semantic_evaluator_evidence" in outcome


def test_step2_fault_uses_step4_handoff_for_semantic_evaluation() -> None:
    envelope = _envelope()
    step2_fault = {
        "abstract_step": 2,
        "fault_applied": True,
        "original_message": {
            "tool": "set_range_filter",
            "arguments": {"field": "ID", "from_value": "299", "to_value": "299"},
        },
        "delivered_message": {
            "tool": "set_range_filter",
            "arguments": {"field": "ID", "from_value": "309", "to_value": "309"},
        },
        "delivered_messages": [
            {
                "tool": "set_range_filter",
                "arguments": {"field": "ID", "from_value": "309", "to_value": "309"},
            }
        ],
        "delivery_count": 1,
        "observed_A_symptom": "A6_message_semantic_corruption",
    }
    step4_handoff = {
        "abstract_step": 4,
        "source_agent": "Evidence Worker",
        "fault_applied": False,
        "original_message": envelope,
        "delivered_message": envelope,
        "delivered_messages": [envelope],
        "delivery_count": 1,
        "observed_A_symptom": "none",
    }
    record = _record(
        condition="semantic_corruption_step2",
        delivered=envelope,
        success=True,
    )
    record["events"] = [step2_fault, step4_handoff]

    outcome = evaluate_main_outcome(record)

    assert outcome["observed_A_symptom"] == ["A6_message_semantic_corruption"]
    assert outcome["observed_M_consequence"] == ["none"]
    assert outcome["propagation_class"] == "exposed_at_A_only"
