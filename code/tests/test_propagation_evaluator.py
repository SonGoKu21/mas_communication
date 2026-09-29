from mas_faults.llm.propagation_evaluator import evaluate_propagation


def evidence(**overrides):
    value = {
        "task_id": "shopping-001",
        "product_title": "Current product",
        "product_id": "current-product",
        "sku": "CURRENT-SKU",
        "requested_quantity": 1,
        "observed_quantity": 1,
        "cart_verified": True,
        "evidence": "{\"task_id\": \"shopping-001\"}",
    }
    value.update(overrides)
    return value


TASK = {"task_id": "shopping-001", "product_title": "Current product", "quantity": 1}


def test_m14_requires_accepted_partial_evidence():
    result = evaluate_propagation(
        task=TASK,
        evidence={"task_id": "shopping-001", "cart_verified": True},
        decision="accept",
        final_task_success=False,
    )
    assert result.consequences == [
        "M14_partial_tool_or_message_result_acceptance",
        "M2_task_timeout_or_failure",
    ]


def test_stale_rejection_does_not_assign_m5_or_m6():
    result = evaluate_propagation(
        task=TASK,
        evidence=evidence(task_id="previous-task", product_id="old-product"),
        decision="reject",
        final_task_success=False,
    )
    assert "M5_stale_context_acceptance" not in result.consequences
    assert "M6_state_inconsistency" not in result.consequences


def test_accepted_stale_evidence_assigns_m5_and_internal_inconsistency_assigns_m6():
    result = evaluate_propagation(
        task=TASK,
        evidence=evidence(task_id="previous-task", product_id="old-product"),
        decision="accept",
        final_task_success=False,
    )
    assert "M5_stale_context_acceptance" in result.consequences
    assert "M6_state_inconsistency" in result.consequences


def test_m4_requires_wrong_final_decision_with_complete_evidence():
    result = evaluate_propagation(
        task=TASK,
        evidence=evidence(),
        decision="accept",
        final_task_success=False,
    )
    assert result.consequences == ["M4_incorrect_collective_decision"]


def test_final_failure_with_rejected_complete_evidence_is_still_m2():
    result = evaluate_propagation(
        task=TASK,
        evidence=evidence(),
        decision="reject",
        final_task_success=False,
    )

    assert result.consequences == ["M2_task_timeout_or_failure"]


def test_final_success_preserves_m_propagation_only_with_trace_backed_recovery():
    result = evaluate_propagation(
        task=TASK,
        evidence=None,
        decision="reject",
        final_task_success=True,
        recovery_evidence={"mechanism": "native_retry", "accepted": True},
    )
    assert result.consequences == ["M2_task_timeout_or_failure", "M3_incomplete_information_aggregation"]
    assert result.recovery_detected is True
    assert result.propagation_class == "detected_and_recovered"


def test_recovery_is_not_inferred_from_final_success_without_trace_evidence():
    result = evaluate_propagation(
        task=TASK,
        evidence=None,
        decision="reject",
        final_task_success=True,
    )
    assert result.recovery_detected is False
    assert result.propagation_class == "silent_propagation_to_M"
