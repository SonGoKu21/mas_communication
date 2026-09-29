import json

from mas_faults.llm.consequence_axes import SystemRuntimeEvidence
from mas_faults.llm.shopping_strict_evaluator import (
    evaluate_shopping_run_consequences,
    is_structurally_complete_shopping_evidence,
    is_valid_shopping_evidence,
    shopping_evidence_is_internally_inconsistent,
)


TASK = {
    "task_id": "shopping-1",
    "product_title": "Tea",
    "quantity": 1,
}


def _evidence(**overrides):
    nested = {
        "task_id": "shopping-1",
        "product_title": "Tea",
        "product_id": "p-1",
        "sku": "sku-1",
        "requested_quantity": 1,
        "observed_quantity": 1,
        "cart_verified": True,
    }
    value = {
        "task_id": "shopping-1",
        "product_title": "Tea",
        "product_id": "p-1",
        "sku": "sku-1",
        "requested_quantity": 1,
        "observed_quantity": 1,
        "cart_verified": True,
        "evidence": json.dumps(nested),
    }
    value.update(overrides)
    return value


def _runtime(**overrides):
    values = {
        "workflow_completed": True,
        "final_task_success": True,
        "expected_delegations": 1,
        "completed_delegations": 1,
        "execution_count": 1,
        "expected_execution_count": 1,
    }
    values.update(overrides)
    return SystemRuntimeEvidence(**values)


def test_clean_complete_accept_has_no_axis_or_m_consequence():
    evidence = _evidence()

    result = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=evidence,
        decision_evidence=evidence,
        verification={"decision": "accept"},
        final_verdict={"decision": "accept"},
        final_task_success=True,
        runtime_evidence=_runtime(),
    )

    assert result["system_consequences"] == ["none"]
    assert result["semantic_consequences"] == ["none"]
    assert result["observed_M_consequence"] == ["none"]
    assert result["final_decision_correct"] is True


def test_rejected_missing_evidence_is_m3_and_final_failure_but_not_system_failure():
    result = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=None,
        decision_evidence=None,
        verification={"decision": "reject"},
        final_verdict={"decision": "reject"},
        final_task_success=False,
        runtime_evidence=_runtime(final_task_success=False),
    )

    assert result["system_consequences"] == ["none"]
    assert result["semantic_consequences"] == ["evidence_omission"]
    assert result["observed_M_consequence"] == [
        "M2_task_timeout_or_failure",
        "M3_incomplete_information_aggregation",
    ]


def test_accepted_stale_evidence_is_m5_and_wrong_decision_without_false_m6():
    stale = _evidence(
        task_id="previous-task",
        product_title="Old Tea",
        evidence=json.dumps({
            "task_id": "previous-task",
            "product_title": "Old Tea",
            "product_id": "p-1",
            "observed_quantity": 1,
            "cart_verified": True,
        }),
    )

    result = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=stale,
        decision_evidence=stale,
        verification={"decision": "accept"},
        final_verdict={"decision": "accept"},
        final_task_success=False,
        runtime_evidence=_runtime(final_task_success=False),
    )

    assert "constraint_loss" in result["semantic_consequences"]
    assert "stale_belief_acceptance" in result["semantic_consequences"]
    assert "state_inconsistency" not in result["semantic_consequences"]
    assert "M4_incorrect_collective_decision" in result["observed_M_consequence"]
    assert "M5_stale_context_acceptance" in result["observed_M_consequence"]
    assert "M6_state_inconsistency" not in result["observed_M_consequence"]


def test_duplicate_execution_is_system_consequence_not_evidence_omission():
    mismatched = _evidence(observed_quantity=2, cart_verified=False)

    result = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=mismatched,
        decision_evidence=mismatched,
        verification={"decision": "reject"},
        final_verdict={"decision": "reject"},
        final_task_success=False,
        runtime_evidence=_runtime(
            final_task_success=False,
            execution_count=2,
        ),
    )

    assert result["system_consequences"] == ["duplicate_execution"]
    assert result["semantic_consequences"] == ["none"]
    assert result["observed_M_consequence"] == ["M2_task_timeout_or_failure"]


def test_direct_state_and_rejecting_secondary_branch_produce_authority_conflict():
    current = _evidence()

    result = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=None,
        decision_evidence=current,
        verification={"decision": "reject"},
        final_verdict={"decision": "accept"},
        final_task_success=True,
        runtime_evidence=_runtime(expected_delegations=2, completed_delegations=1),
    )

    assert result["system_consequences"] == ["failed_delegation"]
    assert result["semantic_consequences"] == ["authority_state_conflict"]
    assert result["observed_M_consequence"] == ["none"]


def test_nested_conflict_is_m6_only_when_the_state_is_accepted():
    poisoned = _evidence(
        observed_quantity=2,
        evidence=json.dumps({
            "task_id": "shopping-1",
            "product_title": "Tea",
            "product_id": "p-1",
            "observed_quantity": 1,
            "cart_verified": True,
        }),
    )
    assert shopping_evidence_is_internally_inconsistent(poisoned) is True

    rejected = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=poisoned,
        decision_evidence=poisoned,
        verification={"decision": "reject"},
        final_verdict={"decision": "reject"},
        final_task_success=False,
        runtime_evidence=_runtime(final_task_success=False),
    )
    accepted = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=poisoned,
        decision_evidence=poisoned,
        verification={"decision": "accept"},
        final_verdict={"decision": "accept"},
        final_task_success=False,
        runtime_evidence=_runtime(final_task_success=False),
    )

    assert "state_inconsistency" not in rejected["semantic_consequences"]
    assert "M6_state_inconsistency" not in rejected["observed_M_consequence"]
    assert "state_inconsistency" in accepted["semantic_consequences"]
    assert "M6_state_inconsistency" in accepted["observed_M_consequence"]


def test_contract_types_are_not_structurally_complete():
    drifted = _evidence(observed_quantity="1", cart_verified="true")

    assert is_structurally_complete_shopping_evidence(drifted) is False
    assert is_valid_shopping_evidence(TASK, drifted) is False


def test_accepted_type_drift_is_not_classified_as_m14_partial_acceptance():
    drifted = _evidence(observed_quantity="1", cart_verified="true")

    result = evaluate_shopping_run_consequences(
        task=TASK,
        primary_evidence=drifted,
        decision_evidence=drifted,
        verification={"decision": "accept"},
        final_verdict={"decision": "accept"},
        final_task_success=False,
        runtime_evidence=_runtime(final_task_success=False),
    )

    assert "contract_semantic_drift" in result["semantic_consequences"]
    assert "partial_result_acceptance" not in result["semantic_consequences"]
    assert "M14_partial_tool_or_message_result_acceptance" not in result["observed_M_consequence"]


def test_nested_sku_or_requested_quantity_conflict_is_internal_inconsistency():
    poisoned = _evidence(evidence=json.dumps({
        "task_id": "shopping-1",
        "product_title": "Tea",
        "product_id": "p-1",
        "sku": "sku-other",
        "requested_quantity": 2,
        "observed_quantity": 1,
        "cart_verified": True,
    }))

    assert shopping_evidence_is_internally_inconsistent(poisoned) is True
