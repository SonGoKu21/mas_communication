from mas_faults.llm.consequence_axes import classify_consequence_axes


TASK = {"task_id": "shopping-001", "product_title": "Tea", "quantity": 1}
COMPLETE = {
    "task_id": "shopping-001", "product_title": "Tea", "product_id": "42", "sku": "TEA-001",
    "requested_quantity": 1, "observed_quantity": 1, "cart_verified": True, "evidence": "cart state",
}


def test_missing_primary_evidence_and_rejected_final_decision_is_system_and_semantic():
    result = classify_consequence_axes(
        task=TASK,
        primary_evidence=None,
        decision_evidence=None,
        verification={"decision": "reject"},
        final_decision="reject",
        final_task_success=False,
        duplicate_execution_count=0,
    )

    assert result.system == ["task_failure"]
    assert result.semantic == ["evidence_omission"]


def test_correct_direct_evidence_rejected_by_verifier_is_verification_and_authority_conflict():
    result = classify_consequence_axes(
        task=TASK,
        primary_evidence=None,
        decision_evidence=COMPLETE,
        verification={"decision": "reject", "reason": "missing primary evidence"},
        final_decision="reject",
        final_task_success=False,
        duplicate_execution_count=0,
    )

    assert result.system == ["task_failure"]
    assert result.semantic == ["incorrect_verification", "authority_state_conflict"]


def test_duplicate_cart_execution_requires_observed_duplicate_effect():
    result = classify_consequence_axes(
        task=TASK,
        primary_evidence=COMPLETE,
        decision_evidence=COMPLETE,
        verification={"decision": "reject"},
        final_decision="reject",
        final_task_success=False,
        duplicate_execution_count=2,
        observed_quantity=2,
    )

    assert result.system == ["task_failure", "duplicate_execution"]
    assert result.semantic == []


def test_stale_belief_requires_stale_evidence_to_be_accepted():
    stale = {**COMPLETE, "task_id": "previous-task"}
    result = classify_consequence_axes(
        task=TASK,
        primary_evidence=stale,
        decision_evidence=stale,
        verification={"decision": "accept"},
        final_decision="accept",
        final_task_success=False,
        duplicate_execution_count=0,
    )

    assert result.semantic == ["stale_belief_acceptance"]


def test_domain_adapter_can_supply_nonshopping_state_completeness():
    result = classify_consequence_axes(
        task={"task_id": "reddit-1", "product_title": "unused", "quantity": 1},
        primary_evidence=None,
        decision_evidence={"task_id": "reddit-1", "body": "current state"},
        verification={"decision": "reject"},
        final_decision="reject",
        final_task_success=False,
        duplicate_execution_count=0,
        primary_complete=False,
        decision_complete=True,
    )

    assert result.semantic == ["incorrect_verification", "authority_state_conflict"]


def test_adapter_trace_can_record_accepted_inconsistent_state_without_fault_name_inference():
    result = classify_consequence_axes(
        task=TASK,
        primary_evidence=COMPLETE,
        decision_evidence=COMPLETE,
        verification={"decision": "accept"},
        final_decision="accept",
        final_task_success=True,
        duplicate_execution_count=0,
        state_inconsistent=True,
    )

    assert result.system == []
    assert result.semantic == ["state_inconsistency"]
