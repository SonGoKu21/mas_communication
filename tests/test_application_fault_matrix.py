from mas_faults.llm.application_fault_matrix import APPLICATION_FAULTS, fault_names_for_steps
from mas_faults.llm.communication_interceptor import CommunicationInterceptor, MessageEnvelope
from mas_faults.llm.fault_scenarios import FaultScenario


def test_full_application_taxonomy_has_a1_through_a15_with_explicit_steps():
    assert tuple(spec.code for spec in APPLICATION_FAULTS) == tuple(f"A{index}" for index in range(1, 16))
    assert set(fault_names_for_steps((2, 3, 4))) == {spec.name for spec in APPLICATION_FAULTS}


def test_semantic_corruption_keeps_json_shape_but_changes_quantity():
    payload = {"task_id": "shopping-001", "quantity": 1, "cart_verified": True}
    result = CommunicationInterceptor(FaultScenario(fault="message_corruption")).transmit(
        MessageEnvelope("Worker", "Shopping", payload)
    )

    assert result.a_layer_symptom == "A6"
    assert result.delivered[0].payload["task_id"] == "shopping-001"
    assert result.delivered[0].payload["quantity"] == 2


def test_schema_mismatch_renames_contract_fields_without_malforming_json():
    payload = {"task_id": "shopping-001", "observed_quantity": 1, "cart_verified": True}
    result = CommunicationInterceptor(FaultScenario(fault="schema_mismatch")).transmit(
        MessageEnvelope("Worker", "Verifier", payload)
    )

    assert result.a_layer_symptom == "A11"
    assert result.delivered[0].payload == {
        "task_id": "shopping-001",
        "quantity_seen": 1,
        "verified_cart": True,
    }


def test_contract_violation_keeps_fields_but_changes_required_scalar_types():
    payload = {"task_id": "shopping-001", "observed_quantity": 1, "cart_verified": True}
    result = CommunicationInterceptor(FaultScenario(fault="contract_violation")).transmit(
        MessageEnvelope("Worker", "Verifier", payload)
    )

    assert result.a_layer_symptom == "A15"
    assert result.delivered[0].payload["observed_quantity"] == "1"
    assert result.delivered[0].payload["cart_verified"] == "true"


def test_reordering_replays_prior_same_session_message_and_marks_a10():
    interceptor = CommunicationInterceptor(FaultScenario(fault="none"))
    interceptor.transmit(MessageEnvelope("Worker", "Verifier", {"state_version": 1}))
    interceptor.scenario = FaultScenario(fault="reordering")

    result = interceptor.transmit(MessageEnvelope("Worker", "Verifier", {"state_version": 2}))

    assert result.a_layer_symptom == "A10"
    assert result.delivered[0].payload == {"state_version": 1}
