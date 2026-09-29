from mas_faults.webarena_shopping_real import CORE_CONDITIONS, apply_message_fault, resolve_conditions


def test_core_matrix_has_clean_plus_exactly_six_fault_conditions():
    assert CORE_CONDITIONS == (
        "clean",
        "a1_moderate_delay",
        "a1_deadline_delay",
        "a5_omission",
        "a6_inner_evidence_poisoning",
        "a8_truncation",
        "a12_stale_replay",
    )
    assert resolve_conditions(None) == CORE_CONDITIONS


def test_clean_condition_delivers_the_original_message_without_fault():
    message = {"task_id": "shopping-001", "cart_verified": True}
    delivered, symptom, applied = apply_message_fault(message, "clean")
    assert delivered == message
    assert symptom == "none"
    assert applied is False


def test_truncation_preserves_a_message_but_marks_the_a_layer_symptom():
    message = {"task_id": "shopping-001", "cart_verified": True, "evidence": "x" * 200}
    delivered, symptom, applied = apply_message_fault(message, "a8_truncation")
    assert delivered["task_id"] == message["task_id"]
    assert len(delivered["evidence"]) < len(message["evidence"])
    assert symptom == "A8_message_truncation"
    assert applied is True


def test_stale_replay_delivers_the_previous_task_message():
    current = {"task_id": "shopping-002"}
    stale = {"task_id": "shopping-001"}
    delivered, symptom, applied = apply_message_fault(current, "a12_stale_replay", stale)
    assert delivered == stale
    assert symptom == "A12_timing_or_session_mismatch"
    assert applied is True


def test_hard_omission_is_distinct_from_recoverable_omission():
    message = {"task_id": "shopping-001", "cart_verified": True}
    delivered, symptom, applied = apply_message_fault(message, "a5_omission_hard")
    assert delivered is None
    assert symptom == "A5_message_omission"
    assert applied is True


def test_hard_truncation_removes_verification_fields():
    message = {"task_id": "shopping-001", "cart_verified": True, "evidence": "complete evidence"}
    delivered, symptom, applied = apply_message_fault(message, "a8_truncation_hard")
    assert delivered["task_id"] == message["task_id"]
    assert delivered["cart_verified"] is False
    assert delivered["evidence"] == ""
    assert symptom == "A8_message_truncation"
    assert applied is True


def test_semantic_truncation_keeps_valid_syntax_but_drops_required_fields():
    message = {"task_id": "shopping-001", "product_id": "104499", "sku": "B08PCSHBXY", "cart_verified": True, "evidence": "complete"}
    delivered, symptom, applied = apply_message_fault(message, "a8_semantic_truncation")
    assert delivered["task_id"] == message["task_id"]
    assert "product_id" not in delivered
    assert "sku" not in delivered
    assert delivered["cart_verified"] is True
    assert delivered["truncated"] is False
    assert symptom == "A8_message_truncation"
    assert applied is True


def test_poisoned_stale_replay_keeps_task_id_but_changes_semantics():
    current = {"task_id": "shopping-002", "product_title": "Current product"}
    stale = {"task_id": "shopping-001", "product_title": "Old product", "sku": "OLD-SKU", "cart_verified": True}
    delivered, symptom, applied = apply_message_fault(current, "a12_stale_replay_poisoned", stale)
    assert delivered["task_id"] == current["task_id"]
    assert delivered["product_title"] == current["product_title"]
    assert delivered["sku"] == stale["sku"]
    assert delivered["stale_replayed"] is True
    assert symptom == "A12_timing_or_session_mismatch"
    assert applied is True


def test_silent_stale_replay_looks_current_at_task_level():
    current = {"task_id": "shopping-002", "product_title": "Current product"}
    stale = {"task_id": "shopping-001", "product_title": "Old product", "sku": "OLD-SKU", "cart_verified": True}
    delivered, symptom, applied = apply_message_fault(current, "a12_silent_stale_replay", stale)
    assert delivered["task_id"] == current["task_id"]
    assert delivered["product_title"] == current["product_title"]
    assert delivered["sku"] == stale["sku"]
    assert "stale_replayed" not in delivered
    assert symptom == "A12_timing_or_session_mismatch"
    assert applied is True


def test_semantic_faults_preserve_the_outer_message_shape_as_requested():
    current = {
        "task_id": "shopping-002",
        "product_title": "Current product",
        "product_id": "NEW-ID",
        "sku": "NEW-SKU",
        "requested_quantity": 1,
        "observed_quantity": 1,
        "cart_verified": True,
        "evidence": '{"task_id":"shopping-002","product_id":"NEW-ID","observed_quantity":1}',
    }
    previous = {
        "task_id": "shopping-001",
        "product_title": "Old product",
        "product_id": "OLD-ID",
        "sku": "OLD-SKU",
        "evidence": '{"task_id":"shopping-001","product_id":"OLD-ID","observed_quantity":1}',
        "message_id": "old-message",
    }
    swapped = apply_message_fault(current, "a6_product_id_sku_swap", previous)[0]
    assert swapped["task_id"] == current["task_id"]
    assert swapped["product_title"] == current["product_title"]
    assert swapped["cart_verified"] is True
    assert swapped["product_id"] == previous["product_id"]
    assert swapped["sku"] == previous["sku"]
    drifted = apply_message_fault(current, "a19_contract_semantic_drift", previous)[0]
    assert "cart_verified" not in drifted
    assert drifted["verified_cart"] is True
    assert drifted["quantity_seen"] == "1"


if __name__ == "__main__":
    test_clean_condition_delivers_the_original_message_without_fault()
    test_truncation_preserves_a_message_but_marks_the_a_layer_symptom()
    test_stale_replay_delivers_the_previous_task_message()
