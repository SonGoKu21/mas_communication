import copy
import json
import pytest

from mas_faults.multimechanism_faults import SingleBoundaryFault


def evidence(version=2, task="t1", sku="SKU"):
    payload = {"task_id": task, "product_title": "Product", "product_id": "12", "sku": sku,
               "requested_quantity": version, "observed_quantity": version, "cart_verified": True}
    return {"evidence_id": f"e{version}-{task}", "task_id": task, "session_id": f"s-{task}",
            "entity_id": "cart", "version": version, "action_id": f"a{version}",
            "source": "worker", "payload": {**payload, "evidence": json.dumps(payload)}}


def test_only_registered_boundary_is_changed_and_only_once():
    fault = SingleBoundaryFault("request_non_delivery")
    message = {"action_id": "a", "quantity": 1}
    assert fault.deliver("action_ack", message) == [message]
    assert fault.deliver("action_request", message) == []
    assert fault.deliver("action_request", message) == [message]
    assert len(fault.events) == 1


@pytest.mark.parametrize("condition,boundary", [("request_non_delivery", "action_request"),
                                               ("acknowledgement_loss", "action_ack")])
def test_non_delivery_is_at_explicit_boundary(condition, boundary):
    fault = SingleBoundaryFault(condition)
    assert fault.deliver(boundary, {"value": 1}) == []


def test_duplicate_retains_same_action_id_without_aliases():
    original = {"action_id": "a", "parameters": {"quantity": 2}}
    result = SingleBoundaryFault("duplicate_action_delivery").deliver("action_request", original)
    assert result == [original, original]
    result[0]["parameters"]["quantity"] = 4
    assert result[1] == original


def test_valid_partial_remains_json_but_missing_real_evidence():
    original = evidence()
    fault = SingleBoundaryFault("valid_partial")
    delivered = fault.deliver("evidence_handoff", original)[0]
    assert set(delivered["payload"]) == {"task_id", "product_title", "requested_quantity", "cart_verified"}
    assert "sku" in original["payload"]
    json.dumps(delivered)


def test_same_session_reordering_requires_real_older_source():
    new, old = evidence(2), evidence(1)
    fault = SingleBoundaryFault("same_session_reordering", old_evidence=old)
    assert fault.deliver("observation_handoff", new) == [new, old]
    assert fault.events[0]["source_evidence_id"] == old["evidence_id"]
    bad = SingleBoundaryFault("same_session_reordering", old_evidence=evidence(3))
    with pytest.raises(ValueError):
        bad.deliver("observation_handoff", new)


def test_cross_task_replay_preserves_source_binding():
    old = evidence(1, "other")
    fault = SingleBoundaryFault("cross_task_replay", cross_task_evidence=old)
    assert fault.deliver("evidence_handoff", evidence()) == [old]


def test_stale_judgment_uses_observed_old_judgment_not_generated_acceptance():
    old = {"judgment_id": "j1", "evidence_ids": ["e1"], "verdict": {"decision": "reject"}}
    new = {"judgment_id": "j2", "evidence_ids": ["e2"], "verdict": {"decision": "accept"}}
    fault = SingleBoundaryFault("stale_judgment_replay", old_judgment=old)
    assert fault.deliver("judgment_handoff", new) == [old]
    assert fault.events[0]["source_judgment_id"] == "j1"


def test_conflicting_observation_does_not_supply_pristine_original():
    original = evidence()
    delivered = SingleBoundaryFault("conflicting_observation").deliver("observation_handoff", original)
    assert len(delivered) == 1
    assert delivered[0] != original
    assert delivered[0]["source"] == original["source"]
    inner = json.loads(delivered[0]["payload"]["evidence"])
    assert inner["observed_quantity"] == delivered[0]["payload"]["observed_quantity"]


def test_coherent_corruption_uses_alternate_real_identity_not_schema_error():
    other = evidence(1, "other", "OTHER-SKU")
    other["payload"]["product_id"] = "33"
    other["payload"]["product_title"] = "Different product"
    current = evidence()
    fault = SingleBoundaryFault("contract_consistent_identity_corruption", cross_task_evidence=other)
    delivered = fault.deliver("evidence_handoff", current)[0]
    assert delivered["payload"]["sku"] == "OTHER-SKU"
    assert delivered["payload"]["task_id"] == "t1"
    assert delivered["payload"]["observed_quantity"] == 2
    inner = json.loads(delivered["payload"]["evidence"])
    assert inner == {k: v for k, v in delivered["payload"].items() if k != "evidence"}


@pytest.mark.parametrize("condition,boundary", [("cross_task_replay", "evidence_handoff"),
    ("contract_consistent_identity_corruption", "evidence_handoff"),
    ("same_session_reordering", "observation_handoff"),
    ("stale_judgment_replay", "judgment_handoff")])
def test_missing_real_sources_fail_closed(condition, boundary):
    fault = SingleBoundaryFault(condition)
    with pytest.raises(ValueError):
        fault.deliver(boundary, evidence())
    assert not fault.events


def test_clean_changes_nothing_and_records_no_injection():
    source = evidence()
    fault = SingleBoundaryFault("clean")
    result = fault.deliver("evidence_handoff", source)
    assert result == [source]
    result[0]["payload"].clear()
    assert source["payload"]
    assert fault.events == []
