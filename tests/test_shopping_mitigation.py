import copy
import importlib
import inspect
import json
from types import SimpleNamespace

import pytest

from mas_faults.webarena_shopping_real import ShoppingHTTPExecutor


TASK = {"task_id": "task-1", "product_title": "Tea", "quantity": 2,
        "product_url": "http://shopping.invalid/tea.html"}


def evidence():
    state = {"task_id": "task-1", "product_title": "Tea", "product_id": "10",
             "sku": "TEA", "requested_quantity": 2, "observed_quantity": 2,
             "cart_verified": True}
    return {**state, "evidence": json.dumps(state)}


def api():
    return importlib.import_module("mas_faults.shopping_mitigation")


def test_valid_evidence_passes_without_readback():
    policy = api().EvidencePolicy("guarded_recheck", TASK)
    result = policy.receive(evidence(), lambda: pytest.fail("unexpected read"), receiver="Verifier")
    assert result == evidence()
    assert policy.readback_count == 0
    assert not policy.events[-1]["triggered"]


@pytest.mark.parametrize("field,value", [
    ("task_id", "task-previous"), ("observed_quantity", 3),
    ("observed_quantity", "2"), ("observed_quantity", True),
    ("cart_verified", "true"), ("product_id", ""), ("evidence", ""),
])
def test_guard_rechecks_semantically_invalid_messages(field, value):
    damaged = evidence()
    damaged[field] = value
    policy = api().EvidencePolicy("guarded_recheck", TASK)
    assert policy.receive(damaged, evidence, receiver="Verifier") == evidence()
    assert policy.readback_count == 1
    assert policy.events[-1]["triggered"]
    assert policy.events[-1]["before_issues"]


@pytest.mark.parametrize("damaged", [None, "{broken", {"task_id": "task-1", "cart_verified": True}])
def test_guard_handles_non_delivery_malformed_and_partial(damaged):
    policy = api().EvidencePolicy("guarded_recheck", TASK)
    assert policy.receive(damaged, evidence, receiver="Verifier") == evidence()
    assert policy.readback_count == 1


def test_inner_poisoning_is_detected_without_access_to_clean_original():
    damaged = evidence()
    nested = json.loads(damaged["evidence"])
    nested["task_id"] = "previous-task"
    damaged["evidence"] = json.dumps(nested)
    check = api().check_evidence(TASK, damaged)
    assert "inner_outer:task_id" in check.issues
    parameters = inspect.signature(api().EvidencePolicy.receive).parameters
    assert not {"fault", "original", "expected_answer", "ground_truth"}.intersection(parameters)


def test_consistent_identity_corruption_is_not_claimed_detectable_without_observation():
    changed = evidence()
    changed["product_id"] = "999"
    changed["sku"] = "OTHER"
    nested = json.loads(changed["evidence"])
    nested.update(product_id="999", sku="OTHER")
    changed["evidence"] = json.dumps(nested)
    assert api().check_evidence(TASK, changed).valid


def test_plain_text_evidence_is_not_rejected_by_an_invented_nested_json_contract():
    current = evidence()
    current["evidence"] = "Observed two units of Tea in the current cart."
    assert api().check_evidence(TASK, current).valid


def test_baseline_never_rechecks_or_rewrites_received_input():
    damaged = {"task_id": "previous"}
    policy = api().EvidencePolicy("baseline", TASK)
    assert policy.receive(damaged, lambda: pytest.fail("baseline read"), receiver="Verifier") == damaged
    assert not policy.events[-1]["triggered"]


def test_always_recheck_uses_the_same_one_read_budget_across_receivers():
    calls = []
    def readback():
        calls.append(1)
        return evidence()
    policy = api().EvidencePolicy("always_recheck", TASK)
    assert policy.receive(evidence(), readback, receiver="Verifier") == evidence()
    assert policy.receive(evidence(), readback, receiver="Coordinator") == evidence()
    assert len(calls) == policy.readback_count == 1
    assert [e["readback_called"] for e in policy.events] == [True, False]


def test_failed_readback_is_bounded_and_fails_closed_for_invalid_input():
    def fail():
        raise TimeoutError("do not record secret-bearing response text")
    policy = api().EvidencePolicy("guarded_recheck", TASK)
    assert policy.receive(None, fail, receiver="Verifier") is None
    assert policy.receive(None, fail, receiver="Coordinator") is None
    assert policy.readback_count == 1
    assert policy.events[0]["readback_error"] == "TimeoutError"


def test_valid_original_is_retained_when_unconditional_readback_fails():
    def fail():
        raise TimeoutError()
    policy = api().EvidencePolicy("always_recheck", TASK)
    assert policy.receive(evidence(), fail, receiver="Verifier") == evidence()


def test_invalid_readback_does_not_become_a_recovery():
    policy = api().EvidencePolicy("guarded_recheck", TASK)
    assert policy.receive(None, lambda: {"cart_verified": True}, receiver="Verifier") is None
    assert not policy.events[-1]["replacement_used"]


def test_successfully_observed_negative_state_replaces_old_positive_claim():
    negative = evidence()
    negative.update(observed_quantity=1, cart_verified=False)
    inner = json.loads(negative["evidence"])
    inner.update(observed_quantity=1, cart_verified=False)
    negative["evidence"] = json.dumps(inner)
    policy = api().EvidencePolicy("always_recheck", TASK)
    assert policy.receive(evidence(), lambda: negative, receiver="Verifier") == negative
    assert policy.events[-1]["after_issues"]


def test_successfully_returned_but_invalid_readback_does_not_preserve_old_success():
    policy = api().EvidencePolicy("always_recheck", TASK)
    assert policy.receive(evidence(), lambda: {"cart_verified": False}, receiver="Verifier") is None


def test_readback_gets_identity_from_live_catalog_and_cart_without_posting():
    requests = []
    class Response:
        text = '<input name="product" value="10"><button data-product-sku="TEA">'
        def raise_for_status(self):
            pass
        def json(self):
            return [{"sku": "TEA", "name": "Tea", "qty": 2}]
    class Session:
        def get(self, url, **kwargs):
            requests.append(url)
            return Response()
        def post(self, *args, **kwargs):
            pytest.fail("readback must not execute a write")
    executor = ShoppingHTTPExecutor("http://shopping.invalid")
    executor.session = Session()
    executor.guest_cart_id = "current-cart"
    result = executor.reobserve_cart(TASK)
    assert result["product_id"] == "10"
    assert result["sku"] == "TEA"
    assert result["observed_quantity"] == 2
    assert result["cart_verified"] is True
    assert json.loads(result["evidence"])["sku"] == "TEA"
    assert len(requests) == executor.readback_http_request_count == 2
    assert any("current-cart/items" in url for url in requests)


def test_reobserve_without_an_executed_cart_never_fabricates_success():
    executor = ShoppingHTTPExecutor("http://shopping.invalid")
    result = executor.reobserve_cart(TASK)
    assert result["cart_verified"] is False
    assert executor.readback_http_request_count == 0


def test_valid_partial_operator_keeps_json_valid_but_drops_content():
    from mas_faults.llm.communication_interceptor import CommunicationInterceptor, MessageEnvelope
    from mas_faults.llm.fault_scenarios import FaultScenario
    original = evidence()
    delivery = CommunicationInterceptor(FaultScenario(fault="valid_partial")).transmit(
        MessageEnvelope("Worker", "Verifier", original))
    partial = delivery.delivered[0].payload
    assert isinstance(partial, dict)
    assert partial["task_id"] == "task-1"
    assert partial["cart_verified"] is True
    assert "sku" not in partial and "evidence" not in partial
    assert original == evidence()
    assert delivery.a_layer_symptom == "A8"


@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
@pytest.mark.parametrize("fault", ["omission", "malformed_json"])
def test_real_autogen_runner_routes_faulted_step4_through_policy(monkeypatch, topology, fault):
    import asyncio
    import run_webarena_architecture_rq2 as runner
    class Executor:
        readback_http_request_count = 0
        def __init__(self, base_url):
            pass
        def add_to_cart(self, action):
            return evidence()
        def reobserve_cart(self, task):
            return evidence()
    class Client:
        call_count = prompt_tokens = completion_tokens = 0
        model_info = SimpleNamespace(model="unit-test-only", provider="unit-test")
        def complete(self, prompt):
            self.call_count += 1
            if "Convert the delivered" in prompt:
                return json.dumps(evidence())
            if "action MUST" in prompt:
                return json.dumps({"action": "add_to_cart"})
            return json.dumps({"decision": "accept", "task_id": "task-1", "reason": "fixture"})
    monkeypatch.setattr(runner, "ShoppingHTTPExecutor", Executor)
    policy = api().EvidencePolicy("guarded_recheck", TASK)
    row = asyncio.run(runner.run_one(Client(), TASK, topology, fault, 4, 1,
                                    "http://shopping.invalid", receiver_policy=policy))
    assert row["final_task_success"]
    assert row["observed_A_symptom"] == (["A5"] if fault == "omission" else ["A7"])
    assert row["decision_evidence"] == evidence()
    if fault == "omission":
        assert row["mitigation_events"][0]["before"] is None
    else:
        assert isinstance(row["mitigation_events"][0]["before"], str)
    assert row["mitigation_events"][0]["replacement_used"]
    assert sum(e["fault_applied"] for e in row["events"]) == 1
    assert policy.readback_count == 1
