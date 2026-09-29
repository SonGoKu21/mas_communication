import copy
import importlib
import json

import pytest


@pytest.fixture
def api():
    return importlib.import_module("mas_faults.mitigation_protocol")


@pytest.fixture
def graph(api):
    return api.EvidenceGraph("task-1", "session-1")


def evidence(graph, evidence_id, entity="cart", version=1, payload=None, **kwargs):
    return graph.add_evidence(
        evidence_id, entity_id=entity, version=version,
        payload={"quantity": 1} if payload is None else payload,
        source="worker", task_id="task-1", session_id="session-1", **kwargs,
    )


def judgment(graph, judgment_id, provided=(), cited=(), parents=(), **kwargs):
    return graph.add_judgment(
        judgment_id, provided_evidence_ids=provided, cited_evidence_ids=cited,
        parent_judgment_ids=parents, task_id="task-1", session_id="session-1",
        status=kwargs.pop("status", "valid"), **kwargs,
    )


@pytest.mark.parametrize("kind,limit", [("get", 4), ("model_call", 3), ("replay", 1)])
def test_budget_exhaustion_is_atomic_and_independent(api, kind, limit):
    budget = api.Budget()
    budget.consume(kind, limit - 1)
    with pytest.raises(api.BudgetExhausted, match="budget exhausted"):
        budget.consume(kind, 2)
    assert budget.snapshot()["used"][kind] == limit - 1
    budget.consume(kind)
    with pytest.raises(api.BudgetExhausted):
        budget.consume(kind)
    assert budget.snapshot()["remaining"][kind] == 0
    assert sum(budget.snapshot()["used"].values()) == limit
    assert budget.events[-1]["event_type"] == "budget_exhausted"
    json.dumps(budget.snapshot(), allow_nan=False)
    json.dumps(budget.events, allow_nan=False)


@pytest.mark.parametrize("count", [-1, True, 1.5, "1"])
def test_budget_rejects_invalid_counts_without_mutating(api, count):
    budget = api.Budget()
    before = budget.snapshot()
    with pytest.raises(ValueError):
        budget.consume("get", count)
    assert budget.snapshot() == before


def test_custom_budget_zero_and_unknown_kind(api):
    budget = api.Budget(max_gets=0, max_model_calls=1, max_replays=0)
    budget.consume("get", 0)
    with pytest.raises(api.BudgetExhausted):
        budget.consume("get")
    with pytest.raises(ValueError):
        budget.consume("typo")
    with pytest.raises(ValueError):
        api.Budget(max_gets=-1)
    with pytest.raises(ValueError):
        api.Budget(max_replays=True)
    snapshot = budget.snapshot()
    snapshot["used"]["model_call"] = 99
    budget.consume("model_call")
    assert budget.snapshot()["used"]["model_call"] == 1


def test_new_version_invalidates_downstream_not_independent_entities(graph):
    evidence(graph, "cart-v1")
    evidence(graph, "catalog-v20", entity="catalog", version=20)
    judgment(graph, "cart-check", ["cart-v1"])
    judgment(graph, "final", parents=["cart-check"])
    judgment(graph, "catalog-check", ["catalog-v20"])
    assert graph.accepted("final")
    evidence(graph, "cart-v2", version=2)
    states = graph.snapshot()["judgments"]
    assert states["cart-check"]["status"] == "invalidated"
    assert states["final"]["status"] == "invalidated"
    assert graph.accepted("catalog-check")
    assert not graph.accepted("final")
    assert graph.snapshot()["known_versions"] == {"cart": 2, "catalog": 20}


def test_all_provided_inputs_and_cited_ids_are_dependencies(graph):
    evidence(graph, "visible", entity="cart")
    evidence(graph, "cited", entity="catalog")
    record = judgment(graph, "check", ["visible", "visible"], ["cited"])
    assert record["evidence_ids"] == ["visible", "cited"]
    evidence(graph, "visible-new", version=2)
    assert not graph.accepted("check")
    judgment(graph, "check-2", ["visible-new"], ["cited"])
    evidence(graph, "cited-new", entity="catalog", version=2)
    assert not graph.accepted("check-2")


@pytest.mark.parametrize("kwargs", [
    {"provided": ["missing"]}, {"cited": ["missing"]}, {"parents": ["missing"]},
])
def test_unknown_dependencies_reject_without_partial_registration(graph, kwargs):
    before = graph.snapshot()
    with pytest.raises(ValueError, match="unknown"):
        judgment(graph, "bad", **kwargs)
    assert graph.snapshot() == before


@pytest.mark.parametrize("method", ["add_evidence", "add_judgment"])
@pytest.mark.parametrize("field,value", [("task_id", "other"), ("session_id", "other")])
def test_scope_mismatch_rejects_before_state_changes(graph, method, field, value):
    kwargs = {"task_id": "task-1", "session_id": "session-1", field: value}
    if method == "add_evidence":
        kwargs.update(entity_id="cart", version=999, payload={}, source="worker")
    else:
        kwargs.update(provided_evidence_ids=[])
    before = graph.snapshot()
    with pytest.raises(ValueError, match="scope"):
        getattr(graph, method)("foreign", **kwargs)
    assert graph.snapshot() == before


def test_late_old_evidence_never_regresses_known_version_or_revives_judgment(graph):
    evidence(graph, "v1")
    judgment(graph, "old", ["v1"])
    evidence(graph, "v3", version=3)
    judgment(graph, "current", ["v3"])
    evidence(graph, "late-v2", version=2)
    assert graph.snapshot()["known_versions"]["cart"] == 3
    assert graph.accepted("current")
    assert not graph.accepted("old")
    assert judgment(graph, "late-check", ["late-v2"])["status"] == "invalidated"
    assert set(graph.context_snapshot()["evidence"]) == {"v3"}


def test_same_version_conflict_is_unresolved_and_propagates(graph):
    evidence(graph, "a")
    judgment(graph, "check", ["a"])
    judgment(graph, "final", parents=["check"])
    evidence(graph, "independent", entity="catalog")
    judgment(graph, "independent-check", ["independent"])
    evidence(graph, "b", payload={"quantity": 2})
    state = graph.snapshot()
    assert state["evidence"]["a"]["status"] == "unresolved"
    assert state["evidence"]["b"]["status"] == "unresolved"
    assert state["judgments"]["check"]["status"] == "unresolved"
    assert state["judgments"]["final"]["status"] == "unresolved"
    assert not graph.accepted("check")
    assert graph.accepted("independent-check")
    assert set(graph.context_snapshot()["evidence"]) == {"independent"}
    evidence(graph, "v2", version=2)
    assert not graph.accepted("check")
    assert judgment(graph, "recomputed", ["v2"])["status"] == "valid"


def test_same_version_agreement_does_not_conflict_across_sources(graph):
    evidence(graph, "a")
    graph.add_evidence("b", entity_id="cart", version=1, payload={"quantity": 1},
                       source="independent-read", task_id="task-1", session_id="session-1")
    judgment(graph, "check", ["a", "b"])
    assert graph.accepted("check")


@pytest.mark.parametrize("version", [None, "opaque-version"])
def test_incomparable_versions_never_overwrite_known_state(graph, version):
    evidence(graph, "known", version=3)
    judgment(graph, "check", ["known"])
    evidence(graph, "incomparable", version=version)
    assert not graph.accepted("check")
    assert graph.snapshot()["judgments"]["check"]["status"] == "unresolved"
    assert graph.snapshot()["known_versions"]["cart"] == 3
    assert graph.context_snapshot()["evidence"] == {}


def test_provisional_unresolved_and_empty_judgments_are_not_accepted(graph):
    evidence(graph, "a")
    judgment(graph, "provisional", ["a"], status="provisional")
    judgment(graph, "unresolved", ["a"], status="unresolved")
    judgment(graph, "empty")
    judgment(graph, "downstream", parents=["provisional"])
    for name in ["provisional", "unresolved", "empty", "downstream"]:
        assert not graph.accepted(name)
    with pytest.raises(ValueError):
        judgment(graph, "bad-status", ["a"], status="anything")


def test_explicit_invalidation_is_transitive_idempotent_and_audited(graph):
    evidence(graph, "a")
    judgment(graph, "first", ["a"])
    judgment(graph, "second", parents=["first"])
    assert graph.invalidate("first", reason="action-needs-verification") == ["first", "second"]
    assert graph.invalidate("first", reason="again") == []
    assert graph.snapshot()["judgments"]["second"]["status"] == "invalidated"
    assert any(event["event_type"] == "judgment_state_changed" for event in graph.events)
    with pytest.raises(ValueError, match="unknown"):
        graph.invalidate("unknown")


def test_ids_cannot_be_rebound_but_identical_evidence_delivery_is_idempotent(graph):
    first = evidence(graph, "a")
    assert evidence(graph, "a") == first
    with pytest.raises(ValueError, match="already"):
        evidence(graph, "a", version=2)
    judgment(graph, "check", ["a"])
    with pytest.raises(ValueError, match="already"):
        judgment(graph, "check", ["a"])
    assert graph.accepted("check")


def test_payloads_states_and_events_are_detached_json_data(graph):
    payload = {"quantity": 1, "details": ["observed"]}
    result = evidence(graph, "a", payload=payload)
    payload["details"].append("mutated")
    result["payload"]["quantity"] = 99
    judgment(graph, "check", ["a"], payload={"decision": "accept"})
    state = graph.snapshot()
    state["evidence"]["a"]["payload"]["quantity"] = 100
    events = graph.events
    events.clear()
    assert graph.snapshot()["evidence"]["a"]["payload"] == {
        "quantity": 1, "details": ["observed"],
    }
    assert graph.events
    assert json.loads(json.dumps(graph.snapshot(), allow_nan=False)) == graph.snapshot()
    json.dumps(graph.context_snapshot(), allow_nan=False)
    json.dumps(graph.events, allow_nan=False)


@pytest.mark.parametrize("payload", [{"bad": float("nan")}, {"bad": object()}, {1: "bad"}])
def test_non_json_payload_rejected_atomically(graph, payload):
    before = graph.snapshot()
    with pytest.raises(ValueError, match="JSON"):
        evidence(graph, "bad", payload=payload)
    assert graph.snapshot() == before


def test_runner_envelopes_bind_action_and_verdict_without_adapters(graph):
    envelope = {"evidence_id": "cart-1", "task_id": "task-1", "session_id": "session-1",
                "entity_id": "cart", "version": 1, "action_id": "action-1",
                "source": "worker", "payload": {"quantity": 2}}
    assert graph.add_evidence(**envelope)["action_id"] == "action-1"
    decision = {"judgment_id": "check", "evidence_ids": ["cart-1"],
                "verdict": {"decision": "accept"}}
    record = graph.add_judgment(**decision, status="valid")
    assert record["task_id"] == "task-1"
    assert record["session_id"] == "session-1"
    assert record["verdict"] == {"decision": "accept"}
    assert record["provided_evidence_ids"] == ["cart-1"]
    assert graph.accepted("check")
    with pytest.raises(ValueError, match="already"):
        graph.add_evidence(**{**envelope, "action_id": "other-action"})
    assert graph.add_judgment(**{**decision, "judgment_id": "draft"})["status"] == "provisional"


def test_envelope_ids_and_explicit_inputs_are_conservatively_unioned(graph):
    evidence(graph, "a", entity="a")
    evidence(graph, "b", entity="b")
    evidence(graph, "c", entity="c")
    record = graph.add_judgment("check", evidence_ids=["a"],
                                provided_evidence_ids=["b"], cited_evidence_ids=["c"],
                                verdict={"decision": "reject"}, status="valid")
    assert record["evidence_ids"] == ["b", "a", "c"]
    assert graph.accepted("check")
    evidence(graph, "a-new", entity="a", version=2)
    assert not graph.accepted("check")


def test_conflicting_verdict_aliases_reject_atomically(graph):
    evidence(graph, "a")
    before = graph.snapshot()
    with pytest.raises(ValueError, match="verdict"):
        graph.add_judgment("check", evidence_ids=["a"], verdict={"decision": "accept"},
                           payload={"decision": "reject"})
    assert graph.snapshot() == before


def test_duplicate_id_cannot_change_json_types_that_compare_equal_in_python(graph):
    evidence(graph, "a", payload={"quantity": 1})
    before = graph.snapshot()
    with pytest.raises(ValueError, match="already"):
        evidence(graph, "a", payload={"quantity": True})
    assert graph.snapshot() == before


def test_verdict_alias_comparison_preserves_json_types(graph):
    evidence(graph, "a")
    with pytest.raises(ValueError, match="verdict"):
        graph.add_judgment("check", evidence_ids=["a"], verdict={"accept": True},
                           payload={"accept": 1})


def shopping_payload():
    state = {"task_id": "task-1", "product_title": "Tea", "product_id": "10",
             "sku": "TEA", "requested_quantity": 2, "observed_quantity": 2,
             "cart_verified": True}
    return {**state, "evidence": json.dumps(state), "http_receipt_indices": [1, 2]}


@pytest.mark.parametrize("change", ["receipts", "formatting", "nested_receipts", "all"])
def test_same_shopping_state_with_new_diagnostics_remains_valid(graph, change):
    first = shopping_payload()
    fresh = copy.deepcopy(first)
    if change in ("receipts", "all"):
        fresh["http_receipt_indices"] = [7, 8]
    nested = json.loads(fresh["evidence"])
    if change in ("nested_receipts", "all"):
        nested["http_receipt_indices"] = [9]
    if change in ("formatting", "nested_receipts", "all"):
        fresh["evidence"] = json.dumps(dict(reversed(list(nested.items()))), indent=4)
    evidence(graph, "first", payload=first)
    judgment(graph, "check", ["first"])
    judgment(graph, "downstream", parents=["check"])
    result = evidence(graph, "fresh", payload=fresh)
    assert result["status"] == "valid"
    assert graph.accepted("check")
    assert graph.accepted("downstream")
    assert set(graph.context_snapshot()["evidence"]) == {"first", "fresh"}
    assert graph.snapshot()["evidence"]["fresh"]["payload"] == fresh
    assert any(event.get("evidence", {}).get("payload") == fresh
               for event in graph.events if event["event_type"] == "evidence_registered")


@pytest.mark.parametrize("location", ["outer", "nested"])
@pytest.mark.parametrize("field,value", [
    ("task_id", "other"), ("product_title", "Coffee"), ("product_id", "11"),
    ("sku", "COFFEE"), ("requested_quantity", 3), ("observed_quantity", 3),
    ("cart_verified", False), ("unknown_semantic_field", "different"),
    ("diagnostics", {"http_receipt_indices": [99]}),
])
def test_shopping_projection_keeps_state_and_unknown_content_conflicts(graph, location, field, value):
    first = shopping_payload()
    conflicting = copy.deepcopy(first)
    if location == "outer":
        conflicting[field] = value
    else:
        nested = json.loads(conflicting["evidence"])
        nested[field] = value
        conflicting["evidence"] = json.dumps(nested, indent=2)
    evidence(graph, "first", payload=first)
    judgment(graph, "check", ["first"])
    assert evidence(graph, "conflicting", payload=conflicting)["status"] == "unresolved"
    assert not graph.accepted("check")


@pytest.mark.parametrize("payload", [
    {"content": "a", "http_receipt_indices": [1]},
    {"evidence": '{"content": "a"}'},
    {"nested": {"http_receipt_indices": [1]}},
])
def test_generic_payloads_keep_all_content_semantics(graph, payload):
    evidence(graph, "first", payload=payload)
    changed = copy.deepcopy(payload)
    if "http_receipt_indices" in changed:
        changed["http_receipt_indices"] = [2]
    elif "evidence" in changed:
        changed["evidence"] = '{"content": "b"}'
    else:
        changed["nested"]["http_receipt_indices"] = [2]
    assert evidence(graph, "changed", payload=changed)["status"] == "unresolved"


@pytest.mark.parametrize("nested", [
    '{"observed_quantity": 999, "observed_quantity": 2}',
    '{"observed_quantity": NaN}', "{broken", "observed two units",
])
def test_ambiguous_or_non_json_evidence_is_not_discarded(graph, nested):
    first = shopping_payload()
    changed = {**first, "evidence": nested}
    evidence(graph, "first", payload=first)
    assert evidence(graph, "changed", payload=changed)["status"] == "unresolved"


def test_receipt_normalization_does_not_allow_rebinding_an_evidence_id(graph):
    first = shopping_payload()
    evidence(graph, "same-id", payload=first)
    with pytest.raises(ValueError, match="already"):
        evidence(graph, "same-id", payload={**first, "http_receipt_indices": [99]})
