"""Unit-only protocol tests: real SQLite, fake HTTP transport, no real network.

These tests are not the real-environment Shopping gate.
"""

import importlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import requests

from mas_faults.webarena_shopping_real import ShoppingHTTPExecutor


@pytest.fixture
def protocol():
    name = "mas_faults.shopping_action_protocol"
    assert importlib.util.find_spec(name) is not None, "T2 protocol is not implemented"
    return importlib.import_module(name)


PARAMS = {"operation": "add_quantity", "sku": "TEA", "quantity": 1}
RECEIPT = {"task_id": "task-1", "cart_verified": True, "observed_quantity": 3,
           "http_receipt_indices": [0, 1, 2]}


def invoke(ledger, execute, **kwargs):
    return ledger.execute_once("task-1", "session-1", "action-1", PARAMS, execute, **kwargs)


def test_unit_pending_binding_and_confirmed_receipt_survive_reopen(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    pending = ledger.register("task-1", "session-1", "action-1", PARAMS)
    assert pending["state"] == "pending"
    assert pending["attempts"] == 0
    assert pending["params"] == PARAMS
    assert len(pending["params_sha256"]) == 64
    calls = []

    def execute():
        assert protocol.ActionLedger(path).get("action-1")["state"] == "executing"
        calls.append("write")
        return RECEIPT.copy()

    first = invoke(ledger, execute)
    assert first["state"] == "confirmed"
    assert first["receipt"] == RECEIPT
    assert first["replayed"] is False
    first["receipt"]["observed_quantity"] = 999
    replay = invoke(protocol.ActionLedger(path), execute)
    assert replay["replayed"] is True
    assert replay["receipt"] == RECEIPT
    assert calls == ["write"]
    events = ledger.events("action-1")
    assert any(e["event"] == "confirmed_receipt_replayed" for e in events)
    assert ledger.get("absent") is None


@pytest.mark.parametrize("changes", [
    {"task_id": "task-2"}, {"session_id": "session-2"},
    {"params": {**PARAMS, "quantity": 2}},
])
def test_unit_action_id_cannot_be_rebound(protocol, tmp_path, changes):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    invoke(ledger, lambda: RECEIPT)
    binding = dict(task_id="task-1", session_id="session-1", action_id="action-1", params=PARAMS)
    binding.update(changes)
    with pytest.raises(protocol.ActionBindingError):
        ledger.execute_once(**binding, execute=lambda: pytest.fail("rebound action executed"))
    assert ledger.get("action-1")["receipt"] == RECEIPT


def test_unit_canonical_params_ignore_dictionary_key_order(protocol, tmp_path):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    invoke(ledger, lambda: RECEIPT)
    result = ledger.execute_once("task-1", "session-1", "action-1",
                                 dict(reversed(list(PARAMS.items()))),
                                 lambda: pytest.fail("duplicate executed"))
    assert result["replayed"] is True


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt])
def test_unit_execute_exception_is_unknown_and_never_retried(protocol, tmp_path, failure):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    writes = []

    def execute():
        writes.append("external write")
        raise failure("private transport text")

    with pytest.raises(failure):
        invoke(ledger, execute)
    assert ledger.get("action-1")["state"] == "unknown"
    with pytest.raises(protocol.ActionUnresolved, match="unknown"):
        invoke(protocol.ActionLedger(path), execute)
    assert writes == ["external write"]
    assert "private transport text" not in json.dumps(ledger.events("action-1"))


def test_unit_database_confirmation_failure_does_not_repeat_external_write(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    ledger.register("task-1", "session-1", "action-1", PARAMS)
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TRIGGER fail_confirm BEFORE UPDATE ON actions
                      WHEN NEW.state = 'confirmed'
                      BEGIN SELECT RAISE(ABORT, 'confirmation interrupted'); END""")
    writes = []

    def execute():
        writes.append(1)
        return RECEIPT

    with pytest.raises(sqlite3.DatabaseError):
        invoke(ledger, execute)
    assert protocol.ActionLedger(path).get("action-1")["state"] == "unknown"
    with pytest.raises(protocol.ActionUnresolved):
        invoke(ledger, execute)
    assert writes == [1]


def test_unit_preexecution_failure_can_retry_once_before_any_execute(protocol, tmp_path):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    order = []

    def preexecute():
        order.append("prepare")
        if len(order) == 1:
            raise TimeoutError("request not delivered to execute")

    result = invoke(ledger, lambda: order.append("execute") or RECEIPT,
                    preexecute=preexecute)
    assert order == ["prepare", "prepare", "execute"]
    assert result["state"] == "confirmed"
    assert result["attempts"] == 2
    assert result["execution_count"] == 1
    assert any(e["event"] == "preexecution_failed_unexecuted"
               for e in ledger.events("action-1"))


def test_unit_preexecution_retry_budget_is_persistent(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    calls = []

    def preexecute():
        calls.append(1)
        raise RuntimeError("not sent")

    with pytest.raises(protocol.PreexecutionFailed):
        invoke(protocol.ActionLedger(path), lambda: pytest.fail("must not execute"),
               preexecute=preexecute)
    state = protocol.ActionLedger(path).get("action-1")
    assert state["state"] == "pending"
    assert state["execution_count"] == 0
    assert state["attempts"] == 2
    with pytest.raises(protocol.PreexecutionFailed):
        invoke(protocol.ActionLedger(path), lambda: pytest.fail("must not execute"),
               preexecute=preexecute)
    assert calls == [1, 1]


def test_unit_execute_cannot_signal_safe_retry_after_entry(protocol, tmp_path):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")

    def execute():
        raise protocol.PreexecutionFailed("untrusted claim after execute entry")

    with pytest.raises(protocol.PreexecutionFailed):
        invoke(ledger, execute)
    assert ledger.get("action-1")["state"] == "unknown"
    with pytest.raises(protocol.ActionUnresolved):
        invoke(ledger, execute)


def test_unit_concurrent_instances_claim_only_one_execution(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    entered, release = Event(), Event()
    writes = []

    def execute():
        writes.append(1)
        entered.set()
        assert release.wait(5)
        return RECEIPT

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(invoke, ledger, execute)
        try:
            assert entered.wait(5)
            with pytest.raises(protocol.ActionUnresolved, match="executing"):
                invoke(protocol.ActionLedger(path), execute)
        finally:
            release.set()
        assert owner.result(timeout=5)["state"] == "confirmed"
    assert writes == [1]
    assert ledger.get("action-1")["execution_count"] == 1


def test_unit_abandoned_executing_record_is_never_reclaimed(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    ledger.register("task-1", "session-1", "action-1", PARAMS)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE actions SET state = 'executing' WHERE action_id = 'action-1'")
    with pytest.raises(protocol.ActionUnresolved, match="executing"):
        invoke(protocol.ActionLedger(path), lambda: pytest.fail("abandoned claim rerun"))


@pytest.mark.parametrize("receipt", [None, {}, {"cart_verified": False},
                                      {**RECEIPT, "value": object()}])
def test_unit_malformed_or_unpersistable_receipt_stays_unknown(protocol, tmp_path, receipt):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    with pytest.raises((ValueError, TypeError)):
        invoke(ledger, lambda: receipt)
    assert ledger.get("action-1")["state"] == "unknown"


@pytest.mark.parametrize("params", [{"qty": float("nan")}, {1: "ambiguous key"},
                                    {"qty": object()}])
def test_unit_invalid_params_rejected_before_registration(protocol, tmp_path, params):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    with pytest.raises((ValueError, TypeError)):
        ledger.execute_once("task-1", "session-1", "action-1", params,
                            lambda: pytest.fail("invalid params executed"))
    assert ledger.get("action-1") is None


BASE = "http://shopping.invalid"
TOKEN = "private-guest-cart-token"
TASK = {"task_id": "task-1", "product_title": "Tea", "product_url": "/tea.html", "quantity": 2}
HTML = '<input name="product" value="10"><button data-product-sku="TEA">'
ITEM = {"item_id": 7, "sku": "TEA", "name": "Tea", "qty": 2, "quote_id": TOKEN,
        "extension_attributes": {"cookie": "private-cookie"}}


def response(payload=None, *, body=None, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.encoding = "utf-8"
    result._content = body if body is not None else json.dumps(payload).encode()
    result.headers.update(headers or {})
    return result


@pytest.fixture
def transport(monkeypatch):
    calls, replies = [], []

    def send(session, method, url, **kwargs):
        calls.append((method.upper(), url, kwargs))
        assert replies, "unexpected HTTP request in unit-only transport"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        reply.url = url
        return reply

    monkeypatch.setattr(requests.Session, "request", send)
    return calls, replies


def initialized(protocol):
    executor = protocol.MultiStateShoppingExecutor(BASE)
    executor.guest_cart_id = TOKEN
    return executor


def queue_update(replies, before=2, after=1, *, write_qty=None):
    replies.extend([response(body=HTML.encode()), response([{**ITEM, "qty": before}]),
                    response({**ITEM, "qty": after if write_qty is None else write_qty}),
                    response([{**ITEM, "qty": after}])])


def test_unit_initial_cart_creation_uses_inherited_executor(protocol, transport):
    calls, replies = transport
    replies.extend([response(body=HTML.encode()), response(TOKEN), response(ITEM)])
    executor = protocol.MultiStateShoppingExecutor(BASE)
    assert isinstance(executor, ShoppingHTTPExecutor)
    assert executor.add_to_cart.__func__ is ShoppingHTTPExecutor.add_to_cart
    receipt = executor.add_to_cart(TASK)
    assert receipt["observed_quantity"] == 2
    assert receipt["cart_verified"] is True
    assert executor.guest_cart_id == TOKEN
    assert [call[0] for call in calls] == ["GET", "POST", "POST"]
    assert calls[1][1] == BASE + "/rest/V1/guest-carts"
    assert calls[2][2]["json"] == {"cartItem": {"sku": "TEA", "qty": 2}}


def test_unit_set_quantity_puts_discovered_item_id_and_reads_fresh_cart(protocol, transport):
    calls, replies = transport
    queue_update(replies)
    executor = initialized(protocol)
    receipt = executor.set_quantity(TASK, 1)
    assert [call[0] for call in calls] == ["GET", "GET", "PUT", "GET"]
    assert calls[2][1] == BASE + "/rest/V1/guest-carts/" + TOKEN + "/items/7"
    assert calls[2][2]["json"] == {"cartItem": {"item_id": 7, "sku": "TEA", "qty": 1}}
    assert receipt["requested_quantity"] == 1
    assert receipt["observed_quantity"] == 1
    assert receipt["cart_verified"] is True
    assert json.loads(receipt["evidence"])["observed_quantity"] == 1
    assert TASK["quantity"] == 2
    assert executor.guest_cart_id == TOKEN
    assert executor.readback_http_request_count == 3
    assert receipt["http_receipt_indices"] == [0, 1, 2, 3]
    assert not replies


def test_unit_add_quantity_is_same_cart_post_and_unguarded_duplicates_add_twice(protocol, transport):
    calls, replies = transport
    queue_update(replies, before=2, after=3)
    queue_update(replies, before=3, after=4)
    executor = initialized(protocol)
    assert executor.add_quantity(TASK, 1)["observed_quantity"] == 3
    second = executor.add_quantity(TASK, 1)
    assert second["observed_quantity"] == 4
    assert second["requested_quantity"] == 4
    writes = [call for call in calls if call[0] != "GET"]
    assert len(writes) == 2
    for method, url, kwargs in writes:
        assert method == "POST"
        assert url == BASE + "/rest/V1/guest-carts/" + TOKEN + "/items"
        assert kwargs["json"] == {"cartItem": {"sku": "TEA", "qty": 1}}
    assert executor.guest_cart_id == TOKEN


def test_unit_guarded_duplicate_add_returns_historical_receipt_without_http(protocol, transport, tmp_path):
    calls, replies = transport
    queue_update(replies, before=2, after=3)
    executor = initialized(protocol)
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    first = invoke(ledger, lambda: executor.add_quantity(TASK, 1))
    duplicate = invoke(ledger, lambda: executor.add_quantity(TASK, 1))
    assert duplicate["receipt"] == first["receipt"]
    assert duplicate["replayed"] is True
    assert duplicate["receipt"]["observed_quantity"] == 3
    assert len(calls) == 4


@pytest.mark.parametrize("observed", [2, 1.5, "1", True, None])
def test_unit_readback_does_not_report_write_ack_as_final_quantity(protocol, transport, observed):
    _, replies = transport
    queue_update(replies, after=observed, write_qty=1)
    result = initialized(protocol).set_quantity(TASK, 1)
    assert result["observed_quantity"] == observed
    assert type(result["observed_quantity"]) is type(observed)
    assert result["cart_verified"] is False
    assert json.loads(result["evidence"])["observed_quantity"] == observed


@pytest.mark.parametrize("method", ["set_quantity", "add_quantity"])
@pytest.mark.parametrize("quantity", [0, -1, True, 1.5, "2", None])
def test_unit_invalid_requested_quantity_never_sends_http(protocol, transport, method, quantity):
    calls, _ = transport
    with pytest.raises(ValueError, match="quantity"):
        getattr(initialized(protocol), method)(TASK, quantity)
    assert calls == []


@pytest.mark.parametrize("method", ["set_quantity", "add_quantity"])
def test_unit_state_update_without_cart_does_not_initialize_one(protocol, transport, method):
    calls, _ = transport
    executor = protocol.MultiStateShoppingExecutor(BASE)
    with pytest.raises(ValueError, match="cart"):
        getattr(executor, method)(TASK, 1)
    assert calls == []
    assert executor.guest_cart_id is None


@pytest.mark.parametrize("method", ["set_quantity", "add_quantity", "reobserve_cart"])
@pytest.mark.parametrize("url", ["http://other.invalid/tea.html", "//other.invalid/tea.html",
                                 "https://shopping.invalid/tea.html", "http://shopping.invalid:81/tea.html",
                                 "http://user:private-password@shopping.invalid/tea.html"])
def test_unit_adapter_rejects_foreign_origin_and_credentials_before_http(protocol, transport, method, url):
    calls, _ = transport
    executor = initialized(protocol)
    args = ({**TASK, "product_url": url},) + (() if method == "reobserve_cart" else (1,))
    with pytest.raises(ValueError, match="origin|credentials"):
        getattr(executor, method)(*args)
    assert calls == []
    assert executor.http_receipts == []


@pytest.mark.parametrize("items", [[], [{**ITEM, "sku": "OTHER"}], [ITEM, ITEM],
                                   [{**ITEM, "item_id": "//other.invalid"}],
                                   [{**ITEM, "item_id": True}],
                                   [{**ITEM, "name": "Different product"}], {"items": [ITEM]}])
def test_unit_missing_ambiguous_or_invalid_cart_item_prevents_write(protocol, transport, items):
    calls, replies = transport
    replies.extend([response(body=HTML.encode()), response(items)])
    with pytest.raises(ValueError):
        initialized(protocol).set_quantity(TASK, 1)
    assert all(call[0] == "GET" for call in calls)


@pytest.mark.parametrize("stage", ["identity", "lookup", "write", "readback"])
def test_unit_redirects_never_follow_or_trigger_another_write(protocol, transport, stage):
    calls, replies = transport
    queue_update(replies)
    index = ["identity", "lookup", "write", "readback"].index(stage)
    replies[index] = response(body=b"redirect", status=307,
                              headers={"Location": "http://other.invalid/?token=private-token"})
    executor = initialized(protocol)
    with pytest.raises(RuntimeError, match="redirect"):
        executor.set_quantity(TASK, 1)
    assert len(calls) == index + 1
    assert all(call[2]["allow_redirects"] is False for call in calls)
    assert executor.http_receipts[-1]["status_code"] == 307
    assert "private-token" not in json.dumps(executor.http_receipts)


def test_unit_new_receipts_preserve_redaction_hashes_and_actual_fields(protocol, transport):
    calls, replies = transport
    queue_update(replies)
    executor = initialized(protocol)
    executor.session.headers["Authorization"] = "Bearer private-authorization"
    result = executor.set_quantity({**TASK, "product_url": "/tea.html?token=private-query"}, 1)
    receipts = executor.http_receipts
    assert [r["receipt_index"] for r in receipts] == [0, 1, 2, 3]
    assert [r["request_method"] for r in receipts] == ["GET", "GET", "PUT", "GET"]
    assert all(len(r["response_sha256"]) == 64 for r in receipts)
    assert receipts[2]["request_url"] == BASE + "/rest/V1/guest-carts/[REDACTED]/items/7"
    assert receipts[-1]["response_payload"] == [{"item_id": 7, "sku": "TEA", "name": "Tea", "qty": 1}]
    serialized = json.dumps([receipts, result])
    for secret in [TOKEN, "private-cookie", "private-authorization", "private-query"]:
        assert secret not in serialized
    assert all(call[2]["headers"]["Cache-Control"] == "no-cache"
               for call in calls if call[0] == "GET")


def test_unit_readback_timeout_after_write_marks_unknown_without_second_write(protocol, transport, tmp_path):
    calls, replies = transport
    queue_update(replies, after=3)
    replies[-1] = requests.Timeout("private transport failure")
    executor = initialized(protocol)
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    with pytest.raises(requests.Timeout):
        invoke(ledger, lambda: executor.add_quantity(TASK, 1))
    assert ledger.get("action-1")["state"] == "unknown"
    with pytest.raises(protocol.ActionUnresolved):
        invoke(ledger, lambda: executor.add_quantity(TASK, 1))
    assert len(calls) == 4
    assert executor.http_receipts[-1]["error_type"] == "Timeout"
    assert executor.readback_http_request_count == 3


def test_unit_readback_is_new_gets_not_cached_action_receipt(protocol, transport):
    calls, replies = transport
    queue_update(replies)
    replies.extend([response(body=HTML.encode()), response([{**ITEM, "qty": 4}])])
    executor = initialized(protocol)
    assert executor.set_quantity(TASK, 1)["observed_quantity"] == 1
    latest = executor.reobserve_cart({**TASK, "quantity": 1})
    assert latest["observed_quantity"] == 4
    assert latest["cart_verified"] is False
    assert [call[0] for call in calls[-2:]] == ["GET", "GET"]
    assert executor.readback_http_request_count == 2


def observe(ledger, execute):
    return ledger.execute_observed("task-1", "session-1", "action-1", PARAMS, execute)


@pytest.mark.parametrize("method", ["execute_once", "execute_observed"])
@pytest.mark.parametrize("quantity", [0, 0.0, 1.5, 2])
def test_unit_negative_observation_confirms_response_not_goal_success(protocol, tmp_path, method, quantity):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    receipt = {**RECEIPT, "cart_verified": False, "observed_quantity": quantity}
    result = getattr(ledger, method)("task-1", "session-1", "action-1", PARAMS, lambda: receipt)
    assert result["state"] == "confirmed"
    assert result["receipt"] == receipt
    assert result["receipt"]["cart_verified"] is False
    assert ledger.get("action-1")["state"] == "confirmed"
    replay = invoke(ledger, lambda: pytest.fail("negative receipt must still suppress guarded replay"))
    assert replay["replayed"] is True
    assert replay["receipt"] == receipt


@pytest.mark.parametrize("method", ["execute_once", "execute_observed"])
@pytest.mark.parametrize("bad_fields", [
    {"cart_verified": 0}, {"cart_verified": 1}, {"cart_verified": "false"}, {"cart_verified": None},
    {"observed_quantity": None}, {"observed_quantity": True}, {"observed_quantity": "0"},
    {"observed_quantity": -1}, {"observed_quantity": float("nan")},
    {"observed_quantity": float("inf")}, {"observed_quantity": float("-inf")},
    {"task_id": "different-task"},
])
def test_unit_receipt_validation_is_strict_in_both_modes(protocol, tmp_path, method, bad_fields):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    receipt = {**RECEIPT, **bad_fields}
    with pytest.raises((ValueError, TypeError)):
        getattr(ledger, method)("task-1", "session-1", "action-1", PARAMS, lambda: receipt)
    assert ledger.get("action-1")["state"] == "unknown"


@pytest.mark.parametrize("method", ["execute_once", "execute_observed"])
@pytest.mark.parametrize("missing", ["task_id", "cart_verified", "observed_quantity"])
def test_unit_missing_receipt_fields_are_not_defaulted(protocol, tmp_path, method, missing):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    receipt = {key: value for key, value in RECEIPT.items() if key != missing}
    with pytest.raises((ValueError, TypeError)):
        getattr(ledger, method)("task-1", "session-1", "action-1", PARAMS, lambda: receipt)
    assert ledger.get("action-1")["state"] == "unknown"


def test_unit_observed_duplicates_write_twice_and_persist_each_response(protocol, transport, tmp_path):
    calls, replies = transport
    queue_update(replies, before=2, after=3)
    queue_update(replies, before=3, after=4)
    executor = initialized(protocol)
    path = tmp_path / "actions.sqlite"
    first = observe(protocol.ActionLedger(path), lambda: executor.add_quantity(TASK, 1))
    second = observe(protocol.ActionLedger(path), lambda: executor.add_quantity(TASK, 1))
    assert first["receipt"]["observed_quantity"] == 3
    assert second["receipt"]["observed_quantity"] == 4
    assert first["replayed"] is second["replayed"] is False
    assert first["delivery_id"] != second["delivery_id"]
    assert [call[0] for call in calls].count("POST") == 2
    ledger = protocol.ActionLedger(path)
    assert ledger.get("action-1")["execution_count"] == 2
    logged = [e["details"]["receipt"] for e in ledger.events("action-1")
              if e["event"] == "observed_delivery_confirmed"]
    assert [r["observed_quantity"] for r in logged] == [3, 4]


def test_unit_observed_delivery_has_no_guarded_two_attempt_limit(protocol, tmp_path):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    writes = []
    for _ in range(4):
        result = observe(ledger, lambda: writes.append(1) or RECEIPT)
        assert result["replayed"] is False
    assert len(writes) == 4
    assert ledger.get("action-1")["execution_count"] == 4


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt])
def test_unit_observed_exception_blocks_known_unknown_in_both_modes(protocol, tmp_path, failure):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    observe(ledger, lambda: RECEIPT)
    writes = []

    def execute():
        writes.append(1)
        raise failure("private operation error")

    with pytest.raises(failure):
        observe(ledger, execute)
    assert ledger.get("action-1")["state"] == "unknown"
    assert ledger.get("action-1")["execution_count"] == 2
    assert ledger.get("action-1")["receipt"] is None
    for method in (invoke, observe):
        with pytest.raises(protocol.ActionUnresolved, match="unknown"):
            method(protocol.ActionLedger(path), execute)
    assert writes == [1]
    assert "private operation error" not in json.dumps(ledger.events("action-1"))


@pytest.mark.parametrize("changes", [{"task_id": "task-2"}, {"session_id": "session-2"},
                                    {"params": {**PARAMS, "quantity": 9}}])
def test_unit_observed_duplicate_cannot_rebind_action(protocol, tmp_path, changes):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    observe(ledger, lambda: RECEIPT)
    binding = dict(task_id="task-1", session_id="session-1", action_id="action-1", params=PARAMS)
    binding.update(changes)
    with pytest.raises(protocol.ActionBindingError):
        ledger.execute_observed(**binding, execute=lambda: pytest.fail("rebound action executed"))


def test_unit_observed_mode_cannot_bypass_an_inflight_claim(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    entered, release = Event(), Event()

    def execute():
        entered.set()
        assert release.wait(5)
        return RECEIPT

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(observe, ledger, execute)
        try:
            assert entered.wait(5)
            with pytest.raises(protocol.ActionUnresolved, match="executing"):
                observe(protocol.ActionLedger(path), lambda: pytest.fail("inflight claim bypassed"))
        finally:
            release.set()
        assert future.result(timeout=5)["state"] == "confirmed"
    assert ledger.get("action-1")["execution_count"] == 1


def test_unit_guarded_unknown_cannot_be_bypassed_by_observed_mode(protocol, tmp_path):
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")

    def execute():
        raise TimeoutError("written but no reply")

    with pytest.raises(TimeoutError):
        invoke(ledger, execute)
    with pytest.raises(protocol.ActionUnresolved, match="unknown"):
        observe(ledger, lambda: pytest.fail("must not resend unknown write"))


def test_unit_real_adapter_negative_readback_is_recorded_not_infrastructure_error(protocol, transport, tmp_path):
    _, replies = transport
    queue_update(replies, before=2, after=2, write_qty=3)
    executor = initialized(protocol)
    ledger = protocol.ActionLedger(tmp_path / "actions.sqlite")
    result = invoke(ledger, lambda: executor.add_quantity(TASK, 1))
    assert result["state"] == "confirmed"
    assert result["receipt"]["cart_verified"] is False
    assert result["receipt"]["observed_quantity"] == 2


@pytest.mark.parametrize("state", ["confirmed", "unknown", "executing"])
def test_unit_original_database_migration_preserves_outcomes_and_bindings(protocol, tmp_path, state):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE actions (
            action_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, session_id TEXT NOT NULL,
            params_json TEXT NOT NULL, params_sha256 TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending','executing','confirmed','unknown')),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 2),
            execution_count INTEGER NOT NULL DEFAULT 0 CHECK(execution_count BETWEEN 0 AND 1),
            owner TEXT, receipt_json TEXT, error_type TEXT)""")
        db.execute("INSERT INTO actions VALUES(?,?,?,?,?,?,1,1,NULL,?,NULL)",
                   ("action-1", "task-1", "session-1",
                    json.dumps(PARAMS, sort_keys=True, separators=(",", ":")), "legacy-digest", state,
                    json.dumps(RECEIPT) if state == "confirmed" else None))
    ledger = protocol.ActionLedger(path)
    assert ledger.get("action-1")["state"] == state
    assert ledger.get("action-1")["params"] == PARAMS
    if state == "confirmed":
        replay = invoke(ledger, lambda: pytest.fail("legacy confirmed action repeated by guard"))
        assert replay["receipt"] == RECEIPT
        assert observe(ledger, lambda: RECEIPT)["execution_count"] == 2
        assert observe(protocol.ActionLedger(path), lambda: RECEIPT)["execution_count"] == 3
    else:
        for method in (invoke, observe):
            with pytest.raises(protocol.ActionUnresolved):
                method(ledger, lambda: pytest.fail("legacy unresolved action repeated"))


def test_unit_observed_confirmation_failure_preserves_previous_receipt_history(protocol, tmp_path):
    path = tmp_path / "actions.sqlite"
    ledger = protocol.ActionLedger(path)
    observe(ledger, lambda: RECEIPT)
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TRIGGER fail_confirm BEFORE UPDATE ON actions
                      WHEN NEW.state = 'confirmed'
                      BEGIN SELECT RAISE(ABORT, 'confirmation interrupted'); END""")
    with pytest.raises(sqlite3.DatabaseError):
        observe(ledger, lambda: {**RECEIPT, "observed_quantity": 4})
    assert ledger.get("action-1")["state"] == "unknown"
    history = [e["details"]["receipt"] for e in ledger.events("action-1")
               if e["event"] == "observed_delivery_confirmed"]
    assert history == [RECEIPT]
    with pytest.raises(protocol.ActionUnresolved):
        observe(ledger, lambda: pytest.fail("unknown was repeated"))
