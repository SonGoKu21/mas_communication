"""Offline transport fixtures for the real Shopping executor, not a runtime backend."""

import hashlib
import json
from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from mas_faults.webarena_shopping_real import ShoppingHTTPExecutor, parse_verdict


BASE = "http://shopping.invalid"
TOKEN = "private-guest-cart-token"
TOKEN_HASH = hashlib.sha256(TOKEN.encode()).hexdigest()
TASK = {"task_id": "task-1", "product_title": "Tea", "product_url": "/tea.html", "quantity": 2}
HTML = '<input name="product" value="10"><button data-product-sku="TEA">'
ITEM = {"item_id": 7, "sku": "TEA", "name": "Tea", "qty": 2, "quote_id": TOKEN,
        "extension_attributes": {"cookie": "private-cookie"}, "password": "private-password"}


def response(payload=None, *, body=None, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.encoding = "utf-8"
    result._content = body if body is not None else json.dumps(payload).encode()
    result.headers.update(headers or {})
    return result


@pytest.fixture
def transport(monkeypatch):
    calls = []
    replies = []

    def send(session, method, url, **kwargs):
        calls.append((method.upper(), url, kwargs))
        assert replies, "unexpected HTTP request"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        reply.url = url
        return reply

    monkeypatch.setattr(requests.Session, "request", send)
    return calls, replies


@pytest.mark.parametrize("raw", ["[]", "null", '"accept"', "true", "42",
                               '[{"decision":"accept","task_id":"task-1"}]',
                               json.dumps('{"decision":"accept","task_id":"task-1"}')])
def test_non_object_verdict_explicitly_rejects_without_extracting_nested_object(raw):
    verdict = parse_verdict(raw)
    assert verdict == {"decision": "reject", "task_id": "", "reason": "non_object_verdict"}


@pytest.mark.parametrize("raw", [
    '{"decision":"accept","task_id":"other-task","reason":"observed"}',
    '```json\n{"decision":"accept","task_id":"other-task","reason":"observed"}\n```',
])
def test_verdict_preserves_received_task_binding(raw):
    assert parse_verdict(raw) == {"decision": "accept", "task_id": "other-task", "reason": "observed"}
    assert parse_verdict('{"decision":"accept"}')["task_id"] == ""
    assert parse_verdict("not json")["reason"] == "non_json_verdict"


@pytest.mark.parametrize("product_url", ["tea.html?color=green#details", "/tea.html?color=green#details",
                                          "http://SHOPPING.invalid:80/tea.html?color=green#details"])
def test_add_and_reobserve_canonicalize_the_same_product_url(transport, product_url):
    calls, replies = transport
    replies.extend([response(body=HTML.encode()), response(TOKEN), response(ITEM),
                    response(body=HTML.encode()), response([ITEM])])
    executor = ShoppingHTTPExecutor(BASE)
    task = {**TASK, "product_url": product_url}
    assert executor.add_to_cart(task)["cart_verified"] is True
    assert executor.reobserve_cart(task)["cart_verified"] is True
    for offset, cache_key in [(0, "__mas_run"), (3, "__mas_readback")]:
        parsed = urlsplit(calls[offset][1])
        assert (parsed.scheme, parsed.netloc, parsed.path, parsed.fragment) == (
            "http", "shopping.invalid", "/tea.html", "")
        assert parse_qs(parsed.query)["color"] == ["green"]
        assert cache_key in parse_qs(parsed.query)
    assert executor.readback_http_request_count == 2


@pytest.mark.parametrize("method", ["add_to_cart", "reobserve_cart"])
@pytest.mark.parametrize("url", ["http://localhost:7770/tea.html", "http://other.invalid/tea.html",
                                  "//other.invalid/tea.html", "https://shopping.invalid/tea.html",
                                  "http://shopping.invalid:81/tea.html",
                                  "http://user:private-password@shopping.invalid/tea.html"])
def test_execution_rejects_foreign_origin_and_userinfo_before_http(transport, method, url):
    calls, _ = transport
    executor = ShoppingHTTPExecutor(BASE)
    executor.guest_cart_id = TOKEN
    with pytest.raises(ValueError, match="origin|credentials"):
        getattr(executor, method)({**TASK, "product_url": url})
    assert calls == []
    assert executor.http_receipts == []
    assert executor.readback_http_request_count == 0


def test_only_discovery_rebases_baked_in_localhost_catalog_links(transport):
    calls, replies = transport
    homepage = '<a title="Tea" href="http://localhost:7770/tea.html" class="product-item-link">Tea</a>'
    replies.append(response(body=homepage.encode()))
    executor = ShoppingHTTPExecutor(BASE)
    discovered = executor.discover_tasks(1)
    assert discovered[0]["product_url"] == BASE + "/tea.html"
    with pytest.raises(ValueError, match="origin"):
        executor.add_to_cart({**TASK, "product_url": "http://localhost:7770/tea.html"})
    assert len(calls) == 1


def test_receipts_cover_writes_reads_hashes_and_sanitized_observed_fields(transport):
    calls, replies = transport
    raw_responses = [response(body=(HTML + '<input name="form_key" value="private-form-key">').encode()),
                     response(TOKEN), response(ITEM), response([ITEM]),
                     response(body=HTML.encode()), response([ITEM])]
    replies.extend(raw_responses)
    executor = ShoppingHTTPExecutor(BASE)
    executor.session.headers["Authorization"] = "Bearer private-authorization"
    executor.session.cookies.set("session", "private-cookie")
    task = {**TASK, "product_url": "/tea.html?password=private-query-password&token=private-query-token"}
    executor.add_to_cart(task)
    executor.verify_cart(task)
    executor.reobserve_cart(task)
    receipts = executor.http_receipts
    assert [r["receipt_index"] for r in receipts] == list(range(6))
    assert [r["purpose"] for r in receipts] == [
        "add_to_cart.product_page", "add_to_cart.create_cart", "add_to_cart.add_item",
        "verify_cart.items", "reobserve_cart.product_page", "reobserve_cart.items"]
    assert [r["request_method"] for r in receipts] == ["GET", "POST", "POST", "GET", "GET", "GET"]
    assert all(r["status_code"] == 200 for r in receipts)
    assert all(datetime.fromisoformat(r["timestamp"]).utcoffset().total_seconds() == 0 for r in receipts)
    assert [r["response_sha256"] for r in receipts] == [hashlib.sha256(r.content).hexdigest() for r in raw_responses]
    assert all(r["response_hash_source"] == "content" for r in receipts)
    assert receipts[0]["response_payload"] == {"product_id": "10", "sku": "TEA"}
    assert receipts[1]["response_payload"] == {"guest_cart_id_sha256": TOKEN_HASH}
    assert receipts[2]["response_payload"] == {"item_id": 7, "sku": "TEA", "name": "Tea", "qty": 2}
    assert receipts[3]["response_payload"] == [receipts[2]["response_payload"]]
    assert all(r["guest_cart_id_sha256"] == TOKEN_HASH for r in receipts[1:])
    assert receipts[2]["request_url"] == BASE + "/rest/V1/guest-carts/[REDACTED]/items"
    assert all(call[2].get("allow_redirects") is False for call in calls)
    serialized = json.dumps(receipts)
    for secret in [TOKEN, "private-password", "private-cookie", "private-form-key", "private-authorization",
                   "private-query-password", "private-query-token"]:
        assert secret not in serialized
    assert executor.readback_http_request_count == 2


def test_readback_counts_attempts_per_call_and_receipts_survive_http_errors(transport):
    _, replies = transport
    replies.extend([response(body=HTML.encode()), response([ITEM]),
                    response(body=b"private-error-body", status=503)])
    executor = ShoppingHTTPExecutor(BASE)
    executor.guest_cart_id = TOKEN
    executor.reobserve_cart(TASK)
    assert executor.readback_http_request_count == 2
    with pytest.raises(requests.HTTPError):
        executor.reobserve_cart(TASK)
    assert executor.readback_http_request_count == 1
    assert [r["receipt_index"] for r in executor.http_receipts] == [0, 1, 2]
    failure = executor.http_receipts[-1]
    assert failure["status_code"] == 503
    assert failure["response_sha256"] == hashlib.sha256(b"private-error-body").hexdigest()
    assert failure["error_type"] == "HTTPError"
    assert "private-error-body" not in json.dumps(executor.http_receipts)
    executor.guest_cart_id = None
    assert executor.reobserve_cart(TASK)["cart_verified"] is False
    assert executor.readback_http_request_count == 0
    assert len(executor.http_receipts) == 3


def test_transport_failure_is_not_fabricated_as_an_http_response(transport):
    _, replies = transport
    replies.append(requests.Timeout("private-transport-error"))
    executor = ShoppingHTTPExecutor(BASE)
    executor.guest_cart_id = TOKEN
    with pytest.raises(requests.Timeout):
        executor.reobserve_cart(TASK)
    assert executor.readback_http_request_count == 1
    receipt = executor.http_receipts[0]
    assert receipt["status_code"] is None
    assert receipt["response_sha256"] is None
    assert receipt["error_type"] == "Timeout"
    assert "private-transport-error" not in json.dumps(receipt)


def test_redirect_is_recorded_but_never_followed_or_written_through(transport):
    calls, replies = transport
    replies.append(response(body=b"redirect", status=302,
                            headers={"Location": "http://localhost:7770/tea.html", "Set-Cookie": "private-cookie"}))
    executor = ShoppingHTTPExecutor(BASE)
    with pytest.raises(RuntimeError, match="redirect"):
        executor.add_to_cart(TASK)
    assert len(calls) == 1
    assert calls[0][2]["allow_redirects"] is False
    assert executor.http_receipts[0]["status_code"] == 302
    assert "private-cookie" not in json.dumps(executor.http_receipts)


def test_receipts_support_minimal_legacy_response_without_inventing_status():
    class Response:
        text = HTML

        def raise_for_status(self):
            pass

        def json(self):
            return [{"sku": "TEA", "name": "Tea", "qty": 2}]

    class Session:
        def get(self, url, **kwargs):
            return Response()

    executor = ShoppingHTTPExecutor(BASE)
    executor.session = Session()
    executor.guest_cart_id = TOKEN
    assert executor.reobserve_cart(TASK)["cart_verified"] is True
    assert executor.readback_http_request_count == 2
    assert all(r["status_code"] is None for r in executor.http_receipts)
    assert all(r["response_hash_source"] == "text_utf8" for r in executor.http_receipts)
    assert all(r["response_sha256"] == hashlib.sha256(HTML.encode()).hexdigest() for r in executor.http_receipts)
