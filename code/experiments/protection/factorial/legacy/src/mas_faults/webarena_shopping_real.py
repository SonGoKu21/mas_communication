"""Live WebArena Shopping workflow with communication-fault tracing.

The executor uses the deployed Magento pages and forms directly. It is an
HTTP page/form executor, not a claim of pixel-level browser automation; the
communication boundary and task-state evaluator are real.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_task_selection import load_task_manifest


CONDITIONS = [
    "clean",
    "a1_moderate_delay",
    "a1_deadline_delay",
    "a5_omission",
    "a8_truncation",
    "a8_semantic_truncation",
    "a12_stale_replay",
    "a12_silent_stale_replay",
]

SEMANTIC_CONDITIONS = [
    "clean",
    "a6_product_id_sku_swap",
    "a6_inner_evidence_poisoning",
    "a5_semantic_evidence_omission",
    "a10_stale_state_reordering",
    "a19_contract_semantic_drift",
    "a6_quantity_corruption",
    "a14_task_binding_error",
]

CORE_CONDITIONS = (
    "clean",
    "a1_moderate_delay",
    "a1_deadline_delay",
    "a5_omission",
    "a6_inner_evidence_poisoning",
    "a8_truncation",
    "a12_stale_replay",
)


def resolve_conditions(requested: list[str] | None) -> tuple[str, ...]:
    if requested is None:
        return CORE_CONDITIONS
    supported = set(CONDITIONS) | set(SEMANTIC_CONDITIONS)
    unknown = [condition for condition in requested if condition not in supported]
    if unknown:
        raise ValueError(f"unsupported conditions: {', '.join(unknown)}")
    if "clean" not in requested:
        raise ValueError("the experiment matrix must include clean")
    return tuple(requested)


def apply_message_fault(message, condition, stale_message=None):
    """Return delivered message, A symptom, and whether a fault was applied."""
    if condition == "clean":
        return copy.deepcopy(message), "none", False
    if condition == "a1_moderate_delay":
        time.sleep(0.25)
        return copy.deepcopy(message), "A1_message_latency", True
    if condition == "a1_deadline_delay":
        return None, "A2_message_timeout", True
    if condition == "a1_deadline_no_retry":
        return None, "A2_message_timeout", True
    if condition == "a5_omission":
        return None, "A5_message_omission", True
    if condition == "a5_omission_hard":
        return None, "A5_message_omission", True
    if condition == "a8_truncation":
        partial = copy.deepcopy(message)
        partial["evidence"] = str(partial.get("evidence", ""))[:96]
        partial["truncated"] = True
        return partial, "A8_message_truncation", True
    if condition == "a8_truncation_hard":
        partial = copy.deepcopy(message)
        partial["evidence"] = ""
        partial["cart_verified"] = False
        partial["truncated"] = True
        return partial, "A8_message_truncation", True
    if condition == "a8_semantic_truncation":
        partial = copy.deepcopy(message)
        partial.pop("product_id", None)
        partial.pop("sku", None)
        partial["evidence"] = json.dumps({"task_id": message.get("task_id"), "cart_verified": True})
        partial["cart_verified"] = True
        partial["truncated"] = False
        return partial, "A8_message_truncation", True
    if condition == "a6_product_id_sku_swap":
        swapped = copy.deepcopy(message)
        swapped["product_id"] = str(stale_message.get("product_id", "swapped-product")) if stale_message else "swapped-product"
        swapped["sku"] = str(stale_message.get("sku", "SWAPPED-SKU")) if stale_message else "SWAPPED-SKU"
        return swapped, "A6_message_semantic_corruption", True
    if condition == "a6_inner_evidence_poisoning":
        poisoned = copy.deepcopy(message)
        if stale_message is None:
            raise ValueError("inner evidence poisoning needs a prior worker message")
        poisoned["evidence"] = stale_message.get("evidence", "previous-task-evidence")
        return poisoned, "A6_message_semantic_corruption", True
    if condition == "a5_semantic_evidence_omission":
        partial = copy.deepcopy(message)
        for key in ["product_id", "sku", "observed_quantity", "evidence"]:
            partial.pop(key, None)
        return partial, "A5_message_omission", True
    if condition == "a10_stale_state_reordering":
        if stale_message is None:
            raise ValueError("state reordering needs a prior worker message")
        reordered = copy.deepcopy(stale_message)
        reordered["task_id"] = message.get("task_id")
        reordered["product_title"] = message.get("product_title")
        reordered["state_version"] = 1
        return reordered, "A10_state_reordering", True
    if condition == "a19_contract_semantic_drift":
        drifted = copy.deepcopy(message)
        drifted["verified_cart"] = drifted.pop("cart_verified", True)
        drifted["quantity_seen"] = str(drifted.pop("observed_quantity", "1"))
        return drifted, "A19_contract_semantic_drift", True
    if condition == "a6_quantity_corruption":
        corrupted = copy.deepcopy(message)
        corrupted["observed_quantity"] = int(message.get("requested_quantity", 1)) + 1
        return corrupted, "A6_message_semantic_corruption", True
    if condition == "a14_task_binding_error":
        if stale_message is None:
            raise ValueError("task binding error needs a prior worker message")
        bound = copy.deepcopy(message)
        bound["message_id"] = stale_message.get("message_id", "previous-message")
        bound["evidence"] = stale_message.get("evidence", "previous-task-evidence")
        bound["source_session"] = stale_message.get("source_session", "previous-session")
        return bound, "A14_task_binding_error", True
    if condition == "a12_stale_replay":
        if stale_message is None:
            raise ValueError("stale replay needs a prior worker message")
        return copy.deepcopy(stale_message), "A12_timing_or_session_mismatch", True
    if condition == "a12_stale_replay_poisoned":
        if stale_message is None:
            raise ValueError("stale replay needs a prior worker message")
        poisoned = copy.deepcopy(stale_message)
        poisoned["task_id"] = message.get("task_id")
        poisoned["product_title"] = message.get("product_title")
        poisoned["stale_replayed"] = True
        return poisoned, "A12_timing_or_session_mismatch", True
    if condition == "a12_silent_stale_replay":
        if stale_message is None:
            raise ValueError("stale replay needs a prior worker message")
        stale = copy.deepcopy(stale_message)
        stale["task_id"] = message.get("task_id")
        stale["product_title"] = message.get("product_title")
        return stale, "A12_timing_or_session_mismatch", True
    raise ValueError(f"unsupported condition: {condition}")


def _first(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else default


class ShoppingHTTPExecutor:
    """Live HTTP execution with sanitized, zero-based, append-only request receipts.

    Receipts contain response-derived fields only, never task/oracle assertions.
    Redirects fail closed; callers must configure the actual Magento origin.
    """

    def __init__(self, base_url: str):
        from urllib.parse import urlsplit

        if urlsplit(base_url).query or urlsplit(base_url).fragment:
            raise ValueError("Shopping base URL cannot contain a query or fragment")
        self.base_url = self._canonical_url(base_url).rstrip("/")
        self.session = requests.Session()
        self.guest_cart_id = None
        self.http_receipts: list[dict[str, Any]] = []
        self._next_http_receipt_index = 0
        self.readback_http_request_count = 0

    @staticmethod
    def _canonical_url(value: str) -> str:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(value)
        if parts.username is not None or parts.password is not None:
            raise ValueError("Shopping URL credentials are not allowed")
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Shopping URL must have an HTTP(S) origin")
        host = parts.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = parts.port
        if port is not None and port != {"http": 80, "https": 443}[parts.scheme]:
            host = f"{host}:{port}"
        return urlunsplit((parts.scheme, host, parts.path or "/", parts.query, ""))

    def _url(self, value: str) -> str:
        from urllib.parse import urljoin, urlsplit

        url = self._canonical_url(urljoin(self.base_url + "/", value))
        base, target = urlsplit(self.base_url), urlsplit(url)
        if (target.scheme, target.netloc) != (base.scheme, base.netloc):
            raise ValueError("Shopping URL origin does not match executor origin")
        return url

    def _discovery_url(self, value: str) -> str:
        """Rebase only catalog links baked into the local Magento image at discovery."""
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(self._canonical_url(value)) if value.startswith(("http://", "https://")) else None
        if parts and (parts.scheme, parts.netloc) == ("http", "localhost:7770"):
            base = urlsplit(self.base_url)
            value = urlunsplit((base.scheme, base.netloc, parts.path, parts.query, ""))
        return self._url(value)

    def _cart_hash(self) -> str | None:
        import hashlib

        return hashlib.sha256(self.guest_cart_id.encode("utf-8")).hexdigest() if self.guest_cart_id else None

    def _cart_items_url(self) -> str:
        from urllib.parse import quote

        return f"{self.base_url}/rest/V1/guest-carts/{quote(self.guest_cart_id, safe='')}/items"

    @staticmethod
    def _receipt_url(url: str) -> str:
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

        parts = urlsplit(url)
        path = re.sub(r"(/guest-carts/)[^/]+", r"\1[REDACTED]", parts.path)
        query = urlencode([(key, "[REDACTED]") for key, _ in parse_qsl(parts.query, keep_blank_values=True)])
        return urlunsplit((parts.scheme, parts.netloc, path, query, ""))

    @staticmethod
    def _item_fields(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        return {key: item[key] for key in ("item_id", "sku", "name", "qty")
                if key in item and (item[key] is None or type(item[key]) in {str, int, float, bool})}

    def _request(self, method: str, url: str, purpose: str, **kwargs):
        import hashlib

        url = self._url(url)
        receipt = {
            "receipt_index": self._next_http_receipt_index, "purpose": purpose,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_method": method, "request_url": self._receipt_url(url),
            "guest_cart_id_sha256": self._cart_hash(), "status_code": None,
            "response_sha256": None, "response_hash_source": "unavailable",
            "response_payload": None,
        }
        self._next_http_receipt_index += 1
        self.http_receipts.append(receipt)
        try:
            response = getattr(self.session, method.lower())(url, allow_redirects=False, **kwargs)
            status = getattr(response, "status_code", None)
            receipt["status_code"] = status if type(status) is int else None
            body = getattr(response, "content", None)
            if isinstance(body, bytes):
                receipt["response_hash_source"] = "content"
            else:
                body = getattr(response, "text", None)
                if isinstance(body, str):
                    body = body.encode("utf-8")
                    receipt["response_hash_source"] = "text_utf8"
            if isinstance(body, bytes):
                receipt["response_sha256"] = hashlib.sha256(body).hexdigest()
            if type(status) is int and 300 <= status < 400:
                raise RuntimeError("Shopping HTTP redirect refused")
            response_url = getattr(response, "url", None)
            if isinstance(response_url, str) and response_url:
                self._url(response_url)
            response.raise_for_status()
        except Exception as exc:
            receipt["error_type"] = type(exc).__name__
            raise
        return response, receipt

    def discover_tasks(self, limit: int) -> list[dict[str, Any]]:
        response, receipt = self._request("GET", f"{self.base_url}/", "discover_tasks.homepage", timeout=30)
        products = re.findall(
            r'<a\s+title="([^"]+)"\s+href="([^"]+)"\s+class="product-item-link"',
            response.text,
            flags=re.IGNORECASE,
        )
        tasks = []
        seen = set()
        for title, href in products:
            if href in seen:
                continue
            seen.add(href)
            tasks.append({"task_id": f"shopping-{len(tasks)+1:03d}", "product_title": title, "product_url": self._discovery_url(href), "quantity": 1})
            if len(tasks) >= limit:
                break
        receipt["response_payload"] = {"product_count": len(tasks)}
        if not tasks:
            raise RuntimeError("Shopping homepage exposed no product tasks")
        return tasks

    def add_to_cart(self, task: dict[str, Any]) -> dict[str, Any]:
        url = self._url(task["product_url"])
        separator = "&" if "?" in url else "?"
        page, page_receipt = self._request(
            "GET", f"{url}{separator}__mas_run={uuid.uuid4().hex}", "add_to_cart.product_page",
            timeout=30, headers={"Cache-Control": "no-cache"})
        html = page.text
        sku = _first(r'data-product-sku="([^"]+)"', html)
        product_id = _first(r'name="product"\s+value="([^"]+)"', html)
        page_receipt["response_payload"] = {"product_id": product_id, "sku": sku}
        if not sku or not product_id:
            raise RuntimeError("Shopping product page did not expose a product SKU")
        cart_response, cart_receipt = self._request(
            "POST", f"{self.base_url}/rest/V1/guest-carts", "add_to_cart.create_cart", timeout=30)
        self.guest_cart_id = cart_response.json()
        if not isinstance(self.guest_cart_id, str) or not self.guest_cart_id:
            self.guest_cart_id = None
            raise ValueError("Shopping guest cart response is not a nonempty token")
        cart_receipt["guest_cart_id_sha256"] = self._cart_hash()
        cart_receipt["response_payload"] = {"guest_cart_id_sha256": self._cart_hash()}
        item_response, item_receipt = self._request(
            "POST", self._cart_items_url(), "add_to_cart.add_item",
            json={"cartItem": {"sku": sku, "qty": task["quantity"]}},
            timeout=30,
        )
        item = item_response.json()
        item_receipt["response_payload"] = self._item_fields(item)
        title_in_cart = item.get("name") == task["product_title"]
        observed_quantity = int(item.get("qty", 0))
        return {
            "task_id": task["task_id"],
            "product_title": task["product_title"],
            "product_id": product_id,
            "sku": sku,
            "requested_quantity": task["quantity"],
            "observed_quantity": observed_quantity,
            "cart_verified": bool(title_in_cart and observed_quantity == task["quantity"]),
            "evidence": json.dumps({"task_id": task["task_id"], "product_title": task["product_title"], "product_id": product_id, "observed_quantity": observed_quantity, "cart_verified": title_in_cart and observed_quantity == task["quantity"]}, ensure_ascii=False),
        }

    def verify_cart(self, task: dict[str, Any]) -> dict[str, Any]:
        if not self.guest_cart_id:
            return {"task_id": task["task_id"], "product_title": task["product_title"], "observed_quantity": 0, "cart_verified": False, "evidence": "no guest cart"}
        cart, receipt = self._request("GET", self._cart_items_url(), "verify_cart.items", timeout=30)
        items = cart.json()
        receipt["response_payload"] = [self._item_fields(item) for item in items] if isinstance(items, list) else None
        matched = next((item for item in items if item.get("name") == task["product_title"]), None)
        title_in_cart = matched is not None
        observed_quantity = int(matched.get("qty", 0)) if matched else 0
        return {"task_id": task["task_id"], "product_title": task["product_title"], "observed_quantity": observed_quantity, "cart_verified": bool(title_in_cart and observed_quantity == task["quantity"]), "evidence": "recovery cart recheck"}

    def reobserve_cart(self, task: dict[str, Any]) -> dict[str, Any]:
        """Read fresh identity and cart state; never create a cart or repeat a write."""
        self.readback_http_request_count = 0
        if not self.guest_cart_id:
            return {"task_id": task["task_id"], "cart_verified": False, "status": "no executed cart"}
        url = self._url(task["product_url"])
        separator = "&" if "?" in url else "?"
        self.readback_http_request_count += 1
        page, page_receipt = self._request(
            "GET", f"{url}{separator}__mas_readback={uuid.uuid4().hex}", "reobserve_cart.product_page",
            timeout=30, headers={"Cache-Control": "no-cache"})
        product_id = _first(r'name="product"\s+value="([^"]+)"', page.text)
        sku = _first(r'data-product-sku="([^"]+)"', page.text)
        page_receipt["response_payload"] = {"product_id": product_id, "sku": sku}
        if not product_id or not sku:
            raise RuntimeError("readback product identity unavailable")
        self.readback_http_request_count += 1
        cart, cart_receipt = self._request(
            "GET", self._cart_items_url(), "reobserve_cart.items",
            timeout=30, headers={"Cache-Control": "no-cache"})
        items = cart.json()
        cart_receipt["response_payload"] = [self._item_fields(item) for item in items] if isinstance(items, list) else None
        if not isinstance(items, list):
            raise ValueError("readback cart response is not a list")
        matched = next((item for item in items if isinstance(item, dict) and item.get("sku") == sku), None)
        quantity = matched.get("qty", 0) if matched else 0
        if isinstance(quantity, float) and quantity.is_integer():
            quantity = int(quantity)
        observed = {
            "task_id": task["task_id"], "product_title": matched.get("name", "") if matched else "",
            "product_id": product_id, "sku": sku, "requested_quantity": task["quantity"],
            "observed_quantity": quantity,
            "cart_verified": bool(matched and matched.get("name") == task["product_title"]
                                  and type(quantity) is int and quantity == task["quantity"]),
        }
        return {**observed, "evidence": json.dumps(observed, ensure_ascii=False)}


def parse_verdict(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        try:
            value = json.loads(match.group(0) if match else raw)
        except json.JSONDecodeError:
            return {"decision": "reject", "task_id": "", "reason": "non_json_verdict"}
    if not isinstance(value, dict):
        return {"decision": "reject", "task_id": "", "reason": "non_object_verdict"}
    return {"decision": "accept" if str(value.get("decision", "reject")).lower() == "accept" else "reject", "task_id": str(value.get("task_id", "")), "reason": str(value.get("reason", ""))}


def coordinator_verdict(client, task: dict[str, Any], delivered: dict[str, Any] | None) -> dict[str, str]:
    payload = "missing" if delivered is None else json.dumps(delivered, ensure_ascii=False)
    prompt = (
        "You are the Coordinator for a real Shopping task. Return strict JSON only: "
        '{"decision":"accept"|"reject","task_id":string,"reason":string}. '
        "Accept only when the worker evidence is complete, belongs to the current task, "
        "has the exact current product title, "
        "and cart_verified is true. Reject missing or truncated evidence.\n"
        f"Current task: {json.dumps(task, ensure_ascii=False)}\nWorker message: {payload}"
    )
    return parse_verdict(client.complete(prompt))


def consequence(condition: str, original: dict[str, Any], delivered: dict[str, Any] | None, verdict: dict[str, str], task: dict[str, Any]) -> tuple[list[str], bool, str, str]:
    accepted = verdict["decision"] == "accept"
    evidence = {}
    if delivered and isinstance(delivered.get("evidence"), str):
        try:
            evidence = json.loads(delivered["evidence"])
        except json.JSONDecodeError:
            evidence = {}
    complete = bool(
        delivered
        and not delivered.get("truncated")
        and delivered.get("task_id") == task["task_id"]
        and delivered.get("product_title") == task["product_title"]
        and delivered.get("product_id") == original.get("product_id")
        and delivered.get("sku") == original.get("sku")
        and delivered.get("observed_quantity") == task["quantity"]
        and delivered.get("cart_verified") is True
        and evidence.get("task_id") == task["task_id"]
        and evidence.get("product_id") == original.get("product_id")
        and evidence.get("observed_quantity") == task["quantity"]
        and verdict["task_id"] == task["task_id"]
    )
    if accepted and complete:
        if condition == "a1_moderate_delay":
            return [], True, "framework_recovery", "message delivered after moderate delay"
        return [], False, "none", ""
    if condition in {"a1_deadline_delay", "a5_omission"}:
        return ["M2_task_timeout_or_failure", "M3_incomplete_information_aggregation"], True, "timeout_or_retry_recovery", "Coordinator requested a cart recheck"
    if condition == "a8_truncation":
        return ["M3_incomplete_information_aggregation"], True, "reasoning_recovery", "Coordinator rejected truncated evidence and rechecked the cart"
    if condition == "a8_semantic_truncation":
        if accepted:
            return ["M14_partial_tool_or_message_result_acceptance", "M4_incorrect_collective_decision"], False, "none", "Coordinator accepted syntactically valid but incomplete evidence"
        return [], True, "reasoning_recovery", "Coordinator rejected incomplete evidence and rechecked the cart"
    if condition == "a12_stale_replay":
        return [], True, "reasoning_recovery", "Coordinator rejected evidence from a different task and rechecked the cart"
    if condition in SEMANTIC_CONDITIONS:
        if not accepted:
            return [], True, "reasoning_recovery", "Coordinator rejected the semantic communication fault and rechecked the cart"
        consequences = []
        if condition in {"a6_product_id_sku_swap", "a6_inner_evidence_poisoning", "a6_quantity_corruption", "a14_task_binding_error"}:
            consequences.extend(["M4_incorrect_collective_decision", "M6_state_inconsistency"])
        elif condition == "a5_semantic_evidence_omission":
            consequences.extend(["M14_partial_tool_or_message_result_acceptance", "M4_incorrect_collective_decision"])
        elif condition == "a10_stale_state_reordering":
            consequences.extend(["M5_stale_context_acceptance", "M4_incorrect_collective_decision", "M6_state_inconsistency"])
        elif condition == "a19_contract_semantic_drift":
            consequences.extend(["M14_partial_tool_or_message_result_acceptance", "M6_state_inconsistency"])
        return consequences, False, "none", "Coordinator accepted semantically invalid evidence without triggering recovery"
    return ["M2_task_timeout_or_failure"], False, "none", ""


def write_event(handle, run_id, trace_id, task_id, condition, layer, event_type, effect, label, details):
    handle.write(json.dumps({"run_id": run_id, "trace_id": trace_id, "timestamp": datetime.now(timezone.utc).isoformat(), "event_layer": layer, "component": "shopping_http_executor", "event_type": event_type, "source": "ShoppingWorker", "target": "Coordinator", "condition": condition, "effect": effect, "propagation_label": label, "task_id": task_id, "details": details}, ensure_ascii=False) + "\n")


def propagation_class(applied: bool, consequences: list[str], recovered: bool, success: bool, accepted: bool) -> str:
    if not applied:
        return "clean"
    if recovered and success:
        return "detected_and_recovered"
    if consequences and not recovered and not success:
        return "propagated_to_M_final_failure"
    if consequences and not recovered and accepted:
        return "silent_propagation_to_M"
    return "detected_but_unrecovered"


def run_one(client, executor, task, condition, stale, event_handle):
    run_id, trace_id = f"{task['task_id']}-{condition}-{uuid.uuid4().hex[:8]}", f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = (client.call_count, client.prompt_tokens, client.completion_tokens)
    original = executor.add_to_cart(task)
    original["source_agent"], original["target_agent"] = "ShoppingWorker", "Coordinator"
    original["message_id"] = f"message-{uuid.uuid4()}"
    original["source_session"] = f"session-{task['task_id']}"
    original["state_version"] = 2
    original["send_timestamp"] = datetime.now(timezone.utc).isoformat()
    delivered, a_symptom, applied = apply_message_fault(original, condition, stale)
    original["delivery_timestamp"] = datetime.now(timezone.utc).isoformat() if delivered else None
    if condition == "a10_stale_state_reordering":
        write_event(event_handle, run_id, trace_id, task["task_id"], condition, "A", "message_delivered", "newer_state_version_2", "pre_injection", {"message": original, "state_version": 2})
        write_event(event_handle, run_id, trace_id, task["task_id"], condition, "A", "message_reordered", "older_state_version_1_delivered_last", "injected", {"message": delivered, "state_version": 1})
    write_event(event_handle, run_id, trace_id, task["task_id"], condition, "A", "fault_applied" if applied else "message_delivered", a_symptom, "injected" if applied else "pre_injection", {"fault_applied": applied, "original": original, "delivered": delivered})
    verdict = coordinator_verdict(client, task, delivered)
    consequences, recovered, recovery_type, recovery_evidence = consequence(condition, original, delivered, verdict, task)
    if recovered and (condition in SEMANTIC_CONDITIONS or condition in {"a8_semantic_truncation", "a12_stale_replay"} or not delivered or delivered.get("truncated") or not delivered.get("cart_verified")):
        recovery = executor.verify_cart(task)
        recovery_evidence = f"{recovery_evidence}; cart_verified={recovery.get('cart_verified')}"
        success = bool(recovery.get("cart_verified"))
    else:
        success = not consequences and verdict["decision"] == "accept"
    run_propagation_class = propagation_class(applied, consequences, recovered, success, verdict["decision"] == "accept")
    for item in consequences:
        write_event(event_handle, run_id, trace_id, task["task_id"], condition, "M", "m_consequence_observed", item, "propagated", {"verdict": verdict})
    write_event(event_handle, run_id, trace_id, task["task_id"], condition, "M", "final_consequence", "task_success" if success else "task_failure", "recovered" if recovered else ("pre_injection" if condition == "clean" else "final_failure"), {"verdict": verdict})
    return normalize_run_record({
        "run_id": run_id, "trace_id": trace_id, "benchmark": "WebArena-Verified-Shopping", "execution_mode": "live_site_http", "scenario": "shopping_cart_verification", "task_id": task["task_id"], "condition": condition, "model": client.model_info.model, "provider": client.model_info.provider, "fault_id": "none" if condition == "clean" else f"fault-{run_id}", "fault_type": condition, "fault_applied": applied, "source_agent": "ShoppingWorker", "target_agent": "Coordinator", "original_message": original, "delivered_message": delivered, "first_divergence": "A:fault_applied" if applied else "none", "observed_runtime_effect": a_symptom, "observed_A_symptom": a_symptom, "observed_M_consequence": consequences or ["none"], "recovery_detected": recovered, "recovery_type": recovery_type, "recovery_evidence": recovery_evidence, "propagation_class": run_propagation_class, "expected_answer": {"task_id": task["task_id"], "product_title": task["product_title"], "quantity": task["quantity"], "cart_verified": True}, "final_answer": {"verdict": verdict, "cart_verified": success}, "task_score": 1.0 if success else 0.0, "final_task_success": success, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "api_call_count": client.call_count - before[0], "prompt_tokens": client.prompt_tokens - before[1], "completion_tokens": client.completion_tokens - before[2], "total_tokens": (client.prompt_tokens - before[1]) + (client.completion_tokens - before[2]), "error": None,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("SHOPPING_BASE_URL", "http://localhost:7770"))
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--task-manifest", help="Frozen task manifest created by mas_faults.webarena_task_selection.")
    parser.add_argument("--runs-per-condition", type=int, default=2)
    parser.add_argument("--output-dir", default="results/webarena_shopping_real_smoke")
    parser.add_argument("--matrix", choices=["primary", "semantic"], default="primary")
    parser.add_argument("--conditions", nargs="+", help="Explicit condition list; defaults to the frozen six-fault core matrix.")
    args = parser.parse_args()
    selected_conditions = resolve_conditions(args.conditions) if args.conditions else tuple(CONDITIONS if args.matrix == "primary" else SEMANTIC_CONDITIONS)
    load_dotenv(); load_dotenv(".env.local")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory exists: {output}")
    output.mkdir(parents=True)
    client = get_llm_client()
    discovery = ShoppingHTTPExecutor(args.base_url)
    candidates = load_task_manifest(Path(args.task_manifest)) if args.task_manifest else discovery.discover_tasks(max(args.tasks * 2, args.tasks))
    requested_tasks = len(candidates) if args.task_manifest else args.tasks
    tasks, preflight_skipped = [], []
    for candidate in candidates:
        try:
            ShoppingHTTPExecutor(args.base_url).add_to_cart(candidate)
            tasks.append(candidate)
        except requests.RequestException as exc:
            preflight_skipped.append({"task_id": candidate["task_id"], "product_title": candidate["product_title"], "error": str(exc)})
        if len(tasks) >= requested_tasks:
            break
    repeated_task_instances = 0
    while not args.task_manifest and tasks and len(tasks) < requested_tasks:
        repeated = copy.deepcopy(tasks[len(tasks) % len(tasks)])
        repeated["task_id"] = f"shopping-{len(tasks)+1:03d}-repeat"
        tasks.append(repeated)
        repeated_task_instances += 1
    if not tasks:
        raise RuntimeError("No Shopping products passed Guest Cart preflight")
    rows, stale = [], {"task_id": "shopping-seed-000", "product_title": "stale seed from an earlier task instance", "cart_verified": True, "evidence": "seed message"}
    event_path = output / "shopping_causal_events.jsonl"
    with event_path.open("w", encoding="utf-8") as event_handle:
        for task in tasks:
            task_stale = stale
            for condition in selected_conditions:
                for _ in range(args.runs_per_condition):
                    executor = ShoppingHTTPExecutor(args.base_url)
                    row = run_one(client, executor, task, condition, task_stale, event_handle)
                    rows.append(row)
            stale = rows[-1]["original_message"]
    with (output / "shopping_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (output / "shopping_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    summary = {"benchmark": "WebArena-Verified-Shopping", "execution_mode": "live_site_http", "runs": len(rows), "final_success": sum(bool(r["final_task_success"]) for r in rows), "a_exposure": sum(r["observed_A_symptom"] != ["none"] for r in rows), "m_propagation": sum(r["observed_M_consequence"] != ["none"] for r in rows), "recovery": sum(bool(r["recovery_detected"]) for r in rows), "final_failure": sum(not bool(r["final_task_success"]) for r in rows), "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 3), "mean_total_tokens": round(sum(r["total_tokens"] for r in rows) / len(rows), 3)}
    (output / "shopping_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "shopping_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    md = ["# WebArena Verified Shopping Communication Experiment", "", f"- Execution mode: `{summary['execution_mode']}`", f"- Runs: {summary['runs']}", f"- Final task success: {summary['final_success']}/{summary['runs']}", f"- A-layer exposure: {summary['a_exposure']}/{summary['runs']}", f"- M-layer propagation: {summary['m_propagation']}/{summary['runs']}", f"- Recovery: {summary['recovery']}/{summary['runs']}", f"- Final failure: {summary['final_failure']}/{summary['runs']}", ""]
    (output / "shopping_summary.md").write_text("\n".join(md), encoding="utf-8")
    (output / "experiment_config.json").write_text(json.dumps({"benchmark": "WebArena-Verified-Shopping", "execution_mode": "live_site_http", "matrix": args.matrix, "base_url": args.base_url, "task_manifest": args.task_manifest, "model": client.model_info.model, "provider": client.model_info.provider, "conditions": selected_conditions, "tasks_requested": requested_tasks, "tasks_executed": len(tasks), "runs_per_condition": args.runs_per_condition, "repeated_task_instances": repeated_task_instances, "preflight_skipped": preflight_skipped, "task_definition": "add the requested quantity of a selected local Shopping product to a fresh Guest Cart, and verify SKU/title/quantity", "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
