"""Persistent Shopping execute-once guard, not an exactly-once guarantee.

An external write and a SQLite commit cannot be atomic. Unknown or abandoned
executing actions require investigation, never an automatic lease takeover.
The ledger stores legitimate runtime receipts, not injector message caches.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mas_faults.webarena_shopping_real import ShoppingHTTPExecutor, _first


class ActionBindingError(ValueError):
    """An existing action ID was reused with a different immutable envelope."""


class ActionUnresolved(RuntimeError):
    """Execution may already have happened; another write is unsafe."""


class PreexecutionFailed(RuntimeError):
    """Both permitted preexecution attempts failed without entering execute."""


def _json(value: Any) -> str:
    def validate(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise TypeError("Action data must be JSON values")

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ActionLedger:
    """One global action ID binds task, session, and canonical JSON parameters.

    Use a persistent local database shared by all deliveries of the same action.
    A confirmed response means execution returned a structurally valid receipt,
    not that the shopping goal succeeded: cart_verified may be False.

    execute_once suppresses confirmed duplicates; execute_observed records and
    executes each confirmed duplicate again. Neither bypasses executing/unknown
    outcomes. Execution counts measure callback entries, not HTTP writes.

    ``preexecute`` is a trusted side-effect-free delivery/validation hook. Only
    its failures can retry, once across restarts; it must not perform the action.
    Replayed receipts are historical confirmation, not fresh observations.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if not self.path or self.path == ":memory:":
            raise ValueError("ActionLedger requires a persistent SQLite path")
        with self._transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 2}:
                raise ValueError("Unsupported ActionLedger schema version")
            existing = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='actions'").fetchone()
            if existing and version == 0:
                # Rebuild atomically to lift v1's single-execution CHECK limits.
                objects = db.execute("""SELECT sql FROM sqlite_master WHERE tbl_name='actions'
                    AND type IN ('trigger','index') AND sql IS NOT NULL""").fetchall()
                db.execute("ALTER TABLE actions RENAME TO actions_v1")
                self._create_actions(db)
                db.execute("INSERT INTO actions SELECT * FROM actions_v1")
                db.execute("DROP TABLE actions_v1")
                for obj in objects:
                    db.execute(obj[0])
            else:
                self._create_actions(db)
            db.execute("""CREATE TABLE IF NOT EXISTS action_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details_json TEXT NOT NULL
            )""")
            db.execute("PRAGMA user_version = 2")

    @staticmethod
    def _create_actions(db):
        db.execute("""CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                params_json TEXT NOT NULL,
                params_sha256 TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','executing','confirmed','unknown')),
                attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                execution_count INTEGER NOT NULL DEFAULT 0 CHECK(execution_count >= 0),
                owner TEXT,
                receipt_json TEXT,
                error_type TEXT
            )""")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous = FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _event(db, action_id, event, **details):
        db.execute("INSERT INTO action_events(action_id,timestamp,event,details_json) VALUES(?,?,?,?)",
                   (action_id, datetime.now(timezone.utc).isoformat(), event, _json(details)))

    @staticmethod
    def _record(row):
        if row is None:
            return None
        result = dict(row)
        result["params"] = json.loads(result.pop("params_json"))
        raw = result.pop("receipt_json")
        result["receipt"] = json.loads(raw) if raw is not None else None
        result.pop("owner")
        return result

    def register(self, task_id: str, session_id: str, action_id: str,
                 params: dict[str, Any]) -> dict[str, Any]:
        for value in (task_id, session_id, action_id):
            if type(value) is not str or not value.strip():
                raise ValueError("task/session/action IDs must be nonempty strings")
        if not isinstance(params, dict):
            raise TypeError("Action parameters must be an object")
        canonical = _json(params)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if row is None:
                db.execute("""INSERT INTO actions
                    (action_id,task_id,session_id,params_json,params_sha256,state)
                    VALUES(?,?,?,?,?,'pending')""", (action_id, task_id, session_id, canonical, digest))
                self._event(db, action_id, "registered", params_sha256=digest)
                row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            elif (row["task_id"], row["session_id"], row["params_json"]) != (task_id, session_id, canonical):
                raise ActionBindingError("Action ID is bound to different task/session/parameters")
            return self._record(row)

    def get(self, action_id: str) -> dict[str, Any] | None:
        with self._transaction() as db:
            row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return self._record(row)

    def events(self, action_id: str) -> list[dict[str, Any]]:
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM action_events WHERE action_id=? ORDER BY event_id",
                              (action_id,)).fetchall()
            result = []
            for row in rows:
                event = dict(row)
                event["details"] = json.loads(event.pop("details_json"))
                result.append(event)
            return result

    def _transition(self, action_id, owner, state, event, *, receipt=None, error_type=None):
        with self._transaction() as db:
            changed = db.execute("""UPDATE actions SET state=?, receipt_json=?, error_type=?, owner=NULL
                WHERE action_id=? AND state='executing' AND owner=?""",
                (state, receipt, error_type, action_id, owner)).rowcount
            if changed != 1:
                raise ActionUnresolved("Execution claim no longer belongs to this caller")
            details = {"error_type": error_type, "delivery_id": owner}
            if receipt is not None:
                details["receipt"] = json.loads(receipt)
            self._event(db, action_id, event, **details)
            row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            return self._record(row)

    @staticmethod
    def _validate_receipt(receipt, task_id):
        if not isinstance(receipt, dict) or type(receipt.get("cart_verified")) is not bool:
            raise ValueError("Shopping receipt requires a boolean cart_verified")
        if receipt.get("task_id") != task_id:
            raise ActionBindingError("Receipt task does not match action")
        quantity = receipt.get("observed_quantity")
        if (type(quantity) not in {int, float} or quantity < 0
                or (type(quantity) is float and not math.isfinite(quantity))):
            raise ValueError("Shopping receipt requires a finite nonnegative numeric observed_quantity")
        return _json(receipt)

    def execute_once(self, task_id: str, session_id: str, action_id: str,
                     params: dict[str, Any], execute: Callable[[], dict[str, Any]], *,
                     preexecute: Callable[[], None] | None = None) -> dict[str, Any]:
        """Execute a pending action or return its recorded positive/negative receipt."""
        return self._execute(task_id, session_id, action_id, params, execute,
                             observed=False, preexecute=preexecute)

    def execute_observed(self, task_id: str, session_id: str, action_id: str,
                         params: dict[str, Any], execute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Record an unguarded delivery, executing confirmed duplicates again.

        No retry is performed. Executing and unknown actions raise ActionUnresolved.
        Each completed delivery has a distinct delivery_id and a receipt in events;
        get(action_id) returns the latest state and cumulative execution_count.
        """
        return self._execute(task_id, session_id, action_id, params, execute, observed=True)

    def _execute(self, task_id, session_id, action_id, params, execute, *, observed, preexecute=None):
        self.register(task_id, session_id, action_id, params)
        while True:
            owner = uuid.uuid4().hex
            with self._transaction() as db:
                row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
                if row["state"] == "confirmed" and not observed:
                    self._event(db, action_id, "confirmed_receipt_replayed")
                    return {**self._record(row), "replayed": True}
                if row["state"] not in {"pending", "confirmed"}:
                    raise ActionUnresolved(f"Action is {row['state']}; no automatic reexecution")
                if row["state"] == "pending" and row["attempts"] >= 2:
                    raise PreexecutionFailed("Preexecution retry budget exhausted; action unexecuted")
                db.execute("""UPDATE actions SET state='executing', attempts=attempts+1, owner=?,
                              receipt_json=NULL, error_type=NULL WHERE action_id=?""", (owner, action_id))
                self._event(db, action_id, "claimed", attempt=row["attempts"] + 1,
                            delivery_id=owner, mode="observed" if observed else "guarded")

            if preexecute is not None:
                try:
                    preexecute()
                except Exception as exc:
                    self._transition(action_id, owner, "pending", "preexecution_failed_unexecuted",
                                     error_type=type(exc).__name__)
                    continue
                except BaseException as exc:
                    self._transition(action_id, owner, "pending", "preexecution_failed_unexecuted",
                                     error_type=type(exc).__name__)
                    raise

            try:
                # Commit the execution boundary before calling any external code.
                with self._transaction() as db:
                    changed = db.execute("""UPDATE actions SET execution_count=execution_count+1
                        WHERE action_id=? AND state='executing' AND owner=?""", (action_id, owner)).rowcount
                    if changed != 1:
                        raise ActionUnresolved("Execution claim lost before callback")
                    self._event(db, action_id, "execute_entered", delivery_id=owner,
                                mode="observed" if observed else "guarded")
                receipt = execute()
                serialized = self._validate_receipt(receipt, task_id)
                result = self._transition(action_id, owner, "confirmed",
                                          "observed_delivery_confirmed" if observed else "confirmed",
                                          receipt=serialized)
            except BaseException as exc:
                try:
                    self._transition(action_id, owner, "unknown", "execution_outcome_unknown",
                                     error_type=type(exc).__name__)
                except (sqlite3.Error, ActionUnresolved):
                    # A failed journal update must not replace the original failure
                    # or make a persisted executing claim eligible for replay.
                    pass
                raise
            return {**result, "replayed": False, "delivery_id": owner}


class MultiStateShoppingExecutor(ShoppingHTTPExecutor):
    """Single-cart state updates using the base executor's guarded HTTP transport.

Initial ``add_to_cart`` is inherited unchanged. Each update performs a product
GET, a cart-item GET, one PUT/POST, and a fresh cart GET. ``quantity`` is an
absolute positive integer for set_quantity and a positive increment for
add_quantity. Unguarded repeated increments deliberately issue repeated POSTs.
The adapter does not retry writes; wrap it in ActionLedger when guarding them.
"""

    @staticmethod
    def _quantity(quantity):
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("Shopping quantity must be a positive integer")
        return quantity

    def _identity(self, task, purpose):
        url = self._url(task["product_url"])
        separator = "&" if "?" in url else "?"
        self.readback_http_request_count += 1
        response, receipt = self._request(
            "GET", f"{url}{separator}__mas_readback={uuid.uuid4().hex}", purpose,
            timeout=30, headers={"Cache-Control": "no-cache"})
        identity = {
            "product_id": _first(r'name="product"\s+value="([^"]+)"', response.text),
            "sku": _first(r'data-product-sku="([^"]+)"', response.text),
        }
        receipt["response_payload"] = identity.copy()
        if not all(identity.values()):
            raise ValueError("Shopping product identity unavailable")
        return identity

    def _items(self, purpose):
        self.readback_http_request_count += 1
        response, receipt = self._request("GET", self._cart_items_url(), purpose,
                                          timeout=30, headers={"Cache-Control": "no-cache"})
        items = response.json()
        receipt["response_payload"] = ([self._item_fields(item) for item in items]
                                       if isinstance(items, list) else None)
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("Shopping cart response must be a list of items")
        return items

    @staticmethod
    def _matching_item(items, sku):
        matches = [item for item in items if item.get("sku") == sku]
        if len(matches) > 1:
            raise ValueError("Shopping cart has ambiguous duplicate SKU items")
        return matches[0] if matches else None

    def _observation(self, task, identity, quantity, purpose, start):
        item = self._matching_item(self._items(purpose), identity["sku"])
        observed_quantity = item.get("qty") if item else 0
        if type(observed_quantity) is float and observed_quantity.is_integer():
            observed_quantity = int(observed_quantity)
        observed = {
            "task_id": task["task_id"], "product_title": item.get("name", "") if item else "",
            **identity, "requested_quantity": quantity, "observed_quantity": observed_quantity,
            "cart_verified": bool(item and item.get("name") == task["product_title"]
                                  and type(observed_quantity) is int and observed_quantity == quantity),
        }
        return {**observed, "evidence": json.dumps(observed, ensure_ascii=False),
                "http_receipt_indices": [r["receipt_index"] for r in self.http_receipts[start:]]}

    def _change_quantity(self, task, quantity, *, additive):
        self.readback_http_request_count = 0
        self._quantity(quantity)
        self._url(task["product_url"])
        if not isinstance(self.guest_cart_id, str) or not self.guest_cart_id:
            raise ValueError("State update requires an existing guest cart")
        purpose = "add_quantity" if additive else "set_quantity"
        start = len(self.http_receipts)
        identity = self._identity(task, purpose + ".product_page")
        item = self._matching_item(self._items(purpose + ".lookup_items"), identity["sku"])
        if item is None or item.get("name") != task["product_title"]:
            raise ValueError("Requested product is not present in the existing cart")
        item_id = item.get("item_id")
        if type(item_id) is not int or item_id <= 0:
            raise ValueError("Shopping cart item ID must be a positive integer")
        payload = {"sku": identity["sku"], "qty": quantity}
        url = self._cart_items_url()
        if additive:
            before = item.get("qty")
            if type(before) is float and before.is_integer():
                before = int(before)
            self._quantity(before)
            expected = before + quantity
            method, operation = "POST", ".add_item"
        else:
            expected = quantity
            payload["item_id"] = item_id
            url += f"/{item_id}"
            method, operation = "PUT", ".set_item"
        response, receipt = self._request(method, url, purpose + operation,
                                          json={"cartItem": payload}, timeout=30)
        receipt["request_payload"] = {"cartItem": payload.copy()}
        receipt["response_payload"] = self._item_fields(response.json())
        return self._observation(task, identity, expected, purpose + ".readback_items", start)

    def set_quantity(self, task: dict[str, Any], quantity: int) -> dict[str, Any]:
        """PUT an absolute quantity to the item ID freshly found in this cart."""
        return self._change_quantity(task, quantity, additive=False)

    def add_quantity(self, task: dict[str, Any], quantity: int) -> dict[str, Any]:
        """POST an increment to this cart; duplicate unguarded calls can add twice."""
        return self._change_quantity(task, quantity, additive=True)

    def reobserve_cart(self, task: dict[str, Any]) -> dict[str, Any]:
        """Read fresh identity and cart state with no writes or cached receipts."""
        self.readback_http_request_count = 0
        self._url(task["product_url"])
        self._quantity(task["quantity"])
        if not self.guest_cart_id:
            return {"task_id": task["task_id"], "cart_verified": False, "status": "no executed cart"}
        start = len(self.http_receipts)
        identity = self._identity(task, "reobserve_cart.product_page")
        return self._observation(task, identity, task["quantity"], "reobserve_cart.items", start)
