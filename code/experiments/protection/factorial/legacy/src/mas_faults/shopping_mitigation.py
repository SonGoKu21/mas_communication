"""Receiver-side mitigation without access to injector state or evaluator truth."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

MODES = ("baseline", "always_recheck", "guarded_recheck")
TEXT_FIELDS = ("task_id", "product_title", "product_id", "sku", "evidence")
INTEGER_FIELDS = ("requested_quantity", "observed_quantity")
REQUIRED_FIELDS = TEXT_FIELDS + INTEGER_FIELDS + ("cart_verified",)


@dataclass(frozen=True)
class EvidenceCheck:
    issues: tuple[str, ...]
    missing_fields: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def check_evidence(task: dict[str, Any], payload: Any, *, require_success: bool = True) -> EvidenceCheck:
    if not isinstance(payload, dict):
        return EvidenceCheck(("non_delivery" if payload is None else "not_object",), REQUIRED_FIELDS)
    missing = tuple(key for key in REQUIRED_FIELDS if key not in payload)
    issues = [f"missing:{key}" for key in missing]
    for key in TEXT_FIELDS:
        if key in payload and (not isinstance(payload[key], str) or not payload[key].strip()):
            issues.append(f"type:{key}")
    for key in INTEGER_FIELDS:
        if key in payload and type(payload[key]) is not int:
            issues.append(f"type:{key}")
        elif key in payload and payload[key] < 0:
            issues.append(f"range:{key}")
    if "cart_verified" in payload and type(payload["cart_verified"]) is not bool:
        issues.append("type:cart_verified")
    constraints = [("task_id", task["task_id"]), ("product_title", task["product_title"]),
                   ("requested_quantity", task["quantity"])]
    if require_success:
        constraints += [("observed_quantity", task["quantity"]), ("cart_verified", True)]
    elif payload.get("cart_verified") is True:
        constraints += [("observed_quantity", task["quantity"])]
    for key, expected in constraints:
        if key in payload and payload[key] != expected:
            issues.append(f"constraint:{key}")
    detail = payload.get("evidence")
    # Historical workers may return prose. Compare nested fields only when JSON is supplied.
    if isinstance(detail, str) and detail.lstrip().startswith(("{", "[")):
        try:
            nested = json.loads(detail)
        except (ValueError, TypeError):
            issues.append("nested_unparseable")
        else:
            if not isinstance(nested, dict):
                issues.append("nested_not_object")
            else:
                for key in REQUIRED_FIELDS:
                    if key == "evidence" or key not in nested:
                        continue
                    if key not in payload or type(nested[key]) is not type(payload[key]) or nested[key] != payload[key]:
                        issues.append(f"inner_outer:{key}")
    return EvidenceCheck(tuple(issues), missing)


class EvidencePolicy:
    """One optional readback per run, shared by the receiving nodes of that run."""

    def __init__(self, mode: str, task: dict[str, Any]):
        if mode not in MODES:
            raise ValueError(f"unsupported mitigation mode: {mode}")
        self.mode = mode
        self.task = {key: copy.deepcopy(task[key]) for key in ("task_id", "product_title", "quantity")}
        self.readback_count = 0
        self.events: list[dict[str, Any]] = []
        self._fresh: Any = None
        self._readback_error: str | None = None

    def receive(self, payload: Any, readback: Callable[[], Any], *, receiver: str) -> Any:
        started = time.perf_counter()
        before = check_evidence(self.task, payload)
        triggered = self.mode == "guarded_recheck" and not before.valid
        requested = self.mode == "always_recheck" or triggered
        called = False
        used = False
        result = copy.deepcopy(payload) if isinstance(payload, dict) else None
        if requested:
            if self.readback_count == 0:
                called = True
                self.readback_count += 1
                try:
                    self._fresh = copy.deepcopy(readback())
                except Exception as exc:
                    self._readback_error = type(exc).__name__
            if self._readback_error is None and check_evidence(self.task, self._fresh, require_success=False).valid:
                result = copy.deepcopy(self._fresh)
                used = True
            elif self._readback_error is None or not before.valid:
                result = None
        self.events.append({
            "event_type": "receiver_mitigation", "mode": self.mode, "receiver": receiver,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "before": copy.deepcopy(payload), "before_issues": list(before.issues),
            "triggered": triggered, "readback_requested": requested, "readback_called": called,
            "readback_count": self.readback_count, "readback_error": self._readback_error,
            "readback_response": copy.deepcopy(self._fresh) if called else None,
            "replacement_used": used, "after": copy.deepcopy(result),
            "after_issues": list(check_evidence(self.task, result).issues),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        return result
