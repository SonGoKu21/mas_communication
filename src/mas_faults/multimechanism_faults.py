"""Single-boundary operators; instances are owned by the injector, not policies."""
from __future__ import annotations

import copy
import json

from mas_faults.multimechanism_matrix import CELLS, config_digest


class SingleBoundaryFault:
    def __init__(self, condition, *, old_evidence=None, cross_task_evidence=None, old_judgment=None):
        if condition not in CELLS:
            raise ValueError("unknown condition")
        self.condition = condition
        self.boundary = CELLS[condition]
        self.old_evidence = copy.deepcopy(old_evidence)
        self.cross_task_evidence = copy.deepcopy(cross_task_evidence)
        self.old_judgment = copy.deepcopy(old_judgment)
        self.events = []

    def deliver(self, boundary, message):
        current = copy.deepcopy(message)
        if self.boundary is None or self.events or boundary != self.boundary:
            return [current]
        event = {"boundary": boundary, "condition": self.condition,
                 "original_sha256": config_digest(current)}
        condition = self.condition
        if condition in {"request_non_delivery", "acknowledgement_loss"}:
            delivered = []
        elif condition == "duplicate_action_delivery":
            delivered = [current, copy.deepcopy(current)]
        elif condition == "valid_partial":
            current["payload"] = {key: value for key, value in current["payload"].items()
                                  if key in {"task_id", "product_title", "requested_quantity", "cart_verified"}}
            delivered = [current]
        elif condition == "same_session_reordering":
            old = self.old_evidence
            if (not isinstance(old, dict) or not old.get("evidence_id")
                    or any(old.get(k) != current.get(k) for k in ("task_id", "session_id", "entity_id"))
                    or type(old.get("version")) is not int or type(current.get("version")) is not int
                    or old["version"] >= current["version"]):
                raise ValueError("reordering requires a real earlier same-session entity observation")
            event["source_evidence_id"] = old["evidence_id"]
            event["source_sha256"] = config_digest(old)
            delivered = [current, copy.deepcopy(old)]
        elif condition in {"cross_task_replay", "contract_consistent_identity_corruption"}:
            source = self.cross_task_evidence
            if (not isinstance(source, dict) or not source.get("evidence_id")
                    or not source.get("task_id") or source["task_id"] == current.get("task_id")
                    or not isinstance(source.get("payload"), dict)
                    or not source["payload"].get("sku") or not source["payload"].get("product_id")):
                raise ValueError("a recorded cross-task identity source is required")
            event["source_evidence_id"] = source["evidence_id"]
            event["source_sha256"] = config_digest(source)
            if condition == "cross_task_replay":
                delivered = [copy.deepcopy(source)]
            else:
                if source["payload"]["sku"] == current["payload"].get("sku"):
                    raise ValueError("identity corruption requires a different product SKU")
                for key in ("sku", "product_id"):
                    current["payload"][key] = source["payload"][key]
                current["payload"]["evidence"] = json.dumps(
                    {k: v for k, v in current["payload"].items() if k != "evidence"}, ensure_ascii=False)
                delivered = [current]
        elif condition == "stale_judgment_replay":
            old = self.old_judgment
            if (not isinstance(old, dict) or not old.get("judgment_id") or not old.get("evidence_ids")
                    or not isinstance(old.get("verdict"), dict)):
                raise ValueError("a recorded prior judgment is required")
            event["source_judgment_id"] = old["judgment_id"]
            event["source_sha256"] = config_digest(old)
            delivered = [copy.deepcopy(old)]
        else:
            alternate = copy.deepcopy(current)
            payload = alternate["payload"]
            if type(payload.get("observed_quantity")) is not int:
                raise ValueError("conflict source needs an observed integer quantity")
            payload["observed_quantity"] += 1
            payload["cart_verified"] = payload["observed_quantity"] == payload.get("requested_quantity")
            payload["evidence"] = json.dumps({k: v for k, v in payload.items() if k != "evidence"}, ensure_ascii=False)
            alternate["evidence_id"] += "-conflicting"
            delivered = [alternate]
        event["delivered_sha256"] = [config_digest(value) for value in delivered]
        event["delivered_count"] = len(delivered)
        self.events.append(event)
        return delivered
