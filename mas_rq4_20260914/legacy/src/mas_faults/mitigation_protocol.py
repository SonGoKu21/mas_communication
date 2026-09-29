"""Runtime-only budgets and evidence dependencies for mitigation experiments."""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable
from typing import Any


def _nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _json_copy(value: Any) -> Any:
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("payload must contain only finite JSON data with string keys")

    try:
        check(value)
    except RecursionError as exc:
        raise ValueError("payload must be acyclic JSON data") from exc
    return copy.deepcopy(value)


def _ids(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("IDs must be an iterable of strings, not a string")
    result = []
    for value in values:
        _identifier(value, "dependency ID")
        if value not in result:
            result.append(value)
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


_SHOPPING_STATE_FIELDS = (
    "task_id", "product_title", "product_id", "sku", "requested_quantity",
    "observed_quantity", "cart_verified",
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous duplicate JSON key")
        result[key] = value
    return result


def _shopping_payload_projection(payload: Any) -> Any:
    """Compare Shopping state plus all unknown content, without receipt indices.

    Recognition requires all seven _SHOPPING_STATE_FIELDS at the top level.
    Other payloads, including partial Shopping payloads, are compared in full.
    The ONLY ignored field is ``http_receipt_indices``, at the recognized
    payload's top level and in its immediate ``evidence`` JSON object. It is
    not removed recursively from arbitrary content or generic payloads.

    Object-valued evidence is compared structurally; JSON object strings are
    decoded, making key order, whitespace, and escaping immaterial. Unknown
    fields and outer/inner disagreements remain part of the comparison.
    Malformed, duplicate-key, nonfinite, and non-object JSON evidence retains
    its original representation. Raw registration and audit data is unaltered.
    """
    if not isinstance(payload, dict) or not all(key in payload for key in _SHOPPING_STATE_FIELDS):
        return payload
    state = {key: payload[key] for key in _SHOPPING_STATE_FIELDS}
    state.update({key: value for key, value in payload.items()
                  if key not in _SHOPPING_STATE_FIELDS and key != "http_receipt_indices"})
    nested = state.get("evidence")
    if isinstance(nested, str):
        try:
            nested = _json_copy(json.loads(nested, object_pairs_hook=_unique_json_object))
        except (ValueError, RecursionError):
            return state
    if isinstance(nested, dict):
        state["evidence"] = {key: value for key, value in nested.items()
                             if key != "http_receipt_indices"}
    return state


class BudgetExhausted(RuntimeError):
    """The requested operation would exceed a shared mitigation budget."""


class Budget:
    """In-memory per-run accounting; consume before attempting an operation.

    Kinds are ``get``, ``model_call``, and ``replay``. Failed attempts still
    consume budget. This counter does not authorize a replay or perform I/O.
    """

    def __init__(self, max_gets: int = 4, max_model_calls: int = 3, max_replays: int = 1):
        self._limits = {"get": max_gets, "model_call": max_model_calls, "replay": max_replays}
        for kind, limit in self._limits.items():
            _nonnegative_int(limit, kind)
        self._used = dict.fromkeys(self._limits, 0)
        self._events: list[dict[str, Any]] = []

    def consume(self, kind: str, count: int = 1) -> dict[str, Any]:
        if kind not in self._limits:
            raise ValueError(f"unknown budget kind: {kind}")
        _nonnegative_int(count, "count")
        exhausted = self._used[kind] + count > self._limits[kind]
        if not exhausted:
            self._used[kind] += count
        state = self.snapshot()
        self._events.append({
            "event_type": "budget_exhausted" if exhausted else "budget_consumed",
            "kind": kind, "count": count, **state,
        })
        if exhausted:
            raise BudgetExhausted(f"budget exhausted: {kind}")
        return copy.deepcopy(state)

    def snapshot(self) -> dict[str, Any]:
        return {"limits": dict(self._limits), "used": dict(self._used),
                "remaining": {kind: limit - self._used[kind]
                              for kind, limit in self._limits.items()}}

    @property
    def events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)


class EvidenceGraph:
    """One task/session's observed evidence and immutable judgment DAG.

    Nonnegative integer versions are runtime logical clocks per entity, not
    environment truth. Strings are opaque versions (equal tokens can agree);
    different tokens, mixed clock types, and None require external verification.
    Inputs and scope metadata must come from the runtime, not model assertions.

    ``valid`` means dependency-valid, not semantically correct or complete.
    Callers check task-defined obligations and semantic decisions separately.
    No I/O, model calls, automatic replay, or external-action rollback occurs.

    Conflict comparison uses _shopping_payload_projection for recognized
    Shopping state: only top-level and immediate evidence-object
    ``http_receipt_indices`` are ignored, and nested JSON objects are compared
    structurally. All unknown fields still count; generic payloads are intact.
    """

    def __init__(self, task_id: str, session_id: str):
        _identifier(task_id, "task_id")
        _identifier(session_id, "session_id")
        self._task_id = task_id
        self._session_id = session_id
        self._evidence: dict[str, dict[str, Any]] = {}
        self._judgments: dict[str, dict[str, Any]] = {}
        self._known_versions: dict[str, int] = {}
        self._events: list[dict[str, Any]] = []

    def _scope(self, task_id: str, session_id: str) -> None:
        if task_id != self._task_id or session_id != self._session_id:
            raise ValueError("task/session scope mismatch")

    def _event(self, event_type: str, **details: Any) -> None:
        self._events.append(copy.deepcopy({
            "event_type": event_type, "sequence": len(self._events) + 1,
            "task_id": self._task_id, "session_id": self._session_id, **details,
        }))

    def add_evidence(
        self, evidence_id: str, *, entity_id: str, version: int | str | None,
        payload: Any, source: str, task_id: str, session_id: str,
        action_id: str | None = None,
    ) -> dict[str, Any]:
        """Register an observation; an identical ID delivery is idempotent."""
        self._scope(task_id, session_id)
        for name, value in (("evidence_id", evidence_id), ("entity_id", entity_id),
                            ("source", source)):
            _identifier(value, name)
        if action_id is not None:
            _identifier(action_id, "action_id")
        if type(version) is int:
            _nonnegative_int(version, "version")
        elif isinstance(version, str):
            _identifier(version, "version")
        elif version is not None:
            raise ValueError("version must be an integer, opaque string, or None")
        record = {"evidence_id": evidence_id, "task_id": task_id, "session_id": session_id,
                  "entity_id": entity_id, "version": version,
                  "payload": _json_copy(payload), "source": source, "action_id": action_id}
        if evidence_id in self._evidence:
            existing = self._evidence[evidence_id]
            if _canonical({key: existing[key] for key in record}) != _canonical(record):
                raise ValueError(f"evidence ID already registered: {evidence_id}")
            self._event("evidence_duplicate", evidence_id=evidence_id)
            return copy.deepcopy(existing)
        self._evidence[evidence_id] = {**record, "status": "unresolved"}
        if type(version) is int:
            self._known_versions[entity_id] = max(
                version, self._known_versions.get(entity_id, version))
        self._event("evidence_registered", evidence=record)
        self._refresh_entity(entity_id)
        self._refresh_judgments(reason="evidence_updated")
        return copy.deepcopy(self._evidence[evidence_id])

    def _refresh_entity(self, entity_id: str) -> None:
        records = [e for e in self._evidence.values() if e["entity_id"] == entity_id]
        known = self._known_versions.get(entity_id)
        current = [e for e in records if type(e["version"]) is not int or e["version"] == known]
        versions = {(type(e["version"]).__name__, e["version"]) for e in current}
        payloads = {_canonical(_shopping_payload_projection(e["payload"])) for e in current}
        ambiguous = len(versions) != 1 or any(e["version"] is None for e in current)
        conflict = len(payloads) > 1
        for record in records:
            stale = type(record["version"]) is int and record["version"] < known
            status = "stale" if stale else "unresolved" if ambiguous or conflict else "valid"
            before = record["status"]
            record["status"] = status
            if before != status:
                self._event("evidence_state_changed", evidence_id=record["evidence_id"],
                            before=before, after=status)

    def add_judgment(
        self, judgment_id: str, *, provided_evidence_ids: Iterable[str] = (),
        evidence_ids: Iterable[str] | None = None, verdict: Any = None,
        task_id: str | None = None, session_id: str | None = None,
        cited_evidence_ids: Iterable[str] = (),
        parent_judgment_ids: Iterable[str] = (), payload: Any = None,
        status: str = "provisional",
    ) -> dict[str, Any]:
        """Bind actual inputs union explicit citations, plus delivered parents.

        ``evidence_ids`` is an alias for actual provided inputs; ``verdict``
        aliases payload. Omitted scope inherits this runtime-owned graph.
        Parents must already exist, so registration order is topological.
        Recompute with a new ID; invalidated judgments cannot be resurrected.
        """
        task_id = self._task_id if task_id is None else task_id
        session_id = self._session_id if session_id is None else session_id
        self._scope(task_id, session_id)
        _identifier(judgment_id, "judgment_id")
        if judgment_id in self._judgments:
            raise ValueError(f"judgment ID already registered: {judgment_id}")
        if status not in ("provisional", "valid", "invalidated", "unresolved"):
            raise ValueError(f"unknown judgment status: {status}")
        provided, cited, parents = map(_ids, (provided_evidence_ids, cited_evidence_ids,
                                            parent_judgment_ids))
        provided = _ids(provided + _ids(() if evidence_ids is None else evidence_ids))
        dependencies = _ids(provided + cited)
        for ids, registry in ((dependencies, self._evidence), (parents, self._judgments)):
            for dependency in ids:
                if dependency not in registry:
                    raise ValueError(f"unknown dependency ID: {dependency}")
        payload = _json_copy(payload)
        verdict = _json_copy(verdict)
        if verdict is not None:
            if payload is not None and _canonical(payload) != _canonical(verdict):
                raise ValueError("verdict and payload disagree")
            payload = verdict
        record = {
            "judgment_id": judgment_id, "task_id": task_id, "session_id": session_id,
            "provided_evidence_ids": provided, "cited_evidence_ids": cited,
            "evidence_ids": dependencies, "parent_judgment_ids": parents,
            "payload": payload, "verdict": copy.deepcopy(payload), "status": status,
        }
        self._judgments[judgment_id] = record
        self._event("judgment_registered", judgment=record)
        self._refresh_judgments(reason="dependency_check")
        return copy.deepcopy(record)

    def _set_status(self, judgment: dict[str, Any], status: str, reason: str) -> bool:
        before = judgment["status"]
        if before == status or before == "invalidated":
            return False
        judgment["status"] = status
        self._event("judgment_state_changed", judgment_id=judgment["judgment_id"],
                    before=before, after=status, reason=reason)
        return True

    def _refresh_judgments(self, reason: str) -> list[str]:
        changed = []
        # Existing parents precede children: one forward pass covers the DAG.
        for judgment_id, record in self._judgments.items():
            dependencies = [self._evidence[e]["status"] for e in record["evidence_ids"]]
            dependencies += [self._judgments[j]["status"] for j in record["parent_judgment_ids"]]
            if any(status in ("stale", "invalidated") for status in dependencies):
                status = "invalidated"
            elif not dependencies or any(status != "valid" for status in dependencies):
                status = "unresolved"
            else:
                continue
            if self._set_status(record, status, reason):
                changed.append(judgment_id)
        return changed

    def invalidate(self, judgment_id: str, reason: str = "explicit_invalidation") -> list[str]:
        """Invalidate a judgment and its downstream dependents; return changed IDs."""
        _identifier(reason, "reason")
        if judgment_id not in self._judgments:
            raise ValueError(f"unknown judgment ID: {judgment_id}")
        changed = []
        if self._set_status(self._judgments[judgment_id], "invalidated", reason):
            changed.append(judgment_id)
        return changed + self._refresh_judgments(reason=reason)

    def accepted(self, judgment_id: str) -> bool:
        """Whether a registered judgment is currently dependency-valid."""
        if judgment_id not in self._judgments:
            raise ValueError(f"unknown judgment ID: {judgment_id}")
        return self._judgments[judgment_id]["status"] == "valid"

    def snapshot(self) -> dict[str, Any]:
        """Detached diagnostic state, including stale and rejected dependencies."""
        return copy.deepcopy({
            "task_id": self._task_id, "session_id": self._session_id,
            "known_versions": self._known_versions,
            "evidence": self._evidence, "judgments": self._judgments,
        })

    def context_snapshot(self) -> dict[str, Any]:
        """Fresh context containing only usable evidence and valid judgments.

        Snapshot creation is audited. The caller charges ensuing model calls
        to its shared Budget; this method does not perform or charge I/O.
        """
        state = self.snapshot()
        for key in ("evidence", "judgments"):
            state[key] = {name: record for name, record in state[key].items()
                          if record["status"] == "valid"}
        self._event("context_snapshot", evidence_ids=list(state["evidence"]),
                    judgment_ids=list(state["judgments"]))
        return state

    @property
    def events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)
