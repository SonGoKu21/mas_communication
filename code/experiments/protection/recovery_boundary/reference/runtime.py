"""Real multi-state Shopping mitigation runtime and offline outcome helpers."""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from requests import RequestException

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from mas_faults.mitigation_protocol import Budget, BudgetExhausted, EvidenceGraph
from mas_faults.multimechanism_faults import SingleBoundaryFault
from mas_faults.multimechanism_matrix import ARMS, TOPOLOGIES, config_digest
from mas_faults.shopping_action_protocol import ActionLedger, ActionUnresolved
from mas_faults.shopping_mitigation import check_evidence
from design import mechanisms
from contract import ReceiverContract
from exposure import FiniteExposure


def make_envelope(payload, *, task_id, session_id, entity_id, version, action_id, evidence_id, source):
    return {"task_id": task_id, "session_id": session_id, "entity_id": entity_id,
            "version": version, "action_id": action_id, "evidence_id": evidence_id,
            "source": source, "payload": copy.deepcopy(payload)}


def validate_action(action, task, operation, quantity):
    return (isinstance(action, dict) and action.get("task_id") == task["task_id"]
            and action.get("operation") == operation and type(action.get("quantity")) is int
            and action["quantity"] == quantity)


def semantic_success(task, evidence, environment):
    return bool(isinstance(environment, dict) and check_evidence(task, evidence).valid
                and all(evidence.get(k) == environment.get(k)
                        for k in ("task_id", "product_title", "product_id", "sku", "observed_quantity"))
                and environment.get("cart_verified") is True)


def evidence_state_digest(envelope):
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        return config_digest(payload)
    state = {k: v for k, v in payload.items() if k not in {"http_receipt_indices", "status"}}
    nested = state.get("evidence")
    if isinstance(nested, str):
        try:
            state["evidence"] = json.loads(nested)
        except ValueError:
            pass
    return config_digest(state)


def verify_action_protocol_event(row, event):
    """Cross-check a response recovery/prevention claim, never a goal-success claim.

    Schema v1 links an event to the final action ledger snapshot, its append-only
    events, and the original adapter write/readback HTTP receipts. No new reads.
    """
    outcomes = {"request_redelivery": "recovery", "acknowledgement_retrieval": "recovery",
                "duplicate_prevention": "prevention"}
    if not isinstance(event, dict) or row.get("arm") not in {"action_only", "combined"}:
        return False
    kind = event.get("kind")
    state = row.get("action_ledger_state")
    if (event.get("schema_version") != 1 or kind not in outcomes
            or event.get("outcome") != outcomes[kind] or event.get("response_used") is not True
            or row.get("action_outcome_unknown") or not isinstance(state, dict)
            or state.get("state") != "confirmed" or state.get("execution_count") != 1):
        return False
    if any(event.get(k) != state.get(k) or not event.get(k) for k in ("task_id", "session_id", "action_id")):
        return False
    if event["task_id"] != row["task"]["task_id"]:
        return False
    expected_before = ("pending", 0) if kind == "request_redelivery" else ("confirmed", 1)
    if ((event.get("before_state"), event.get("before_execution_count")) != expected_before
            or (event.get("after_state"), event.get("after_execution_count")) != ("confirmed", 1)):
        return False
    receipt = state.get("receipt")
    if (not isinstance(receipt, dict) or receipt.get("task_id") != event["task_id"]
            or type(receipt.get("cart_verified")) is not bool
            or config_digest(receipt) != event.get("receipt_sha256")):
        return False
    ledger_ids = event.get("ledger_event_ids")
    ledger = {e["event_id"]: e for e in row.get("action_ledger_events", []) if isinstance(e, dict) and "event_id" in e}
    if (not isinstance(ledger_ids, list) or not ledger_ids or any(type(i) is not int for i in ledger_ids)
            or len(set(ledger_ids)) != len(ledger_ids) or any(i not in ledger for i in ledger_ids)):
        return False
    linked = [ledger[i] for i in ledger_ids]
    if any(e.get("action_id") != event["action_id"] for e in linked):
        return False
    confirmations = [e for e in linked if e.get("event") == "confirmed"]
    entries = [e for e in linked if e.get("event") == "execute_entered"]
    registrations = [e for e in linked if e.get("event") == "registered"]
    if len(confirmations) != 1 or len(entries) != 1 or len(registrations) != 1:
        return False
    confirmation, entry, registration = confirmations[0], entries[0], registrations[0]
    details = confirmation.get("details", {})
    if (config_digest(details.get("receipt")) != event["receipt_sha256"]
            or not details.get("delivery_id") or details["delivery_id"] != entry.get("details", {}).get("delivery_id")
            or entry.get("details", {}).get("mode") != "guarded"
            or not registration["event_id"] < entry["event_id"] < confirmation["event_id"]
            or registration.get("details", {}).get("params_sha256") != state.get("params_sha256")
            or any(e.get("event") == "execution_outcome_unknown" for e in linked)):
        return False
    if kind == "duplicate_prevention" and not any(
            e.get("event") == "confirmed_receipt_replayed" and e["event_id"] > confirmation["event_id"] for e in linked):
        return False
    acknowledgments = [ack for e in row.get("events", []) if e.get("role") == "ActionExecutor"
                       for ack in e.get("output", []) if isinstance(ack, dict)]
    if not any(a.get("action_id") == event["action_id"] and config_digest(a.get("receipt")) == event["receipt_sha256"]
               for a in acknowledgments):
        return False
    indices = event.get("receipt_indices")
    http = {r["receipt_index"]: r for r in row.get("http_receipts", []) if isinstance(r, dict) and "receipt_index" in r}
    if (not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices)) or indices != receipt.get("http_receipt_indices")
            or any(i not in http for i in indices)):
        return False
    added = event.get("new_http_receipt_indices")
    if added != (indices if kind == "request_redelivery" else []):
        return False
    traces = [http[i] for i in indices]
    if any(type(r.get("status_code")) is not int or not 200 <= r["status_code"] < 300
           or r.get("error_type") or not isinstance(r.get("response_sha256"), str)
           or len(r["response_sha256"]) != 64 for r in traces):
        return False
    writes = [r for r in traces if r.get("request_method") in {"POST", "PUT"}]
    if len(writes) != 1:
        return False
    write = writes[0]
    params = state.get("params", {})
    expected_method = {"add_quantity": "POST", "set_quantity": "PUT"}.get(params.get("operation"))
    request_item = (write.get("request_payload") or {}).get("cartItem", {})
    if (write.get("request_method") != expected_method or request_item.get("sku") != receipt.get("sku")
            or request_item.get("qty") != params.get("quantity") or not write.get("guest_cart_id_sha256")):
        return False
    readbacks = [r for r in traces if r.get("request_method") == "GET" and r["receipt_index"] > write["receipt_index"]
                 and isinstance(r.get("response_payload"), list)]
    if len(readbacks) != 1 or readbacks[0].get("guest_cart_id_sha256") != write["guest_cart_id_sha256"]:
        return False
    matches = [item for item in readbacks[0]["response_payload"] if isinstance(item, dict) and item.get("sku") == receipt.get("sku")]
    return bool(len(matches) == 1 and matches[0].get("name") == receipt.get("product_title")
                and type(matches[0].get("qty")) in {int, float}
                and matches[0]["qty"] == receipt.get("observed_quantity"))


def verify_action_consequences(row):
    """Return ledger/HTTP-backed execution consequences without using fault labels.

    Failed delegation requires a bound, valid action with an explicitly empty
    delivered input and a registered, unexecuted pending ledger record. Duplicate
    execution requires distinct confirmed deliveries and distinct successful
    adjustment writes for the same parameters, cart, and product, excluding setup.
    """
    consequences = set()
    task, state = row.get("task", {}), row.get("action_ledger_state")
    if not isinstance(state, dict) or state.get("task_id") != task.get("task_id"):
        return consequences
    binding = {key: state.get(key) for key in ("task_id", "session_id", "action_id")}
    if any(type(value) is not str or not value for value in binding.values()):
        return consequences
    if any(row.get(key) != binding[key] for key in ("session_id", "action_id")):
        return consequences
    params = state.get("params")
    if not isinstance(params, dict) or any(params.get(key) != value for key, value in binding.items()):
        return consequences
    initial, target = task.get("initial_quantity"), task.get("quantity")
    if type(initial) is not int or type(target) is not int:
        return consequences
    operation = "add_quantity" if target > initial else "set_quantity"
    quantity = target - initial if operation == "add_quantity" else target
    if not validate_action(params, task, operation, quantity):
        return consequences
    ledger = [e for e in row.get("action_ledger_events", [])
              if isinstance(e, dict) and e.get("action_id") == binding["action_id"]]
    registrations = [e for e in ledger if e.get("event") == "registered"
                     and e.get("details", {}).get("params_sha256") == state.get("params_sha256")]
    count = state.get("execution_count")
    if len(registrations) != 1 or type(count) is not int:
        return consequences
    executors = [e for e in row.get("events", []) if e.get("role") == "ActionExecutor"
                 and all(e.get(key) == value for key, value in binding.items())]
    executed = {"execute_entered", "confirmed", "observed_delivery_confirmed", "execution_outcome_unknown"}
    if (row.get("action_contract_valid") is True and state.get("state") == "pending" and count == 0
            and state.get("receipt") is None and not row.get("action_outcome_unknown")
            and not any(e.get("event") in executed for e in ledger)
            and len(executors) == 1 and executors[0].get("action_valid") is True
            and executors[0].get("input") == [] and executors[0].get("output") == []):
        consequences.add("failed_delegation")
    if count <= 1:
        return consequences
    http_rows = row.get("http_receipts", [])
    http = {r["receipt_index"]: r for r in http_rows if isinstance(r, dict) and type(r.get("receipt_index")) is int}
    if len(http) != len(http_rows):
        return consequences
    confirmed_writes = {}
    expected_method, expected_purpose = (("POST", "add_quantity.add_item") if operation == "add_quantity"
                                         else ("PUT", "set_quantity.set_item"))
    for confirmation in ledger:
        if confirmation.get("event") not in {"confirmed", "observed_delivery_confirmed"}:
            continue
        details = confirmation.get("details", {})
        delivery_id, receipt = details.get("delivery_id"), details.get("receipt")
        if not isinstance(delivery_id, str) or not delivery_id or not isinstance(receipt, dict):
            continue
        entries = [e for e in ledger if e.get("event") == "execute_entered"
                   and e.get("details", {}).get("delivery_id") == delivery_id]
        if (len(entries) != 1 or any(type(e.get("event_id")) is not int for e in (registrations[0], entries[0], confirmation))
                or not registrations[0]["event_id"] < entries[0]["event_id"] < confirmation["event_id"]
                or receipt.get("task_id") != binding["task_id"] or receipt.get("product_title") != task.get("product_title")
                or not receipt.get("sku") or not receipt.get("product_id")):
            continue
        indices = receipt.get("http_receipt_indices")
        if (not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices) or any(i not in http for i in indices)):
            continue
        traces = [http[i] for i in indices]
        writes = [r for r in traces if r.get("request_method") in {"POST", "PUT"}]
        if len(writes) != 1:
            continue
        write = writes[0]
        request_item = (write.get("request_payload") or {}).get("cartItem", {})
        response_item = write.get("response_payload") or {}
        if (write.get("request_method") != expected_method or write.get("purpose") != expected_purpose
                or type(write.get("status_code")) is not int or not 200 <= write["status_code"] < 300
                or write.get("error_type") or not isinstance(write.get("response_sha256"), str)
                or len(write["response_sha256"]) != 64 or not write.get("guest_cart_id_sha256")
                or request_item.get("sku") != receipt["sku"] or type(request_item.get("qty")) is not int
                or request_item["qty"] != quantity or not isinstance(response_item, dict)
                or response_item.get("sku") != receipt["sku"] or response_item.get("name") != task["product_title"]):
            continue
        identities = [r for r in traces if r.get("request_method") == "GET" and r["receipt_index"] < write["receipt_index"]
                      and isinstance(r.get("response_payload"), dict)
                      and r["response_payload"].get("sku") == receipt["sku"]
                      and r["response_payload"].get("product_id") == receipt["product_id"]]
        if not identities:
            continue
        product_cart = (write["guest_cart_id_sha256"], receipt["sku"], receipt["product_id"])
        confirmed_writes.setdefault(product_cart, {})[delivery_id] = write["receipt_index"]
    if any(len(deliveries) > 1 and len(set(deliveries.values())) > 1 for deliveries in confirmed_writes.values()):
        consequences.add("duplicate_execution")
    return consequences


def evaluate_trial(row):
    """Called only after execution; no policy can read the evaluator's environment."""
    result = copy.deepcopy(row)
    task, environment = row["task"], row["environment_state"]
    evidence, verdict = row.get("final_evidence"), row.get("final_verdict", {})
    environment_ok = bool(isinstance(environment, dict) and environment.get("cart_verified") is True
                          and environment.get("observed_quantity") == task["quantity"]
                          and environment.get("product_title") == task["product_title"])
    environment_known = (isinstance(environment, dict) and type(environment.get("cart_verified")) is bool
                         and type(environment.get("observed_quantity")) is int
                         and bool(environment.get("product_id")) and bool(environment.get("sku")))
    environment_known |= environment.get("status") == "no executed cart"
    grounded = semantic_success(task, evidence, environment)
    expected = "accept" if grounded else "reject"
    correct = (isinstance(verdict, dict) and verdict.get("task_id") == task["task_id"]
               and verdict.get("decision") == expected)
    m = verify_action_consequences(row)
    if environment_known and not environment_ok:
        m.add("task_incomplete")
    acceptance_errors = 0
    for judgment in row.get("judgments", []):
        jv, p = judgment.get("verdict", {}), judgment.get("accepted_payload")
        if not isinstance(jv, dict) or jv.get("decision") != "accept":
            continue
        checks = check_evidence(task, p)
        error = False
        if checks.missing_fields:
            m.add("M14")
            error = True
        if judgment.get("stale") or isinstance(p, dict) and p.get("task_id") != task["task_id"]:
            m.add("M5")
            error = True
        inconsistent = any(issue.startswith("inner_outer:") for issue in checks.issues)
        if isinstance(p, dict) and p.get("cart_verified") is True:
            inconsistent |= (type(p.get("observed_quantity")) is int
                             and type(p.get("requested_quantity")) is int
                             and p["observed_quantity"] != p["requested_quantity"])
        if inconsistent:
            m.add("M6")
            error = True
        if judgment.get("semantic_mismatch"):
            m.add("incorrect_verification")
            error = True
        acceptance_errors += int(error)
    if not correct and environment_known:
        m.add("M4")
    success = (environment_ok and grounded and correct and verdict.get("decision") == "accept"
               and row.get("final_commit_allowed", True))
    readback_recoveries = [e for e in row.get("recovery_events", [])
                          if e.get("verified") is True and e.get("replacement_used") is True and e.get("receipt_indices")
                          and e.get("before_sha256") != e.get("after_sha256")]
    action_events = result.setdefault("action_protocol_events", [])
    seen = set()
    for event in action_events:
        key = (event.get("kind"), event.get("action_id"), tuple(event.get("ledger_event_ids", [])))
        event["verified"] = key not in seen and verify_action_protocol_event(row, event)
        seen.add(key)
    action_recoveries = [e for e in action_events if e["verified"] and e["outcome"] == "recovery"]
    preventions = [e for e in action_events if e["verified"] and e["outcome"] == "prevention"]
    recovered = bool(readback_recoveries or action_recoveries)
    detected = bool(row.get("detection_events"))
    if m and not success:
        propagation = "propagated_to_M_final_failure"
    elif recovered:
        propagation = "detected_and_recovered"
    elif preventions:
        propagation = "detected_and_prevented"
    elif detected:
        propagation = "detected_but_unrecovered"
    elif m:
        propagation = "silent_propagation_to_M"
    else:
        propagation = "A_only" if row.get("fault_events") else "clean"
    result.update(environment_task_success=environment_ok if environment_known else None,
                  decision_correct=correct if environment_known else None,
                  final_task_success=bool(success and environment_known), task_score=int(environment_ok) if environment_known else None,
                  evidence_acceptance_errors=acceptance_errors, observed_M_consequence=sorted(m),
                  observed_A_symptom=[e["condition"] for e in row.get("fault_events", [])],
                  recovery_detected=recovered, recovery_type=sorted({e.get("kind", "unknown")
                    for e in readback_recoveries + action_recoveries}),
                  action_recovery_count=len(action_recoveries), action_prevention_count=len(preventions),
                  prevention_detected=bool(preventions), prevention_type=sorted({e["kind"] for e in preventions}),
                  propagation_class=propagation)
    return result


@dataclass
class AgentRequest:
    prompt: str


@dataclass
class AgentReply:
    value: dict


class ProtocolAgent(RoutedAgent):
    def __init__(self, client, role):
        super().__init__(role)
        self.client, self.role = client, role

    @message_handler
    async def respond(self, message: AgentRequest, ctx) -> AgentReply:
        raw = self.client.complete(f"You are the {self.role} in a real Shopping workflow. " + message.prompt,
                                   json_mode=True)
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = {"parse_error": "invalid_json"}
        if not isinstance(value, dict):
            value = {"parse_error": "not_object"}
        return AgentReply(value)


async def run_trial(task, job, executor, client, *, cross_task_evidence=None, ledger_path):
    """Run actual AutoGen agents and the supplied real HTTP executor.

    The CLI admits only the local model. Unit tests may supply deterministic
    transports but there is no simulated experiment branch in this function.
    """
    arm, topology = job["arm"], job["topology"]
    if arm not in {'baseline', 'action_only', 'semantic_only', 'combined'} or topology not in {'sequential', 'flat'}:
        raise ValueError("unsupported strategy or topology")
    session = uuid.uuid4().hex
    run_id = "mm-" + session
    switches = mechanisms(arm)
    dep = independent = False
    action_guard = switches['action']
    semantic_guard = switches['semantic']
    budget = Budget(max_replays=2)
    receiver = None
    path_design = job.get('path_design')
    graph = EvidenceGraph(task["task_id"], session)
    ledger = ActionLedger(ledger_path)
    runtime = SingleThreadedAgentRuntime()
    verifier_role = "Supervisor" if topology == "hierarchical" else "Verifier"
    decision_role = "Supervisor" if topology == "hierarchical" else "Coordinator"
    roles = ("Worker", verifier_role, decision_role, "EvidencePeer")
    for role in dict.fromkeys(roles):
        await ProtocolAgent.register(runtime, role, lambda role=role: ProtocolAgent(client, role))
    events, judgments, detections, recoveries = [], [], [], []
    action_protocol_events = []
    graph_errors = []
    action_unknown = False
    fault = None
    model_start = len(client.request_log)
    initial_task = {**task, "quantity": task["initial_quantity"]}
    current_action_id = "adjust-" + session
    entity = "cart-" + session

    async def ask(role, prompt, inputs, *, extra=False):
        if extra:
            budget.consume("model_call")
        request = AgentRequest(prompt + "\nInput: " + json.dumps(inputs, ensure_ascii=False))
        response = await runtime.send_message(request, AgentId(role, session))
        events.append({"event_id": len(events), "role": role, "input": copy.deepcopy(inputs),
                       "output": copy.deepcopy(response.value), "extra_model_call": extra})
        return response.value

    def envelope(p, label, version, action_id, source="Worker"):
        return make_envelope(p, task_id=task["task_id"], session_id=session, entity_id=entity,
                             version=version, action_id=action_id, evidence_id=label + "-" + session,
                             source=source)

    def register(e):
        if not isinstance(e, dict):
            return False
        try:
            graph.add_evidence(**e)
            return True
        except ValueError as exc:
            graph_errors.append({"evidence_id": e.get("evidence_id"), "error_type": type(exc).__name__})
            return False

    def dependency_valid(judgment):
        if not isinstance(judgment, dict):
            return False
        try:
            return graph.accepted(judgment["judgment_id"])
        except (ValueError, KeyError):
            return False
    async def worker_evidence(observations, expected, label):
        answer = await ask("Worker", "Select the current observation for the task. Do not invent missing evidence. "
                           'Return {"selected_evidence_id":string|null,"payload":object|null}; payload must preserve '
                           "task_id, product_title, product_id, sku, requested_quantity, observed_quantity, "
                           "cart_verified and evidence if present. Older observations do not override newer states.",
                           {"task": expected, "observations": observations})
        # Keep the model event, but absence of an observation is not evidence.
        if not observations:
            return None
        selected = next((e for e in observations if isinstance(e, dict)
                         and e.get("evidence_id") == answer.get("selected_evidence_id")), None)
        # A single input has unambiguous runtime provenance, even if the LLM omits its ID.
        if selected is None and len(observations) == 1:
            selected = observations[0]
        payload = answer.get("payload")
        result = envelope(payload if isinstance(payload, dict) else {}, label,
                          selected.get("version") if selected else None,
                          selected.get("action_id") if selected else None)
        if selected is not None:
            result["task_id"], result["session_id"] = selected["task_id"], selected["session_id"]
        return result

    async def judge(e, expected, label, *, parent=None, extra=False, role=None):
        role = role or verifier_role
        p = e.get("payload") if isinstance(e, dict) else None
        verdict = await ask(role, 'Return {"task_id":string,"decision":"accept"|"reject",'
                            '"reason":string,"evidence_ids":[string],"judgment_ids":[string]}. Accept only complete, current evidence '
                            "of the exact requested product and quantity. A prior judgment is not a current observation.",
                            {"task": expected, "evidence": e, "prior_judgment": parent}, extra=extra)
        record = {"judgment_id": label + "-" + session,
                  "evidence_ids": [e["evidence_id"]] if isinstance(e, dict) else [],
                  "verdict": verdict}
        judgments.append({**record, "role": role, "accepted_payload": copy.deepcopy(p),
                          "expected_quantity": expected["quantity"],
                          "stage": "initial" if label == "judgment-initial" else "current",
                          "input_parent_judgment": copy.deepcopy(parent),
                          "stale": bool(expected["quantity"] == task["quantity"] and isinstance(e, dict)
                                        and e.get("action_id") != current_action_id)})
        if e is None or register(e):
            cites = verdict.get("evidence_ids", [])
            try:
                graph.add_judgment(record["judgment_id"], provided_evidence_ids=record["evidence_ids"],
                                   cited_evidence_ids=cites if isinstance(cites, list) else ["invalid-citation"],
                                   parent_judgment_ids=[parent["judgment_id"]] if parent else [],
                                   verdict=verdict, status="unresolved" if e is None else "valid")
            except ValueError:
                graph_errors.append({"judgment_id": record["judgment_id"], "error_type": "invalid_dependency"})
        return record

    def readback(kind, before, *, common=False):
        if not common:
            budget.consume("get", 2)
        start = len(executor.http_receipts)
        try:
            p = executor.reobserve_cart(task)
        except (RequestException, TimeoutError, ValueError) as exc:
            p = {"task_id": task["task_id"], "status": "observation_unavailable", "error_type": type(exc).__name__}
        indices = list(range(start, len(executor.http_receipts)))
        fresh = envelope(p, f"readback-{len(recoveries)}", 3 + len(recoveries), current_action_id,
                         source="EnvironmentReadback")
        delivered = fault.deliver('evidence_handoff', fresh, recovery=True)
        fresh = delivered[-1] if delivered else None
        recoveries.append({"kind": kind, "common": common, "receipt_indices": indices,
                           "before_sha256": evidence_state_digest(before), "after_sha256": evidence_state_digest(fresh),
                           "verified": bool(indices and isinstance(fresh, dict)
                                            and check_evidence(task, fresh.get('payload'), require_success=False).valid),
                           "replacement_used": True,
                           "before": copy.deepcopy(before), "after": copy.deepcopy(fresh)})
        return fresh

    def record_action_protocol(kind, before, after, new_indices):
        receipt = after.get("receipt") or {}
        action_protocol_events.append({
            "schema_version": 1, "event_id": len(action_protocol_events), "kind": kind,
            "outcome": "prevention" if kind == "duplicate_prevention" else "recovery",
            "task_id": task["task_id"], "session_id": session, "action_id": current_action_id,
            "before_state": before["state"], "before_execution_count": before["execution_count"],
            "after_state": after["state"], "after_execution_count": after["execution_count"],
            "ledger_event_ids": [e["event_id"] for e in ledger.events(current_action_id)],
            "receipt_sha256": config_digest(receipt), "receipt_indices": list(receipt.get("http_receipt_indices", [])),
            "new_http_receipt_indices": list(new_indices), "response_used": True,
        })

    runtime.start()
    try:
        first_action = await ask("Worker", 'Return {"task_id":string,"operation":"add_to_cart","quantity":integer}. '
                                 "Execute the initial stage specified in the task.", {"task": initial_task})
        initial_ok = validate_action(first_action, task, "add_to_cart", initial_task["quantity"])
        if initial_ok:
            executor.add_to_cart(initial_task)
        initial_observation = envelope(executor.reobserve_cart(initial_task), "observation-initial", 1, "initial-" + session)
        initial_e = await worker_evidence([initial_observation], initial_task, "worker-initial")
        old_judgment = await judge(initial_e, initial_task, "judgment-initial")
        fault = FiniteExposure(job["condition"], count=job['planned_exposures'],
                               expose_recovery=job['recovery_path_exposed'],
                               old_evidence=initial_observation, old_judgment=old_judgment,
                               cross_task_evidence=cross_task_evidence)
        # The initial live observation is available to all agents before injection.
        # Pin identity only from that observation, never from a clean-run answer.
        receiver = ReceiverContract(task, session_id=session, action_id=current_action_id,
                                    entity_id=entity, minimum_version=2,
                                    known_identity={k: initial_observation['payload'][k]
                                                    for k in ('sku', 'product_id')
                                                    if initial_observation['payload'].get(k)},
                                    identity_provenance='pre_action_observation', max_readbacks=1)
        operation = "add_quantity" if task["quantity"] > task["initial_quantity"] else "set_quantity"
        quantity = task["quantity"] - task["initial_quantity"] if operation == "add_quantity" else task["quantity"]
        proposal = await ask("Worker", 'Return {"task_id":string,"operation":string,"quantity":integer}. '
                            "For add_quantity, quantity is the increment; for set_quantity it is the absolute target. "
                            "Use the specified operation and argument.",
                            {"task": task, "operation": operation, "argument": quantity})
        action_valid = initial_ok and validate_action(proposal, task, operation, quantity)
        request = {"task_id": task["task_id"], "session_id": session,
                   "action_id": current_action_id, "operation": proposal.get("operation"),
                   "quantity": proposal.get("quantity")}
        delivered_requests = fault.deliver("action_request", request)
        if action_valid:
            ledger.register(task["task_id"], session, current_action_id, request)
        redelivery_before = None
        replay_limit = 2
        replay_count = 0
        while action_guard and action_valid and not delivered_requests and replay_count < replay_limit:
            # The persistent ledger proves that this action never entered execution.
            state = ledger.get(current_action_id)
            if state["state"] == "pending" and state["execution_count"] == 0:
                budget.consume("replay")
                replay_count += 1
                detections.append({"kind": "missing_action_receipt", "action_id": current_action_id})
                redelivery_before = state
                delivered_requests = fault.deliver("action_request", request, recovery=True)
            else:
                break
        acknowledgments = []
        for received in delivered_requests:
            if not action_valid or not validate_action(received, task, operation, quantity):
                continue
            execute = lambda: getattr(executor, operation)(task, quantity)
            before_action = ledger.get(current_action_id)
            operation_start = len(executor.http_receipts)
            try:
                operation_wrapper = ledger.execute_once if action_guard else ledger.execute_observed
                result = operation_wrapper(task["task_id"], session, current_action_id, request, execute)
                receipt = result["receipt"]
            except (RequestException, TimeoutError, ValueError, ActionUnresolved) as exc:
                action_unknown = True
                detections.append({"kind": "action_outcome_unknown", "action_id": current_action_id,
                                   "error_type": type(exc).__name__})
                break
            acknowledgments.extend(fault.deliver("action_ack", {"action_id": current_action_id, "receipt": receipt}))
            if action_guard and result["replayed"]:
                detections.append({"kind": "duplicate_prevented", "action_id": current_action_id})
                record_action_protocol("duplicate_prevention", before_action, result,
                                       range(operation_start, len(executor.http_receipts)))
            elif redelivery_before is not None and acknowledgments:
                record_action_protocol("request_redelivery", redelivery_before, result,
                                       range(operation_start, len(executor.http_receipts)))
                redelivery_before = None
        if action_guard and not acknowledgments and action_valid:
            state = ledger.get(current_action_id)
            if state and state["state"] == "confirmed":
                detections.append({"kind": "confirmation_retrieved", "action_id": current_action_id})
                acknowledgments = [{"action_id": current_action_id, "receipt": state["receipt"]}]
                record_action_protocol("acknowledgement_retrieval", state, state, [])
        events.append({"event_id": len(events), "role": "ActionExecutor", "input": delivered_requests,
                       "task_id": task["task_id"], "session_id": session, "action_id": current_action_id,
                       "output": copy.deepcopy(acknowledgments), "action_valid": action_valid})
        current_observation = envelope(executor.reobserve_cart(task), "observation-current", 2, current_action_id) if acknowledgments else None
        observations = fault.deliver("observation_handoff", current_observation) if current_observation else []
        current_e = await worker_evidence(observations, task, "worker-current")
        handed = fault.deliver("evidence_handoff", current_e) if current_e is not None else []
        primary = handed[-1] if handed else None
        if semantic_guard:
            before = primary
            primary = receiver.accept(primary, lambda: readback('semantic_contract', before))
            detections.extend({'kind': 'semantic_contract', **event} for event in receiver.events[-1:]
                              if event['before_issues'])
            if recoveries and receiver.events[-1]['readback_called']:
                recoveries[-1]['replacement_used'] = primary is not None
                recoveries[-1]['verified'] = primary is not None
        register(primary)
        checks = check_evidence(task, primary.get("payload") if isinstance(primary, dict) else None)
        binding_bad = (not isinstance(primary, dict) or primary.get("task_id") != task["task_id"]
                       or primary.get("session_id") != session or primary.get("action_id") != current_action_id)
        if arm in {"always_recheck", "action_always"}:
            primary = readback("fixed_readback", primary)
        elif arm == "guarded_recheck" and not checks.valid:
            detections.append({"kind": "contract_check", "issues": list(checks.issues)})
            primary = readback("guarded_readback", primary)
        elif dep and (not checks.valid or binding_bad):
            detections.append({"kind": "evidence_obligation", "issues": list(checks.issues), "binding_bad": binding_bad})
            primary = readback("dependency_readback", primary)
        if independent:
            fresh = readback("independent_observation", primary)
            register(fresh)
            recoveries[-1]["replacement_used"] = False
            comparison = ("task_id", "product_id", "sku", "observed_quantity", "cart_verified")
            before = primary.get("payload", {}) if isinstance(primary, dict) else {}
            if any(before.get(k) != fresh["payload"].get(k) for k in comparison) or not check_evidence(task, before).valid:
                detections.append({"kind": "independent_disagreement"})
                candidates = [e for e in (primary, fresh) if isinstance(e, dict)]
                resolution = await ask("EvidencePeer" if topology == "flat" else verifier_role,
                                       'Compare the candidate observations against the task, '
                                       'their action bindings, versions and sources. Return {"selected_evidence_id":string|null,"reason":string}. '
                                       "Select a supported current observation, or null if unresolved. Do not vote or invent state.",
                                       {"task": task, "candidate_observations": candidates}, extra=True)
                primary = next((e for e in candidates if e["evidence_id"] == resolution.get("selected_evidence_id")), None)
                recoveries[-1]["replacement_used"] = primary is fresh
        current_judgment = await judge(primary, task, "judgment-current")
        reports = fault.deliver("judgment_handoff", current_judgment)
        report = reports[-1] if reports else None
        if dep and report is not None and not dependency_valid(report):
            detections.append({"kind": "invalidated_judgment", "judgment_id": report["judgment_id"]})
            try:
                live = graph.context_snapshot()["evidence"].values()
                candidates = [e for e in live if e.get("action_id") == current_action_id
                              and check_evidence(task, e["payload"], require_success=False).valid]
                if candidates:
                    chosen = max(candidates, key=lambda e: e["version"] if type(e["version"]) is int else -1)
                    primary = {k: v for k, v in chosen.items() if k != "status"}
                    for recovery in recoveries:
                        if recovery.get("after", {}).get("evidence_id") == primary["evidence_id"]:
                            recovery["replacement_used"] = True
                report = await judge(primary, task, "judgment-recomputed", extra=True)
            except BudgetExhausted:
                report = None
            if not dependency_valid(report):
                report = None
        final_e = primary
        if topology == "flat" or path_design in {'duplicate_forwarding', 'independent_observation'}:
            # Existing structural bypass is available in every strategy arm.
            direct = (copy.deepcopy(primary) if path_design == 'duplicate_forwarding' else
                      envelope(executor.reobserve_cart(task), "direct-peer", 4, current_action_id, source="EvidencePeer"))
            peer = await ask("EvidencePeer", 'Return {"payload":object}. Preserve the exact independent observation; '
                             "do not fill missing fields.", {"task": task, "observation": direct})
            final_e = {**direct, "payload": peer.get("payload", {})} if isinstance(direct, dict) else None
        if semantic_guard:
            before = final_e
            final_e = receiver.accept(final_e, lambda: readback('semantic_final_contract', before))
            detections.extend({'kind': 'semantic_contract', **event} for event in receiver.events[-1:]
                              if event['before_issues'])
        final = await judge(final_e, task, "judgment-final", parent=report, role=decision_role)
        final_verdict = final["verdict"]
        if (final_verdict.get("decision") == "reject" or final_verdict.get("task_id") != task["task_id"]):
            detections.append({"kind": "common_recovery_trigger"})
            final_e = readback("common_recovery", final_e, common=True)
            if semantic_guard:
                issues = receiver.issues(final_e)
                receiver.events.append({'before_issues': list(issues), 'after_issues': list(issues),
                                        'readback_called': False, 'accepted': not issues,
                                        'stage': 'common_recovery'})
                if issues:
                    final_e = None
            final = await judge(final_e, task, "judgment-common-recovery", role=decision_role)
            final_verdict = final["verdict"]
        commit_allowed = True
        if dep and not dependency_valid(final):
            detections.append({"kind": "decision_commit_blocked", "judgment_id": final["judgment_id"]})
            try:
                final = await judge(final_e, task, "judgment-commit-recheck", role=decision_role, extra=True)
                final_verdict = final["verdict"]
            except BudgetExhausted:
                pass
            commit_allowed = dependency_valid(final)
        # This read is offline evaluation only: it happens after all model decisions.
        evaluation_start = len(executor.http_receipts)
        executor.begin_evaluation()
        try:
            truth = executor.reobserve_cart(task)
        except (RequestException, TimeoutError, ValueError) as exc:
            truth = {"status": "observation_unavailable", "error_type": type(exc).__name__}
        for record in judgments:
            p = record.get("accepted_payload")
            expected = {**task, "quantity": record["expected_quantity"]}
            reference = initial_observation["payload"] if record["stage"] == "initial" else truth
            reference_available = type(reference.get("observed_quantity")) is int and bool(reference.get("sku"))
            record["semantic_reference_available"] = reference_available
            record["semantic_reference"] = {k: reference.get(k) for k in ("product_id", "sku", "observed_quantity")}
            record["semantic_mismatch"] = bool(
                not check_evidence(expected, p).valid
                or reference_available and isinstance(p, dict)
                and any(p.get(k) != reference.get(k) for k in ("product_id", "sku", "observed_quantity")))
            parent = record.get("input_parent_judgment")
            if parent and parent["judgment_id"] == old_judgment["judgment_id"]:
                cites = record["verdict"].get("evidence_ids", [])
                j_cites = record["verdict"].get("judgment_ids", [])
                record["stale"] |= (isinstance(cites, list) and bool(set(old_judgment["evidence_ids"]) & {c for c in cites if isinstance(c, str)})
                                     or isinstance(j_cites, list) and parent["judgment_id"] in j_cites)
        usage = client.request_log[model_start:]
        token_usage, known_usage = {}, {}
        for field in ("prompt_tokens", "completion_tokens"):
            counts = [r.get(field) for r in usage]
            valid = [value for value in counts if type(value) is int and value >= 0]
            known_usage["known_" + field] = sum(valid)
            token_usage[field] = sum(valid) if len(valid) == len(counts) else None
        known_usage["known_total_tokens"] = sum(known_usage.values())
        token_usage["total_tokens"] = (sum(token_usage.values())
                                       if all(value is not None for value in token_usage.values()) else None)
        row = {**job, "run_id": run_id, "task": copy.deepcopy(task),
               "session_id": session, "action_id": current_action_id,
               "model": client.model_info.model, "provider": client.model_info.provider,
               "model_seed": None, "events": events, "judgments": judgments,
               "final_evidence": final_e.get("payload") if isinstance(final_e, dict) else None,
               "final_verdict": final_verdict, "environment_state": truth,
               "final_commit_allowed": commit_allowed,
               "fault_events": fault.events, "detection_events": detections, "recovery_events": recoveries,
               "graph": graph.snapshot(), "graph_errors": graph_errors, "budget": budget.snapshot(),
               "action_ledger_events": ledger.events(current_action_id),
               "action_ledger_state": ledger.get(current_action_id), "action_protocol_events": action_protocol_events,
               "http_receipts": copy.deepcopy(executor.http_receipts),
               "evaluation_receipt_indices": list(range(evaluation_start, len(executor.http_receipts))),
               "source_evidence": current_e, "common_recovery_enabled": True,
               "rq4_mechanisms": switches, "path_design": path_design,
               "semantic_contract_events": receiver.events,
               "exposure_deliveries": fault.deliveries,
               "llm_request_log": usage, "action_contract_valid": action_valid,
               "action_outcome_unknown": action_unknown,
               "token_usage": token_usage, **known_usage}
        return evaluate_trial(row)
    except Exception as exc:
        exc.partial_trial = {"events": copy.deepcopy(events), "judgments": copy.deepcopy(judgments),
                             "fault_events": copy.deepcopy(fault.events) if fault else [],
                             "exposure_deliveries": copy.deepcopy(fault.deliveries) if fault else [],
                             "semantic_contract_events": copy.deepcopy(receiver.events) if receiver else [],
                             "detection_events": copy.deepcopy(detections), "recovery_events": copy.deepcopy(recoveries),
                             "action_protocol_events": copy.deepcopy(action_protocol_events),
                             "action_ledger_state": ledger.get(current_action_id),
                             "action_ledger_events": ledger.events(current_action_id),
                             "http_receipts": copy.deepcopy(executor.http_receipts),
                             "budget": budget.snapshot(), "graph_errors": copy.deepcopy(graph_errors)}
        raise
    finally:
        await runtime.stop()
