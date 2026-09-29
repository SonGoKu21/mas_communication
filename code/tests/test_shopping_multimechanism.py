import json
import pytest

from mas_faults.shopping_multimechanism import make_envelope, semantic_success, evaluate_trial, validate_action


def test_real_runner_entrypoint_is_async_and_has_no_simulation_flag():
    import inspect
    from mas_faults.shopping_multimechanism import run_trial
    assert inspect.iscoroutinefunction(run_trial)
    assert "mock" not in inspect.signature(run_trial).parameters


def task():
    return {"task_id": "t", "product_title": "Product", "quantity": 2, "initial_quantity": 1}


def payload(q=2, sku="SKU"):
    p = {"task_id": "t", "product_title": "Product", "product_id": "12", "sku": sku,
         "requested_quantity": 2, "observed_quantity": q, "cart_verified": q == 2}
    return {**p, "evidence": json.dumps(p)}


def test_envelope_binding_owned_by_runtime_and_no_oracle_fields():
    e = make_envelope(payload(), task_id="t", session_id="s", entity_id="cart", version=2,
                      action_id="a", evidence_id="e", source="worker")
    assert e["payload"] == payload()
    assert e["task_id"] == "t"
    assert set(e) == {"payload", "task_id", "session_id", "entity_id", "version", "action_id", "evidence_id", "source"}


def test_action_not_silently_replaced_with_task_truth():
    good = {"task_id": "t", "operation": "add_quantity", "quantity": 1}
    assert validate_action(good, task(), "add_quantity", 1)
    assert not validate_action({**good, "quantity": 2}, task(), "add_quantity", 1)
    assert not validate_action({**good, "quantity": True}, task(), "add_quantity", 1)
    assert not validate_action({**good, "task_id": "other"}, task(), "add_quantity", 1)


def test_evaluation_is_semantic_not_missing_protocol_ids():
    assert semantic_success(task(), payload(), payload())
    assert not semantic_success(task(), payload(sku="wrong"), payload())
    partial = {"task_id": "t", "product_title": "Product", "requested_quantity": 2, "cart_verified": True}
    assert not semantic_success(task(), partial, payload())


def test_intermediate_partial_acceptance_retained_after_final_recovery():
    partial = {"task_id": "t", "cart_verified": True}
    row = {"task": task(), "environment_state": payload(), "final_evidence": payload(),
           "final_verdict": {"task_id": "t", "decision": "accept"},
           "judgments": [{"verdict": {"decision": "accept", "task_id": "t"},
                          "accepted_payload": partial, "stale": False}],
           "recovery_events": [], "fault_events": [{"condition": "valid_partial"}]}
    evaluated = evaluate_trial(row)
    assert evaluated["environment_task_success"]
    assert evaluated["final_task_success"]
    assert "M14" in evaluated["observed_M_consequence"]
    assert "M4" not in evaluated["observed_M_consequence"]


def test_safe_rejection_does_not_count_as_task_success():
    row = {"task": task(), "environment_state": payload(3), "final_evidence": payload(3),
           "final_verdict": {"task_id": "t", "decision": "reject"}, "judgments": [],
           "recovery_events": [], "fault_events": []}
    evaluated = evaluate_trial(row)
    assert evaluated["decision_correct"]
    assert not evaluated["environment_task_success"]
    assert not evaluated["final_task_success"]
    assert "M4" not in evaluated["observed_M_consequence"]


def test_wrong_acceptance_m4_does_not_require_explicit_reject():
    row = {"task": task(), "environment_state": payload(3), "final_evidence": payload(2),
           "final_verdict": {"task_id": "t", "decision": "accept"}, "judgments": [],
           "recovery_events": [], "fault_events": [{"condition": "contract_consistent_identity_corruption"}]}
    evaluated = evaluate_trial(row)
    assert not evaluated["decision_correct"]
    assert "M4" in evaluated["observed_M_consequence"]


class UnitClient:
    """Scripted transport for integration tests, never a runnable experiment mode."""
    def __init__(self):
        from types import SimpleNamespace
        self.model_info = SimpleNamespace(model="unit-only", provider="unit-only")
        self.request_log = []

    def complete(self, prompt, *, json_mode=False):
        from mas_faults.shopping_mitigation import check_evidence
        inputs = json.loads(prompt.split("\nInput: ", 1)[1])
        t = inputs["task"]
        if "observations" in inputs:
            obs = inputs["observations"]
            selected = max(obs, key=lambda e: e.get("version") or 0) if obs else None
            result = {"selected_evidence_id": selected["evidence_id"] if selected else None,
                      "payload": selected["payload"] if selected else None}
        elif "candidate_observations" in inputs:
            result = {"selected_evidence_id": inputs["candidate_observations"][-1]["evidence_id"]}
        elif "prior_judgment" in inputs:
            e = inputs["evidence"]
            p = e.get("payload") if e else None
            result = {"task_id": t["task_id"], "decision": "accept" if check_evidence(t, p).valid else "reject",
                      "reason": "unit transport", "evidence_ids": [e["evidence_id"]] if e else []}
        elif "observation" in inputs:
            result = {"payload": inputs["observation"]["payload"]}
        else:
            result = {"task_id": t["task_id"], "operation": inputs.get("operation", "add_to_cart"),
                      "quantity": inputs.get("argument", t["quantity"])}
        self.request_log.append({"prompt_tokens": 10, "completion_tokens": 2})
        return json.dumps(result)


class UnitExecutor:
    """In-memory unit transport, not evidence of actual Magento behavior."""
    def __init__(self):
        self.q = 0
        self.http_receipts = []

    def add_to_cart(self, t):
        self.q = t["quantity"]
        return self.reobserve_cart(t)

    def add_quantity(self, t, q):
        self.q += q
        return self.reobserve_cart({**t, "quantity": self.q})

    def set_quantity(self, t, q):
        self.q = q
        return self.reobserve_cart({**t, "quantity": q})

    def reobserve_cart(self, t):
        for purpose in ("product_page", "items"):
            self.http_receipts.append({"receipt_index": len(self.http_receipts),
                                       "purpose": purpose, "request_method": "GET", "status_code": 200})
        p = {**payload(self.q), "requested_quantity": t["quantity"], "cart_verified": self.q == t["quantity"]}
        p["evidence"] = json.dumps({k: v for k, v in p.items() if k != "evidence"})
        return p


@pytest.mark.parametrize("usage", [{"prompt_tokens": None, "completion_tokens": 2},
    {"completion_tokens": 2}, {"prompt_tokens": True, "completion_tokens": 2},
    {"prompt_tokens": -1, "completion_tokens": 2}])
def test_runtime_token_aggregation_preserves_unknown_and_known_counts(usage, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs

    class NullableClient(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = super().complete(prompt, json_mode=json_mode)
            self.request_log[-1] = dict(usage)
            return result

    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = next(j for j in build_jobs([t], 1) if j["arm"] == "baseline"
               and j["condition"] == "clean" and j["topology"] == "sequential")
    result = asyncio.run(run_trial(t, job, UnitExecutor(), NullableClient(),
                                   ledger_path=tmp_path / "ledger.sqlite"))
    assert result["final_task_success"] is True
    assert result["token_usage"] == {"prompt_tokens": None,
        "completion_tokens": 2 * len(result["llm_request_log"]), "total_tokens": None}
    assert result["known_prompt_tokens"] == 0
    assert result["known_completion_tokens"] == result["known_total_tokens"] == 2 * len(result["llm_request_log"])


@pytest.mark.parametrize("arm", ["baseline", "always_recheck", "guarded_recheck", "dependency",
                                  "independent", "action_protocol", "combined"])
@pytest.mark.parametrize("condition", ["clean", "request_non_delivery", "acknowledgement_loss",
    "duplicate_action_delivery", "valid_partial", "same_session_reordering", "cross_task_replay",
    "stale_judgment_replay", "conflicting_observation", "contract_consistent_identity_corruption"])
@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
def test_unit_all_arms_cross_all_conditions_execute_real_autogen_runtime(arm, condition, topology, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs
    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = next(j for j in build_jobs([t], 1) if j["arm"] == arm and j["condition"] == condition
               and j["topology"] == topology)
    source = make_envelope({**payload(sku="OTHER"), "task_id": "other", "product_id": "44"},
        task_id="other", session_id="other", entity_id="other", version=1,
        action_id="other", evidence_id="other", source="unit-fixture")
    result = asyncio.run(run_trial(t, job, UnitExecutor(), UnitClient(),
                                   cross_task_evidence=source, ledger_path=tmp_path / "ledger.sqlite"))
    assert len(result["fault_events"]) == (0 if condition == "clean" else 1)
    assert result["common_recovery_enabled"]
    assert result["token_usage"]["total_tokens"] == 12 * len(result["llm_request_log"])
    assert result["budget"]["used"]["get"] <= 4
    assert result["budget"]["used"]["model_call"] <= 3
    if condition == "clean":
        assert result["final_task_success"]
        assert not result["recovery_detected"]
        if topology == "hierarchical":
            assert {j["role"] for j in result["judgments"]} == {"Supervisor"}
    if condition == "duplicate_action_delivery":
        assert result["environment_task_success"] is (arm in {"action_protocol", "combined"})
    if arm in {"independent", "combined"} and condition == "contract_consistent_identity_corruption":
        expected_role = {"sequential": "Verifier", "hierarchical": "Supervisor", "flat": "EvidencePeer"}[topology]
        assert any(e.get("role") == expected_role and e.get("extra_model_call")
                   for e in result["events"])


def test_invalid_dependency_citations_do_not_crash_or_commit(tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs

    class UnknownCitationClient(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = json.loads(super().complete(prompt, json_mode=json_mode))
            if "evidence_ids" in result:
                result["evidence_ids"] = ["never-observed"]
            return json.dumps(result)

    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = next(j for j in build_jobs([t], 1) if j["arm"] == "dependency" and j["condition"] == "clean"
               and j["topology"] == "sequential")
    row = asyncio.run(run_trial(t, job, UnitExecutor(), UnknownCitationClient(), ledger_path=tmp_path / "l.sqlite"))
    assert row["environment_task_success"]
    assert not row["final_commit_allowed"]
    assert not row["final_task_success"]
    assert row["graph_errors"]


@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
@pytest.mark.parametrize("hallucinate", [False, True])
def test_ack_loss_dependency_recovers_without_registering_absent_observation(topology, hallucinate, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial

    class EmptyObservationClient(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = super().complete(prompt, json_mode=json_mode)
            inputs = json.loads(prompt.split("\nInput: ", 1)[1])
            if inputs.get("observations") == [] and hallucinate:
                return json.dumps({"selected_evidence_id": "invented", "payload": payload(sku="HALLUCINATED")})
            return result

    rows = {}
    for arm in ("baseline", "dependency"):
        rows[arm] = asyncio.run(run_trial(task(), {"arm": arm, "condition": "acknowledgement_loss",
            "topology": topology}, UnitExecutor(), EmptyObservationClient(),
            ledger_path=tmp_path / f"{arm}.sqlite"))
    for row in rows.values():
        assert row["environment_task_success"] and row["final_task_success"]
        assert row["final_commit_allowed"] and row["common_recovery_enabled"]
        assert row["source_evidence"] is None
        empty_events = [e for e in row["events"] if isinstance(e.get("input"), dict)
                        and e["input"].get("observations") == []]
        assert len(empty_events) == 1 and empty_events[0]["role"] == "Worker"
        assert empty_events[0]["extra_model_call"] is False
        if hallucinate:
            assert empty_events[0]["output"]["payload"]["sku"] == "HALLUCINATED"
        assert all(e["version"] is not None and e["payload"].get("sku") != "HALLUCINATED"
                   for e in row["graph"]["evidence"].values())
        assert len(row["fault_events"]) == 1
        assert row["fault_events"][0]["boundary"] == "action_ack"
        assert row["fault_events"][0]["delivered_count"] == 0
        assert row["budget"]["limits"] == {"get": 4, "model_call": 3, "replay": 1}
    assert rows["dependency"]["budget"]["used"] == {
        "get": 2, "model_call": 1 if topology == "flat" else 0, "replay": 0}
    assert rows["baseline"]["budget"]["used"] == {"get": 0, "model_call": 0, "replay": 0}
    recovery, = rows["dependency"]["recovery_events"]
    assert recovery["kind"] == "dependency_readback" and recovery["common"] is False
    assert recovery["before"] is None and recovery["after"]["version"] == 3
    assert any(d["kind"] == "decision_commit_blocked"
               for d in rows["dependency"]["detection_events"]) is (topology == "flat")
    assert [r["kind"] for r in rows["baseline"]["recovery_events"]] == ([] if topology == "flat" else ["common_recovery"])


@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
def test_ack_loss_dependency_keeps_common_recovery_after_explicit_rejection(topology, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial

    class RejectParentClient(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = json.loads(super().complete(prompt, json_mode=json_mode))
            inputs = json.loads(prompt.split("\nInput: ", 1)[1])
            if inputs.get("prior_judgment") is not None:
                result["decision"] = "reject"
            return json.dumps(result)

    row = asyncio.run(run_trial(task(), {"arm": "dependency", "condition": "acknowledgement_loss",
        "topology": topology}, UnitExecutor(), RejectParentClient(), ledger_path=tmp_path / "common.sqlite"))
    assert row["final_task_success"] and row["final_commit_allowed"]
    assert row["common_recovery_enabled"]
    assert [(r["kind"], r["common"]) for r in row["recovery_events"]] == [
        ("dependency_readback", False), ("common_recovery", True)]
    assert row["budget"]["used"] == {"get": 2, "model_call": 0, "replay": 0}


@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
@pytest.mark.parametrize("version", [None, "opaque-version"])
def test_actual_delivered_ambiguous_version_remains_blocked(topology, version, tmp_path, monkeypatch):
    import asyncio
    from mas_faults import shopping_multimechanism as runtime

    class AmbiguousObservation(runtime.SingleBoundaryFault):
        def deliver(self, boundary, message):
            delivered = super().deliver(boundary, message)
            if boundary == "observation_handoff":
                delivered[0]["version"] = version
            return delivered

    monkeypatch.setattr(runtime, "SingleBoundaryFault", AmbiguousObservation)
    row = asyncio.run(runtime.run_trial(task(), {"arm": "dependency", "condition": "clean",
        "topology": topology}, UnitExecutor(), UnitClient(), ledger_path=tmp_path / "ambiguous.sqlite"))
    assert row["source_evidence"]["version"] == version
    assert row["source_evidence"]["action_id"] is not None
    assert row["environment_task_success"]
    assert not row["final_commit_allowed"] and not row["final_task_success"]
    assert any(e["version"] == version and e["status"] == "unresolved"
               for e in row["graph"]["evidence"].values())
    assert any(d["kind"] == "decision_commit_blocked" for d in row["detection_events"])


def test_no_observation_does_not_activate_evidence_payload_fault(tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial

    class InvalidActionClient(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = json.loads(super().complete(prompt, json_mode=json_mode))
            inputs = json.loads(prompt.split("\nInput: ", 1)[1])
            if "operation" in inputs:
                result["quantity"] = -1
            return json.dumps(result)

    row = asyncio.run(run_trial(task(), {"arm": "dependency", "condition": "valid_partial",
        "topology": "sequential"}, UnitExecutor(), InvalidActionClient(), ledger_path=tmp_path / "absent.sqlite"))
    assert row["source_evidence"] is None
    assert row["fault_events"] == []
    assert not row["action_contract_valid"] and not row["final_task_success"]


@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
@pytest.mark.parametrize("condition", ["clean", "acknowledgement_loss"])
@pytest.mark.parametrize("arm", ["baseline", "dependency", "combined"])
def test_ack_gate_valid_outputs_have_complete_judgment_graph(topology, condition, arm, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial

    row = asyncio.run(run_trial(task(), {"arm": arm, "condition": condition, "topology": topology},
        UnitExecutor(), UnitClient(), ledger_path=tmp_path / "graph.sqlite"))
    assert row["final_task_success"] and row["final_commit_allowed"]
    assert row["graph_errors"] == []
    nodes = row["graph"]["judgments"]
    assert set(nodes) == {j["judgment_id"] for j in row["judgments"]}
    for judgment in row["judgments"]:
        parent = judgment["input_parent_judgment"]
        assert nodes[judgment["judgment_id"]]["parent_judgment_ids"] == ([parent["judgment_id"]] if parent else [])
    if condition == "acknowledgement_loss" and arm == "baseline":
        current, = [node for key, node in nodes.items() if key.startswith("judgment-current-")]
        assert current["status"] == "unresolved" and current["evidence_ids"] == []
        assert current["provided_evidence_ids"] == []
        final, = [node for key, node in nodes.items() if key.startswith("judgment-final-")]
        assert final["parent_judgment_ids"] == [current["judgment_id"]]
        assert final["status"] == "unresolved"


@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
def test_absent_evidence_judgment_still_rejects_fabricated_citations(topology, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial

    class FabricatedCitation(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = json.loads(super().complete(prompt, json_mode=json_mode))
            inputs = json.loads(prompt.split("\nInput: ", 1)[1])
            if "prior_judgment" in inputs and inputs["evidence"] is None:
                result["evidence_ids"] = ["never-observed"]
            return json.dumps(result)

    row = asyncio.run(run_trial(task(), {"arm": "baseline", "condition": "acknowledgement_loss",
        "topology": topology}, UnitExecutor(), FabricatedCitation(), ledger_path=tmp_path / "unknown.sqlite"))
    current, = [j for j in row["judgments"] if j["judgment_id"].startswith("judgment-current-")]
    assert current["judgment_id"] not in row["graph"]["judgments"]
    assert {"judgment_id": current["judgment_id"], "error_type": "invalid_dependency"} in row["graph_errors"]
    assert "never-observed" not in row["graph"]["evidence"]


def test_combined_does_not_forget_newer_unselected_independent_evidence(tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs

    class PreferDamagedSource(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = json.loads(super().complete(prompt, json_mode=json_mode))
            inputs = json.loads(prompt.split("\nInput: ", 1)[1])
            if "candidate_observations" in inputs:
                result = {"selected_evidence_id": inputs["candidate_observations"][0]["evidence_id"]}
            return json.dumps(result)

    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = next(j for j in build_jobs([t], 1) if j["arm"] == "combined"
               and j["condition"] == "contract_consistent_identity_corruption" and j["topology"] == "sequential")
    source = make_envelope({**payload(sku="OTHER"), "task_id": "other", "product_id": "44"},
        task_id="other", session_id="other", entity_id="other", version=1,
        action_id="other", evidence_id="other", source="unit-fixture")
    row = asyncio.run(run_trial(t, job, UnitExecutor(), PreferDamagedSource(),
                               cross_task_evidence=source, ledger_path=tmp_path / "l.sqlite"))
    assert row["final_evidence"]["sku"] == "SKU"
    assert any(e["source"] == "EnvironmentReadback" for e in row["graph"]["evidence"].values())
    assert row["final_task_success"]
    assert row["evidence_acceptance_errors"] > 0
    assert "incorrect_verification" in row["observed_M_consequence"]


def test_unknown_environment_is_not_claimed_wrong_decision():
    row = {"task": task(), "environment_state": {"status": "observation_unavailable"},
           "final_evidence": payload(), "final_verdict": {"task_id": "t", "decision": "accept"},
           "judgments": [], "recovery_events": [], "fault_events": []}
    result = evaluate_trial(row)
    assert result["decision_correct"] is None
    assert result["environment_task_success"] is None
    assert result["task_score"] is None
    assert "M4" not in result["observed_M_consequence"]


@pytest.mark.parametrize("arm", ["baseline", "action_protocol"])
def test_unknown_action_keeps_trace_and_never_reexecutes(arm, tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs

    class WriteThenTimeout(UnitExecutor):
        def __init__(self):
            super().__init__()
            self.writes = 0

        def add_quantity(self, t, q):
            self.q += q
            self.writes += 1
            raise TimeoutError("unit post-write timeout")

    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = next(j for j in build_jobs([t], 1) if j["arm"] == arm
               and j["condition"] == "duplicate_action_delivery" and j["topology"] == "sequential")
    ex = WriteThenTimeout()
    row = asyncio.run(run_trial(t, job, ex, UnitClient(), ledger_path=tmp_path / "l.sqlite"))
    assert ex.writes == 1
    assert row["action_outcome_unknown"]
    assert row["action_ledger_events"]
    assert row["http_receipts"]


def test_model_timeout_carries_partial_causal_trace(tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs

    class FailingClient(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            if len(self.request_log) == 3:
                raise RuntimeError("unit model timeout")
            return super().complete(prompt, json_mode=json_mode)

    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = build_jobs([t], 1)[0]
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run_trial(t, job, UnitExecutor(), FailingClient(), ledger_path=tmp_path / "l.sqlite"))
    assert len(caught.value.partial_trial["events"]) == 3
    assert caught.value.partial_trial["judgments"]


def test_fabricated_target_quantity_counts_historical_wrong_acceptance(tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    from mas_faults.multimechanism_matrix import build_jobs

    class TargetFabricatingWorker(UnitClient):
        def complete(self, prompt, *, json_mode=False):
            result = json.loads(super().complete(prompt, json_mode=json_mode))
            inputs = json.loads(prompt.split("\nInput: ", 1)[1])
            if "observations" in inputs and result.get("payload"):
                p = result["payload"]
                p["observed_quantity"] = inputs["task"]["quantity"]
                p["cart_verified"] = True
                p["evidence"] = json.dumps({k: v for k, v in p.items() if k != "evidence"})
            return json.dumps(result)

    t = {**task(), "product_url": "http://localhost:17770/product"}
    job = next(j for j in build_jobs([t], 1) if j["arm"] == "baseline"
               and j["condition"] == "duplicate_action_delivery" and j["topology"] == "sequential")
    row = asyncio.run(run_trial(t, job, UnitExecutor(), TargetFabricatingWorker(), ledger_path=tmp_path / "l.sqlite"))
    assert row["environment_state"]["observed_quantity"] == 3
    assert row["evidence_acceptance_errors"] >= 2
    assert "incorrect_verification" in row["observed_M_consequence"]
    assert row["judgments"][0]["semantic_mismatch"] is False


@pytest.fixture
def action_trace_run(tmp_path, monkeypatch):
    """Unit-only HTTP transport exercising the actual adapter and SQLite ledger."""
    import asyncio
    import requests
    from urllib.parse import urlsplit
    from mas_faults.shopping_action_protocol import MultiStateShoppingExecutor
    from mas_faults.shopping_multimechanism import run_trial

    def run(condition, *, arm="action_protocol", fail_write=False, wrong_quantity=False):
        state = {"quantity": 0, "writes": 0}

        def send(session, method, url, **kwargs):
            path = urlsplit(url).path
            response = requests.Response()
            response.status_code, response.url, response.encoding = 200, url, "utf-8"
            if path == "/product":
                response._content = b'<input name="product" value="12"><button data-product-sku="SKU">'
                return response
            if method.upper() == "POST" and path.endswith("/guest-carts"):
                value = "unit-private-cart"
            else:
                if method.upper() in {"PUT", "POST"}:
                    state["writes"] += 1
                    q = kwargs["json"]["cartItem"]["qty"]
                    state["quantity"] = state["quantity"] + q if method.upper() == "POST" else q
                    if state["writes"] == 2 and fail_write:
                        raise requests.Timeout("unit write outcome unknown")
                    if state["writes"] == 2 and wrong_quantity:
                        state["quantity"] = 1
                item = {"item_id": 7, "sku": "SKU", "name": "Product", "qty": state["quantity"]}
                value = [item] if method.upper() == "GET" else item
            response._content = json.dumps(value).encode()
            return response

        monkeypatch.setattr(requests.Session, "request", send)
        t = {**task(), "product_url": "http://shopping.invalid/product"}
        row = asyncio.run(run_trial(t, {"arm": arm, "condition": condition, "topology": "sequential"},
                                   MultiStateShoppingExecutor("http://shopping.invalid"), UnitClient(),
                                   ledger_path=tmp_path / f"{condition}-{arm}-{fail_write}-{wrong_quantity}.sqlite"))
        return row, state
    return run


@pytest.mark.parametrize("condition,kind", [("request_non_delivery", "request_redelivery"),
                                            ("acknowledgement_loss", "acknowledgement_retrieval")])
def test_unit_action_rescue_is_trace_verified_recovery_without_extra_gets(action_trace_run, condition, kind):
    clean, _ = action_trace_run("clean")
    row, state = action_trace_run(condition)
    assert row["recovery_detected"] is True
    assert row["action_recovery_count"] == 1
    assert kind in row["recovery_type"]
    assert row["propagation_class"] == "detected_and_recovered"
    assert row["action_prevention_count"] == 0
    event, = row["action_protocol_events"]
    assert event["kind"] == kind and event["verified"] is True
    assert event["outcome"] == "recovery"
    assert event["ledger_event_ids"] and event["receipt_indices"]
    assert event["action_id"] == row["action_ledger_state"]["action_id"]
    assert state["writes"] == 2  # Initial add and one adjustment, unit transport only.
    assert row["budget"]["used"]["get"] == clean["budget"]["used"]["get"] == 0
    assert len(row["http_receipts"]) == len(clean["http_receipts"])


def test_unit_duplicate_replay_is_prevention_not_recovery(action_trace_run):
    row, state = action_trace_run("duplicate_action_delivery")
    event, = row["action_protocol_events"]
    assert event["kind"] == "duplicate_prevention" and event["verified"] is True
    assert event["outcome"] == "prevention"
    assert event["new_http_receipt_indices"] == []
    assert row["action_prevention_count"] == 1
    assert row["prevention_detected"] is True
    assert row["recovery_detected"] is False
    assert row["recovery_type"] == []
    assert row["propagation_class"] == "detected_and_prevented"
    assert state["writes"] == 2


@pytest.mark.parametrize("condition", ["request_non_delivery", "acknowledgement_loss", "duplicate_action_delivery"])
def test_unit_unknown_write_cannot_count_as_action_recovery_or_prevention(action_trace_run, condition):
    row, _ = action_trace_run(condition, fail_write=True)
    assert row["action_outcome_unknown"] is True
    assert row["action_recovery_count"] == row["action_prevention_count"] == 0
    assert not any(e["verified"] for e in row["action_protocol_events"])


@pytest.mark.parametrize("damage", ["ledger_missing", "ledger_unknown", "ledger_binding", "confirmation_missing",
                                    "receipt_missing", "http_failure", "readback_mismatch", "foreign_cart",
                                    "no_write", "response_unused", "not_pending", "digest_changed"])
def test_unit_recovery_claim_must_be_supported_by_ledger_and_http(action_trace_run, damage):
    import copy
    row, _ = action_trace_run("request_non_delivery")
    row = copy.deepcopy(row)
    event = row["action_protocol_events"][0]
    event["verified"] = True  # Evaluator must not trust this assertion.
    indices = event["receipt_indices"]
    if damage == "ledger_missing":
        row["action_ledger_events"] = []
    elif damage == "ledger_unknown":
        row["action_ledger_state"]["state"] = "unknown"
    elif damage == "ledger_binding":
        row["action_ledger_state"]["session_id"] = "other"
    elif damage == "confirmation_missing":
        row["action_ledger_events"] = [e for e in row["action_ledger_events"] if e["event"] != "confirmed"]
    elif damage == "receipt_missing":
        row["http_receipts"] = [r for r in row["http_receipts"] if r["receipt_index"] != indices[-1]]
    elif damage == "http_failure":
        row["http_receipts"][indices[-1]]["status_code"] = 503
    elif damage == "readback_mismatch":
        row["http_receipts"][indices[-1]]["response_payload"][0]["qty"] = 9
    elif damage == "foreign_cart":
        row["http_receipts"][indices[-1]]["guest_cart_id_sha256"] = "another-cart"
    elif damage == "no_write":
        for i in indices:
            row["http_receipts"][i]["request_method"] = "GET"
    elif damage == "response_unused":
        next(e for e in row["events"] if e.get("role") == "ActionExecutor")["output"] = []
    elif damage == "not_pending":
        event["before_state"] = "unknown"
    else:
        event["receipt_sha256"] = "false-digest"
    result = evaluate_trial(row)
    assert result["action_recovery_count"] == 0
    assert result["recovery_detected"] is False
    assert result["action_protocol_events"][0]["verified"] is False


def test_unit_detection_and_untraced_unit_executor_are_not_action_recovery(tmp_path):
    import asyncio
    from mas_faults.shopping_multimechanism import run_trial
    t = {**task(), "product_url": "http://shopping.invalid/product"}
    row = asyncio.run(run_trial(t, {"arm": "action_protocol", "condition": "acknowledgement_loss",
                                    "topology": "sequential"}, UnitExecutor(), UnitClient(),
                               ledger_path=tmp_path / "l.sqlite"))
    assert row["detection_events"]
    assert row["action_recovery_count"] == 0
    assert row["recovery_detected"] is False


def test_unit_confirmed_negative_response_recovery_does_not_imply_task_success(action_trace_run):
    row, _ = action_trace_run("request_non_delivery", wrong_quantity=True)
    assert row["action_ledger_state"]["receipt"]["cart_verified"] is False
    assert row["action_recovery_count"] == 1
    assert row["environment_task_success"] is False
    assert row["final_task_success"] is False


def test_unit_lost_request_correct_rejection_still_has_system_consequences(action_trace_run):
    from mas_faults.shopping_multimechanism import verify_action_consequences
    row, state = action_trace_run("request_non_delivery", arm="baseline")
    assert set(row["observed_M_consequence"]) == {"task_incomplete", "failed_delegation"}
    assert verify_action_consequences(row) == {"failed_delegation"}
    assert row["decision_correct"] is True
    assert "M4" not in row["observed_M_consequence"]
    assert row["evidence_acceptance_errors"] == 0
    assert row["propagation_class"] == "propagated_to_M_final_failure"
    assert state["writes"] == 1  # Only the initial add happened.


def test_unit_duplicate_actual_deliveries_count_without_false_decision_error(action_trace_run):
    from mas_faults.shopping_multimechanism import verify_action_consequences
    row, state = action_trace_run("duplicate_action_delivery", arm="baseline")
    assert set(row["observed_M_consequence"]) == {"task_incomplete", "duplicate_execution"}
    assert verify_action_consequences(row) == {"duplicate_execution"}
    assert row["decision_correct"] is True
    assert row["evidence_acceptance_errors"] == 0
    assert row["propagation_class"] == "propagated_to_M_final_failure"
    assert state["writes"] == 3


@pytest.mark.parametrize("environment,commit,expected", [
    (payload(1), True, True),
    (payload(2), False, False),
    ({"status": "observation_unavailable"}, True, False),
])
def test_unit_task_incomplete_requires_known_goal_violation(environment, commit, expected):
    row = {"task": task(), "environment_state": environment, "final_evidence": payload(),
           "final_verdict": {"task_id": "t", "decision": "reject"}, "judgments": [],
           "final_commit_allowed": commit, "recovery_events": [], "fault_events": []}
    result = evaluate_trial(row)
    assert ("task_incomplete" in result["observed_M_consequence"]) is expected


@pytest.mark.parametrize("damage", ["action_invalid", "task_binding", "session_binding", "action_binding",
                                    "missing_input", "delivered_input", "missing_executor", "executed_output",
                                    "count_one", "unknown", "execution_event", "missing_registration"])
def test_unit_failed_delegation_requires_bound_empty_delivery_and_unexecuted_ledger(action_trace_run, damage):
    from mas_faults.shopping_multimechanism import verify_action_consequences
    row, _ = action_trace_run("request_non_delivery", arm="baseline")
    action = next(e for e in row["events"] if e.get("role") == "ActionExecutor")
    if damage == "action_invalid":
        row["action_contract_valid"] = False
    elif damage in {"task_binding", "session_binding", "action_binding"}:
        action[damage.removesuffix("_binding") + "_id"] = "other"
    elif damage == "missing_input":
        action.pop("input")
    elif damage == "delivered_input":
        action["input"] = [row["action_ledger_state"]["params"]]
    elif damage == "missing_executor":
        row["events"].remove(action)
    elif damage == "executed_output":
        action["output"] = [{"action_id": row["action_ledger_state"]["action_id"], "receipt": payload()}]
    elif damage == "count_one":
        row["action_ledger_state"]["execution_count"] = 1
    elif damage == "unknown":
        row["action_outcome_unknown"] = True
    elif damage == "execution_event":
        row["action_ledger_events"].append({"action_id": row["action_ledger_state"]["action_id"],
                                            "event": "execute_entered"})
    else:
        row["action_ledger_events"] = []
    assert "failed_delegation" not in verify_action_consequences(row)
    assert "failed_delegation" not in evaluate_trial(row)["observed_M_consequence"]


@pytest.mark.parametrize("damage", ["count_one", "missing_confirmation", "same_delivery_id", "missing_receipt",
                                    "reused_receipts", "failed_http", "wrong_quantity", "wrong_cart",
                                    "wrong_product", "initial_write", "missing_entry"])
def test_unit_duplicate_execution_requires_distinct_matching_successful_action_writes(action_trace_run, damage):
    from mas_faults.shopping_multimechanism import verify_action_consequences
    row, _ = action_trace_run("duplicate_action_delivery", arm="baseline")
    confirmations = [e for e in row["action_ledger_events"] if e["event"] == "observed_delivery_confirmed"]
    last = confirmations[-1]
    indices = last["details"]["receipt"]["http_receipt_indices"]
    write = next(r for r in row["http_receipts"] if r["receipt_index"] in indices and r["request_method"] == "POST")
    if damage == "count_one":
        row["action_ledger_state"]["execution_count"] = 1
    elif damage == "missing_confirmation":
        row["action_ledger_events"].remove(last)
    elif damage == "same_delivery_id":
        last["details"]["delivery_id"] = confirmations[0]["details"]["delivery_id"]
    elif damage == "missing_receipt":
        row["http_receipts"].remove(write)
    elif damage == "reused_receipts":
        last["details"]["receipt"]["http_receipt_indices"] = confirmations[0]["details"]["receipt"]["http_receipt_indices"]
    elif damage == "failed_http":
        write["status_code"] = 503
    elif damage == "wrong_quantity":
        write["request_payload"]["cartItem"]["qty"] = 99
    elif damage == "wrong_cart":
        write["guest_cart_id_sha256"] = "different-cart"
    elif damage == "wrong_product":
        write["request_payload"]["cartItem"]["sku"] = "OTHER"
    elif damage == "initial_write":
        write["purpose"] = "add_to_cart.add_item"
    else:
        row["action_ledger_events"] = [e for e in row["action_ledger_events"]
                                        if not (e["event"] == "execute_entered"
                                                and e["details"]["delivery_id"] == last["details"]["delivery_id"])]
    assert "duplicate_execution" not in verify_action_consequences(row)
    assert "duplicate_execution" not in evaluate_trial(row)["observed_M_consequence"]


@pytest.mark.parametrize("condition", ["request_non_delivery", "duplicate_action_delivery"])
def test_unit_consequences_do_not_depend_on_fault_labels(action_trace_run, condition):
    from mas_faults.shopping_multimechanism import verify_action_consequences
    row, _ = action_trace_run(condition, arm="baseline")
    expected = set(row["observed_M_consequence"])
    action_expected = verify_action_consequences(row)
    row["condition"] = "clean"
    row["fault_events"] = []
    row["detection_events"] = []
    assert set(evaluate_trial(row)["observed_M_consequence"]) == expected
    assert verify_action_consequences(row) == action_expected


def test_unit_action_recovery_can_coexist_with_task_incomplete(action_trace_run):
    row, _ = action_trace_run("request_non_delivery", wrong_quantity=True)
    assert row["recovery_detected"] is True
    assert row["action_recovery_count"] == 1
    assert "task_incomplete" in row["observed_M_consequence"]
    assert row["propagation_class"] == "propagated_to_M_final_failure"
    assert row["decision_correct"] is True
    assert "M4" not in row["observed_M_consequence"]
