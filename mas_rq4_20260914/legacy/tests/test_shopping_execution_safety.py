import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
import requests

import run_shopping_mitigation as matrix
import run_webarena_architecture_rq2 as workflow
from mas_faults.shopping_mitigation import EvidencePolicy
from test_shopping_mitigation import TASK, evidence


def test_infrastructure_timeout_is_not_returned_as_task_failure():
    class Executor:
        def add_to_cart(self, action):
            raise requests.Timeout("site unavailable")

    with pytest.raises(requests.Timeout):
        workflow.execute_delivered_actions(Executor(), [{**TASK, "action": "add_to_cart"}], TASK)


@pytest.mark.parametrize("mode", ["baseline", "always_recheck", "guarded_recheck"])
@pytest.mark.parametrize("topology", ["sequential", "flat", "hierarchical"])
def test_rejected_final_gets_one_common_recovery_in_every_arm(monkeypatch, mode, topology):
    class Executor:
        def __init__(self, base_url):
            self.http_receipts = []
            self.readback_http_request_count = 0

        def add_to_cart(self, action):
            return evidence()

        def reobserve_cart(self, task):
            self.readback_http_request_count = 2
            for purpose in ("readback_product", "readback_cart"):
                self.http_receipts.append({"index": len(self.http_receipts), "purpose": purpose,
                                           "method": "GET", "status_code": 200})
            return evidence()

    class Client:
        call_count = prompt_tokens = completion_tokens = 0
        final_calls = 0
        model_info = SimpleNamespace(model="unit-test-only", provider="unit-test")

        def complete(self, prompt):
            self.call_count += 1
            if "action MUST" in prompt:
                return json.dumps({"action": "add_to_cart"})
            if "Convert the delivered" in prompt:
                return json.dumps(evidence())
            if "independent verifier" not in prompt:
                self.final_calls += 1
                choice = "accept" if self.final_calls == 2 else "reject"
            else:
                choice = "reject"
            return json.dumps({"decision": choice, "task_id": TASK["task_id"]})

    monkeypatch.setattr(workflow, "ShoppingHTTPExecutor", Executor)
    client = Client()
    result = asyncio.run(workflow.run_one(client, TASK, topology, "omission", 4, 1,
                         "http://shopping.invalid", receiver_policy=EvidencePolicy(mode, TASK),
                         common_recovery=True))
    assert result["final_task_success"]
    assert client.final_calls == 2
    assert len(result["common_recovery_events"]) == 1
    event = result["common_recovery_events"][0]
    assert event["before_verdict"]["decision"] == "reject"
    assert event["after_verdict"]["decision"] == "accept"
    assert event["readback_called"] and len(event["readback_receipts"]) == 2
    assert result["common_recovery_http_requests"] == 2
    assert result["additional_http_requests"] == (0 if mode == "baseline" else 2)
    assert sum(e["fault_applied"] for e in result["events"]) == 1


def clean_source(task_id, run_id):
    payload = {**evidence(), "task_id": task_id}
    nested = json.loads(payload["evidence"])
    nested["task_id"] = task_id
    payload["evidence"] = json.dumps(nested)
    return {"task_id": task_id, "run_id": run_id, "topology": "sequential", "repeat_index": 1,
            "fault_type": "none", "mitigation_mode": "baseline", "final_task_success": True,
            "environment_state": copy.deepcopy(payload),
            "model": "unit-test-only", "provider": "unit-test",
            "events": [{"step_index": 4, "original_message": payload,
                        "source_agent": "Shopping Worker", "target_agent": "Verifier",
                        "message_id": run_id + ":step4"}]}


def test_stale_source_is_frozen_real_prior_clean_not_placeholder(tmp_path):
    rows = [clean_source("task-1", "clean-1"), clean_source("task-2", "clean-2")]
    job = {"task_id": "task-1", "topology": "sequential", "repeat_index": 1}
    first = matrix.freeze_stale_source(rows, job, tmp_path)
    assert first["source_run_id"] == "clean-2"
    assert first["source_message_id"] == "clean-2:step4"
    assert first["payload"] == rows[1]["events"][0]["original_message"]
    assert first["payload"]["product_id"] == "10"
    assert matrix.freeze_stale_source(list(reversed(rows)), job, tmp_path) == first


def test_stale_source_refuses_same_task_or_failed_clean(tmp_path):
    rows = [clean_source("task-1", "clean-1"), {**clean_source("task-2", "clean-2"), "final_task_success": False}]
    with pytest.raises(ValueError, match="different task"):
        matrix.freeze_stale_source(rows, {"task_id": "task-1", "topology": "sequential", "repeat_index": 1}, tmp_path)


def test_stale_source_skips_invalid_original_even_when_clean_was_later_recovered(tmp_path):
    broken = clean_source("task-0", "recovered-clean")
    broken["events"][0]["original_message"].pop("sku")
    rows = [broken, clean_source("task-2", "clean-2")]
    chosen = matrix.freeze_stale_source(rows, {"task_id": "task-1", "topology": "sequential", "repeat_index": 1}, tmp_path)
    assert chosen["source_run_id"] == "clean-2"


def test_pilot_stale_injection_requires_actual_source_before_model_calls():
    client = SimpleNamespace(model_info=SimpleNamespace(model="unit-test-only"))
    with pytest.raises(ValueError, match="real clean source"):
        asyncio.run(workflow.run_one(client, TASK, "sequential", "stale_replay", 4, 1,
                                    "http://shopping.invalid", receiver_policy=EvidencePolicy("baseline", TASK)))


def test_summary_keeps_common_and_added_recovery_costs_separate(tmp_path):
    from test_mitigation_matrix import row
    value = {**row(), "pair_key": "unit-pair", "job_key": "unit-pair:baseline", "common_recovery_http_requests": 2, "additional_http_requests": 0,
             "common_recovery_detected": True, "mitigation_recovery_detected": False,
             "common_recovery_events": [{"trigger": "final_rejected_or_unbound"}],
             "http_receipts": [{"method": "GET"}], "model_requests": [{"prompt_tokens": 2}]}
    matrix.append_jsonl(tmp_path / "main_runs.jsonl", value)
    matrix.write_reports(tmp_path, [])
    counts = json.loads((tmp_path / "summary.json").read_text())["by_mode"]["baseline"]
    assert counts["common_recovery_http_requests"] == 2
    assert counts["http_requests_added"] == 0
    assert counts["common_recovery_successes"] == 1
    groups = {e["trace_group"] for e in matrix.read_jsonl(tmp_path / "traces.jsonl")}
    assert {"common_recovery_events", "http_receipts", "model_requests"}.issubset(groups)
