"""Offline result-audit contracts. Fixtures are synthetic, never experiment runs."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from run_shopping_mitigation import VERSION, audit_outcome, build_jobs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_shopping_mitigation.py"
MODES = ("baseline", "always_recheck", "guarded_recheck")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evidence(task):
    state = {"task_id": task["task_id"], "product_title": task["product_title"],
             "product_id": task["task_id"], "sku": task["task_id"].upper(),
             "requested_quantity": 1, "observed_quantity": 1, "cart_verified": True}
    return {**state, "evidence": json.dumps(state)}


def cart_receipt():
    return {"receipt_index": 0, "purpose": "add_to_cart.create_cart", "request_method": "POST",
            "request_url": "http://127.0.0.1/carts", "status_code": 200,
            "response_sha256": "a" * 64, "guest_cart_id_sha256": "b" * 64,
            "response_payload": {"guest_cart_id_sha256": "b" * 64}}


def receipts(payload, start=1):
    bodies = [{"product_id": payload["product_id"], "sku": payload["sku"]},
              [{"item_id": 7, "sku": payload["sku"], "name": payload["product_title"],
                "qty": payload["observed_quantity"]}]]
    return [{"receipt_index": i, "purpose": purpose, "request_method": "GET",
             "request_url": "http://127.0.0.1/items", "status_code": 200,
             "response_sha256": "a" * 64, "guest_cart_id_sha256": "b" * 64,
             "response_payload": bodies[i - start]}
            for i, purpose in enumerate(("reobserve_cart.product_page", "reobserve_cart.items"), start)]


def add_common_readback(row):
    payload = copy.deepcopy(row["environment_state"])
    start = len(row["http_receipts"])
    additions = receipts(payload, start)
    row["http_receipts"].extend(additions)
    row["common_recovery_http_requests"] = 2
    row["common_recovery_events"] = [{"readback_called": True,
        "readback_receipt_indices": [start, start + 1], "readback_receipts": copy.deepcopy(additions),
        "receiver": row["decision_evidence_receiver"], "readback": payload,
        "before_verdict": {"task_id": row["task_id"], "decision": "reject"},
        "before_primary_evidence": copy.deepcopy(row["primary_evidence"]),
        "before_decision_evidence": copy.deepcopy(row["decision_evidence"]),
        "before_verification_verdict": copy.deepcopy(row["verification_verdict"]),
        "after_verdict": copy.deepcopy(row["final_answer"]["verdict"]), "replacement_used": True}]
    row["decision_evidence"] = payload


def fixture():
    tasks = [{"task_id": name, "product_title": name, "quantity": 1,
              "product_url": "http://127.0.0.1/" + name} for name in ("tea", "juice")]
    jobs = build_jobs(tasks, 1)
    manifest = {"version": VERSION, "tasks": tasks, "jobs": jobs, "repetitions": 1,
                "provider": "modelscope_local", "model": "Qwen/Qwen3.8-27B",
                "max_attempts_per_job": 2, "max_readbacks_per_run": 1,
                "common_recovery": {"enabled": True, "max_readbacks": 1},
                "inference_settings": {"api_base_url": "http://127.0.0.1:18001/v1"},
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [ROOT / "run_shopping_mitigation.py",
                                            ROOT / "run_webarena_architecture_rq2.py",
                                            ROOT / "src/mas_faults/shopping_mitigation.py"]}}
    rows = []
    task_map = {task["task_id"]: task for task in tasks}
    for n, job in enumerate(jobs):
        task = task_map[job["task_id"]]
        payload = evidence(task)
        receiver = "Supervisor" if job["topology"] == "hierarchical" else "Verifier"
        decision = {"task_id": task["task_id"], "decision": "accept"}
        event = {"message_id": f"run-{n}:step4", "run_id": f"run-{n}",
                 "source_agent": "Shopping Worker", "target_agent": receiver,
                 "step_index": 4, "original_message": payload,
                 "delivered_message": [payload], "fault_type": job["fault"],
                 "fault_applied": job["fault"] != "none"}
        row = {**job, "run_id": f"run-{n}", "fault_type": job["fault"],
               "condition": "clean" if job["fault"] == "none" else job["fault"],
               "fault_applied": job["fault"] != "none", "events": [event],
               "model": manifest["model"], "provider": manifest["provider"],
               "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
               "api_call_count": 1, "model_requests": [{"request_index": 100 + n,
                   "prompt_tokens": 10, "completion_tokens": 2,
                   "model": manifest["model"], "provider": manifest["provider"]}],
               "latency_ms": 100, "common_recovery_enabled": True,
               "common_recovery_events": [], "common_recovery_http_requests": 0,
               "mitigation_events": [], "additional_http_requests": 0, "http_receipts": [cart_receipt()],
               "primary_evidence": payload, "decision_evidence": payload,
               "environment_state": payload, "primary_receiver": receiver,
               "decision_evidence_receiver": "Supervisor" if job["topology"] == "hierarchical" else "Coordinator",
               "verification_verdict": None if job["topology"] == "hierarchical" else decision,
               "final_answer": {"verdict": decision, "cart_verified": True}}
        if job["mitigation_mode"] == "always_recheck":
            row["http_receipts"].extend(receipts(payload))
            row["additional_http_requests"] = 2
            row["mitigation_events"] = [{"mode": "always_recheck", "receiver": receiver,
                "readback_called": True, "readback_count": 1, "readback_receipt_indices": [1, 2],
                "readback_response": payload, "before": payload, "before_issues": [], "after_issues": [],
                "replacement_used": True, "after": payload}]
        if job["fault"] == "stale_replay":
            source = next(r for r in rows if r["fault_type"] == "none"
                          and r["mitigation_mode"] == "baseline" and r["task_id"] != task["task_id"]
                          and r["topology"] == job["topology"])
            original = source["events"][0]
            row["stale_replay_source"] = {"source_run_id": source["run_id"],
                "source_task_id": source["task_id"], "source_message_id": original["message_id"],
                "model": source["model"], "provider": source["provider"],
                "payload": copy.deepcopy(original["original_message"]),
                "payload_sha256": digest(original["original_message"])}
            event["delivered_message"] = [copy.deepcopy(row["stale_replay_source"]["payload"])]
        rows.append(audit_outcome(row, task))
    attempts = [{**job, "attempt": 1, "model": manifest["model"],
                 "provider": manifest["provider"]} for job in jobs]
    return manifest, rows, attempts


def write_snapshot(path, manifest, rows, attempts, errors=(), infra=()):
    path.mkdir()
    (path / "matrix_manifest.json").write_text(json.dumps(manifest))
    for name, values in (("main_runs.jsonl", rows), ("run_attempts.jsonl", attempts),
                         ("run_errors.jsonl", errors), ("infrastructure_events.jsonl", infra)):
        (path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    for row in rows:
        source = row.get("stale_replay_source")
        if source:
            key = [row["task_id"], row["topology"], row["repeat_index"]]
            name = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20] + ".json"
            directory = path / "stale_sources"
            directory.mkdir(exist_ok=True)
            (directory / name).write_text(json.dumps(source))


@pytest.fixture
def auditor():
    def audit_results(*args, **kwargs):
        assert SCRIPT.is_file(), "offline audit script has not been implemented"
        spec = importlib.util.spec_from_file_location("shopping_result_audit", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.audit_results(*args, **kwargs)
    return SimpleNamespace(audit_results=audit_results)


def run_audit(tmp_path, auditor, data=None, **kwargs):
    path = tmp_path / "input"
    write_snapshot(path, *(data or fixture()), **kwargs)
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    report = auditor.audit_results(path, source_root=ROOT)
    assert before == {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    return report


def codes(report):
    return {issue["code"] for issue in report["findings"]}


def test_valid_matrix_has_three_arm_pairs_and_global_request_indices(tmp_path, auditor):
    report = run_audit(tmp_path, auditor)
    assert report["status"] == "complete"
    assert report["audit_status"] == "passed", report["findings"]
    assert report["matrix"]["expected_runs"] == 90
    assert report["pairing"]["complete_three_arm_pairs"] == 30
    assert report["pairing"]["comparisons"]["guarded_recheck"]["matched_clean_fault_pairs"] == 24
    assert report["usage"]["completed_known_total_tokens"] == 1080
    assert report["source_hashes"]["status"] == "verified"


@pytest.mark.parametrize("field,code", [("run_id", "duplicate_run_id"), ("job_key", "duplicate_job_key")])
def test_duplicate_run_or_job_is_not_silently_collapsed(tmp_path, auditor, field, code):
    m, rows, attempts = fixture()
    rows[1][field] = rows[0][field]
    report = run_audit(tmp_path, auditor, (m, rows, attempts))
    assert code in codes(report)
    assert report["audit_status"] == "failed"


def test_missing_job_is_incomplete_not_a_success_or_failure_run(tmp_path, auditor):
    m, rows, attempts = fixture()
    missing = rows.pop()
    report = run_audit(tmp_path, auditor, (m, rows, attempts))
    assert report["status"] == "incomplete"
    assert report["matrix"]["missing_job_keys"] == [missing["job_key"]]
    assert report["pairing"]["complete_three_arm_pairs"] == 29


@pytest.mark.parametrize("change,code", [
    (lambda m: m["jobs"].pop(), "manifest_jobs_mismatch"),
    (lambda m: m["jobs"].append(copy.deepcopy(m["jobs"][0])), "duplicate_manifest_job_key"),
    (lambda m: m.update(expected_runs=12), "expected_count_mismatch"),
    (lambda m: m.update(max_attempts_per_job=3), "attempt_budget"),
    (lambda m: m["common_recovery"].update(enabled=False), "common_recovery_disabled"),
    (lambda m: m["inference_settings"].update(api_base_url="https://example.com/v1"), "nonlocal_endpoint"),
])
def test_manifest_constraints(tmp_path, auditor, change, code):
    data = fixture()
    change(data[0])
    assert code in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("change,code", [
    ({"repeat_index": 9}, "job_metadata_mismatch"),
    ({"model": "remote-model"}, "model_mismatch"),
    ({"provider": "openai"}, "provider_mismatch"),
    ({"common_recovery_enabled": False}, "common_recovery_disabled"),
    ({"total_tokens": 13}, "token_total_mismatch"),
    ({"prompt_tokens": 9, "total_tokens": 11}, "request_usage_mismatch"),
    ({"final_task_success": False}, "derived_field_mismatch"),
    ({"common_recovery_http_requests": 3}, "http_budget"),
])
def test_row_constraints(tmp_path, auditor, change, code):
    data = fixture()
    data[1][0].update(change)
    assert code in codes(run_audit(tmp_path, auditor, data))


def test_missing_usage_remains_unknown_not_zero(tmp_path, auditor):
    data = fixture()
    data[1][0].pop("prompt_tokens")
    data[1][0].pop("total_tokens")
    data[1][1].pop("model_requests")
    report = run_audit(tmp_path, auditor, data)
    assert "token_usage_unknown" in codes(report)
    assert "request_usage_unknown" in codes(report)
    assert report["usage"]["completed_total_tokens"] is None
    assert report["audit_status"] == "unknown"


def test_mismatched_request_usage_is_not_counted_as_reconciled(tmp_path, auditor):
    data = fixture()
    data[1][0]["model_requests"][0]["prompt_tokens"] = 11
    report = run_audit(tmp_path, auditor, data)
    assert report["usage"]["request_reconciled_runs"] == 89
    assert report["usage"]["completed_total_tokens"] is None


def test_nonboolean_derived_success_cannot_pass_by_python_numeric_equality(tmp_path, auditor):
    data = fixture()
    data[1][0]["final_task_success"] = 1
    assert "derived_field_mismatch" in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("field,value", [("run_id", []), ("job_key", {}), ("task_id", []),
                                         ("events", None), ("environment_state", None)])
def test_malformed_row_is_reported_without_crashing(tmp_path, auditor, field, value):
    data = fixture()
    data[1][0][field] = value
    report = run_audit(tmp_path, auditor, data)
    assert report["audit_status"] != "passed"
    assert report["matrix"]["completed_runs"] == 90


def test_legacy_evaluation_is_deliberately_ignored(tmp_path, auditor):
    data = fixture()
    data[1][0]["legacy_evaluation"] = {"legacy_evaluation": {"final_task_success": False}}
    assert "derived_field_mismatch" not in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("change,code", [
    (lambda r: r["events"][0].update(fault_applied=False), "fault_event_count"),
    (lambda r: r["events"][0].update(step_index=3), "fault_event_location"),
    (lambda r: r["events"].append(copy.deepcopy(r["events"][0])), "fault_event_count"),
])
def test_fault_must_be_once_at_worker_step_four(tmp_path, auditor, change, code):
    data = fixture()
    change(next(r for r in data[1] if r["fault_type"] == "omission"))
    assert code in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("change,code", [
    (lambda r: r["stale_replay_source"].update(source_run_id="nonexistent"), "stale_source_missing_run"),
    (lambda r: r["stale_replay_source"].update(source_message_id="fabricated"), "stale_source_message"),
    (lambda r: r["stale_replay_source"].update(payload_sha256="0" * 64), "stale_source_hash"),
    (lambda r: r["stale_replay_source"]["payload"].update(sku="FORGED"), "stale_source_payload"),
    (lambda r: r["events"][0].update(delivered_message=[{}]), "stale_delivery_mismatch"),
])
def test_tampered_stale_source_is_detected(tmp_path, auditor, change, code):
    data = fixture()
    change(next(r for r in data[1] if r["fault_type"] == "stale_replay"))
    assert code in codes(run_audit(tmp_path, auditor, data))


def test_rehashing_forged_payload_cannot_replace_actual_worker_message(tmp_path, auditor):
    data = fixture()
    source = next(r for r in data[1] if r["fault_type"] == "stale_replay")["stale_replay_source"]
    source["payload"]["sku"] = "FORGED"
    source["payload_sha256"] = digest(source["payload"])
    assert "stale_source_payload" in codes(run_audit(tmp_path, auditor, data))


def test_stale_pair_arms_must_share_source(tmp_path, auditor):
    data = fixture()
    stale = [r for r in data[1] if r["fault_type"] == "stale_replay"]
    stale[0]["stale_replay_source"] = copy.deepcopy(stale[3]["stale_replay_source"])
    assert "stale_pair_source_mismatch" in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("field,value", [("source_run_id", []), ("source_task_id", "tea"),
                                         ("source_message_id", "missing")])
def test_stale_source_invalid_identity_is_reported(tmp_path, auditor, field, value):
    data = fixture()
    source = next(r for r in data[1] if r["fault_type"] == "stale_replay")
    source["stale_replay_source"][field] = value
    assert run_audit(tmp_path, auditor, data)["audit_status"] == "failed"


@pytest.mark.parametrize("indices", [[-1, 0], [0, 99], [0, 0], [False, 1], [1, 0]])
def test_http_indices_are_real_unique_ordered_indices(tmp_path, auditor, indices):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "always_recheck")
    row["mitigation_events"][0]["readback_receipt_indices"] = indices
    assert "http_receipt_indices" in codes(run_audit(tmp_path, auditor, data))


def test_added_and_common_readbacks_have_independent_one_call_budgets(tmp_path, auditor):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "always_recheck")
    row["mitigation_events"] *= 2
    row["common_recovery_events"] = [{"readback_called": True}] * 2
    assert "readback_budget" in codes(run_audit(tmp_path, auditor, data))


def test_attempts_and_error_tokens_are_not_added_as_completed_runs(tmp_path, auditor):
    m, rows, attempts = fixture()
    error = {**m["jobs"][0], "attempt": 1, "known_total_tokens": 5, "usage_complete": False}
    attempts.insert(1, {**attempts[0], "attempt": 2})
    report = run_audit(tmp_path, auditor, (m, rows, attempts), errors=[error])
    assert report["matrix"]["completed_runs"] == 90
    assert report["attempts"]["error_attempts"] == 1
    assert report["usage"]["error_known_total_tokens"] == 5
    assert report["usage"]["error_total_tokens"] is None


def test_third_attempt_and_duplicate_attempt_are_errors(tmp_path, auditor):
    data = fixture()
    data[2].extend([copy.deepcopy(data[2][0]), {**data[2][1], "attempt": 3}])
    result = codes(run_audit(tmp_path, auditor, data))
    assert {"duplicate_attempt", "attempt_budget"} <= result


def test_completed_run_cannot_be_assigned_to_earlier_attempt_after_last_attempt_failed(tmp_path, auditor):
    m, rows, attempts = fixture()
    attempts.append({**attempts[0], "attempt": 2})
    error = {**m["jobs"][0], "attempt": 2, "known_total_tokens": 0, "usage_complete": False}
    report = run_audit(tmp_path, auditor, (m, rows, attempts), errors=[error])
    assert "attempt_outcome_conflict" in codes(report)


def test_unadmitted_faults_are_excluded_from_analysis_pairs(tmp_path, auditor):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "baseline" and r["fault_type"] == "none")
    row["decision_evidence"] = None
    row["final_answer"]["verdict"] = {"task_id": row["task_id"], "decision": "reject"}
    add_common_readback(row)
    row.update(audit_outcome(row, data[0]["tasks"][0]))
    report = run_audit(tmp_path, auditor, data)
    assert "baseline_clean_not_admitted" in codes(report)
    # Twelve unadmitted task faults plus three stale arms using that failed clean.
    assert report["matrix"]["analysis_runs"] == 75


def test_successful_common_recovery_is_counted_separately(tmp_path, auditor):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "baseline" and r["fault_type"] == "omission")
    add_common_readback(row)
    row.update(audit_outcome(row, data[0]["tasks"][0]))
    report = run_audit(tmp_path, auditor, data)
    assert report["by_mode"]["baseline"]["common_recovery_detected"] == 1
    assert report["by_mode"]["baseline"]["mitigation_recovery_detected"] == 0
    assert report["by_mode"]["baseline"]["common_recovery_http_requests"] == 2


def recovery_fixture(group):
    data = fixture()
    mode = "always_recheck" if group == "mitigation" else "baseline"
    row = next(r for r in data[1] if r["mitigation_mode"] == mode and r["fault_type"] == "omission")
    if group == "mitigation":
        row["mitigation_events"][0].update(before=None, before_issues=["non_delivery"])
    else:
        add_common_readback(row)
    task = next(t for t in data[0]["tasks"] if t["task_id"] == row["task_id"])
    row.update(audit_outcome(row, task))
    assert row[group + "_recovery_detected"] is True
    return data, row, task


@pytest.mark.parametrize("issues", [[], ["constraint:task_id"]])
def test_before_issues_must_match_actual_before_evidence(tmp_path, auditor, issues):
    data, row, task = recovery_fixture("mitigation")
    row["mitigation_events"][0]["before_issues"] = issues
    row.update(audit_outcome(row, task))
    assert "before_issues_mismatch" in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("group", ["mitigation", "common"])
@pytest.mark.parametrize("change,code", [
    ("empty_items", "readback_payload_mismatch"),
    ("wrong_product", "readback_payload_mismatch"),
    ("wrong_quantity", "readback_payload_mismatch"),
    ("different_cart", "cart_identity_mismatch"),
    ("create_cart_payload", "cart_identity_mismatch"),
])
def test_recovery_requires_matching_http_payload_and_cart(tmp_path, auditor, group, change, code):
    data, row, _ = recovery_fixture(group)
    page, items = row["http_receipts"][-2:]
    if change == "empty_items":
        items["response_payload"] = []
    elif change == "wrong_product":
        page["response_payload"]["product_id"] = "different-product"
    elif change == "wrong_quantity":
        items["response_payload"][0]["qty"] = 0
    elif change == "different_cart":
        items["guest_cart_id_sha256"] = "0" * 64
    else:
        row["http_receipts"][0]["response_payload"]["guest_cart_id_sha256"] = "0" * 64
    # Keep embedded receipts synchronized so the new content check is decisive.
    if group == "common":
        row["common_recovery_events"][0]["readback_receipts"] = copy.deepcopy([page, items])
    report = run_audit(tmp_path, auditor, data)
    assert code in codes(report)
    assert report["by_mode"][row["mitigation_mode"]][group + "_recovery_detected"] == 0


@pytest.mark.parametrize("field", ["readback_response", "after"])
def test_both_readback_and_replacement_must_match_http_payload(tmp_path, auditor, field):
    data, row, task = recovery_fixture("mitigation")
    row["mitigation_events"][0][field] = {**row["mitigation_events"][0][field], "observed_quantity": 9}
    row.update(audit_outcome(row, task))
    assert "readback_payload_mismatch" in codes(run_audit(tmp_path, auditor, data))


def test_missing_http_payload_is_unknown_and_does_not_certify_recovery(tmp_path, auditor):
    data, row, _ = recovery_fixture("mitigation")
    row["http_receipts"][-1].pop("response_payload")
    report = run_audit(tmp_path, auditor, data)
    assert "readback_payload_unknown" in codes(report)
    assert report["audit_status"] == "unknown"
    assert report["by_mode"]["always_recheck"]["mitigation_recovery_detected"] == 0


def test_completed_always_arm_cannot_skip_its_readback(tmp_path, auditor):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "always_recheck" and r["fault_type"] == "none")
    row.update(mitigation_events=[], additional_http_requests=0, http_receipts=[cart_receipt()])
    row.update(audit_outcome(row, data[0]["tasks"][0]))
    assert "required_mitigation_readback" in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("decision,bound", [("reject", True), ("accept", False)])
@pytest.mark.parametrize("stage", ["final", "before_common"])
def test_reject_or_unbound_requires_one_common_readback(tmp_path, auditor, decision, bound, stage):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "baseline" and r["fault_type"] == "omission")
    verdict = {"task_id": row["task_id"] if bound else "different-task", "decision": decision}
    if stage == "final":
        row["final_answer"]["verdict"] = verdict
    else:
        row["common_recovery_events"] = [{"before_verdict": verdict, "readback_called": False,
                                          "readback_receipt_indices": [], "replacement_used": False}]
    row.update(audit_outcome(row, data[0]["tasks"][0]))
    assert "required_common_readback" in codes(run_audit(tmp_path, auditor, data))


@pytest.mark.parametrize("http_count", [0, 1])
def test_required_readback_is_a_call_not_two_guaranteed_successful_gets(tmp_path, auditor, http_count):
    data = fixture()
    row = next(r for r in data[1] if r["mitigation_mode"] == "always_recheck" and r["fault_type"] == "none")
    row["http_receipts"] = row["http_receipts"][:1 + http_count]
    if http_count:
        row["http_receipts"][-1].update(status_code=500, response_payload=None)
    row["additional_http_requests"] = http_count
    row["mitigation_events"][0].update(readback_error="HTTPError", readback_response=None,
        replacement_used=False, readback_receipt_indices=list(range(1, 1 + http_count)))
    row.update(audit_outcome(row, data[0]["tasks"][0]))
    report = run_audit(tmp_path, auditor, data)
    assert report["audit_status"] == "passed", report["findings"]
    assert report["by_mode"]["always_recheck"]["mitigation_recovery_detected"] == 0


@pytest.mark.parametrize("topology,baseline_success", [("sequential", False), ("hierarchical", False), ("flat", True)])
def test_partial_acceptance_can_fail_evidence_or_succeed_via_flat_direct(tmp_path, auditor, topology, baseline_success):
    data = fixture()
    task = data[0]["tasks"][0]
    partial = {"task_id": task["task_id"], "product_title": task["product_title"],
               "requested_quantity": 1, "cart_verified": True, "status": "success"}
    targets = [r for r in data[1] if r["task_id"] == task["task_id"]
               and r["fault_type"] == "valid_partial" and r["topology"] == topology]
    for row in targets:
        row["events"][0]["delivered_message"] = [copy.deepcopy(partial)]
        if row["mitigation_mode"] == "baseline":
            row["primary_evidence"] = copy.deepcopy(partial)
            if topology != "flat":
                row["decision_evidence"] = copy.deepcopy(partial)
        else:
            fresh = row["environment_state"]
            row["http_receipts"] = [cart_receipt(), *receipts(fresh)]
            row["additional_http_requests"] = 2
            row["mitigation_events"] = [{"mode": row["mitigation_mode"], "receiver": row["primary_receiver"],
                "before": copy.deepcopy(partial), "before_issues": ["missing:product_id", "missing:sku", "missing:evidence", "missing:observed_quantity"],
                "readback_called": True, "readback_count": 1, "readback_receipt_indices": [1, 2],
                "readback_response": fresh, "after": fresh, "after_issues": [], "replacement_used": True}]
        row.update(audit_outcome(row, task))
    baseline = next(r for r in targets if r["mitigation_mode"] == "baseline")
    assert baseline["final_task_success"] is baseline_success
    assert "M14_partial_tool_or_message_result_acceptance" in baseline["observed_M_consequence"]
    assert all(r["environment_state"]["cart_verified"] is True for r in targets)
    assert all(r["mitigation_recovery_detected"] is True for r in targets if r is not baseline)
    report = run_audit(tmp_path, auditor, data)
    assert report["audit_status"] == "passed", report["findings"]
    assert report["matrix"]["analysis_runs"] == 90
    assert report["by_mode"]["baseline"]["m_propagated"] == 1
    assert report["by_mode"]["baseline"]["final_task_success"] == (30 if baseline_success else 29)


def test_frozen_stale_file_must_match_embedded_source(tmp_path, auditor):
    path = tmp_path / "input"
    write_snapshot(path, *fixture())
    frozen = next((path / "stale_sources").iterdir())
    frozen.write_text('{"payload": "changed"}')
    assert "stale_frozen_file_mismatch" in codes(auditor.audit_results(path, ROOT))


@pytest.mark.parametrize("exists", [True, False])
def test_empty_stale_file_is_mismatch_but_missing_file_is_unknown(tmp_path, auditor, exists):
    path = tmp_path / "input"
    write_snapshot(path, *fixture())
    frozen = next((path / "stale_sources").iterdir())
    if exists:
        frozen.write_text("{}")
    else:
        frozen.unlink()
    report = auditor.audit_results(path, ROOT)
    if exists:
        assert "stale_frozen_file_mismatch" in codes(report)
        assert report["audit_status"] == "failed"
    else:
        assert "missing_input" in codes(report)
        assert "stale_frozen_file_mismatch" not in codes(report)
        assert report["status"] == "incomplete"


def test_matched_clean_improvement_regression_and_recovery_types(tmp_path, auditor):
    data = fixture()
    targets = [r for r in data[1] if r["fault_type"] == "omission" and r["topology"] == "sequential"]
    for task, mode in (("tea", "baseline"), ("juice", "guarded_recheck")):
        row = next(r for r in targets if r["task_id"] == task and r["mitigation_mode"] == mode)
        row["decision_evidence"] = None
        row["final_answer"]["verdict"] = {"task_id": task, "decision": "reject"}
        add_common_readback(row)
        row.update(audit_outcome(row, next(t for t in data[0]["tasks"] if t["task_id"] == task)))
    report = run_audit(tmp_path, auditor, data)
    comparison = report["pairing"]["comparisons"]["guarded_recheck"]
    assert comparison["improvements"] == 1
    assert comparison["regressions"] == 1
    assert report["by_mode"]["baseline"]["environment_verified"] == 30
    assert report["by_mode"]["baseline"]["final_task_success"] == 29
    assert report["by_mode"]["always_recheck"]["additional_http_requests"] == 60
    assert report["by_mode"]["always_recheck"]["common_recovery_http_requests"] == 0


def test_infrastructure_exclusions_and_inflight_caveat(tmp_path, auditor):
    data = fixture()
    ids = [data[1][0]["run_id"], data[1][1]["run_id"]]
    report = run_audit(tmp_path, auditor, data, infra=[{"completed_run_ids_at_recording": ids}])
    assert report["latency"]["recorded_excluded_run_ids"] == sorted(ids)
    assert len(report["latency"]["effective_excluded_run_ids"]) == 18
    assert report["latency"]["possibly_uncovered_inflight"] is True
    assert sum(v["latency_included_runs"] for v in report["by_mode"].values()) == 72


def test_source_root_is_optional_but_hash_mismatch_fails(tmp_path, auditor):
    path = tmp_path / "input"
    data = fixture()
    data[0]["source_hashes"]["run_shopping_mitigation.py"] = "0" * 64
    write_snapshot(path, *data)
    assert auditor.audit_results(path)["source_hashes"]["status"] == "unknown"
    assert "source_hash_mismatch" in codes(auditor.audit_results(path, source_root=ROOT))


def test_torn_snapshot_tail_is_preserved_and_incomplete(tmp_path, auditor):
    path = tmp_path / "input"
    write_snapshot(path, *fixture())
    journal = path / "main_runs.jsonl"
    with journal.open("ab") as handle:
        handle.write(b'{"run_id":')
    before = journal.read_bytes()
    result = auditor.audit_results(path)
    assert result["status"] == "incomplete"
    assert "torn_jsonl_tail" in codes(result)
    assert journal.read_bytes() == before


def test_cli_writes_chinese_reports_and_refuses_any_existing_output(tmp_path, auditor):
    path, output = tmp_path / "input", tmp_path / "audit"
    write_snapshot(path, *fixture())
    args = [sys.executable, "-B", str(SCRIPT), "--result-dir", str(path),
            "--source-root", str(ROOT), "--output-dir", str(output)]
    result = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    assert {p.name for p in output.iterdir()} == {"audit.json", "audit.md", "paired.csv"}
    assert "审计" in (output / "audit.md").read_text()
    assert "官方" in (output / "audit.md").read_text()
    assert "environment_verified" in (output / "audit.md").read_text()
    with (output / "paired.csv").open() as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 30
    assert {"baseline_run_id", "guarded_recheck_run_id", "always_recheck_run_id"} <= paired[0].keys()
    before = (output / "audit.json").read_bytes()
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0
    assert (output / "audit.json").read_bytes() == before


def test_cli_refuses_empty_existing_output_and_output_inside_snapshot(tmp_path, auditor):
    path = tmp_path / "input"
    write_snapshot(path, *fixture())
    existing = tmp_path / "existing"
    existing.mkdir()
    for output in (existing, path / "new-audit"):
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(SCRIPT),
            "--result-dir", str(path), "--output-dir", str(output)], capture_output=True, text=True)
        assert result.returncode == 2
    assert not (path / "new-audit").exists()
