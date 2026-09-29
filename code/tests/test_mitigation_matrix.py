import copy
import importlib
import json

import pytest

from test_shopping_mitigation import TASK, evidence


def api():
    return importlib.import_module("run_shopping_mitigation")


def readback_receipts():
    return [{"receipt_index": index, "purpose": purpose, "request_method": "GET",
             "request_url": "http://shopping.invalid/readback", "status_code": 200,
             "response_sha256": "a" * 64}
            for index, purpose in enumerate(("reobserve_cart.product_page", "reobserve_cart.items"))]


def row():
    return {"run_id": "unit-run", "fault_type": "none", "condition": "clean", "mitigation_mode": "baseline",
            "task_id": TASK["task_id"], "topology": "sequential", "repeat_index": 1,
            "primary_evidence": evidence(), "decision_evidence": evidence(), "environment_state": evidence(),
            "primary_receiver": "Verifier", "decision_evidence_receiver": "Coordinator",
            "verification_verdict": {"decision": "accept", "task_id": TASK["task_id"]},
            "final_answer": {"verdict": {"decision": "accept", "task_id": TASK["task_id"]}},
            "mitigation_events": [], "final_task_success": True, "observed_M_consequence": ["none"],
            "recovery_detected": False, "recovery_type": "none", "propagation_class": "none",
            "latency_ms": 1, "total_tokens": 2, "additional_http_requests": 0}


def test_minimal_plan_contains_135_unique_jobs_and_three_arms_per_pair():
    tasks = [{**TASK, "task_id": f"task-{i}"} for i in range(3)]
    jobs = api().build_jobs(tasks, 1)
    assert len(jobs) == len({j["job_key"] for j in jobs}) == 135
    assert len({j["pair_key"] for j in jobs}) == 45
    assert {j["topology"] for j in jobs} == {"sequential", "flat", "hierarchical"}
    assert {j["injection_step"] for j in jobs} == {4}
    assert all(j["fault"] == "none" for j in jobs[:27])


def test_plan_rejects_duplicate_tasks():
    with pytest.raises(ValueError, match="duplicate"):
        api().build_jobs([TASK, TASK], 1)


def test_m14_only_for_accepted_partial_state():
    value = row()
    value["primary_evidence"] = value["decision_evidence"] = {"task_id": "task-1", "cart_verified": True}
    result = api().audit_outcome(value, TASK)
    assert "M14_partial_tool_or_message_result_acceptance" in result["observed_M_consequence"]
    value["verification_verdict"] = {"decision": "reject", "task_id": TASK["task_id"]}
    value["final_answer"]["verdict"] = {"decision": "reject", "task_id": TASK["task_id"]}
    assert "M14_partial_tool_or_message_result_acceptance" not in api().audit_outcome(value, TASK)["observed_M_consequence"]


def test_m5_and_m6_are_not_assigned_simply_because_fault_was_injected():
    value = row()
    value["fault_type"] = "stale_replay"
    result = api().audit_outcome(value, TASK)
    assert result["observed_M_consequence"] == ["none"]
    assert result["final_task_success"]


def test_bad_primary_acceptance_is_retained_even_when_flat_final_succeeds():
    value = row()
    value["topology"] = "flat"
    value["primary_evidence"] = {"task_id": "old", "cart_verified": True}
    result = api().audit_outcome(value, TASK)
    assert result["final_task_success"]
    assert "M5_stale_context_acceptance" in result["observed_M_consequence"]
    assert result["propagation_class"] == "silent_propagation_to_M"


def test_wrong_quantity_is_not_m14_when_all_fields_are_present():
    value = row()
    value["primary_evidence"] = None
    value["verification_verdict"] = None
    value["decision_evidence"]["observed_quantity"] = 3
    result = api().audit_outcome(value, TASK)
    assert "M14_partial_tool_or_message_result_acceptance" not in result["observed_M_consequence"]
    assert "M6_state_inconsistency" in result["observed_M_consequence"]
    assert "M4_incorrect_collective_decision" in result["observed_M_consequence"]


def test_consistently_wrong_identity_can_pass_guard_but_fails_external_evaluation():
    value = row()
    altered = evidence()
    altered["product_id"] = "999"
    nested = json.loads(altered["evidence"])
    nested["product_id"] = "999"
    altered["evidence"] = json.dumps(nested)
    value["decision_evidence"] = altered
    assert not api().audit_outcome(value, TASK)["final_task_success"]


def test_readback_attempt_alone_is_not_recovery():
    value = row()
    value["mitigation_events"] = [{"before_issues": ["missing:evidence"], "triggered": True,
                                   "replacement_used": False, "readback_called": True}]
    assert not api().audit_outcome(value, TASK)["recovery_detected"]


def test_recovery_requires_fresh_valid_evidence_used_by_accepting_receiver():
    value = row()
    value["http_receipts"] = readback_receipts()
    value["mitigation_events"] = [{"receiver": "Verifier", "before_issues": ["non_delivery"],
                                   "triggered": True, "replacement_used": True, "readback_called": True,
                                   "readback_response": evidence(), "readback_receipt_indices": [0, 1],
                                   "after": evidence(), "after_issues": []}]
    result = api().audit_outcome(value, TASK)
    assert result["recovery_detected"]
    assert result["propagation_class"] == "detected_and_recovered"


def test_journal_never_reruns_completed_failure_and_caps_error_attempts(tmp_path):
    jobs = api().build_jobs([TASK], 1)
    completed = {**row(), **jobs[0], "final_task_success": False}
    api().append_jsonl(tmp_path / "main_runs.jsonl", completed)
    api().append_jsonl(tmp_path / "run_errors.jsonl", {"job_key": jobs[1]["job_key"], "attempt": 1})
    api().append_jsonl(tmp_path / "run_errors.jsonl", {"job_key": jobs[1]["job_key"], "attempt": 2})
    pending, blocked = api().pending_jobs(jobs, tmp_path)
    assert jobs[0] not in pending and jobs[1] not in pending
    assert jobs[1]["job_key"] in blocked


def test_duplicate_checkpoint_is_rejected(tmp_path):
    job = api().build_jobs([TASK], 1)[0]
    api().append_jsonl(tmp_path / "main_runs.jsonl", job)
    api().append_jsonl(tmp_path / "main_runs.jsonl", job)
    with pytest.raises(ValueError, match="duplicate"):
        api().pending_jobs([job], tmp_path)


def test_paired_summary_does_not_fill_missing_arms_or_call_clean_intervention_false_positive():
    a = {**row(), "pair_key": "x", "mitigation_mode": "baseline"}
    b = {**row(), "pair_key": "x", "mitigation_mode": "guarded_recheck", "final_task_success": False}
    result = api().build_summary([a, b], [])
    assert result["complete_three_arm_pairs"] == 0
    assert result["baseline_guarded_pairs"] == 1
    assert result["clean_success_regressions"] == 1


def test_resume_rejects_changed_manifest(tmp_path):
    config = {"model": "Qwen", "task_sha256": "original"}
    api().freeze_manifest(tmp_path, config, resume=False)
    with pytest.raises(ValueError, match="manifest"):
        api().freeze_manifest(tmp_path, {**config, "task_sha256": "different"}, resume=True)


def test_plan_only_loads_real_manifest_schema_without_model_calls(tmp_path, monkeypatch):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"tasks": [{**TASK, "task_id": f"t-{i}"} for i in range(3)]}))
    output = tmp_path / "plan"
    monkeypatch.setattr(api(), "get_llm_client", lambda: pytest.fail("plan must not call model"))
    monkeypatch.setattr("sys.argv", ["runner", "--task-manifest", str(manifest), "--output-dir", str(output),
                                    "--required-model", "unit-test", "--plan-only"])
    api().main()
    assert len(json.loads((output / "matrix_manifest.json").read_text())["jobs"]) == 135


def test_interrupted_attempts_also_count_against_the_two_attempt_budget(tmp_path):
    job = api().build_jobs([TASK], 1)[0]
    for attempt in (1, 2):
        api().append_jsonl(tmp_path / "run_attempts.jsonl", {"job_key": job["job_key"], "attempt": attempt})
    pending, blocked = api().pending_jobs([job], tmp_path)
    assert pending == []
    assert blocked == [job["job_key"]]


def test_concurrent_runner_cannot_take_the_same_output_lock(tmp_path):
    with api().locked_output(tmp_path):
        with pytest.raises(RuntimeError, match="running"):
            with api().locked_output(tmp_path):
                pytest.fail("second runner acquired lock")


def test_reports_export_json_csv_and_causal_trace_without_model_calls(tmp_path):
    job = api().build_jobs([TASK], 1)[0]
    value = {**row(), **job, "events": [{"event_type": "test-only"}]}
    api().append_jsonl(tmp_path / "main_runs.jsonl", value)
    api().write_reports(tmp_path, [])
    assert all((tmp_path / name).is_file() for name in
               ("summary.json", "summary.md", "main_runs.csv", "traces.jsonl", "representative_traces.md"))
    assert json.loads((tmp_path / "summary.json").read_text())["completed_runs"] == 1


def test_flat_rejected_readback_is_not_credited_for_identical_direct_evidence():
    value = row()
    value["http_receipts"] = readback_receipts()
    value.update(topology="flat", decision_evidence_receiver="Coordinator", fault_type="omission",
                 verification_verdict={"decision": "reject", "task_id": TASK["task_id"]})
    value["mitigation_events"] = [{"receiver": "Verifier", "before_issues": ["non_delivery"],
                                   "triggered": True, "replacement_used": True,
                                   "readback_called": True, "readback_response": evidence(),
                                   "readback_receipt_indices": [0, 1],
                                   "after": evidence(), "after_issues": []}]
    result = api().audit_outcome(value, TASK)
    assert result["final_task_success"]
    assert not result["recovery_detected"]


def test_rejected_stale_evidence_is_task_failure_but_not_stale_acceptance():
    value = row()
    stale = {**evidence(), "task_id": "previous-task"}
    value.update(primary_evidence=stale, decision_evidence=stale, fault_type="stale_replay",
                 verification_verdict={"decision": "reject", "task_id": TASK["task_id"]},
                 final_answer={"verdict": {"decision": "reject", "task_id": TASK["task_id"]}})
    result = api().audit_outcome(value, TASK)
    assert not result["final_task_success"]
    assert "M2_task_timeout_or_failure" in result["observed_M_consequence"]
    assert "M5_stale_context_acceptance" not in result["observed_M_consequence"]
    assert result["propagation_class"] == "propagated_to_M_final_failure"


def test_audit_preserves_agent_claim_but_aligns_derived_success_fields():
    value = row()
    value["final_answer"]["cart_verified"] = True
    value["decision_evidence"]["product_id"] = "wrong"
    result = api().audit_outcome(value, TASK)
    assert result["final_answer"]["verdict"]["decision"] == "accept"
    assert result["legacy_evaluation"]["final_answer"]["cart_verified"] is True
    assert result["final_answer"]["cart_verified"] == result["final_task_success"] is False


def test_inference_settings_freeze_effective_generation_and_deadlines(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:18001/v1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1234")
    monkeypatch.setenv("LLM_DISABLE_THINKING", "true")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", "100")
    value = api().inference_settings()
    assert value == {"api_base_url": "http://127.0.0.1:18001/v1", "temperature": 0,
                     "max_tokens": 1234, "disable_thinking": True,
                     "socket_timeout_seconds": 90, "total_timeout_seconds": 100}


def test_torn_tail_recovery_preserves_original_and_complete_rows(tmp_path):
    path = tmp_path / "main_runs.jsonl"
    original = b'{"job_key":"done"}\n{"job_key":"partial'
    path.write_bytes(original)
    with api().locked_output(tmp_path):
        api().recover_torn_journals(tmp_path)
    assert api().read_jsonl(path) == [{"job_key": "done"}]
    backups = list(tmp_path.glob("main_runs.jsonl.torn-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert api().read_jsonl(tmp_path / "journal_repairs.jsonl")[0]["file"] == "main_runs.jsonl"


def test_corrupt_middle_journal_is_not_silently_repaired(tmp_path):
    path = tmp_path / "main_runs.jsonl"
    original = b'{"job_key":"done"}\ninvalid\n{"job_key":"later"}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError, match="journal"):
        api().recover_torn_journals(tmp_path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("base", ["http://localhost:18001", "http://localhost:18001/v1", "http://localhost:18001/v1/"])
def test_model_preflight_and_completion_share_the_same_v1_base(base, monkeypatch):
    from mas_faults.llm_client import OpenAICompatibleHTTPClient
    from unittest.mock import MagicMock
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"choices":[{"message":{"content":"test"}}]}'
    opened = []
    def urlopen(request, **kwargs):
        opened.append(request.full_url)
        return response
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = OpenAICompatibleHTTPClient(api_key="test", base_url=base, model="test", provider="test")
    client.complete("test")
    assert opened == ["http://localhost:18001/v1/chat/completions"]
    assert api().model_listing_url(base) == "http://localhost:18001/v1/models"
