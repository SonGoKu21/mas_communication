"""Offline unit fixtures only; these are not real Shopping experiment results."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from mas_faults import multimechanism_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]


def auditor_module():
    path = ROOT / "scripts/audit_shopping_multimechanism.py"
    assert path.is_file(), "offline auditor implementation is missing"
    spec = importlib.util.spec_from_file_location("offline_mm_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(i=0):
    return {"task_id": f"offline-{i}", "product_title": f"Offline product {i}",
            "product_url": f"http://shopping.invalid/product-{i}.html",
            "initial_quantity": 1, "quantity": 2}


def payload(t):
    value = {"task_id": t["task_id"], "product_title": t["product_title"],
             "product_id": t["task_id"], "sku": t["task_id"], "requested_quantity": 2,
             "observed_quantity": 2, "cart_verified": True}
    return {**value, "evidence": json.dumps(value)}


def receipt(index, *, after=True):
    product = index % 2 == 0
    return {"receipt_index": index, "request_method": "GET", "status_code": 200,
            "request_url": "http://shopping.invalid/product.html" if product else
            "http://shopping.invalid/rest/V1/guest-carts/[REDACTED]/items",
            "purpose": "reobserve_cart.product_page" if product else "reobserve_cart.items",
            "timestamp": "2026-09-11T12:00:02+00:00" if after else "2026-09-11T12:00:00+00:00",
            "response_sha256": "a" * 64, "guest_cart_id_sha256": "b" * 64,
            "response_payload": {"product_id": "offline-0", "sku": "offline-0"} if product else
            [{"sku": "offline-0", "qty": 2, "name": "Offline product 0", "item_id": 1}]}


def completed(job, digest, t=None, index=1):
    t = t or task()
    p = payload(t)
    verdict = {"task_id": t["task_id"], "decision": "accept"}
    request = {"request_index": index, "model": "Qwen/Qwen3.8-27B", "provider": "modelscope_local",
               "prompt_tokens": 11, "completion_tokens": 4,
               "started_at": "2026-09-11T12:00:00+00:00", "completed_at": "2026-09-11T12:00:01+00:00"}
    value = {**job, "attempt": 1, "attempt_id": f"attempt-{index}", "run_id": f"run-{index}",
             "config_digest": digest, "model": request["model"], "provider": request["provider"],
             "task": t, "environment_state": p, "final_evidence": p, "final_verdict": verdict,
             "common_recovery_enabled": True, "action_contract_valid": True,
             "fault_events": [], "judgments": [], "detection_events": [], "recovery_events": [],
             "events": [{"event_id": 0, "role": "Coordinator", "input": {"task": t},
                         "output": verdict, "extra_model_call": False}],
             "http_receipts": [receipt(0), receipt(1)], "evaluation_receipt_indices": [0, 1],
             "model_requests": [request], "llm_request_log": [request], "model_calls": 1,
             "prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15,
             "known_prompt_tokens": 11, "known_completion_tokens": 4, "known_total_tokens": 15,
             "token_usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
             "usage_complete": True, "cross_task_source": None, "action_ledger_events": [],
             "budget": {"limits": {"get": 4, "model_call": 3, "replay": 1},
                        "used": {"get": 0, "model_call": 0, "replay": 0},
                        "remaining": {"get": 4, "model_call": 3, "replay": 1}},
             "source_evidence": {"task_id": t["task_id"], "session_id": f"session-{index}",
                 "entity_id": f"entity-{index}", "action_id": f"action-{index}", "evidence_id": f"e-{index}",
                 "version": 2, "source": "Worker", "payload": p}}
    value["environment_state"] = {**p, "http_receipt_indices": [0, 1]}
    return auditor_module().load_evaluator()[0](value)


def snapshot(tmp_path, *, tasks=None):
    root = tmp_path / "offline-unit-input"
    root.mkdir()
    ts = tasks or [task()]
    jobs = matrix.build_jobs(ts, 1)
    config = {"version": matrix.VERSION, "runner_schema": 1, "tasks": ts, "jobs": jobs,
              "repetitions": 1, "planned_runs": len(jobs), "shard_runs": len(jobs),
              "shard_index": 0, "shard_count": 1, "max_attempts_per_job": 2,
              "model": "Qwen/Qwen3.8-27B", "provider": "modelscope_local",
              "source_hashes": {}, "inference_settings": {
                  "model": "Qwen/Qwen3.8-27B", "provider": "modelscope_local",
                  "api_base_url": "http://127.0.0.1:18001/v1"}}
    digest = matrix.config_digest(config)
    (root / "matrix_manifest.json").write_text(json.dumps({"config": config, "config_digest": digest}))
    for name in ("main_runs", "run_attempts", "run_errors"):
        (root / f"{name}.jsonl").write_text("")
    return root, config, digest


def write_rows(root, rows, errors=(), starts=None):
    if starts is None:
        starts = [{k: v for k, v in r.items() if k in {
            *matrix.build_jobs([task()], 1)[0], "attempt", "attempt_id", "config_digest", "model", "provider",
            "cross_task_source"}} for r in [*rows, *errors]]
    for name, values in (("main_runs", rows), ("run_errors", errors), ("run_attempts", starts)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in values))


def audit(root):
    return auditor_module().Auditor(root).audit()


def codes(report):
    return {f["code"] for f in report["findings"]}


def test_wrapped_manifest_missing_jobs_is_incomplete_not_completed(tmp_path):
    root, config, _ = snapshot(tmp_path)
    result = audit(root)
    assert result["status"] == "incomplete"
    assert result["coverage"]["planned_runs"] == 210
    assert result["coverage"]["missing_job_keys"] == [j["job_key"] for j in config["jobs"]]
    assert result["evaluation"]["independent_runtime_validation"] is False


@pytest.mark.parametrize("field,value,code", [
    ("config_digest", "wrong", "config_digest_mismatch"),
    ("model", "paid-model", "model_mismatch"),
    ("provider", "deepseek", "provider_mismatch"),
    ("common_recovery_enabled", False, "common_recovery_disabled"),
    ("topology", "flat", "job_metadata_mismatch"),
    ("total_tokens", 999, "token_total_mismatch"),
    ("final_task_success", False, "evaluation_mismatch"),
])
def test_row_integrity_mismatches(tmp_path, field, value, code):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    row[field] = value
    write_rows(root, [row])
    assert code in codes(audit(root))


def test_duplicate_runs_attempts_and_completed_jobs_are_not_silently_deduplicated(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    write_rows(root, [row, row])
    result = audit(root)
    assert {"duplicate_run_id", "duplicate_attempt_id", "duplicate_completed_job"} <= codes(result)
    assert result["status"] != "complete"


def test_unknown_interrupted_attempt_usage_is_not_zero(tmp_path):
    root, config, digest = snapshot(tmp_path)
    start = {**config["jobs"][0], "attempt": 1, "attempt_id": "interrupted", "config_digest": digest,
             "model": config["model"], "provider": config["provider"]}
    write_rows(root, [], starts=[start])
    result = audit(root)
    assert result["tokens"]["total_tokens"] is None
    assert result["tokens"]["unknown_attempts"] == 1
    assert "unfinished_attempt" in codes(result)


def test_exposure_does_not_promote_planned_or_invalid_actions(tmp_path):
    root, config, digest = snapshot(tmp_path)
    jobs = [j for j in config["jobs"] if j["condition"] == "request_non_delivery"][:3]
    rows = [completed(j, digest, index=i + 1) for i, j in enumerate(jobs)]
    event = {"condition": "request_non_delivery", "boundary": "action_request",
             "original_sha256": "a" * 64, "delivered_sha256": [], "delivered_count": 0}
    rows[1]["fault_events"] = [event]
    rows[1]["action_contract_valid"] = False
    rows[2]["fault_events"] = [event]
    rows = [auditor_module().load_evaluator()[0](r) for r in rows]
    write_rows(root, rows)
    result = audit(root)
    assert [r["exposure"] for r in result["cases"]] == ["unreached", "prior_contract_failure", "exposed"]
    assert result["exposure"]["planned_fault_runs"] == 3
    assert result["exposure"]["exposed_fault_runs"] == 1


def test_request_tokens_reconcile_both_logs_and_keep_partial_unknown(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    row["model_requests"][0]["prompt_tokens"] = 20
    write_rows(root, [row])
    assert "request_usage_mismatch" in codes(audit(root))
    row["model_requests"][0]["prompt_tokens"] = None
    row["usage_complete"] = False
    write_rows(root, [row])
    result = audit(root)
    assert result["tokens"]["total_tokens"] is None


def test_evaluator_preserves_history_without_protocol_field_advantage():
    row = completed(matrix.build_jobs([task()], 1)[0], "fixture")
    row["judgments"] = [{"verdict": row["final_verdict"], "accepted_payload": {"task_id": "offline-0"}}]
    result = auditor_module().load_evaluator()[0](row)
    assert result["final_task_success"] is True
    assert result["environment_task_success"] is True
    assert result["evidence_acceptance_errors"] == 1
    assert "M14" in result["observed_M_consequence"]
    assert "evidence_id" not in row["final_evidence"]


def test_cli_requires_new_output_and_never_mutates_inputs(tmp_path):
    root, _, _ = snapshot(tmp_path)
    module = auditor_module()
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    out = tmp_path / "offline-unit-report"
    assert module.main(["--result-dir", str(root), "--output-dir", str(out)]) == 1
    assert {"summary.json", "cases.csv", "pairs.csv", "summary.md"} <= {p.name for p in out.iterdir()}
    assert "显著性" in (out / "summary.md").read_text()
    assert module.main(["--result-dir", str(root), "--output-dir", str(out)]) == 2
    assert module.main(["--result-dir", str(root), "--output-dir", str(root / "audit")]) == 2
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def add_recovery(row, gets=2, *, changed=True, used=True):
    before = copy.deepcopy(row["source_evidence"])
    if changed:
        before["payload"] = {"task_id": row["task_id"]}
    after = copy.deepcopy(row["source_evidence"])
    after["evidence_id"] = "fresh-offline"
    row["http_receipts"] = [receipt(i, after=False) for i in range(gets)] + [receipt(gets), receipt(gets + 1)]
    row["evaluation_receipt_indices"] = [gets, gets + 1]
    row["environment_state"]["http_receipt_indices"] = [gets, gets + 1]
    after["payload"]["http_receipt_indices"] = list(range(gets))
    row["events"][0]["input"]["evidence"] = after if used else before
    digest = auditor_module().load_evaluator()[0].__globals__["evidence_state_digest"]
    row["recovery_events"] = [{"kind": "fixed_readback", "common": False, "verified": True,
        "replacement_used": used, "before": before, "after": after, "receipt_indices": list(range(gets)),
        "before_sha256": digest(before), "after_sha256": digest(after)}]
    row["budget"]["used"]["get"] = gets
    row["budget"]["remaining"]["get"] = 4 - gets
    return auditor_module().load_evaluator()[0](row)


@pytest.mark.parametrize("change,code", [
    ("cap", "mitigation_get_cap"), ("counter", "budget_receipt_mismatch"),
    ("duplicate", "receipt_reference_invalid"), ("missing", "receipt_reference_invalid"),
    ("evaluation_reused", "evaluation_receipt_leak"),
    ("early_evaluation", "evaluation_before_decision"),
    ("private_url", "private_cart_url"), ("private_key", "private_token_field"),
    ("response", "environment_receipt_mismatch"),
])
def test_receipt_budget_privacy_and_truth_isolation(tmp_path, change, code):
    root, config, digest = snapshot(tmp_path)
    row = add_recovery(completed(config["jobs"][1], digest), gets=6 if change == "cap" else 2)
    if change == "counter":
        row["budget"]["used"]["get"] = 0
    elif change == "duplicate":
        row["recovery_events"][0]["receipt_indices"] = [0, 0]
    elif change == "missing":
        row["recovery_events"][0]["receipt_indices"] = [0, 99]
    elif change == "evaluation_reused":
        row["events"][0]["input"]["leaked"] = {"http_receipt_indices": [2, 3]}
    elif change == "early_evaluation":
        row["http_receipts"][2]["timestamp"] = "2026-09-11T11:00:00+00:00"
    elif change == "private_url":
        row["http_receipts"][3]["request_url"] = "http://shopping.invalid/rest/V1/guest-carts/PRIVATE-SECRET/items"
    elif change == "private_key":
        row["events"][0]["input"]["guest_cart_id"] = "PRIVATE-SECRET"
    elif change == "response":
        row["http_receipts"][3]["response_payload"][0]["qty"] = 88
    write_rows(root, [row])
    report = audit(root)
    assert code in codes(report)
    assert "PRIVATE-SECRET" not in json.dumps(report)


def test_meaningful_recovery_requires_used_payload_change_and_real_receipts(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = add_recovery(completed(config["jobs"][1], digest))
    write_rows(root, [row])
    assert audit(root)["cases"][0]["receipt_backed_recovery"] is True
    row = add_recovery(completed(config["jobs"][1], digest), changed=False)
    write_rows(root, [row])
    assert audit(root)["cases"][0]["receipt_backed_recovery"] is False
    row = add_recovery(completed(config["jobs"][1], digest), used=False)
    write_rows(root, [row])
    assert audit(root)["cases"][0]["receipt_backed_recovery"] is False


def test_extra_llm_count_from_events_not_budget_claim(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    row["events"] *= 4
    row["events"] = [{**e, "event_id": i, "extra_model_call": True} for i, e in enumerate(row["events"])]
    write_rows(root, [row])
    result = audit(root)
    assert {"mitigation_model_cap", "budget_event_mismatch", "model_event_count_mismatch"} <= codes(result)
    assert result["cases"][0]["mitigation_model_calls"] == 4


def test_action_replay_needs_nonexecution_proof_and_actual_write_receipt(tmp_path):
    root, config, digest = snapshot(tmp_path)
    job = next(j for j in config["jobs"] if j["condition"] == "request_non_delivery" and j["arm"] == "action_protocol")
    row = completed(job, digest)
    row["detection_events"] = [{"kind": "missing_action_receipt", "action_id": "a"}]
    row["budget"]["used"]["replay"] = 1
    row["budget"]["remaining"]["replay"] = 0
    write_rows(root, [row])
    result = audit(root)
    assert "replay_authorization_unproven" in codes(result)
    assert result["cases"][0]["action_replays"] == 1
    assert result["cases"][0]["write_requests"] == 0


def test_pairs_require_seven_arms_and_common_clean_success(tmp_path):
    root, config, digest = snapshot(tmp_path)
    jobs = [j for j in config["jobs"] if j["topology"] == "sequential" and
            j["condition"] in {"clean", "valid_partial"}]
    rows = [completed(j, digest, index=i + 1) for i, j in enumerate(jobs)]
    for row in rows:
        if row["condition"] == "valid_partial":
            row["fault_events"] = [{"condition": "valid_partial", "boundary": "evidence_handoff",
                "original_sha256": "a" * 64, "delivered_sha256": ["b" * 64], "delivered_count": 1}]
            if row["arm"] == "baseline":
                row["final_verdict"] = {"task_id": row["task_id"], "decision": "reject"}
    rows = [auditor_module().load_evaluator()[0](r) for r in rows]
    write_rows(root, rows)
    report = audit(root)
    assert report["pairing"]["complete_seven_arm_pairs"] == 2
    pairs = [p for p in report["pairs"] if p["condition"] == "valid_partial"]
    assert len(pairs) == 6
    assert all(p["common_clean_success"] and p["all_fault_arms_exposed"] for p in pairs)
    assert all(p["final_task_success_change"] == "improvement" for p in pairs)
    assert all(p["environment_task_success_change"] == "unchanged" for p in pairs)
    assert report["comparisons"]["common_clean_success"]["combined"]["final_task_success"]["improvements"] == 1
    rows[0]["final_verdict"] = {"task_id": "wrong", "decision": "accept"}
    rows[0] = auditor_module().load_evaluator()[0](rows[0])
    write_rows(root, rows)
    assert not any(p["common_clean_success"] for p in audit(root)["pairs"])
    write_rows(root, rows[:-1])
    assert audit(root)["pairing"]["complete_seven_arm_pairs"] == 1


def test_source_provenance_requires_file_hash_and_actual_other_clean(tmp_path):
    root, config, digest = snapshot(tmp_path, tasks=[task(), task(1)])
    source_job = next(j for j in config["jobs"] if j["task_id"] == "offline-1" and
                      j["condition"] == "clean" and j["arm"] == "baseline")
    target_job = next(j for j in config["jobs"] if j["task_id"] == "offline-0" and j["condition"] == "cross_task_replay")
    source = completed(source_job, digest, task(1))
    target = completed(target_job, digest, index=2)
    frozen = {"config_digest": digest, "target": [target["task_id"], target["topology"], target["repeat_index"]],
              "source_task_id": source["task_id"], "source_run_id": source["run_id"],
              "source_job_key": source["job_key"], "envelope": source["source_evidence"],
              "envelope_sha256": matrix.config_digest(source["source_evidence"])}
    path = root / "cross_task_sources" / (matrix.config_digest(frozen["target"]) + ".json")
    path.parent.mkdir()
    path.write_text(json.dumps(frozen))
    target["cross_task_source"] = {**frozen, "file": str(path.relative_to(root)),
                                 "file_sha256": auditor_module().sha256(path.read_bytes())}
    target["fault_events"] = [{"condition": "cross_task_replay", "boundary": "evidence_handoff",
        "original_sha256": "a" * 64, "delivered_sha256": [frozen["envelope_sha256"]], "delivered_count": 1,
        "source_evidence_id": source["source_evidence"]["evidence_id"], "source_sha256": frozen["envelope_sha256"]}]
    target = auditor_module().load_evaluator()[0](target)
    write_rows(root, [source, target])
    assert audit(root)["cases"][1]["cross_source_verified"] is True
    path.write_text(json.dumps({**frozen, "source_run_id": "fabricated"}))
    result = audit(root)
    assert "cross_source_file_hash_mismatch" in codes(result)
    assert result["cases"][1]["cross_source_verified"] is False


def test_summary_counts_are_reconciled_and_torn_tail_not_repaired(tmp_path):
    root, _, _ = snapshot(tmp_path)
    (root / "summary.json").write_text(json.dumps({"completed_runs": 6300, "status": "complete"}))
    (root / "main_runs.jsonl").write_text('{"unfinished":')
    result = audit(root)
    assert {"summary_count_mismatch", "torn_jsonl_tail"} <= codes(result)
    assert (root / "main_runs.jsonl").read_text() == '{"unfinished":'


def test_optional_runtime_guard_and_unknown_action_are_reported_without_requiring_them(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    write_rows(root, [row])
    result = audit(root)
    assert not [f for f in result["findings"] if f["severity"] == "error"]
    assert result["cases"][0]["final_task_success"] is True
    row.update(final_commit_allowed=False, action_outcome_unknown=True)
    row = auditor_module().load_evaluator()[0](row)
    write_rows(root, [row])
    result = audit(root)
    assert result["cases"][0]["action_outcome_unknown"] is True
    assert result["cases"][0]["final_commit_allowed"] is False
    assert result["outcomes"]["unresolved_completed_trials"] == 1
    assert result["coverage"]["error_attempts"] == 0


def test_explicit_nonapplied_fault_not_exposure_and_prior_invalid_is_separate(tmp_path):
    root, config, digest = snapshot(tmp_path)
    job = next(j for j in config["jobs"] if j["condition"] == "request_non_delivery")
    row = completed(job, digest)
    row["fault_events"] = [{"condition": job["condition"], "boundary": job["boundary"], "applied": False,
        "original_sha256": "a" * 64, "delivered_count": 0, "delivered_sha256": []}]
    write_rows(root, [row])
    assert audit(root)["cases"][0]["exposure"] != "exposed"
    row.update(fault_events=[], action_contract_valid=False)
    write_rows(root, [row])
    assert audit(root)["cases"][0]["exposure"] == "prior_contract_failure"


def test_local_replay_requires_actual_earlier_source_hash(tmp_path):
    root, config, digest = snapshot(tmp_path)
    job = next(j for j in config["jobs"] if j["condition"] == "stale_judgment_replay")
    row = completed(job, digest)
    row["fault_events"] = [{"condition": job["condition"], "boundary": job["boundary"],
        "original_sha256": "a" * 64, "delivered_count": 1, "delivered_sha256": ["b" * 64],
        "source_sha256": "b" * 64, "source_judgment_id": "invented-source"}]
    write_rows(root, [row])
    assert "local_replay_source_unproven" in codes(audit(root))


def test_inference_config_is_checked_and_failed_attempt_costs_are_separate(tmp_path):
    root, config, digest = snapshot(tmp_path)
    config["inference_settings"]["api_base_url"] = "https://paid.example/v1"
    digest = matrix.config_digest(config)
    (root / "matrix_manifest.json").write_text(json.dumps({"config": config, "config_digest": digest}))
    row = completed(config["jobs"][0], digest)
    error = {**row, "status": "infra_error", "error_type": "OfflineUnitFailure", "usage_complete": False}
    write_rows(root, [], [error])
    result = audit(root)
    assert "nonlocal_inference_endpoint" in codes(result)
    assert result["tokens"]["total_tokens"] is None
    assert result["outcomes"]["infra_error_attempts"] == 1
    assert result["attempt_cases"][0]["known_total_tokens"] == 15


def test_nonfinite_duplicate_keys_and_wrong_legacy_schema_rejected(tmp_path):
    module = auditor_module()
    for value in ('{"x": 1, "x": 2}', '{"x": NaN}'):
        with pytest.raises(ValueError):
            module.strict_json(value)
    root, config, _ = snapshot(tmp_path)
    (root / "matrix_manifest.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config_digest"):
        audit(root)


def test_reports_include_separate_outcomes_pair_cases_and_no_significance(tmp_path):
    root, config, digest = snapshot(tmp_path)
    rows = [completed(j, digest, index=i + 1) for i, j in enumerate(config["jobs"][:7])]
    write_rows(root, rows)
    out = tmp_path / "new-report"
    assert auditor_module().main(["--result-dir", str(root), "--output", str(out), "--no-strict"]) == 0
    md = (out / "summary.md").read_text()
    assert "环境成功" in md and "历史错误接受" in md and "代表配对" in md
    assert "run-1" in md
    assert (out / "attempts.csv").is_file()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["coverage"]["planned_clean_runs"] == 21
    assert summary["coverage"]["planned_fault_runs"] == 189
    assert summary["statistical_inference"]["performed"] is False


@pytest.mark.parametrize("change,code", [
    ("cart", "cart_identity_mismatch"), ("time", "recovery_after_decision"),
    ("method", "recovery_receipt_invalid"), ("hash", "recovery_payload_hash_mismatch"),
    ("common", "common_recovery_trigger_unproven"),
])
def test_recovery_proof_rejects_bad_cart_timing_method_hash_and_trigger(tmp_path, change, code):
    root, config, digest = snapshot(tmp_path)
    row = add_recovery(completed(config["jobs"][1], digest))
    if change == "cart":
        row["http_receipts"][1]["guest_cart_id_sha256"] = "c" * 64
    elif change == "time":
        row["http_receipts"][0]["timestamp"] = "2026-09-11T12:01:00+00:00"
    elif change == "method":
        row["http_receipts"][0]["request_method"] = "POST"
    elif change == "hash":
        row["recovery_events"][0]["before_sha256"] = "invented"
    elif change == "common":
        row["recovery_events"][0].update(common=True, kind="common_recovery")
    write_rows(root, [row])
    result = audit(root)
    assert code in codes(result)
    assert result["cases"][0]["receipt_backed_recovery"] is False


def test_frozen_core_hashes_and_local_evaluator_provenance_are_required(tmp_path):
    root, config, _ = snapshot(tmp_path)
    config["source_hashes"] = {"run_shopping_multimechanism.py": "a" * 64,
        "src/mas_faults/shopping_multimechanism.py": "b" * 64}
    digest = matrix.config_digest(config)
    (root / "matrix_manifest.json").write_text(json.dumps({"config": config, "config_digest": digest}))
    report = auditor_module().Auditor(root, ROOT).audit()
    assert {"core_source_hash_missing", "source_hash_mismatch", "evaluator_source_mismatch"} <= codes(report)


def test_output_path_symlink_to_source_is_rejected(tmp_path):
    root, _, _ = snapshot(tmp_path)
    link = tmp_path / "not-new"
    link.symlink_to(root, target_is_directory=True)
    assert auditor_module().main(["--result-dir", str(root), "--output", str(link)]) == 2


def test_regressions_not_only_improvements_and_error_usage_stays_visible(tmp_path):
    root, config, digest = snapshot(tmp_path)
    rows = [completed(j, digest, index=i + 1) for i, j in enumerate(config["jobs"][:7])]
    combined = next(r for r in rows if r["arm"] == "combined")
    combined["final_verdict"]["decision"] = "reject"
    rows = [auditor_module().load_evaluator()[0](r) for r in rows]
    write_rows(root, rows)
    report = audit(root)
    assert report["comparisons"]["all_samples"]["combined"]["final_task_success"]["regressions"] == 1
    assert report["by_cell"][0]["runs"] == 7


def test_unknown_attempt_usage_not_summarized_as_complete_from_claimed_totals(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    row.pop("model_requests")
    write_rows(root, [row])
    result = audit(root)
    assert result["tokens"]["total_tokens"] is None
    assert result["tokens"]["unknown_attempts"] == 1


def test_blocked_and_exhausted_frozen_jobs_remain_in_coverage(tmp_path):
    root, config, digest = snapshot(tmp_path)
    errors = []
    for attempt in (1, 2):
        row = completed(config["jobs"][0], digest, index=attempt)
        row.update(attempt=attempt, status="infra_error", usage_complete=False)
        errors.append(row)
    write_rows(root, [], errors)
    (root / "blocked_cells.jsonl").write_text(json.dumps({"pair_key": config["jobs"][7]["pair_key"],
        "config_digest": digest, "reason": "missing_actual_baseline_clean_source"}) + "\n")
    result = audit(root)
    assert result["coverage"]["exhausted_job_keys"] == [config["jobs"][0]["job_key"]]
    assert result["coverage"]["blocked_pair_keys"] == [config["jobs"][7]["pair_key"]]


def test_unavailable_final_observation_is_unknown_not_a_fabricated_failure(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    row["environment_state"] = {"status": "observation_unavailable", "error_type": "OfflineTimeout"}
    row["http_receipts"][1].update(status_code=None, response_sha256=None, response_payload=None)
    row["action_outcome_unknown"] = True
    row = auditor_module().load_evaluator()[0](row)
    write_rows(root, [row])
    result = audit(root)
    assert result["cases"][0]["environment_task_success"] is None
    assert result["cases"][0]["decision_correct"] is None
    assert "evaluation_observation_unavailable" in codes(result)
    assert "evaluation_receipt_invalid" not in codes(result)


def test_observed_delivery_receipts_count_real_writes_and_reject_bad_links(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    writes = [{**receipt(i, after=False), "request_method": "POST", "purpose": "add_quantity.add_item"}
              for i in (0, 1)]
    row["http_receipts"] = writes + [receipt(2), receipt(3)]
    row["evaluation_receipt_indices"] = [2, 3]
    row["environment_state"]["http_receipt_indices"] = [2, 3]
    row["action_ledger_events"] = [
        {"event": "observed_delivery_confirmed", "action_id": "action", "details": {
            "delivery_id": f"delivery-{i}", "receipt": {"http_receipt_indices": [i]}}} for i in (0, 1)]
    write_rows(root, [row])
    result = audit(root)
    assert result["cases"][0]["ledger_confirmed_deliveries"] == 2
    assert result["cases"][0]["duplicate_adjustment_write_receipts"] == 1
    row["action_ledger_events"][0]["details"]["receipt"]["http_receipt_indices"] = [99]
    write_rows(root, [row])
    assert "receipt_reference_invalid" in codes(audit(root))


def runner_reports(root, config, rows, errors=()):
    """Exercise only the real runner's offline report writer, never execution."""
    from run_shopping_multimechanism import write_reports

    return write_reports(root, config["jobs"], {"rows": rows, "errors": list(errors),
        "exhausted": [], "consecutive_error_attempts": 0})


def test_runner_produced_seven_clean_summary_matches_and_preserves_ledger_unknowns(tmp_path):
    root, config, digest = snapshot(tmp_path)
    rows = [completed(j, digest, index=i + 1) for i, j in enumerate(config["jobs"][:7])]
    rows[0]["action_ledger_events"] = [{"event": "confirmed", "details": {}}]
    write_rows(root, rows)
    runner_summary = runner_reports(root, config, rows)
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    result = audit(root)
    assert not [f for f in result["findings"] if f["severity"] == "error"]
    assert result["status"] == "incomplete"
    assert result["coverage"]["completed_runs"] == 7
    assert result["tokens"]["total_tokens"] == runner_summary["total_tokens_completed"] == 105
    assert {"ledger_delivery_id_unknown", "ledger_confirmation_receipts_unknown"} <= codes(result)
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize("scope,mutation", [
    ("arm", "increment"), ("arm", "remove"), ("arm", "scalar"), ("arm", "bool"),
    ("total", "increment"),
])
def test_runner_exposure_summary_tampering_still_fails(tmp_path, scope, mutation):
    root, config, digest = snapshot(tmp_path)
    rows = [completed(config["jobs"][0], digest)]
    write_rows(root, rows)
    summary = runner_reports(root, config, rows)
    container = summary["by_arm"]["baseline"] if scope == "arm" else summary
    if mutation == "increment":
        container["exposure"]["clean_runs"] += 1
    elif mutation == "remove":
        del container["exposure"]["clean_runs"]
    elif mutation == "bool":
        container["exposure"]["clean_runs"] = True
    else:
        container["exposure"] = 1
    (root / "summary.json").write_text(json.dumps(summary))
    result = audit(root)
    prefix = "by_arm.baseline.exposure" if scope == "arm" else "exposure"
    findings = [f for f in result["findings"] if f["code"] == "summary_count_mismatch"]
    assert any(f.get("field", "").startswith(prefix) for f in findings)
    assert not any(f.get("field", "").startswith("by_arm.always_recheck") for f in findings)


def test_runner_exposure_summary_covers_fault_categories_and_error_attempts(tmp_path):
    root, config, digest = snapshot(tmp_path)
    jobs = [j for j in config["jobs"] if j["condition"] == "request_non_delivery"][:5]
    rows = [completed(j, digest, index=i + 1) for i, j in enumerate(jobs)]
    event = {"condition": "request_non_delivery", "boundary": "action_request",
             "original_sha256": "a" * 64, "delivered_sha256": [], "delivered_count": 0}
    rows[0]["fault_events"] = [event]
    rows[1].update(fault_events=[event], action_contract_valid=False)
    rows[2].update(fault_events=[], action_contract_valid=False)
    rows[3]["fault_events"] = [{"condition": "request_non_delivery"}]
    rows = [auditor_module().load_evaluator()[0](r) for r in rows]
    error = completed(jobs[0], digest, index=99)
    error.update(attempt=2, status="infra_error", usage_complete=False)
    write_rows(root, rows, [error])
    runner_reports(root, config, rows, [error])
    result = audit(root)
    # Malformed trace and retry-after-completion remain errors, but summary
    # reconciliation must compare the runner's counters, not a different schema.
    assert not [f for f in result["findings"] if f["code"] == "summary_count_mismatch"]
    assert "fault_event_mismatch" in codes(result)
    assert result["tokens"]["total_tokens"] is None


@pytest.mark.parametrize("task_count,repetitions", [(3, 1), (10, 3)])
def test_limitations_describe_frozen_scale_not_a_static_pilot(tmp_path, task_count, repetitions):
    root, config, _ = snapshot(tmp_path, tasks=[task(i) for i in range(task_count)])
    jobs = matrix.build_jobs(config["tasks"], repetitions)
    config.update(repetitions=repetitions, jobs=jobs, planned_runs=len(jobs), shard_runs=len(jobs))
    (root / "matrix_manifest.json").write_text(json.dumps({"config": config, "config_digest": matrix.config_digest(config)}))
    result = audit(root)
    text = "\n".join(result["limitations"])
    assert f"{task_count} 个任务实例" in text
    assert f"{repetitions} 次重复" in text
    assert "不作统计显著性" in text
    assert result["evaluation"]["limitations"] == result["limitations"]
    if task_count == 10:
        assert "三任务一次重复" not in text
        assert "3 个任务实例" not in text
    out = tmp_path / "offline-unit-dynamic-report"
    auditor_module().write_reports(out, result)
    assert text in (out / "summary.md").read_text()


def action_protocol_row(kind, digest="offline-unit", *, negative=False):
    """Minimal JSON v1 ledger/HTTP trace, not a live execution or experiment."""
    evaluate_trial, _ = auditor_module().load_evaluator()

    condition = {"request_redelivery": "request_non_delivery",
                 "acknowledgement_retrieval": "acknowledgement_loss",
                 "duplicate_prevention": "duplicate_action_delivery"}[kind]
    job = next(j for j in matrix.build_jobs([task()], 1) if j["condition"] == condition
               and j["topology"] == "sequential" and j["arm"] == "action_protocol")
    row = completed(job, digest)
    params = {"task_id": task()["task_id"], "session_id": "offline-session", "action_id": "offline-action",
              "operation": "add_quantity", "quantity": 1}
    row.update(session_id=params["session_id"], action_id=params["action_id"])
    p = payload(task())
    if negative:
        p.update(observed_quantity=1, cart_verified=False)
        p["evidence"] = json.dumps({k: v for k, v in p.items() if k != "evidence"})
    p["http_receipt_indices"] = [0, 1, 2, 3]
    row["action_ledger_state"] = {**{k: params[k] for k in ("task_id", "session_id", "action_id")},
        "params": params, "params_sha256": matrix.config_digest(params), "state": "confirmed",
        "execution_count": 1, "receipt": copy.deepcopy(p)}
    row["action_ledger_events"] = [
        {"event_id": 1, "event": "registered", "action_id": params["action_id"],
         "details": {"params_sha256": matrix.config_digest(params)}},
        {"event_id": 2, "event": "claimed", "action_id": params["action_id"],
         "details": {"delivery_id": "offline-delivery", "mode": "guarded"}},
        {"event_id": 3, "event": "execute_entered", "action_id": params["action_id"],
         "details": {"delivery_id": "offline-delivery", "mode": "guarded"}},
        {"event_id": 4, "event": "confirmed", "action_id": params["action_id"],
         "details": {"delivery_id": "offline-delivery", "receipt": copy.deepcopy(p)}}]
    if kind == "duplicate_prevention":
        row["action_ledger_events"].append({"event_id": 5, "event": "confirmed_receipt_replayed",
                                          "action_id": params["action_id"], "details": {}})
    row["http_receipts"] = [receipt(i, after=i >= 4) for i in range(6)]
    for i, purpose in enumerate(("product_page", "lookup_items", "add_item", "readback_items")):
        row["http_receipts"][i]["purpose"] = "add_quantity." + purpose
    write = row["http_receipts"][2]
    write.update(request_method="POST", request_url=row["http_receipts"][1]["request_url"],
                 request_payload={"cartItem": {"sku": p["sku"], "qty": 1}},
                 response_payload={"sku": p["sku"], "qty": p["observed_quantity"]})
    row["http_receipts"][1]["response_payload"][0]["qty"] = 1
    for i in (3, 5):
        row["http_receipts"][i]["response_payload"][0]["qty"] = p["observed_quantity"]
    row["environment_state"] = {**copy.deepcopy(p), "http_receipt_indices": [4, 5]}
    row["evaluation_receipt_indices"] = [4, 5]
    row["final_evidence"] = copy.deepcopy(p)
    if negative:
        row["final_verdict"]["decision"] = "reject"
    row["events"].insert(0, {"event_id": 0, "role": "ActionExecutor", "input": [copy.deepcopy(params)],
        "output": [{"action_id": params["action_id"], "receipt": copy.deepcopy(p)}], "action_valid": True,
        **{k: params[k] for k in ("task_id", "session_id", "action_id")}})
    row["events"][1]["event_id"] = 1
    if kind == "duplicate_prevention":
        row["events"][0]["input"] *= 2
    detection = {"request_redelivery": "missing_action_receipt", "acknowledgement_retrieval": "confirmation_retrieved",
                 "duplicate_prevention": "duplicate_prevented"}[kind]
    row["detection_events"] = [{"kind": detection, "action_id": params["action_id"]}]
    row["action_outcome_unknown"] = False
    is_redelivery = kind == "request_redelivery"
    row["action_protocol_events"] = [{"schema_version": 1, "event_id": 0, "kind": kind,
        "outcome": "prevention" if kind == "duplicate_prevention" else "recovery",
        **{k: params[k] for k in ("task_id", "session_id", "action_id")},
        "before_state": "pending" if is_redelivery else "confirmed",
        "before_execution_count": 0 if is_redelivery else 1, "after_state": "confirmed", "after_execution_count": 1,
        "ledger_event_ids": [e["event_id"] for e in row["action_ledger_events"]],
        "receipt_sha256": matrix.config_digest(p), "receipt_indices": [0, 1, 2, 3],
        "new_http_receipt_indices": [0, 1, 2, 3] if is_redelivery else [], "response_used": True}]
    if is_redelivery:
        row["budget"]["used"]["replay"] = 1
        row["budget"]["remaining"]["replay"] = 0
    delivered = [matrix.config_digest(params)] * (2 if kind == "duplicate_prevention" else 0)
    row["fault_events"] = [{"condition": condition, "boundary": job["boundary"],
        "original_sha256": matrix.config_digest(params), "delivered_count": len(delivered), "delivered_sha256": delivered}]
    return evaluate_trial(row)


def test_offline_evaluator_loads_pure_action_verifier():
    evaluate, _ = auditor_module().load_evaluator()
    assert "verify_action_protocol_event" in evaluate.__globals__
    assert not {"ActionLedger", "run_trial", "requests", "urllib"} & evaluate.__globals__.keys()
    row = action_protocol_row("request_redelivery")
    assert evaluate(row)["action_recovery_count"] == 1


@pytest.mark.parametrize("kind", ["request_redelivery", "acknowledgement_retrieval", "duplicate_prevention"])
def test_action_protocol_v1_independent_counts_and_prevention_class(tmp_path, kind):
    root, _, digest = snapshot(tmp_path)
    row = action_protocol_row(kind, digest)
    assert row["action_protocol_events"][0]["verified"] is True
    original = copy.deepcopy(row)
    write_rows(root, [row])
    result = audit(root)
    assert not [f for f in result["findings"] if f["severity"] == "error"]
    case = result["cases"][0]
    prevention = int(kind == "duplicate_prevention")
    assert case["receipt_backed_action_prevention_count"] == prevention
    assert case["receipt_backed_action_recovery_count"] == 1 - prevention
    assert case["receipt_backed_readback_recovery_count"] == 0
    assert case["receipt_backed_recovery"] is False
    assert case["propagation_class"] == ("detected_and_prevented" if prevention else "detected_and_recovered")
    assert result["action_protocol"]["verified_preventions"] == prevention
    assert result["action_protocol"]["verified_recoveries"] == 1 - prevention
    assert row == original


@pytest.mark.parametrize("damage", ["ledger_missing", "unknown", "session", "duplicate_ledger_id", "hidden_execution",
    "write_qty", "write_cart", "readback_qty", "response_hash", "receipt_hash", "params_hash",
    "evaluation_receipts", "unused", "schema", "new_http", "wrong_arm", "missing_detection"])
def test_action_protocol_false_claims_cannot_earn_independent_credit(tmp_path, damage):
    root, _, digest = snapshot(tmp_path)
    row = action_protocol_row("request_redelivery", digest)
    event = row["action_protocol_events"][0]
    if damage == "ledger_missing":
        row["action_ledger_events"] = []
    elif damage == "unknown":
        row["action_outcome_unknown"] = True
    elif damage == "session":
        row["action_ledger_state"]["session_id"] = "different"
    elif damage == "duplicate_ledger_id":
        row["action_ledger_events"].append(copy.deepcopy(row["action_ledger_events"][-1]))
    elif damage == "hidden_execution":
        row["action_ledger_events"].append({**row["action_ledger_events"][2], "event_id": 99})
    elif damage == "write_qty":
        row["http_receipts"][2]["request_payload"]["cartItem"]["qty"] = 99
    elif damage == "write_cart":
        row["http_receipts"][2]["guest_cart_id_sha256"] = "c" * 64
    elif damage == "readback_qty":
        row["http_receipts"][3]["response_payload"][0]["qty"] = 99
    elif damage == "response_hash":
        row["http_receipts"][2]["response_sha256"] = "z" * 64
    elif damage == "receipt_hash":
        event["receipt_sha256"] = "a" * 64
    elif damage == "params_hash":
        row["action_ledger_state"]["params"]["quantity"] = 99
    elif damage == "evaluation_receipts":
        event["receipt_indices"] = [4, 5]
    elif damage == "unused":
        row["events"][0]["output"] = []
    elif damage == "schema":
        event["schema_version"] = 2
    elif damage == "new_http":
        event["new_http_receipt_indices"] = []
    elif damage == "wrong_arm":
        row["arm"] = "baseline"
    else:
        row["detection_events"] = []
    write_rows(root, [row])
    result = audit(root)
    assert result["cases"][0]["receipt_backed_action_recovery_count"] == 0
    assert "action_protocol_event_invalid" in codes(result)


def test_action_checker_does_not_delegate_independent_validation_to_runtime():
    module = auditor_module()
    row = action_protocol_row("request_redelivery")
    row["http_receipts"][2]["response_sha256"] = "z" * 64
    auditor = module.Auditor(ROOT)
    auditor.evaluate.__globals__["verify_action_protocol_event"] = lambda *_: True
    assert auditor.action_protocol(row)["receipt_backed_action_recovery_count"] == 0


def test_action_prevention_duplicate_events_do_not_double_count(tmp_path):
    root, _, digest = snapshot(tmp_path)
    row = action_protocol_row("duplicate_prevention", digest)
    row["action_protocol_events"].append({**copy.deepcopy(row["action_protocol_events"][0]), "event_id": 1})
    write_rows(root, [row])
    result = audit(root)
    assert result["cases"][0]["receipt_backed_action_prevention_count"] == 1
    assert "action_protocol_duplicate" in codes(result)


def test_action_response_recovery_is_not_task_success(tmp_path):
    root, _, digest = snapshot(tmp_path)
    row = action_protocol_row("request_redelivery", digest, negative=True)
    write_rows(root, [row])
    case = audit(root)["cases"][0]
    assert case["receipt_backed_action_recovery_count"] == 1
    assert case["environment_task_success"] is False
    assert case["final_task_success"] is False


def test_action_derived_fields_and_saved_verified_flags_are_compared(tmp_path):
    root, _, digest = snapshot(tmp_path)
    row = action_protocol_row("duplicate_prevention", digest)
    row["action_prevention_count"] = 0
    row["action_protocol_events"][0]["verified"] = False
    write_rows(root, [row])
    result = audit(root)
    assert any(f["code"] == "evaluation_mismatch" and f.get("field") == "action_prevention_count" for f in result["findings"])
    assert "action_protocol_verification_mismatch" in codes(result)


def test_legacy_missing_action_trace_is_unknown_not_required_protocol_failure(tmp_path):
    root, config, digest = snapshot(tmp_path)
    row = completed(config["jobs"][0], digest)
    for key in ("action_protocol_events", "action_recovery_count", "action_prevention_count", "prevention_detected", "prevention_type"):
        row.pop(key, None)
    write_rows(root, [row])
    result = audit(root)
    assert result["cases"][0]["receipt_backed_action_recovery_count"] is None
    assert not [f for f in result["findings"] if f["severity"] == "error"]


def test_action_and_readback_recovery_counts_remain_separate_in_reports(tmp_path):
    root, _, digest = snapshot(tmp_path)
    row = action_protocol_row("request_redelivery", digest)
    write_rows(root, [row])
    report = audit(root)
    out = tmp_path / "offline-action-report"
    auditor_module().write_reports(out, report)
    assert "动作响应恢复" in (out / "summary.md").read_text()
    assert "重复预防" in (out / "summary.md").read_text()
    assert report["by_arm"]["action_protocol"]["receipt_backed_action_recovery_count"] == 1


def test_offline_evaluator_includes_consequence_helper_and_its_pure_dependency():
    evaluate, _ = auditor_module().load_evaluator()
    assert {"verify_action_consequences", "validate_action"} <= evaluate.__globals__.keys()
    assert not {"ActionLedger", "run_trial", "requests", "urllib"} & evaluate.__globals__.keys()


def test_offline_known_task_failure_with_correct_rejection_does_not_become_m4():
    evaluate, _ = auditor_module().load_evaluator()
    row = action_protocol_row("request_redelivery", negative=True)
    result = evaluate(row)
    assert result["decision_correct"] is True
    assert "task_incomplete" in result["observed_M_consequence"]
    assert "M4" not in result["observed_M_consequence"]
    assert result["action_recovery_count"] == 1
    row["environment_state"] = {"status": "observation_unavailable"}
    assert "task_incomplete" not in evaluate(row)["observed_M_consequence"]


def test_offline_failed_delegation_requires_bound_pending_empty_delivery_not_fault_label():
    evaluate, _ = auditor_module().load_evaluator()
    row = action_protocol_row("request_redelivery", negative=True)
    row.update(arm="baseline", action_protocol_events=[], detection_events=[], condition="clean", fault_events=[])
    row["action_ledger_state"].update(state="pending", execution_count=0, receipt=None)
    row["action_ledger_events"] = row["action_ledger_events"][:1]
    row["events"][0].update(input=[], output=[])
    result = evaluate(row)
    assert set(result["observed_M_consequence"]) == {"task_incomplete", "failed_delegation"}
    row["events"][0]["session_id"] = "other"
    assert "failed_delegation" not in evaluate(row)["observed_M_consequence"]


def test_offline_duplicate_execution_requires_distinct_confirmed_http_writes():
    evaluate, _ = auditor_module().load_evaluator()
    row = action_protocol_row("request_redelivery")
    row.update(arm="baseline", action_protocol_events=[], condition="clean", fault_events=[], detection_events=[])
    row["action_ledger_state"]["execution_count"] = 2
    original = row["action_ledger_events"][-1]["details"]["receipt"]
    second = copy.deepcopy(original)
    second["http_receipt_indices"] = [4, 5, 6, 7]
    traces = copy.deepcopy(row["http_receipts"][:4])
    for i, r in enumerate(traces, 4):
        r["receipt_index"] = i
    row["http_receipts"] = row["http_receipts"][:4] + traces
    for i in (2, 6):
        row["http_receipts"][i]["response_payload"]["name"] = task()["product_title"]
    row["action_ledger_events"] += [
        {"event_id": 5, "event": "execute_entered", "action_id": row["action_id"],
         "details": {"delivery_id": "second-delivery", "mode": "observed"}},
        {"event_id": 6, "event": "observed_delivery_confirmed", "action_id": row["action_id"],
         "details": {"delivery_id": "second-delivery", "receipt": second}}]
    assert "duplicate_execution" in evaluate(row)["observed_M_consequence"]
    row["action_ledger_events"][-1]["details"]["receipt"]["http_receipt_indices"] = [0, 1, 2, 3]
    assert "duplicate_execution" not in evaluate(row)["observed_M_consequence"]
