"""Small offline unit fixtures, never real Shopping experiment results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "always_recheck", "guarded_recheck", "dependency",
        "independent", "action_protocol", "combined")
KEY = ("task_id", "topology", "condition", "boundary", "repeat_index")


def module():
    path = ROOT / "scripts/summarize_multimechanism_formal.py"
    assert path.is_file(), "T7 offline postprocessor is not implemented"
    spec = importlib.util.spec_from_file_location("formal_stats", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fixture(products=("a", "a", "b"), repetitions=2):
    tasks = [{"task_id": f"unit-{i}", "product_url": f"https://offline.invalid/{p}",
              "product_title": p, "initial_quantity": 1, "quantity": 2} for i, p in enumerate(products)]
    jobs, cases = [], []
    for task in tasks:
        for repeat in range(1, repetitions + 1):
            for condition, boundary in (("clean", None), ("request_non_delivery", "action_request")):
                key = (task["task_id"], "sequential", condition, boundary, repeat)
                pair_key = json.dumps(key, separators=(",", ":"))
                for arm in ARMS:
                    job = dict(zip(KEY, key), arm=arm, pair_key=pair_key, job_key=pair_key + ":" + arm)
                    jobs.append(job)
                    success = condition == "clean" or arm == "baseline" or task["product_title"] == "a"
                    cases.append({**job, "run_id": f"run-{len(cases)}", "attempt_id": f"attempt-{len(cases)}",
                        "status": "completed", "final_task_success": success, "environment_task_success": success,
                        "decision_correct": True, "evidence_acceptance_errors": 0 if success else 2,
                        "total_tokens": 10 if arm == "baseline" else 14, "known_total_tokens": 10 if arm == "baseline" else 14,
                        "mitigation_gets": 0 if arm == "baseline" else 1,
                        "common_recovery_gets": 0, "base_gets": 3, "evaluation_gets": 2,
                        "mitigation_model_calls": 0, "base_and_common_model_calls": 2,
                        "action_replays": 0, "write_requests": 1, "state_write_requests": 1,
                        "actual_backend_write_count": None})
    config = {"tasks": tasks, "jobs": jobs, "repetitions": repetitions, "planned_runs": len(jobs),
              "shard_runs": len(jobs), "shard_count": 1, "shard_index": 0}
    manifest = {"config": config, "config_digest": digest(config)}
    audit = {"audit_version": "shopping-multimechanism-offline-v1", "offline": True,
        "status": "complete", "findings": [], "cases": cases, "inputs": {},
        "coverage": {"planned_runs": len(jobs), "shard_runs": len(jobs), "completed_runs": len(cases),
                     "completed_unique_jobs": len(cases), "missing_job_keys": [], "unfinished_attempts": 0,
                     "blocked_pair_keys": [], "full_matrix_coverage": True},
        "task_count": len(tasks), "product_count_by_manifest_url": len(set(products)), "repetitions": repetitions}
    refresh(audit)
    return audit, manifest


def refresh(audit):
    cases = audit["cases"]
    audit["by_arm"] = {arm: {"runs": sum(c["arm"] == arm for c in cases)} for arm in ARMS}
    audit["pairs"] = []
    groups = {}
    for case in cases:
        groups.setdefault(case["pair_key"], {})[case["arm"]] = case
    for group in groups.values():
        base = group.get("baseline")
        if base is None or len(group) != 7:
            continue
        clean = [c for c in cases if c["condition"] == "clean" and all(c[k] == base[k] for k in
                 ("task_id", "topology", "repeat_index"))]
        common = len(clean) == 7 and all(c["final_task_success"] is True for c in clean)
        for arm in ARMS[1:]:
            audit["pairs"].append({**{k: base[k] for k in KEY if k != "boundary"}, "pair_key": base["pair_key"],
                "arm": arm, "baseline_run_id": base["run_id"], "strategy_run_id": group[arm]["run_id"],
                "common_clean_success": common})


def inputs(tmp_path, audit=None, manifest=None):
    if audit is None:
        audit, manifest = fixture()
    source = tmp_path / "input"
    source.mkdir(exist_ok=True)
    mp = source / "matrix_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    audit["inputs"]["matrix_manifest.json"] = {"present": True,
        "sha256": hashlib.sha256(mp.read_bytes()).hexdigest(), "bytes": mp.stat().st_size}
    ap = source / "summary.json"
    ap.write_text(json.dumps(audit), encoding="utf-8")
    return ap, mp


def stat(report, metric="final_task_success", subset="all_samples", scope="fault", arm="combined"):
    return next(s for s in report["summaries"] if s["metric"] == metric and s["subset"] == subset
                and s["scope"] == scope and s["arm"] == arm)


def test_product_equal_mean_not_row_weighted_and_deterministic(tmp_path):
    ap, mp = inputs(tmp_path)
    m = module()
    report = m.summarize(ap, mp, draws=1000, seed=20260911)
    s = stat(report)
    assert s["paired_mean"] == -0.5  # Four product-a rows, two product-b rows.
    assert s["ci95"] == [-1.0, 0.0]
    assert (s["improvements"], s["regressions"], s["unchanged"]) == (0, 2, 4)
    assert s["product_clusters"] == 2 and s["paired_rows"] == 6
    assert m.summarize(ap, mp, draws=1000, seed=20260911) == report
    assert report["formal_eligible"] is False and report["status"] == "complete"
    assert report["design"]["actual_tasks"] == 3 and report["design"]["actual_products"] == 2
    assert report["design"]["planned_runs"] == report["design"]["actual_runs"] == 84
    assert stat(report, "total_tokens")["paired_mean"] is None
    assert stat(report, "total_tokens")["interval_reason"] == "verified_all_attempt_cost_required"
    assert stat(report, "evidence_acceptance_errors")["paired_mean"] == 1
    assert stat(report, "environment_task_success")["paired_mean"] == -0.5
    assert stat(report, scope="clean")["paired_mean"] == 0
    assert any(s["scope"] == "topology:sequential" for s in report["summaries"])
    assert any(s["scope"] == "condition:request_non_delivery" for s in report["summaries"])


def test_clean_failure_excludes_all_conditions_all_arms_for_that_context(tmp_path):
    audit, manifest = fixture()
    bad = next(c for c in audit["cases"] if c["task_id"] == "unit-0" and c["condition"] == "clean"
               and c["repeat_index"] == 1 and c["arm"] == "independent")
    bad["final_task_success"] = False
    refresh(audit)
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=50)
    assert stat(report)["paired_rows"] == 6
    assert stat(report, subset="common_clean_success")["paired_rows"] == 5
    exclusions = [e for e in report["exclusions"] if e["reason"] == "clean_not_successful"]
    assert len(exclusions) == 12
    assert all(e["subset"] == "common_clean_success" for e in exclusions)
    assert report["formal_eligible"] is False and report["status"] == "complete"


@pytest.mark.parametrize("field", ["total_tokens", "mitigation_gets"])
def test_unknown_cost_is_not_zero_or_silently_complete_case(tmp_path, field):
    audit, manifest = fixture()
    next(c for c in audit["cases"] if c["arm"] == "combined" and c["condition"] != "clean")[field] = None
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=50)
    s = stat(report, field)
    assert s["paired_mean"] is None and s["ci95"] is None
    assert s["unknown_pairs"] == 1 and s["known_pairs"] == 5
    assert any(e["reason"] == "metric_unknown" and e["metric"] == field for e in report["exclusions"])
    assert "actual_backend_write_count" not in {s["metric"] for s in report["summaries"]}


@pytest.mark.parametrize("damage", ["duplicate_case", "duplicate_run", "duplicate_job", "duplicate_pair",
                                   "boundary", "pair_link", "by_arm"])
def test_conflicting_or_duplicate_inputs_are_rejected_not_deduplicated(tmp_path, damage):
    audit, manifest = fixture()
    if damage == "duplicate_case":
        audit["cases"].append(copy.deepcopy(audit["cases"][0]))
    elif damage == "duplicate_run":
        audit["cases"][1]["run_id"] = audit["cases"][0]["run_id"]
    elif damage == "duplicate_job":
        manifest["config"]["jobs"].append(copy.deepcopy(manifest["config"]["jobs"][0]))
        manifest["config_digest"] = digest(manifest["config"])
    elif damage == "duplicate_pair":
        audit["pairs"].append(copy.deepcopy(audit["pairs"][0]))
    elif damage == "boundary":
        audit["cases"][0]["boundary"] = "wrong-boundary"
    elif damage == "pair_link":
        audit["pairs"][0]["strategy_run_id"] = "wrong-run"
    else:
        audit["by_arm"]["combined"]["runs"] += 1
    with pytest.raises(ValueError):
        module().summarize(*inputs(tmp_path, audit, manifest), draws=10)


@pytest.mark.parametrize("damage", ["missing", "pending"])
def test_missing_or_pending_pair_explicitly_incomplete(tmp_path, damage):
    audit, manifest = fixture()
    c = next(c for c in audit["cases"] if c["arm"] == "combined" and c["condition"] != "clean")
    if damage == "missing":
        audit["cases"].remove(c)
    else:
        c["status"] = "pending"
    refresh(audit)
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=20)
    assert report["status"] == "incomplete" and report["formal_eligible"] is False
    assert report["summaries"] == []
    assert sum(p["arm"] == "combined" and p["condition"] != "clean" for p in report["paired_rows"]) == 5
    assert any(e["reason"] == ("missing_case" if damage == "missing" else "nonterminal_case")
               for e in report["exclusions"])


@pytest.mark.parametrize("damage", ["coverage", "count", "design"])
def test_claimed_complete_audit_with_own_blockers_emits_diagnostics_only(tmp_path, damage):
    audit, manifest = fixture()
    if damage == "coverage":
        audit["coverage"]["full_matrix_coverage"] = False
    elif damage == "count":
        audit["coverage"]["completed_runs"] -= 1
    else:
        audit["task_count"] += 1
    assert audit["status"] == "complete" and audit["findings"] == []
    m = module()
    report = m.summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert report["blockers"] and report["status"] == "incomplete"
    assert report["formal_eligible"] is False and report["summaries"] == []
    assert len(report["paired_rows"]) == 72 and "exclusions" in report
    assert report["design"]["actual_runs"] == 84
    out = tmp_path / "blocked-report"
    m.write_reports(out, report)
    assert len((out / "effects.csv").read_text().splitlines()) == 1
    assert json.loads((out / "summary.json").read_text())["summaries"] == []


@pytest.mark.parametrize("damage", ["status", "error", "unknown"])
def test_unapproved_audit_cannot_produce_effect_estimates(tmp_path, damage):
    audit, manifest = fixture()
    if damage == "status":
        audit["status"] = "incomplete"
    else:
        audit["findings"] = [{"severity": damage, "code": "offline-unit-finding"}]
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert report["status"] == "ineligible"
    assert report["summaries"] == [] and report["formal_eligible"] is False


def test_hash_exact_bytes_and_digest_required(tmp_path):
    ap, mp = inputs(tmp_path)
    mp.write_bytes(mp.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        module().summarize(ap, mp, draws=10)


def test_single_cluster_has_no_interval_and_unknown_success_is_explicit(tmp_path):
    audit, manifest = fixture(products=("a",), repetitions=1)
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert stat(report)["ci95"] is None
    assert stat(report)["interval_reason"] == "fewer_than_two_products"
    audit["cases"][-1]["final_task_success"] = None
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert stat(report)["paired_mean"] is None


@pytest.mark.parametrize("draws", [0, 100001, True, 1.5])
def test_draw_bound(tmp_path, draws):
    with pytest.raises(ValueError, match="draws"):
        module().summarize(*inputs(tmp_path), draws=draws)


def test_exclusive_outputs_hashes_chinese_markdown_and_offline_cli(tmp_path, monkeypatch):
    ap, mp = inputs(tmp_path)
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: pytest.fail("No API/network allowed"))
    m = module()
    before = (ap.read_bytes(), mp.read_bytes())
    out = tmp_path / "new-report"
    assert m.main([str(ap), "--manifest", str(mp), "--output-dir", str(out), "--draws", "20"]) == 0
    report = json.loads((out / "summary.json").read_text())
    assert report["provenance"]["audit_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert report["provenance"]["code_sha256"] == hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
    assert report["bootstrap"]["seed"] == 20260911 and report["bootstrap"]["draws"] == 20
    assert (out / "effects.csv").is_file() and (out / "exclusions.csv").is_file()
    md = (out / "summary.md").read_text()
    assert "商品" in md and "探索" in md and "独立" in md
    assert "https://offline.invalid/" not in (out / "summary.json").read_text()
    assert m.main([str(ap), "--manifest", str(mp), "--output-dir", str(out)]) == 2
    assert m.main([str(ap), "--manifest", str(mp), "--output-dir", str(ap.parent / "nested")]) == 2
    assert before == (ap.read_bytes(), mp.read_bytes())


def test_order_invariance_and_same_url_merges_variants(tmp_path):
    audit, manifest = fixture()
    paths = inputs(tmp_path, audit, manifest)
    m = module()
    original = m.summarize(*paths, draws=100)
    audit["cases"].reverse()
    audit["pairs"].reverse()
    reordered = m.summarize(*inputs(tmp_path, audit, manifest), draws=100)
    assert original["summaries"] == reordered["summaries"]
    assert len(original["design"]["product_task_counts"]) == 2
    assert sorted(original["design"]["product_task_counts"].values()) == [1, 2]


def test_full_frozen_6300_without_result_binding_is_descriptive_not_formal(tmp_path):
    from mas_faults import multimechanism_matrix as matrix
    audit, manifest = fixture(products=tuple(str(i // 2) for i in range(10)), repetitions=3)
    config = manifest["config"]
    jobs = matrix.build_jobs(config["tasks"], repetitions=3)
    template = audit["cases"][0]
    audit["cases"] = [{**template, **job, "run_id": f"unit-run-{i}", "attempt_id": f"unit-attempt-{i}"}
                      for i, job in enumerate(jobs)]
    config.update(jobs=jobs, planned_runs=6300, shard_runs=6300)
    manifest["config_digest"] = digest(config)
    audit["coverage"].update(planned_runs=6300, shard_runs=6300, completed_runs=6300, completed_unique_jobs=6300)
    refresh(audit)
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert report["formal_eligible"] is False
    assert report["eligibility"] is None
    assert report["design"]["actual_runs"] == 6300 and report["design"]["actual_products"] == 5
    assert stat(report)["paired_rows"] == 810 and stat(report)["product_clusters"] == 5
    assert stat(report)["paired_mean"] == 0 and stat(report)["ci95"] == [0, 0]
    assert any("五" in limitation and "不稳定" in limitation for limitation in report["limitations"])


def test_absent_success_and_tool_fields_are_unknown_not_schema_crash(tmp_path):
    audit, manifest = fixture()
    del audit["cases"][0]["final_task_success"]
    for case in audit["cases"]:
        case.pop("mitigation_gets")
    # The auditor's saved clean-subset flag treats a missing derived outcome as not successful.
    for pair in audit["pairs"]:
        if pair["task_id"] == "unit-0" and pair["repeat_index"] == 1:
            pair["common_clean_success"] = False
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert stat(report, "mitigation_gets")["paired_mean"] is None
    assert stat(report, subset="common_clean_success")["paired_rows"] == 5


@pytest.mark.parametrize("damage", ["split_pair_key", "saved_change"])
def test_no_silent_pair_schema_disagreement(tmp_path, damage):
    audit, manifest = fixture()
    if damage == "split_pair_key":
        manifest["config"]["jobs"][0]["pair_key"] = "split-pair"
        audit["cases"][0]["pair_key"] = "split-pair"
        manifest["config_digest"] = digest(manifest["config"])
    else:
        audit["pairs"][0]["final_task_success_change"] = "regression"
    with pytest.raises(ValueError, match="pair"):
        module().summarize(*inputs(tmp_path, audit, manifest), draws=10)


def test_unresolved_terminal_failure_remains_in_estimand(tmp_path):
    audit, manifest = fixture()
    case = next(c for c in audit["cases"] if c["condition"] != "clean" and c["arm"] == "combined"
                and c["task_id"] == "unit-0")
    case.update(status="unresolved", action_outcome_unknown=True, final_task_success=False)
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=10)
    assert report["status"] == "complete"
    assert stat(report)["paired_rows"] == 6 and stat(report)["regressions"] == 3


def test_ineligible_cli_emits_diagnostic_report_and_nonzero(tmp_path):
    audit, manifest = fixture()
    audit["status"] = "incomplete"
    ap, mp = inputs(tmp_path, audit, manifest)
    out = tmp_path / "diagnostics"
    assert module().main([str(ap), "--manifest", str(mp), "--output-dir", str(out), "--draws", "1"]) == 1
    assert json.loads((out / "summary.json").read_text())["summaries"] == []


def test_symlink_output_cannot_reenter_input(tmp_path):
    ap, mp = inputs(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(ap.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        module().check_output(alias / "nested", [ap, mp])


def test_bootstrap_linear_percentiles_known_values():
    m = module()
    assert m.quantile([0, 10], 0.025) == 0.25
    assert m.quantile([0, 10], 0.975) == 9.75
    assert m.cluster_interval((-1.0, 1.0), 10000, 20260911) == (-1.0, 1.0)


def bound_inputs(tmp_path, *, unknown=False, formal=True, damage=None):
    from test_multimechanism_eligibility import fixture as eligible_fixture, save
    values = eligible_fixture(formal=formal, unknown=unknown)
    audit, manifest, logs = values
    refresh(audit)
    if damage == "finding":
        audit["findings"].append({"code": "future_unknown_code", "severity": "info"})
    elif damage == "request_count":
        logs["run_errors.jsonl"][0]["model_calls"] += 1
    ap, root = save(tmp_path, values)
    return ap, root / "matrix_manifest.json", root


def test_t7_derived_null_cost_preserves_outcome_denominator_and_blocks_savings(tmp_path):
    m = module()
    before_dir, after_dir = tmp_path / "known", tmp_path / "unknown"
    before_dir.mkdir()
    after_dir.mkdir()
    ap, mp, root = bound_inputs(before_dir)
    before = m.summarize(ap, mp, result_dir=root, draws=10)
    ap, mp, root = bound_inputs(after_dir, unknown=True)
    original = {p: p.read_bytes() for p in [ap, *root.rglob("*")] if p.is_file()}
    after = m.summarize(ap, mp, result_dir=root, draws=10)
    assert before["formal_eligible"] and after["formal_eligible"]
    assert before["outcome_eligible"] and after["outcome_eligible"]
    assert before["exact_cost_comparison_eligible"] and not after["exact_cost_comparison_eligible"]
    assert before["outcomes"] == after["outcomes"]
    assert after["outcomes"]["denominator"] == 6300
    assert after["outcomes"]["error_attempts"] == after["outcomes"]["retried_jobs"] == 1
    assert after["outcomes"]["first_attempt_completed"] == 6299
    assert after["source_audit_status"] == "incomplete"
    assert after["eligibility"]["provenance"]["audit_sha256"] == hashlib.sha256(original[ap]).hexdigest()
    assert before["outcome_summaries"] == after["outcome_summaries"]
    assert stat(after)["paired_rows"] == 810
    assert after["cost"]["exact_total_tokens"] is None
    assert after["cost"]["known_lower_bound_tokens"] == 630020
    job = next(j for j in after["cost_jobs"] if j["attempt_count"] == 2)
    assert job["known_lower_bound_tokens"] == 120 and job["exact_total_tokens"] is None
    assert after["cost_by_arm"]["baseline"]["unknown_attempts"] == 1
    assert after["cost_by_arm"]["baseline"]["outcomes"]["error_attempts"] == 1
    for s in after["cost_summaries"]:
        assert s["paired_mean"] is None and s["ci95"] is None
        assert s["improvements"] is None and s["regressions"] is None
        assert s["comparison_eligible"] is False
    known_cost = stat(before, "total_tokens", scope="clean")
    assert known_cost["paired_mean"] == pytest.approx(-30 / 90)
    assert before["cost"]["exact_total_tokens"] == 630030
    assert before["cost_jobs"][0]["attempts"]
    out = tmp_path / "t7"
    m.write_reports(out, after)
    assert (out / "cost_jobs.csv").is_file() and (out / "cost_by_arm.csv").is_file()
    md = (out / "summary.md").read_text()
    assert "成本节省" in md and "下限" in md and "首尝试" in md
    assert original == {p: p.read_bytes() for p in original}


@pytest.mark.parametrize("damage", ["finding", "request_count", "pilot"])
def test_t7_derived_other_defects_or_small_design_cannot_enable_inference(tmp_path, damage):
    ap, mp, root = bound_inputs(tmp_path, unknown=True, formal=damage != "pilot", damage=damage)
    report = module().summarize(ap, mp, result_dir=root, draws=1)
    assert not report["formal_eligible"] and not report["outcome_eligible"]
    assert not report["exact_cost_comparison_eligible"]
    assert report["summaries"] == report["outcome_summaries"] == report["cost_summaries"] == []
    assert report["blockers"] and report["source_audit_status"] == "incomplete"


@pytest.mark.parametrize("damage", ["raw_bytes", "audit_bytes", "wrong_result_dir", "manifest_copy"])
def test_t7_derived_requires_explicit_matching_real_input_bytes(tmp_path, damage):
    ap, mp, root = bound_inputs(tmp_path, formal=False)
    m = module()
    if damage == "raw_bytes":
        path = root / "run_errors.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "audit_bytes":
        audit = json.loads(ap.read_bytes())
        audit["inputs"]["run_errors.jsonl"]["sha256"] = "0" * 64
        ap.write_text(json.dumps(audit))
    elif damage == "wrong_result_dir":
        root = tmp_path / "wrong"
        root.mkdir()
    else:
        other = tmp_path / "manifest.json"
        other.write_bytes(mp.read_bytes())
        mp = other
    with pytest.raises((ValueError, OSError)):
        m.summarize(ap, mp, result_dir=root, draws=1)


def test_t7_legacy_never_silently_enables_failed_usage_exception(tmp_path):
    ap, mp, _ = bound_inputs(tmp_path, unknown=True, formal=False)
    report = module().summarize(ap, mp, draws=1)
    assert report["summaries"] == [] and not report["formal_eligible"]
    assert report["eligibility"] is None and not report["exact_cost_comparison_eligible"]


def test_t7_cli_result_dir_and_output_must_remain_outside_raw_inputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    ap, mp, root = bound_inputs(source, formal=False)
    before = {p: p.read_bytes() for p in [ap, *root.rglob("*")] if p.is_file()}
    m = module()
    out = tmp_path / "diagnostics"
    argv = [str(ap), "--manifest", str(mp), "--result-dir", str(root), "--draws", "1"]
    assert m.main([*argv, "--output-dir", str(out)]) == 1
    report = json.loads((out / "summary.json").read_text())
    assert report["eligibility"]["design"]["planned_jobs"] == 14
    assert report["provenance"]["result_dir"] == str(root.resolve())
    assert m.main([*argv, "--output-dir", str(root / "new")]) == 2
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("during", ["summarize", "write"])
def test_t7_rechecks_input_bytes_before_emitting_derived_artifact(tmp_path, monkeypatch, during):
    source = tmp_path / "input"
    source.mkdir()
    ap, mp, root = bound_inputs(source, formal=False)
    m = module()
    raw = root / "run_errors.jsonl"
    if during == "summarize":
        original_validate = m.validate_inputs

        def mutate(audit, manifest):
            result = original_validate(audit, manifest)
            raw.write_bytes(raw.read_bytes() + b"\n")
            return result

        monkeypatch.setattr(m, "validate_inputs", mutate)
        with pytest.raises(ValueError, match="changed|hash"):
            m.summarize(ap, mp, result_dir=root, draws=1)
    else:
        report = m.summarize(ap, mp, result_dir=root, draws=1)
        raw.write_bytes(raw.read_bytes() + b"\n")
        output = tmp_path / "report"
        with pytest.raises(ValueError, match="changed|hash"):
            m.write_reports(output, report)
        assert not output.exists()


def test_t7_default_unknown_finding_is_not_an_implicit_allowlist(tmp_path):
    audit, manifest = fixture()
    audit["findings"].append({"code": "future_unknown_code", "severity": "info"})
    report = module().summarize(*inputs(tmp_path, audit, manifest), draws=1)
    assert not report["formal_eligible"] and report["summaries"] == []
