#!/usr/bin/env python3
"""T7 offline product-cluster summaries with optional derived eligibility.

Only local JSON files are read. No runtime imports, network, or model calls.
Exit 0: complete descriptive report; 1: ineligible/incomplete; 2: invalid input
or unsafe output. --result-dir verifies the raw audit-bound snapshot through
the eligibility sidecar. Without it, only strict legacy descriptive outcomes
are available; formal eligibility and all-attempt cost comparisons fail closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean


ARMS = ("baseline", "always_recheck", "guarded_recheck", "dependency",
        "independent", "action_protocol", "combined")
KEY = ("task_id", "topology", "condition", "boundary", "repeat_index")
CELLS = {"clean": None, "request_non_delivery": "action_request", "acknowledgement_loss": "action_ack",
         "duplicate_action_delivery": "action_request", "valid_partial": "evidence_handoff",
         "same_session_reordering": "observation_handoff", "cross_task_replay": "evidence_handoff",
         "stale_judgment_replay": "judgment_handoff", "conflicting_observation": "observation_handoff",
         "contract_consistent_identity_corruption": "evidence_handoff"}
SUCCESS = ("final_task_success", "environment_task_success", "decision_correct")
OUTCOME_METRICS = (*SUCCESS, "evidence_acceptance_errors")
METRICS = (*SUCCESS, "evidence_acceptance_errors", "total_tokens", "mitigation_gets",
           "common_recovery_gets", "base_gets", "evaluation_gets", "mitigation_model_calls",
           "base_and_common_model_calls", "action_replays", "write_requests", "state_write_requests")
TERMINAL = {"completed", "unresolved"}
LIMITATIONS = [
    "仅作描述性、探索性商品聚类区间；不计算 p 值或统计显著性，不作因果或模型泛化主张。",
    "正式设计仅五个商品簇；任务变体、拓扑、条件及重复行相互依赖，行数不是独立样本量。",
    "每行先计算策略减基线，再在商品内平均，最后商品等权平均；重采样完整商品簇，保留配对和重复依赖。",
    "95% 区间为商品均值有放回重采样的百分位区间；少于两个商品不报区间，五簇区间仍很不稳定。",
    "共同 clean 成功子集要求同任务、拓扑、重复的七组全部成功；全样本始终保留，排除原因单列。",
    "任一配对指标未知时，该汇总的主均值与区间留空，不以零或完整案例估计替代；原始变化计数仅描述已知配对。",
    "工具指标仅为审计导出的请求或事件计数，不是货币成本或后端实际写入次数；评价 GET 单列。",
    "正式资格须显式提供 result-dir 并通过审计真实字节绑定的派生资格核验；旧调用仅保留严格审计门槛下的描述性结果。",
    "结果估计对象是最多两次尝试的冻结流程，不是首尝试成功率；错误、重试及首尝试情况另列。",
    "成本按 job_key 汇总全部尝试；未知成本仅报告已知下限，不能由两个下限的差推断成本节省。",
    "操作计数沿用审计中完成尝试的诊断字段，不包含失败尝试，不能解释为流程成本节省。",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("input must be a local file")
    data = path.read_bytes()
    value = json.loads(data, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return path, data, value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key_of(row):
    require(isinstance(row, dict), "row must be an object")
    require(all(isinstance(row.get(k), str) and row[k] for k in KEY[:3]), "invalid pairing dimensions")
    require(row.get("boundary") is None or isinstance(row["boundary"], str), "invalid boundary")
    require(type(row.get("repeat_index")) is int and row["repeat_index"] > 0, "invalid repeat index")
    return tuple(row[k] for k in KEY)


def validate_inputs(audit, manifest):
    require(audit.get("audit_version") == "shopping-multimechanism-offline-v1" and audit.get("offline") is True,
            "unsupported audit schema")
    config = manifest.get("config")
    require(isinstance(config, dict) and manifest.get("config_digest") == sha256(canonical(config).encode()),
            "manifest config digest mismatch")
    tasks = config.get("tasks")
    require(isinstance(tasks, list) and tasks, "manifest tasks required")
    products = {}
    for task in tasks:
        require(isinstance(task, dict) and isinstance(task.get("task_id"), str) and task["task_id"]
                and isinstance(task.get("product_url"), str) and task["product_url"], "task product mapping required")
        require(task["task_id"] not in products, "duplicate manifest task")
        # The exact frozen URL is an opaque identifier, never parsed or fetched.
        products[task["task_id"]] = sha256(task["product_url"].encode())
    require(type(config.get("repetitions")) is int and config["repetitions"] > 0, "invalid repetitions")
    jobs = config.get("jobs")
    require(isinstance(jobs, list) and jobs, "frozen jobs required")
    planned, job_ids, pair_ids, group_ids = {}, set(), {}, {}
    for job in jobs:
        key = key_of(job)
        require(key[0] in products and key[4] <= config["repetitions"] and job.get("arm") in ARMS, "invalid frozen job")
        identity = (key, job["arm"])
        require(identity not in planned and isinstance(job.get("job_key"), str) and job["job_key"] not in job_ids,
                "duplicate frozen job")
        require(isinstance(job.get("pair_key"), str) and job["pair_key"], "pair_key required")
        require(pair_ids.get(job["pair_key"], key) == key, "pair_key dimension collision")
        require(group_ids.get(key, job["pair_key"]) == job["pair_key"], "split frozen pair_key")
        pair_ids[job["pair_key"]] = key
        group_ids[key] = job["pair_key"]
        planned[identity] = job
        job_ids.add(job["job_key"])
    groups = {key for key, _ in planned}
    require(all({arm for k, arm in planned if k == key} == set(ARMS) for key in groups), "frozen pair missing arms")
    require(config.get("shard_runs") == len(jobs) and type(config.get("planned_runs")) is int
            and config["planned_runs"] >= len(jobs), "manifest planned counts mismatch")
    cases = audit.get("cases")
    require(isinstance(cases, list), "audit cases required")
    indexed, seen = {}, {k: set() for k in ("run_id", "job_key", "attempt_id")}
    for case in cases:
        key = key_of(case)
        identity = (key, case.get("arm"))
        require(identity in planned and identity not in indexed, "unknown or duplicate case pairing")
        require(all(case.get(k) == planned[identity][k] for k in ("job_key", "pair_key")), "case frozen job mismatch")
        for field, values in seen.items():
            value = case.get(field)
            require(isinstance(value, str) and value and value not in values, "missing or duplicate " + field)
            values.add(value)
        for metric in METRICS:
            value = case.get(metric)
            require(value is None or (type(value) is bool if metric in SUCCESS else type(value) is int and value >= 0),
                    "invalid metric type: " + metric)
        indexed[identity] = case
    by_arm = audit.get("by_arm")
    require(isinstance(by_arm, dict) and set(by_arm) == set(ARMS), "audit by_arm schema mismatch")
    for arm in ARMS:
        rows = [c for c in cases if c["arm"] == arm]
        require(isinstance(by_arm[arm], dict) and by_arm[arm].get("runs") == len(rows), "audit by_arm runs mismatch")
        for metric in (*SUCCESS, "evidence_acceptance_errors", "known_total_tokens"):
            if metric not in by_arm[arm]:
                continue
            values = [c.get(metric) for c in rows]
            expected = sum(v is True for v in values) if metric in SUCCESS else (
                sum(values) if all(type(v) is int and v >= 0 for v in values) else None)
            require(expected is not None and by_arm[arm][metric] == expected, "audit by_arm metric mismatch")
    exported = audit.get("pairs")
    require(isinstance(exported, list), "audit pairs required")
    seen_pairs = set()
    for pair in exported:
        require(isinstance(pair, dict) and pair.get("pair_key") in pair_ids, "unknown exported pair")
        key, arm = pair_ids[pair["pair_key"]], pair.get("arm")
        require(arm in ARMS[1:] and (key, arm) not in seen_pairs, "duplicate or invalid exported pair")
        seen_pairs.add((key, arm))
        base, other = indexed.get((key, "baseline")), indexed.get((key, arm))
        require(base is not None and other is not None and all(pair.get(k) == base[k] for k in KEY if k != "boundary")
                and pair.get("baseline_run_id") == base["run_id"] and pair.get("strategy_run_id") == other["run_id"],
                "exported pair linkage mismatch")
        clean = (key[0], key[1], "clean", None, key[4])
        common = all(indexed.get((clean, a), {}).get("final_task_success") is True for a in ARMS)
        require(pair.get("common_clean_success") is common, "exported clean subset mismatch")
        if "token_delta" in pair:
            a, b = base.get("total_tokens"), other.get("total_tokens")
            require(pair["token_delta"] == (None if a is None or b is None else b - a), "exported token delta mismatch")
        for metric in (*SUCCESS, "decision_correct", "evidence_acceptance_errors"):
            if metric + "_change" not in pair:
                continue
            a, b = base.get(metric), other.get(metric)
            expected = "unknown" if a is None or b is None else "unchanged" if a == b else (
                "improvement" if (b < a if metric == "evidence_acceptance_errors" else b > a) else "regression")
            require(pair[metric + "_change"] == expected, "exported pair change mismatch")
    expected_pairs = {(key, arm) for key in groups if all((key, a) in indexed for a in ARMS) for arm in ARMS[1:]}
    require(seen_pairs == expected_pairs, "exported pair coverage mismatch")
    return config, products, planned, indexed


def quantile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


@lru_cache(maxsize=512)
def cluster_interval(means, draws, seed):
    rng = random.Random(seed)
    n = len(means)
    samples = sorted(fmean(means[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return (quantile(samples, 0.025), quantile(samples, 0.975))


def effect(rows, metric, draws, seed):
    clusters, changes = defaultdict(list), Counter()
    unknown = 0
    for pair in rows:
        a, b = pair["baseline"].get(metric), pair["strategy"].get(metric)
        if a is None or b is None:
            unknown += 1
            continue
        delta = int(b) - int(a)
        clusters[pair["product_id"]].append(delta)
        changes["unchanged" if delta == 0 else "improvements" if (delta > 0) == (metric in SUCCESS) else "regressions"] += 1
    means = tuple(fmean(clusters[p]) for p in sorted(clusters))
    reason = "unknown_pairs" if unknown else "no_pairs" if not means else "fewer_than_two_products" if len(means) < 2 else None
    return {"paired_rows": len(rows), "known_pairs": len(rows) - unknown, "unknown_pairs": unknown,
            "product_clusters": len({r["product_id"] for r in rows}), "known_product_clusters": len(means),
            "paired_mean": fmean(means) if means and not unknown else None,
            "ci95": list(cluster_interval(means, draws, seed)) if reason is None else None,
            "interval_reason": reason, **{k: changes[k] for k in ("improvements", "regressions", "unchanged")}}


def derived_eligibility(audit_path, manifest_path, result_dir):
    """Recompute qualification from bytes, never trust caller-supplied flags."""
    root = Path(result_dir).resolve(strict=True)
    require(root.is_dir() and manifest_path == (root / "matrix_manifest.json").resolve(strict=True),
            "manifest must be the explicit result-dir manifest")
    path = Path(__file__).with_name("assess_multimechanism_eligibility.py").resolve(strict=True)
    helper_bytes = path.read_bytes()
    spec = importlib.util.spec_from_file_location("t7_eligibility", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    _, _, _, _, snapshot = helper.read_inputs(audit_path, root)
    value = helper.assess(audit_path, root)
    require(value.get("version") == "shopping-multimechanism-eligibility-v1"
            and value.get("policy_version") == "formal-statistical-eligibility-20260912-v1"
            and value.get("offline") is True, "unsupported derived eligibility schema")
    require(value["provenance"]["audit_sha256"] == sha256(snapshot[audit_path])
            and value["provenance"]["helper_sha256"] == sha256(helper_bytes),
            "derived eligibility byte binding mismatch")
    snapshot[path] = helper_bytes
    return value, snapshot


def summarize(audit_path, manifest_path, *, draws=10000, seed=20260911, result_dir=None):
    require(type(draws) is int and 1 <= draws <= 100000, "draws must be an integer in 1..100000")
    require(type(seed) is int and 0 <= seed <= 2**64 - 1, "seed must be an integer in 0..2**64-1")
    ap, audit_bytes, audit = read_json(audit_path)
    mp, manifest_bytes, manifest = read_json(manifest_path)
    code_path = Path(__file__).resolve()
    code_bytes = code_path.read_bytes()
    eligibility = None
    snapshot = {ap: audit_bytes, mp: manifest_bytes, code_path: code_bytes}
    if result_dir is not None:
        eligibility, bound_snapshot = derived_eligibility(ap, mp, result_dir)
        require(bound_snapshot[ap] == audit_bytes and bound_snapshot[mp] == manifest_bytes,
                "input changed during eligibility assessment")
        snapshot.update(bound_snapshot)
    recorded = audit.get("inputs", {}).get("matrix_manifest.json", {})
    require(recorded.get("present") is True and recorded.get("sha256") == sha256(manifest_bytes), "manifest input hash mismatch")
    config, products, planned, indexed = validate_inputs(audit, manifest)
    groups = sorted({k for k, _ in planned}, key=canonical)
    exclusions, blockers, pairs = [], [], []
    findings = audit.get("findings")
    require(isinstance(findings, list) and all(isinstance(f, dict) and f.get("severity") in {"error", "unknown", "warning", "info"}
                                             for f in findings), "invalid audit findings")
    gated = audit.get("status") != "complete" or bool(findings)
    if eligibility is not None:
        gated = eligibility["outcome_eligible"] is not True
        if gated:
            blockers.extend("derived:" + reason for reason in eligibility["blockers"])
    elif gated:
        blockers.append("audit_not_complete_or_has_unapproved_findings")
    terminal = {i: c for i, c in indexed.items() if c.get("status") in TERMINAL}
    if len(terminal) != len(planned):
        blockers.append("missing_or_nonterminal_cases")
    coverage = audit.get("coverage", {})
    if (coverage.get("planned_runs") != config["planned_runs"] or coverage.get("completed_runs") != len(terminal)
            or coverage.get("completed_unique_jobs") != len(terminal) or coverage.get("full_matrix_coverage") is not True
            or coverage.get("missing_job_keys") or coverage.get("unfinished_attempts") or coverage.get("blocked_pair_keys")
            or config.get("shard_count") != 1 or config["planned_runs"] != len(planned)):
        blockers.append("coverage_not_complete_or_counts_mismatch")
    if audit.get("task_count") != len(products) or audit.get("repetitions") != config["repetitions"]:
        blockers.append("design_metadata_mismatch")
    cost_jobs = {j["job_key"]: j for j in eligibility["jobs"]} if eligibility is not None else {}
    exact_cost_eligible = bool(eligibility is not None
        and eligibility["exact_cost_comparison_eligible"] is True and not blockers)
    for key in groups:
        clean_key = (key[0], key[1], "clean", None, key[4])
        clean = [terminal.get((clean_key, arm)) for arm in ARMS]
        clean_reason = "clean_missing_or_nonterminal" if any(c is None for c in clean) else (
            "clean_not_successful" if not all(c.get("final_task_success") is True for c in clean) else None)
        for arm in ARMS[1:]:
            identity = {**dict(zip(KEY, key)), "arm": arm, "product_id": products[key[0]]}
            absent = [a for a in ("baseline", arm) if (key, a) not in terminal]
            if absent:
                for a in absent:
                    exclusions.append({**identity, "subset": "all_samples", "excluded_arm": a,
                        "reason": "missing_case" if (key, a) not in indexed else "nonterminal_case"})
                continue
            base, other = terminal[key, "baseline"], terminal[key, arm]
            if eligibility is not None:
                # Saved audit pairs describe completed attempts. Validate them
                # above, then copy rather than rewrite their original values.
                base = {**base, "total_tokens": cost_jobs[base["job_key"]]["exact_total_tokens"]}
                other = {**other, "total_tokens": cost_jobs[other["job_key"]]["exact_total_tokens"]}
            pair = {**identity, "baseline_run_id": base["run_id"], "strategy_run_id": other["run_id"],
                    "common_clean_success": clean_reason is None, "baseline": base, "strategy": other}
            pairs.append(pair)
            if clean_reason:
                exclusions.append({**identity, "subset": "common_clean_success", "reason": clean_reason})
            for metric in METRICS:
                if base.get(metric) is None or other.get(metric) is None:
                    exclusions.append({**identity, "subset": "all_samples", "reason": "metric_unknown", "metric": metric,
                                       "also_common_clean": clean_reason is None})
    actual_tasks = {key[0] for key, _ in terminal}
    product_counts = dict(sorted(Counter(products.values()).items()))
    formal_groups = {(task, top, condition, boundary, repeat) for task, top, (condition, boundary), repeat in
                     itertools.product(products, ("sequential", "flat", "hierarchical"), CELLS.items(), range(1, 4))}
    formal_design = (len(products) == 10 and len(product_counts) == 5 and set(product_counts.values()) == {2}
                     and config["repetitions"] == 3 and config["planned_runs"] == 6300 and set(groups) == formal_groups)
    report = {"version": "shopping-multimechanism-formal-stats-v1", "offline": True,
        "status": "ineligible" if gated else "incomplete" if blockers else "complete",
        "formal_eligible": formal_design and not blockers and eligibility is not None,
        "outcome_eligible": formal_design and not blockers and eligibility is not None,
        "exact_cost_comparison_eligible": exact_cost_eligible,
        "cost_complete": eligibility["cost_complete"] if eligibility is not None else False,
        "eligibility": eligibility, "source_audit_status": audit["status"],
        "source_findings": findings, "outcomes": eligibility["outcomes"] if eligibility is not None else None,
        "cost": eligibility["cost"] if eligibility is not None else None,
        "cost_jobs": list(cost_jobs.values()),
        "cost_by_arm": eligibility["by_arm"] if eligibility is not None else {},
        "blockers": blockers,
        "design": {"formal_target": {"runs": 6300, "tasks": 10, "products": 5, "repetitions": 3},
            "planned_runs": config["planned_runs"], "manifest_job_rows": len(planned), "actual_runs": len(terminal),
            "input_case_rows": len(indexed), "planned_tasks": len(products), "actual_tasks": len(actual_tasks),
            "planned_products": len(product_counts), "actual_products": len({products[t] for t in actual_tasks}),
            "planned_repetitions": config["repetitions"], "actual_repeat_indices": sorted({k[4] for k, _ in terminal}),
            "actual_repetitions": len({k[4] for k, _ in terminal}), "product_task_counts": product_counts,
            "task_product_ids": products, "planned_topologies": sorted({k[1] for k in groups}),
            "planned_conditions": sorted({k[2] for k in groups}), "matches_formal_design": formal_design},
        "provenance": {"audit_file": str(ap), "manifest_file": str(mp), "audit_sha256": sha256(audit_bytes),
            "manifest_sha256": sha256(manifest_bytes), "config_digest": manifest["config_digest"],
            "result_dir": str(Path(result_dir).resolve()) if result_dir is not None else None,
            "code_file": str(code_path), "code_sha256": sha256(code_bytes),
            "input_snapshot": {str(path): {"present": data is not None, "bytes": len(data) if data is not None else None,
                "sha256": sha256(data) if data is not None else None} for path, data in snapshot.items()}},
        "bootstrap": {"draws": draws, "seed": seed, "confidence": 0.95, "unit": "product",
            "method": "paired product-equal percentile bootstrap; linear quantiles; sorted opaque product hashes",
            "delta": "strategy_minus_baseline", "independent_row_n": False},
        "metric_directions": {m: "higher_is_better" if m in SUCCESS else "lower_is_better" for m in METRICS},
        "exclusions": exclusions, "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
        "paired_rows": [{k: v for k, v in p.items() if k not in {"baseline", "strategy"}} for p in pairs],
        "summaries": [], "outcome_summaries": [], "cost_summaries": [], "operation_summaries": [],
        "limitations": LIMITATIONS}
    if not blockers:
        scopes = {"all": pairs, "clean": [p for p in pairs if p["condition"] == "clean"],
                  "fault": [p for p in pairs if p["condition"] != "clean"]}
        for field in ("topology", "condition"):
            for value in sorted({k[KEY.index(field)] for k in groups}):
                scopes[field + ":" + value] = [p for p in pairs if p[field] == value]
        for top, condition in sorted({(k[1], k[2]) for k in groups}):
            scopes[f"topology_condition:{top}:{condition}"] = [p for p in pairs if p["topology"] == top and p["condition"] == condition]
        for subset in ("all_samples", "common_clean_success"):
            for scope, rows in scopes.items():
                for arm in ARMS[1:]:
                    selected = [p for p in rows if p["arm"] == arm and (subset == "all_samples" or p["common_clean_success"])]
                    for metric in METRICS:
                        value = {"subset": subset, "scope": scope, "arm": arm, "metric": metric,
                                 **effect(selected, metric, draws, seed)}
                        domain = "outcome" if metric in OUTCOME_METRICS else "cost" if metric == "total_tokens" else "operation"
                        value["domain"] = domain
                        value["comparison_eligible"] = domain != "cost" or exact_cost_eligible
                        if domain == "cost" and not exact_cost_eligible:
                            value.update(paired_mean=None, ci95=None, improvements=None, regressions=None,
                                unchanged=None, interval_reason="unknown_all_attempt_cost" if eligibility is not None
                                else "verified_all_attempt_cost_required")
                        if domain == "operation":
                            value["basis"] = "completed_attempt_diagnostic_only_not_workflow_cost"
                        report["summaries"].append(value)
                        report[domain + "_summaries"].append(value)
    for path, data in snapshot.items():
        require(not path.exists() if data is None else path.is_file() and path.read_bytes() == data,
                "input changed during T7 summarization")
    return report


def check_output(output, inputs):
    output = Path(output).resolve()
    require(not output.exists(), "output must be a NEW directory")
    require(all(not output.is_relative_to(Path(p).resolve().parent) for p in inputs),
            "output must be outside both input source directories")
    return output


def write_reports(output, report):
    provenance = report["provenance"]
    output = check_output(output, [provenance["audit_file"], provenance["manifest_file"]])
    for name, expected in provenance["input_snapshot"].items():
        path = Path(name)
        if not expected["present"]:
            require(not path.exists(), "input changed before T7 report write")
        else:
            require(path.is_file(), "input changed before T7 report write")
            data = path.read_bytes()
            require(len(data) == expected["bytes"] and sha256(data) == expected["sha256"],
                    "input hash changed before T7 report write")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    for filename, key in (("effects.csv", "summaries"), ("exclusions.csv", "exclusions"), ("pairs.csv", "paired_rows")):
        rows = report[key]
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ["reason"]
        with (output / filename).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    for filename, rows in (("cost_jobs.csv", report["cost_jobs"]),
                           ("cost_by_arm.csv", [{"arm": arm, **row} for arm, row in report["cost_by_arm"].items()])):
        with (output / filename).open("x", encoding="utf-8", newline="") as handle:
            fields = list(dict.fromkeys(k for row in rows for k in row)) or ["reason"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    d = report["design"]
    lines = ["# 多机制商品聚类配对汇总", "", f"状态：{report['status']}；正式设计合格：{report['formal_eligible']}。",
        f"计划/实际：{d['planned_runs']}/{d['actual_runs']} 行，{d['planned_tasks']}/{d['actual_tasks']} 任务，"
        f"{d['planned_products']}/{d['actual_products']} 商品，{d['planned_repetitions']} 次计划重复；"
        f"实际重复编号：{d['actual_repeat_indices']}。", "",
        f"Bootstrap：{report['bootstrap']['draws']} 次；seed={report['bootstrap']['seed']}。",
        "差值均为策略减基线；结果与成本资格分开。未知成本不计算差值或成本节省。计数不是独立 N。", "",
        "| 子集 | 范围 | 策略 | 指标 | 配对行 | 商品簇 | 均值 | 探索性 95% 区间 | 改善/退化/不变 | 未知 |",
        "|---|---|---|---|---:|---:|---:|---|---|---:|"]
    for s in report["summaries"]:
        if s["scope"] not in {"clean", "fault"} or s["metric"] not in (*SUCCESS, "evidence_acceptance_errors", "total_tokens"):
            continue
        mean = "未知" if s["paired_mean"] is None else f"{s['paired_mean']:.6g}"
        ci = "不报告" if s["ci95"] is None else f"[{s['ci95'][0]:.6g}, {s['ci95'][1]:.6g}]"
        changes = "不推断" if not s["comparison_eligible"] else f"{s['improvements']}/{s['regressions']}/{s['unchanged']}"
        lines.append(f"| {s['subset']} | {s['scope']} | {s['arm']} | {s['metric']} | {s['paired_rows']} | "
                     f"{s['product_clusters']} | {mean} | {ci} | {changes} | {s['unknown_pairs']} |")
    if report["outcomes"] is not None:
        o = report["outcomes"]
        lines += ["", "## 尝试与成本", "",
            f"原始审计状态：{report['source_audit_status']}（不修改）；结果资格：{report['outcome_eligible']}；"
            f"精确成本比较资格：{report['exact_cost_comparison_eligible']}。",
            f"结果分母 {o['denominator']}；错误尝试 {o['error_attempts']}；重试 job {o['retried_jobs']}；"
            f"首尝试完成 {o['first_attempt_completed']}；首尝试成功 {o['first_attempt_successes']}。",
            "逐 job 的全部尝试见 cost_jobs.csv；下限不是精确总量，不对下限作差。",
            "| 策略 | 已知 token 下限 | 精确总量 | 未知尝试 | 尝试数 |",
            "|---|---:|---:|---:|---:|"]
        for arm, row in report["cost_by_arm"].items():
            exact = "未知" if row["exact_total_tokens"] is None else row["exact_total_tokens"]
            lines.append(f"| {arm} | {row['known_lower_bound_tokens']} | {exact} | {row['unknown_attempts']} | {row['attempt_count']} |")
    lines += ["", "## 排除与限制", "", "完整拓扑/条件/工具指标分层见 effects.csv；每个配对的排除原因见 exclusions.csv。",
              "排除计数（跨策略/条件可重复，不是独立样本）：" + canonical(report["exclusion_counts"]),
              "阻断原因：" + canonical(report["blockers"]), "", *["- " + item for item in LIMITATIONS]]
    with (output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_summary", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args(argv)
    try:
        output = check_output(args.output_dir, [args.audit_summary, args.manifest])
        report = summarize(args.audit_summary, args.manifest, draws=args.draws, seed=args.seed, result_dir=args.result_dir)
        write_reports(output, report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"formal summary error: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status']}; formal_eligible={report['formal_eligible']}; {output}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
