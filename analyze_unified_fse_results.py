#!/usr/bin/env python3
"""Generate the unified FSE-oriented analysis without rerunning experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence

from mas_faults.unified_fse_analysis import (
    canonical_job_key,
    cluster_bootstrap_paired_difference,
    cluster_bootstrap_rate,
    cluster_sign_flip_test,
    cochran_q_test,
    derive_propagation_class,
    holm_adjust,
    injection_step,
    is_clean,
    labels,
    load_and_audit_manifest,
    mcnemar_exact_test,
    repeat_index,
    summarize_metrics,
    summarize_repetition_stability,
)


OUTCOME_METRICS = (
    "clean_final_success",
    "fault_final_success",
    "a_exposure",
    "m_propagation",
    "recovery",
)


def _metric_value(row: dict[str, Any], metric: str) -> bool:
    if metric in {"clean_final_success", "fault_final_success"}:
        return bool(row.get("final_task_success"))
    if metric == "a_exposure":
        return bool(labels(row.get("observed_A_symptom")))
    if metric == "m_propagation":
        return bool(labels(row.get("observed_M_consequence")))
    if metric == "recovery":
        return bool(row.get("recovery_detected"))
    raise KeyError(metric)


def _metric_eligible(row: dict[str, Any], metric: str) -> bool:
    if metric == "clean_final_success":
        return is_clean(row)
    return bool(row.get("fault_applied"))


def _cluster_id(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}::{row.get('task_id')}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _csv_value(value) for key, value in metrics.items()}


def _clean_match_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("dataset", "")),
        str(row.get("model_label") or row.get("model") or ""),
        str(row.get("task_id")),
        str(row.get("topology")),
        repeat_index(row),
    )


def _summary_with_cluster_intervals(
    rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    records = list(rows)
    summary = summarize_metrics(records)
    clean = [row for row in records if is_clean(row)]
    faults = [row for row in records if bool(row.get("fault_applied"))]
    clean_by_key = {_clean_match_key(row): row for row in clean}
    conditional = [
        row
        for row in faults
        if _clean_match_key(row) in clean_by_key
        and bool(clean_by_key[_clean_match_key(row)].get("final_task_success"))
    ]
    specifications = (
        ("clean_success_rate", clean, lambda row: bool(row.get("final_task_success"))),
        (
            "a_exposure_rate",
            faults,
            lambda row: bool(labels(row.get("observed_A_symptom"))),
        ),
        (
            "m_propagation_rate",
            faults,
            lambda row: bool(labels(row.get("observed_M_consequence"))),
        ),
        ("recovery_rate", faults, lambda row: bool(row.get("recovery_detected"))),
        (
            "fault_final_failure_rate",
            faults,
            lambda row: not bool(row.get("final_task_success")),
        ),
        (
            "clean_conditional_loss_rate",
            conditional,
            lambda row: not bool(row.get("final_task_success")),
        ),
    )
    for metric_index, (metric, source, value) in enumerate(specifications):
        derived = [
            {
                "cluster_id": _cluster_id(row),
                "value": value(row),
            }
            for row in source
        ]
        interval = cluster_bootstrap_rate(
            derived,
            value_key="value",
            cluster_key="cluster_id",
            resamples=bootstrap_samples,
            seed=seed + metric_index,
        )
        summary[f"{metric}_lower_95"] = interval["lower_95"]
        summary[f"{metric}_upper_95"] = interval["upper_95"]
        summary[f"{metric}_cluster_count"] = interval["cluster_count"]
    return summary


def _group_summary(
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = []
        for key in keys:
            if key == "injection_step":
                values.append(injection_step(row))
            else:
                values.append(str(row.get(key, "")))
        grouped[tuple(values)].append(row)
    output = []
    for group_index, (group_key, group) in enumerate(sorted(grouped.items())):
        output.append(
            {
                **dict(zip(keys, group_key)),
                **_summary_with_cluster_intervals(
                    group,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + group_index * 10,
                ),
            }
        )
    return output


def _apply_holm_families(
    rows: Sequence[dict[str, Any]], *, family_keys: Sequence[str]
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        grouped[tuple(str(row.get(key, "")) for key in family_keys)].append(index)
    for indices in grouped.values():
        adjusted = holm_adjust(
            [float(output[index]["cluster_sign_flip_p_value_raw"]) for index in indices]
        )
        mcnemar_adjusted = holm_adjust(
            [float(output[index]["mcnemar_p_value_raw"]) for index in indices]
        )
        for index, corrected, mcnemar_corrected in zip(
            indices, adjusted, mcnemar_adjusted
        ):
            output[index]["p_value_raw"] = output[index][
                "cluster_sign_flip_p_value_raw"
            ]
            output[index]["p_value_holm"] = corrected
            output[index]["mcnemar_p_value_holm"] = mcnemar_corrected
    return output


def _run_paired_family(
    *,
    indices: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    labels_in_order: Sequence[str],
    group_fields: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    omnibus_common = set.intersection(
        *(set(indices[label]) for label in labels_in_order)
    )
    results: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(OUTCOME_METRICS):
        omnibus_eligible_keys = [
            key
            for key in sorted(omnibus_common)
            if all(_metric_eligible(indices[label][key], metric) for label in labels_in_order)
        ]
        if omnibus_eligible_keys:
            matrix = [
                [
                    _metric_value(indices[label][key], metric)
                    for label in labels_in_order
                ]
                for key in omnibus_eligible_keys
            ]
            omnibus = cochran_q_test(matrix)
        else:
            omnibus = {
                "q_statistic": None,
                "degrees_of_freedom": len(labels_in_order) - 1,
                "p_value": None,
            }
        pair_rows = []
        for pair_index, (left_label, right_label) in enumerate(
            itertools.combinations(labels_in_order, 2)
        ):
            pair_common = set(indices[left_label]) & set(indices[right_label])
            eligible_keys = [
                key
                for key in sorted(pair_common)
                if _metric_eligible(indices[left_label][key], metric)
                and _metric_eligible(indices[right_label][key], metric)
            ]
            if not eligible_keys:
                continue
            pair_records = []
            for key in eligible_keys:
                left_row = indices[left_label][key]
                right_row = indices[right_label][key]
                pair_records.append(
                    {
                        "cluster_id": _cluster_id(left_row),
                        "left": _metric_value(left_row, metric),
                        "right": _metric_value(right_row, metric),
                    }
                )
            exact = mcnemar_exact_test(
                [record["left"] for record in pair_records],
                [record["right"] for record in pair_records],
            )
            interval = cluster_bootstrap_paired_difference(
                pair_records,
                left_key="left",
                right_key="right",
                cluster_key="cluster_id",
                resamples=bootstrap_samples,
                seed=seed + metric_index * 100 + pair_index,
            )
            cluster_test = cluster_sign_flip_test(
                pair_records,
                left_key="left",
                right_key="right",
                cluster_key="cluster_id",
                resamples=bootstrap_samples,
                seed=seed + 10_000 + metric_index * 100 + pair_index,
            )
            pair_rows.append(
                {
                    **group_fields,
                    "metric": metric,
                    "left": left_label,
                    "right": right_label,
                    "common_job_count": len(pair_common),
                    "eligible_pair_count": len(eligible_keys),
                    "omnibus_common_job_count": len(omnibus_common),
                    "omnibus_eligible_count": len(omnibus_eligible_keys),
                    "omnibus_q": omnibus["q_statistic"],
                    "omnibus_df": omnibus["degrees_of_freedom"],
                    "omnibus_p_value": omnibus["p_value"],
                    "risk_difference": interval["estimate"],
                    "risk_difference_lower_95": interval["lower_95"],
                    "risk_difference_upper_95": interval["upper_95"],
                    "job_weighted_risk_difference": exact["risk_difference"],
                    "matched_odds_ratio": exact["matched_odds_ratio"],
                    "discordant_count": exact["discordant_count"],
                    "left_success_right_failure": exact[
                        "left_success_right_failure"
                    ],
                    "left_failure_right_success": exact[
                        "left_failure_right_success"
                    ],
                    "cluster_equal_weight_risk_difference": cluster_test[
                        "task_weighted_risk_difference"
                    ],
                    "cluster_sign_flip_p_value_raw": cluster_test["p_value"],
                    "cluster_sign_flip_exact": cluster_test["exact"],
                    "cluster_sign_flip_permutations": cluster_test[
                        "permutations"
                    ],
                    "mcnemar_p_value_raw": exact["p_value"],
                    "bootstrap_cluster_count": interval["cluster_count"],
                    "bootstrap_samples": interval["bootstrap_samples"],
                }
            )
        results.extend(pair_rows)
    return results


def _paired_model_tests(
    rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset"))].append(row)
    results = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        model_labels = sorted({str(row.get("model_label")) for row in dataset_rows})
        if len(model_labels) < 2:
            continue
        indices: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
        for label in model_labels:
            index = {}
            for row in dataset_rows:
                if str(row.get("model_label")) != label:
                    continue
                key = canonical_job_key(row)
                if key in index:
                    raise ValueError(f"duplicate model paired key: {dataset}/{label}/{key}")
                index[key] = row
            indices[label] = index
        results.extend(
            _run_paired_family(
                indices=indices,
                labels_in_order=model_labels,
                group_fields={"dataset": dataset},
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    return _apply_holm_families(results, family_keys=("dataset", "metric"))


def _topology_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("task_id")),
        str(row.get("condition")),
        injection_step(row),
        repeat_index(row),
    )


def _paired_topology_tests(
    rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("model_label")))].append(row)
    results = []
    preferred = ("sequential", "flat", "hierarchical")
    for (dataset, model_label), group in sorted(grouped.items()):
        available = {str(row.get("topology")) for row in group}
        topologies = [item for item in preferred if item in available]
        topologies.extend(sorted(available - set(topologies)))
        if len(topologies) < 2:
            continue
        indices: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
        for topology in topologies:
            index = {}
            for row in group:
                if str(row.get("topology")) != topology:
                    continue
                key = _topology_key(row)
                if key in index:
                    raise ValueError(
                        f"duplicate topology paired key: {dataset}/{model_label}/{topology}/{key}"
                    )
                index[key] = row
            indices[topology] = index
        results.extend(
            _run_paired_family(
                indices=indices,
                labels_in_order=topologies,
                group_fields={"dataset": dataset, "model_label": model_label},
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1000,
            )
        )
    return _apply_holm_families(results, family_keys=("dataset", "metric"))


def _axis_group(row: dict[str, Any]) -> str:
    system = bool(labels(row.get("system_consequences")))
    semantic = bool(labels(row.get("semantic_consequences")))
    if system and semantic:
        return "system_and_semantic"
    if system:
        return "system_only"
    if semantic:
        return "semantic_only"
    return "neither_axis"


def _system_semantic_summary(
    rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("fault_applied")):
            continue
        grouped[
            (
                str(row.get("dataset")),
                str(row.get("model_label")),
                _axis_group(row),
            )
        ].append(row)
    output = []
    for index, ((dataset, model_label, group), records) in enumerate(
        sorted(grouped.items())
    ):
        derived = [
            {
                **row,
                "cluster_id": _cluster_id(row),
                "final_failure": not bool(row.get("final_task_success")),
                "recovered": bool(row.get("recovery_detected")),
            }
            for row in records
        ]
        failure = cluster_bootstrap_rate(
            derived,
            value_key="final_failure",
            cluster_key="cluster_id",
            resamples=bootstrap_samples,
            seed=seed + index * 2,
        )
        recovery = cluster_bootstrap_rate(
            derived,
            value_key="recovered",
            cluster_key="cluster_id",
            resamples=bootstrap_samples,
            seed=seed + index * 2 + 1,
        )
        output.append(
            {
                "scope": "model_domain",
                "dataset": dataset,
                "model_label": model_label,
                "axis_group": group,
                "run_count": len(records),
                "final_failure_rate": failure["estimate"],
                "final_failure_lower_95": failure["lower_95"],
                "final_failure_upper_95": failure["upper_95"],
                "recovery_rate": recovery["estimate"],
                "recovery_lower_95": recovery["lower_95"],
                "recovery_upper_95": recovery["upper_95"],
                "cluster_count": failure["cluster_count"],
            }
        )
    return output


def _system_semantic_global_summary(
    rows: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("fault_applied")):
            grouped[_axis_group(row)].append(row)
    output = []
    for index, (group, records) in enumerate(sorted(grouped.items())):
        derived = [
            {
                **row,
                "cluster_id": _cluster_id(row),
                "final_failure": not bool(row.get("final_task_success")),
                "recovered": bool(row.get("recovery_detected")),
            }
            for row in records
        ]
        failure = cluster_bootstrap_rate(
            derived,
            value_key="final_failure",
            cluster_key="cluster_id",
            resamples=bootstrap_samples,
            seed=seed + 5000 + index * 2,
        )
        recovery = cluster_bootstrap_rate(
            derived,
            value_key="recovered",
            cluster_key="cluster_id",
            resamples=bootstrap_samples,
            seed=seed + 5000 + index * 2 + 1,
        )
        output.append(
            {
                "scope": "global",
                "dataset": "ALL",
                "model_label": "ALL",
                "axis_group": group,
                "run_count": len(records),
                "final_failure_rate": failure["estimate"],
                "final_failure_lower_95": failure["lower_95"],
                "final_failure_upper_95": failure["upper_95"],
                "recovery_rate": recovery["estimate"],
                "recovery_lower_95": recovery["lower_95"],
                "recovery_upper_95": recovery["upper_95"],
                "cluster_count": failure["cluster_count"],
            }
        )
    return output


def _repeat_outputs(
    rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    overall = summarize_repetition_stability(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("model_label")))].append(row)
    variance_rows = []
    stability_summary = []
    for (dataset, model_label), group in sorted(grouped.items()):
        summary = summarize_repetition_stability(group)
        stability_summary.append(
            {
                "dataset": dataset,
                "model_label": model_label,
                "complete_cell_count": summary["complete_cell_count"],
                "incomplete_cell_count": summary["incomplete_cell_count"],
                "outcome_stability_rate": summary["outcome_stability_rate"],
                "propagation_class_stability_rate": summary[
                    "propagation_class_stability_rate"
                ],
                "m_consequence_stability_rate": summary[
                    "m_consequence_stability_rate"
                ],
            }
        )
        for metric, values in summary["repeat_variance"].items():
            variance_rows.append(
                {
                    "dataset": dataset,
                    "model_label": model_label,
                    "metric": metric,
                    **values,
                    "repeat_1": summary["repeat_metrics"][1][metric],
                    "repeat_2": summary["repeat_metrics"][2][metric],
                    "repeat_3": summary["repeat_metrics"][3][metric],
                }
            )
    return overall, variance_rows, stability_summary


def _representative_cases(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = (
        "masked",
        "exposed_at_A_only",
        "propagated_to_M_recovered",
        "propagated_to_M_final_success",
        "propagated_to_M_final_failure",
    )
    selected = []
    for propagation_class in preferred:
        candidates = sorted(
            (
                row
                for row in rows
                if derive_propagation_class(row) == propagation_class
            ),
            key=lambda row: (
                str(row.get("dataset")),
                str(row.get("model_label")),
                str(row.get("run_id")),
            ),
        )
        seen_datasets = set()
        for row in candidates:
            dataset = str(row.get("dataset"))
            if dataset in seen_datasets:
                continue
            seen_datasets.add(dataset)
            selected.append(
                {
                    "propagation_class": propagation_class,
                    "dataset": dataset,
                    "model_label": row.get("model_label"),
                    "run_id": row.get("run_id"),
                    "trace_id": row.get("trace_id"),
                    "task_id": row.get("task_id"),
                    "topology": row.get("topology"),
                    "condition": row.get("condition"),
                    "injection_step": row.get("injection_step"),
                    "observed_A_symptom": row.get("observed_A_symptom"),
                    "observed_M_consequence": row.get("observed_M_consequence"),
                    "recovery_detected": row.get("recovery_detected"),
                    "recovery_type": row.get("recovery_type"),
                    "recovery_evidence": row.get("recovery_evidence"),
                    "final_task_success": row.get("final_task_success"),
                    "original_message": row.get("original_message"),
                    "delivered_message": row.get("delivered_message"),
                    "event_chain": [
                        {
                            key: event.get(key)
                            for key in (
                                "abstract_step",
                                "source_agent",
                                "target_agent",
                                "fault_applied",
                                "observed_A_symptom",
                                "observed_runtime_effect",
                                "delivery_count",
                            )
                        }
                        for event in row.get("events", [])
                    ],
                }
            )
            if len(seen_datasets) >= 2:
                break
    return selected


def _protocol(
    config: dict[str, Any], bootstrap_samples: int, analysis_seed: int
) -> dict[str, Any]:
    return {
        "analysis_id": config.get("analysis_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_type": "controlled repeated benchmark experiment",
        "repeat_interpretation": (
            "three repeated executions; independence is not established by stored traces"
        ),
        "primary_unit": "task-level clustered communication-fault run",
        "exact_pairing_key": (
            "dataset × task_id × topology × condition × injection_step × repeat_index"
        ),
        "primary_metrics": [
            "clean final success rate",
            "actually-applied-fault final success/failure rate",
            "A-layer exposure rate",
            "M-layer propagation rate",
            "recovery rate",
            "fault final failure rate",
            "clean-conditional outcome loss",
            "propagation class",
        ],
        "repetition_metrics": [
            "final outcome stability",
            "propagation-class stability",
            "M-consequence stability",
            "mean and sample standard deviation across repetitions",
        ],
        "paired_tests": {
            "primary_pairwise": (
                "two-sided task-cluster sign-flip permutation test; exact for "
                "at most 12 tasks and Monte Carlo otherwise"
            ),
            "sensitivity_omnibus": "job-level Cochran's Q",
            "sensitivity_pairwise": "job-level two-sided exact McNemar",
            "effect_size": (
                "task-equal-weight paired risk difference with task-cluster bootstrap "
                "CI; job-weighted risk difference and matched odds ratio are sensitivity "
                "statistics"
            ),
            "multiple_comparison": "Holm correction within dataset and metric",
        },
        "uncertainty": {
            "method": "task-cluster percentile bootstrap",
            "samples": bootstrap_samples,
            "confidence_level": 0.95,
            "seed": analysis_seed,
        },
        "causal_boundary": (
            "System/semantic consequences are post-fault states; associations with "
            "failure are not interpreted as causal effects. Cross-domain and cross-model "
            "contrasts are configuration-level associations because tasks, providers, and "
            "deployments are not jointly randomized."
        ),
        "latency_boundary": (
            "Cross-provider latency is descriptive because API and local hardware differ."
        ),
        "token_boundary": (
            "Cross-model token totals are descriptive because tokenizers differ."
        ),
        "source_standard": [
            "Zhang et al., Not as Sweet by Another Name, ASE 2026",
            "ACM SIGSOFT Empirical Standards",
        ],
    }


def run_analysis(
    config: dict[str, Any], *, bootstrap_samples: int = 10_000, analysis_seed: int = 20260825
) -> dict[str, Any]:
    loaded = load_and_audit_manifest(config)
    rows = loaded["rows"]
    repetition, repeat_variance, stability_summary = _repeat_outputs(rows)
    system_semantic_detail = _system_semantic_summary(
        rows, bootstrap_samples, analysis_seed
    )
    system_semantic_global = _system_semantic_global_summary(
        rows, bootstrap_samples, analysis_seed
    )
    result = {
        "analysis_config": config,
        "analysis_protocol": _protocol(config, bootstrap_samples, analysis_seed),
        "input_audit": {
            "audit": loaded["audit"],
            "inputs": loaded["inputs"],
            "experiments": loaded["experiments"],
        },
        "overall_summary": _summary_with_cluster_intervals(
            rows, bootstrap_samples=bootstrap_samples, seed=analysis_seed + 20_000
        ),
        "summary_by_model": _group_summary(
            rows,
            ("model_label",),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 21_000,
        ),
        "summary_by_domain": _group_summary(
            rows,
            ("dataset",),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 22_000,
        ),
        "summary_by_model_domain": _group_summary(
            rows,
            ("dataset", "model_label"),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 23_000,
        ),
        "summary_by_topology": _group_summary(
            rows,
            ("dataset", "model_label", "topology"),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 24_000,
        ),
        "summary_by_topology_global": _group_summary(
            rows,
            ("topology",),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 25_000,
        ),
        "summary_by_fault": _group_summary(
            rows,
            ("dataset", "model_label", "condition", "injection_step"),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 26_000,
        ),
        "summary_by_fault_global": _group_summary(
            rows,
            ("condition", "injection_step"),
            bootstrap_samples=bootstrap_samples,
            seed=analysis_seed + 27_000,
        ),
        "repetition_stability": repetition,
        "repeat_variance": repeat_variance,
        "stability_by_model_domain": stability_summary,
        "paired_model_tests": _paired_model_tests(
            rows, bootstrap_samples, analysis_seed
        ),
        "paired_topology_tests": _paired_topology_tests(
            rows, bootstrap_samples, analysis_seed
        ),
        "system_semantic_summary": system_semantic_global + system_semantic_detail,
        "system_semantic_global_summary": system_semantic_global,
        "representative_cases": _representative_cases(rows),
    }
    return result


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.1f}%"


def _signed_pct(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):+.1f}%"


def _pvalue(value: Any) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    return f"{numeric:.2e}" if numeric < 0.001 else f"{numeric:.3f}"


def _pct_ci(row: dict[str, Any], metric: str) -> str:
    return (
        f"{_pct(row.get(metric))} "
        f"[{_pct(row.get(f'{metric}_lower_95'))}, "
        f"{_pct(row.get(f'{metric}_upper_95'))}]"
    )


def _success_ci_from_failure(row: dict[str, Any]) -> str:
    failure = float(row["fault_final_failure_rate"])
    lower = row.get("fault_final_failure_rate_lower_95")
    upper = row.get("fault_final_failure_rate_upper_95")
    return (
        f"{_pct(1.0 - failure)} "
        f"[{_pct(None if upper is None else 1.0 - float(upper))}, "
        f"{_pct(None if lower is None else 1.0 - float(lower))}]"
    )


def _render_findings(result: dict[str, Any]) -> str:
    overall = result["overall_summary"]
    transitions = overall["transition_counts"]
    m_success = transitions.get("propagated_to_M_recovered", 0) + transitions.get(
        "propagated_to_M_final_success", 0
    )
    lines = [
        "# MAS 通信故障统一统计与论文 Finding",
        "",
        "## 实验与统计口径",
        "",
        (
            f"本报告分析 {overall['run_count']:,} 条正式记录。三轮仅解释为“三次重复执行”；"
            "存档 trace 不能证明它们统计独立，也不宣称为 API 可控随机种子。模型和拓扑比较只使用完全一致的 task、condition、"
            "injection step 与 repeat 交集。"
        ),
        "",
        (
            "置信区间采用 task-cluster bootstrap；主显著性检验以 task 为推断单位，使用"
            "双侧 cluster sign-flip permutation，并在每个 dataset × metric 家族内进行 "
            "Holm 校正。逐 job 的 Cochran's Q 与 exact McNemar 仅作为敏感性分析。"
        ),
        "",
        "## 总体结果",
        "",
        "| Runs | Faults | A 暴露 [95% CI] | M 传播 [95% CI] | Recovery [95% CI] | Fault 最终失败 [95% CI] |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['run_count']:,} | {overall['fault_count']:,} | "
            f"{_pct_ci(overall, 'a_exposure_rate')} | "
            f"{_pct_ci(overall, 'm_propagation_rate')} | "
            f"{_pct_ci(overall, 'recovery_rate')} | "
            f"{_pct_ci(overall, 'fault_final_failure_rate')} |"
        ),
        "",
        "## Finding 1：故障传播与最终任务结果是不同变量",
        "",
        (
            f"在实际注入 fault 中，{overall['m_propagation_count']:,} 条传播到 M 层；其中至少 "
            f"{m_success:,} 条最终仍成功。最终成功不等于故障被屏蔽，必须同时检查 A 症状、"
            "M consequence 和 recovery trace。"
        ),
        "",
        (
            f"计划执行 {overall['scheduled_fault_count']:,} 条 fault runs，其中 "
            f"{overall['pre_injection_failure_count']:,} 条在故障应用前已发生模型失败；"
            "这些预注入失败单独记录，不计入 A/M/recovery 的 fault-effect 分母。"
        ),
        "",
        "| Propagation class | Count |",
        "|---|---:|",
    ]
    for name, count in sorted(transitions.items()):
        lines.append(f"| `{name}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Finding 2：当前 Flat 配置的较高韧性与独立证据旁路相一致",
            "",
            "| Topology | Fault success [95% CI] | M propagation [95% CI] | Recovery [95% CI] | Conditional loss [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    topology_rows = sorted(
        result["summary_by_topology_global"],
        key=lambda row: str(row["topology"]),
    )
    for row in topology_rows:
        lines.append(
            f"| {str(row['topology']).title()} | "
            f"{_success_ci_from_failure(row)} | "
            f"{_pct_ci(row, 'm_propagation_rate')} | "
            f"{_pct_ci(row, 'recovery_rate')} | "
            f"{_pct_ci(row, 'clean_conditional_loss_rate')} |"
        )
    best_topology = max(
        topology_rows,
        key=lambda row: 1.0 - float(row["fault_final_failure_rate"]),
        default=None,
    )
    if best_topology:
        lines.extend(
            [
                "",
                (
                    f"在当前实现中，`{best_topology['topology']}` 的 fault-run 成功率最高。"
                    "该结果只支持“当前旁路/汇总结构”的结论，不泛化为同名拓扑天然更鲁棒。"
                ),
            ]
        )
    flat_pairs = [
        row
        for row in result["paired_topology_tests"]
        if row["metric"] == "fault_final_success"
        and "flat" in {row["left"], row["right"]}
    ]
    flat_higher = [
        row
        for row in flat_pairs
        if (
            row["left"] == "flat" and float(row["risk_difference"]) > 0
        )
        or (
            row["right"] == "flat" and float(row["risk_difference"]) < 0
        )
    ]
    flat_higher_significant = sum(
        float(row["p_value_holm"]) < 0.05 for row in flat_higher
    )
    non_flat_significant = sum(
        row["metric"] == "fault_final_success"
        and "flat" not in {row["left"], row["right"]}
        and float(row["p_value_holm"]) < 0.05
        for row in result["paired_topology_tests"]
    )
    lines.extend(
        [
            "",
            (
                f"在 {len(flat_pairs)} 个 Flat 与非 Flat 的严格配对比较中，"
                f"Flat 点估计更高 {len(flat_higher)} 次，其中 {flat_higher_significant} 次"
                "在 task-cluster Holm 校正后达到 0.05；"
                f"Sequential 与 Hierarchical 之间显著差异为 {non_flat_significant} 次。"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Finding 3：在当前配置中，故障机制和注入位置与不可逆性相关",
            "",
            (
                "下表跨模型、任务域和拓扑作描述性汇总；不同 condition 的覆盖构成可能不同，"
                "因此不能把条件间比例差异单独解释为 fault type 或注入位置的因果效应。"
            ),
            "",
            "| Condition | Step | Fault failure [95% CI] | M propagation [95% CI] | Recovery [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    fault_rows = [
        row
        for row in result["summary_by_fault_global"]
        if row["fault_count"] > 0
    ]
    for row in sorted(
        fault_rows,
        key=lambda item: float(item["fault_final_failure_rate"]),
        reverse=True,
    ):
        lines.append(
            f"| `{row['condition']}` | {row['injection_step'] or '-'} | "
            f"{_pct_ci(row, 'fault_final_failure_rate')} | "
            f"{_pct_ci(row, 'm_propagation_rate')} | "
            f"{_pct_ci(row, 'recovery_rate')} |"
        )

    lines.extend(
        [
            "",
            "## Finding 4：System consequence 与 Semantic consequence 的传播路径不同",
            "",
            (
                "System/semantic 是 fault 之后观察到的状态，不是随机分配的处理变量。"
                "因此下表用于描述关联，不解释为 consequence 对失败的因果效应。"
            ),
            "",
            "| Dataset | Model | Axis | Runs | Failure [95% CI] | Recovery [95% CI] |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in result["system_semantic_global_summary"]:
        lines.append(
            f"| {row['dataset']} | {row['model_label']} | {row['axis_group']} | "
            f"{row['run_count']} | {_pct(row['final_failure_rate'])} "
            f"[{_pct(row['final_failure_lower_95'])}, {_pct(row['final_failure_upper_95'])}] | "
            f"{_pct(row['recovery_rate'])} "
            f"[{_pct(row['recovery_lower_95'])}, {_pct(row['recovery_upper_95'])}] |"
        )

    lines.extend(
        [
            "",
            "## Finding 5：不同任务域呈现不同 consequence profile",
            "",
            "| Dataset | Clean success [95% CI] | Fault failure [95% CI] | M propagation [95% CI] | Conditional loss [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(result["summary_by_domain"], key=lambda item: item["dataset"]):
        lines.append(
            f"| {row['dataset']} | {_pct_ci(row, 'clean_success_rate')} | "
            f"{_pct_ci(row, 'fault_final_failure_rate')} | "
            f"{_pct_ci(row, 'm_propagation_rate')} | "
            f"{_pct_ci(row, 'clean_conditional_loss_rate')} |"
        )
    lines.extend(
        [
            "",
            (
                "跨域任务不是同一批样本，不能用配对检验把这些比例解释为 benchmark 难度排名。"
                "域间结论以 task-cluster 区间、条件损失和 consequence 构成为主。"
            ),
            "",
            "## Finding 6：不同模型配置呈现不同基础可靠性，但均未消除结构性通信故障",
            "",
            "| Model | Clean success [95% CI] | Fault success [95% CI] | M propagation [95% CI] | Recovery [95% CI] | Conditional loss [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(result["summary_by_model"], key=lambda item: item["model_label"]):
        lines.append(
            f"| {row['model_label']} | {_pct_ci(row, 'clean_success_rate')} | "
            f"{_success_ci_from_failure(row)} | "
            f"{_pct_ci(row, 'm_propagation_rate')} | "
            f"{_pct_ci(row, 'recovery_rate')} | "
            f"{_pct_ci(row, 'clean_conditional_loss_rate')} |"
        )
    model_effect_rows = sorted(
        (
            row
            for row in result["paired_model_tests"]
            if row["metric"] == "fault_final_success"
        ),
        key=lambda row: (str(row["dataset"]), str(row["left"]), str(row["right"])),
    )
    lines.extend(
        [
            "",
            (
                "模型、provider 与部署方式在本实验中并未完全解耦；因此以下差异描述的是"
                "当前模型配置组合，不能单独归因于模型能力。"
            ),
            "",
            "### 配对模型效应量",
            "",
            (
                "风险差定义为 left − right，先在 task 内求均值再对 task 等权；"
                "区间为 task-cluster bootstrap 95% CI。"
                "主检验以 task 为推断单位，McNemar 仅用于逐 job 敏感性分析。"
            ),
            "",
            (
                "| Dataset | Comparison | Paired jobs | Risk difference [95% CI] | "
                "Task-cluster Holm p | McNemar Holm p（敏感性） |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in model_effect_rows:
        lines.append(
            f"| {row['dataset']} | {row['left']} − {row['right']} | "
            f"{row['eligible_pair_count']} | {_signed_pct(row['risk_difference'])} "
            f"[{_signed_pct(row['risk_difference_lower_95'])}, "
            f"{_signed_pct(row['risk_difference_upper_95'])}] | "
            f"{_pvalue(row['p_value_holm'])} | "
            f"{_pvalue(row['mcnemar_p_value_holm'])} |"
        )
    significant_model = [
        row
        for row in result["paired_model_tests"]
        if row["metric"] == "fault_final_success" and row["p_value_holm"] < 0.05
    ]
    lines.extend(
        [
            "",
            (
                f"严格配对的实际注入 fault 最终成功比较中，有 {len(significant_model)} 组两两差异在 Holm "
                "校正后达到 0.05 阈值。统计显著不等于实际重要，正文应同时引用风险差和区间。"
            ),
            "",
            "## Finding 7：三次重复揭示最终结果之外的 trace 不稳定性",
            "",
            "| Dataset | Model | Outcome stability | Propagation stability | M-consequence stability |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in result["stability_by_model_domain"]:
        lines.append(
            f"| {row['dataset']} | {row['model_label']} | "
            f"{_pct(row['outcome_stability_rate'])} | "
            f"{_pct(row['propagation_class_stability_rate'])} | "
            f"{_pct(row['m_consequence_stability_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界与投稿前检查",
            "",
            "1. 三轮是重复执行；现有 trace 不足以证明统计独立，也不保证是 API 可控 random seeds。",
            "2. 主结论来自 confirmation matrices；历史 discovery runs 不混入显著性检验。",
            "3. 不同 provider/hardware 的 latency 只作描述，不解释为模型固有速度。",
            "4. 不同 tokenizer 的 token 数只用于成本审计。",
            "5. 建议由两位研究者分层复核 30–50 条 trace，并报告 Cohen's kappa。",
            "6. 可在小型代表子集上增加 k=5 敏感性分析，但它不阻塞当前主统计。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "analysis_protocol.json", result["analysis_protocol"])
    _write_json(output_dir / "analysis_config.json", result["analysis_config"])
    _write_json(output_dir / "input_audit.json", result["input_audit"])
    _write_json(
        output_dir / "overall_summary.json",
        {
            "overall": result["overall_summary"],
            "by_model": result["summary_by_model"],
            "by_domain": result["summary_by_domain"],
            "stability_by_model_domain": result["stability_by_model_domain"],
        },
    )
    _write_csv(
        output_dir / "summary_by_model_domain.csv",
        result["summary_by_model_domain"],
    )
    _write_csv(output_dir / "summary_by_topology.csv", result["summary_by_topology"])
    _write_csv(output_dir / "summary_by_fault.csv", result["summary_by_fault"])
    _write_csv(output_dir / "repeat_variance.csv", result["repeat_variance"])
    _write_csv(
        output_dir / "execution_stability.csv",
        result["repetition_stability"]["cells"],
    )
    _write_csv(
        output_dir / "paired_model_tests.csv", result["paired_model_tests"]
    )
    _write_csv(
        output_dir / "paired_topology_tests.csv", result["paired_topology_tests"]
    )
    _write_csv(
        output_dir / "system_semantic_summary.csv",
        result["system_semantic_summary"],
    )
    (output_dir / "representative_cases.jsonl").write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
            for case in result["representative_cases"]
        ),
        encoding="utf-8",
    )
    (output_dir / "paper_findings_zh.md").write_text(
        _render_findings(result), encoding="utf-8"
    )

    output_files = sorted(
        path for path in output_dir.iterdir() if path.name != "reproducibility_manifest.json"
    )
    manifest = {
        "analysis_id": result["analysis_protocol"].get("analysis_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_rows": result["input_audit"]["audit"]["total_rows"],
        "input_files": result["input_audit"]["inputs"],
        "outputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in output_files
        ],
        "no_api_calls": True,
    }
    _write_json(output_dir / "reproducibility_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_analysis(
        config,
        bootstrap_samples=args.bootstrap_samples,
        analysis_seed=args.analysis_seed,
    )
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows": result["input_audit"]["audit"]["total_rows"],
                "paired_model_tests": len(result["paired_model_tests"]),
                "paired_topology_tests": len(result["paired_topology_tests"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
