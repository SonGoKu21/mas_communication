"""Auditable statistics for the unified FSE communication-fault study."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence


NONE_LABELS = {"", "none", "null", "clean"}


def labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        values = (value,)
    return tuple(
        str(item)
        for item in values
        if item is not None and str(item).strip().lower() not in NONE_LABELS
    )


def is_clean(row: dict[str, Any]) -> bool:
    condition = str(row.get("condition") or "").strip().lower()
    if condition:
        return condition in {"clean", "none"}
    return (
        str(row.get("fault_type") or "").strip().lower() in {"clean", "none"}
        and not bool(row.get("fault_applied"))
    )


def repeat_index(row: dict[str, Any]) -> int:
    return int(row.get("repeat_index", row.get("seed_or_run_index", 1)) or 1)


def injection_step(row: dict[str, Any]) -> str:
    value = row.get("injection_step")
    if value not in {None, "", "none"}:
        return str(value)
    match = re.search(r"_step([234])(?:$|_)", condition_name(row))
    return match.group(1) if match else ""


def condition_name(row: dict[str, Any]) -> str:
    return str(row.get("condition") or row.get("fault_type") or "unknown")


def derive_propagation_class(row: dict[str, Any]) -> str:
    if is_clean(row):
        return "clean"
    if not bool(row.get("fault_applied")):
        return "pre_injection_failure"
    a_exposed = bool(labels(row.get("observed_A_symptom")))
    m_propagated = bool(labels(row.get("observed_M_consequence")))
    success = bool(row.get("final_task_success"))
    response_detected = bool(row.get("recovery_detected"))
    if not a_exposed and not m_propagated:
        return "masked"
    if not m_propagated:
        return "exposed_at_A_only" if success else "exposed_at_A_final_failure"
    if response_detected and success:
        return "propagated_to_M_recovered"
    if success:
        return "propagated_to_M_final_success"
    return "propagated_to_M_final_failure"


def canonical_job_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("task_id")),
        str(row.get("topology")),
        condition_name(row),
        injection_step(row),
        repeat_index(row),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {error}") from error
    return rows


def load_and_audit_manifest(config: dict[str, Any]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    global_ids: set[tuple[str, str, str]] = set()
    for spec in config.get("experiments", []):
        dataset = str(spec["dataset"])
        label = str(spec["label"])
        experiment_rows: list[dict[str, Any]] = []
        for raw_path in spec["paths"]:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded = _load_jsonl(path)
            inputs.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "path": str(path),
                    "row_count": len(loaded),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            experiment_rows.extend(loaded)

        run_ids = [str(row.get("run_id")) for row in experiment_rows]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError(f"duplicate run_id in {dataset}/{label}")
        expected_rows = spec.get("expected_rows")
        if expected_rows is not None and len(experiment_rows) != int(expected_rows):
            raise ValueError(
                f"row count mismatch in {dataset}/{label}: "
                f"{len(experiment_rows)} != {expected_rows}"
            )
        actual_repeats = sorted({repeat_index(row) for row in experiment_rows})
        expected_repeats = spec.get("expected_repeats")
        if expected_repeats is not None and actual_repeats != sorted(
            int(item) for item in expected_repeats
        ):
            raise ValueError(
                f"repeat coverage mismatch in {dataset}/{label}: "
                f"{actual_repeats} != {expected_repeats}"
            )
        if expected_repeats is not None:
            expected_repeat_tuple = tuple(
                sorted(int(item) for item in expected_repeats)
            )
            repeat_cells: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
            for row in experiment_rows:
                cell = (
                    str(row.get("task_id")),
                    str(row.get("topology")),
                    condition_name(row),
                    injection_step(row),
                )
                repeat_cells[cell].append(repeat_index(row))
            for cell, repeats in repeat_cells.items():
                if tuple(sorted(repeats)) != expected_repeat_tuple:
                    raise ValueError(
                        f"cell repeat coverage mismatch in {dataset}/{label}: "
                        f"{cell} has {sorted(repeats)} != {list(expected_repeat_tuple)}"
                    )

        normalized = []
        for raw in experiment_rows:
            row = {
                **raw,
                "dataset": dataset,
                "model_label": label,
                "repeat_index": repeat_index(raw),
                "condition": condition_name(raw),
                "injection_step_normalized": injection_step(raw),
            }
            row["propagation_class_normalized"] = derive_propagation_class(row)
            identity = (dataset, label, str(row.get("run_id")))
            if identity in global_ids:
                raise ValueError(f"duplicate run_id identity: {identity}")
            global_ids.add(identity)
            normalized.append(row)
        all_rows.extend(normalized)
        experiments.append(
            {
                "dataset": dataset,
                "label": label,
                "row_count": len(normalized),
                "task_count": len({str(row.get("task_id")) for row in normalized}),
                "repeat_indices": actual_repeats,
                "models": sorted({str(row.get("model")) for row in normalized}),
                "providers": sorted(
                    {str(row.get("provider")) for row in normalized}
                ),
            }
        )

    expected_total = config.get("expected_total_rows")
    if expected_total is not None and len(all_rows) != int(expected_total):
        raise ValueError(f"total row mismatch: {len(all_rows)} != {expected_total}")
    actual_labels = sorted({str(spec["label"]) for spec in config.get("experiments", [])})
    expected_models = config.get("expected_models")
    if expected_models is not None and actual_labels != sorted(
        str(item) for item in expected_models
    ):
        raise ValueError(
            f"model label coverage mismatch: {actual_labels} != {expected_models}"
        )
    actual_topologies = sorted({str(row.get("topology")) for row in all_rows})
    expected_topologies = config.get("expected_topologies")
    if expected_topologies is not None and actual_topologies != sorted(
        str(item) for item in expected_topologies
    ):
        raise ValueError(
            f"topology coverage mismatch: {actual_topologies} != {expected_topologies}"
        )
    return {
        "rows": all_rows,
        "inputs": inputs,
        "experiments": experiments,
        "audit": {
            "passed": True,
            "total_rows": len(all_rows),
            "unique_experiment_run_ids": len(global_ids),
            "experiment_count": len(experiments),
            "model_labels": actual_labels,
            "topologies": actual_topologies,
        },
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    clean = [row for row in row_list if is_clean(row)]
    scheduled_faults = [row for row in row_list if not is_clean(row)]
    faults = [row for row in scheduled_faults if bool(row.get("fault_applied"))]
    a_count = sum(bool(labels(row.get("observed_A_symptom"))) for row in faults)
    m_count = sum(bool(labels(row.get("observed_M_consequence"))) for row in faults)
    recovery_count = sum(bool(row.get("recovery_detected")) for row in faults)
    failure_count = sum(not bool(row.get("final_task_success")) for row in faults)
    success_count = sum(bool(row.get("final_task_success")) for row in row_list)
    tokens = [int(row.get("total_tokens") or 0) for row in row_list]
    latencies = [float(row.get("latency_ms") or 0.0) for row in row_list]
    transitions = Counter(derive_propagation_class(row) for row in scheduled_faults)

    def clean_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
        return (
            str(row.get("dataset", "")),
            str(row.get("model_label") or row.get("model") or ""),
            str(row.get("task_id")),
            str(row.get("topology")),
            repeat_index(row),
        )

    clean_by_key = {clean_key(row): row for row in clean}
    matched = [
        (clean_by_key[clean_key(row)], row)
        for row in faults
        if clean_key(row) in clean_by_key
    ]
    clean_success_matched = [
        (baseline, fault)
        for baseline, fault in matched
        if bool(baseline.get("final_task_success"))
    ]
    conditional_losses = sum(
        not bool(fault.get("final_task_success"))
        for _, fault in clean_success_matched
    )
    return {
        "run_count": len(row_list),
        "task_count": len({str(row.get("task_id")) for row in row_list}),
        "clean_count": len(clean),
        "clean_success_count": sum(
            bool(row.get("final_task_success")) for row in clean
        ),
        "clean_success_rate": _rate(
            sum(bool(row.get("final_task_success")) for row in clean), len(clean)
        ),
        "scheduled_fault_count": len(scheduled_faults),
        "fault_count": len(faults),
        "pre_injection_failure_count": sum(
            bool(row.get("pre_injection_model_failure"))
            or (
                not bool(row.get("fault_applied"))
                and not bool(row.get("final_task_success"))
            )
            for row in scheduled_faults
        ),
        "a_exposure_count": a_count,
        "a_exposure_rate": _rate(a_count, len(faults)),
        "m_propagation_count": m_count,
        "m_propagation_rate": _rate(m_count, len(faults)),
        "recovery_count": recovery_count,
        "recovery_rate": _rate(recovery_count, len(faults)),
        "fault_final_failure_count": failure_count,
        "fault_final_failure_rate": _rate(failure_count, len(faults)),
        "final_success_count": success_count,
        "final_success_rate": _rate(success_count, len(row_list)),
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "total_tokens": sum(tokens),
        "mean_tokens": mean(tokens) if tokens else 0.0,
        "clean_matched_fault_count": len(matched),
        "clean_success_matched_fault_count": len(clean_success_matched),
        "clean_success_to_fault_failure_count": conditional_losses,
        "clean_conditional_loss_rate": _rate(
            conditional_losses, len(clean_success_matched)
        ),
        "transition_counts": dict(sorted(transitions.items())),
    }


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    values = list(values)
    return {
        "mean": mean(values) if values else 0.0,
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def summarize_repetition_stability(
    rows: Sequence[dict[str, Any]], *, expected_repeats: Sequence[int] = (1, 2, 3)
) -> dict[str, Any]:
    expected = tuple(sorted(int(item) for item in expected_repeats))
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("model_label") or row.get("model") or ""),
            str(row.get("task_id")),
            str(row.get("topology")),
            condition_name(row),
            injection_step(row),
        )
        grouped[key].append(row)

    cells = []
    incomplete = []
    for key, group in sorted(grouped.items()):
        repeats = sorted(repeat_index(row) for row in group)
        if tuple(repeats) != expected:
            incomplete.append({"cell_key": list(key), "repeat_indices": repeats})
            continue
        outcome_values = [bool(row.get("final_task_success")) for row in group]
        propagation_values = [derive_propagation_class(row) for row in group]
        m_values = [
            tuple(sorted(labels(row.get("observed_M_consequence")))) for row in group
        ]
        cells.append(
            {
                "dataset": key[0],
                "model_label": key[1],
                "task_id": key[2],
                "topology": key[3],
                "condition": key[4],
                "injection_step": key[5] or None,
                "repeat_indices": repeats,
                "outcome_stable": len(set(outcome_values)) == 1,
                "propagation_class_stable": len(set(propagation_values)) == 1,
                "m_consequence_stable": len(set(m_values)) == 1,
                "outcomes": outcome_values,
                "propagation_classes": propagation_values,
                "m_consequences": [list(value) for value in m_values],
            }
        )

    repeat_metrics = {
        repeat: summarize_metrics(
            [row for row in rows if repeat_index(row) == repeat]
        )
        for repeat in expected
    }
    variance_metrics = {}
    for metric in (
        "final_success_rate",
        "a_exposure_rate",
        "m_propagation_rate",
        "recovery_rate",
        "fault_final_failure_rate",
    ):
        variance_metrics[metric] = _sample_summary(
            [repeat_metrics[repeat][metric] for repeat in expected]
        )
    return {
        "expected_repeats": list(expected),
        "complete_cell_count": len(cells),
        "incomplete_cell_count": len(incomplete),
        "outcome_stable_count": sum(cell["outcome_stable"] for cell in cells),
        "propagation_class_stable_count": sum(
            cell["propagation_class_stable"] for cell in cells
        ),
        "m_consequence_stable_count": sum(
            cell["m_consequence_stable"] for cell in cells
        ),
        "outcome_stability_rate": _rate(
            sum(cell["outcome_stable"] for cell in cells), len(cells)
        ),
        "propagation_class_stability_rate": _rate(
            sum(cell["propagation_class_stable"] for cell in cells), len(cells)
        ),
        "m_consequence_stability_rate": _rate(
            sum(cell["m_consequence_stable"] for cell in cells), len(cells)
        ),
        "repeat_metrics": repeat_metrics,
        "repeat_variance": variance_metrics,
        "cells": cells,
        "incomplete_cells": incomplete,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute quantile of empty data")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_samples(
    records: Sequence[dict[str, Any]], cluster_key: str
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    clusters = sorted({str(record.get(cluster_key)) for record in records})
    by_cluster = {
        cluster: [
            record for record in records if str(record.get(cluster_key)) == cluster
        ]
        for cluster in clusters
    }
    return clusters, by_cluster


def cluster_bootstrap_rate(
    records: Sequence[dict[str, Any]],
    *,
    value_key: str,
    cluster_key: str,
    resamples: int = 1000,
    seed: int = 20260825,
) -> dict[str, Any]:
    records = list(records)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    clusters, by_cluster = _cluster_samples(records, cluster_key)
    if not records or not clusters:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "record_count": len(records),
            "cluster_count": len(clusters),
            "bootstrap_samples": 0,
        }
    estimate = sum(bool(record.get(value_key)) for record in records) / len(records)
    numerators = [
        sum(bool(record.get(value_key)) for record in by_cluster[cluster])
        for cluster in clusters
    ]
    denominators = [len(by_cluster[cluster]) for cluster in clusters]
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sampled = rng.choices(range(len(clusters)), k=len(clusters))
        numerator = sum(numerators[index] for index in sampled)
        denominator = sum(denominators[index] for index in sampled)
        values.append(numerator / denominator)
    return {
        "estimate": estimate,
        "lower_95": _quantile(values, 0.025),
        "upper_95": _quantile(values, 0.975),
        "record_count": len(records),
        "cluster_count": len(clusters),
        "bootstrap_samples": len(values),
    }


def cluster_bootstrap_paired_difference(
    records: Sequence[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    cluster_key: str,
    resamples: int = 1000,
    seed: int = 20260825,
) -> dict[str, Any]:
    records = list(records)
    clusters, by_cluster = _cluster_samples(records, cluster_key)
    if not records or not clusters:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "record_count": len(records),
            "cluster_count": len(clusters),
            "bootstrap_samples": 0,
        }

    effects = [
        mean(
            bool(record.get(left_key)) - bool(record.get(right_key))
            for record in by_cluster[cluster]
        )
        for cluster in clusters
    ]
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        values.append(mean(rng.choices(effects, k=len(effects))))
    return {
        "estimate": mean(effects),
        "lower_95": _quantile(values, 0.025),
        "upper_95": _quantile(values, 0.975),
        "record_count": len(records),
        "cluster_count": len(clusters),
        "bootstrap_samples": len(values),
    }


def cluster_sign_flip_test(
    records: Sequence[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    cluster_key: str,
    resamples: int = 10000,
    seed: int = 20260825,
    exact_cluster_limit: int = 12,
) -> dict[str, Any]:
    """Test a paired difference while treating clusters as inference units."""
    records = list(records)
    clusters, by_cluster = _cluster_samples(records, cluster_key)
    if not clusters:
        return {
            "cluster_count": 0,
            "task_weighted_risk_difference": None,
            "p_value": None,
            "exact": True,
            "permutations": 0,
        }

    effects = [
        mean(
            bool(record.get(left_key)) - bool(record.get(right_key))
            for record in by_cluster[cluster]
        )
        for cluster in clusters
    ]
    observed = abs(mean(effects))
    tolerance = 1e-12

    if len(effects) <= exact_cluster_limit:
        permutations = 2 ** len(effects)
        extreme = 0
        for mask in range(permutations):
            permuted = mean(
                effect if mask & (1 << index) else -effect
                for index, effect in enumerate(effects)
            )
            extreme += abs(permuted) >= observed - tolerance
        p_value = extreme / permutations
        exact = True
    else:
        if resamples <= 0:
            raise ValueError("resamples must be positive for Monte Carlo testing")
        rng = random.Random(seed)
        extreme = 0
        for _ in range(resamples):
            permuted = mean(
                effect if rng.random() < 0.5 else -effect for effect in effects
            )
            extreme += abs(permuted) >= observed - tolerance
        p_value = (extreme + 1) / (resamples + 1)
        permutations = resamples
        exact = False

    return {
        "cluster_count": len(effects),
        "task_weighted_risk_difference": mean(effects),
        "p_value": p_value,
        "exact": exact,
        "permutations": permutations,
    }


def mcnemar_exact_test(
    left: Sequence[bool], right: Sequence[bool]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired vectors must have equal length")
    b = sum(bool(a) and not bool(c) for a, c in zip(left, right))
    c = sum(not bool(a) and bool(c) for a, c in zip(left, right))
    discordant = b + c
    if discordant:
        tail = sum(
            math.comb(discordant, index) for index in range(min(b, c) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    n = len(left)
    return {
        "pair_count": n,
        "left_success_right_failure": b,
        "left_failure_right_success": c,
        "discordant_count": discordant,
        "p_value": p_value,
        "risk_difference": (
            sum(bool(value) for value in left)
            - sum(bool(value) for value in right)
        )
        / n
        if n
        else 0.0,
        "matched_odds_ratio": (b + 0.5) / (c + 0.5),
    }


def _regularized_gamma_q(shape: float, value: float) -> float:
    if value < 0.0 or shape <= 0.0:
        raise ValueError("invalid gamma parameters")
    if value == 0.0:
        return 1.0
    epsilon = 3e-14
    tiny = 1e-300
    max_iterations = 1000
    log_factor = -value + shape * math.log(value) - math.lgamma(shape)
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        ap = shape
        for _ in range(max_iterations):
            ap += 1.0
            term *= value / ap
            total += term
            if abs(term) < abs(total) * epsilon:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(log_factor)))

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0:
        d = -d
    h = d
    for index in range(1, max_iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return max(0.0, min(1.0, math.exp(log_factor) * h))


def cochran_q_test(matrix: Sequence[Sequence[bool]]) -> dict[str, Any]:
    matrix = [list(row) for row in matrix]
    if not matrix:
        raise ValueError("matrix must not be empty")
    group_count = len(matrix[0])
    if group_count < 2 or any(len(row) != group_count for row in matrix):
        raise ValueError("matrix must be rectangular with at least two groups")
    column_sums = [
        sum(bool(row[column]) for row in matrix) for column in range(group_count)
    ]
    row_sums = [sum(bool(value) for value in row) for row in matrix]
    total = sum(column_sums)
    denominator = group_count * total - sum(value * value for value in row_sums)
    if denominator == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (group_count - 1) * (
            group_count * sum(value * value for value in column_sums)
            - total * total
        ) / denominator
        p_value = _regularized_gamma_q((group_count - 1) / 2.0, statistic / 2.0)
    return {
        "pair_count": len(matrix),
        "group_count": group_count,
        "q_statistic": statistic,
        "degrees_of_freedom": group_count - 1,
        "p_value": p_value,
        "group_success_counts": column_sums,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    indexed = sorted(enumerate(float(value) for value in p_values), key=lambda x: x[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (original_index, value) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[original_index] = running
    return adjusted
