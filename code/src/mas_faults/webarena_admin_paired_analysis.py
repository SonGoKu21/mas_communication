"""Baseline-paired outcome analysis for WebArena Admin matrices."""

from __future__ import annotations

from typing import Any


def _pair_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("task_id")),
        str(row.get("topology")),
        int(row.get("repeat_index", 1)),
    )


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = {
        "clean_success_fault_failure": 0,
        "clean_success_fault_success": 0,
        "clean_failure_fault_failure": 0,
        "clean_failure_fault_success": 0,
    }
    for record in records:
        transitions[record["paired_outcome_transition"]] += 1
    eligible = (
        transitions["clean_success_fault_failure"]
        + transitions["clean_success_fault_success"]
    )
    loss_rate = (
        round(transitions["clean_success_fault_failure"] / eligible, 4)
        if eligible
        else None
    )
    return {
        "fault_runs": len(records),
        "clean_eligible_count": eligible,
        **transitions,
        "conditional_outcome_loss_rate": loss_rate,
    }


def _group_metrics(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(record.get(key, "unknown")) for record in records})
    return {
        value: _metrics(
            [record for record in records if str(record.get(key, "unknown")) == value]
        )
        for value in values
    }


def summarize_paired_outcomes(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "clean":
            continue
        key = _pair_key(row)
        if key in clean_by_key:
            raise ValueError(f"duplicate clean baseline: {key}")
        clean_by_key[key] = row

    paired_records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("condition") == "clean":
            continue
        key = _pair_key(row)
        if key not in clean_by_key:
            raise ValueError(f"missing clean baseline: {key}")
        clean_success = bool(clean_by_key[key].get("final_task_success"))
        fault_success = bool(row.get("final_task_success"))
        transition = (
            f"clean_{'success' if clean_success else 'failure'}_"
            f"fault_{'success' if fault_success else 'failure'}"
        )
        paired_records.append({**row, "paired_outcome_transition": transition})

    return {
        "overall": _metrics(paired_records),
        "by_topology": _group_metrics(paired_records, "topology"),
        "by_condition": _group_metrics(paired_records, "condition"),
        "by_fault_family": _group_metrics(paired_records, "fault_family"),
        "by_task_stratum": _group_metrics(paired_records, "task_stratum"),
    }
