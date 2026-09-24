"""Admission report for repeated WebArena Shopping clean runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_clean_runs(rows: list[dict[str, Any]], *, expected_repeats: int) -> dict[str, Any]:
    if expected_repeats < 1:
        raise ValueError("expected_repeats must be at least 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("condition") == "clean":
            grouped[str(row["task_id"])].append(row)

    tasks: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        trials = grouped[task_id]
        passed = sum(bool(row.get("final_task_success")) for row in trials)
        errors = sum(bool(row.get("error")) for row in trials)
        stable = len(trials) == expected_repeats and passed == expected_repeats and errors == 0
        tasks.append({
            "task_id": task_id,
            "clean_trials": len(trials),
            "clean_successes": passed,
            "errors": errors,
            "stable_for_fault_matrix": stable,
            "admission_reason": "all_clean_repeats_passed" if stable else "clean_baseline_not_stable",
        })
    stable_task_ids = [task["task_id"] for task in tasks if task["stable_for_fault_matrix"]]
    return {
        "expected_clean_repeats": expected_repeats,
        "candidate_tasks": len(tasks),
        "stable_tasks": len(stable_task_ids),
        "stable_task_ids": stable_task_ids,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated WebArena Shopping clean runs.")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--expected-repeats", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.runs).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_clean_runs(rows, expected_repeats=args.expected_repeats)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "clean_baseline_admission.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "stable_task_ids.json").write_text(json.dumps(summary["stable_task_ids"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_tasks": summary["candidate_tasks"], "stable_tasks": summary["stable_tasks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
