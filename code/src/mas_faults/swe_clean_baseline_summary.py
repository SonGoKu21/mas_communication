from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_clean_runs(rows: list[dict[str, Any]], expected_repeats: int) -> dict[str, Any]:
    if expected_repeats < 1:
        raise ValueError("expected_repeats must be at least 1")
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("condition") == "clean":
            by_instance[str(row["instance_id"])].append(row)

    instances: list[dict[str, Any]] = []
    for instance_id in sorted(by_instance):
        trials = sorted(by_instance[instance_id], key=lambda item: int(item.get("repeat_index", 1)))
        passed = sum(bool(item.get("official_test_passed")) for item in trials)
        invalid = sum(item.get("execution_status") == "agent_patch_invalid" for item in trials)
        environment_errors = sum(
            bool(item.get("official_runner_error")) and item.get("official_runner_error") != "agent_patch_invalid"
            for item in trials
        )
        stable = len(trials) == expected_repeats and passed == expected_repeats and not invalid and not environment_errors
        instances.append({
            "instance_id": instance_id,
            "clean_trials": len(trials),
            "clean_passes": passed,
            "agent_patch_invalid": invalid,
            "environment_errors": environment_errors,
            "stable_for_fault_matrix": stable,
            "admission_reason": "all_clean_repeats_passed" if stable else "clean_baseline_not_stable",
        })

    stable_ids = [item["instance_id"] for item in instances if item["stable_for_fault_matrix"]]
    return {
        "expected_clean_repeats": expected_repeats,
        "candidate_instances": len(instances),
        "stable_instances": len(stable_ids),
        "stable_instance_ids": stable_ids,
        "instances": instances,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# SWE-bench Clean 基线准入汇总",
        "",
        f"- 每任务重复次数：{summary['expected_clean_repeats']}",
        f"- 候选任务数：{summary['candidate_instances']}",
        f"- 可进入故障矩阵的稳定任务数：{summary['stable_instances']}",
        "",
        "| 任务 | Clean 通过 | 补丁无效 | 环境错误 | 准入 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in summary["instances"]:
        lines.append(
            f"| {item['instance_id']} | {item['clean_passes']}/{item['clean_trials']} | "
            f"{item['agent_patch_invalid']} | {item['environment_errors']} | {item['admission_reason']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated SWE-bench clean baseline runs.")
    parser.add_argument("--runs", required=True, help="Canonical swe_bench_runs.jsonl path.")
    parser.add_argument("--expected-repeats", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.runs).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_clean_runs(rows, args.expected_repeats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clean_baseline_admission.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "clean_baseline_admission.md").write_text(markdown_report(summary), encoding="utf-8")
    (output_dir / "stable_instance_ids.json").write_text(json.dumps(summary["stable_instance_ids"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_instances": summary["candidate_instances"], "stable_instances": summary["stable_instances"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
