from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_case_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_condition: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for row in rows:
        topology = str(row["topology"])
        label = str(row.get("architecture_manifestation", "none"))
        fault = str(row.get("fault_type", "none"))
        step = str(row.get("injection_step", "none"))
        counts[topology][label] += 1
        by_condition[topology][fault][step][label] += 1
    return {
        "manifestation_counts": {topology: dict(sorted(labels.items())) for topology, labels in sorted(counts.items())},
        "manifestation_counts_by_fault_and_step": {
            topology: {
                fault: {step: dict(sorted(labels.items())) for step, labels in sorted(steps.items())}
                for fault, steps in sorted(faults.items())
            }
            for topology, faults in sorted(by_condition.items())
        },
    }


def write_representative_cases(rows: list[dict[str, Any]], output: Path) -> None:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("architecture_manifestation", "none"))
        if label != "none" and label not in chosen:
            chosen[label] = row
    lines = ["# RQ2 Representative Framework Consequence Cases", "", "These are trace-backed structure manifestations. They supplement, rather than replace, the frozen generic M-layer evaluator.", ""]
    if not chosen:
        lines.extend(["No non-`none` manifestation was observed in this matrix.", ""])
    for label, row in sorted(chosen.items()):
        lines.extend([
            f"## {label}",
            "",
            f"- Run: `{row['run_id']}`",
            f"- Topology: `{row['topology']}`; fault: `{row['fault_type']}`; injection step: `{row['injection_step']}`",
            f"- Generic M consequence: `{', '.join(row.get('observed_M_consequence', ['none']))}`; final success: `{row.get('final_task_success')}`",
            f"- Message path: `{' -> '.join(f'{source}->{target}' for source, target in row.get('message_path', []))}`",
            f"- Manifestation evidence: `{json.dumps(row.get('architecture_manifestation_evidence', {}), ensure_ascii=False, sort_keys=True)}`",
            "",
        ])
    (output / "rq2_representative_cases.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-json", required=True, help="JSON array of RQ2 run records")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    summary = build_case_summary(rows)
    (output / "rq2_consequence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_representative_cases(rows, output)


if __name__ == "__main__":
    main()
