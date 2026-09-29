"""Post-process canonical benchmark JSONL runs into research-facing summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _observed(value: Any) -> bool:
    if isinstance(value, str):
        return value != "none"
    return any(item != "none" for item in (value or []))


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    transitions = Counter(str(row.get("propagation_class", "unknown")) for row in rows)
    a_count = sum(_observed(row.get("observed_A_symptom")) for row in rows)
    m_count = sum(_observed(row.get("observed_M_consequence")) for row in rows)
    recovery_count = sum(bool(row.get("recovery_detected")) for row in rows)
    success_count = sum(bool(row.get("final_task_success")) for row in rows)
    return {
        "runs": count,
        "a_exposure_count": a_count,
        "a_exposure_rate": _rate(a_count, count),
        "m_consequence_count": m_count,
        "m_propagation_count": m_count,
        "m_propagation_rate": _rate(m_count, count),
        "recovery_count": recovery_count,
        "recovery_rate": _rate(recovery_count, count),
        "final_success_count": success_count,
        "final_success_rate": _rate(success_count, count),
        "final_failure_count": count - success_count,
        "final_failure_rate": _rate(count - success_count, count),
        "mean_latency_ms": round(sum(float(row.get("latency_ms") or 0) for row in rows) / count, 3) if count else 0.0,
        "mean_total_tokens": round(sum(float(row.get("total_tokens") or 0) for row in rows) / count, 3) if count else 0.0,
        "transition_counts": dict(sorted(transitions.items())),
    }


def summarize_runs(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_condition_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition_rows.setdefault(str(row.get("condition", "unknown")), []).append(row)
    by_condition = {condition: _condition_summary(group) for condition, group in sorted(by_condition_rows.items())}
    overall = _condition_summary(rows)
    overall["runs"] = len(rows)
    overall["fault_runs"] = sum(bool(row.get("fault_applied")) for row in rows)
    overall["transition_counts"] = dict(sorted(Counter(str(row.get("propagation_class", "unknown")) for row in rows).items()))
    return overall, by_condition


def _markdown(overall: dict[str, Any], by_condition: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# 通信故障实验汇总",
        "",
        f"- 总运行数：{overall['runs']}",
        f"- 最终任务成功：{overall['final_success_count']}/{overall['runs']} ({overall['final_success_rate']:.1%})",
        f"- A 层暴露：{overall['a_exposure_count']}/{overall['runs']} ({overall['a_exposure_rate']:.1%})",
        f"- M 层传播：{overall['m_consequence_count']}/{overall['runs']} ({overall['m_propagation_rate']:.1%})",
        f"- 恢复：{overall['recovery_count']}/{overall['runs']} ({overall['recovery_rate']:.1%})",
        "",
        "| 条件 | Runs | A 暴露 | M 传播 | 恢复 | 成功 | 失败 | 平均延迟 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, item in by_condition.items():
        lines.append(
            f"| {condition} | {item['runs']} | {item['a_exposure_count']} | {item['m_consequence_count']} | "
            f"{item['recovery_count']} | {item['final_success_count']} | {item['final_failure_count']} | {item['mean_latency_ms']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create aggregate summaries from canonical benchmark JSONL runs.")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="benchmark")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.runs).read_text(encoding="utf-8").splitlines() if line.strip()]
    overall, by_condition = summarize_runs(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.prefix}_summary.json").write_text(
        json.dumps({"overall": overall, "by_condition": by_condition}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["condition", "runs", "a_exposure_count", "m_consequence_count", "recovery_count", "final_success_count", "final_failure_count", "mean_latency_ms", "mean_total_tokens"]
    with (output / f"{args.prefix}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, item in by_condition.items():
            writer.writerow({"condition": condition, **{field: item[field] for field in fields if field != "condition"}})
    (output / f"{args.prefix}_summary.md").write_text(_markdown(overall, by_condition), encoding="utf-8")


if __name__ == "__main__":
    main()
