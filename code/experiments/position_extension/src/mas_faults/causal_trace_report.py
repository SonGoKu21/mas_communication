from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_EVENT_FIELDS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "fault_id",
    "carrier_id",
    "carrier_instance_id",
    "duplicate_index",
    "logical_message_id",
    "pair_id",
    "trace_variant",
    "timestamp",
    "event_layer",
    "component",
    "event_type",
    "event_status",
    "source",
    "target",
    "injection_operator_code",
    "injection_point_kind",
    "injected_fault_code",
    "injected_fault_layer",
    "expected_manifest_code",
    "expected_manifest_layer",
    "observed_effect",
    "propagation_label",
    "evidence",
}

ALLOWED_PROPAGATION_LABELS = {
    None,
    "pre_injection",
    "injected",
    "masked",
    "partial",
    "propagated",
}


SUMMARY_FIELDS = [
    "trace_id",
    "pair_id",
    "trace_variant",
    "task_id",
    "fault_id",
    "injection_operator_code",
    "injection_point_kind",
    "injected_fault_code",
    "injected_fault_layer",
    "expected_manifest_code",
    "expected_manifest_layer",
    "carrier_id",
    "carrier_instance_count",
    "carrier_instance_ids",
    "injection_event",
    "first_effect_layer",
    "first_effect_event",
    "first_observed_effect",
    "masking_layer",
    "masking_mechanism",
    "final_mas_consequence",
    "classification",
    "first_divergence_layer",
    "first_divergence_event",
    "event_count",
    "latency_ms",
    "evidence_file",
]


@dataclass
class TraceAnalysis:
    row: dict[str, Any]
    chain: dict[str, Any]


def validate_event_schema(event: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_EVENT_FIELDS - event.keys())
    if missing:
        raise ValueError(
            f"trace event {event.get('trace_id', '<unknown>')} missing fields: {missing}"
        )
    ambiguous = sorted({"fault_code", "fault_type", "layer"} & event.keys())
    if ambiguous:
        raise ValueError(
            f"trace event {event.get('trace_id', '<unknown>')} contains ambiguous fields: "
            f"{ambiguous}"
        )
    label = event.get("propagation_label")
    if label not in ALLOWED_PROPAGATION_LABELS:
        raise ValueError(f"unsupported propagation_label={label!r}")
    injected_code = event.get("injected_fault_code")
    injected_layer = event.get("injected_fault_layer")
    if injected_code and injected_layer != injected_code[0]:
        raise ValueError(
            f"injected fault {injected_code} conflicts with layer {injected_layer}"
        )
    operator_code = event.get("injection_operator_code")
    if operator_code and not event.get("injection_point_kind"):
        raise ValueError(
            f"injection operator {operator_code} lacks injection_point_kind"
        )


def validate_trace_semantics(events: list[dict[str, Any]]) -> None:
    event_types = {event["event_type"] for event in events}
    span_ids = {event["span_id"] for event in events}
    dangling_parents = sorted(
        {
            event["parent_span_id"]
            for event in events
            if event["parent_span_id"] is not None
            and event["parent_span_id"] not in span_ids
        }
    )
    if dangling_parents:
        raise ValueError(
            f"trace {events[0]['trace_id']} has dangling parent spans: "
            f"{dangling_parents}"
        )
    if "fault_applied" not in event_types:
        return
    required = {"fault_applied", "runtime_effect_observed", "final_consequence"}
    missing = sorted(required - event_types)
    if missing:
        raise ValueError(
            f"fault trace {events[0]['trace_id']} lacks downstream evidence: {missing}"
        )
    if not {"message_delivered", "message_dropped"} & event_types:
        raise ValueError(
            f"fault trace {events[0]['trace_id']} lacks a carrier delivery outcome"
        )
    worker_delivery = any(
        event["event_type"] == "message_delivered"
        and event.get("component") == "WorkerAgent"
        for event in events
    )
    if worker_delivery:
        agent_required = {"agent_message_received", "agent_visible_effect"}
        agent_missing = sorted(agent_required - event_types)
        if agent_missing:
            raise ValueError(
                f"fault trace {events[0]['trace_id']} lacks agent evidence: "
                f"{agent_missing}"
            )
    operator_codes = {
        event["injection_operator_code"]
        for event in events
        if event.get("injection_operator_code")
    }
    if "C8" in operator_codes:
        carrier_instances = {
            event["carrier_instance_id"]
            for event in events
            if event["event_type"] == "runtime_send_attempted"
            and event.get("carrier_instance_id")
        }
        if len(carrier_instances) != 2:
            raise ValueError(
                f"C8 trace {events[0]['trace_id']} expected two carrier instances, "
                f"observed {sorted(carrier_instances)}"
            )


def _timestamp(event: dict[str, Any]) -> float:
    return float(event.get("timestamp_unix", event.get("ts", 0.0)) or 0.0)


def _event_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    evidence = event.get("evidence") or {}
    return (
        event.get("event_layer", event.get("layer")),
        event.get("event_type", event.get("stage")),
        event.get("component", event.get("role")),
        event.get("observed_effect"),
        evidence.get("task_success"),
        evidence.get("final_answer_correct"),
    )


def _task_id(events: list[dict[str, Any]]) -> str:
    for event in events:
        evidence = event.get("evidence") or {}
        value = evidence.get("task_id") or event.get("task_id")
        if value:
            return str(value)
    return ""


def _classification(events: list[dict[str, Any]], fault_event: dict[str, Any] | None) -> str:
    if fault_event is None:
        return "clean"
    labels = {event.get("propagation_label") for event in events}
    m_events = [
        event
        for event in events
        if event.get("event_layer", event.get("layer")) == "M"
    ]
    harmful_m_event = any(
        event.get("event_type") == "final_consequence"
        and event.get("event_status") in {"missing", "incorrect", "failed"}
        or (event.get("evidence") or {}).get("task_success") is False
        or (event.get("evidence") or {}).get("final_answer_correct") is False
        for event in m_events
    )
    if harmful_m_event:
        return "propagated"
    if "masked" in labels:
        return "masked"
    effect_events = [
        event
        for event in events
        if _timestamp(event) >= _timestamp(fault_event)
        and event.get("observed_effect")
        and event.get("event_type") != "fault_applied"
    ]
    if effect_events or "partial" in labels:
        return "partial"
    return "masked"


def _first_divergence(
    clean_events: list[dict[str, Any]],
    fault_events: list[dict[str, Any]],
) -> tuple[str, str]:
    clean_signatures = [_event_signature(event) for event in clean_events]
    fault_signatures = [_event_signature(event) for event in fault_events]
    for index in range(max(len(clean_signatures), len(fault_signatures))):
        clean_signature = clean_signatures[index] if index < len(clean_signatures) else None
        fault_signature = fault_signatures[index] if index < len(fault_signatures) else None
        if clean_signature != fault_signature:
            event = fault_events[index] if index < len(fault_events) else clean_events[index]
            return (
                str(event.get("event_layer", event.get("layer")) or ""),
                str(event.get("event_type") or event.get("stage") or ""),
            )
    return "", ""


def analyze_trace(
    trace_id: str,
    events: list[dict[str, Any]],
    evidence_file: str,
    clean_events: list[dict[str, Any]] | None,
) -> TraceAnalysis:
    ordered = sorted(events, key=_timestamp)
    fault_event = next(
        (
            event
            for event in ordered
            if event.get("event_type", event.get("stage")) == "fault_applied"
        ),
        None,
    )
    fault_time = _timestamp(fault_event) if fault_event else float("inf")
    first_effect = next(
        (
            event
            for event in ordered
            if _timestamp(event) >= fault_time
            and event.get("observed_effect")
            and event is not fault_event
        ),
        None,
    )
    final_event = next(
        (
            event
            for event in reversed(ordered)
            if event.get("event_layer", event.get("layer")) == "M"
        ),
        None,
    )
    classification = _classification(ordered, fault_event)
    masking_layer = ""
    masking_mechanism = ""
    if classification == "masked":
        masking_layer = "T"
        masking_mechanism = "reliable transport recovery or suppression"
    injected_fault_code = (fault_event or {}).get("injected_fault_code")
    injected_fault_layer = (fault_event or {}).get("injected_fault_layer")
    injection_operator_code = (fault_event or {}).get("injection_operator_code")
    injection_point_kind = (fault_event or {}).get("injection_point_kind")
    expected_manifest_code = (fault_event or {}).get("expected_manifest_code")
    expected_manifest_layer = (fault_event or {}).get("expected_manifest_layer")
    fault_id = (fault_event or {}).get("fault_id")
    carrier_id = (fault_event or ordered[0]).get("carrier_id") if ordered else ""
    carrier_instance_ids = sorted(
        {
            str(event["carrier_instance_id"])
            for event in ordered
            if event.get("carrier_instance_id")
        }
    )
    first_divergence_layer = ""
    first_divergence_event = ""
    if clean_events is not None and fault_event is not None:
        first_divergence_layer, first_divergence_event = _first_divergence(clean_events, ordered)
    start = _timestamp(ordered[0]) if ordered else 0.0
    end = _timestamp(ordered[-1]) if ordered else start
    final_evidence = (final_event or {}).get("evidence") or {}
    consequence = final_evidence.get("mas_consequence") or (final_event or {}).get("observed_effect") or ""
    pair_id = str((ordered[0] if ordered else {}).get("pair_id") or f"pair-{_task_id(ordered)}")
    variant = str((ordered[0] if ordered else {}).get("trace_variant") or ("fault" if fault_event else "clean"))
    row = {
        "trace_id": trace_id,
        "pair_id": pair_id,
        "trace_variant": variant,
        "task_id": _task_id(ordered),
        "fault_id": fault_id or "",
        "injection_operator_code": injection_operator_code or "",
        "injection_point_kind": injection_point_kind or "",
        "injected_fault_code": injected_fault_code or "",
        "injected_fault_layer": injected_fault_layer or "",
        "expected_manifest_code": expected_manifest_code or "",
        "expected_manifest_layer": expected_manifest_layer or "",
        "carrier_id": carrier_id or "",
        "carrier_instance_count": len(carrier_instance_ids),
        "carrier_instance_ids": "|".join(carrier_instance_ids),
        "injection_event": (fault_event or {}).get("event_type", ""),
        "first_effect_layer": (first_effect or {}).get(
            "event_layer",
            (first_effect or {}).get("layer", ""),
        ),
        "first_effect_event": (first_effect or {}).get("event_type", ""),
        "first_observed_effect": (first_effect or {}).get("observed_effect", ""),
        "masking_layer": masking_layer,
        "masking_mechanism": masking_mechanism,
        "final_mas_consequence": consequence,
        "classification": classification,
        "first_divergence_layer": first_divergence_layer,
        "first_divergence_event": first_divergence_event,
        "event_count": len(ordered),
        "latency_ms": round((end - start) * 1000.0, 3),
        "evidence_file": evidence_file,
    }
    chain_events = [
        {
            "span_id": event.get("span_id"),
            "parent_span_id": event.get("parent_span_id"),
            "fault_id": event.get("fault_id"),
            "carrier_id": event.get("carrier_id"),
            "carrier_instance_id": event.get("carrier_instance_id"),
            "duplicate_index": event.get("duplicate_index"),
            "timestamp": event.get("timestamp"),
            "event_layer": event.get("event_layer", event.get("layer")),
            "component": event.get("component"),
            "event_type": event.get("event_type", event.get("stage")),
            "source": event.get("source"),
            "target": event.get("target"),
            "observed_effect": event.get("observed_effect"),
            "propagation_label": event.get("propagation_label"),
            "event_status": event.get("event_status"),
            "injection_operator_code": event.get("injection_operator_code"),
            "injection_point_kind": event.get("injection_point_kind"),
            "injected_fault_code": event.get("injected_fault_code"),
            "injected_fault_layer": event.get("injected_fault_layer"),
            "expected_manifest_code": event.get("expected_manifest_code"),
            "expected_manifest_layer": event.get("expected_manifest_layer"),
            "evidence": event.get("evidence", {}),
        }
        for event in ordered
    ]
    return TraceAnalysis(row=row, chain={"trace_id": trace_id, "summary": row, "events": chain_events})


def read_events(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.jsonl"))
    events: list[dict[str, Any]] = []
    evidence_files: dict[str, str] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                validate_event_schema(event)
                trace_id = str(event.get("trace_id") or "trace-unknown")
                events.append(event)
                evidence_files.setdefault(trace_id, str(path))
    return events, evidence_files


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(input_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events, evidence_files = read_events(input_path)
    normalized_events = sorted(events, key=lambda item: (str(item.get("trace_id")), _timestamp(item)))
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in normalized_events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in normalized_events:
        grouped[str(event.get("trace_id") or "trace-unknown")].append(event)

    clean_by_pair: dict[str, list[dict[str, Any]]] = {}
    for trace_events in grouped.values():
        ordered = sorted(trace_events, key=_timestamp)
        first = ordered[0] if ordered else {}
        if first.get("trace_variant") == "clean" or not any(
            event.get("event_type", event.get("stage")) == "fault_applied"
            and event.get("injected_fault_code")
            for event in ordered
        ):
            pair_id = str(first.get("pair_id") or f"pair-{_task_id(ordered)}")
            clean_by_pair[pair_id] = ordered

    analyses: list[TraceAnalysis] = []
    for trace_id, trace_events in sorted(grouped.items()):
        ordered = sorted(trace_events, key=_timestamp)
        validate_trace_semantics(ordered)
        pair_id = str((ordered[0] if ordered else {}).get("pair_id") or f"pair-{_task_id(ordered)}")
        clean_events = clean_by_pair.get(pair_id)
        analyses.append(
            analyze_trace(
                trace_id,
                ordered,
                evidence_files.get(trace_id, ""),
                None if clean_events is ordered else clean_events,
            )
        )

    rows = [analysis.row for analysis in analyses]
    write_csv(output_dir / "trace_summary.csv", rows, SUMMARY_FIELDS)
    (output_dir / "propagation_chains.json").write_text(
        json.dumps([analysis.chain for analysis in analyses], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    chain_lines = [
        "# Forward Fault Propagation Chains",
        "",
        "Each chain preserves TraceID, SpanID, FaultID, and CarrierID. "
        "Injected fault source, expected manifestation, and event layer are recorded separately.",
        "",
    ]
    for analysis in analyses:
        row = analysis.row
        chain_lines.extend(
            [
                f"## {row['trace_id']} — {row['classification']}",
                "",
                f"- Pair: `{row['pair_id']}` / `{row['trace_variant']}`",
                f"- Injection operator: `{row['injection_operator_code'] or 'none'}` at `{row['injection_point_kind'] or 'n/a'}`",
                f"- Injected taxonomy fault: `{row['injected_fault_code'] or 'none'}` at `{row['injected_fault_layer'] or 'n/a'}`",
                f"- Expected manifestation: `{row['expected_manifest_code'] or 'none'}` at `{row['expected_manifest_layer'] or 'n/a'}`",
                f"- FaultID: `{row['fault_id'] or 'n/a'}`",
                f"- Carrier: `{row['carrier_id']}`",
                f"- Carrier instances: `{row['carrier_instance_ids'] or 'none'}`",
                f"- First divergence: `{row['first_divergence_layer'] or 'n/a'}:{row['first_divergence_event'] or 'n/a'}`",
                f"- First observed effect: `{row['first_effect_layer'] or 'n/a'}:{row['first_effect_event'] or 'n/a'}` — {row['first_observed_effect'] or 'none'}",
                f"- Final MAS consequence: {row['final_mas_consequence'] or 'none'}",
                "",
                "| Time | Event layer | Carrier instance | Component | Event | Status | Effect | Propagation |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for event in analysis.chain["events"]:
            chain_lines.append(
                "| {timestamp} | {layer} | {carrier_instance} | {component} | {event_type} | {status} | {effect} | {label} |".format(
                    timestamp=event.get("timestamp") or "",
                    layer=event.get("event_layer") or "",
                    carrier_instance=event.get("carrier_instance_id") or "",
                    component=event.get("component") or "",
                    event_type=event.get("event_type") or "",
                    status=event.get("event_status") or "",
                    effect=(event.get("observed_effect") or "").replace("|", "/"),
                    label=event.get("propagation_label") or "",
                )
            )
        chain_lines.append("")
    (output_dir / "propagation_chains.md").write_text("\n".join(chain_lines), encoding="utf-8")

    classification_counts = Counter(str(row["classification"]) for row in rows)
    path_counts = Counter(
        (
            str(row["expected_manifest_code"] or "none"),
            str(row["first_effect_layer"] or "none"),
            str(row["final_mas_consequence"] or "none"),
            str(row["classification"]),
        )
        for row in rows
    )
    statistics_rows = [
        {
            "expected_manifest_code": fault_type,
            "first_effect_layer": first_layer,
            "final_mas_consequence": consequence,
            "classification": classification,
            "count": count,
            "share": round(count / len(rows), 4) if rows else 0.0,
        }
        for (fault_type, first_layer, consequence, classification), count in sorted(path_counts.items())
    ]
    write_csv(
        output_dir / "propagation_statistics.csv",
        statistics_rows,
        ["expected_manifest_code", "first_effect_layer", "final_mas_consequence", "classification", "count", "share"],
    )
    stats_lines = [
        "# Forward Propagation Statistics",
        "",
        f"- Total traces: {len(rows)}",
        f"- Clean: {classification_counts.get('clean', 0)}",
        f"- Masked: {classification_counts.get('masked', 0)}",
        f"- Partial: {classification_counts.get('partial', 0)}",
        f"- Propagated: {classification_counts.get('propagated', 0)}",
        "",
        "## Dominant Paths",
        "",
        "| Fault | First effect layer | Final MAS consequence | Classification | Count | Share |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in statistics_rows:
        stats_lines.append(
            f"| {row['expected_manifest_code']} | {row['first_effect_layer']} | "
            f"{row['final_mas_consequence']} | {row['classification']} | "
            f"{row['count']} | {row['share']:.4f} |"
        )
    stats_lines.extend(
        [
            "",
            "## Causal Boundary",
            "",
            "This report is a forward causal evidence chain for controlled fault experiments, not a production-grade "
            "backward observability system. Clean/fault pairing localizes the first recorded divergence, while deterministic "
            "tasks reduce—but do not by themselves eliminate—unobserved confounding.",
        ]
    )
    (output_dir / "propagation_statistics.md").write_text("\n".join(stats_lines), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward causal fault propagation traces.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = build_report(args.input, args.output)
    print(json.dumps({"trace_count": len(rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
