from __future__ import annotations

import asyncio
import copy
import csv
import json
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TRACE_SCHEMA_VERSION = "unified-mas-trace-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BridgeCondition:
    name: str
    source_fault_ids: tuple[str, ...]
    source_layers: tuple[str, ...]
    transport_mediation: tuple[str, ...]
    mechanism: str
    injection_step: int
    a_operator: str
    severity: str = "controlled"


@dataclass(frozen=True)
class BridgeRunSpec:
    task_id: str
    topology: str
    condition: str
    repeat_index: int
    injection_step: int

    @property
    def job_key(self) -> str:
        return "::".join(
            (
                self.task_id,
                self.topology,
                self.condition,
                str(self.injection_step),
                str(self.repeat_index),
            )
        )


@dataclass(frozen=True)
class BridgeExposure:
    condition: str
    source_fault_ids: tuple[str, ...]
    source_layers: tuple[str, ...]
    transport_mediation: tuple[str, ...]
    mechanism: str
    fault_triggered: bool
    masked_before_a: bool
    observed_a_faults: tuple[str, ...]
    a_operator: str
    injection_step: int
    severity: str
    started_at: str
    completed_at: str
    evidence: dict[str, Any]


BRIDGE_CONDITIONS: tuple[BridgeCondition, ...] = (
    BridgeCondition(
        name="clean",
        source_fault_ids=(),
        source_layers=(),
        transport_mediation=("T1",),
        mechanism="reliable transport delivers the original logical message",
        injection_step=4,
        a_operator="none",
        severity="none",
    ),
    BridgeCondition(
        name="physical_bit_flip_masked",
        source_fault_ids=("P1", "P2"),
        source_layers=("P", "D", "T"),
        transport_mediation=("D2", "D3", "T1", "T5"),
        mechanism="one payload bit is corrupted; checksum detection drops the frame and reliable transport retransmits the original bytes",
        injection_step=4,
        a_operator="none",
    ),
    BridgeCondition(
        name="deadline_induced_omission",
        source_fault_ids=("N2", "N3", "T5"),
        source_layers=("N", "T"),
        transport_mediation=("T5",),
        mechanism="network delay exceeds the application deadline; the late logical message is discarded",
        injection_step=4,
        a_operator="timeout",
    ),
    BridgeCondition(
        name="stream_interruption_truncation",
        source_fault_ids=("T4", "T11"),
        source_layers=("T",),
        transport_mediation=("T4", "T11"),
        mechanism="the JSON stream starts normally and is interrupted after its first fragment",
        injection_step=4,
        a_operator="truncation",
    ),
    BridgeCondition(
        name="retry_induced_duplication",
        source_fault_ids=("T5", "T12"),
        source_layers=("T",),
        transport_mediation=("T5", "T12"),
        mechanism="the receiver commits a side effect, its acknowledgement times out, and a non-idempotent retry executes the message again",
        injection_step=2,
        a_operator="duplicate_request",
    ),
    BridgeCondition(
        name="async_logical_reordering",
        source_fault_ids=("N2", "T6"),
        source_layers=("N", "T"),
        transport_mediation=("T6",),
        mechanism="two logical states are sent concurrently and runtime scheduling delivers the newer state first",
        injection_step=3,
        a_operator="reordering",
    ),
)

CONDITIONS_BY_NAME = {condition.name: condition for condition in BRIDGE_CONDITIONS}


def get_bridge_condition(name: str) -> BridgeCondition:
    try:
        return CONDITIONS_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(CONDITIONS_BY_NAME)
        raise ValueError(f"unsupported bridge condition={name!r}; supported: {supported}") from exc


def build_bridge_specs(
    *,
    tasks: Sequence[dict[str, Any]],
    topologies: Sequence[str],
    repetitions: int,
    conditions: Sequence[str] | None = None,
) -> list[BridgeRunSpec]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    selected = tuple(conditions or CONDITIONS_BY_NAME)
    for condition_name in selected:
        get_bridge_condition(condition_name)
    specs = [
        BridgeRunSpec(
            task_id=str(task["task_id"]),
            topology=topology,
            condition=condition.name,
            repeat_index=repeat_index,
            injection_step=condition.injection_step,
        )
        for topology in topologies
        for task in tasks
        for condition_name in selected
        for condition in (get_bridge_condition(condition_name),)
        for repeat_index in range(1, repetitions + 1)
    ]
    keys = [spec.job_key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("bridge matrix contains duplicate job keys")
    return specs


async def probe_lower_layer(condition_name: str, payload: Any, *, seed: int) -> BridgeExposure:
    condition = get_bridge_condition(condition_name)
    started_at = now()
    evidence: dict[str, Any]
    observed: tuple[str, ...]
    masked = False
    triggered = condition.name != "clean"
    operator = condition.a_operator

    if condition.name == "clean":
        evidence = {"message_delivered": True, "payload_preserved": True}
        observed = ("none",)
    elif condition.name == "physical_bit_flip_masked":
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        bit_index = seed % max(1, len(encoded) * 8)
        byte_index, bit_offset = divmod(bit_index, 8)
        corrupted = bytearray(encoded)
        corrupted[byte_index] ^= 1 << bit_offset
        original_checksum = zlib.crc32(encoded)
        corrupted_checksum = zlib.crc32(corrupted)
        checksum_mismatch = original_checksum != corrupted_checksum
        retransmitted = checksum_mismatch
        evidence = {
            "bit_index": bit_index,
            "original_checksum": original_checksum,
            "corrupted_checksum": corrupted_checksum,
            "checksum_mismatch": checksum_mismatch,
            "corrupted_frame_delivered_to_application": False,
            "retransmission_delivered_original": retransmitted,
            "payload_preserved": retransmitted,
        }
        observed = ("none",) if retransmitted else ("A6",)
        masked = retransmitted
        operator = "none" if masked else "message_corruption"
    elif condition.name == "deadline_induced_omission":
        timeout_seconds = 0.002
        response_delay_seconds = 0.010
        timed_out = False
        try:
            await asyncio.wait_for(
                _delayed_value(payload, response_delay_seconds),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
        late_message_discarded = timed_out
        evidence = {
            "timeout_ms": timeout_seconds * 1000,
            "response_delay_ms": response_delay_seconds * 1000,
            "timeout_triggered": timed_out,
            "late_message_discarded": late_message_discarded,
            "message_delivered": not late_message_discarded,
        }
        observed = ("A2", "A5") if timed_out and late_message_discarded else ("none",)
        masked = observed == ("none",)
        operator = condition.a_operator if not masked else "none"
    elif condition.name == "stream_interruption_truncation":
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        fragment = encoded[: max(1, len(encoded) // 2)]
        parse_failed = False
        try:
            json.loads(fragment)
        except json.JSONDecodeError:
            parse_failed = True
        evidence = {
            "original_bytes": len(encoded.encode("utf-8")),
            "delivered_bytes": len(fragment.encode("utf-8")),
            "stream_interrupted": len(fragment) < len(encoded),
            "parse_failed": parse_failed,
            "sample_fragment": fragment,
        }
        observed = ("A7", "A8") if evidence["stream_interrupted"] and parse_failed else ("none",)
        masked = observed == ("none",)
        operator = condition.a_operator if not masked else "none"
    elif condition.name == "retry_induced_duplication":
        state = {"execution_count": 0}

        async def execute_then_ack() -> None:
            state["execution_count"] += 1
            await asyncio.sleep(0.010)

        timeout_triggered = False
        retry_count = 0
        try:
            await asyncio.wait_for(execute_then_ack(), timeout=0.002)
        except asyncio.TimeoutError:
            timeout_triggered = True
            retry_count = 1
            state["execution_count"] += 1
        duplicate_execution = state["execution_count"] > 1
        evidence = {
            "timeout_triggered": timeout_triggered,
            "retry_count": retry_count,
            "idempotency_key_used": False,
            "execution_count": state["execution_count"],
            "duplicate_execution": duplicate_execution,
        }
        observed = ("A9",) if duplicate_execution else ("none",)
        masked = observed == ("none",)
        operator = condition.a_operator if not masked else "none"
    elif condition.name == "async_logical_reordering":
        sent_order = ["state-v1", "state-v2"]
        arrivals = await asyncio.gather(
            _arrive(sent_order[0], 0.010),
            _arrive(sent_order[1], 0.001),
        )
        arrival_order = [item[0] for item in sorted(arrivals, key=lambda item: item[1])]
        reordered = arrival_order != sent_order
        evidence = {
            "sent_order": sent_order,
            "arrival_order": arrival_order,
            "logical_reordering_observed": reordered,
            "concurrent_send_enabled": True,
        }
        observed = ("A10",) if reordered else ("none",)
        masked = observed == ("none",)
        operator = condition.a_operator if not masked else "none"
    else:
        raise AssertionError(f"unimplemented bridge condition: {condition.name}")

    return BridgeExposure(
        condition=condition.name,
        source_fault_ids=condition.source_fault_ids,
        source_layers=condition.source_layers,
        transport_mediation=condition.transport_mediation,
        mechanism=condition.mechanism,
        fault_triggered=triggered,
        masked_before_a=masked,
        observed_a_faults=observed,
        a_operator=operator,
        injection_step=condition.injection_step,
        severity=condition.severity,
        started_at=started_at,
        completed_at=now(),
        evidence=evidence,
    )


async def _delayed_value(value: Any, delay_seconds: float) -> Any:
    await asyncio.sleep(delay_seconds)
    return value


async def _arrive(message_id: str, delay_seconds: float) -> tuple[str, float]:
    await asyncio.sleep(delay_seconds)
    return message_id, time.perf_counter()


def enrich_run_record(
    base_row: dict[str, Any],
    exposure: BridgeExposure,
    spec: BridgeRunSpec,
    *,
    scenario_name: str = "bottom_up_webarena_shopping_bridge",
) -> dict[str, Any]:
    row = copy.deepcopy(base_row)
    bridge_id = f"bridge-{row['run_id']}"
    base_a_fault = str(row.get("fault_type", exposure.a_operator))
    base_a_applied = bool(row.get("fault_applied", False))
    a_symptoms = _merge_labels(exposure.observed_a_faults, row.get("observed_A_symptom", []))
    m_consequences = _normalize_labels(row.get("observed_M_consequence", []))
    final_success = bool(row.get("final_task_success", False))
    recovery = bool(row.get("recovery_detected", False))
    propagation_class = classify_bridge_propagation(
        condition=spec.condition,
        masked_before_a=exposure.masked_before_a,
        a_symptoms=a_symptoms,
        m_consequences=m_consequences,
        recovery_detected=recovery,
        final_task_success=final_success,
    )

    lower_layer = exposure.source_layers[0] if exposure.source_layers else "T"
    source_event = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": row["run_id"],
        "trace_id": row["trace_id"],
        "bridge_id": bridge_id,
        "event_id": f"{bridge_id}:lower-source",
        "timestamp": exposure.started_at,
        "event_layer": lower_layer,
        "event_type": "lower_layer_fault_injected" if exposure.fault_triggered else "clean_transport_started",
        "source_agent": "communication_substrate",
        "target_agent": "transport_stack",
        "condition": exposure.condition,
        "fault_applied": exposure.fault_triggered,
        "source_fault_ids": list(exposure.source_fault_ids),
        "observed_A_symptom": ["none"],
        "observed_M_consequence": ["none"],
        "propagation_label": "injected" if exposure.fault_triggered else "clean",
        "evidence": exposure.evidence,
    }
    mediation_event = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": row["run_id"],
        "trace_id": row["trace_id"],
        "bridge_id": bridge_id,
        "event_id": f"{bridge_id}:transport-mediation",
        "timestamp": exposure.completed_at,
        "event_layer": "T",
        "event_type": "transport_mediation_observed",
        "source_agent": "transport_stack",
        "target_agent": "application_communication_boundary",
        "condition": exposure.condition,
        "fault_applied": exposure.fault_triggered,
        "transport_mediation": list(exposure.transport_mediation),
        "masked_before_a": exposure.masked_before_a,
        "observed_A_symptom": list(exposure.observed_a_faults),
        "observed_M_consequence": ["none"],
        "propagation_label": "masked" if exposure.masked_before_a else ("exposed_at_A" if exposure.fault_triggered else "clean"),
        "evidence": exposure.evidence,
    }
    application_events = []
    for event in row.get("events", []):
        item = copy.deepcopy(event)
        item["bridge_id"] = bridge_id
        item["event_layer"] = "A"
        item["lower_layer_condition"] = exposure.condition
        application_events.append(item)
    final_event = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": row["run_id"],
        "trace_id": row["trace_id"],
        "bridge_id": bridge_id,
        "event_id": f"{bridge_id}:final",
        "timestamp": now(),
        "event_layer": "M",
        "event_type": "final_outcome",
        "source_agent": "task_evaluator",
        "target_agent": "MAS_output",
        "condition": exposure.condition,
        "observed_A_symptom": a_symptoms,
        "observed_M_consequence": m_consequences,
        "recovery_detected": recovery,
        "recovery_type": row.get("recovery_type", "none"),
        "final_task_success": final_success,
        "propagation_class": propagation_class,
        "propagation_label": "success" if final_success else "final_failure",
    }
    events = [source_event, mediation_event, *application_events, final_event]
    propagation_path = [*exposure.source_fault_ids]
    if exposure.masked_before_a:
        propagation_path.append("masked_before_A")
    else:
        propagation_path.extend(label for label in a_symptoms if label != "none")
        propagation_path.extend(label for label in m_consequences if label != "none")
    propagation_path.append("final_success" if final_success else "final_failure")

    row.update(
        {
            "bridge_id": bridge_id,
            "scenario": scenario_name,
            "condition": exposure.condition,
            "job_key": spec.job_key,
            "repeat_index": spec.repeat_index,
            "seed_or_run_index": spec.repeat_index,
            "injection_step": spec.injection_step,
            "fault_id": "none" if not exposure.fault_triggered else f"lower-{bridge_id}",
            "fault_type": exposure.condition,
            "fault_severity": exposure.severity,
            "fault_applied": exposure.fault_triggered,
            "a_fault_type": base_a_fault,
            "a_fault_applied": base_a_applied and not exposure.masked_before_a,
            "source_fault_ids": list(exposure.source_fault_ids),
            "source_layers": list(exposure.source_layers),
            "transport_mediation": list(exposure.transport_mediation),
            "lower_layer_mechanism": exposure.mechanism,
            "lower_layer_evidence": exposure.evidence,
            "masked_before_a": exposure.masked_before_a,
            "observed_A_symptom": a_symptoms,
            "observed_M_consequence": m_consequences,
            "propagation_class": propagation_class,
            "propagation_path": propagation_path,
            "first_divergence": (
                f"{lower_layer}:lower_fault_injected" if exposure.fault_triggered else "none"
            ),
            "events": events,
        }
    )
    return row


def classify_bridge_propagation(
    *,
    condition: str,
    masked_before_a: bool,
    a_symptoms: Sequence[str],
    m_consequences: Sequence[str],
    recovery_detected: bool,
    final_task_success: bool,
) -> str:
    if condition == "clean":
        return "clean"
    if masked_before_a:
        return "masked_before_A"
    m_exposed = any(label != "none" for label in m_consequences)
    a_exposed = any(label != "none" for label in a_symptoms)
    if a_exposed and not m_exposed:
        return "exposed_at_A_only"
    if m_exposed and recovery_detected and final_task_success:
        return "propagated_to_M_recovered"
    if m_exposed and not final_task_success:
        return "propagated_to_M_final_failure"
    if m_exposed:
        return "propagated_to_M_final_success"
    return "detected_but_unrecovered"


def _normalize_labels(value: Any) -> list[str]:
    if value in (None, "", []):
        return ["none"]
    if isinstance(value, str):
        labels = [part.strip() for part in value.replace(",", "/").split("/") if part.strip()]
    else:
        labels = [str(part) for part in value if str(part)]
    non_none = sorted({label for label in labels if label != "none"})
    return non_none or ["none"]


def _merge_labels(*values: Any) -> list[str]:
    labels: list[str] = []
    for value in values:
        labels.extend(_normalize_labels(value))
    non_none = sorted({label for label in labels if label != "none"})
    return non_none or ["none"]


def build_bridge_summary(
    rows: Sequence[dict[str, Any]],
    *,
    experiment_name: str = "bottom_up_webarena_shopping_bridge",
) -> dict[str, Any]:
    fault_rows = [row for row in rows if row.get("condition") != "clean"]
    denominator = len(fault_rows) or 1
    a_count = sum(_has_non_none(row.get("observed_A_symptom")) for row in fault_rows)
    m_count = sum(_has_non_none(row.get("observed_M_consequence")) for row in fault_rows)
    recovery_count = sum(bool(row.get("recovery_detected")) for row in fault_rows)
    failure_count = sum(not bool(row.get("final_task_success")) for row in fault_rows)
    transitions = Counter(str(row.get("propagation_class", "unknown")) for row in rows)

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row.get("condition")) for row in rows}):
        selected = [row for row in rows if row.get("condition") == condition]
        by_condition[condition] = _group_metrics(selected)
    by_topology: dict[str, dict[str, Any]] = {}
    for topology in sorted({str(row.get("topology")) for row in rows}):
        selected = [row for row in rows if row.get("topology") == topology]
        by_topology[topology] = _group_metrics(selected)

    return {
        "experiment": experiment_name,
        "total_runs": len(rows),
        "fault_runs": len(fault_rows),
        "metrics": {
            "A_layer_exposure_rate": round(a_count / denominator, 6),
            "M_layer_propagation_rate": round(m_count / denominator, 6),
            "recovery_rate": round(recovery_count / denominator, 6),
            "final_failure_rate": round(failure_count / denominator, 6),
            "final_task_success_rate_all_runs": round(
                sum(bool(row.get("final_task_success")) for row in rows) / max(1, len(rows)), 6
            ),
            "mean_latency_ms": round(
                sum(float(row.get("latency_ms", 0) or 0) for row in rows) / max(1, len(rows)), 3
            ),
            "mean_total_tokens": round(
                sum(int(row.get("total_tokens", 0) or 0) for row in rows) / max(1, len(rows)), 3
            ),
        },
        "transition_counts": {
            "fault_injected_to_masked_before_A": transitions["masked_before_A"],
            "fault_injected_to_exposed_at_A_only": transitions["exposed_at_A_only"],
            "fault_injected_to_propagated_to_M_recovered": transitions["propagated_to_M_recovered"],
            "fault_injected_to_propagated_to_M_final_success": transitions["propagated_to_M_final_success"],
            "fault_injected_to_propagated_to_M_final_failure": transitions["propagated_to_M_final_failure"],
            "clean": transitions["clean"],
        },
        "by_condition": by_condition,
        "by_topology": by_topology,
    }


def _group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "masked_before_A": sum(bool(row.get("masked_before_a")) for row in rows),
        "A_exposure": sum(_has_non_none(row.get("observed_A_symptom")) for row in rows),
        "M_consequence": sum(_has_non_none(row.get("observed_M_consequence")) for row in rows),
        "recovery": sum(bool(row.get("recovery_detected")) for row in rows),
        "final_success": sum(bool(row.get("final_task_success")) for row in rows),
        "final_failure": sum(not bool(row.get("final_task_success")) for row in rows),
    }


def _has_non_none(value: Any) -> bool:
    return any(label != "none" for label in _normalize_labels(value))


def write_bridge_outputs(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_config: dict[str, Any],
    condition_definitions: Sequence[Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [{key: value for key, value in row.items() if key != "events"} for row in rows]
    _write_jsonl(output_dir / "llm_communication_runs.jsonl", run_rows)
    _write_jsonl(
        output_dir / "llm_communication_traces.jsonl",
        [event for row in rows for event in row.get("events", [])],
    )
    _write_csv(output_dir / "llm_communication_runs.csv", run_rows)

    experiment_name = str(
        experiment_config.get("experiment", "bottom_up_webarena_shopping_bridge")
    )
    summary = build_bridge_summary(rows, experiment_name=experiment_name)
    (output_dir / "llm_communication_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_rows = [
        {"group": "condition", "name": name, **metrics}
        for name, metrics in summary["by_condition"].items()
    ] + [
        {"group": "topology", "name": name, **metrics}
        for name, metrics in summary["by_topology"].items()
    ]
    _write_csv(output_dir / "llm_communication_summary.csv", summary_rows)
    (output_dir / "llm_communication_summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    (output_dir / "representative_causal_traces.md").write_text(
        _representative_traces_markdown(rows), encoding="utf-8"
    )
    definitions = condition_definitions or BRIDGE_CONDITIONS
    config = {**experiment_config, "conditions": [asdict(item) for item in definitions]}
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    run_ids = [str(row.get("run_id")) for row in rows]
    job_keys = [str(row.get("job_key") or _fallback_job_key(row)) for row in rows]
    expected_runs = int(experiment_config.get("expected_runs", len(rows)))
    gate = {
        "passed": (
            len(rows) == expected_runs
            and len(run_ids) == len(set(run_ids))
            and len(job_keys) == len(set(job_keys))
            and not any(row.get("error") for row in rows)
        ),
        "expected_runs": expected_runs,
        "run_count": len(rows),
        "unique_run_ids": len(set(run_ids)),
        "unique_job_keys": len(set(job_keys)),
        "error_count": sum(bool(row.get("error")) for row in rows),
        "models": sorted({str(row.get("model")) for row in rows}),
        "providers": sorted({str(row.get("provider")) for row in rows}),
        "total_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in rows),
    }
    (output_dir / "matrix_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _fallback_job_key(row: dict[str, Any]) -> str:
    return "::".join(
        str(row.get(key, ""))
        for key in ("task_id", "topology", "condition", "injection_step", "repeat_index")
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = [
        {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _summary_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# 底层到 MAS 的桥接实验汇总",
        "",
        f"- 总运行数：{summary['total_runs']}",
        f"- Fault 运行数：{summary['fault_runs']}",
        f"- A 层暴露率：{metrics['A_layer_exposure_rate']:.3f}",
        f"- M 层传播率：{metrics['M_layer_propagation_rate']:.3f}",
        f"- 恢复率：{metrics['recovery_rate']:.3f}",
        f"- 最终失败率：{metrics['final_failure_rate']:.3f}",
        "",
        "## 按底层机制",
        "",
        "| Condition | Runs | Masked before A | A exposure | M consequence | Recovery | Final success | Final failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, values in summary["by_condition"].items():
        lines.append(
            f"| {condition} | {values['runs']} | {values['masked_before_A']} | "
            f"{values['A_exposure']} | {values['M_consequence']} | {values['recovery']} | "
            f"{values['final_success']} | {values['final_failure']} |"
        )
    lines.extend(["", "## 转移计数", "", "```json", json.dumps(summary["transition_counts"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def _representative_traces_markdown(rows: Sequence[dict[str, Any]]) -> str:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        chosen.setdefault(str(row.get("propagation_class")), row)
    lines = ["# 代表性端到端因果 Trace", ""]
    for label, row in sorted(chosen.items()):
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Run: `{row.get('run_id')}`",
                f"- Task: `{row.get('task_id')}`",
                f"- Topology: `{row.get('topology')}`",
                f"- Condition: `{row.get('condition')}`",
                f"- Chain: `{' -> '.join(row.get('propagation_path', []))}`",
                f"- Final success: `{row.get('final_task_success')}`",
                f"- Recovery: `{row.get('recovery_detected')}` / `{row.get('recovery_type', 'none')}`",
                "",
            ]
        )
    return "\n".join(lines)
