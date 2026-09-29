import asyncio
import json

import pytest

from mas_faults.bottom_up_bridge import (
    BRIDGE_CONDITIONS,
    BridgeRunSpec,
    build_bridge_specs,
    build_bridge_summary,
    enrich_run_record,
    probe_lower_layer,
    write_bridge_outputs,
)
from run_bottom_up_bridge_experiment import (
    load_checkpoint_rows,
    pending_specs,
    select_frozen_tasks,
    validate_local_model,
)


TASKS = [
    {"task_id": "shopping-001-q1"},
    {"task_id": "shopping-005-q2"},
    {"task_id": "shopping-009-q3"},
]


def test_bridge_manifest_contains_exactly_108_unique_units():
    specs = build_bridge_specs(
        tasks=TASKS,
        topologies=("sequential", "flat"),
        repetitions=3,
    )

    assert len(BRIDGE_CONDITIONS) == 6
    assert len(specs) == 108
    assert len({spec.job_key for spec in specs}) == 108
    assert {spec.injection_step for spec in specs if spec.condition == "retry_induced_duplication"} == {2}
    assert {spec.injection_step for spec in specs if spec.condition == "async_logical_reordering"} == {3}
    assert {
        spec.injection_step
        for spec in specs
        if spec.condition in {"deadline_induced_omission", "stream_interruption_truncation"}
    } == {4}


def test_physical_bit_flip_is_triggered_but_masked_before_a():
    payload = {"task_id": "shopping-001-q1", "cart_verified": True}

    exposure = asyncio.run(probe_lower_layer("physical_bit_flip_masked", payload, seed=1))

    assert exposure.fault_triggered is True
    assert exposure.masked_before_a is True
    assert exposure.observed_a_faults == ("none",)
    assert exposure.a_operator == "none"
    assert exposure.evidence["checksum_mismatch"] is True
    assert exposure.evidence["retransmission_delivered_original"] is True


def test_exposed_lower_layer_mechanisms_are_inferred_from_runtime_evidence():
    payload = {"task_id": "shopping-001-q1", "cart_verified": True}

    expected = {
        "deadline_induced_omission": (("A2", "A5"), "timeout"),
        "stream_interruption_truncation": (("A7", "A8"), "truncation"),
        "retry_induced_duplication": (("A9",), "duplicate_request"),
        "async_logical_reordering": (("A10",), "reordering"),
    }
    for index, (condition, (symptoms, operator)) in enumerate(expected.items(), start=1):
        exposure = asyncio.run(probe_lower_layer(condition, payload, seed=index))
        assert exposure.fault_triggered is True
        assert exposure.masked_before_a is False
        assert exposure.observed_a_faults == symptoms
        assert exposure.a_operator == operator


def test_enriched_record_has_ordered_lower_a_m_final_trace():
    spec = BridgeRunSpec(
        task_id="shopping-001-q1",
        topology="sequential",
        condition="deadline_induced_omission",
        repeat_index=1,
        injection_step=4,
    )
    exposure = asyncio.run(probe_lower_layer(spec.condition, {"task_id": spec.task_id}, seed=1))
    base = {
        "run_id": "base-run",
        "trace_id": "trace-1",
        "model": "Qwen3.8-27B",
        "provider": "modelscope_local",
        "task_id": spec.task_id,
        "topology": spec.topology,
        "fault_type": "timeout",
        "fault_applied": True,
        "observed_A_symptom": ["A2"],
        "observed_M_consequence": ["M3_incomplete_information_aggregation"],
        "recovery_detected": False,
        "recovery_type": "none",
        "final_task_success": False,
        "task_score": 0.0,
        "latency_ms": 100.0,
        "total_tokens": 20,
        "events": [
            {
                "run_id": "base-run",
                "trace_id": "trace-1",
                "step_id": "step-004",
                "step_index": 4,
                "timestamp": "2026-08-25T00:00:01+00:00",
                "observed_A_symptom": "A2",
            }
        ],
    }

    row = enrich_run_record(base, exposure, spec)

    assert row["fault_type"] == spec.condition
    assert row["a_fault_type"] == "timeout"
    assert row["fault_applied"] is True
    assert row["a_fault_applied"] is True
    assert row["observed_A_symptom"] == ["A2", "A5"]
    assert row["propagation_class"] == "propagated_to_M_final_failure"
    assert row["propagation_path"] == [
        "N2",
        "N3",
        "T5",
        "A2",
        "A5",
        "M3_incomplete_information_aggregation",
        "final_failure",
    ]
    assert [event["event_layer"] for event in row["events"]] == ["N", "T", "A", "M"]


def test_masked_fault_is_not_mislabeled_as_clean_or_a_exposure():
    spec = BridgeRunSpec(
        task_id="shopping-001-q1",
        topology="flat",
        condition="physical_bit_flip_masked",
        repeat_index=1,
        injection_step=4,
    )
    exposure = asyncio.run(probe_lower_layer(spec.condition, {"task_id": spec.task_id}, seed=1))
    base = {
        "run_id": "masked-run",
        "trace_id": "trace-masked",
        "model": "Qwen3.8-27B",
        "provider": "modelscope_local",
        "task_id": spec.task_id,
        "topology": spec.topology,
        "fault_type": "none",
        "fault_applied": False,
        "observed_A_symptom": ["none"],
        "observed_M_consequence": ["none"],
        "recovery_detected": False,
        "recovery_type": "none",
        "final_task_success": True,
        "task_score": 1.0,
        "latency_ms": 10.0,
        "total_tokens": 10,
        "events": [],
    }

    row = enrich_run_record(base, exposure, spec)

    assert row["condition"] == "physical_bit_flip_masked"
    assert row["fault_applied"] is True
    assert row["a_fault_applied"] is False
    assert row["observed_A_symptom"] == ["none"]
    assert row["propagation_class"] == "masked_before_A"


def test_summary_and_outputs_keep_a_m_and_final_outcomes_separate(tmp_path):
    rows = [
        {
            "run_id": "masked",
            "trace_id": "t1",
            "task_id": "shopping-001-q1",
            "topology": "sequential",
            "condition": "physical_bit_flip_masked",
            "model": "Qwen3.8-27B",
            "provider": "modelscope_local",
            "fault_applied": True,
            "masked_before_a": True,
            "observed_A_symptom": ["none"],
            "observed_M_consequence": ["none"],
            "recovery_detected": False,
            "final_task_success": True,
            "propagation_class": "masked_before_A",
            "latency_ms": 10.0,
            "total_tokens": 10,
            "events": [{"event_layer": "P", "event_type": "lower_layer_fault_injected"}],
        },
        {
            "run_id": "failed",
            "trace_id": "t2",
            "task_id": "shopping-001-q1",
            "topology": "sequential",
            "condition": "deadline_induced_omission",
            "model": "Qwen3.8-27B",
            "provider": "modelscope_local",
            "fault_applied": True,
            "masked_before_a": False,
            "observed_A_symptom": ["A2", "A5"],
            "observed_M_consequence": ["M3_incomplete_information_aggregation"],
            "recovery_detected": False,
            "final_task_success": False,
            "propagation_class": "propagated_to_M_final_failure",
            "latency_ms": 30.0,
            "total_tokens": 30,
            "events": [{"event_layer": "M", "event_type": "final_outcome"}],
        },
    ]

    summary = build_bridge_summary(rows)
    write_bridge_outputs(rows, tmp_path, experiment_config={"expected_runs": 2})

    assert summary["metrics"]["A_layer_exposure_rate"] == 0.5
    assert summary["metrics"]["M_layer_propagation_rate"] == 0.5
    assert summary["metrics"]["final_failure_rate"] == 0.5
    assert summary["transition_counts"] == {
        "fault_injected_to_masked_before_A": 1,
        "fault_injected_to_exposed_at_A_only": 0,
        "fault_injected_to_propagated_to_M_recovered": 0,
        "fault_injected_to_propagated_to_M_final_success": 0,
        "fault_injected_to_propagated_to_M_final_failure": 1,
        "clean": 0,
    }
    assert len((tmp_path / "llm_communication_runs.jsonl").read_text().splitlines()) == 2
    assert len((tmp_path / "llm_communication_traces.jsonl").read_text().splitlines()) == 2
    gate = json.loads((tmp_path / "matrix_gate.json").read_text())
    assert gate["passed"] is True
    assert gate["run_count"] == 2
    assert (tmp_path / "representative_causal_traces.md").is_file()


def test_runner_selects_exact_frozen_tasks_in_requested_order():
    available = [
        {"task_id": "shopping-001-q1", "quantity": 1},
        {"task_id": "shopping-005-q2", "quantity": 2},
        {"task_id": "shopping-009-q3", "quantity": 3},
    ]

    selected = select_frozen_tasks(
        available,
        ("shopping-009-q3", "shopping-001-q1"),
    )

    assert [task["task_id"] for task in selected] == ["shopping-009-q3", "shopping-001-q1"]
    with pytest.raises(ValueError, match="missing frozen task"):
        select_frozen_tasks(available, ("shopping-missing",))


def test_resume_checkpoint_excludes_completed_job_without_duplicate_runs(tmp_path):
    checkpoint = tmp_path / "bridge_full.checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps({"run_id": "done", "job_key": "shopping-001-q1::flat::clean::4::1", "events": []}) + "\n",
        encoding="utf-8",
    )
    specs = [
        BridgeRunSpec("shopping-001-q1", "flat", "clean", 1, 4),
        BridgeRunSpec("shopping-001-q1", "flat", "clean", 2, 4),
    ]

    rows = load_checkpoint_rows(checkpoint)
    remaining = pending_specs(specs, rows)

    assert [spec.repeat_index for spec in remaining] == [2]


def test_runner_refuses_paid_or_wrong_local_model():
    class Client:
        class Info:
            model = "Qwen3.8-27B"
            provider = "modelscope_local"

        model_info = Info()

    validate_local_model(Client(), required_model="Qwen3.8-27B", required_provider="modelscope_local")
    with pytest.raises(ValueError, match="provider"):
        validate_local_model(Client(), required_model="Qwen3.8-27B", required_provider="deepseek")
    with pytest.raises(ValueError, match="model"):
        validate_local_model(Client(), required_model="Qwen3.5-9B", required_provider="modelscope_local")
