import asyncio
import json
from unittest.mock import patch

import pytest

from mas_faults.llm.architecture_manifestations import classify_manifestation
from mas_faults.llm.communication_interceptor import MessageEnvelope
from summarize_rq2_cases import build_case_summary
from mas_faults.llm.webarena_topologies import SUPPORTED_TOPOLOGIES, get_topology
from run_webarena_architecture_rq2 import append_checkpoint, decision_evidence_for_topology, decision_role_for_topology, delivery_payloads, execute_delivered_actions, guard_deepseek_schedule, iter_run_specs, load_checkpoint_rows, normalize_single_action_request, pending_run_specs, repeat_metadata, require_model, resolve_tasks, run_with_retries


CURRENT_EVIDENCE = {
    "task_id": "shopping-001",
    "product_title": "Orange Vanilla Tea",
    "product_id": "orange-vanilla-tea",
    "sku": "TEA-001",
    "requested_quantity": 1,
    "observed_quantity": 1,
    "cart_verified": True,
    "evidence": "current cart observation",
}
STALE_EVIDENCE = {
    **CURRENT_EVIDENCE,
    "task_id": "previous-task",
    "product_id": "old-product",
    "sku": "OLD-SKU",
    "evidence": "previous cart observation",
}
REJECTED_VERIFICATION = {"decision": "reject", "task_id": "shopping-001", "reason": "stale evidence"}


def test_supported_topologies_include_the_baseline_sequential_chain():
    assert SUPPORTED_TOPOLOGIES == ("sequential", "flat", "hierarchical", "team", "hybrid")


def test_sequential_definition_has_no_direct_worker_decider_bypass():
    topology = get_topology("sequential")

    assert topology.architecture_taxonomy == "sequential"
    assert topology.message_path == (
        ("Coordinator", "Shopping Worker"),
        ("Shopping Worker", "WebArena Shopping"),
        ("WebArena Shopping", "Shopping Worker"),
        ("Shopping Worker", "Verifier"),
        ("Verifier", "Coordinator"),
    )
    assert ("Shopping Worker", "Coordinator") not in topology.message_path
    assert decision_role_for_topology("sequential") == "Coordinator"


def test_flat_has_peer_worker_verifier_edge():
    topology = get_topology("flat")
    assert topology.architecture_taxonomy == "flat"
    assert ("Shopping Worker", "Verifier") in topology.message_path


def test_flat_definition_has_direct_worker_coordinator_edge():
    topology = get_topology("flat")
    assert ("Shopping Worker", "Coordinator") in topology.message_path


def test_hierarchical_has_supervisor_worker_edge():
    topology = get_topology("hierarchical")
    assert topology.architecture_taxonomy == "hierarchical"
    assert ("Supervisor", "Shopping Worker") in topology.message_path


def test_team_has_execution_verification_handoff():
    topology = get_topology("team")
    assert topology.architecture_taxonomy == "team"
    assert ("Execution Team", "Verification Team") in topology.message_path


def test_hybrid_has_peer_and_supervisor_edges():
    topology = get_topology("hybrid")
    assert topology.architecture_taxonomy == "hybrid"
    assert ("Shopping Worker", "Verifier") in topology.message_path
    assert ("Verifier", "Supervisor") in topology.message_path
    assert ("Shopping Worker", "Supervisor") in topology.message_path


def test_unknown_topology_is_rejected():
    with pytest.raises(ValueError, match="unsupported topology"):
        get_topology("society")


def test_hierarchical_marks_control_bottleneck_when_only_evidence_is_missing():
    result = classify_manifestation("hierarchical", None, None, None)
    assert result.label == "hierarchical_control_bottleneck"


def test_team_marks_cross_boundary_handoff_failure_when_only_handoff_is_missing():
    result = classify_manifestation("team", None, None, None)
    assert result.label == "team_cross_boundary_handoff_failure"


def test_flat_marks_peer_divergence_for_current_direct_and_stale_primary_payloads():
    result = classify_manifestation("flat", STALE_EVIDENCE, CURRENT_EVIDENCE, REJECTED_VERIFICATION)
    assert result.label == "flat_peer_state_divergence"


def test_hybrid_marks_authority_conflict_for_current_direct_and_stale_primary_payloads():
    result = classify_manifestation("hybrid", STALE_EVIDENCE, CURRENT_EVIDENCE, REJECTED_VERIFICATION)
    assert result.label == "hybrid_authority_conflict"


def test_case_summary_groups_by_manifestation():
    summary = build_case_summary([
        {"topology": "flat", "fault_type": "stale_replay", "injection_step": 4, "architecture_manifestation": "flat_peer_state_divergence"}
    ])
    assert summary["manifestation_counts"]["flat"]["flat_peer_state_divergence"] == 1


def test_single_action_contract_normalizes_worker_planning_noise():
    task = {
        "task_id": "shopping-001",
        "product_title": "Orange Vanilla Tea",
        "product_url": "http://localhost:7770/orange-vanilla-tea.html",
        "quantity": 1,
    }
    action = normalize_single_action_request({"action": "navigate", "task_id": "wrong-task"}, task)
    assert action == {
        "task_id": "shopping-001",
        "action": "add_to_cart",
        "product_title": "Orange Vanilla Tea",
        "product_url": "http://localhost:7770/orange-vanilla-tea.html",
        "quantity": 1,
    }


def test_matrix_includes_every_selected_task_for_each_topology_condition_and_step():
    tasks = [{"task_id": "shopping-001"}, {"task_id": "shopping-002"}]
    specs = list(iter_run_specs(("flat", "team"), ("none", "omission"), (4,), 1, tasks))
    assert len(specs) == 8
    assert {(topology, fault, task["task_id"]) for topology, fault, _, _, task in specs} == {
        (topology, fault, task["task_id"])
        for topology in ("flat", "team")
        for fault in ("none", "omission")
        for task in tasks
    }


def test_full_application_profile_uses_only_semantically_valid_injection_steps():
    specs = list(iter_run_specs(
        ("flat",),
        ("none", "duplicate_request", "reordering", "prompt_injection"),
        (2, 3, 4),
        1,
        [{"task_id": "shopping-001"}],
        application_layer_full=True,
    ))

    assert {(fault, step) for _, fault, step, _, _ in specs} == {
        ("none", 2),
        ("duplicate_request", 2),
        ("reordering", 3),
        ("reordering", 4),
        ("prompt_injection", 3),
        ("prompt_injection", 4),
    }


def test_matrix_can_start_repeat_indices_after_existing_r1():
    specs = list(iter_run_specs(
        ("flat",),
        ("schema_mismatch",),
        (4,),
        2,
        [{"task_id": "shopping-001"}],
        run_index_start=2,
    ))

    assert [(fault, step, run_index) for _, fault, step, run_index, _ in specs] == [
        ("schema_mismatch", 4, 2),
        ("schema_mismatch", 4, 3),
    ]


def test_repeat_metadata_is_explicit_in_canonical_run_records():
    assert repeat_metadata(3) == {
        "seed_or_run_index": 3,
        "repeat_index": 3,
        "model_seed": None,
    }


def test_duplicate_delivery_executes_every_valid_action_and_rechecks_final_cart_state():
    class Executor:
        def __init__(self):
            self.executed = []

        def add_to_cart(self, action):
            self.executed.append(action)
            return {"cart_verified": True, "observed_quantity": 1}

        def verify_cart(self, task):
            return {"task_id": task["task_id"], "cart_verified": False, "observed_quantity": 2, "evidence": "duplicate item"}

    task = {"task_id": "shopping-001", "quantity": 1}
    action = {"action": "add_to_cart", "task_id": "shopping-001"}
    environment = execute_delivered_actions(Executor(), [action, action], task)

    assert environment["observed_quantity"] == 2
    assert environment["cart_verified"] is False


def test_m_evaluator_receives_the_message_that_final_decider_actually_used():
    injected = {"task_id": "shopping-001", "cart_verified": False}
    direct = {"task_id": "shopping-001", "cart_verified": True}
    assert decision_evidence_for_topology("hierarchical", injected, direct) == injected
    assert decision_evidence_for_topology("team", injected, direct) == injected
    assert decision_evidence_for_topology("sequential", injected, direct) == injected
    assert decision_evidence_for_topology("flat", injected, direct) == direct
    assert decision_evidence_for_topology("hybrid", injected, direct) == direct


def test_trace_keeps_non_json_delivered_payloads_from_the_interceptor():
    payloads = delivery_payloads([
        MessageEnvelope("Shopping Worker", "Verifier", "{\"task_id\": \"shopping-001\""),
    ])
    assert payloads == ['{"task_id": "shopping-001"']


def test_manifest_tasks_are_used_without_live_discovery(tmp_path):
    manifest = tmp_path / "frozen_tasks.json"
    expected = [
        {"task_id": "shopping-002-q1", "product_title": "Tea", "product_url": "http://shop/tea", "quantity": 1},
        {"task_id": "shopping-004-q2", "product_title": "Coffee", "product_url": "http://shop/coffee", "quantity": 2},
    ]
    manifest.write_text(json.dumps({"tasks": expected}), encoding="utf-8")

    selected = resolve_tasks(str(manifest), task_count=2, base_url="http://unused")

    assert selected == expected


def test_run_with_retries_replays_only_a_transient_runtime_error():
    attempts = []

    async def transient_runner():
        attempts.append("called")
        if len(attempts) == 1:
            raise RuntimeError("deepseek request failed: deadline")
        return {"run_id": "recovered-run"}

    result = asyncio.run(run_with_retries(transient_runner, attempts=2, delay_seconds=0))

    assert result == {"run_id": "recovered-run"}
    assert len(attempts) == 2


def test_append_checkpoint_persists_one_run_and_its_trace_events(tmp_path):
    row = {
        "run_id": "rq2-flat-shopping-001-none-step2-r1",
        "final_task_success": True,
        "events": [{"run_id": "rq2-flat-shopping-001-none-step2-r1", "step_id": "step-001"}],
    }

    append_checkpoint(row, tmp_path)

    run_lines = (tmp_path / "llm_communication_runs.checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
    trace_lines = (tmp_path / "llm_communication_traces.checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in run_lines] == [{"run_id": row["run_id"], "final_task_success": True}]
    assert [json.loads(line) for line in trace_lines] == row["events"]


def test_resume_loads_atomic_full_rows_and_skips_completed_specs(tmp_path):
    completed = {
        "run_id": "rq2-flat-shopping-001-schema_mismatch-step4-r2-uuid",
        "task_id": "shopping-001",
        "topology": "flat",
        "fault_type": "schema_mismatch",
        "injection_step": 4,
        "repeat_index": 2,
        "final_task_success": True,
        "events": [{"run_id": "rq2-flat-shopping-001-schema_mismatch-step4-r2-uuid", "step_id": "step-001"}],
    }
    append_checkpoint(completed, tmp_path)
    loaded = load_checkpoint_rows(tmp_path)
    specs = list(iter_run_specs(
        ("flat",),
        ("schema_mismatch",),
        (4,),
        2,
        [{"task_id": "shopping-001"}],
        run_index_start=2,
    ))

    assert loaded == [completed]
    assert [(fault, run_index) for _, fault, _, run_index, _ in pending_run_specs(specs, loaded)] == [
        ("schema_mismatch", 3),
    ]


def test_required_model_gate_rejects_accidental_default_model():
    class Client:
        model_info = type("ModelInfo", (), {"model": "deepseek-chat"})()

    with pytest.raises(ValueError, match="deepseek-v4-pro"):
        require_model(Client(), "deepseek-v4-pro")


def test_each_shopping_run_uses_the_shared_deepseek_schedule_guard():
    class Client:
        model_info = type("ModelInfo", (), {"model": "deepseek-v4-flash"})()

    with patch("run_webarena_architecture_rq2.ensure_deepseek_offpeak") as guard:
        guard_deepseek_schedule(Client())

    guard.assert_called_once_with("deepseek-v4-flash")
