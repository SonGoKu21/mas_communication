import asyncio
import json

import pytest

from mas_faults.webarena_admin_main_matrix import (
    CONDITION_BY_NAME,
    MainCommunicationInterceptor,
)
from mas_faults.webarena_admin_topologies import (
    TOPOLOGY_POLICIES,
    run_decision_topology,
)


class RecordingClient:
    def __init__(self) -> None:
        self.calls = []

    def complete_with_metadata(self, prompt, *, json_mode, metadata):
        self.calls.append((metadata["agent_role"], prompt, json_mode))
        role = metadata["agent_role"]
        if role in {"Verifier", "Supervisor Verify"}:
            if "null" in prompt or '{"message_id":' in prompt:
                return json.dumps(
                    {"decision": "reject", "answer": "N/A", "reason": "missing"}
                )
            return json.dumps(
                {"decision": "accept", "answer": "299", "reason": "complete"}
            )
        return json.dumps(
            {"decision": "accept", "answer": "299", "reason": "accepted branch"}
        )


def _task():
    return {"task_id": "199", "intent": "Get the newest pending order ID"}


def _envelope():
    return {
        "message_id": "m-current",
        "task_id": "199",
        "source_session": "run-current",
        "state_version": 3,
        "payload": {
            "evidence_result": {
                "candidate_answer": "299",
                "evidence_summary": "row 1",
                "evidence_row_indices": [1],
            },
            "visible_evidence": [["ID", "Status"], ["299", "Pending"]],
            "structured_task_evidence": {"row_count": 1},
        },
    }


def test_topology_policies_freeze_three_distinct_message_graphs() -> None:
    assert TOPOLOGY_POLICIES["sequential"].declared_edges == (
        ("Evidence Worker", "Verifier"),
        ("Verifier", "Coordinator"),
    )
    assert TOPOLOGY_POLICIES["flat"].declared_edges == (
        ("Evidence Worker", "Verifier"),
        ("Evidence Worker", "Coordinator"),
        ("Verifier", "Coordinator"),
    )
    assert TOPOLOGY_POLICIES["hierarchical"].declared_edges == (
        ("Evidence Worker", "Supervisor"),
    )


def test_each_topology_executes_only_its_declared_communication_edges() -> None:
    clean = MainCommunicationInterceptor(CONDITION_BY_NAME["clean"]).intercept(
        4, _envelope(), context={"eligible": True}
    )

    for topology, policy in TOPOLOGY_POLICIES.items():
        client = RecordingClient()
        result = asyncio.run(
            run_decision_topology(
                client,
                topology=topology,
                task=_task(),
                delivery=clean,
                direct_delivery=clean if topology == "flat" else None,
            )
        )
        assert result.used_edges == policy.declared_edges
        assert result.final_decision["answer"] == "299"
        assert len(client.calls) == 2
        assert all("authoritative_browser_state" not in prompt for _, prompt, _ in client.calls)


def test_flat_direct_branch_is_constant_and_can_supply_trace_backed_redundancy() -> None:
    envelope = _envelope()
    dropped = MainCommunicationInterceptor(
        CONDITION_BY_NAME["non_delivery_step4"]
    ).intercept(4, envelope, context={"eligible": True})
    direct = MainCommunicationInterceptor(CONDITION_BY_NAME["clean"]).intercept(
        4, envelope, context={"eligible": False, "branch": "direct"}
    )

    result = asyncio.run(
        run_decision_topology(
            RecordingClient(),
            topology="flat",
            task=_task(),
            delivery=dropped,
            direct_delivery=direct,
        )
    )

    assert result.branch_inputs["verifier"] is None
    assert result.branch_inputs["direct"] == envelope
    assert result.final_decision["answer"] == "299"
    assert result.recovery_evidence == (
        "flat_direct_evidence_branch_used_after_verifier_branch_fault",
    )


def test_flat_requires_an_independently_delivered_direct_branch() -> None:
    clean = MainCommunicationInterceptor(CONDITION_BY_NAME["clean"]).intercept(
        4, _envelope(), context={"eligible": True}
    )

    with pytest.raises(ValueError, match="direct delivery"):
        asyncio.run(
            run_decision_topology(
                RecordingClient(),
                topology="flat",
                task=_task(),
                delivery=clean,
            )
        )


def test_sequential_and_hierarchical_do_not_receive_a_hidden_clean_bypass() -> None:
    dropped = MainCommunicationInterceptor(
        CONDITION_BY_NAME["non_delivery_step4"]
    ).intercept(4, _envelope(), context={"eligible": True})

    for topology in ("sequential", "hierarchical"):
        result = asyncio.run(
            run_decision_topology(
                RecordingClient(),
                topology=topology,
                task=_task(),
                delivery=dropped,
            )
        )
        assert "direct" not in result.branch_inputs
        assert result.recovery_evidence == ()


def test_sequential_coordinator_knows_null_direct_branch_is_expected() -> None:
    clean = MainCommunicationInterceptor(CONDITION_BY_NAME["clean"]).intercept(
        4, _envelope(), context={"eligible": True}
    )
    client = RecordingClient()

    asyncio.run(
        run_decision_topology(
            client,
            topology="sequential",
            task=_task(),
            delivery=clean,
        )
    )

    coordinator_prompt = next(
        prompt for role, prompt, _ in client.calls if role == "Coordinator"
    )
    assert "SEQUENTIAL MODE" in coordinator_prompt
    assert "DIRECT_EVIDENCE is expected to be null" in coordinator_prompt
    assert "VERIFICATION is the only authoritative branch input" in coordinator_prompt


def test_hierarchical_supervisor_uses_the_same_strict_output_contract() -> None:
    clean = MainCommunicationInterceptor(CONDITION_BY_NAME["clean"]).intercept(
        4, _envelope(), context={"eligible": True}
    )
    client = RecordingClient()

    asyncio.run(
        run_decision_topology(
            client,
            topology="hierarchical",
            task=_task(),
            delivery=clean,
        )
    )

    supervisor_prompts = [
        prompt for role, prompt, _ in client.calls if role.startswith("Supervisor")
    ]
    assert len(supervisor_prompts) == 2
    for prompt in supervisor_prompts:
        assert "decision (accept or reject)" in prompt
        assert "answer (string)" in prompt
        assert "reason (string)" in prompt
