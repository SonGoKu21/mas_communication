"""AutoGen decision graphs for the WebArena Admin main matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler

from mas_faults.webarena_admin_controlled import (
    AdminAgentMessage,
    parse_coordinator_decision,
)
from mas_faults.webarena_admin_main_matrix import MainDeliveryBatch


@dataclass(frozen=True)
class TopologyPolicy:
    name: str
    declared_edges: tuple[tuple[str, str], ...]


TOPOLOGY_POLICIES = {
    "sequential": TopologyPolicy(
        "sequential",
        (("Evidence Worker", "Verifier"), ("Verifier", "Coordinator")),
    ),
    "flat": TopologyPolicy(
        "flat",
        (
            ("Evidence Worker", "Verifier"),
            ("Evidence Worker", "Coordinator"),
            ("Verifier", "Coordinator"),
        ),
    ),
    "hierarchical": TopologyPolicy(
        "hierarchical", (("Evidence Worker", "Supervisor"),)
    ),
}


@dataclass(frozen=True)
class TopologyDecisionResult:
    topology: str
    declared_edges: tuple[tuple[str, str], ...]
    used_edges: tuple[tuple[str, str], ...]
    branch_inputs: dict[str, Any]
    verification: dict[str, str]
    final_decision: dict[str, str]
    recovery_evidence: tuple[str, ...]


def _complete(client: Any, prompt: str, role: str) -> str:
    method = getattr(client, "complete_with_metadata", None)
    if callable(method):
        return str(
            method(prompt, json_mode=True, metadata={"agent_role": role})
        ).strip()
    return str(client.complete(prompt, json_mode=True)).strip()


def _decision(raw: str) -> dict[str, str]:
    return parse_coordinator_decision(raw)


def _candidate(envelope: Any) -> str | None:
    if not isinstance(envelope, dict):
        return None
    payload = envelope.get("payload")
    evidence = payload.get("evidence_result") if isinstance(payload, dict) else None
    value = evidence.get("candidate_answer") if isinstance(evidence, dict) else None
    return value if isinstance(value, str) else None


class TopologyVerifierAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("WebArena Admin topology verifier")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        prompt = (
            "You are the Verifier in a read-only WebArena MAS. Use only DELIVERED_EVIDENCE. "
            "Return strict JSON with exactly decision (accept or reject), answer (string), and "
            "reason (string). Reject missing, malformed, stale, partial, type-invalid, or "
            "internally inconsistent evidence. Do not use hidden browser state.\n"
            f"TASK: {payload['task']['intent']}\n"
            f"EXPECTED_TASK_ID: {payload['task']['task_id']}\n"
            f"DELIVERED_EVIDENCE: {json.dumps(payload.get('evidence'), ensure_ascii=False)}"
        )
        return AdminAgentMessage(
            "verification",
            {"raw": _complete(self.client, prompt, "Verifier")},
        )


class TopologyCoordinatorAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("WebArena Admin topology coordinator")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        topology = str(payload.get("topology", "sequential"))
        if topology == "sequential":
            mode_rule = (
                "SEQUENTIAL MODE: VERIFICATION is the only authoritative branch input. "
                "DIRECT_EVIDENCE is expected to be null and its absence is not missing evidence. "
                "When VERIFICATION accepts a supported answer, preserve that answer unless the "
                "verification object is itself malformed or contradictory."
            )
        else:
            mode_rule = (
                "FLAT MODE: compare VERIFICATION with the intentional DIRECT_EVIDENCE branch. "
                "A valid direct branch may support the final answer when the verifier branch rejects."
            )
        prompt = (
            "You are the Coordinator. Return strict JSON with exactly decision (accept or reject), "
            "answer (string), and reason (string). Use only the branch inputs supplied below. "
            "A direct evidence branch, when present, is an intentional topology path. Reject when "
            "the available branches do not entail an answer. Do not use hidden browser state.\n"
            f"{mode_rule}\n"
            f"TASK: {payload['task']['intent']}\n"
            f"EXPECTED_TASK_ID: {payload['task']['task_id']}\n"
            f"VERIFICATION: {json.dumps(payload.get('verification'), ensure_ascii=False)}\n"
            f"DIRECT_EVIDENCE: {json.dumps(payload.get('direct_evidence'), ensure_ascii=False)}"
        )
        return AdminAgentMessage(
            "decision",
            {"raw": _complete(self.client, prompt, "Coordinator")},
        )


class TopologySupervisorAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("WebArena Admin hierarchical supervisor")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        if message.kind == "supervisor_verify":
            prompt = (
                "You are the centralized Supervisor verification phase. Use only DELIVERED_EVIDENCE. "
                "Return strict JSON with exactly decision (accept or reject), answer (string), and "
                "reason (string). Reject missing, malformed, "
                "stale, partial, type-invalid, or inconsistent evidence.\n"
                f"TASK: {payload['task']['intent']}\n"
                f"EXPECTED_TASK_ID: {payload['task']['task_id']}\n"
                f"DELIVERED_EVIDENCE: {json.dumps(payload.get('evidence'), ensure_ascii=False)}"
            )
            role = "Supervisor Verify"
            kind = "supervisor_verification"
        else:
            prompt = (
                "You are the centralized Supervisor decision phase. Return strict JSON with exactly "
                "decision (accept or reject), answer (string), and reason (string), using only your "
                "supplied verification result.\n"
                f"TASK: {payload['task']['intent']}\n"
                f"VERIFICATION: {json.dumps(payload.get('verification'), ensure_ascii=False)}"
            )
            role = "Supervisor Decide"
            kind = "supervisor_decision"
        return AdminAgentMessage(
            kind,
            {"raw": _complete(self.client, prompt, role)},
        )


async def _ask(
    runtime: SingleThreadedAgentRuntime,
    recipient: AgentId,
    message: AdminAgentMessage,
) -> AdminAgentMessage:
    response = await runtime.send_message(message, recipient)
    if not isinstance(response, AdminAgentMessage):
        raise TypeError(f"unexpected AutoGen response: {response!r}")
    return response


async def run_decision_topology(
    client: Any,
    *,
    topology: str,
    task: dict[str, Any],
    delivery: MainDeliveryBatch,
    direct_delivery: MainDeliveryBatch | None = None,
) -> TopologyDecisionResult:
    try:
        policy = TOPOLOGY_POLICIES[topology]
    except KeyError as exc:
        raise ValueError(f"unknown topology: {topology}") from exc

    if topology == "flat" and direct_delivery is None:
        raise ValueError("flat topology requires an independent direct delivery")

    delivered = delivery.delivered_messages[0] if delivery.delivered_messages else None
    runtime = SingleThreadedAgentRuntime()
    verifier_id = AgentId("admin_main_verifier", "default")
    coordinator_id = AgentId("admin_main_coordinator", "default")
    supervisor_id = AgentId("admin_main_supervisor", "default")
    await runtime.register_agent_instance(TopologyVerifierAgent(client), verifier_id)
    await runtime.register_agent_instance(TopologyCoordinatorAgent(client), coordinator_id)
    await runtime.register_agent_instance(TopologySupervisorAgent(client), supervisor_id)
    runtime.start()
    branch_inputs: dict[str, Any]
    recovery_evidence: tuple[str, ...] = ()
    try:
        if topology in {"sequential", "flat"}:
            verification_message = await _ask(
                runtime,
                verifier_id,
                AdminAgentMessage(
                    "verify", {"task": task, "evidence": delivered}
                ),
            )
            verification = _decision(str(verification_message.payload["raw"]))
            direct = None
            if topology == "flat" and direct_delivery is not None:
                direct = (
                    direct_delivery.delivered_messages[0]
                    if direct_delivery.delivered_messages
                    else None
                )
            decision_message = await _ask(
                runtime,
                coordinator_id,
                AdminAgentMessage(
                    "decide",
                    {
                        "task": task,
                        "topology": topology,
                        "verification": verification,
                        "direct_evidence": direct,
                    },
                ),
            )
            final_decision = _decision(str(decision_message.payload["raw"]))
            branch_inputs = {"verifier": delivered}
            if topology == "flat":
                branch_inputs["direct"] = direct
                direct_candidate = _candidate(direct)
                verifier_candidate = verification.get("answer")
                if (
                    delivery.fault_applied
                    and final_decision.get("decision") == "accept"
                    and direct_candidate is not None
                    and final_decision.get("answer") == direct_candidate
                    and (
                        verification.get("decision") == "reject"
                        or verifier_candidate != direct_candidate
                    )
                ):
                    recovery_evidence = (
                        "flat_direct_evidence_branch_used_after_verifier_branch_fault",
                    )
        else:
            verification_message = await _ask(
                runtime,
                supervisor_id,
                AdminAgentMessage(
                    "supervisor_verify", {"task": task, "evidence": delivered}
                ),
            )
            verification = _decision(str(verification_message.payload["raw"]))
            decision_message = await _ask(
                runtime,
                supervisor_id,
                AdminAgentMessage(
                    "supervisor_decide",
                    {"task": task, "verification": verification},
                ),
            )
            final_decision = _decision(str(decision_message.payload["raw"]))
            branch_inputs = {"supervisor": delivered}
    finally:
        await runtime.stop()

    return TopologyDecisionResult(
        topology=topology,
        declared_edges=policy.declared_edges,
        used_edges=policy.declared_edges,
        branch_inputs=branch_inputs,
        verification=verification,
        final_decision=final_decision,
        recovery_evidence=recovery_evidence,
    )
