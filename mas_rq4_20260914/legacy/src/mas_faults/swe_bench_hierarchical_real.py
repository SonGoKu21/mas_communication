from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.causal_trace_report import build_report
from mas_faults.decision_message_faults import replay_stale_guidance
from mas_faults.llm_client import ChatClient, get_llm_client
from mas_faults.swe_bench_verified_real import TestEvidence
from mas_faults.swe_bench_verified_real import (
    ContextBudget,
    SWEHarness,
    SWEInstance,
    append_jsonl_record,
    build_coder_prompt,
    build_execution_schedule,
    build_planner_prompt,
    build_run_id,
    collect_repository_context_with_metadata,
    generate_valid_patch,
    intercept_evidence,
    intercept_text_message,
    load_completed_run_rows,
    load_verified_instances,
    local_image_exists,
    prepare_output_dir,
    run_official_evaluation,
    select_runnable_instances,
    swe_condition_summary_rows,
    write_event,
)


@dataclass(frozen=True)
class HierarchicalInjectionSlot:
    name: str
    source: str
    target: str
    semantic_slot: str
    edge_role: str
    delivery_scope: str = "unicast"


HIERARCHICAL_INJECTION_SLOTS = {
    "manager_to_coder": HierarchicalInjectionSlot(
        "manager_to_coder", "Manager", "Coder", "S1_control_context", "control", "manager_relay"
    ),
    "coder_to_manager": HierarchicalInjectionSlot(
        "coder_to_manager", "Coder", "Manager", "S2_artifact_state", "artifact_state", "manager_relay"
    ),
    "tester_to_manager": HierarchicalInjectionSlot(
        "tester_to_manager", "Tester", "Manager", "S3_evidence_observation", "evidence", "manager_relay"
    ),
    "reviewer_to_manager": HierarchicalInjectionSlot(
        "reviewer_to_manager", "Reviewer", "Manager", "S4_feedback_decision", "feedback", "manager_relay"
    ),
    "manager_guidance_to_coder": HierarchicalInjectionSlot(
        "manager_guidance_to_coder", "Manager", "Coder", "S4_feedback_decision", "control", "manager_relay"
    ),
}

STALE_EVIDENCE_CONDITIONS = frozenset({"a6_inner_evidence_poisoning", "a12_stale_replay"})


def validate_hierarchical_condition_slot(condition: str, injection_slot: str) -> None:
    if injection_slot not in HIERARCHICAL_INJECTION_SLOTS:
        raise ValueError(f"unsupported Hierarchical injection slot={injection_slot!r}")
    if condition in STALE_EVIDENCE_CONDITIONS and injection_slot != "tester_to_manager":
        raise ValueError(
            f"{condition} requires S3_evidence_observation at tester_to_manager"
        )
    if condition == "a12_guidance_stale_replay" and injection_slot != "manager_guidance_to_coder":
        raise ValueError("a12_guidance_stale_replay requires manager_guidance_to_coder")


def build_hierarchical_cid(
    injection_slot: str,
    *,
    message_rank: int,
    eligible_messages: int,
) -> dict[str, object]:
    if injection_slot not in HIERARCHICAL_INJECTION_SLOTS:
        raise ValueError(f"unsupported Hierarchical injection slot={injection_slot!r}")
    if eligible_messages < 1 or not 1 <= message_rank <= eligible_messages:
        raise ValueError("message_rank must be within the eligible message count")

    slot = HIERARCHICAL_INJECTION_SLOTS[injection_slot]
    phase_percentile = message_rank / eligible_messages
    phase = "early" if phase_percentile <= 1 / 3 else "middle" if phase_percentile <= 2 / 3 else "late"
    return {
        "semantic_slot": slot.semantic_slot,
        "phase": phase,
        "phase_percentile": phase_percentile,
        "delivery_scope": slot.delivery_scope,
        "edge_role": slot.edge_role,
        "realization_source": slot.source,
        "realization_target": slot.target,
    }


def parse_hierarchical_revision_assessment(raw: str) -> dict[str, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"evidence_assessment": "unknown", "review_assessment": "unknown"}
    if not isinstance(payload, dict):
        return {"evidence_assessment": "unknown", "review_assessment": "unknown"}

    allowed_evidence = {"complete", "missing", "partial", "stale", "conflicting", "unknown"}
    evidence = str(payload.get("evidence_assessment", "unknown")).lower()
    review = str(payload.get("review_assessment", "unknown")).lower()
    return {
        "evidence_assessment": evidence if evidence in allowed_evidence else "unknown",
        "review_assessment": review,
    }


def evaluate_hierarchical_run(
    *,
    condition: str,
    final_evidence: TestEvidence,
    coder_assessment: dict[str, str],
) -> dict[str, object]:
    a_symptom = {
        "clean": "none",
        "a1_moderate_delay": "A1_message_latency",
        "a1_deadline_delay": "A2_message_timeout",
        "a5_omission": "A5_message_omission",
        "a6_inner_evidence_poisoning": "A6_message_semantic_corruption",
        "a8_truncation": "A8_message_truncation",
        "a12_stale_replay": "A12_timing_or_session_mismatch",
        "a12_guidance_stale_replay": "A12_timing_or_session_mismatch",
    }[condition]
    evidence_assessment = coder_assessment.get("evidence_assessment", "unknown")
    review_assessment = coder_assessment.get("review_assessment", "unknown")
    accepted_as_complete = evidence_assessment == "complete"
    detected = evidence_assessment in {"missing", "partial", "stale", "conflicting"}
    consequences: list[str] = []

    if condition == "a8_truncation" and accepted_as_complete:
        consequences.extend([
            "M3_incomplete_information_aggregation",
            "M14_partial_tool_or_message_result_acceptance",
        ])
    if condition in {"a12_stale_replay", "a12_guidance_stale_replay"} and accepted_as_complete:
        consequences.extend(["M5_stale_context_acceptance", "M6_state_inconsistency"])
    if condition == "a6_inner_evidence_poisoning" and accepted_as_complete:
        consequences.append("M6_state_inconsistency")

    final_success = bool(final_evidence.passed)
    if not final_success and condition != "clean":
        if review_assessment == "accepted":
            consequences.append("M4_incorrect_collective_decision")
        consequences.append("M2_task_timeout_or_failure")
    consequences = list(dict.fromkeys(consequences))

    if condition == "clean":
        propagation_class = "clean"
    elif consequences and not final_success:
        propagation_class = "propagated_to_M_final_failure"
    elif consequences:
        propagation_class = "silent_propagation_to_M"
    elif detected:
        propagation_class = "detected_but_unrecovered"
    elif final_success:
        propagation_class = "exposed_at_A_only"
    else:
        propagation_class = "fault_not_observed"

    return {
        "observed_A_symptom": [a_symptom],
        "observed_M_consequence": consequences or ["none"],
        "final_task_success": final_success,
        "recovery_detected": False,
        "recovery_type": "none",
        "recovery_evidence": "",
        "propagation_class": propagation_class,
    }


def build_hierarchical_experiment_config(
    instance_ids: list[str],
    conditions: list[str],
    repeats: int,
    model: str,
    provider: str,
    injection_slot: str,
) -> dict[str, object]:
    return {
        "benchmark": "SWE-bench Verified",
        "topology": "hierarchical",
        "model": model,
        "provider": provider,
        "llm_seed": int(os.environ["LLM_SEED"]) if os.environ.get("LLM_SEED") else None,
        "instance_ids": instance_ids,
        "conditions": conditions,
        "runs_per_condition": repeats,
        "injection_slot": injection_slot,
        "workflow": "Manager -> Coder -> Manager -> Tester/Reviewer -> Manager -> Coder -> official SWE evaluator",
        "cid_slots": {
            name: {
                "source": slot.source,
                "target": slot.target,
                "semantic_slot": slot.semantic_slot,
                "edge_role": slot.edge_role,
                "delivery_scope": slot.delivery_scope,
            }
            for name, slot in HIERARCHICAL_INJECTION_SLOTS.items()
        },
    }


def build_hierarchical_tester_prompt(instance: SWEInstance, patch: str | None, evidence: TestEvidence) -> str:
    return (
        "You are the Tester in a hierarchical peer software-repair team. Analyze the real evaluator evidence for the "
        "candidate patch. Return strict JSON with fields assessment (complete|missing|partial|failed|passed), "
        "summary, and risks. Do not fabricate test execution.\n"
        f"Current instance: {instance.instance_id}\nCandidate patch:\n{patch or 'missing'}\n"
        f"Official evaluator evidence:\n{serialize_evidence(evidence)}\n"
    )


def build_hierarchical_reviewer_prompt(instance: SWEInstance, task_card: str, patch: str | None) -> str:
    return (
        "You are the Reviewer in a hierarchical peer software-repair team. Inspect the candidate patch against the issue. "
        "Return strict JSON with fields assessment (accepted|rejected|uncertain), summary, and risks. "
        "Do not claim that tests passed.\n"
        f"Repository: {instance.repo}\nIssue:\n{instance.problem_statement}\n"
        f"Manager task card:\n{task_card}\nCandidate patch:\n{patch or 'missing'}\n"
    )


def build_hierarchical_manager_prompt(
    instance: SWEInstance,
    tester_report: str,
    evidence: TestEvidence | None,
    reviewer_report: str | None,
) -> str:
    return (
        "You are the Manager of a hierarchical software-repair workflow. Workers report only to you. "
        "Return strict JSON with evidence_assessment (complete|missing|partial|stale|conflicting|unknown), "
        "review_assessment, and directive for the Coder. Do not fabricate evidence.\n"
        f"Current instance: {instance.instance_id}\nTester report:\n{tester_report[:1800]}\n"
        f"Delivered evaluator evidence:\n{serialize_evidence(evidence)}\n"
        f"Reviewer report:\n{(reviewer_report or 'missing')[:1800]}\n"
    )


def build_hierarchical_coder_revision_prompt(
    instance: SWEInstance,
    task_card: str,
    candidate_patch: str | None,
    tester_report: str,
    delivered_evidence: TestEvidence | None,
    reviewer_report: str | None,
    repository_context: str,
) -> str:
    # The local Qwen endpoint has an 8192-token context window. The revision
    # round combines peer artifacts, so bound each independently rather than
    # letting a large patch or log crowd out the source context.
    bounded_task_card = task_card[:1200]
    bounded_context = repository_context[:4500]
    bounded_patch = (candidate_patch or "missing")[:2500]
    bounded_tester_report = tester_report[:1500]
    bounded_reviewer_report = (reviewer_report or "missing")[:1500]
    base = build_coder_prompt(instance, bounded_task_card, bounded_context)
    return (
        "You are now revising a candidate patch after peer feedback in a hierarchical software-repair team. "
        "Return only one JSON object with evidence_assessment, review_assessment, and edits. "
        "evidence_assessment must be exactly one of complete, missing, partial, stale, conflicting, unknown. "
        "review_assessment records the received reviewer conclusion. Do not invent missing evidence or claim recovery. "
        "The edits array follows the same path, old, new contract below.\n\n"
        f"{base}\nInitial candidate patch:\n{bounded_patch}\n"
        "The following is the Manager's delivered directive. Do not infer unseen worker messages.\n"
        f"Manager directive:\n{bounded_tester_report}\n"
    )


def serialize_evidence(evidence: TestEvidence | None) -> str:
    if evidence is None:
        return "missing"
    return json.dumps(
        {
            "instance_id": evidence.instance_id,
            "passed": evidence.passed,
            "tests": evidence.tests[:20],
            "log": evidence.log[:1200],
            "runner_error": evidence.runner_error,
            "evidence_origin_instance_id": evidence.evidence_origin_instance_id,
        },
        ensure_ascii=False,
    )


def _hierarchical_text_delivery(message: str, condition: str, injection_slot: str, current_slot: str) -> tuple[str, str | None, str, bool]:
    if injection_slot != current_slot:
        return message, message, "none", False
    intercepted = intercept_text_message(message, condition)
    return intercepted.original, intercepted.delivered, intercepted.observed_a_symptom, intercepted.fault_applied


def _hierarchical_evidence_delivery(
    evidence: TestEvidence,
    condition: str,
    injection_slot: str,
    stale: TestEvidence | None,
) -> tuple[TestEvidence, TestEvidence | None, str, bool]:
    if injection_slot != "tester_to_manager":
        return evidence, evidence, "none", False
    intercepted = intercept_evidence(evidence, condition, stale)
    return intercepted.original, intercepted.delivered, intercepted.observed_a_symptom, intercepted.fault_applied


def _write_hierarchical_delivery(
    events: Path,
    *,
    trace_id: str,
    run_id: str,
    instance: SWEInstance,
    condition: str,
    source: str,
    target: str,
    cid: dict[str, object],
    fault_applied: bool,
    observed_a_symptom: str,
    delivered: object | None,
) -> None:
    write_event(
        events,
        trace_id=trace_id,
        run_id=run_id,
        instance=instance,
        condition=condition,
        event_type="fault_applied" if fault_applied else "message_delivered",
        layer="A",
        component="communication_interceptor" if fault_applied else target,
        status="applied" if fault_applied else "delivered",
        effect=observed_a_symptom,
        label="injected" if fault_applied else "pre_injection",
        evidence={"cid": cid, "delivered": delivered is not None},
        source=source,
        target=target,
        injection_point_kind="hierarchical_communication_interceptor",
    )
    if fault_applied:
        write_hierarchical_runtime_effect(
            events,
            trace_id=trace_id,
            run_id=run_id,
            instance=instance,
            condition=condition,
            source=source,
            target=target,
            cid=cid,
            observed_a_symptom=observed_a_symptom,
        )
        write_hierarchical_delivery_outcome(
            events,
            trace_id=trace_id,
            run_id=run_id,
            instance=instance,
            condition=condition,
            source=source,
            target=target,
            cid=cid,
            observed_a_symptom=observed_a_symptom,
            delivered=delivered is not None,
        )


def write_hierarchical_runtime_effect(
    events: Path,
    *,
    trace_id: str,
    run_id: str,
    instance: SWEInstance,
    condition: str,
    source: str,
    target: str,
    cid: dict[str, object],
    observed_a_symptom: str,
) -> None:
    write_event(
        events,
        trace_id=trace_id,
        run_id=run_id,
        instance=instance,
        condition=condition,
        event_type="runtime_effect_observed",
        layer="A",
        component="communication_interceptor",
        status="effect_observed",
        effect=observed_a_symptom,
        label="propagated",
        evidence={"cid": cid},
        source=source,
        target=target,
        injection_point_kind="hierarchical_communication_interceptor",
    )


def write_hierarchical_delivery_outcome(
    events: Path,
    *,
    trace_id: str,
    run_id: str,
    instance: SWEInstance,
    condition: str,
    source: str,
    target: str,
    cid: dict[str, object],
    observed_a_symptom: str,
    delivered: bool,
) -> None:
    write_event(
        events,
        trace_id=trace_id,
        run_id=run_id,
        instance=instance,
        condition=condition,
        event_type="message_delivered" if delivered else "message_dropped",
        layer="A",
        component=target,
        status="delivered" if delivered else "dropped",
        effect=observed_a_symptom,
        label="propagated",
        evidence={"cid": cid},
        source=source,
        target=target,
        injection_point_kind="hierarchical_communication_interceptor",
    )


def run_hierarchical_instance(
    client: ChatClient,
    harness: SWEHarness,
    instance: SWEInstance,
    condition: str,
    output_dir: Path,
    stale: TestEvidence | None,
    repository_context: str,
    context_metadata: dict[str, int],
    *,
    repeat_index: int,
    injection_slot: str,
    stale_guidance: tuple[str, str] | None = None,
) -> tuple[dict[str, object], TestEvidence]:
    validate_hierarchical_condition_slot(condition, injection_slot)
    run_id = build_run_id(instance.instance_id, condition, repeat_index)
    trace_id = f"trace-{uuid.uuid4()}"
    events = output_dir / "swe_hierarchical_causal_events.jsonl"
    cid = build_hierarchical_cid(injection_slot, message_rank=6, eligible_messages=8)
    slot = HIERARCHICAL_INJECTION_SLOTS[injection_slot]
    calls_before, prompt_before, completion_before = client.call_count, client.prompt_tokens, client.completion_tokens
    started = time.perf_counter()
    write_event(
        events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition,
        event_type="workflow_started", layer="A", component="HierarchicalTeam", status="started",
        effect="hierarchical SWE-bench MAS started", label="pre_injection", evidence={"topology": "hierarchical"},
    )

    task_card = client.complete(build_planner_prompt(instance, repository_context))
    original_message: object | None = None
    delivered_message: object | None = None
    observed_a_symptom = "none"
    fault_applied = False
    plan_original, plan_delivered, symptom, applied = _hierarchical_text_delivery(task_card, condition, injection_slot, "manager_to_coder")
    if injection_slot == "manager_to_coder":
        original_message, delivered_message, observed_a_symptom, fault_applied = plan_original, plan_delivered, symptom, applied
        _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Manager", target="Coder", cid=cid, fault_applied=applied, observed_a_symptom=symptom, delivered=plan_delivered)
    plan_for_coder = plan_delivered or "[Manager task card was not delivered due to a communication fault.]"

    coder_initial = client.complete(build_coder_prompt(instance, plan_for_coder, repository_context), json_mode=True)
    candidate_patch, candidate_error, candidate_attempts = generate_valid_patch(client, instance, coder_initial, repository_context)

    patch_original, patch_delivered, symptom, applied = _hierarchical_text_delivery(candidate_patch, condition, injection_slot, "coder_to_manager")
    if injection_slot == "coder_to_manager":
        original_message, delivered_message, observed_a_symptom, fault_applied = patch_original, patch_delivered, symptom, applied
        _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Coder", target="Manager", cid=cid, fault_applied=applied, observed_a_symptom=symptom, delivered=patch_delivered)
    patch_for_workers = patch_delivered
    _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Manager", target="Tester", cid=cid, fault_applied=False, observed_a_symptom="none", delivered=patch_for_workers)
    _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Manager", target="Reviewer", cid=cid, fault_applied=False, observed_a_symptom="none", delivered=patch_for_workers)
    if candidate_error:
        initial_evidence = TestEvidence(instance.instance_id, False, instance.fail_to_pass, candidate_error, "agent_patch_invalid")
    elif patch_for_workers is None:
        initial_evidence = TestEvidence(instance.instance_id, False, instance.fail_to_pass, "candidate patch was not delivered to Tester", "communication_delivery_failed")
    else:
        initial_evidence = run_official_evaluation(
            harness, instance, patch_for_workers,
            dataset_name=str((output_dir / "swe_bench_verified_instances.json").resolve()),
            split="test", output_dir=output_dir, run_id=f"{run_id}-candidate",
        )

    tester_report = client.complete(build_hierarchical_tester_prompt(instance, patch_for_workers, initial_evidence), json_mode=True)
    reviewer_report = client.complete(build_hierarchical_reviewer_prompt(instance, task_card, patch_for_workers), json_mode=True)
    review_original, review_delivered, symptom, applied = _hierarchical_text_delivery(reviewer_report, condition, injection_slot, "reviewer_to_manager")
    if injection_slot == "reviewer_to_manager":
        original_message, delivered_message, observed_a_symptom, fault_applied = review_original, review_delivered, symptom, applied
        _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Reviewer", target="Manager", cid=cid, fault_applied=applied, observed_a_symptom=symptom, delivered=review_delivered)

    evidence_original, evidence_delivered, symptom, applied = _hierarchical_evidence_delivery(initial_evidence, condition, injection_slot, stale)
    if injection_slot == "tester_to_manager":
        original_message = evidence_original.__dict__
        delivered_message = None if evidence_delivered is None else evidence_delivered.__dict__
        observed_a_symptom, fault_applied = symptom, applied
        _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Tester", target="Manager", cid=cid, fault_applied=applied, observed_a_symptom=symptom, delivered=evidence_delivered)

    manager_guidance = client.complete(
        build_hierarchical_manager_prompt(instance, tester_report, evidence_delivered, review_delivered),
        json_mode=True,
    )
    manager_assessment = parse_hierarchical_revision_assessment(manager_guidance)
    delivered_manager_guidance = manager_guidance
    if injection_slot == "manager_guidance_to_coder" and condition == "a12_guidance_stale_replay":
        if stale_guidance is None:
            raise RuntimeError("a12_guidance_stale_replay requires prior clean guidance")
        stale_message, stale_task_id = stale_guidance
        intercepted = replay_stale_guidance(manager_guidance, stale_message, stale_task_id=stale_task_id)
        original_message, delivered_message = intercepted.original, intercepted.delivered
        observed_a_symptom, fault_applied = intercepted.observed_a_symptom, True
        delivered_manager_guidance = intercepted.delivered
        _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Manager", target="Coder", cid=cid, fault_applied=True, observed_a_symptom=observed_a_symptom, delivered=delivered_manager_guidance)
    else:
        _write_hierarchical_delivery(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, source="Manager", target="Coder", cid=cid, fault_applied=False, observed_a_symptom="none", delivered=manager_guidance)

    revision_raw = client.complete(
        build_hierarchical_coder_revision_prompt(
            instance, task_card, candidate_patch, delivered_manager_guidance, None, None, repository_context,
        ),
        json_mode=True,
    )
    coder_assessment = parse_hierarchical_revision_assessment(revision_raw)
    final_patch, final_patch_error, final_attempts = generate_valid_patch(client, instance, revision_raw, repository_context)
    final_patch_fallback = False
    if final_patch_error and candidate_patch:
        final_patch, final_patch_fallback = candidate_patch, True
        final_patch_error = None
    if final_patch_error:
        final_evidence = TestEvidence(instance.instance_id, False, instance.fail_to_pass, final_patch_error, "agent_patch_invalid")
    else:
        final_evidence = run_official_evaluation(
            harness, instance, final_patch,
            dataset_name=str((output_dir / "swe_bench_verified_instances.json").resolve()),
            split="test", output_dir=output_dir, run_id=f"{run_id}-final",
        )
    outcome = evaluate_hierarchical_run(condition=condition, final_evidence=final_evidence, coder_assessment=coder_assessment if injection_slot == "manager_guidance_to_coder" else manager_assessment)
    for consequence in outcome["observed_M_consequence"]:
        if consequence != "none":
            write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="m_consequence_observed", layer="M", component="task_evaluator", status="observed", effect=str(consequence), label="propagated", evidence={"cid": cid})
    write_event(
        events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition,
        event_type="final_consequence", layer="M", component="task_evaluator",
        status="preserved" if outcome["final_task_success"] else "failed",
        effect="hierarchical final official evaluation", label="pre_injection" if condition == "clean" else "propagated",
        evidence={"task_success": outcome["final_task_success"], "mas_consequence": outcome["observed_M_consequence"]},
    )
    row: dict[str, object] = {
        "run_id": run_id, "trace_id": trace_id, "benchmark": "SWE-bench Verified", "topology": "hierarchical",
        "execution_mode": "official_swe_harness", "scenario": "code_repair_hierarchical_peer_review", "task_id": instance.instance_id,
        "instance_id": instance.instance_id, "condition": condition, "fault_type": condition,
        "fault_severity": "none" if condition == "clean" else "default", "fault_parameters": {},
        "model": client.model_info.model, "provider": client.model_info.provider,
        "fault_id": "none" if condition == "clean" else f"fault-{run_id}", "fault_applied": fault_applied,
        "source_agent": slot.source, "target_agent": slot.target, "injection_slot": injection_slot,
        "cid": cid, "first_divergence": "A:fault_applied" if fault_applied else "none",
        "observed_runtime_effect": observed_a_symptom, "original_message": original_message,
        "delivered_message": delivered_message, "planner_output": task_card, "candidate_patch": candidate_patch,
        "coder_initial_output": coder_initial, "tester_report": tester_report, "reviewer_report": reviewer_report,
        "manager_guidance": manager_guidance, "manager_assessment": manager_assessment,
        "delivered_manager_guidance": delivered_manager_guidance,
        "coder_revision_output": revision_raw, "coder_assessment": coder_assessment, "candidate_patch_attempts": candidate_attempts,
        "final_patch": final_patch, "final_patch_attempts": final_attempts, "final_patch_used_candidate_fallback": final_patch_fallback,
        "context_metadata": context_metadata, "expected_answer": {"resolved": True, "instance_id": instance.instance_id},
        "final_answer": {"official_test_passed": final_evidence.passed, "manager_assessment": manager_assessment},
        **outcome, "official_test_passed": final_evidence.passed, "official_runner_error": final_evidence.runner_error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "api_call_count": client.call_count - calls_before,
        "prompt_tokens": client.prompt_tokens - prompt_before,
        "completion_tokens": client.completion_tokens - completion_before,
    }
    row["total_tokens"] = int(row["prompt_tokens"]) + int(row["completion_tokens"])
    return normalize_run_record(row), initial_evidence


def render_hierarchical_summary_markdown(summary: dict[str, object]) -> str:
    return (
        "# SWE-bench Hierarchical 通信故障实验汇总\n\n"
        f"- 运行数：{summary['runs']}\n"
        f"- 官方测试通过：{summary['official_test_passed']}\n"
        f"- MAS 最终成功：{summary['final_success']}\n"
        f"- A 层暴露：{summary['a_exposure']}\n"
        f"- M 层传播：{summary['m_propagation']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real LLM SWE-bench Hierarchical topology communication-fault experiments.")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--instance-ids", nargs="*", default=[])
    parser.add_argument("--conditions", nargs="+", default=["clean", "a1_deadline_delay", "a5_omission", "a8_truncation"])
    parser.add_argument("--injection-slot", choices=sorted(HIERARCHICAL_INJECTION_SLOTS), default="tester_to_manager")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", default="results/swe_bench_hierarchical_real")
    parser.add_argument("--dataset-dir", default="data/modelscope_swe_bench_verified")
    parser.add_argument("--context-budget-chars", type=int, default=8000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for condition in args.conditions:
        validate_hierarchical_condition_slot(condition, args.injection_slot)
    load_dotenv(); load_dotenv(".env.local")
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, resume=args.resume)
    client, harness = get_llm_client(), SWEHarness()
    candidates, raw_instances = load_verified_instances(Path(args.dataset_dir), max(500, args.instances * 4))
    if args.instance_ids:
        wanted = set(args.instance_ids)
        candidates = [item for item in candidates if item.instance_id in wanted]
        missing = wanted - {item.instance_id for item in candidates}
        if missing:
            raise ValueError(f"requested SWE instances not found: {', '.join(sorted(missing))}")
    instances, skipped = select_runnable_instances(candidates, args.instances, image_exists=local_image_exists)
    if not instances:
        raise SystemExit("no selected SWE instances have a local evaluation image")
    selected_ids = {item.instance_id for item in instances}
    (output_dir / "swe_bench_verified_instances.json").write_text(json.dumps([row for row in raw_instances if row["instance_id"] in selected_ids], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "instance_selection.json").write_text(json.dumps({"selected": sorted(selected_ids), "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    config = build_hierarchical_experiment_config([item.instance_id for item in instances], args.conditions, args.repeats, client.model_info.model, client.model_info.provider, args.injection_slot)
    (output_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contexts = {item.instance_id: collect_repository_context_with_metadata(item, ContextBudget(max_total_chars=args.context_budget_chars)) for item in instances}
    rows_path = output_dir / "swe_hierarchical_runs.jsonl"
    rows = load_completed_run_rows(rows_path) if args.resume else []
    completed = {str(row.get("run_id", "")) for row in rows}
    evidence_by_clean_run: dict[tuple[str, int], TestEvidence] = {}
    guidance_by_clean_run: dict[tuple[str, int], str] = {
        (str(row.get("instance_id")), int(row.get("repeat_index", 1))): str(row["manager_guidance"])
        for row in rows if row.get("condition") == "clean" and row.get("manager_guidance")
    }
    by_id = {item.instance_id: item for item in instances}
    for instance_id, condition, repeat_index, stale_source_id in build_execution_schedule([item.instance_id for item in instances], args.conditions, repeats=args.repeats):
        run_id = build_run_id(instance_id, condition, repeat_index)
        if run_id in completed:
            continue
        stale = evidence_by_clean_run.get((str(stale_source_id), repeat_index)) if condition in STALE_EVIDENCE_CONDITIONS else None
        if condition in STALE_EVIDENCE_CONDITIONS and stale is None:
            raise RuntimeError(f"no prior clean Hierarchical evidence available for {condition} on {instance_id}")
        stale_guidance = None
        if condition == "a12_guidance_stale_replay":
            candidates = [(task_id, guidance) for (task_id, index), guidance in guidance_by_clean_run.items() if index == repeat_index and task_id != instance_id]
            if not candidates:
                raise RuntimeError("a12_guidance_stale_replay requires at least two clean task guidances")
            stale_task_id, guidance = candidates[0]
            stale_guidance = (guidance, stale_task_id)
        context, metadata = contexts[instance_id]
        row, initial_evidence = run_hierarchical_instance(client, harness, by_id[instance_id], condition, output_dir, stale, context, metadata, repeat_index=repeat_index, injection_slot=args.injection_slot, stale_guidance=stale_guidance)
        row["repeat_index"] = repeat_index
        rows.append(row)
        append_jsonl_record(rows_path, row)
        if condition == "clean":
            evidence_by_clean_run[(instance_id, repeat_index)] = initial_evidence
            guidance_by_clean_run[(instance_id, repeat_index)] = str(row["manager_guidance"])
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "swe_hierarchical_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    summary = {
        "runs": len(rows), "official_test_passed": sum(bool(row["official_test_passed"]) for row in rows),
        "final_success": sum(bool(row["final_task_success"]) for row in rows),
        "a_exposure": sum(row["observed_A_symptom"] != ["none"] for row in rows),
        "m_propagation": sum(row["observed_M_consequence"] != ["none"] for row in rows),
    }
    (output_dir / "swe_hierarchical_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "swe_hierarchical_summary.md").write_text(render_hierarchical_summary_markdown(summary), encoding="utf-8")
    condition_summary = swe_condition_summary_rows(rows)
    with (output_dir / "swe_hierarchical_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_summary[0]) if condition_summary else ["condition"])
        writer.writeheader(); writer.writerows(condition_summary)
    build_report(output_dir / "swe_hierarchical_causal_events.jsonl", output_dir / "causal_trace_report")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
