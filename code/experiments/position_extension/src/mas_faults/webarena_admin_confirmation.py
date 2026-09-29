"""Controlled WebArena Admin workflow for the frozen main matrix."""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogen_core import AgentId, SingleThreadedAgentRuntime

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.webarena_admin_controlled import (
    AdminAgentMessage,
    ControlledEvidenceAgent,
    ControlledPlannerAgent,
    ControlledRunError,
    ControlledToolNavigatorAgent,
    _ask,
    agent_task_view,
    build_structured_numeric_evidence,
    build_structured_task_evidence,
    parse_evidence_result,
    write_sanitized_browser_config,
)
from mas_faults.webarena_admin_fault_matrix import make_evidence_envelope
from mas_faults.webarena_admin_main_matrix import (
    MainCommunicationInterceptor,
    MainConditionCell,
    MainDeliveryBatch,
)
from mas_faults.webarena_admin_main_evaluator import evaluate_main_outcome
from mas_faults.webarena_admin_tools import (
    parse_admin_tool_request,
    validate_admin_tool_call,
)
from mas_faults.webarena_admin_topologies import (
    TOPOLOGY_POLICIES,
    run_decision_topology,
)


STEP2_ELIGIBLE_TOOLS = frozenset(
    {
        "set_date_range",
        "set_select_filter",
        "set_range_filter",
        "set_text_filter",
        "read_visible_table",
    }
)

TASK_EVIDENCE_COLUMNS = {
    "sales_ranking": ("Interval", "Product", "Order Quantity"),
    "sales_product_ranking": ("Interval", "Product", "Order Quantity"),
    "sales_report_aggregation": (
        "Interval",
        "Product",
        "Order Quantity",
        "Orders",
    ),
    "temporal_sales_aggregation": ("Interval", "Orders"),
    "inventory_attribute_lookup": ("SKU", "Name", "Quantity"),
    "order_payment_aggregation": (
        "ID",
        "Purchase Date",
        "Grand Total",
        "Grand Total (Base)",
        "Status",
    ),
    "customer_order_aggregation": (
        "ID",
        "Purchase Date",
        "Bill-to Name",
        "Status",
    ),
    "customer_contact_lookup": ("Name", "Email", "Phone"),
    "order_state_lookup": (
        "ID",
        "Purchase Date",
        "Bill-to Name",
        "Ship-to Name",
        "Status",
    ),
    "customer_cancellation_aggregation": (
        "ID",
        "Purchase Date",
        "Bill-to Name",
        "Grand Total",
        "Status",
    ),
}


def project_visible_evidence(
    task: dict[str, Any], visible_evidence: list[list[str]]
) -> list[list[str]]:
    """Keep task-relevant columns without selecting rows or an answer."""
    if not visible_evidence:
        return []
    header = visible_evidence[0]
    requested = TASK_EVIDENCE_COLUMNS.get(str(task.get("task_stratum")), ())
    indices = [header.index(name) for name in requested if name in header]
    if not indices:
        return visible_evidence
    return [
        [row[index] if index < len(row) else "" for index in indices]
        for row in visible_evidence
    ]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_event(
    *,
    run_id: str,
    trace_id: str,
    event_index: int,
    abstract_step: int | None,
    source_agent: str,
    target_agent: str,
    original_message: Any,
    delivered_messages: tuple[Any, ...] | list[Any],
    effect: str,
    message_id: str | None = None,
    delivery: MainDeliveryBatch | None = None,
) -> dict[str, Any]:
    delivered_list = list(delivered_messages)
    timestamp = delivery.send_timestamp if delivery else _timestamp()
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "timestamp": timestamp,
        "step_index": event_index,
        "step_id": f"step-{event_index:03d}",
        "abstract_step": abstract_step,
        "message_id": message_id or f"{run_id}:step-{event_index:03d}",
        "source_agent": source_agent,
        "target_agent": target_agent,
        "original_message": deepcopy(original_message),
        "delivered_message": (
            deepcopy(delivered_list[-1]) if delivered_list else None
        ),
        "delivered_messages": deepcopy(delivered_list),
        "delivery_count": len(delivered_list),
        "fault_id": delivery.fault_id if delivery else "none",
        "fault_type": delivery.fault_type if delivery else "clean",
        "fault_family": delivery.fault_family if delivery else "clean",
        "fault_cause": delivery.fault_cause if delivery else "none",
        "fault_applied": delivery.fault_applied if delivery else False,
        "fault_parameters": deepcopy(delivery.fault_parameters) if delivery else {},
        "send_timestamp": timestamp,
        "delivery_timestamp": (
            delivery.delivery_timestamps[-1]
            if delivery and delivery.delivery_timestamps
            else (timestamp if delivered_list else None)
        ),
        "delivery_timestamps": (
            list(delivery.delivery_timestamps) if delivery else [timestamp]
        ),
        "observed_runtime_effect": effect,
        "observed_A_symptom": (
            delivery.observed_a_symptom if delivery else "none"
        ),
        "observed_M_consequence": ["none"],
    }


def _usage(client: Any) -> tuple[int, int, int]:
    return (
        int(getattr(client, "call_count", 0)),
        int(getattr(client, "prompt_tokens", 0)),
        int(getattr(client, "completion_tokens", 0)),
    )


async def run_admin_confirmation_task(
    client: Any,
    browser: Any,
    evaluator: Any,
    task: dict[str, Any],
    *,
    original_config_file: Path,
    sanitized_config_dir: Path,
    topology: str,
    condition_cell: MainConditionCell,
    run_index: int,
    stale_message: dict[str, Any] | None = None,
    max_steps: int = 10,
    max_evidence_chars: int = 64000,
) -> dict[str, Any]:
    """Run one valid-or-auditable main-matrix attempt."""
    if topology not in TOPOLOGY_POLICIES:
        raise ValueError(f"unknown topology: {topology}")
    run_id = (
        f"admin-main-{topology}-{task['task_id']}-r{run_index}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    trace_id = f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = _usage(client)
    request_log_before = len(getattr(client, "request_log", []))
    task_view = agent_task_view(task)
    interceptor = MainCommunicationInterceptor(
        condition_cell, stale_message=stale_message
    )
    sanitized = write_sanitized_browser_config(
        Path(original_config_file), sanitized_config_dir
    )
    events: list[dict[str, Any]] = []
    event_index = 0
    runtime = SingleThreadedAgentRuntime()
    planner_id = AgentId("admin_main_planner", "default")
    navigator_id = AgentId("admin_main_navigator", "default")
    evidence_id = AgentId("admin_main_evidence", "default")
    await runtime.register_agent_instance(ControlledPlannerAgent(client), planner_id)
    await runtime.register_agent_instance(
        ControlledToolNavigatorAgent(client), navigator_id
    )
    await runtime.register_agent_instance(ControlledEvidenceAgent(client), evidence_id)
    runtime.start()

    authoritative_state: dict[str, Any] = {}
    consumed_state: dict[str, Any] = {}
    accepted_visible_evidence: list[list[str]] = []
    history: list[dict[str, Any]] = []
    state_version = 0
    termination_reason = "max_steps"
    topology_result = None
    try:
        authoritative_state = browser.reset(str(sanitized.path))
        authoritative_state["state_version"] = state_version
        consumed_state = deepcopy(authoritative_state)
        event_index += 1
        events.append(
            _trace_event(
                run_id=run_id,
                trace_id=trace_id,
                event_index=event_index,
                abstract_step=3,
                source_agent="WebArena Tool Worker",
                target_agent="Tool Navigator",
                original_message={"sanitized_config_sha256": sanitized.sha256},
                delivered_messages=(
                    {
                        "url": consumed_state.get("url", ""),
                        "title": consumed_state.get("title", ""),
                        "state_version": state_version,
                    },
                ),
                effect="sanitized_environment_reset",
            )
        )

        plan_message = await _ask(
            runtime,
            planner_id,
            AdminAgentMessage("task", {"task": task_view}),
        )
        plan = str(plan_message.payload["plan"])
        event_index += 1
        events.append(
            _trace_event(
                run_id=run_id,
                trace_id=trace_id,
                event_index=event_index,
                abstract_step=1,
                source_agent="Planner",
                target_agent="Tool Navigator",
                original_message={"task": task_view},
                delivered_messages=({"plan": plan},),
                effect="plan_handoff",
            )
        )

        for _ in range(max_steps):
            navigator_message = await _ask(
                runtime,
                navigator_id,
                AdminAgentMessage(
                    "navigate",
                    {
                        "task": task_view,
                        "plan": plan,
                        "page": {
                            "url": consumed_state.get("url", ""),
                            "title": consumed_state.get("title", ""),
                            "available_tools": consumed_state.get(
                                "available_tools", []
                            ),
                        },
                        "history": history,
                        "evidence_row_count": len(accepted_visible_evidence),
                    },
                ),
            )
            raw_request = str(navigator_message.payload["raw"])
            try:
                request = parse_admin_tool_request(raw_request)
            except ValueError:
                termination_reason = "tool_request_parse_failure"
                event_index += 1
                events.append(
                    _trace_event(
                        run_id=run_id,
                        trace_id=trace_id,
                        event_index=event_index,
                        abstract_step=2,
                        source_agent="Tool Navigator",
                        target_agent="Tool Protocol Validator",
                        original_message=raw_request,
                        delivered_messages=(),
                        effect=termination_reason,
                    )
                )
                raise
            request_message = {
                "tool": request.name,
                "arguments": request.arguments,
            }
            step2_delivery = interceptor.intercept(
                2,
                request_message,
                context={"eligible": request.name in STEP2_ELIGIBLE_TOOLS},
            )
            event_index += 1
            events.append(
                _trace_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    event_index=event_index,
                    abstract_step=2,
                    source_agent="Tool Navigator",
                    target_agent="WebArena Tool Worker",
                    original_message=request_message,
                    delivered_messages=step2_delivery.delivered_messages,
                    effect=step2_delivery.observed_runtime_effect,
                    delivery=step2_delivery,
                )
            )

            if not step2_delivery.delivered_messages:
                history.append(
                    {
                        "tool": request.name,
                        "arguments": request.arguments,
                        "status": "not_delivered",
                        "url": consumed_state.get("url", ""),
                        "evidence_rows": len(accepted_visible_evidence),
                    }
                )
                continue

            for delivered_request in step2_delivery.delivered_messages:
                validated = validate_admin_tool_call(
                    delivered_request.get("tool"),
                    delivered_request.get("arguments"),
                )
                available = consumed_state.get("available_tools")
                if isinstance(available, list) and validated.name not in available:
                    termination_reason = "tool_not_available_in_delivered_state"
                    raise ValueError(
                        f"tool {validated.name!r} is unavailable in delivered state"
                    )
                authoritative_state = browser.tool(
                    validated.name, validated.arguments
                )
                state_version += 1
                authoritative_state["state_version"] = state_version
                current_evidence = authoritative_state.get("visible_evidence") or []
                step3_delivery = interceptor.intercept(
                    3,
                    authoritative_state,
                    context={
                        "eligible": bool(current_evidence and consumed_state),
                        "older_message": consumed_state,
                    },
                )
                event_index += 1
                events.append(
                    _trace_event(
                        run_id=run_id,
                        trace_id=trace_id,
                        event_index=event_index,
                        abstract_step=3,
                        source_agent="WebArena Tool Worker",
                        target_agent="Tool Navigator",
                        original_message=authoritative_state,
                        delivered_messages=step3_delivery.delivered_messages,
                        effect=step3_delivery.observed_runtime_effect,
                        delivery=step3_delivery,
                    )
                )
                if step3_delivery.delivered_messages:
                    consumed_state = deepcopy(step3_delivery.delivered_messages[-1])
                consumed_evidence = consumed_state.get("visible_evidence") or []
                if consumed_evidence:
                    accepted_visible_evidence = project_visible_evidence(
                        task, consumed_evidence
                    )
                history.append(
                    {
                        "tool": validated.name,
                        "arguments": validated.arguments,
                        "status": consumed_state.get("tool_status", "unknown"),
                        "url": consumed_state.get("url", ""),
                        "evidence_rows": len(consumed_evidence),
                        "state_version": consumed_state.get("state_version"),
                    }
                )

            if request.name == "finish_with_evidence":
                termination_reason = "navigator_finish"
                break

        encoded_evidence = json.dumps(accepted_visible_evidence, ensure_ascii=False)
        if len(encoded_evidence) > max_evidence_chars:
            termination_reason = "evidence_prompt_budget_exceeded"
            raise ValueError(
                f"projected visible evidence exceeds prompt budget: {len(encoded_evidence)}"
            )
        evidence_message = await _ask(
            runtime,
            evidence_id,
            AdminAgentMessage(
                "extract",
                {
                    "task": task_view,
                    "visible_evidence": accepted_visible_evidence,
                },
            ),
        )
        evidence_result = parse_evidence_result(
            str(evidence_message.payload["raw"])
        )
        envelope = make_evidence_envelope(
            message_id=f"{run_id}:evidence-handoff",
            task_id=str(task["task_id"]),
            source_session=run_id,
            state_version=state_version,
            payload={
                "evidence_result": evidence_result,
                "visible_evidence": accepted_visible_evidence,
                "structured_numeric_evidence": build_structured_numeric_evidence(
                    accepted_visible_evidence
                ),
                "structured_task_evidence": build_structured_task_evidence(
                    accepted_visible_evidence
                ),
            },
        )
        step4_delivery = interceptor.intercept(
            4, envelope, context={"eligible": True}
        )
        step4_target = TOPOLOGY_POLICIES[topology].declared_edges[0][1]
        event_index += 1
        events.append(
            _trace_event(
                run_id=run_id,
                trace_id=trace_id,
                event_index=event_index,
                abstract_step=4,
                source_agent="Evidence Worker",
                target_agent=step4_target,
                original_message=envelope,
                delivered_messages=step4_delivery.delivered_messages,
                effect=step4_delivery.observed_runtime_effect,
                message_id=envelope["message_id"],
                delivery=step4_delivery,
            )
        )

        direct_delivery = None
        if topology == "flat":
            direct_delivery = interceptor.intercept(
                4,
                envelope,
                context={"eligible": False, "branch": "direct"},
            )
            event_index += 1
            events.append(
                _trace_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    event_index=event_index,
                    abstract_step=4,
                    source_agent="Evidence Worker",
                    target_agent="Coordinator",
                    original_message=envelope,
                    delivered_messages=direct_delivery.delivered_messages,
                    effect=direct_delivery.observed_runtime_effect,
                    message_id=f"{envelope['message_id']}:direct",
                    delivery=direct_delivery,
                )
            )

        topology_result = await run_decision_topology(
            client,
            topology=topology,
            task=task_view,
            delivery=step4_delivery,
            direct_delivery=direct_delivery,
        )
        for source, target in topology_result.used_edges[1:]:
            if (source, target) == ("Evidence Worker", "Coordinator"):
                continue
            event_index += 1
            delivered_value: Any = topology_result.final_decision
            if source in {"Verifier", "Supervisor"}:
                delivered_value = topology_result.verification
            events.append(
                _trace_event(
                    run_id=run_id,
                    trace_id=trace_id,
                    event_index=event_index,
                    abstract_step=4,
                    source_agent=source,
                    target_agent=target,
                    original_message=delivered_value,
                    delivered_messages=(delivered_value,),
                    effect="topology_message_delivery",
                )
            )

        final_decision = topology_result.final_decision
        evaluation = evaluator.evaluate(
            str(original_config_file), final_decision["answer"]
        )
        score = float(evaluation["score"])
        final_success = score == 1.0
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        after = _usage(client)
        fault_event = next(
            (event for event in events if event.get("fault_applied")), None
        )
        injection_valid = condition_cell.condition == "clean" or fault_event is not None
        record = {
            "run_id": run_id,
            "trace_id": trace_id,
            "scenario": "webarena_shopping_admin_main_confirmation",
            "dataset": "WebArena",
            "benchmark": "WebArena Shopping Admin",
            "framework": "AutoGen",
            "model": client.model_info.model,
            "provider": client.model_info.provider,
            "topology": topology,
            "declared_edges": [list(edge) for edge in topology_result.declared_edges],
            "used_edges": [list(edge) for edge in topology_result.used_edges],
            "task_id": str(task["task_id"]),
            "task_stratum": task["task_stratum"],
            "intent": task["intent"],
            "condition": condition_cell.condition,
            "fault_family": condition_cell.fault_family,
            "injection_step": condition_cell.injection_step,
            "fault_id": fault_event.get("fault_id", "none") if fault_event else "none",
            "fault_type": fault_event.get("fault_type", "clean") if fault_event else "clean",
            "fault_cause": fault_event.get("fault_cause", "none") if fault_event else "none",
            "fault_severity": condition_cell.severity,
            "fault_parameters": condition_cell.parameters,
            "fault_applied": bool(fault_event),
            "injection_valid": injection_valid,
            "source_agent": fault_event.get("source_agent") if fault_event else "none",
            "target_agent": fault_event.get("target_agent") if fault_event else "none",
            "original_message": fault_event.get("original_message") if fault_event else None,
            "delivered_message": fault_event.get("delivered_message") if fault_event else None,
            "first_divergence": (
                f"step_{fault_event['abstract_step']}:{condition_cell.condition}"
                if fault_event
                else "none"
            ),
            "observed_runtime_effect": (
                fault_event.get("observed_runtime_effect") if fault_event else "clean"
            ),
            "observed_A_symptom": [
                fault_event.get("observed_A_symptom", "none")
                if fault_event
                else "none"
            ],
            "observed_M_consequence": ["none"],
            "system_consequences": ["none"],
            "semantic_consequences": ["none"],
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": list(topology_result.recovery_evidence),
            "propagation_class": "masked" if fault_event else "clean",
            "expected_answer": task.get("eval", {}).get("reference_answers", {}),
            "final_answer": final_decision,
            "verification": topology_result.verification,
            "task_score": score,
            "final_task_success": final_success,
            "termination_reason": termination_reason,
            "accepted_visible_evidence": accepted_visible_evidence,
            "structured_task_evidence": build_structured_task_evidence(
                accepted_visible_evidence
            ),
            "sanitized_config_sha256": sanitized.sha256,
            "real_site": True,
            "official_task_config": True,
            "official_final_answer_evaluator": True,
            "final_evaluator_mode": evaluation.get(
                "evaluator_mode", "official_or_test_evaluator"
            ),
            "official_evaluator_input": "final_stop_only",
            "process_trajectory_type": "controlled_tool_trace",
            "latency_ms": elapsed_ms,
            "api_call_count": after[0] - before[0],
            "prompt_tokens": after[1] - before[1],
            "completion_tokens": after[2] - before[2],
            "total_tokens": (after[1] - before[1]) + (after[2] - before[2]),
            "llm_calls": list(getattr(client, "request_log", []))[
                request_log_before:
            ],
            "error": None,
            "events": events,
            "topology_recovery_evidence": list(topology_result.recovery_evidence),
            "axis_evaluation_mode": "strict_evidence_v1",
        }
        evaluated_record = evaluate_main_outcome(record)
        return normalize_run_record(
            evaluated_record, axis_evaluation_mode="strict_evidence_v1"
        )
    except Exception as exc:
        raise ControlledRunError(
            exc,
            events=events,
            termination_reason=termination_reason,
            browser_state=authoritative_state,
        ) from exc
    finally:
        await runtime.stop()
