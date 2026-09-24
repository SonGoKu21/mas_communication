"""Controlled-tool MAS support for real WebArena Shopping Admin tasks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.webarena_admin_real import AdminAgentMessage
from mas_faults.webarena_admin_fault_matrix import (
    AdminCommunicationInterceptor,
    build_admin_outcome,
    make_evidence_envelope,
)
from mas_faults.webarena_admin_tools import parse_admin_tool_request, tool_catalog_text


AGENT_TASK_FIELDS = ("task_id", "task_stratum", "sites", "intent")


@dataclass(frozen=True)
class SanitizedBrowserConfig:
    path: Path
    sha256: str


class ControlledRunError(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        *,
        events: list[dict[str, Any]],
        termination_reason: str,
        browser_state: dict[str, Any],
    ) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.events = list(events)
        self.termination_reason = termination_reason
        self.browser_state = dict(browser_state)


class EvaluatorWorkerClient:
    """JSONL client for the reference-answer-only evaluator capability domain."""

    def __init__(
        self,
        *,
        process: Any | None = None,
        python_executable: str = "/data2/system5/mas/venvs/webarena/bin/python",
        worker_script: str = "/home/systemai_5/code/mas/scripts/smoke/webarena_admin_evaluator_worker.py",
        webarena_root: str = "/data2/system5/mas/third_party/webarena",
        env: dict[str, str] | None = None,
    ) -> None:
        if process is None:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            process = subprocess.Popen(
                [
                    python_executable,
                    worker_script,
                    "--webarena-root",
                    webarena_root,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=process_env,
            )
        self.process = process

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"WebArena evaluator worker exited: {stderr[-1000:]}")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("WebArena evaluator worker has no JSONL streams")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(
                f"WebArena evaluator worker returned no response: {stderr[-1000:]}"
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                f"WebArena evaluator worker {response.get('error_type', 'Error')}: "
                f"{response.get('error', 'unknown error')}"
            )
        return {key: value for key, value in response.items() if key != "ok"}

    def evaluate(self, config_file: str, answer: str) -> dict[str, Any]:
        return self._request(
            {
                "command": "evaluate",
                "config_file": config_file,
                "answer": answer,
            }
        )

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"command": "close"})
            finally:
                if self.process.poll() is None:
                    self.process.terminate()


def agent_task_view(task: dict[str, Any]) -> dict[str, Any]:
    """Construct the only task DTO permitted to cross an Agent boundary."""
    missing = [field for field in AGENT_TASK_FIELDS if field not in task]
    if missing:
        raise ValueError(f"task is missing Agent fields: {missing}")
    return {field: task[field] for field in AGENT_TASK_FIELDS}


def write_sanitized_browser_config(
    original_path: Path, output_dir: Path
) -> SanitizedBrowserConfig:
    """Create a browser config whose capability domain contains no evaluator data."""
    original_path = original_path.resolve()
    payload = json.loads(original_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("WebArena config must be a JSON object")
    sanitized = {key: value for key, value in payload.items() if key != "eval"}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{original_path.stem}.browser.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SanitizedBrowserConfig(path=path, sha256=digest)


def _strict_json_object(raw: str, expected_keys: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("model output is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("model JSON does not match the required schema")
    return value


def parse_evidence_result(raw: str) -> dict[str, Any]:
    value = _strict_json_object(
        raw,
        {"candidate_answer", "evidence_summary", "evidence_row_indices"},
    )
    if not isinstance(value["candidate_answer"], str):
        raise ValueError("candidate_answer must be a string")
    if not isinstance(value["evidence_summary"], str):
        raise ValueError("evidence_summary must be a string")
    indices = value["evidence_row_indices"]
    if not isinstance(indices, list) or any(
        not isinstance(index, int) or isinstance(index, bool) for index in indices
    ):
        raise ValueError("evidence_row_indices must contain integers")
    return value


def parse_coordinator_decision(raw: str) -> dict[str, str]:
    value = _strict_json_object(raw, {"decision", "answer", "reason"})
    if value["decision"] not in {"accept", "reject"}:
        raise ValueError("coordinator decision must be accept or reject")
    if not isinstance(value["answer"], str) or not isinstance(value["reason"], str):
        raise ValueError("coordinator answer and reason must be strings")
    return value


def _complete_for_role(
    client: Any,
    prompt: str,
    *,
    role: str,
    json_mode: bool = False,
) -> str:
    method = getattr(client, "complete_with_metadata", None)
    if callable(method):
        return str(
            method(
                prompt,
                json_mode=json_mode,
                metadata={"agent_role": role},
            )
        )
    return str(client.complete(prompt, json_mode=json_mode))


class ControlledPlannerAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("Shopping Admin Controlled Planner")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        task = message.payload["task"]
        prompt = (
            "You are the Planner in a read-only WebArena Shopping Admin MAS. "
            "Write a short plan using only information in the task. Do not answer the task. "
            "The Tool Navigator will choose controlled read-only browser tools.\n"
            f"TASK: {task['intent']}"
        )
        return AdminAgentMessage(
            "plan",
            {"plan": _complete_for_role(self.client, prompt, role="Planner").strip()},
        )


class ControlledToolNavigatorAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("Shopping Admin Controlled Tool Navigator")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        prompt = (
            "You are the Tool Navigator in a read-only WebArena Shopping Admin MAS. "
            "Choose exactly one next tool. Return strict JSON only with exactly the keys tool and "
            "arguments. The arguments must be a JSON object, never a JSON-encoded string. "
            "Zero-argument tools must use exactly an empty object and must not carry parameters for a later tool. "
            "Examples: {\"tool\":\"open_bestsellers_report\",\"arguments\":{}} and "
            "{\"tool\":\"open_sales_orders_report\",\"arguments\":{}}. "
            "Open the destination page first; send filters or dates only in a later turn when that tool is listed. "
            "Do not use Markdown fences. Do not guess the final answer. Call read_visible_table "
            "before finish_with_evidence. finish_with_evidence has no side effects and only signals that "
            "the current explicit evidence is ready. Use read_table_head instead of a full table read only "
            "for newest or most-recent tasks. Never call read_table_head after read_visible_table because it "
            "would replace complete evidence with a subset. A tool error ends the run; there is no retry.\n"
            f"TOOLS:\n{tool_catalog_text(available_tools=payload['page']['available_tools'])}\n"
            f"TASK: {payload['task']['intent']}\n"
            f"PLAN: {payload['plan']}\n"
            f"PAGE: {json.dumps(payload['page'], ensure_ascii=False)}\n"
            f"HISTORY: {json.dumps(payload['history'], ensure_ascii=False)}\n"
            f"CURRENT EVIDENCE ROWS: {payload['evidence_row_count']}"
        )
        return AdminAgentMessage(
            "tool_completion",
            {
                "raw": _complete_for_role(
                    self.client,
                    prompt,
                    role="Tool Navigator",
                    json_mode=True,
                ).strip()
            },
        )


def build_structured_numeric_evidence(
    visible_evidence: list[list[str]],
) -> dict[str, Any]:
    """Derive auditable sums from named columns without ranking or answer selection."""
    if not visible_evidence:
        return {}
    header = visible_evidence[0]
    if "Product" not in header or "Order Quantity" not in header:
        return {}
    group_index = header.index("Product")
    value_index = header.index("Order Quantity")
    totals: dict[str, float] = {}
    row_indices: dict[str, list[int]] = {}
    for index, row in enumerate(visible_evidence[1:], start=1):
        if max(group_index, value_index) >= len(row):
            continue
        group = row[group_index].strip()
        raw_value = row[value_index].strip().replace(",", "")
        if not group:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        totals[group] = totals.get(group, 0.0) + value
        row_indices.setdefault(group, []).append(index)
    return {
        "operation": "sum_by_group",
        "group_column": "Product",
        "value_column": "Order Quantity",
        "groups": [
            {
                "group": group,
                "total": total,
                "row_indices": row_indices[group],
            }
            for group, total in totals.items()
        ],
    }


def build_structured_task_evidence(
    visible_evidence: list[list[str]],
) -> dict[str, Any]:
    """Derive generic auditable aggregations without selecting a task answer."""
    if not visible_evidence:
        return {"columns": [], "row_count": 0, "aggregations": []}

    header = visible_evidence[0]
    rows = visible_evidence[1:]
    result: dict[str, Any] = {
        "columns": list(header),
        "row_count": len(rows),
        "aggregations": [],
    }

    numeric = build_structured_numeric_evidence(visible_evidence)
    if numeric:
        result["aggregations"].append(numeric)

    if "Bill-to Name" in header:
        group_index = header.index("Bill-to Name")
        counts: Counter[str] = Counter()
        row_indices: dict[str, list[int]] = {}
        for index, row in enumerate(rows, start=1):
            if group_index >= len(row):
                continue
            group = row[group_index].strip()
            if not group:
                continue
            counts[group] += 1
            row_indices.setdefault(group, []).append(index)
        result["aggregations"].append(
            {
                "operation": "count_by_group",
                "group_column": "Bill-to Name",
                "groups": [
                    {
                        "group": group,
                        "count": count,
                        "row_indices": row_indices[group],
                    }
                    for group, count in counts.items()
                ],
            }
        )

    if "Interval" in header and "Orders" in header:
        interval_index = header.index("Interval")
        orders_index = header.index("Orders")
        series: list[dict[str, Any]] = []
        for row in rows:
            if max(interval_index, orders_index) >= len(row):
                continue
            interval = row[interval_index].strip()
            raw_orders = row[orders_index].strip().replace(",", "")
            if not interval or interval.lower() == "total":
                continue
            try:
                orders = int(raw_orders)
            except ValueError:
                continue
            series.append({"interval": interval, "orders": orders})
        result["temporal_series"] = series

    if "Search Query" in header and "Uses" in header:
        label_index = header.index("Search Query")
        value_index = header.index("Uses")
        ranking_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows, start=1):
            if max(label_index, value_index) >= len(row):
                continue
            label = row[label_index].strip()
            raw_value = row[value_index].strip().replace(",", "")
            if not label or not raw_value:
                continue
            try:
                value = int(raw_value)
            except ValueError:
                continue
            ranking_rows.append(
                {"label": label, "value": value, "row_index": row_index}
            )
        if ranking_rows:
            ranking_rows.sort(key=lambda item: (-item["value"], item["row_index"]))
            result["rankings"] = [
                {
                    "operation": "sort_by_numeric",
                    "label_column": "Search Query",
                    "value_column": "Uses",
                    "direction": "descending",
                    "rows": ranking_rows,
                }
            ]

    return result


def build_evidence_prompt(
    task: dict[str, Any], visible_evidence: list[list[str]]
) -> str:
    structured = build_structured_numeric_evidence(visible_evidence)
    structured_task = build_structured_task_evidence(visible_evidence)
    return (
        "You are the Evidence Worker. Extract an answer only from the supplied real visible table. "
        "When the task asks for ranking over a time range and rows are split by interval, group identical "
        "Product values and sum their Order Quantity before ranking. Do not claim aggregation is impossible "
        "when the supplied generic count-by-group or temporal-series evidence contains all required rows. "
        "When the task asks for a brand or product type rather than a product, group rows by the requested "
        "brand or product type using explicit words in the Product names, sum Order Quantity for each group, "
        "and use provided numeric rankings when the task asks for top search terms. "
        "and return the requested category rather than an individual product. Return strict JSON only with exactly "
        "candidate_answer (string), evidence_summary (string), and evidence_row_indices (integer list). "
        "Row indices are zero-based in VISIBLE_EVIDENCE. Use candidate_answer N/A and an empty index list "
        "only when evidence is genuinely insufficient.\n"
        f"TASK: {task['intent']}\n"
        f"STRUCTURED_TASK_EVIDENCE: {json.dumps(structured_task, ensure_ascii=False)}\n"
        f"STRUCTURED_NUMERIC_EVIDENCE: {json.dumps(structured, ensure_ascii=False)}\n"
        f"VISIBLE_EVIDENCE: {json.dumps(visible_evidence, ensure_ascii=False)}"
    )


class ControlledEvidenceAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("Shopping Admin Evidence Worker")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        prompt = build_evidence_prompt(
            payload["task"],
            payload["visible_evidence"],
        )
        return AdminAgentMessage(
            "evidence_completion",
            {
                "raw": _complete_for_role(
                    self.client,
                    prompt,
                    role="Evidence Worker",
                    json_mode=True,
                ).strip()
            },
        )


class ControlledCoordinatorAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("Shopping Admin Controlled Coordinator")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        prompt = (
            "You are the Coordinator. Verify the candidate using only the delivered evidence envelope. Return "
            "strict JSON only with exactly decision (accept or reject), answer (string), and reason "
            "(string). Check task/session metadata, contract completeness, cited rows, and any structured "
            "evidence. If the delivered message is missing, stale, partial, internally inconsistent, or does "
            "not entail the answer, reject with answer N/A. "
            "There is no fallback and no retry.\n"
            f"TASK: {payload['task']['intent']}\n"
            f"EXPECTED_TASK_ID: {payload['task']['task_id']}\n"
            f"EVIDENCE_ENVELOPE: {json.dumps(payload.get('evidence_envelope'), ensure_ascii=False)}"
        )
        return AdminAgentMessage(
            "coordinator_completion",
            {
                "raw": _complete_for_role(
                    self.client,
                    prompt,
                    role="Coordinator",
                    json_mode=True,
                ).strip()
            },
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


def _event(
    *,
    run_id: str,
    trace_id: str,
    index: int,
    source: str,
    target: str,
    original: Any,
    delivered: Any,
    effect: str,
    message_id: str | None = None,
    fault_type: str = "clean",
    fault_applied: bool = False,
    fault_parameters: dict[str, Any] | None = None,
    send_timestamp: str | None = None,
    delivery_timestamp: str | None = None,
    observed_a_symptom: str = "none",
) -> dict[str, Any]:
    timestamp = send_timestamp or datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "timestamp": timestamp,
        "step_index": index,
        "step_id": f"step-{index:03d}",
        "message_id": message_id or f"{run_id}:step-{index:03d}",
        "source_agent": source,
        "target_agent": target,
        "original_message": original,
        "delivered_message": delivered,
        "fault_type": fault_type,
        "fault_applied": fault_applied,
        "fault_parameters": fault_parameters or {},
        "send_timestamp": send_timestamp or timestamp,
        "delivery_timestamp": delivery_timestamp or timestamp,
        "observed_runtime_effect": effect,
        "observed_A_symptom": observed_a_symptom,
        "observed_M_consequence": ["none"],
    }


async def run_controlled_clean_task(
    client: Any,
    browser: Any,
    evaluator: Any,
    task: dict[str, Any],
    *,
    original_config_file: Path,
    sanitized_config_dir: Path,
    run_index: int,
    max_steps: int = 10,
    max_evidence_chars: int = 24000,
    condition: str = "clean",
    interceptor: Any | None = None,
) -> dict[str, Any]:
    """Run one no-retry controlled-tool task with local message interception."""
    run_id = f"admin-controlled-{task['task_id']}-r{run_index}-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = (client.call_count, client.prompt_tokens, client.completion_tokens)
    request_log_before = len(getattr(client, "request_log", []))
    task_view = agent_task_view(task)
    active_interceptor = interceptor or AdminCommunicationInterceptor(condition)
    sanitized = write_sanitized_browser_config(
        Path(original_config_file), sanitized_config_dir
    )
    events: list[dict[str, Any]] = []
    event_index = 0
    runtime = SingleThreadedAgentRuntime()
    planner_id = AgentId("controlled_admin_planner", "default")
    navigator_id = AgentId("controlled_admin_navigator", "default")
    evidence_id = AgentId("controlled_admin_evidence", "default")
    coordinator_id = AgentId("controlled_admin_coordinator", "default")
    await runtime.register_agent_instance(ControlledPlannerAgent(client), planner_id)
    await runtime.register_agent_instance(ControlledToolNavigatorAgent(client), navigator_id)
    await runtime.register_agent_instance(ControlledEvidenceAgent(client), evidence_id)
    await runtime.register_agent_instance(ControlledCoordinatorAgent(client), coordinator_id)
    runtime.start()

    history: list[dict[str, Any]] = []
    visible_evidence: list[list[str]] = []
    browser_state: dict[str, Any] = {}
    termination_reason = "max_steps"
    try:
        browser_state = browser.reset(str(sanitized.path))
        event_index += 1
        events.append(
            _event(
                run_id=run_id,
                trace_id=trace_id,
                index=event_index,
                source="WebArena Tool Worker",
                target="Tool Navigator",
                original={"sanitized_config_sha256": sanitized.sha256},
                delivered={
                    "url": browser_state.get("url", ""),
                    "title": browser_state.get("title", ""),
                },
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
            _event(
                run_id=run_id,
                trace_id=trace_id,
                index=event_index,
                source="Planner",
                target="Tool Navigator",
                original={"task": task_view},
                delivered={"plan": plan},
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
                            "url": browser_state.get("url", ""),
                            "title": browser_state.get("title", ""),
                            "available_tools": browser_state.get(
                                "available_tools", []
                            ),
                        },
                        "history": history,
                        "evidence_row_count": len(visible_evidence),
                    },
                ),
            )
            raw = str(navigator_message.payload["raw"])
            try:
                request = parse_admin_tool_request(raw)
            except ValueError:
                termination_reason = "tool_request_parse_failure"
                event_index += 1
                events.append(
                    _event(
                        run_id=run_id,
                        trace_id=trace_id,
                        index=event_index,
                        source="Tool Navigator",
                        target="Tool Protocol Validator",
                        original=raw,
                        delivered=None,
                        effect=termination_reason,
                    )
                )
                raise
            event_index += 1
            events.append(
                _event(
                    run_id=run_id,
                    trace_id=trace_id,
                    index=event_index,
                    source="Tool Navigator",
                    target="WebArena Tool Worker",
                    original=raw,
                    delivered={"tool": request.name, "arguments": request.arguments},
                    effect="tool_request",
                )
            )
            available_tools = browser_state.get("available_tools")
            if isinstance(available_tools, list) and request.name not in available_tools:
                termination_reason = "tool_not_available_on_page"
                event_index += 1
                events.append(
                    _event(
                        run_id=run_id,
                        trace_id=trace_id,
                        index=event_index,
                        source="Tool Protocol Validator",
                        target="WebArena Tool Worker",
                        original={"tool": request.name},
                        delivered={"available_tools": available_tools},
                        effect=termination_reason,
                    )
                )
                raise ValueError(
                    f"tool {request.name!r} is not available on the current page"
                )
            browser_state = browser.tool(request.name, request.arguments)
            current_evidence = browser_state.get("visible_evidence") or []
            if current_evidence:
                visible_evidence = current_evidence
            history.append(
                {
                    "tool": request.name,
                    "arguments": request.arguments,
                    "status": browser_state.get("tool_status", "unknown"),
                    "url": browser_state.get("url", ""),
                    "evidence_rows": len(current_evidence),
                }
            )
            event_index += 1
            events.append(
                _event(
                    run_id=run_id,
                    trace_id=trace_id,
                    index=event_index,
                    source="WebArena Tool Worker",
                    target="Tool Navigator",
                    original={"tool": request.name, "arguments": request.arguments},
                    delivered={
                        "tool_status": browser_state.get("tool_status"),
                        "url": browser_state.get("url", ""),
                        "title": browser_state.get("title", ""),
                        "visible_evidence": current_evidence,
                        "tool_subtrace": browser_state.get("tool_subtrace", []),
                    },
                    effect="tool_result",
                )
            )
            if request.name == "finish_with_evidence":
                termination_reason = "navigator_finish"
                break

        encoded_evidence = json.dumps(visible_evidence, ensure_ascii=False)
        if len(encoded_evidence) > max_evidence_chars:
            raise ValueError(
                f"visible evidence exceeds frozen prompt budget: {len(encoded_evidence)}"
            )
        evidence_message = await _ask(
            runtime,
            evidence_id,
            AdminAgentMessage(
                "extract",
                {
                    "task": task_view,
                    "visible_evidence": visible_evidence,
                    "structured_numeric_evidence": build_structured_numeric_evidence(
                        visible_evidence
                    ),
                },
            ),
        )
        raw_evidence = str(evidence_message.payload["raw"])
        try:
            evidence_result = parse_evidence_result(raw_evidence)
        except ValueError:
            termination_reason = "evidence_result_parse_failure"
            event_index += 1
            events.append(
                _event(
                    run_id=run_id,
                    trace_id=trace_id,
                    index=event_index,
                    source="Evidence Worker",
                    target="Result Protocol Validator",
                    original=raw_evidence,
                    delivered=None,
                    effect=termination_reason,
                )
            )
            raise
        structured_numeric_evidence = build_structured_numeric_evidence(
            visible_evidence
        )
        evidence_envelope = make_evidence_envelope(
            message_id=f"{run_id}:evidence-handoff",
            task_id=str(task["task_id"]),
            source_session=run_id,
            state_version=3,
            payload={
                "evidence_result": evidence_result,
                "visible_evidence": visible_evidence,
                "structured_numeric_evidence": structured_numeric_evidence,
            },
        )
        delivery = active_interceptor.intercept(evidence_envelope)
        event_index += 1
        events.append(
            _event(
                run_id=run_id,
                trace_id=trace_id,
                index=event_index,
                source="Evidence Worker",
                target="Coordinator",
                original=delivery.original_message,
                delivered=delivery.delivered_message,
                effect=delivery.observed_runtime_effect,
                message_id=evidence_envelope["message_id"],
                fault_type=delivery.fault_type,
                fault_applied=delivery.fault_applied,
                fault_parameters=delivery.fault_parameters,
                send_timestamp=delivery.send_timestamp,
                delivery_timestamp=delivery.delivery_timestamp,
                observed_a_symptom=delivery.observed_a_symptom,
            )
        )

        coordinator_message = await _ask(
            runtime,
            coordinator_id,
            AdminAgentMessage(
                "coordinate",
                {
                    "task": task_view,
                    "evidence_envelope": delivery.delivered_message,
                },
            ),
        )
        raw_decision = str(coordinator_message.payload["raw"])
        try:
            decision = parse_coordinator_decision(raw_decision)
        except ValueError:
            termination_reason = "coordinator_result_parse_failure"
            event_index += 1
            events.append(
                _event(
                    run_id=run_id,
                    trace_id=trace_id,
                    index=event_index,
                    source="Coordinator",
                    target="Result Protocol Validator",
                    original=raw_decision,
                    delivered=None,
                    effect=termination_reason,
                )
            )
            raise
        event_index += 1
        events.append(
            _event(
                run_id=run_id,
                trace_id=trace_id,
                index=event_index,
                source="Coordinator",
                target="Official WebArena Evaluator",
                original={
                    "evidence_envelope": delivery.delivered_message,
                },
                delivered=decision,
                effect="final_answer_handoff",
            )
        )

        evaluation = evaluator.evaluate(str(original_config_file), decision["answer"])
        score = float(evaluation["score"])
        success = score == 1.0
        outcome = build_admin_outcome(
            expected_task_id=str(task["task_id"]),
            delivery=delivery,
            decision=decision,
            final_task_success=success,
            workflow_completed=True,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        prompt_tokens = client.prompt_tokens - before[1]
        completion_tokens = client.completion_tokens - before[2]
        record = {
            "run_id": run_id,
            "trace_id": trace_id,
            "scenario": (
                "webarena_shopping_admin_controlled_clean"
                if condition == "clean"
                else "webarena_shopping_admin_controlled_fault_matrix"
            ),
            "dataset": "WebArena",
            "benchmark": "WebArena Shopping Admin",
            "framework": "AutoGen",
            "topology": "sequential",
            "task_id": str(task["task_id"]),
            "task_stratum": task["task_stratum"],
            "intent": task["intent"],
            "condition": condition,
            "model": client.model_info.model,
            "provider": client.model_info.provider,
            "seed_or_run_index": run_index,
            "fault_id": delivery.fault_id,
            "fault_type": delivery.fault_type,
            "fault_severity": delivery.fault_severity,
            "fault_parameters": delivery.fault_parameters,
            "fault_applied": delivery.fault_applied,
            "source_agent": "Evidence Worker",
            "target_agent": "Coordinator",
            "original_message": delivery.original_message,
            "delivered_message": delivery.delivered_message,
            "send_timestamp": delivery.send_timestamp,
            "delivery_timestamp": delivery.delivery_timestamp,
            "first_divergence": (
                f"A:evidence_handoff:{condition}"
                if delivery.fault_applied
                else "none"
            ),
            "observed_runtime_effect": delivery.observed_runtime_effect,
            **outcome,
            "parse_retry_count": 0,
            "tool_retry_count": 0,
            "locator_fallback_count": 0,
            "coordinator_retry_count": 0,
            "expected_answer": task.get("eval", {}).get("reference_answers", {}),
            "final_answer": decision,
            "task_score": score,
            "final_task_success": success,
            "termination_reason": termination_reason,
            "browser_final_url": browser_state.get("url", ""),
            "browser_final_title": browser_state.get("title", ""),
            "sanitized_config_sha256": sanitized.sha256,
            "real_site": True,
            "official_task_config": True,
            "official_final_answer_evaluator": True,
            "standard_webarena_action_protocol": False,
            "process_trajectory_type": "controlled_tool_trace",
            "official_evaluator_input": "final_stop_only",
            "latency_ms": elapsed_ms,
            "api_call_count": client.call_count - before[0],
            "llm_calls": list(getattr(client, "request_log", []))[
                request_log_before:
            ],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "error": None,
            "events": events,
            "axis_evaluation_mode": "strict_evidence_v1",
        }
        return normalize_run_record(record, axis_evaluation_mode="strict_evidence_v1")
    except ControlledRunError:
        raise
    except Exception as exc:
        raise ControlledRunError(
            exc,
            events=events,
            termination_reason=termination_reason,
            browser_state=browser_state,
        ) from exc
    finally:
        await runtime.stop()
