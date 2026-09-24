"""Frozen contract for the WebArena Admin main confirmation matrix."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from mas_faults.webarena_admin_tools import validate_admin_tool_call


MAIN_TOPOLOGIES = ("sequential", "flat", "hierarchical")
MAIN_TASK_IDS = (4, 107, 187, 199, 288)


@dataclass(frozen=True)
class MainConditionCell:
    condition: str
    fault_family: str
    injection_step: int | None
    fault_id: str
    fault_type: str
    severity: str = "default"
    fault_cause: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def fault_applied(self) -> bool:
        return self.condition != "clean"


MAIN_CONDITION_CELLS = (
    MainConditionCell("clean", "clean", None, "none", "clean"),
    MainConditionCell(
        "timeliness_moderate_step4",
        "timeliness",
        4,
        "A1",
        "delay",
        "moderate",
        "message_latency",
        {"delay_ms": 250},
    ),
    MainConditionCell(
        "timeliness_deadline_step4",
        "timeliness",
        4,
        "A2",
        "timeout",
        "deadline_exceeding",
        "message_deadline_exceeded",
        {"delay_ms": 1500, "deadline_ms": 500},
    ),
    MainConditionCell(
        "non_delivery_step2",
        "non_delivery",
        2,
        "A5",
        "omission",
        fault_cause="message_omission",
        parameters={"drop_count": 1},
    ),
    MainConditionCell(
        "non_delivery_step4",
        "non_delivery",
        4,
        "A5",
        "omission",
        fault_cause="message_omission",
        parameters={"drop_count": 1},
    ),
    MainConditionCell(
        "semantic_corruption_step2",
        "semantic_corruption",
        2,
        "A6",
        "semantic_corruption",
        fault_cause="task_parameter_corruption",
        parameters={"preserve_tool_contract": True},
    ),
    MainConditionCell(
        "semantic_corruption_step4",
        "semantic_corruption",
        4,
        "A6",
        "semantic_corruption",
        fault_cause="inner_evidence_poisoning",
        parameters={"preserve_outer_binding": True},
    ),
    MainConditionCell(
        "malformed_message_step4",
        "unparseable",
        4,
        "A7",
        "malformed_message",
        fault_cause="message_encoding_or_schema_break",
        parameters={"valid_json": False},
    ),
    MainConditionCell(
        "valid_partial_message_step4",
        "partial",
        4,
        "A8",
        "truncation",
        fault_cause="partial_delivery",
        parameters={"valid_json": True, "remove_required_fields": True},
    ),
    MainConditionCell(
        "duplicate_delivery_step2",
        "duplication",
        2,
        "A9",
        "duplicate",
        fault_cause="retry_induced_duplication",
        parameters={"delivery_count": 2},
    ),
    MainConditionCell(
        "same_session_reordering_step3",
        "ordering_freshness",
        3,
        "A10",
        "reordering",
        fault_cause="same_session_reordering",
        parameters={"newer_first": True, "older_last": True},
    ),
    MainConditionCell(
        "stale_replay_step4",
        "ordering_freshness",
        4,
        "A12",
        "stale_replay",
        fault_cause="cross_session_replay",
        parameters={"replace_entire_envelope": True},
    ),
    MainConditionCell(
        "contract_key_drift_step4",
        "contract_drift",
        4,
        "A11",
        "schema_drift",
        fault_cause="key_drift",
        parameters={"rename_required_keys": True},
    ),
    MainConditionCell(
        "contract_type_drift_step4",
        "contract_drift",
        4,
        "A15",
        "contract_violation",
        fault_cause="field_type_drift",
        parameters={"change_required_field_types": True},
    ),
)

CONDITION_BY_NAME = {cell.condition: cell for cell in MAIN_CONDITION_CELLS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MainDeliveryBatch:
    condition: str
    injection_step: int | None
    fault_id: str
    fault_type: str
    fault_family: str
    fault_cause: str
    fault_parameters: dict[str, Any]
    fault_applied: bool
    original_message: Any
    delivered_messages: tuple[Any, ...]
    send_timestamp: str
    delivery_timestamps: tuple[str, ...]
    observed_runtime_effect: str
    observed_a_symptom: str
    parseable: bool

    @property
    def delivery_count(self) -> int:
        return len(self.delivered_messages)


def _shift_date(value: str) -> str:
    parsed = datetime.strptime(value, "%m/%d/%Y")
    return (parsed + timedelta(days=365)).strftime("%m/%d/%Y")


def _corrupt_tool_request(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    tool = message.get("tool")
    arguments = deepcopy(message.get("arguments"))
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None
    if tool == "set_date_range":
        arguments["from_date"] = _shift_date(str(arguments["from_date"]))
        arguments["to_date"] = _shift_date(str(arguments["to_date"]))
    elif tool == "set_select_filter":
        current = str(arguments.get("value"))
        arguments["value"] = "Complete" if current != "Complete" else "Canceled"
    elif tool == "set_range_filter":
        for key in ("from_value", "to_value"):
            arguments[key] = str(int(str(arguments[key])) + 10)
    elif tool == "set_text_filter":
        arguments["value"] = f"{arguments['value']}-CORRUPTED"
    else:
        return None
    request = validate_admin_tool_call(tool, arguments)
    return {"tool": request.name, "arguments": request.arguments}


class MainCommunicationInterceptor:
    """Apply one local communication fault at a frozen abstract step."""

    def __init__(
        self,
        condition_cell: MainConditionCell,
        *,
        stale_message: Any | None = None,
        sleep_fn: Any = time.sleep,
        now_fn: Any = _utc_now,
    ) -> None:
        self.cell = condition_cell
        self.stale_message = deepcopy(stale_message)
        self.sleep_fn = sleep_fn
        self.now_fn = now_fn
        self.applied = False

    def intercept(
        self,
        step: int,
        message: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> MainDeliveryBatch:
        original = deepcopy(message)
        context = context or {}
        eligible = bool(context.get("eligible", True))
        should_apply = bool(
            self.cell.fault_applied
            and not self.applied
            and self.cell.injection_step == step
            and eligible
        )
        delivered: tuple[Any, ...] = (deepcopy(message),)
        effect = "clean_delivery"
        a_symptom = "none"
        parseable = True
        send_timestamp = self.now_fn()

        if should_apply:
            transformed = self._apply(step, message, context)
            if transformed is not None:
                delivered, effect, a_symptom, parseable = transformed
                self.applied = True
            else:
                should_apply = False

        delivery_timestamps = tuple(self.now_fn() for _ in delivered)
        return MainDeliveryBatch(
            condition=self.cell.condition,
            injection_step=self.cell.injection_step,
            fault_id=self.cell.fault_id if self.applied and should_apply else "none",
            fault_type=self.cell.fault_type if self.applied and should_apply else "clean",
            fault_family=self.cell.fault_family if self.applied and should_apply else "clean",
            fault_cause=self.cell.fault_cause if self.applied and should_apply else "none",
            fault_parameters=(
                deepcopy(self.cell.parameters) if self.applied and should_apply else {}
            ),
            fault_applied=bool(self.applied and should_apply),
            original_message=original,
            delivered_messages=delivered,
            send_timestamp=send_timestamp,
            delivery_timestamps=delivery_timestamps,
            observed_runtime_effect=effect,
            observed_a_symptom=a_symptom,
            parseable=parseable,
        )

    def _apply(
        self,
        step: int,
        message: Any,
        context: dict[str, Any],
    ) -> tuple[tuple[Any, ...], str, str, bool] | None:
        condition = self.cell.condition
        current = deepcopy(message)
        if condition == "timeliness_moderate_step4":
            self.sleep_fn(self.cell.parameters["delay_ms"] / 1000)
            return (current,), "delayed_then_delivered", "A1_message_latency", True
        if condition == "timeliness_deadline_step4":
            self.sleep_fn(self.cell.parameters["deadline_ms"] / 1000)
            return (), "deadline_exceeded_before_delivery", "A2_message_timeout", True
        if condition.startswith("non_delivery_"):
            return (), "message_dropped_before_delivery", "A5_message_omission", True
        if condition == "semantic_corruption_step2":
            corrupted = _corrupt_tool_request(current)
            if corrupted is None:
                return None
            return (
                (corrupted,),
                "task_critical_tool_parameters_corrupted",
                "A6_message_semantic_corruption",
                True,
            )
        if condition == "semantic_corruption_step4":
            stale = self._required_stale_message()
            current["payload"]["evidence_result"] = deepcopy(
                stale["payload"]["evidence_result"]
            )
            return (
                (current,),
                "nested_evidence_replaced_with_prior_task_evidence",
                "A6_message_semantic_corruption",
                True,
            )
        if condition == "malformed_message_step4":
            return (
                ('{"message_id":',),
                "unparseable_message_delivered",
                "A7_malformed_or_unparseable_message",
                False,
            )
        if condition == "valid_partial_message_step4":
            evidence = current.get("payload", {}).get("evidence_result", {})
            current["payload"] = {
                "evidence_result": {
                    "candidate_answer": evidence.get("candidate_answer", "")
                },
                "visible_evidence": deepcopy(
                    current.get("payload", {}).get("visible_evidence", [])[:2]
                ),
            }
            return (
                (current,),
                "syntactically_valid_partial_message_delivered",
                "A8_message_truncation",
                True,
            )
        if condition == "duplicate_delivery_step2":
            return (
                (current, deepcopy(current)),
                "duplicate_message_delivered_twice",
                "A9_message_duplication",
                True,
            )
        if condition == "same_session_reordering_step3":
            older = context.get("older_message")
            if not isinstance(older, dict):
                return None
            return (
                (current, deepcopy(older)),
                "newer_state_delivered_before_older_state",
                "A10_message_reordering",
                True,
            )
        if condition == "stale_replay_step4":
            return (
                (self._required_stale_message(),),
                "cross_session_stale_message_replayed",
                "A12_timing_or_session_mismatch",
                True,
            )
        if condition == "contract_key_drift_step4":
            current["task_binding"] = current.pop("task_id")
            evidence = current["payload"]["evidence_result"]
            evidence["candidate_response"] = evidence.pop("candidate_answer")
            return (
                (current,),
                "required_contract_keys_renamed",
                "A11_protocol_or_schema_mismatch",
                True,
            )
        if condition == "contract_type_drift_step4":
            current["state_version"] = str(current["state_version"])
            evidence = current["payload"]["evidence_result"]
            evidence["candidate_answer"] = [evidence["candidate_answer"]]
            evidence["evidence_row_indices"] = [
                str(value) for value in evidence.get("evidence_row_indices", [])
            ]
            return (
                (current,),
                "required_contract_field_types_changed",
                "A15_output_contract_violation",
                True,
            )
        return None

    def _required_stale_message(self) -> dict[str, Any]:
        if not isinstance(self.stale_message, dict):
            raise ValueError(f"{self.cell.condition} requires a stale_message")
        payload = self.stale_message.get("payload")
        if not isinstance(payload, dict) or not isinstance(
            payload.get("evidence_result"), dict
        ):
            raise ValueError(f"{self.cell.condition} requires stale evidence")
        return deepcopy(self.stale_message)


@dataclass(frozen=True)
class MainMatrixJob:
    topology: str
    task: dict[str, Any]
    condition_cell: MainConditionCell
    repeat_index: int
    matrix_run_index: int

    @property
    def job_key(self) -> str:
        return ":".join(
            (
                self.topology,
                str(self.task["task_id"]),
                self.condition_cell.condition,
                str(self.repeat_index),
            )
        )


def build_main_matrix_jobs(
    tasks: Iterable[dict[str, Any]],
    *,
    repetitions: int,
    topologies: Iterable[str] = MAIN_TOPOLOGIES,
    condition_cells: Iterable[MainConditionCell] = MAIN_CONDITION_CELLS,
    task_ids: Iterable[int] = MAIN_TASK_IDS,
) -> list[MainMatrixJob]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    task_by_id = {int(task["task_id"]): task for task in tasks}
    selected_task_ids = tuple(dict.fromkeys(int(task_id) for task_id in task_ids))
    if not selected_task_ids:
        raise ValueError("at least one task ID must be selected")
    missing = [task_id for task_id in selected_task_ids if task_id not in task_by_id]
    if missing:
        raise ValueError(f"main matrix is missing selected tasks: {missing}")
    ordered_tasks = [task_by_id[task_id] for task_id in selected_task_ids]
    selected_topologies = tuple(topologies)
    unknown = [name for name in selected_topologies if name not in MAIN_TOPOLOGIES]
    if unknown:
        raise ValueError(f"unknown main topology: {unknown}")

    jobs: list[MainMatrixJob] = []
    for topology in selected_topologies:
        for task in ordered_tasks:
            for cell in condition_cells:
                for repeat_index in range(1, repetitions + 1):
                    jobs.append(
                        MainMatrixJob(
                            topology=topology,
                            task=task,
                            condition_cell=cell,
                            repeat_index=repeat_index,
                            matrix_run_index=len(jobs) + 1,
                        )
                    )
    return jobs
