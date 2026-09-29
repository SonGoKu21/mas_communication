"""Shared frozen-evidence matrix contract across admitted benchmark tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MAIN_TOPOLOGIES = ("sequential", "flat", "hierarchical")

# Matches the frozen fourteen-cell WebArena Admin confirmation contract.
FLASH_MAIN_CONDITIONS = (
    "clean",
    "timeliness_moderate_step4",
    "timeliness_deadline_step4",
    "non_delivery_step2",
    "non_delivery_step4",
    "semantic_corruption_step2",
    "semantic_corruption_step4",
    "malformed_message_step4",
    "valid_partial_message_step4",
    "duplicate_delivery_step2",
    "same_session_reordering_step3",
    "stale_replay_step4",
    "contract_key_drift_step4",
    "contract_type_drift_step4",
)


@dataclass(frozen=True)
class BenchmarkCarrier:
    benchmark: str
    task_id: str
    instruction: str
    expected_answer: Any
    envelope: dict[str, Any]
    source_row: dict[str, Any]
    source_jsonl: str


@dataclass(frozen=True)
class MatrixJob:
    benchmark: str
    task_id: str
    topology: str
    condition: str
    repeat_index: int
    run_id: str


def _official_success(row: dict[str, Any], benchmark: str) -> bool:
    if benchmark == "SWE-bench Verified":
        return bool(row.get("official_test_passed"))
    if benchmark == "TheAgentCompany":
        return bool(row.get("official_task_success"))
    if benchmark == "WebArena Reddit":
        return float(row.get("task_score") or 0.0) == 1.0
    raise ValueError(f"unsupported benchmark: {benchmark}")


def _task_id(row: dict[str, Any], benchmark: str) -> str:
    if benchmark == "SWE-bench Verified":
        return str(row.get("instance_id") or "")
    return str(row.get("task_slug") or row.get("task_id") or "")


def _instruction(row: dict[str, Any], benchmark: str, task_id: str) -> str:
    if benchmark == "SWE-bench Verified":
        return str(
            row.get("problem_statement")
            or row.get("instruction")
            or f"Produce and verify the patch for {task_id}."
        )
    if benchmark == "WebArena Reddit":
        if isinstance(row.get("verified_state"), dict):
            return f"Return the exact verified final state for Reddit task {task_id}."
        return str(row.get("intent") or f"Answer Reddit task {task_id}.")
    return str(row.get("instruction") or f"Complete {task_id}.")


def _evidence_summary(row: dict[str, Any], benchmark: str) -> str:
    if benchmark == "SWE-bench Verified":
        evidence = {
            "official_test_passed": bool(row.get("official_test_passed")),
            "test_output": (
                row.get("test_output")
                or row.get("official_test_output")
                or "official evaluator passed"
                if row.get("official_test_passed")
                else None
            ),
            "patch_present": bool(
                row.get("patch")
                or row.get("final_patch")
                or row.get("candidate_patch")
            ),
        }
    elif benchmark == "TheAgentCompany":
        evidence = {
            "official_task_success": bool(row.get("official_task_success")),
            "official_evaluation": row.get("official_evaluation"),
            "operator_exit_code": row.get("action_exit_code"),
        }
    else:
        evidence = {
            "task_score": float(row.get("task_score") or 0.0),
            "evaluator_mode": row.get("evaluator_mode"),
            "worker_mode": row.get("worker_mode"),
            "real_environment_evidence": row.get("evidence"),
            "pre_edit_state": row.get("original_message"),
            "verified_state": row.get("verified_state"),
            "editor_verdict": row.get("editor_verdict"),
        }
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)


def _candidate_answer(row: dict[str, Any], benchmark: str) -> str:
    if benchmark != "WebArena Reddit":
        return "success"
    final_answer = row.get("final_answer")
    answer = final_answer.get("answer") if isinstance(final_answer, dict) else final_answer
    if isinstance(answer, str) and answer.strip():
        return answer
    verified_state = row.get("verified_state")
    verdict = row.get("editor_verdict")
    if (
        isinstance(verified_state, dict)
        and verified_state
        and isinstance(verdict, dict)
        and verdict.get("decision") == "accept"
    ):
        return json.dumps(
            verified_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    raise ValueError("admitted Reddit source row has no real final answer or verified state")


def _expected_answer(row: dict[str, Any], benchmark: str) -> Any:
    if benchmark != "WebArena Reddit":
        return "success"
    expected = row.get("expected_answer")
    if isinstance(expected, dict) and expected:
        return expected
    verified_state = row.get("verified_state")
    verdict = row.get("editor_verdict")
    if (
        isinstance(verified_state, dict)
        and verified_state
        and isinstance(verdict, dict)
        and verdict.get("decision") == "accept"
    ):
        return {"state_assertion": verified_state}
    raise ValueError("admitted Reddit source row has no reference answer or state contract")


def _carrier(
    row: dict[str, Any], *, benchmark: str, source_jsonl: Path
) -> BenchmarkCarrier:
    task_id = _task_id(row, benchmark)
    if not task_id:
        raise ValueError("admitted source row has no task identifier")
    source_run_id = str(row.get("run_id") or f"source-{task_id}")
    summary = _evidence_summary(row, benchmark)
    candidate_answer = _candidate_answer(row, benchmark)
    envelope = {
        "message_id": f"{source_run_id}:normalized-evidence",
        "task_id": task_id,
        "source_session": source_run_id,
        "state_version": 1,
        "payload": {
            "evidence_result": {
                "task_id": task_id,
                "candidate_answer": candidate_answer,
                "evidence_summary": summary,
                "evidence_row_indices": [0],
            },
            "visible_evidence": [summary],
            "structured_task_evidence": {
                "benchmark": benchmark,
                "task_id": task_id,
                "official_success": True,
                "source_run_id": source_run_id,
                "source_trace_id": row.get("trace_id"),
            },
        },
    }
    return BenchmarkCarrier(
        benchmark=benchmark,
        task_id=task_id,
        instruction=_instruction(row, benchmark, task_id),
        expected_answer=_expected_answer(row, benchmark),
        envelope=envelope,
        source_row=row,
        source_jsonl=str(source_jsonl),
    )


def load_admitted_carriers(
    paths: Iterable[Path], *, benchmark: str
) -> tuple[BenchmarkCarrier, ...]:
    """Load one latest Flash clean-admitted carrier per task."""
    admitted: dict[str, BenchmarkCarrier] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") != "clean":
                continue
            if str(row.get("model", "")).lower() != "deepseek-v4-flash":
                continue
            if not _official_success(row, benchmark):
                continue
            if not bool(row.get("final_task_success")):
                continue
            carrier = _carrier(row, benchmark=benchmark, source_jsonl=path)
            admitted[carrier.task_id] = carrier
    if not admitted:
        raise ValueError(f"no admitted clean carriers for {benchmark}")
    return tuple(admitted[key] for key in sorted(admitted))


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def build_execution_schedule(
    carriers: Sequence[BenchmarkCarrier],
    *,
    topologies: Sequence[str] = MAIN_TOPOLOGIES,
    conditions: Sequence[str] = FLASH_MAIN_CONDITIONS,
    repeats: int = 3,
) -> tuple[MatrixJob, ...]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    unknown_topologies = set(topologies) - set(MAIN_TOPOLOGIES)
    unknown_conditions = set(conditions) - set(FLASH_MAIN_CONDITIONS)
    if unknown_topologies:
        raise ValueError(f"unknown topologies: {sorted(unknown_topologies)}")
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {sorted(unknown_conditions)}")
    jobs: list[MatrixJob] = []
    for repeat_index in range(1, repeats + 1):
        for carrier in carriers:
            for condition in conditions:
                for topology in topologies:
                    run_id = (
                        f"{_safe_id(carrier.benchmark)}-{_safe_id(carrier.task_id)}-"
                        f"{topology}-{condition}-r{repeat_index}"
                    )
                    jobs.append(
                        MatrixJob(
                            benchmark=carrier.benchmark,
                            task_id=carrier.task_id,
                            topology=topology,
                            condition=condition,
                            repeat_index=repeat_index,
                            run_id=run_id,
                        )
                    )
    return tuple(jobs)


def select_stale_carrier(
    current: BenchmarkCarrier, carriers: Sequence[BenchmarkCarrier]
) -> BenchmarkCarrier:
    try:
        return next(carrier for carrier in carriers if carrier.task_id != current.task_id)
    except StopIteration as exc:
        raise ValueError("stale replay requires another admitted task") from exc
