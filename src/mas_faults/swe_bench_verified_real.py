from __future__ import annotations

import json
import argparse
import csv
import difflib
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.causal_trace_report import build_report
from mas_faults.llm_client import ChatClient, get_llm_client


@dataclass(frozen=True)
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SWEInstance":
        return cls(
            instance_id=str(record["instance_id"]),
            repo=str(record["repo"]),
            base_commit=str(record["base_commit"]),
            problem_statement=str(record["problem_statement"]),
            fail_to_pass=parse_test_list(record.get("FAIL_TO_PASS")),
            pass_to_pass=parse_test_list(record.get("PASS_TO_PASS")),
        )


@dataclass(frozen=True)
class TestEvidence:
    instance_id: str
    passed: bool
    tests: tuple[str, ...]
    log: str
    runner_error: str | None = None
    evidence_origin_instance_id: str | None = None


@dataclass(frozen=True)
class InterceptedEvidence:
    original: TestEvidence
    delivered: TestEvidence | None
    observed_a_symptom: str
    fault_applied: bool


@dataclass(frozen=True)
class InterceptedTextMessage:
    original: str
    delivered: str | None
    observed_a_symptom: str
    fault_applied: bool


def intercept_text_message(message: str, condition: str) -> InterceptedTextMessage:
    """Apply transport-visible faults to Planner/Coder handoff text."""
    if condition == "clean":
        return InterceptedTextMessage(message, message, "none", False)
    if condition == "a5_omission":
        return InterceptedTextMessage(message, None, "A5_message_omission", True)
    if condition == "a8_truncation":
        return InterceptedTextMessage(message, message[:200], "A8_message_truncation", True)
    if condition == "a1_deadline_delay":
        return InterceptedTextMessage(message, None, "A2_message_timeout", True)
    if condition == "a1_moderate_delay":
        return InterceptedTextMessage(message, message, "A1_message_latency", True)
    raise ValueError(f"condition {condition!r} is unsupported on text messages")


@dataclass(frozen=True)
class ContextBudget:
    max_total_chars: int = 8000
    max_test_chars: int = 1200
    max_source_chars: int = 6000

    def compose(self, test_context: str, source_context: str) -> tuple[str, dict[str, int]]:
        bounded_test = test_context[:self.max_test_chars]
        bounded_source = source_context[:self.max_source_chars]
        context = f"FAILED TEST CONTEXT:\n{bounded_test}\n\nRELATED SOURCE CONTEXT:\n{bounded_source}"
        context = context[:self.max_total_chars]
        return context, {
            "repository_context_chars": len(context),
            "test_context_chars": len(bounded_test),
            "source_context_chars": len(bounded_source),
            "max_total_context_chars": self.max_total_chars,
        }


def intercept_evidence(
    evidence: TestEvidence,
    condition: str,
    stale: TestEvidence | None = None,
) -> InterceptedEvidence:
    origin = evidence.evidence_origin_instance_id or evidence.instance_id
    if condition == "clean":
        current = TestEvidence(evidence.instance_id, evidence.passed, evidence.tests, evidence.log, evidence.runner_error, origin)
        return InterceptedEvidence(current, current, "none", False)
    if condition == "a5_omission":
        return InterceptedEvidence(evidence, None, "A5_message_omission", True)
    if condition == "a8_truncation":
        partial = TestEvidence(evidence.instance_id, evidence.passed, (), evidence.log[:200], evidence.runner_error, origin)
        return InterceptedEvidence(evidence, partial, "A8_message_truncation", True)
    if condition == "a6_inner_evidence_poisoning":
        if stale is None:
            raise ValueError("a6_inner_evidence_poisoning requires prior test evidence")
        if stale.instance_id == evidence.instance_id:
            raise ValueError("a6_inner_evidence_poisoning requires evidence from a different task instance")
        poisoned = TestEvidence(
            evidence.instance_id,
            stale.passed,
            stale.tests,
            stale.log,
            stale.runner_error,
            stale.evidence_origin_instance_id or stale.instance_id,
        )
        return InterceptedEvidence(evidence, poisoned, "A6_message_semantic_corruption", True)
    if condition == "a12_stale_replay":
        if stale is None:
            raise ValueError("a12_stale_replay requires prior test evidence")
        if stale.instance_id == evidence.instance_id:
            raise ValueError("a12_stale_replay requires evidence from a different task instance")
        replayed = TestEvidence(
            stale.instance_id,
            stale.passed,
            stale.tests,
            stale.log,
            stale.runner_error,
            stale.evidence_origin_instance_id or stale.instance_id,
        )
        return InterceptedEvidence(evidence, replayed, "A12_timing_or_session_mismatch", True)
    if condition in {"a1_deadline_delay", "a1_moderate_delay"}:
        if condition == "a1_deadline_delay":
            return InterceptedEvidence(evidence, None, "A2_message_timeout", True)
        return InterceptedEvidence(evidence, evidence, "A1_message_latency", True)
    raise ValueError(f"unsupported condition={condition!r}")


def repeat_condition_schedule(conditions: list[str], repeats: int) -> tuple[tuple[int, str], ...]:
    """Create deterministic independent repetitions for baseline stability checks."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return tuple((repeat_index, condition) for repeat_index in range(1, repeats + 1) for condition in conditions)


STALE_EVIDENCE_CONDITIONS = frozenset({"a6_inner_evidence_poisoning", "a12_stale_replay"})
INJECTION_EDGES = {
    "planner_to_coder": ("Planner", "Coder"),
    "coder_to_tester": ("Coder", "Tester"),
    "tester_to_verifier": ("Tester", "Verifier"),
}


def build_execution_schedule(
    instance_ids: list[str], conditions: list[str], *, repeats: int, external_stale_evidence: bool = False,
) -> tuple[tuple[str, str, int, str | None], ...]:
    """Schedule every clean run before faults needing cross-instance evidence."""
    if len(instance_ids) < 2 and any(condition in STALE_EVIDENCE_CONDITIONS for condition in conditions) and not external_stale_evidence:
        raise ValueError("cross-instance stale faults require at least two instances")
    if "clean" not in conditions:
        raise ValueError("the experiment matrix must include clean")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    ordered: list[tuple[str, str, int, str | None]] = []
    faults = [condition for condition in conditions if condition != "clean"]
    for repeat_index in range(1, repeats + 1):
        for instance_id in instance_ids:
            ordered.append((instance_id, "clean", repeat_index, None))
        for condition in faults:
            for index, instance_id in enumerate(instance_ids):
                stale_source = instance_ids[(index - 1) % len(instance_ids)] if condition in STALE_EVIDENCE_CONDITIONS else None
                ordered.append((instance_id, condition, repeat_index, stale_source))
    return tuple(ordered)


def build_run_id(instance_id: str, condition: str, repeat_index: int = 1) -> str:
    """Keep every official-evaluation workspace unique across repeated trials."""
    if repeat_index < 1:
        raise ValueError("repeat_index must be at least 1")
    return f"{instance_id.replace('/', '_')}-{condition}-r{repeat_index}"


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Persist a completed run immediately so a long official evaluation is interruptible."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_completed_run_rows(path: Path) -> list[dict[str, Any]]:
    """Load durable run records so an interrupted matrix can resume without duplication."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    """Create a new output directory or preserve an existing one for resume mode."""
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise SystemExit(f"output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=resume)


def load_stale_evidence_jsonl(path: Path) -> TestEvidence:
    """Load the latest real TestEvidence message from a prior completed run."""
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        message = record.get("original_message")
        if not isinstance(message, dict) or not message.get("instance_id"):
            continue
        tests = message.get("tests", ())
        if not isinstance(tests, (list, tuple)):
            continue
        return TestEvidence(
            instance_id=str(message["instance_id"]),
            passed=bool(message.get("passed")),
            tests=tuple(map(str, tests)),
            log=str(message.get("log", "")),
            runner_error=None if message.get("runner_error") is None else str(message["runner_error"]),
        )
    raise ValueError(f"no valid original_message evidence found in {path}")


def build_swe_experiment_config(
    instance_ids: list[str],
    conditions: list[str],
    repeats: int,
    model: str,
    provider: str,
    *,
    stale_evidence_jsonl: str | None = None,
    injection_edge: str = "tester_to_verifier",
) -> dict[str, Any]:
    return {
        "benchmark": "SWE-bench Verified",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
        "llm_seed": int(os.environ["LLM_SEED"]) if os.environ.get("LLM_SEED") else None,
        "instance_ids": instance_ids,
        "conditions": conditions,
        "runs_per_condition": repeats,
        "stale_evidence_jsonl": stale_evidence_jsonl,
        "injection_edge": injection_edge,
        "workflow": "Planner -> Coder -> official SWE evaluator -> Tester -> Verifier",
    }


def render_swe_summary_markdown(summary: dict[str, Any]) -> str:
    return (
        "# SWE-bench Verified 通信故障实验汇总\n\n"
        f"- 运行数：{summary.get('runs', 0)}\n"
        f"- 官方测试通过：{summary.get('official_test_passed', 0)}\n"
        f"- MAS 最终成功：{summary.get('final_success', 0)}\n"
        f"- 环境错误：{summary.get('environment_errors', 0)}\n"
        f"- 模型补丁无效：{summary.get('agent_patch_invalid', 0)}\n"
    )


def swe_condition_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        summaries.append({
            "condition": condition,
            "runs": len(group),
            "a_exposure_count": sum(row.get("observed_A_symptom") != ["none"] for row in group),
            "m_consequence_count": sum(row.get("observed_M_consequence") != ["none"] for row in group),
            "recovery_count": sum(bool(row.get("recovery_detected")) for row in group),
            "final_success_count": sum(bool(row.get("final_task_success")) for row in group),
            "final_failure_count": sum(not bool(row.get("final_task_success")) for row in group),
        })
    return summaries


class WorkspaceRunner:
    def __init__(self, timeout_seconds: int = 900) -> None:
        self.timeout_seconds = timeout_seconds

    def run_command(self, workspace: Path, command: list[str], *, instance_id: str) -> TestEvidence:
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return TestEvidence(instance_id, False, (), output, "test_command_timeout")
        except OSError as exc:
            return TestEvidence(instance_id, False, (), "", f"runner_launch_error: {exc}")
        output = f"{completed.stdout}{completed.stderr}"
        return TestEvidence(instance_id, completed.returncode == 0, (), output)


class SWEHarness:
    def evaluation_command(
        self,
        *,
        dataset_name: str,
        split: str,
        instance_id: str,
        predictions_path: Path,
        run_id: str,
        report_dir: Path,
    ) -> list[str]:
        return [
            sys.executable,
            str(Path(__file__).with_name("swebench_offline_evaluation.py")),
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--instance_ids",
            instance_id,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            "1",
            "--namespace",
            "swebench",
            "--run_id",
            run_id,
            "--report_dir",
            str(report_dir),
        ]

    def parse_instance_report(
        self,
        report_path: Path,
        instance_id: str,
        tests: tuple[str, ...],
    ) -> TestEvidence:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            passed = bool(report[instance_id]["resolved"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return TestEvidence(instance_id, False, tests, "", f"official_report_error: {exc}")
        return TestEvidence(instance_id, passed, tests, report_path.read_text(encoding="utf-8"))


def evaluate_run(
    condition: str,
    *,
    delivered_evidence: TestEvidence | None,
    verifier_output: dict[str, Any],
    original_evidence: TestEvidence | None = None,
    communication_prevented_execution: bool = False,
    communication_induced_runtime_failure: bool = False,
) -> dict[str, Any]:
    a_symptom = {
        "clean": "none",
        "a1_moderate_delay": "A1_message_latency",
        "a1_deadline_delay": "A2_message_timeout",
        "a5_omission": "A5_message_omission",
        "a6_inner_evidence_poisoning": "A6_message_semantic_corruption",
        "a8_truncation": "A8_message_truncation",
        "a12_stale_replay": "A12_timing_or_session_mismatch",
    }[condition]
    if communication_prevented_execution:
        return {
            "observed_A_symptom": [a_symptom],
            "observed_M_consequence": ["M3_incomplete_information_aggregation", "M2_task_timeout_or_failure"],
            "benchmark_consequence": "swe.upstream_message_delivery_prevented_execution",
            "execution_status": "communication_delivery_failed",
            "final_task_success": False,
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": "",
            "propagation_class": "propagated_to_M_final_failure",
        }
    if communication_induced_runtime_failure:
        return {
            "observed_A_symptom": [a_symptom],
            "observed_M_consequence": ["M3_incomplete_information_aggregation", "M2_task_timeout_or_failure"],
            "benchmark_consequence": "swe.truncated_patch_rejected_before_tests",
            "execution_status": "communication_delivery_failed",
            "final_task_success": False,
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": "official evaluator reported Patch Apply Failed for the truncated delivered patch",
            "propagation_class": "propagated_to_M_final_failure",
        }
    if original_evidence and original_evidence.runner_error:
        return {
            "observed_A_symptom": [a_symptom],
            "observed_M_consequence": ["none"],
            "benchmark_consequence": "not_evaluable_due_to_agent_or_runtime_failure",
            "execution_status": original_evidence.runner_error,
            "final_task_success": False,
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": "",
            "propagation_class": "not_evaluable",
        }
    consequences: list[str] = []
    accepted = str(verifier_output.get("decision", "")).lower() == "accept"
    original_passed = bool(original_evidence and original_evidence.passed)
    detected = False
    semantic = "none"
    if delivered_evidence is None:
        consequences.append("M3_incomplete_information_aggregation")
        semantic = "swe.patch_rejected_after_missing_test_evidence"
    elif condition == "a8_truncation":
        partial = not delivered_evidence.tests or (
            original_evidence is not None and len(delivered_evidence.log) < len(original_evidence.log)
        )
        if partial:
            consequences.append("M3_incomplete_information_aggregation")
            if accepted:
                consequences.append("M14_partial_tool_or_message_result_acceptance")
                semantic = "swe.verifier_accepted_partial_test_evidence"
            else:
                detected = True
                semantic = "swe.verifier_rejected_partial_test_evidence"
        else:
            semantic = "swe.truncation_not_visible_to_verifier"
    elif condition == "a12_stale_replay":
        stale = original_evidence is not None and delivered_evidence.instance_id != original_evidence.instance_id
        if stale and accepted:
            consequences.extend(["M5_stale_context_acceptance", "M6_state_inconsistency"])
            semantic = "swe.verifier_accepted_stale_test_evidence"
        elif stale:
            detected = True
            semantic = "swe.verifier_rejected_stale_test_evidence"
    elif condition == "a6_inner_evidence_poisoning":
        poisoned = bool(
            original_evidence
            and delivered_evidence
            and (
                delivered_evidence.tests != original_evidence.tests
                or delivered_evidence.log != original_evidence.log
                or delivered_evidence.passed != original_evidence.passed
            )
        )
        if poisoned and accepted:
            consequences.append("M6_state_inconsistency")
            semantic = "swe.verifier_accepted_semantically_poisoned_test_evidence"
        elif poisoned:
            detected = True
            semantic = "swe.verifier_rejected_semantically_poisoned_test_evidence"
    successful = bool(delivered_evidence and delivered_evidence.passed and accepted and not consequences)
    if condition == "a12_stale_replay" and original_evidence and delivered_evidence and delivered_evidence.instance_id != original_evidence.instance_id:
        successful = False
    if condition == "a8_truncation" and delivered_evidence and (not delivered_evidence.tests or (original_evidence and len(delivered_evidence.log) < len(original_evidence.log))):
        successful = False
    if not successful:
        if accepted and condition in {"a8_truncation", "a12_stale_replay"}:
            consequences.append("M4_incorrect_collective_decision")
        consequences.append("M2_task_timeout_or_failure")
    consequences = list(dict.fromkeys(consequences))
    recovery_detected = False
    recovery_type = "none"
    recovery_evidence = ""
    if condition != "clean" and successful and not consequences:
        recovery_evidence = "fault-visible message was delivered and verifier accepted complete current evidence"
    if condition == "clean":
        propagation_class = "clean"
    elif detected and recovery_detected:
        propagation_class = "detected_and_recovered"
    elif detected:
        propagation_class = "detected_but_unrecovered"
    elif consequences:
        propagation_class = "silent_propagation_to_M" if successful else "propagated_to_M_final_failure"
    elif successful:
        propagation_class = "exposed_at_A_only"
    else:
        propagation_class = "fault_not_observed"
    return {
        "observed_A_symptom": [a_symptom],
        "observed_M_consequence": consequences or ["none"],
        "benchmark_consequence": semantic,
        "execution_status": "evaluated",
        "final_task_success": successful,
        "recovery_detected": recovery_detected,
        "recovery_type": recovery_type,
        "recovery_evidence": recovery_evidence,
        "propagation_class": propagation_class,
    }


def load_verified_instances(dataset_dir: Path, limit: int) -> tuple[list[SWEInstance], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet_path = dataset_dir / "data" / "test-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"ModelScope SWE-bench Verified parquet not found: {parquet_path}")
    records = pq.read_table(parquet_path).slice(0, limit).to_pylist()
    return [SWEInstance.from_record(record) for record in records], records


def test_paths_from_tests(tests: tuple[str, ...]) -> tuple[str, ...]:
    paths = {node_id.split("::", 1)[0] for node_id in tests if ".py::" in node_id}
    return tuple(sorted(paths))


test_paths_from_tests.__test__ = False


def select_relevant_test_context(content: str, tests: tuple[str, ...], *, max_chars: int) -> str:
    """Keep the failing test bodies instead of spending the budget on imports."""
    target_names_by_path: dict[str, set[str]] = {}
    for node_id in tests:
        if ".py::" not in node_id:
            continue
        path, target = node_id.split("::", 1)
        name = target.split("::")[-1].split("[", 1)[0]
        if name:
            target_names_by_path.setdefault(path, set()).add(name)

    selected: list[str] = []
    sections = re.split(r"(?=^--- FILE: .+? ---$)", content, flags=re.MULTILINE)
    for section in sections:
        header = re.match(r"^--- FILE: (.+?) ---\n", section)
        if not header:
            continue
        names = target_names_by_path.get(header.group(1), set())
        if not names:
            continue
        matches = [
            match for match in re.finditer(r"^\s*def\s+([A-Za-z_]\w*)\(", section, flags=re.MULTILINE)
            if match.group(1) in names
        ]
        all_functions = list(re.finditer(r"^\s*def\s+[A-Za-z_]\w*\(", section, flags=re.MULTILINE))
        for match in matches:
            next_function = next((item for item in all_functions if item.start() > match.start()), None)
            end = next_function.start() if next_function else len(section)
            selected.append(f"{header.group(0)}{section[match.start():end].strip()}")
    return ("\n\n".join(selected) or content)[:max_chars]


def source_paths_from_test_content(content: str, package_name: str) -> tuple[str, ...]:
    pattern = rf"^\s*from\s+({re.escape(package_name)}(?:\.[A-Za-z_][\w]*)+)\s+import\s+"
    modules = re.findall(pattern, content, flags=re.MULTILINE)
    paths = {module.replace(".", "/") + ".py" for module in modules}

    # Relative imports only have a resolvable package location when the collected
    # test context retains its file header.
    sections = re.split(r"(?=^--- FILE: .+? ---$)", content, flags=re.MULTILINE)
    relative_pattern = r"^\s*from\s+(\.+)([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s+import\s+"
    for section in sections:
        header = re.match(r"^--- FILE: (.+?) ---\n", section)
        if not header:
            continue
        test_package = list(Path(header.group(1)).with_suffix("").parts[:-1])
        if not test_package or test_package[0] != package_name:
            continue
        for match in re.finditer(relative_pattern, section, flags=re.MULTILINE):
            level = len(match.group(1))
            base_package = test_package[: len(test_package) - (level - 1)]
            if not base_package or base_package[0] != package_name:
                continue
            paths.add("/".join((*base_package, *match.group(2).split("."))) + ".py")
    return tuple(sorted(paths))


def select_relevant_source_context(source: str, hints: str, *, max_chars: int) -> str:
    file_sections = re.split(r"(?=^--- FILE: .+? ---$)", source, flags=re.MULTILINE)
    selected_sections: list[str] = []
    fallback_blocks: list[str] = []
    for section in file_sections:
        starts = list(re.finditer(r"^\s*def\s+([A-Za-z_]\w*)\(", section, flags=re.MULTILINE))
        blocks: list[str] = []
        for index, match in enumerate(starts):
            name = match.group(1)
            if not re.search(rf"\b{re.escape(name)}\b", hints):
                continue
            end = starts[index + 1].start() if index + 1 < len(starts) else len(section)
            blocks.append(section[match.start():end].strip())
        if not blocks:
            continue
        excerpt = "\n\n".join(blocks)
        header = re.match(r"^--- FILE: .+? ---\n", section)
        if header:
            selected_sections.append(f"{header.group(0)}{excerpt}")
        else:
            fallback_blocks.extend(blocks)
    excerpt = "\n\n".join((*selected_sections, *fallback_blocks))
    return (excerpt or source)[:max_chars]


def preserve_source_file_sections(source: str, paths: tuple[str, ...], *, max_chars: int, per_file_chars: int = 2000) -> str:
    sections = re.split(r"(?=^--- FILE: .+? ---$)", source, flags=re.MULTILINE)
    by_path = {}
    for section in sections:
        header = re.match(r"^--- FILE: (.+?) ---\n", section)
        if header:
            by_path[header.group(1)] = section
    return "\n\n".join(by_path[path][:per_file_chars] for path in paths if path in by_path)[:max_chars]


def source_search_terms(text: str) -> tuple[str, ...]:
    code_symbols = re.findall(r"`([A-Za-z_][\w.]*)`", text)
    snake_case_symbols = re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", text)
    class_symbols = re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b", text)
    capitalized_symbols = re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", text)
    symbols = code_symbols + snake_case_symbols + class_symbols + capitalized_symbols + re.findall(r"\b__[A-Za-z_]\w*__\b", text)
    ignored = {"AttributeError", "File", "Traceback"}
    return tuple(dict.fromkeys(symbol for symbol in symbols if symbol not in ignored))[:12]


def source_paths_from_issue_text(text: str, package_name: str) -> tuple[str, ...]:
    paths = re.findall(r"\b(?:[A-Za-z_][\w-]*/)+[A-Za-z_][\w-]*\.py\b", text)
    normalized = [path[path.index(f"{package_name}/"):] for path in paths if f"{package_name}/" in path]
    return tuple(dict.fromkeys(normalized))


def prioritize_source_paths(paths: tuple[str, ...], terms: tuple[str, ...]) -> tuple[str, ...]:
    conventional_modules = {
        re.sub(r"(?<!^)(?=[A-Z])", "_", term).lower() + ".py"
        for term in terms
        if re.fullmatch(r"[A-Z][A-Za-z0-9]+", term)
    }
    return tuple(sorted(paths, key=lambda path: (Path(path).name not in conventional_modules, path)))


def prioritize_paths_near_tests(paths: tuple[str, ...], test_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Prefer dependencies in the same package subtree as the failing test."""
    test_directories = [Path(path).parent.parts for path in test_paths]

    def shared_prefix_length(path: str) -> int:
        parts = Path(path).parts
        return max(
            (sum(left == right for left, right in zip(parts, directory)) for directory in test_directories),
            default=0,
        )

    return tuple(sorted(paths, key=lambda path: (-shared_prefix_length(path), path)))


def interleave_source_path_groups(groups: tuple[tuple[str, ...], ...], *, max_paths: int) -> tuple[str, ...]:
    """Allocate a small context-file budget across independent evidence sources."""
    selected: list[str] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index >= len(group) or group[index] in selected:
                continue
            selected.append(group[index])
            if len(selected) >= max_paths:
                return tuple(selected)
    return tuple(selected)


def swe_instance_image(instance: SWEInstance) -> str:
    image_id = instance.instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{image_id}:latest"


def candidate_scan_limit(requested_instances: int, *, has_explicit_ids: bool) -> int:
    """Scan broadly enough to find locally cached representative instances."""
    if has_explicit_ids:
        return 500
    return max(500, requested_instances * 4)


def select_runnable_instances(
    instances: list[SWEInstance], limit: int, *, image_exists: Any,
) -> tuple[list[SWEInstance], list[dict[str, str]]]:
    selected: list[SWEInstance] = []
    skipped: list[dict[str, str]] = []
    for instance in instances:
        if image_exists(instance):
            selected.append(instance)
            if len(selected) >= limit:
                break
        else:
            skipped.append({"instance_id": instance.instance_id, "reason": "local_image_missing"})
    return selected, skipped


def filter_instances_by_id(instances: list[SWEInstance], requested_ids: tuple[str, ...]) -> list[SWEInstance]:
    if not requested_ids:
        return instances
    by_id = {instance.instance_id: instance for instance in instances}
    missing = [instance_id for instance_id in requested_ids if instance_id not in by_id]
    if missing:
        raise ValueError(f"requested SWE instances not found: {', '.join(missing)}")
    return [by_id[instance_id] for instance_id in requested_ids]


def local_image_exists(instance: SWEInstance) -> bool:
    completed = subprocess.run(["docker", "image", "inspect", swe_instance_image(instance)], capture_output=True, text=True, check=False)
    return completed.returncode == 0


def read_container_files(image: str, paths: tuple[str, ...], *, max_chars: int = 12000) -> str:
    if not paths:
        return ""
    script = (
        "cd /testbed && for path in \"$@\"; do "
        "if [ -f \"$path\" ]; then "
        "printf '\\n--- FILE: %s ---\\n' \"$path\"; sed -n '1,900p' \"$path\"; "
        "else printf '\\n[repository context missing: %s]\\n' \"$path\"; fi; "
        "done"
    )
    command = ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", script, "context", *paths]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[repository context unavailable: {exc}]"
    if completed.returncode != 0:
        return f"[repository context unavailable: {completed.stderr[-500:]}]"
    return completed.stdout[:max_chars]


def search_container_source_paths(image: str, terms: tuple[str, ...], *, max_paths: int = 6) -> tuple[str, ...]:
    if not terms:
        return ()
    script = (
        "cd /testbed && find . -path '*/tests/*' -prune -o -name '*.py' -print | "
        "while IFS= read -r path; do score=0; for term in \"$@\"; do "
        "grep -qF \"$term\" \"$path\" 2>/dev/null && score=$((score + 1)); done; "
        "[ \"$score\" -gt 0 ] && printf '%s\\t%s\\n' \"$score\" \"$path\"; done | "
        "sort -rn | cut -f2-"
    )
    command = ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", script, "context", *terms]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ()
    paths = [line.removeprefix("./") for line in completed.stdout.splitlines()]
    return tuple(dict.fromkeys(path for path in paths if path.endswith(".py") and "/tests/" not in path))[:max_paths]


def collect_repository_context_with_metadata(instance: SWEInstance, budget: ContextBudget) -> tuple[str, dict[str, int]]:
    package_name = instance.repo.split("/", 1)[-1].replace("-", "_")
    image = swe_instance_image(instance)
    test_paths = test_paths_from_tests(instance.fail_to_pass)
    raw_test_content = read_container_files(image, test_paths, max_chars=max(budget.max_test_chars * 10, 20_000))
    test_content = select_relevant_test_context(raw_test_content, instance.fail_to_pass, max_chars=budget.max_test_chars)
    terms = source_search_terms(instance.problem_statement)
    imported_paths = prioritize_source_paths(source_paths_from_test_content(raw_test_content, package_name), terms)
    primary_source_paths = imported_paths[:6]
    primary_source_content = read_container_files(image, primary_source_paths, max_chars=max(budget.max_source_chars * 2, 200_000))
    dependency_paths = prioritize_paths_near_tests(
        prioritize_source_paths(source_paths_from_test_content(primary_source_content, package_name), terms),
        test_paths,
    )
    issue_paths = search_container_source_paths(image, terms)
    # Imports from the failing test are stronger dependency evidence than a
    # broad repository text search. Include one import hop (for example,
    # a public class module delegating its validation to a core module) before
    # filling remaining slots with issue-text matches.
    source_paths = interleave_source_path_groups(
        (source_paths_from_issue_text(instance.problem_statement, package_name), prioritize_source_paths(issue_paths, terms), dependency_paths, primary_source_paths),
        max_paths=8,
    )
    raw_source_content = read_container_files(image, source_paths, max_chars=max(budget.max_source_chars * 2, 200000))
    selected_source_content = select_relevant_source_context(
        raw_source_content,
        f"{test_content}\n{instance.problem_statement}",
        max_chars=budget.max_source_chars,
    )
    # The code under repair can live one import hop below the public module
    # named by a test. Keep that dependency evidence even when its function
    # names do not appear verbatim in the issue text.
    preserved_source_content = preserve_source_file_sections(
        raw_source_content,
        tuple(dict.fromkeys((*source_paths_from_issue_text(instance.problem_statement, package_name), *source_paths))),
        max_chars=budget.max_source_chars,
    )
    source_content = (preserved_source_content or selected_source_content)[:budget.max_source_chars]
    return budget.compose(test_content, source_content)


def collect_repository_context(instance: SWEInstance) -> str:
    context, _ = collect_repository_context_with_metadata(instance, ContextBudget())
    return context


def patch_apply_error(instance: SWEInstance, patch: str) -> str | None:
    if not patch:
        return "no unified diff was produced"
    command = [
        "docker", "run", "--rm", "-i", "--entrypoint", "/bin/sh", swe_instance_image(instance),
        "-c", "cd /testbed && git apply --check -",
    ]
    try:
        completed = subprocess.run(command, input=patch, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"patch_validation_unavailable: {exc}"
    if completed.returncode == 0:
        return None
    return (completed.stderr or completed.stdout or "git apply rejected candidate patch")[-3000:]


def parse_structured_edits(raw: str) -> list[dict[str, str]]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    payload = match.group(0) if match else raw
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        # Models sometimes emit Python regexes such as ``\d`` inside an
        # otherwise valid JSON response. Preserve valid JSON escapes while
        # treating only unsupported escapes as literal backslashes.
        try:
            value = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', payload))
        except json.JSONDecodeError:
            return []
    edits = value.get("edits", []) if isinstance(value, dict) else []
    if not isinstance(edits, list):
        return []
    valid: list[dict[str, str]] = []
    for edit in edits[:2]:
        if not isinstance(edit, dict):
            continue
        path, old, new = edit.get("path"), edit.get("old"), edit.get("new")
        if not all(isinstance(value, str) for value in (path, old, new)):
            continue
        if not path.endswith(".py") or path.startswith("/") or ".." in Path(path).parts or is_test_path(path):
            continue
        valid.append({"path": path, "old": old, "new": new})
    return valid


def is_test_path(path: str) -> bool:
    return any(part in {"test", "tests", "testing"} for part in Path(path).parts)


def edit_response_rejection_reason(raw: str) -> str | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return None
    edits = value.get("edits", []) if isinstance(value, dict) else []
    paths = [edit.get("path") for edit in edits if isinstance(edit, dict) and isinstance(edit.get("path"), str)]
    if paths and all(is_test_path(path) for path in paths):
        return "previous response attempted to modify only test files; choose an allowed production source file from Repository context instead"
    return None


def patch_from_structured_edits(files: dict[str, str], edits: list[dict[str, str]]) -> str:
    changed: dict[str, str] = dict(files)
    for edit in edits:
        path, old, new = edit["path"], edit["old"], edit["new"]
        if path not in changed or not old or changed[path].count(old) != 1:
            return ""
        changed[path] = changed[path].replace(old, new, 1)
    sections: list[str] = []
    for path in sorted(changed):
        before, after = files[path], changed[path]
        if before == after:
            continue
        unified = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=3,
        ))
        sections.append(f"diff --git a/{path} b/{path}\n{unified}")
    return "".join(sections)


def structured_edit_rejection_reason(instance: SWEInstance, raw: str) -> str | None:
    """Explain why a JSON edit cannot be converted into a concrete patch."""
    edits = parse_structured_edits(raw)
    if not edits:
        return None
    for edit in edits:
        path = edit["path"]
        source = read_container_file(swe_instance_image(instance), path)
        if not source:
            return f"structured edit target file could not be read: {path}"
        if source.count(edit["old"]) != 1:
            return f"structured edit old snippet was not found in {path}"
    return None


def read_container_file(image: str, path: str) -> str:
    command = ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", "cd /testbed && cat \"$1\"", "context", path]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def patch_from_edit_response(instance: SWEInstance, raw: str) -> str:
    diff = extract_unified_diff(raw)
    if diff:
        return diff
    edits = parse_structured_edits(raw)
    paths = tuple(sorted({edit["path"] for edit in edits}))
    if not edits:
        return ""
    files = {path: read_container_file(swe_instance_image(instance), path) for path in paths}
    if any(not content for content in files.values()):
        return ""
    return patch_from_structured_edits(files, edits)


def context_around_anchor(source: str, anchor: str, *, max_chars: int = 2400) -> str:
    location = source.find(anchor)
    if location < 0:
        location = source.find(anchor.lstrip())
    if location < 0:
        return source[:max_chars]
    line_start = source.rfind("\n", 0, location) + 1
    start = max(0, source.rfind("\n", 0, max(0, line_start - 1200)))
    return source[start:start + max_chars]


def repair_source_context(instance: SWEInstance, raw: str) -> str:
    edits = parse_structured_edits(raw)
    if not edits:
        return ""
    edit = edits[0]
    source = read_container_file(swe_instance_image(instance), edit["path"])
    anchor = next((line for line in edit["old"].splitlines() if line.strip()), "")
    if not source or not anchor:
        return ""
    excerpt = context_around_anchor(source, anchor)
    return f"ACTUAL TARGET FILE: {edit['path']}\n{excerpt}"


def generate_valid_patch(
    client: ChatClient,
    instance: SWEInstance,
    coder_output: str,
    repository_context: str,
    *,
    max_attempts: int = 3,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Convert model-proposed edits into a patch accepted by the real git parser."""
    attempts: list[dict[str, Any]] = []
    output = coder_output
    patch = ""
    error: str | None = None
    for attempt_index in range(1, max_attempts + 1):
        patch = patch_from_edit_response(instance, output)
        error = (
            edit_response_rejection_reason(output)
            or structured_edit_rejection_reason(instance, output)
            or patch_apply_error(instance, patch)
        )
        attempts.append({
            "attempt": attempt_index,
            "model_output": output,
            "candidate_patch": patch,
            "patch_apply_error": error,
            "patch_valid": error is None,
        })
        if error is None or attempt_index == max_attempts:
            break
        exact_context = repair_source_context(instance, output)
        repair_context = repository_context if not exact_context else f"{repository_context}\n\n{exact_context}"
        output = client.complete(build_patch_repair_prompt(patch, error, repair_context), json_mode=True)
    return patch, error, attempts


def parse_verifier_output(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return {"decision": "reject", "reason": "verifier_non_json"}
    decision = str(value.get("decision", "reject")).lower()
    return {"decision": "accept" if decision == "accept" else "reject", "reason": str(value.get("reason", ""))}


def run_official_evaluation(
    harness: SWEHarness,
    instance: SWEInstance,
    patch: str,
    *,
    dataset_name: str,
    split: str,
    output_dir: Path,
    run_id: str,
) -> TestEvidence:
    if not patch:
        return TestEvidence(instance.instance_id, False, instance.fail_to_pass, "", "agent_patch_invalid")
    predictions = output_dir / "official_predictions" / f"{run_id}.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    model_name = "mas-swe-communication"
    predictions.write_text(
        json.dumps({"instance_id": instance.instance_id, "model_name_or_path": model_name, "model_patch": patch}) + "\n",
        encoding="utf-8",
    )
    command = harness.evaluation_command(
        dataset_name=dataset_name,
        split=split,
        instance_id=instance.instance_id,
        predictions_path=predictions.resolve(),
        run_id=run_id,
        report_dir=output_dir,
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=output_dir, capture_output=True, text=True, timeout=2400, check=False)
    except subprocess.TimeoutExpired as exc:
        return TestEvidence(instance.instance_id, False, instance.fail_to_pass, str(exc.stdout or ""), "official_evaluator_timeout")
    log = f"{completed.stdout}{completed.stderr}"
    report = output_dir / "logs" / "run_evaluation" / run_id / model_name / instance.instance_id / "report.json"
    evidence = harness.parse_instance_report(report, instance.instance_id, instance.fail_to_pass + instance.pass_to_pass)
    if evidence.runner_error:
        return TestEvidence(instance.instance_id, False, evidence.tests, log, evidence.runner_error)
    return TestEvidence(instance.instance_id, evidence.passed, evidence.tests, log, None)


def write_event(path: Path, *, trace_id: str, run_id: str, instance: SWEInstance, condition: str, event_type: str, layer: str, component: str, status: str, effect: str, label: str, evidence: dict[str, Any], source: str | None = None, target: str | None = None, injection_point_kind: str | None = None) -> None:
    root = f"span-{uuid.uuid5(uuid.NAMESPACE_URL, trace_id)}"
    event = {
        "trace_id": trace_id, "span_id": f"span-{uuid.uuid4()}", "parent_span_id": root,
        "fault_id": None if condition == "clean" else f"fault-{run_id}", "carrier_id": f"carrier-{run_id}",
        "carrier_instance_id": f"carrier-{run_id}#1", "duplicate_index": 1, "logical_message_id": run_id,
        "pair_id": f"swe-{instance.instance_id}", "trace_variant": "clean" if condition == "clean" else "fault",
        "timestamp": datetime.now(timezone.utc).isoformat(), "timestamp_unix": time.time(), "event_layer": layer,
        "component": component, "event_type": event_type, "event_status": status, "source": source, "target": target,
        "injection_operator_code": {"a1_moderate_delay": "A1", "a1_deadline_delay": "A1", "a5_omission": "A5", "a6_inner_evidence_poisoning": "A6", "a8_truncation": "A8", "a12_stale_replay": "A12"}.get(condition),
        "injection_point_kind": (injection_point_kind or "communication_interceptor") if condition != "clean" else None,
        "injected_fault_code": {"a1_moderate_delay": "A1", "a1_deadline_delay": "A1", "a5_omission": "A5", "a6_inner_evidence_poisoning": "A6", "a8_truncation": "A8", "a12_stale_replay": "A12"}.get(condition),
        "injected_fault_layer": "A" if condition != "clean" else None,
        "expected_manifest_code": {"a1_deadline_delay": "A2", "a5_omission": "A5", "a8_truncation": "A8", "a12_stale_replay": "A12"}.get(condition),
        "expected_manifest_layer": "A" if condition != "clean" else None,
        "observed_effect": effect, "propagation_label": label,
        "evidence": {"run_id": run_id, "task_id": instance.instance_id, **evidence},
    }
    if event_type == "workflow_started":
        event["span_id"] = root
        event["parent_span_id"] = None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def run_instance(client: ChatClient, harness: SWEHarness, instance: SWEInstance, condition: str, output_dir: Path, stale: TestEvidence | None, repository_context: str, context_metadata: dict[str, int], repeat_index: int = 1, injection_edge: str = "tester_to_verifier") -> tuple[dict[str, Any], TestEvidence | None]:
    if injection_edge not in INJECTION_EDGES:
        raise ValueError(f"unsupported injection_edge={injection_edge!r}")
    if injection_edge != "tester_to_verifier" and condition in STALE_EVIDENCE_CONDITIONS:
        raise ValueError(f"{condition} is only defined for tester_to_verifier")
    run_id, trace_id = build_run_id(instance.instance_id, condition, repeat_index), f"trace-{uuid.uuid4()}"
    events = output_dir / "swe_bench_causal_events.jsonl"
    write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="workflow_started", layer="A", component="Coordinator", status="started", effect="SWE-bench MAS started", label="pre_injection", evidence={})
    calls_before, prompt_before, completion_before = client.call_count, client.prompt_tokens, client.completion_tokens
    started = time.perf_counter()
    source_agent, target_agent = INJECTION_EDGES[injection_edge]
    plan = client.complete(build_planner_prompt(instance, repository_context))
    injected_original: Any = None
    injected_delivered: Any = None
    observed_a_symptom = "none"
    fault_applied = False
    if injection_edge == "planner_to_coder":
        intercepted_plan = intercept_text_message(plan, condition)
        injected_original, injected_delivered = intercepted_plan.original, intercepted_plan.delivered
        observed_a_symptom, fault_applied = intercepted_plan.observed_a_symptom, intercepted_plan.fault_applied
        plan_for_coder = intercepted_plan.delivered or "[Planner message was not delivered due to a communication fault.]"
    else:
        plan_for_coder = plan
    coder_output = client.complete(build_coder_prompt(instance, plan_for_coder, repository_context), json_mode=True)
    patch, patch_error, patch_attempts = generate_valid_patch(client, instance, coder_output, repository_context)
    repair_output = "\n\n".join(str(attempt["model_output"]) for attempt in patch_attempts[1:])
    communication_prevented_execution = False
    if injection_edge == "coder_to_tester" and not patch_error:
        intercepted_patch = intercept_text_message(patch, condition)
        injected_original, injected_delivered = intercepted_patch.original, intercepted_patch.delivered
        observed_a_symptom, fault_applied = intercepted_patch.observed_a_symptom, intercepted_patch.fault_applied
        if intercepted_patch.delivered is None:
            communication_prevented_execution = True
            evidence = TestEvidence(instance.instance_id, False, instance.fail_to_pass, "candidate patch was not delivered to Tester", "communication_delivery_failed")
        else:
            patch = intercepted_patch.delivered
            evidence = run_official_evaluation(harness, instance, patch, dataset_name=str((output_dir / "swe_bench_verified_instances.json").resolve()), split="test", output_dir=output_dir, run_id=run_id)
    elif patch_error:
        evidence = TestEvidence(instance.instance_id, False, instance.fail_to_pass, patch_error, "agent_patch_invalid")
    else:
        evidence = run_official_evaluation(harness, instance, patch, dataset_name=str((output_dir / "swe_bench_verified_instances.json").resolve()), split="test", output_dir=output_dir, run_id=run_id)
    if injection_edge == "tester_to_verifier":
        intercepted = intercept_evidence(evidence, condition, stale)
        injected_original, injected_delivered = intercepted.original.__dict__, None if intercepted.delivered is None else intercepted.delivered.__dict__
        observed_a_symptom, fault_applied = intercepted.observed_a_symptom, intercepted.fault_applied
    else:
        intercepted = InterceptedEvidence(evidence, evidence, "none", False)
    write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="fault_applied" if fault_applied else "message_delivered", layer="A", component="communication_interceptor", status="applied" if fault_applied else "delivered", effect=observed_a_symptom, label="injected" if fault_applied else "pre_injection", evidence={}, source=source_agent, target=target_agent, injection_point_kind=f"{injection_edge}_interceptor")
    if fault_applied:
        write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="runtime_effect_observed", layer="A", component="communication_interceptor", status="effect_observed", effect=observed_a_symptom, label="propagated", evidence={}, source=source_agent, target=target_agent, injection_point_kind=f"{injection_edge}_interceptor")
        write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="message_dropped" if injected_delivered is None else "message_delivered", layer="A", component=target_agent, status="dropped" if injected_delivered is None else "delivered", effect=observed_a_symptom, label="propagated", evidence={}, source=source_agent, target=target_agent, injection_point_kind=f"{injection_edge}_interceptor")
    if evidence.runner_error and not communication_prevented_execution:
        verifier = {"decision": "reject", "reason": evidence.runner_error}
    else:
        verifier = parse_verifier_output(client.complete(build_verifier_prompt(instance, intercepted.delivered), json_mode=True))
    communication_induced_runtime_failure = bool(
        injection_edge == "coder_to_tester"
        and condition == "a8_truncation"
        and evidence.runner_error
        and "Patch Apply Failed" in evidence.log
    )
    outcome = evaluate_run(
        condition,
        delivered_evidence=intercepted.delivered,
        verifier_output=verifier,
        original_evidence=evidence,
        communication_prevented_execution=communication_prevented_execution,
        communication_induced_runtime_failure=communication_induced_runtime_failure,
    )
    for consequence in outcome["observed_M_consequence"]:
        if consequence != "none":
            write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="m_consequence_observed", layer="M", component="task_evaluator", status="observed", effect=consequence, label="propagated", evidence={"mas_consequence": consequence})
    final_status = "preserved" if outcome["final_task_success"] else "failed"
    write_event(events, trace_id=trace_id, run_id=run_id, instance=instance, condition=condition, event_type="final_consequence", layer="M", component="task_evaluator", status=final_status, effect=outcome["benchmark_consequence"], label="pre_injection" if condition == "clean" else "propagated", evidence={"mas_consequence": "; ".join(outcome["observed_M_consequence"]), "task_success": outcome["final_task_success"], "final_answer_correct": outcome["final_task_success"]})
    row = {
        "run_id": run_id, "trace_id": trace_id, "benchmark": "SWE-bench Verified", "execution_mode": "official_swe_harness", "scenario": "code_repair_verification", "task_id": instance.instance_id, "instance_id": instance.instance_id,
        "condition": condition, "fault_type": condition, "fault_severity": "none" if condition == "clean" else "default", "fault_parameters": {}, "model": client.model_info.model, "provider": client.model_info.provider,
        "fault_id": "none" if condition == "clean" else f"fault-{run_id}", "fault_applied": fault_applied,
        "source_agent": source_agent, "target_agent": target_agent, "injection_edge": injection_edge, "first_divergence": "A:fault_applied" if fault_applied else "none", "observed_runtime_effect": observed_a_symptom,
        "original_message": injected_original if injected_original is not None else {"plan": plan, "candidate_patch": patch}, "delivered_message": injected_delivered if injected_delivered is not None else {"plan": plan_for_coder, "candidate_patch": patch},
        "candidate_patch": patch, "planner_output": plan, "coder_output": coder_output,
        "patch_repair_output": repair_output, "patch_attempts": patch_attempts, "verifier_output": verifier,
        "context_metadata": context_metadata,
        "expected_answer": {"resolved": True, "instance_id": instance.instance_id}, "final_answer": verifier, **outcome, "official_test_passed": evidence.passed, "official_runner_error": evidence.runner_error,
        "communication_induced_runtime_failure": communication_induced_runtime_failure,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3), "api_call_count": client.call_count - calls_before,
        "prompt_tokens": client.prompt_tokens - prompt_before, "completion_tokens": client.completion_tokens - completion_before,
    }
    row["total_tokens"] = row["prompt_tokens"] + row["completion_tokens"]
    return normalize_run_record(row), evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real API SWE-bench Verified communication-fault experiments.")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=["clean", "a1_deadline_delay", "a5_omission", "a8_truncation", "a12_stale_replay"])
    parser.add_argument("--output-dir", default="results/swe_bench_verified_real")
    parser.add_argument("--dataset-dir", default="data/modelscope_swe_bench_verified")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--context-budget-chars", type=int, default=8000)
    parser.add_argument("--instance-ids", nargs="*", default=[])
    parser.add_argument("--repeats", type=int, default=1, help="Independent repetitions for every instance/condition pair.")
    parser.add_argument("--injection-edge", choices=sorted(INJECTION_EDGES), default="tester_to_verifier")
    parser.add_argument("--resume", action="store_true", help="Resume a non-empty output directory and skip completed run IDs.")
    parser.add_argument(
        "--stale-evidence-jsonl",
        help="A completed real-run JSONL whose original_message is replayed for A12. It must belong to another instance.",
    )
    args = parser.parse_args()
    if args.injection_edge != "tester_to_verifier" and any(condition in STALE_EVIDENCE_CONDITIONS for condition in args.conditions):
        raise SystemExit("a6_inner_evidence_poisoning and a12_stale_replay require --injection-edge tester_to_verifier")
    load_dotenv(); load_dotenv(".env.local")
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, resume=args.resume)
    client, harness = get_llm_client(mock_llm=args.mock_llm), SWEHarness()
    candidate_limit = candidate_scan_limit(args.instances, has_explicit_ids=bool(args.instance_ids))
    candidates, raw_instances = load_verified_instances(Path(args.dataset_dir), candidate_limit)
    candidates = filter_instances_by_id(candidates, tuple(args.instance_ids))
    instances, skipped = select_runnable_instances(candidates, args.instances, image_exists=local_image_exists)
    if not instances:
        raise SystemExit("no selected SWE-bench instances have a local evaluation image")
    (output_dir / "swe_bench_verified_instances.json").write_text(
        json.dumps([record for record in raw_instances if record["instance_id"] in {item.instance_id for item in instances}], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "instance_selection.json").write_text(json.dumps({"selected": [item.instance_id for item in instances], "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    config = build_swe_experiment_config(
        [item.instance_id for item in instances], args.conditions, args.repeats,
        client.model_info.model, client.model_info.provider,
        stale_evidence_jsonl=args.stale_evidence_jsonl,
        injection_edge=args.injection_edge,
    )
    (output_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runs_jsonl = output_dir / "swe_bench_runs.jsonl"
    rows = load_completed_run_rows(runs_jsonl) if args.resume else []
    completed_run_ids = {str(row.get("run_id", "")) for row in rows}
    stale_reference = load_stale_evidence_jsonl(Path(args.stale_evidence_jsonl)) if args.stale_evidence_jsonl else None
    instances_by_id = {instance.instance_id: instance for instance in instances}
    contexts = {
        instance.instance_id: collect_repository_context_with_metadata(
            instance, ContextBudget(max_total_chars=args.context_budget_chars),
        )
        for instance in instances
    }
    clean_evidence: dict[tuple[str, int], TestEvidence] = {}
    for instance_id, condition, repeat_index, stale_source_id in build_execution_schedule(
        [instance.instance_id for instance in instances],
        args.conditions,
        repeats=args.repeats,
        external_stale_evidence=stale_reference is not None,
    ):
        if build_run_id(instance_id, condition, repeat_index) in completed_run_ids:
            continue
        instance = instances_by_id[instance_id]
        repository_context, context_metadata = contexts[instance_id]
        if condition in STALE_EVIDENCE_CONDITIONS:
            stale_for_run = stale_reference if stale_reference is not None else clean_evidence.get((str(stale_source_id), repeat_index))
            if stale_for_run is None:
                raise RuntimeError(f"no clean evidence available for {condition} on {instance_id}")
        else:
            stale_for_run = None
        row, evidence = run_instance(
            client, harness, instance, condition, output_dir, stale_for_run,
            repository_context, context_metadata, repeat_index,
            injection_edge=args.injection_edge,
        )
        row["repeat_index"] = repeat_index
        rows.append(row)
        append_jsonl_record(runs_jsonl, row)
        if condition == "clean" and evidence is not None:
            clean_evidence[(instance_id, repeat_index)] = evidence
    with runs_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "swe_bench_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    summary = {
        "runs": len(rows),
        "official_test_passed": sum(bool(row["official_test_passed"]) for row in rows),
        "final_success": sum(bool(row["final_task_success"]) for row in rows),
        "environment_errors": sum(str(row["execution_status"]).startswith("official_") for row in rows),
        "agent_patch_invalid": sum(row["execution_status"] == "agent_patch_invalid" for row in rows),
    }
    (output_dir / "swe_bench_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "swe_bench_summary.md").write_text(render_swe_summary_markdown(summary), encoding="utf-8")
    condition_summary = swe_condition_summary_rows(rows)
    with (output_dir / "swe_bench_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_summary[0]) if condition_summary else ["condition"])
        writer.writeheader()
        writer.writerows(condition_summary)
    build_report(output_dir / "swe_bench_causal_events.jsonl", output_dir / "causal_trace_report")
    print(json.dumps(summary, sort_keys=True))


def load_instances(path: Path, limit: int) -> list[SWEInstance]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [SWEInstance.from_record(record) for record in records[:limit]]


def parse_test_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(map(str, value))
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("SWE test fields must be JSON arrays")
    return tuple(map(str, parsed))


def extract_unified_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(diff --git .*?)```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("diff --git ") :]
    candidate = candidate.strip()
    if not candidate.startswith("diff --git "):
        return ""
    return candidate


def build_planner_prompt(instance: SWEInstance, repository_context: str = "") -> str:
    return (
        "You are the Planner in a software-repair MAS. Analyze this issue and return a concise repair plan. "
        "Do not claim that tests passed.\n"
        f"Repository: {instance.repo}\nIssue:\n{instance.problem_statement}\n"
        f"Repository context:\n{repository_context}\n"
    )


def context_source_paths(repository_context: str) -> tuple[str, ...]:
    paths = re.findall(r"^--- FILE: (.+?) ---$", repository_context, flags=re.MULTILINE)
    return tuple(dict.fromkeys(path for path in paths if not is_test_path(path)))


def build_coder_prompt(instance: SWEInstance, plan: str, repository_context: str = "") -> str:
    allowed_paths = ", ".join(context_source_paths(repository_context)) or "none"
    return (
        "You are the Coder in a software-repair MAS. Return only one JSON object with an \"edits\" array. "
        "Each edit must contain string fields path, old, and new. "
        f"Allowed source paths: {allowed_paths}. "
        "For path, copy exactly one allowed source path. Never add a repository name, relative/ prefix, or any other prefix. "
        "Only modify files present in Repository context. Make the smallest possible repair: at most two source-file edits and no test files. "
        "The old value must occur exactly once in the current file. Do not return a diff, markdown, or explanation.\n"
        f"Repository: {instance.repo}\nBase commit: {instance.base_commit}\nIssue:\n{instance.problem_statement}\n"
        f"Planner plan:\n{plan}\nRepository context:\n{repository_context}\n"
    )


def build_patch_repair_prompt(candidate_patch: str, apply_error: str, repository_context: str = "") -> str:
    allowed_paths = ", ".join(context_source_paths(repository_context)) or "none"
    return (
        "You are the Patch Repairer. Return only one JSON object with an \"edits\" array. "
        "Each edit must contain string fields path, old, and new. "
        f"Allowed source paths: {allowed_paths}. "
        "For path, copy exactly one allowed source path. Never add a repository name, relative/ prefix, or any other prefix. "
        "Only modify files shown in Repository context. The old value must occur exactly once in the current file. "
        "Do not modify tests or return a diff, markdown, or commentary.\n"
        f"Previous patch validation error:\n{apply_error}\nPrevious candidate diff:\n{candidate_patch}\n"
        f"Repository context:\n{repository_context}\n"
    )


def build_verifier_prompt(instance: SWEInstance, evidence: TestEvidence | None) -> str:
    serialized = "missing" if evidence is None else json.dumps(
        {
            "instance_id": evidence.instance_id,
            "passed": evidence.passed,
            "tests": evidence.tests[:20],
            "test_count": len(evidence.tests),
            "log": evidence.log[:1200],
        },
        ensure_ascii=False,
    )
    return (
        "You are the Verifier in a software-repair MAS. Decide accept only if evidence belongs to the current "
        "instance and reports passing tests. Return strict JSON: {\"decision\": \"accept\"|\"reject\", \"reason\": string}.\n"
        f"Current instance: {instance.instance_id}\nTester evidence: {serialized}\n"
    )


if __name__ == "__main__":
    main()
