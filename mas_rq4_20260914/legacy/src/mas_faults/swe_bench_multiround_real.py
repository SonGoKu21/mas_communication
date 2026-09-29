"""Real multi-round SWE-bench repair workflow with a single feedback fault.

The existing verified runner remains the single-round baseline.  This runner
adds a bounded ``test -> feedback -> repair -> retest`` loop while reusing its
LLM client, local fault interceptor, official evaluator, and trace contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.causal_trace_report import build_report
from mas_faults.llm_client import ChatClient, get_llm_client
from mas_faults.swe_bench_verified_real import (
    ContextBudget,
    SWEHarness,
    SWEInstance,
    TestEvidence,
    append_jsonl_record,
    build_coder_prompt,
    build_feedback_repair_prompt,
    build_patch_repair_prompt,
    build_planner_prompt,
    build_run_id,
    collect_repository_context_with_metadata,
    context_source_paths,
    evaluate_feedback_loop_run,
    extract_unified_diff,
    extract_feedback_assessment,
    generate_valid_patch,
    intercept_feedback_for_round,
    load_verified_instances,
    local_image_exists,
    parse_verifier_output,
    parse_structured_edits,
    patch_apply_error,
    prepare_output_dir,
    run_official_evaluation,
    repair_source_context,
    select_runnable_instances,
    swe_instance_image,
    write_event,
    build_verifier_prompt,
)


def evidence_payload(evidence: TestEvidence | None) -> dict[str, Any] | None:
    return None if evidence is None else evidence.__dict__


def build_unified_diff_coder_prompt(instance: SWEInstance, plan: str, repository_context: str) -> str:
    paths = ", ".join(context_source_paths(repository_context)) or "none"
    return (
        "You are the Coder in a real software-repair workflow. Return only a valid unified git diff, with "
        "diff --git, --- and +++ headers. Modify at most two non-test files from the supplied repository context. "
        "Do not explain the patch, return JSON, markdown fences, or test-file changes. Every removed line must "
        "match the repository context exactly.\n"
        f"Repository: {instance.repo}\nIssue:\n{instance.problem_statement}\nPlanner plan:\n{plan}\n"
        f"Allowed source paths: {paths}\nRepository context:\n{repository_context}\n"
    )


def build_unified_diff_repair_prompt(
    instance: SWEInstance,
    *,
    plan: str,
    prior_patch: str,
    feedback: TestEvidence | None,
    repository_context: str,
) -> str:
    serialized_feedback = "missing" if feedback is None else json.dumps(
        {
            "instance_id": feedback.instance_id,
            "passed": feedback.passed,
            "tests": feedback.tests[:20],
            "log": feedback.log[:2400],
            "runner_error": feedback.runner_error,
        },
        ensure_ascii=False,
    )
    return (
        "You are the repair Coder. The previous candidate was applied to the current workspace and failed. "
        "Use the Tester feedback to return only a valid incremental unified git diff against that current workspace. "
        "Do not repeat the failed patch, return JSON, markdown fences, commentary, or test-file changes.\n"
        f"Repository: {instance.repo}\nIssue:\n{instance.problem_statement}\nPlanner plan:\n{plan}\n"
        f"Previous candidate diff:\n{prior_patch}\nTester feedback:\n{serialized_feedback}\n"
        f"Repository context:\n{repository_context}\n"
    )


def feedback_retry_context(repository_context: str, feedback: TestEvidence | None) -> str:
    """Keep the failed-test evidence available when repairing invalid model edits."""
    if feedback is None:
        return repository_context
    return (
        f"{repository_context}\n\nCURRENT TESTER FEEDBACK:\n"
        + json.dumps(
            {
                "instance_id": feedback.instance_id,
                "passed": feedback.passed,
                "tests": feedback.tests[:20],
                "log": feedback.log[:4000],
                "runner_error": feedback.runner_error,
            },
            ensure_ascii=False,
        )
    )


def generate_cumulative_repair_patch(
    instance: SWEInstance,
    repair_output: str,
    prior_patch: str,
    *,
    client: ChatClient | None = None,
    repository_context: str = "",
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Apply an incremental repair to the prior candidate and export a base-relative patch."""
    incremental_diff = extract_unified_diff(repair_output)
    edits = parse_structured_edits(repair_output) if not incremental_diff else []
    if not incremental_diff and not edits:
        error = "repair response contained neither a unified diff nor structured edits"
        attempts = [{"attempt": 1, "model_output": repair_output, "candidate_patch": "", "patch_apply_error": error, "patch_valid": False}]
        return "", error, attempts

    program = r'''
import json
import pathlib
import subprocess
import sys

payload = json.load(sys.stdin)
prior = payload["prior_patch"]
pathlib.Path("/tmp/prior.patch").write_text(prior, encoding="utf-8")
applied = subprocess.run(["git", "apply", "/tmp/prior.patch"], capture_output=True, text=True)
if applied.returncode:
    print(json.dumps({"error": "prior patch could not be applied: " + applied.stderr[-600:]}))
    raise SystemExit(0)
if payload["incremental_diff"]:
    pathlib.Path("/tmp/repair.patch").write_text(payload["incremental_diff"], encoding="utf-8")
    repaired = subprocess.run(["git", "apply", "/tmp/repair.patch"], capture_output=True, text=True)
    if repaired.returncode:
        print(json.dumps({"error": "incremental patch could not be applied: " + repaired.stderr[-600:]}))
        raise SystemExit(0)
    paths = []
else:
    paths = []
    for edit in payload["edits"]:
        path = edit["path"]
        source = pathlib.Path(path)
        if not source.is_file():
            print(json.dumps({"error": "source file is unavailable: " + path}))
            raise SystemExit(0)
        content = source.read_text(encoding="utf-8")
        occurrences = content.count(edit["old"])
        if occurrences != 1:
            print(json.dumps({"error": f"old snippet occurs {occurrences} times in current workspace: {path}"}))
            raise SystemExit(0)
        source.write_text(content.replace(edit["old"], edit["new"], 1), encoding="utf-8")
        paths.append(path)
diff = subprocess.run(["git", "diff", "--", *paths], capture_output=True, text=True)
print(json.dumps({"patch": diff.stdout}))
'''
    payload = {"prior_patch": prior_patch, "incremental_diff": incremental_diff, "edits": edits}
    command = ["docker", "run", "--rm", "-i", "-w", "/testbed", "--entrypoint", "python", swe_instance_image(instance), "-c", program]
    try:
        completed = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True, timeout=180, check=False)
        response = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        error = f"cumulative repair construction failed: {exc}"
        attempts = [{"attempt": 1, "model_output": repair_output, "candidate_patch": "", "patch_apply_error": error, "patch_valid": False}]
        return "", error, attempts
    error = response.get("error") or (completed.stderr[-600:] if completed.returncode else None)
    patch = str(response.get("patch") or "")
    if not error:
        error = patch_apply_error(instance, patch)
    attempts = [{"attempt": 1, "model_output": repair_output, "candidate_patch": patch, "patch_apply_error": error, "patch_valid": error is None}]
    if error and client is not None:
        exact_context = repair_source_context(instance, repair_output)
        if exact_context:
            corrected_output = client.complete(
                build_patch_repair_prompt(patch, error, f"{repository_context}\n\n{exact_context}"),
                json_mode=True,
            )
            corrected_patch, corrected_error, corrected_attempts = generate_cumulative_repair_patch(
                instance,
                corrected_output,
                prior_patch,
            )
            for attempt in corrected_attempts:
                attempt["attempt"] += len(attempts)
            return corrected_patch, corrected_error, attempts + corrected_attempts
    return patch, error, attempts


def should_attempt_repair(evidence: TestEvidence) -> bool:
    """Keep agent-generated patch errors inside the repair loop, not as runtime aborts."""
    return not evidence.passed and evidence.runner_error not in {
        "official_evaluator_timeout",
        "communication_delivery_failed",
    }


def retain_prior_patch_after_invalid_repair(prior_patch: str, candidate_patch: str, patch_error: str | None) -> str:
    """Keep the last applied workspace state when a later repair cannot be built."""
    return prior_patch if patch_error else candidate_patch


def retain_fault_round_feedback_assessment(
    prior_assessment: str,
    fault_applied: bool,
    current_assessment: str,
) -> str:
    """Preserve the Coder judgement made on the single injected handoff."""
    return current_assessment if fault_applied else prior_assessment


def run_multiround_instance(
    client: ChatClient,
    harness: SWEHarness,
    instance: SWEInstance,
    *,
    condition: str,
    output_dir: Path,
    repository_context: str,
    context_metadata: dict[str, int],
    max_repair_rounds: int,
    fault_round: int,
    stale: TestEvidence | None = None,
    repair_client: ChatClient | None = None,
    patch_format: str = "json_edits",
) -> dict[str, Any]:
    if max_repair_rounds < 1:
        raise ValueError("max_repair_rounds must be at least 1")
    if fault_round < 1 or fault_round >= max_repair_rounds:
        raise ValueError("fault_round must select a feedback handoff before the final repair round")

    run_id = build_run_id(instance.instance_id, condition)
    trace_id = f"trace-{uuid.uuid4()}"
    events_path = output_dir / "swe_bench_causal_events.jsonl"
    write_event(
        events_path,
        trace_id=trace_id,
        run_id=run_id,
        instance=instance,
        condition=condition,
        event_type="workflow_started",
        layer="A",
        component="Coordinator",
        status="started",
        effect="multi-round SWE repair started",
        label="pre_injection",
        evidence={"max_repair_rounds": max_repair_rounds, "fault_round": fault_round},
    )
    repair_client = repair_client or client
    clients = [client] if repair_client is client else [client, repair_client]
    calls_before = sum(item.call_count for item in clients)
    prompt_before = sum(item.prompt_tokens for item in clients)
    completion_before = sum(item.completion_tokens for item in clients)
    started = time.perf_counter()
    plan = client.complete(build_planner_prompt(instance, repository_context))
    if patch_format == "unified_diff":
        coder_output = client.complete(build_unified_diff_coder_prompt(instance, plan, repository_context))
    else:
        coder_output = client.complete(build_coder_prompt(instance, plan, repository_context), json_mode=True)
    patch, patch_error, patch_attempts = generate_valid_patch(client, instance, coder_output, repository_context)
    evidence = (
        TestEvidence(instance.instance_id, False, instance.fail_to_pass, patch_error, "agent_patch_invalid")
        if patch_error
        else run_official_evaluation(
            harness,
            instance,
            patch,
            dataset_name=str((output_dir / "swe_bench_verified_instances.json").resolve()),
            split="test",
            output_dir=output_dir,
            run_id=f"{run_id}-round1",
        )
    )
    round_records: list[dict[str, Any]] = [{
        "repair_round": 1,
        "candidate_patch": patch,
        "patch_attempts": patch_attempts,
        "official_evidence": evidence_payload(evidence),
        "feedback_original": None,
        "feedback_delivered": None,
        "feedback_assessment": "not_applicable",
        "fault_applied": False,
    }]
    original_feedback = evidence
    injected_feedback: TestEvidence | None = evidence
    observed_a_symptom = "none"
    applied = False
    feedback_assessment = "not_applicable"
    fault_feedback_assessment = "not_applicable"

    for feedback_round in range(1, max_repair_rounds):
        if not should_attempt_repair(evidence):
            break
        intercepted = intercept_feedback_for_round(
            evidence,
            condition,
            round_index=feedback_round,
            fault_round=fault_round,
            stale=stale,
        )
        if intercepted.fault_applied:
            applied = True
            original_feedback = intercepted.original
            injected_feedback = intercepted.delivered
            observed_a_symptom = intercepted.observed_a_symptom
            write_event(
                events_path,
                trace_id=trace_id,
                run_id=run_id,
                instance=instance,
                condition=condition,
                event_type="fault_applied",
                layer="A",
                component="communication_interceptor",
                status="applied",
                effect=observed_a_symptom,
                label="injected",
                evidence={"feedback_round": feedback_round, "original": evidence_payload(intercepted.original), "delivered": evidence_payload(intercepted.delivered)},
                source="Tester",
                target="Coder",
                injection_point_kind="tester_to_coder_feedback_interceptor",
            )
        repair_context = feedback_retry_context(repository_context, intercepted.delivered)
        if patch_format == "unified_diff":
            repair_output = repair_client.complete(
                build_unified_diff_repair_prompt(
                    instance,
                    plan=plan,
                    prior_patch=patch,
                    feedback=intercepted.delivered,
                    repository_context=repair_context,
                ),
            )
            feedback_assessment = "unknown"
        else:
            repair_output = repair_client.complete(
                build_feedback_repair_prompt(
                    instance,
                    plan=plan,
                    prior_patch=patch,
                    feedback=intercepted.delivered,
                    repository_context=repair_context,
                ),
                json_mode=True,
            )
            feedback_assessment = extract_feedback_assessment(repair_output)
        fault_feedback_assessment = retain_fault_round_feedback_assessment(
            fault_feedback_assessment,
            intercepted.fault_applied,
            feedback_assessment,
        )
        prior_patch = patch
        candidate_patch, patch_error, patch_attempts = generate_cumulative_repair_patch(
            instance,
            repair_output,
            prior_patch,
            client=repair_client,
            repository_context=repair_context,
        )
        patch = retain_prior_patch_after_invalid_repair(prior_patch, candidate_patch, patch_error)
        repair_round = feedback_round + 1
        evidence = (
            TestEvidence(instance.instance_id, False, instance.fail_to_pass, patch_error, "agent_patch_invalid")
            if patch_error
            else run_official_evaluation(
                harness,
                instance,
                patch,
                dataset_name=str((output_dir / "swe_bench_verified_instances.json").resolve()),
                split="test",
                output_dir=output_dir,
                run_id=f"{run_id}-round{repair_round}",
            )
        )
        round_records.append({
            "repair_round": repair_round,
            "candidate_patch": patch,
            "patch_attempts": patch_attempts,
            "official_evidence": evidence_payload(evidence),
            "feedback_original": evidence_payload(intercepted.original),
            "feedback_delivered": evidence_payload(intercepted.delivered),
            "feedback_assessment": feedback_assessment,
            "fault_applied": intercepted.fault_applied,
        })
        write_event(
            events_path,
            trace_id=trace_id,
            run_id=run_id,
            instance=instance,
            condition=condition,
            event_type="repair_round_completed",
            layer="M",
            component="Coder",
            status="passed" if evidence.passed else "failed",
            effect="official test evidence produced",
            label="propagated" if applied else "pre_injection",
            evidence={"repair_round": repair_round, "feedback_assessment": feedback_assessment, "official_evidence": evidence_payload(evidence)},
            source="Coder",
            target="Tester",
        )

    verifier = (
        {"decision": "reject", "reason": evidence.runner_error}
        if evidence.runner_error
        else parse_verifier_output(repair_client.complete(build_verifier_prompt(instance, evidence), json_mode=True))
    )
    if not applied:
        original_feedback = evidence
        injected_feedback = evidence
    outcome = evaluate_feedback_loop_run(
        condition if applied else "clean",
        injected_feedback=injected_feedback,
        original_feedback=original_feedback,
        final_evidence=evidence,
        verifier_output=verifier,
        feedback_assessment=fault_feedback_assessment,
        repair_rounds=len(round_records),
    )
    for consequence in outcome["observed_M_consequence"]:
        if consequence != "none":
            write_event(
                events_path,
                trace_id=trace_id,
                run_id=run_id,
                instance=instance,
                condition=condition,
                event_type="m_consequence_observed",
                layer="M",
                component="task_evaluator",
                status="observed",
                effect=consequence,
                label="propagated",
                evidence={"mas_consequence": consequence},
            )
    write_event(
        events_path,
        trace_id=trace_id,
        run_id=run_id,
        instance=instance,
        condition=condition,
        event_type="final_consequence",
        layer="M",
        component="task_evaluator",
        status="preserved" if outcome["final_task_success"] else "failed",
        effect=outcome["benchmark_consequence"],
        label="propagated" if applied else "pre_injection",
        evidence={"task_success": outcome["final_task_success"], "repair_rounds": len(round_records)},
    )
    row = {
        "run_id": run_id,
        "trace_id": trace_id,
        "benchmark": "SWE-bench Verified",
        "execution_mode": "official_swe_harness_multiround",
        "patch_format": patch_format,
        "scenario": "code_repair_feedback_loop",
        "task_id": instance.instance_id,
        "instance_id": instance.instance_id,
        "condition": condition,
        "fault_type": condition,
        "fault_id": "none" if condition == "clean" else f"fault-{run_id}",
        "fault_applied": applied,
        "fault_round": fault_round,
        "injection_edge": "tester_to_coder",
        "source_agent": "Tester",
        "target_agent": "Coder",
        "model": repair_client.model_info.model,
        "initial_model": client.model_info.model,
        "repair_model": repair_client.model_info.model,
        "provider": client.model_info.provider,
        "original_message": evidence_payload(original_feedback),
        "delivered_message": evidence_payload(injected_feedback),
        "observed_runtime_effect": observed_a_symptom,
        "first_divergence": "A:fault_applied" if applied else "none",
        "planner_output": plan,
        "repair_rounds": round_records,
        "fault_feedback_assessment": fault_feedback_assessment,
        "final_evidence": evidence_payload(evidence),
        "verifier_output": verifier,
        "context_metadata": context_metadata,
        "official_test_passed": evidence.passed,
        "official_runner_error": evidence.runner_error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "api_call_count": sum(item.call_count for item in clients) - calls_before,
        "prompt_tokens": sum(item.prompt_tokens for item in clients) - prompt_before,
        "completion_tokens": sum(item.completion_tokens for item in clients) - completion_before,
        **outcome,
    }
    row["total_tokens"] = row["prompt_tokens"] + row["completion_tokens"]
    return normalize_run_record(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real multi-round SWE-bench feedback-fault experiments.")
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--instance-ids", nargs="*", default=[])
    parser.add_argument("--conditions", nargs="+", default=["clean", "a5_omission", "a8_truncation", "a12_stale_replay"])
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--fault-round", type=int, default=1)
    parser.add_argument("--repair-model", default=None, help="Optional real API model used after Tester feedback.")
    parser.add_argument("--repair-enable-thinking", action="store_true", help="Enable native reasoning only for the repair client.")
    parser.add_argument("--patch-format", choices=("json_edits", "unified_diff"), default="json_edits")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default="data/modelscope_swe_bench_verified")
    parser.add_argument("--context-budget-chars", type=int, default=8000)
    parser.add_argument("--source-context-chars", type=int, default=6000)
    parser.add_argument("--test-context-chars", type=int, default=1200)
    args = parser.parse_args()
    load_dotenv()
    load_dotenv(".env.local")
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, resume=False)
    client, harness = get_llm_client(), SWEHarness()
    repair_client = client
    if args.repair_model:
        original_model = os.environ.get("LLM_MODEL")
        original_disable_thinking = os.environ.get("LLM_DISABLE_THINKING")
        os.environ["LLM_MODEL"] = args.repair_model
        if args.repair_enable_thinking:
            os.environ["LLM_DISABLE_THINKING"] = "0"
        try:
            repair_client = get_llm_client()
        finally:
            if original_model is None:
                os.environ.pop("LLM_MODEL", None)
            else:
                os.environ["LLM_MODEL"] = original_model
            if original_disable_thinking is None:
                os.environ.pop("LLM_DISABLE_THINKING", None)
            else:
                os.environ["LLM_DISABLE_THINKING"] = original_disable_thinking
    candidates, raw_instances = load_verified_instances(Path(args.dataset_dir), 500 if args.instance_ids else max(args.instances * 4, args.instances))
    wanted = set(args.instance_ids)
    if wanted:
        candidates = [item for item in candidates if item.instance_id in wanted]
    instances, skipped = select_runnable_instances(candidates, args.instances, image_exists=local_image_exists)
    if not instances:
        raise SystemExit("no selected SWE-bench instances have a local evaluation image")
    selected_ids = {item.instance_id for item in instances}
    (output_dir / "swe_bench_verified_instances.json").write_text(
        json.dumps([record for record in raw_instances if record["instance_id"] in selected_ids], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "experiment_config.json").write_text(json.dumps({
        "benchmark": "SWE-bench Verified",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": client.model_info.model,
        "repair_model": repair_client.model_info.model,
        "repair_enable_thinking": args.repair_enable_thinking,
        "patch_format": args.patch_format,
        "provider": client.model_info.provider,
        "conditions": args.conditions,
        "max_repair_rounds": args.max_repair_rounds,
        "fault_round": args.fault_round,
        "injection_edge": "tester_to_coder",
        "selected_instances": sorted(selected_ids),
        "skipped_instances": skipped,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for condition in args.conditions:
        for instance in instances:
            context, metadata = collect_repository_context_with_metadata(
                instance,
                ContextBudget(
                    max_total_chars=args.context_budget_chars,
                    max_source_chars=args.source_context_chars,
                    max_test_chars=args.test_context_chars,
                ),
            )
            row = run_multiround_instance(
                client,
                harness,
                instance,
                condition=condition,
                output_dir=output_dir,
                repository_context=context,
                context_metadata=metadata,
                max_repair_rounds=args.max_repair_rounds,
                fault_round=args.fault_round,
                repair_client=repair_client,
                patch_format=args.patch_format,
            )
            rows.append(row)
            append_jsonl_record(output_dir / "swe_bench_runs.jsonl", row)
    fields = sorted({field for row in rows for field in row})
    with (output_dir / "swe_bench_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    summary = {
        "runs": len(rows),
        "fault_applied": sum(bool(row["fault_applied"]) for row in rows),
        "final_success": sum(bool(row["final_task_success"]) for row in rows),
        "final_failure": sum(not bool(row["final_task_success"]) for row in rows),
        "recovered": sum(bool(row["recovery_detected"]) for row in rows),
    }
    (output_dir / "swe_bench_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_report(output_dir / "swe_bench_causal_events.jsonl", output_dir / "causal_trace_report")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
