"""Linux-only real 18-cell acknowledgement-recovery regression, never a formal run.

Supply the original ten-task gate manifest and one exact q1-to-q2 task ID.
Only clean/acknowledgement_loss x baseline/dependency/combined x three
topologies x one repetition are executed. Outputs must be new; no resume.
The existing full-matrix auditor rejects this subset schema. The targeted
outcome report is not a strict audit pass or authorization for a formal run.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import os
import platform
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_shopping_multimechanism as runner

ARMS = ("baseline", "dependency", "combined")
CONDITIONS = ("clean", "acknowledgement_loss")
PLANNED_RUNS = 18


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:17770")
    parser.set_defaults(repetitions=1, shard_index=0, shard_count=1)
    return parser.parse_args(argv)


def build_config(args, tasks):
    # Validate/hash the original manifest before replacing the execution scope.
    config = runner.build_config(args, tasks)
    if len(tasks) != 10 or args.repetitions != 1 or args.shard_index != 0 or args.shard_count != 1:
        raise ValueError("original ten-task manifest and fixed single repetition required")
    selected = [task for task in tasks if task["task_id"] == args.task_id]
    if len(selected) != 1 or selected[0]["initial_quantity"] != 1 or selected[0]["quantity"] != 2:
        raise ValueError("select one exact existing q1-to-q2 task ID")
    product = urlsplit(runner.validate_loopback_url(selected[0]["product_url"]))
    base = urlsplit(config["base_url"])
    if (product.scheme, product.netloc) != (base.scheme, base.netloc):
        raise ValueError("selected product must use the configured local Shopping origin")
    jobs = [job for job in config["jobs"] if job["task_id"] == args.task_id
            and job["arm"] in ARMS and job["condition"] in CONDITIONS]
    if len(jobs) != PLANNED_RUNS or len({job["job_key"] for job in jobs}) != PLANNED_RUNS:
        raise ValueError("unexpected targeted job coverage")
    config.update(tasks=copy.deepcopy(selected), jobs=jobs, planned_runs=PLANNED_RUNS,
                  shard_runs=PLANNED_RUNS, source_policy="not_applicable_no_cross_task_source_conditions",
                  scope={"kind": "ack_recovery_regression_18_v1", "input_task_count": len(tasks),
                         "selected_task_ids": [args.task_id], "arms": list(ARMS),
                         "conditions": list(CONDITIONS), "topologies": list(runner.matrix.TOPOLOGIES),
                         "repetitions": 1, "selection": "exact_task_id_no_task_mutation",
                         "input_manifest_snapshot": "input_task_manifest.json"},
                  audit_compatibility={"compatible": False,
                      "auditor": "scripts/audit_shopping_multimechanism.py",
                      "reason": "full-matrix auditor reconstructs 210 cells per task, not this 18-cell subset",
                      "expected_findings": ["manifest_jobs_mismatch", "planned_count_mismatch"]})
    config["source_hashes"][Path(__file__).resolve().relative_to(ROOT).as_posix()] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    return config


def validate_new_output(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be a new directory")
    if any(part.lower().startswith(("pilot", "formal")) for part in output.absolute().parts):
        raise ValueError("pilot and formal output paths are forbidden")
    for parent in output.absolute().parents:
        if parent.is_symlink() or any((parent / marker).exists() for marker in (
                "matrix_manifest.json", "probe_manifest.json", "main_runs.jsonl", "run_attempts.jsonl")):
            raise ValueError("output cannot be nested in a saved run or symlinked directory")


def gate_report(config, summary, rows, errors):
    expected = {job["job_key"] for job in config["jobs"]}
    limits = {"get": 4, "model_call": 3, "replay": 1}
    criteria = {
        "exact_18_completed": len(rows) == PLANNED_RUNS and {r.get("job_key") for r in rows} == expected,
        "no_error_attempts": not errors,
        "outcomes_successful": bool(rows) and all(all(row.get(field) is True for field in (
            "environment_task_success", "final_task_success", "final_commit_allowed",
            "common_recovery_enabled")) for row in rows),
        "graph_errors_absent": bool(rows) and all(row.get("graph_errors") == [] for row in rows),
        "fault_exposure_recorded": bool(rows) and all(
            row.get("fault_events") == [] if row["condition"] == "clean" else
            isinstance(row.get("fault_events"), list) and len(row["fault_events"]) == 1
            and row["fault_events"][0].get("condition") == "acknowledgement_loss"
            and row["fault_events"][0].get("boundary") == "action_ack"
            and row["fault_events"][0].get("delivered_count") == 0 for row in rows),
        "budgets_preserved": bool(rows) and all(row.get("budget", {}).get("limits") == limits
            and all(type(row["budget"].get("used", {}).get(kind)) is int
                    and 0 <= row["budget"]["used"][kind] <= limit for kind, limit in limits.items())
            for row in rows),
        "usage_known": bool(rows) and all(row.get("usage_complete") is True
            and type(row.get("total_tokens")) is int and row["total_tokens"] >= 0
            and type(row.get("model_calls")) is int and row["model_calls"] > 0 for row in rows),
        "runner_finished": summary.get("status") == "complete" and not summary.get("exhausted_job_keys")
            and not summary.get("source_blocked_job_keys") and not summary["circuit_breaker"]["tripped"],
    }
    passed = all(criteria.values())
    return {"regression_passed": passed, "status": "passed" if passed else "failed",
            "planned_runs": PLANNED_RUNS, "completed_runs": len(rows), "error_attempts": len(errors),
            "criteria": criteria, "formal_matrix_started": False, "formal_matrix_authorized": False,
            "strict_audit_passed": False, "audit_compatibility": config["audit_compatibility"],
            "config_digest": runner.matrix.config_digest(config)}


def run_regression(args):
    if sys.platform != "linux":
        raise ValueError("real acknowledgement gate requires Linux")
    with runner.locked_workflow():
        validate_new_output(args.output_dir)
        tasks = runner.load_tasks(args.task_manifest)
        config = build_config(args, tasks)
        manifest_bytes = args.task_manifest.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != config["task_manifest_sha256"]:
            raise ValueError("task manifest changed during setup")
        args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        with runner.locked_output(args.output_dir):
            digest = runner.freeze_manifest(args.output_dir, config, resume=False)
            runner.freeze_source_snapshot(args.output_dir, config["source_hashes"], root=ROOT, resume=False)
            with (args.output_dir / "input_task_manifest.json").open("xb") as handle:
                handle.write(manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            runner._sync_directory(args.output_dir)
            state = runner.reconcile_attempts(args.output_dir, config["jobs"], digest)
            runner.write_reports(args.output_dir, config["jobs"], state)
            with runner.local_inference_transport():
                client, preflight = runner.preflight_client(config["inference_settings"])
                runner.append_jsonl(args.output_dir / "preflight.jsonl", {**preflight,
                    "config_digest": digest, "python": platform.python_version(),
                    "platform": platform.system(), "resume": False, "max_jobs": PLANNED_RUNS})
                summary = asyncio.run(runner.execute_jobs(config["tasks"], config["jobs"], args.output_dir,
                    digest, client, config["base_url"]))
            # Verification only: never repair or overwrite frozen source bytes.
            runner.freeze_source_snapshot(args.output_dir, config["source_hashes"], root=ROOT, resume=True)
            if args.task_manifest.read_bytes() != manifest_bytes:
                raise ValueError("original task manifest changed during execution")
            report = gate_report(config, summary, runner.read_jsonl(args.output_dir / "main_runs.jsonl"),
                                 runner.read_jsonl(args.output_dir / "run_errors.jsonl"))
            runner._write_json(args.output_dir / "ack_recovery_gate.json", report, exclusive=True)
            return report


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run_regression(args)
        print(f"Ack recovery regression: {report['status']}; {report['completed_runs']}/18 completed. "
              "Not a strict audit pass or formal-run authorization.")
        return 0 if report["regression_passed"] else 1
    except Exception as exc:
        print(f"Ack regression stopped ({type(exc).__name__}); exception text suppressed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
