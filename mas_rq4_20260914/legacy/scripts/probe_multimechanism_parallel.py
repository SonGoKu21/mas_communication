"""Bounded Linux-only local-Qwen isolation/throughput probe, never a formal runner.

Run only in an idle workflow-lock window. The HTTP gate precedes model preflight.
Two crossover rounds execute exactly eight clean baseline trials without retries.
All detailed receipts remain in a new, private server-side output directory.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_shopping_multimechanism as runner
from mas_faults.shopping_action_protocol import MultiStateShoppingExecutor
from mas_faults.shopping_mitigation import check_evidence


def select_tasks(manifest, task_ids, base_url):
    """Select exact manifest entries; never synthesize quantity variants."""
    tasks = runner.load_tasks(manifest)
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("exactly two distinct manifest task IDs are required")
    by_id = {t["task_id"]: t for t in tasks}
    if any(key not in by_id for key in task_ids):
        raise ValueError("selected task ID is absent from the manifest")
    selected = [copy.deepcopy(by_id[key]) for key in task_ids]
    base = urlsplit(MultiStateShoppingExecutor._canonical_url(runner.validate_loopback_url(base_url)))
    urls = [MultiStateShoppingExecutor._canonical_url(t["product_url"]) for t in selected]
    if any((urlsplit(url).scheme, urlsplit(url).netloc) != (base.scheme, base.netloc) for url in urls):
        raise ValueError("task URL must use the configured Shopping origin")
    if (urls[0] != urls[1] or selected[0]["product_title"] != selected[1]["product_title"]
            or selected[0]["quantity"] == selected[1]["quantity"]):
        raise ValueError("probe requires the same product with two different target quantities")
    return selected


def make_executor(base_url):
    executor = MultiStateShoppingExecutor(base_url)
    executor.session.trust_env = False
    return executor


def _close_executor(executor):
    session = getattr(executor, "session", None)
    if session is not None:
        session.close()


def verify_http_receipts(receipts, cart_hash, task, base_url):
    """Validate the actual ShoppingHTTPExecutor/MultiState receipt contract.

    Only the first product GET precedes cart creation. The base executor fills
    the creation receipt's cart hash after receiving the token; every later
    receipt, including product GETs, must remain bound to that same cart.
    """
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    if not digest(cart_hash) or not isinstance(receipts, list) or len(receipts) < 5:
        return False
    base = urlsplit(MultiStateShoppingExecutor._canonical_url(base_url))
    product = urlsplit(MultiStateShoppingExecutor._canonical_url(task["product_url"]))
    cart_root = base.path.rstrip("/") + "/rest/V1/guest-carts"
    items = cart_root + "/[REDACTED]/items"
    methods_paths = {
        "add_to_cart.product_page": ("GET", product.path),
        "add_to_cart.create_cart": ("POST", cart_root),
        "add_to_cart.add_item": ("POST", items),
        "reobserve_cart.product_page": ("GET", product.path),
        "reobserve_cart.items": ("GET", items),
        "verify_cart.items": ("GET", items),
        "add_quantity.product_page": ("GET", product.path),
        "add_quantity.lookup_items": ("GET", items),
        "add_quantity.add_item": ("POST", items),
        "add_quantity.readback_items": ("GET", items),
        "set_quantity.product_page": ("GET", product.path),
        "set_quantity.lookup_items": ("GET", items),
        "set_quantity.readback_items": ("GET", items),
    }
    for index, receipt in enumerate(receipts):
        if (not isinstance(receipt, dict) or type(receipt.get("receipt_index")) is not int
                or receipt["receipt_index"] != index or type(receipt.get("status_code")) is not int
                or not 200 <= receipt["status_code"] < 300 or receipt.get("error_type")
                or not digest(receipt.get("response_sha256"))
                or receipt.get("response_hash_source") not in {"content", "text_utf8"}):
            return False
        purpose = receipt.get("purpose")
        if index < 3:
            if purpose != ("add_to_cart.product_page", "add_to_cart.create_cart", "add_to_cart.add_item")[index]:
                return False
        elif purpose in {"add_to_cart.product_page", "add_to_cart.create_cart", "add_to_cart.add_item"}:
            return False
        if receipt.get("guest_cart_id_sha256") != (None if index == 0 else cart_hash):
            return False
        url = receipt.get("request_url")
        if not isinstance(url, str):
            return False
        try:
            parsed = urlsplit(url)
            if ((parsed.scheme, parsed.netloc) != (base.scheme, base.netloc)
                    or parsed.username is not None or parsed.password is not None
                    or MultiStateShoppingExecutor._receipt_url(url) != url):
                return False
        except ValueError:
            return False
        if purpose == "set_quantity.set_item":
            suffix = parsed.path.removeprefix(items + "/")
            if not parsed.path.startswith(items + "/") or not suffix.isdecimal() or int(suffix) <= 0:
                return False
            expected = ("PUT", parsed.path)
        else:
            expected = methods_paths.get(purpose)
        if expected != (receipt.get("request_method"), parsed.path):
            return False
    return any(r["purpose"] in {"reobserve_cart.items", "verify_cart.items"} for r in receipts[3:])


def _trial_receipts_verified(row, task, base_url):
    receipts, result = row.get("http_receipts"), row.get("result", {})
    return bool(isinstance(result, dict) and result.get("http_receipts", receipts) == receipts
                and verify_http_receipts(receipts, row.get("cart_id_sha256"), task, base_url))


def http_isolation_gate(tasks, base_url, *, executor_factory=None):
    """Two concurrent carts; both final reads start after both update attempts end."""
    factory = executor_factory or make_executor
    executors = [None, None]
    flows = [{"task": copy.deepcopy(t), "errors": []} for t in tasks]

    def initialize(index):
        executor = executors[index] = factory(base_url)
        initial = {**tasks[index], "quantity": tasks[index]["initial_quantity"]}
        flows[index]["initial_ack"] = executor.add_to_cart(initial)
        flows[index]["initial"] = executor.reobserve_cart(initial)
        if not check_evidence(initial, flows[index]["initial"]).valid:
            raise ValueError("initial cart verification failed")

    def update(index):
        t, executor = tasks[index], executors[index]
        flows[index]["update"] = (executor.add_quantity(t, t["quantity"] - t["initial_quantity"])
                                   if t["quantity"] > t["initial_quantity"] else executor.set_quantity(t, t["quantity"]))

    def recheck(index):
        flows[index]["recheck"] = executors[index].reobserve_cart(tasks[index])

    def guarded(stage, index, operation):
        try:
            operation(index)
        except Exception as exc:
            flows[index]["errors"].append({"stage": stage, "error_type": type(exc).__name__})

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for stage, operation in (("initialize", initialize), ("update", update), ("recheck", recheck)):
                if stage == "update" and any(f["errors"] for f in flows):
                    break
                futures = [pool.submit(guarded, stage, i, operation) for i in range(2)]
                for future in futures:
                    future.result()
        for i, executor in enumerate(executors):
            flows[i]["cart_id_sha256"] = executor._cart_hash() if executor is not None else None
            flows[i]["http_receipts"] = copy.deepcopy(getattr(executor, "http_receipts", []))
            receipts = flows[i]["http_receipts"]
            flows[i]["http_receipts_verified"] = verify_http_receipts(receipts, flows[i]["cart_id_sha256"], tasks[i], base_url)
            flows[i]["passed"] = bool(
                not flows[i]["errors"] and check_evidence(tasks[i], flows[i].get("update")).valid
                and check_evidence(tasks[i], flows[i].get("recheck")).valid
                and flows[i]["http_receipts_verified"])
        identities = [(f.get("initial", {}).get("sku"), f.get("initial", {}).get("product_id")) for f in flows]
        identities += [(f.get("recheck", {}).get("sku"), f.get("recheck", {}).get("product_id")) for f in flows]
        hashes = [f["cart_id_sha256"] for f in flows]
        passed = (all(f["passed"] for f in flows) and all(hashes) and len(set(hashes)) == 2
                  and len(set(identities)) == 1)
        return {"passed": bool(passed), "gate_type": "two_cart_http_isolation", "max_workers": 2,
                "model_calls": 0, "flows": flows}
    finally:
        for executor in executors:
            _close_executor(executor)


def checked_usage(client, before, *, complete):
    """Missing/zero-defaulted usage is unknown, not free inference."""
    usage = runner.usage_since(client, before, complete=complete)
    logs = usage["model_requests"]
    verified = bool(usage["usage_complete"] and usage["model_calls"] > 0 and logs
                    and all(type(r.get(k)) is int and r[k] > 0 for r in logs
                            for k in ("prompt_tokens", "completion_tokens"))
                    and all(r.get("model") == runner.MODEL and r.get("provider") == runner.PROVIDER for r in logs))
    if verified:
        verified = all(sum(r[k] for r in logs) == usage[k] for k in ("prompt_tokens", "completion_tokens"))
    usage["usage_complete"] = verified
    if not verified:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = None
    return usage


def _real_client(settings):
    if sys.platform != "linux" or runner.inference_settings() != settings or not runner.os.environ.get("LLM_API_KEY"):
        raise ValueError("Linux and unchanged explicit local inference settings are required")
    client = runner.get_llm_client()
    info = client.model_info
    if (info.model != runner.MODEL or info.provider != runner.PROVIDER or info.client_type != "http_openai_compatible"
            or runner.openai_api_base(runner.validate_loopback_url(info.base_url)) != settings["api_base_url"]):
        raise ValueError("trial client does not match preflight configuration")
    return client


def _unknown_usage():
    return {"usage_complete": False, "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "model_calls": None, "model_requests": []}


def execute_trial(spec, *, client_factory=None, executor_factory=None, trial=None):
    """Spawn-worker entry: new client/executor/runtime/ledger for one attempt.

    Dependency injection is for offline tests, never a CLI simulation switch.
    Each process owns its urllib opener and Unix signal-based request deadlines.
    """
    if trial is None:
        from mas_faults.shopping_multimechanism import run_trial
        trial = run_trial
    started = time.perf_counter()
    client = executor = before = None
    row = {key: spec[key] for key in ("trial_id", "round", "mode")}
    row.update(task_id=spec["task"]["task_id"], attempt=1, model=runner.MODEL, provider=runner.PROVIDER,
               ledger_path=spec["ledger_path"], passed=False)
    try:
        with runner.local_inference_transport():
            client = (client_factory or _real_client)(spec["settings"])
            before = runner.usage_snapshot(client)
            if before["call_count"] != 0 or before["log_length"] != 0:
                raise ValueError("each trial requires an unused client")
            executor = (executor_factory or make_executor)(spec["base_url"])
            result = asyncio.run(trial(copy.deepcopy(spec["task"]), copy.deepcopy(spec["job"]), executor, client,
                                       ledger_path=Path(spec["ledger_path"])))
            if not isinstance(result, dict):
                raise TypeError("run_trial must return an object")
            usage = checked_usage(client, before, complete=True)
            row.update(result=result, **usage, status="completed")
            row["passed"] = bool(usage["usage_complete"] and result.get("final_task_success") is True
                                 and result.get("environment_task_success") is True and result.get("decision_correct") is True
                                 and result.get("evidence_acceptance_errors") == 0 and result.get("fault_events") == []
                                 and not result.get("action_outcome_unknown")
                                 and result.get("model") == runner.MODEL and result.get("provider") == runner.PROVIDER)
    except Exception as exc:
        row.update(status="error", error_type=type(exc).__name__, partial_trial=runner.partial_trial_context(exc))
        row.update(checked_usage(client, before, complete=False) if client is not None and before is not None else _unknown_usage())
    finally:
        row["http_receipts"] = copy.deepcopy(getattr(executor, "http_receipts", []))
        row["cart_id_sha256"] = executor._cart_hash() if executor is not None else None
        row["latency_seconds"] = time.perf_counter() - started
        _close_executor(executor)
    row["http_receipts_verified"] = _trial_receipts_verified(row, spec["task"], spec["base_url"])
    row["passed"] = row["passed"] is True and row["http_receipts_verified"]
    json.dumps(row, allow_nan=False)
    return row


def _process_pool(max_workers):
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))


def run_batches(tasks, output, settings, base_url, topology, *, trial_runner=None, pool_factory=None, clock=None):
    trial_runner, pool_factory, clock = trial_runner or execute_trial, pool_factory or _process_pool, clock or time.perf_counter
    output = Path(output)
    (output / "action_ledgers").mkdir(exist_ok=False)
    trials, batches = [], []
    for number, (round_number, mode) in enumerate(((1, "sequential"), (1, "parallel"), (2, "parallel"), (2, "sequential"))):
        specs = []
        for task in tasks:
            trial_id = uuid.uuid4().hex
            job = {"task_id": task["task_id"], "topology": topology, "arm": "baseline", "condition": "clean",
                   "boundary": None, "repeat_index": round_number, "job_key": trial_id,
                   "pair_key": f"probe:{round_number}:{task['task_id']}", "attempt": 1}
            spec = {"trial_id": trial_id, "round": round_number, "mode": mode, "task": copy.deepcopy(task),
                    "job": job, "settings": settings, "base_url": base_url,
                    "ledger_path": str((output / "action_ledgers" / (trial_id + ".sqlite3")).resolve())}
            specs.append(spec)
            runner.append_jsonl(output / "trial_attempts.jsonl", {**spec, "attempt": 1})
        batch_rows = []
        width = 2 if mode == "parallel" else 1
        started = clock()
        with pool_factory(max_workers=width) as pool:
            pending = {pool.submit(trial_runner, spec): spec for spec in specs}
            for future in as_completed(pending):
                spec = pending[future]
                try:
                    row = future.result()
                    if not isinstance(row, dict) or row.get("trial_id") != spec["trial_id"]:
                        raise ValueError("worker result is not bound to its trial")
                except Exception as exc:
                    row = {"trial_id": spec["trial_id"], "task_id": spec["task"]["task_id"], "round": round_number,
                           "mode": mode, "status": "error", "passed": False, "error_type": type(exc).__name__,
                           "attempt": 1, "latency_seconds": None, "model": runner.MODEL, "provider": runner.PROVIDER,
                           **_unknown_usage()}
                row["http_receipts_verified"] = _trial_receipts_verified(row, spec["task"], base_url)
                row["passed"] = bool(row.get("passed") is True and row["http_receipts_verified"]
                                     and row.get("status") == "completed" and row.get("usage_complete") is True)
                runner.append_jsonl(output / "trials.jsonl", row)
                batch_rows.append(row)
        wall = clock() - started
        passed = sum(r.get("passed") is True and r.get("usage_complete") is True for r in batch_rows)
        batch = {"batch_index": number, "round": round_number, "mode": mode, "worker_limit": width,
                 "wall_seconds": wall, "trial_ids": [r["trial_id"] for r in batch_rows], "trial_count": len(batch_rows),
                 "passed_trials": passed, "passed": passed == 2 and wall > 0,
                 "completed_throughput_per_second": sum(r.get("status") == "completed" for r in batch_rows) / wall if wall > 0 else None,
                 "successful_throughput_per_second": passed / wall if wall > 0 else None}
        runner.append_jsonl(output / "batches.jsonl", batch)
        batches.append(batch)
        trials.extend(batch_rows)
    rounds = []
    for number in (1, 2):
        pair = {b["mode"]: b for b in batches if b["round"] == number}
        seq, par = pair["sequential"]["wall_seconds"], pair["parallel"]["wall_seconds"]
        rounds.append({"round": number, "sequential_wall_seconds": seq, "parallel_wall_seconds": par,
                       "comparable": pair["sequential"]["passed"] and pair["parallel"]["passed"],
                       "speedup": seq / par if seq > 0 and par > 0 else None})
    usage_complete = all(r.get("usage_complete") is True for r in trials)
    by_mode = {}
    for mode in ("sequential", "parallel"):
        selected = [r for r in trials if r["mode"] == mode]
        wall = sum(b["wall_seconds"] for b in batches if b["mode"] == mode)
        known = all(r.get("usage_complete") is True for r in selected)
        completed = sum(r.get("status") == "completed" for r in selected)
        passed = sum(r.get("passed") is True and r.get("usage_complete") is True for r in selected)
        by_mode[mode] = {"wall_seconds": wall, "passed_trials": sum(r.get("passed") is True for r in selected),
                         "completed_trials": completed, "failed_trials": len(selected) - passed,
                         "completed_throughput_per_second": completed / wall if wall > 0 else None,
                         "successful_throughput_per_second": passed / wall if wall > 0 else None,
                         "latencies_seconds": [r.get("latency_seconds") for r in selected],
                         "prompt_tokens": sum(r["prompt_tokens"] for r in selected) if known else None,
                         "completion_tokens": sum(r["completion_tokens"] for r in selected) if known else None,
                         "total_tokens": sum(r["total_tokens"] for r in selected) if known else None,
                         "model_calls": sum(r["model_calls"] for r in selected) if all(type(r.get("model_calls")) is int for r in selected) else None}
    carts = [r.get("cart_id_sha256") for r in trials]
    sessions = [r.get("result", {}).get("session_id") for r in trials]
    isolated = all(isinstance(value, str) and value for value in carts + sessions)
    isolated = bool(isolated and len(set(carts)) == len(set(sessions)) == 8
                    and all(r["http_receipts_verified"] for r in trials))
    return {"passed": len(trials) == 8 and all(b["passed"] for b in batches) and usage_complete and isolated,
            "trial_isolation_verified": isolated,
            "trials": trials, "batches": batches, "rounds": rounds, "by_mode": by_mode,
            "usage_complete": usage_complete, "total_tokens": sum(r["total_tokens"] for r in trials) if usage_complete else None,
            "two_way_faster_in_both_rounds": isolated and all(r["comparable"] and r["speedup"] is not None and r["speedup"] > 1 for r in rounds)}


def write_reports(output, summary):
    runner._write_json(output / "summary.json", summary)
    fields = ("trial_id", "task_id", "round", "mode", "status", "passed", "model", "provider",
              "latency_seconds", "usage_complete", "prompt_tokens", "completion_tokens", "total_tokens", "model_calls", "error_type")
    with (output / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in summary.get("trials", []))
    lines = ["# 两路并行有界验证", "", f"- 状态：{summary['status']}", f"- 验证通过：{summary['passed']}",
             "- 当前先导不变；本探针不会启动正式矩阵。", "- 原始收据只保存在服务器本目录，不下载。",
             "- 批次时间包含进程启动、排队和收尾；不是并发任务耗时之和。",
             "- 共享模型可能提高单任务延迟；两轮结果不代表统计显著性或必然加速。",
             "- 缺失或零默认 token 使用量按未知处理，不按零成本通过。", ""]
    for row in summary.get("rounds", []):
        lines.append(f"- 第 {row['round']} 轮：串行 {row['sequential_wall_seconds']:.3f}s；双路 {row['parallel_wall_seconds']:.3f}s；实际比值 {row['speedup']}。")
    (output / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args, *, workflow_lock=None, transport=None, preflight=None, executor_factory=None,
              batch_runner=None, platform_name=None):
    """Own the host lock before output/network; injected helpers are unit-only."""
    if (platform_name or sys.platform) != "linux":
        raise ValueError("real probe requires the Linux server")
    with (workflow_lock or runner.locked_workflow)():
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError("probe output must be a new directory")
        settings = runner.inference_settings()
        manifest_bytes = args.task_manifest.read_bytes()
        tasks = select_tasks(args.task_manifest, args.task_ids, args.base_url)
        source_paths = set(ROOT.glob("*.py")) | set((ROOT / "src").rglob("*.py")) | {Path(__file__).resolve()}
        hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
        if args.task_manifest.read_bytes() != manifest_bytes:
            raise ValueError("task manifest changed during setup")
        config = {"probe_schema": 1, "tasks": tasks, "task_ids": args.task_ids, "task_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "source_hashes": hashes, "settings": settings, "base_url": args.base_url, "topology": args.topology,
                  "schedule": [[1, "sequential"], [1, "parallel"], [2, "parallel"], [2, "sequential"]],
                  "model": runner.MODEL, "provider": runner.PROVIDER, "max_workers": 2, "attempts_per_trial": 1,
                  "python": platform.python_version(), "platform": platform.platform(),
                  "dependencies": {name: importlib.metadata.version(name) for name in ("requests", "autogen-core")}}
        args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        runner._write_json(args.output_dir / "probe_manifest.json", {"config": config, "config_digest": runner.matrix.config_digest(config)}, exclusive=True)
        with (args.output_dir / "input_task_manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)
        runner.freeze_source_snapshot(args.output_dir, hashes, root=ROOT, resume=False)
        with (transport or runner.local_inference_transport)():
            gate = http_isolation_gate(tasks, args.base_url, executor_factory=executor_factory)
            runner._write_json(args.output_dir / "http_gate.json", gate, exclusive=True)
            summary = {"passed": False, "status": "http_gate_failed", "http_gate_passed": gate["passed"],
                       "model": runner.MODEL, "provider": runner.PROVIDER, "formal_matrix_started": False,
                       "config_digest": runner.matrix.config_digest(config)}
            if gate["passed"]:
                try:
                    _, checked = (preflight or runner.preflight_client)(settings)
                    runner._write_json(args.output_dir / "preflight.json", checked, exclusive=True)
                    summary.update((batch_runner or run_batches)(tasks, args.output_dir, settings, args.base_url, args.topology))
                    summary["status"] = "passed" if summary["passed"] else "throughput_validation_failed"
                except Exception as exc:
                    summary.update(status="probe_error", passed=False, error_type=type(exc).__name__)
            unchanged = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
            summary["source_unchanged"] = unchanged
            if not unchanged:
                summary.update(passed=False, status="source_changed")
            write_reports(args.output_dir, summary)
            return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--task-ids", required=True, nargs=2, metavar=("TASK_A", "TASK_B"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:17770")
    parser.add_argument("--topology", choices=runner.matrix.TOPOLOGIES, default="sequential")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        summary = run_probe(parse_args(argv))
        print(json.dumps({"status": summary["status"], "passed": summary["passed"]}))
        return 0 if summary["passed"] else 1
    except Exception as exc:
        print(f"Probe refused or stopped ({type(exc).__name__}); busy workflow locks are never bypassed. Details stay server-side.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
