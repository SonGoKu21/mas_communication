#!/usr/bin/env python3
"""Four-cart HTTP isolation only: no LLM, preflight, throughput or formal trials.

Owns the existing workflow lock throughout execution and private report writes.
Selects the first two frozen product pairs without synthesizing task variants.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_multimechanism_parallel import (
    MultiStateShoppingExecutor, check_evidence, make_executor, runner, verify_http_receipts,
)


def select_tasks(manifest, base_url):
    tasks = runner.load_tasks(manifest)
    groups = {}
    for task in tasks:
        url = MultiStateShoppingExecutor._canonical_url(task["product_url"])
        groups.setdefault(url, []).append(task)
    if len(groups) < 2:
        raise ValueError("two frozen product pairs required")
    base = urlsplit(MultiStateShoppingExecutor._canonical_url(runner.validate_loopback_url(base_url)))
    selected = []
    for url, pair in list(groups.items())[:2]:
        parsed = urlsplit(url)
        if ((parsed.scheme, parsed.netloc) != (base.scheme, base.netloc) or len(pair) != 2
                or pair[0]["product_title"] != pair[1]["product_title"]
                or pair[0]["quantity"] == pair[1]["quantity"]):
            raise ValueError("each frozen pair requires one Shopping product and distinct target quantities")
        selected.extend(copy.deepcopy(pair))
    return selected


def http_isolation_gate(tasks, base_url, *, executor_factory=None):
    runner.matrix.validate_tasks(tasks)
    if len(tasks) != 4:
        raise ValueError("exactly four frozen tasks required")
    factory = executor_factory or make_executor
    executors, sessions, mutex = [None] * 4, {}, Lock()
    flows = [{"task": copy.deepcopy(task), "errors": [], "bindings": []} for task in tasks]
    errors = []

    def issue(index, stage, error_type):
        flows[index]["errors"].append({"stage": stage, "error_type": error_type})

    def capture(index, stage):
        executor = executors[index]
        session = getattr(executor, "session", None)
        if session is None:
            raise ValueError("HTTP session missing")
        with mutex:
            sessions[id(session)] = session
        flows[index]["bindings"].append({"stage": stage, "session_object_id": id(session),
                                          "cart_id_sha256": executor._cart_hash()})

    def evidence(index, key, task, operation):
        value = operation()
        flows[index][key] = copy.deepcopy(value)
        capture(index, key)
        if not check_evidence(task, value).valid:
            raise ValueError("cart evidence invalid")

    def initialize(index):
        executor = executors[index] = factory(base_url)
        capture(index, "created")
        initial = {**copy.deepcopy(tasks[index]), "quantity": tasks[index]["initial_quantity"]}
        evidence(index, "initial_ack", initial, lambda: executor.add_to_cart(initial))
        evidence(index, "initial", initial, lambda: executor.reobserve_cart(initial))

    def update(index):
        task, executor = copy.deepcopy(tasks[index]), executors[index]
        operation = (lambda: executor.add_quantity(task, task["quantity"] - task["initial_quantity"])) \
            if task["quantity"] > task["initial_quantity"] else lambda: executor.set_quantity(task, task["quantity"])
        evidence(index, "update", task, operation)

    def recheck(index):
        task = copy.deepcopy(tasks[index])
        evidence(index, "recheck", task, lambda: executors[index].reobserve_cart(task))

    def guarded(index, stage, operation):
        try:
            operation(index)
        except Exception as exc:
            issue(index, stage, type(exc).__name__)
        finally:
            executor = executors[index]
            session = getattr(executor, "session", None)
            if session is not None:
                with mutex:
                    sessions[id(session)] = session

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for stage, operation in (("initialize", initialize), ("update", update), ("recheck", recheck)):
                if stage == "update" and any(flow["errors"] for flow in flows):
                    break
                # Waiting for every future is the barrier between phases.
                futures = [pool.submit(guarded, i, stage, operation) for i in range(4)]
                for future in futures:
                    future.result()
        for i, executor in enumerate(executors):
            flow = flows[i]
            flow.update(cart_id_sha256=None, session_object_id=None, http_receipts=[], http_receipts_verified=False)
            try:
                if executor is not None:
                    capture(i, "final")
                    flow["cart_id_sha256"] = executor._cart_hash()
                    flow["session_object_id"] = id(executor.session)
                    flow["http_receipts"] = copy.deepcopy(executor.http_receipts)
                flow["http_receipts_verified"] = verify_http_receipts(
                    flow["http_receipts"], flow["cart_id_sha256"], tasks[i], base_url)
                if not flow["http_receipts_verified"]:
                    issue(i, "receipts", "InvalidHTTPReceipts")
                bindings = flow["bindings"]
                if (not bindings or len({b["session_object_id"] for b in bindings}) != 1
                        or any(b["cart_id_sha256"] != flow["cart_id_sha256"] for b in bindings if b["stage"] != "created")):
                    issue(i, "bindings", "ChangedCartOrSession")
                values = [flow.get(k) for k in ("initial_ack", "initial", "update", "recheck")]
                identities = [(v.get("sku"), v.get("product_id")) for v in values if isinstance(v, dict)]
                if len(identities) != 4 or len(set(identities)) != 1:
                    issue(i, "identity", "ChangedProductIdentity")
            except Exception as exc:
                issue(i, "verification", type(exc).__name__)
        for first, second in ((0, 1), (2, 3)):
            a, b = flows[first].get("initial", {}), flows[second].get("initial", {})
            if (not isinstance(a, dict) or not isinstance(b, dict)
                    or (a.get("sku"), a.get("product_id")) != (b.get("sku"), b.get("product_id"))):
                errors.append({"stage": "pair_identity", "error_type": "MismatchedProductPair"})
        carts = [f["cart_id_sha256"] for f in flows]
        session_ids = [f["session_object_id"] for f in flows]
        if not all(carts) or len(set(carts)) != 4:
            errors.append({"stage": "cart_isolation", "error_type": "NonuniqueCarts"})
        if not all(session_ids) or len(set(session_ids)) != 4:
            errors.append({"stage": "session_isolation", "error_type": "NonuniqueHTTPSessions"})
    finally:
        for session_id, session in sessions.items():
            try:
                session.close()
            except Exception as exc:
                for i, flow in enumerate(flows):
                    if any(b["session_object_id"] == session_id for b in flow["bindings"]):
                        issue(i, "close", type(exc).__name__)
    for flow in flows:
        flow["passed"] = not flow["errors"] and flow["http_receipts_verified"]
    return {"gate_type": "four_cart_http_isolation", "max_workers": 4, "model_calls": 0,
            "passed": not errors and all(f["passed"] for f in flows), "flows": flows, "errors": errors,
            "model_performance_verified": False,
            "limitations": ["HTTP cart isolation only; no model performance or formal experiment result.",
                            "Session identities refer to Python HTTP session objects, not server-side authentication."]}


def write_private_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_probe(args, *, workflow_lock=None, executor_factory=None):
    with (workflow_lock or runner.locked_workflow)():
        output = args.output_dir
        if output.exists() or output.is_symlink():
            raise FileExistsError("probe output must be a new directory")
        base_url = runner.validate_loopback_url(args.base_url)
        manifest_bytes = args.task_manifest.read_bytes()
        tasks = select_tasks(args.task_manifest, base_url)
        if args.task_manifest.read_bytes() != manifest_bytes:
            raise ValueError("task manifest changed during selection")
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        gate = http_isolation_gate(tasks, base_url, executor_factory=executor_factory)
        if args.task_manifest.read_bytes() != manifest_bytes:
            gate["errors"].append({"stage": "manifest", "error_type": "ChangedTaskManifest"})
            gate["passed"] = False
        detail = {**gate, "tasks": tasks, "base_url": base_url,
                  "task_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
        write_private_json(output / "four_cart_http_receipts.json", detail)
        summary = {k: gate[k] for k in ("gate_type", "max_workers", "model_calls", "passed", "model_performance_verified")}
        summary.update(status="passed" if gate["passed"] else "failed", task_ids=[t["task_id"] for t in tasks],
                       verified_flows=sum(f["passed"] for f in gate["flows"]),
                       error_count=len(gate["errors"]) + sum(len(f["errors"]) for f in gate["flows"]),
                       formal_matrix_started=False)
        write_private_json(output / "summary.json", summary)
        return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:17770")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        summary = run_probe(parse_args(argv))
        print(json.dumps(summary))
        return 0 if summary["passed"] else 1
    except Exception as exc:
        print(f"Four-cart probe refused or stopped ({type(exc).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
