#!/usr/bin/env python3
"""Read-only, offline Shopping pilot audit; never import or run the live runner.

Exit codes: 0 = no integrity errors (inspect status/unknowns), 1 = audit errors,
2 = invalid CLI/input/output. --source-root is for hashing only, not execution.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mas_faults.shopping_mitigation import check_evidence

MODES = ("baseline", "guarded_recheck", "always_recheck")
TOPOLOGIES = ("sequential", "flat", "hierarchical")
FAULTS = ("none", "omission", "message_corruption", "valid_partial", "stale_replay")
JOB_FIELDS = ("job_key", "pair_key", "task_id", "topology", "fault", "injection_step",
              "repeat_index", "mitigation_mode")
DERIVED_FIELDS = ("evaluator_version", "final_task_success", "task_score", "final_decision_correct",
                  "observed_M_consequence", "semantic_consequences", "system_consequences",
                  "mitigation_detected", "recovery_detected", "mitigation_recovery_detected",
                  "common_recovery_detected", "common_recovery_attempted", "recovery_type",
                  "recovery_evidence", "propagation_class", "m_propagated", "final_answer")
CORE_SOURCES = ("run_shopping_mitigation.py", "run_webarena_architecture_rq2.py",
                "src/mas_faults/shopping_mitigation.py")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def payload_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def load_evaluator():
    # Execute only the existing pure evaluator definitions, never runner imports/main.
    path = ROOT / "run_shopping_mitigation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if
                isinstance(node, ast.FunctionDef) and node.name in ("_matches_environment", "audit_outcome")
                or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VERSION"
                                                       for t in node.targets)]
    if len(selected) != 3:
        raise ValueError("offline evaluator definitions are unavailable")
    namespace = {"copy": copy, "check_evidence": check_evidence}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["audit_outcome"], namespace["VERSION"]


class Auditor:
    def __init__(self, result_dir, source_root):
        self.root = Path(result_dir).resolve()
        if not self.root.is_dir():
            raise ValueError("result directory does not exist")
        self.source_root = Path(source_root).resolve() if source_root is not None else None
        self.findings = []
        self.inputs = {}
        self.partial = False

    def issue(self, code, message, row=None, severity="error", **details):
        item = {"severity": severity, "code": code, "message": message, **details}
        if row is not None:
            item.update(run_id=row.get("run_id"), job_key=row.get("job_key"))
        self.findings.append(item)

    def read_bytes(self, relative, required=False):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            self.issue("input_path_escape", "输入文件指向结果目录外", file=str(relative))
            self.partial = True
            return None
        if not path.is_file():
            self.inputs[str(relative)] = {"present": False}
            if required:
                self.issue("missing_input", "快照缺少必要文件", severity="unknown", file=str(relative))
                self.partial = True
            return None
        data = path.read_bytes()
        self.inputs[str(relative)] = {"present": True, "bytes": len(data), "sha256": sha256(data)}
        return data

    def read_json(self, relative, required=False):
        data = self.read_bytes(relative, required)
        if data is None:
            return {}
        try:
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("not an object")
            return value
        except (ValueError, UnicodeDecodeError):
            self.issue("invalid_json", "JSON 对象无法解析", file=str(relative))
            self.partial = True
            return {}

    def read_jsonl(self, relative, required=False):
        data = self.read_bytes(relative, required)
        if data is None:
            return []
        rows = []
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                rows.append(value)
            except (ValueError, UnicodeDecodeError):
                torn = index == len(lines) - 1 and not line.endswith(b"\n")
                self.issue("torn_jsonl_tail" if torn else "invalid_jsonl",
                           "保留损坏记录原文；不修复、不补造结果", file=str(relative),
                           line=index + 1, severity="unknown" if torn else "error")
                self.partial = True
        return rows

    def check_model(self, row, manifest):
        if row.get("model") != manifest.get("model") or not row.get("model"):
            self.issue("model_mismatch", "模型与冻结清单不一致", row)
        if row.get("provider") != "modelscope_local":
            self.issue("provider_mismatch", "provider 必须为 modelscope_local", row)

    def check_manifest(self, manifest):
        tasks = manifest.get("tasks")
        jobs = manifest.get("jobs")
        repetitions = manifest.get("repetitions")
        if not isinstance(tasks, list) or not tasks or not isinstance(jobs, list):
            self.issue("manifest_schema", "manifest 必须包含 tasks 和 jobs")
            self.partial = True
            return {}, {}, {}
        task_map = {}
        for task in tasks:
            if (not isinstance(task, dict) or not isinstance(task.get("task_id"), str)
                    or not all(task.get(k) for k in ("task_id", "product_title", "product_url"))
                    or not integer(task.get("quantity")) or task["quantity"] < 1):
                self.issue("task_metadata", "冻结任务字段不完整或类型错误")
                continue
            if task["task_id"] in task_map:
                self.issue("duplicate_task_id", "冻结任务标识重复")
            task_map[task["task_id"]] = task
        expected = {}
        if not integer(repetitions) or repetitions < 1:
            self.issue("manifest_repetitions", "重复数必须为正整数")
        else:
            for task_id in task_map:
                for topology in TOPOLOGIES:
                    for fault in FAULTS:
                        for repeat in range(1, repetitions + 1):
                            pair = json.dumps([task_id, topology, fault, 4, repeat], separators=(",", ":"))
                            for mode in MODES:
                                key = pair + ":" + mode
                                expected[key] = dict(zip(JOB_FIELDS, (key, pair, task_id, topology, fault, 4, repeat, mode)))
        job_map = {}
        for job in jobs:
            if not isinstance(job, dict) or not isinstance(job.get("job_key"), str):
                self.issue("job_metadata_mismatch", "manifest job 缺少合法 job_key")
                continue
            key = job["job_key"]
            if key in job_map:
                self.issue("duplicate_manifest_job_key", "manifest job_key 重复", job)
            job_map[key] = job
            if key not in expected or any(job.get(k) != expected[key][k] for k in JOB_FIELDS):
                self.issue("job_metadata_mismatch", "manifest job 与固定矩阵元数据不一致", job)
        if set(job_map) != set(expected) or len(jobs) != len(expected):
            self.issue("manifest_jobs_mismatch", "jobs 未覆盖固定 tasks×3拓扑×5条件×3组×重复数",
                       expected_count=len(expected), manifest_count=len(jobs))
            self.partial = True
        for field in ("expected_runs", "planned_runs", "expected_count"):
            if field in manifest and (type(manifest[field]) is not int or manifest[field] != len(expected)):
                self.issue("expected_count_mismatch", "声明的 expected 数量与矩阵不一致", field=field)
        if manifest.get("max_attempts_per_job") != 2:
            self.issue("attempt_budget", "冻结最大尝试数必须为 2")
        common = manifest.get("common_recovery", {})
        if not isinstance(common, dict) or common.get("enabled") is not True:
            self.issue("common_recovery_disabled", "清单没有全组启用共有恢复")
            common = {}
        if common.get("max_readbacks") != 1 or manifest.get("max_readbacks_per_run") != 1:
            self.issue("readback_budget", "共有恢复与新增缓解各自最多一次 readback")
        self.check_model(manifest, manifest)
        settings = manifest.get("inference_settings", {})
        endpoint = settings.get("api_base_url") if isinstance(settings, dict) else None
        if not isinstance(endpoint, str):
            self.issue("local_endpoint_unknown", "未记录模型 endpoint，无法验证本地地址", severity="unknown")
        else:
            try:
                parsed = urlparse(endpoint)
                local = parsed.scheme in ("http", "https") and parsed.hostname in ("localhost", "127.0.0.1", "::1")
            except ValueError:
                local = False
            if not local:
                self.issue("nonlocal_endpoint", "冻结模型 endpoint 不是 loopback 地址")
        return task_map, job_map, expected

    def check_sources(self, manifest):
        hashes = manifest.get("source_hashes")
        result = {"status": "unknown", "verified_files": 0, "source_root": str(self.source_root) if self.source_root else None}
        if not isinstance(hashes, dict) or not hashes:
            self.issue("source_hashes_unknown", "缺少冻结源码 hash", severity="unknown")
            return result
        start = len(self.findings)
        for name in CORE_SOURCES:
            if name not in hashes:
                self.issue("source_hashes_unknown", "缺少关键源码 hash", severity="unknown", file=name)
        for relative, expected in hashes.items():
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts or not valid_hash(expected):
                self.issue("invalid_source_hash", "源码路径或 hash 格式无效", file=relative)
                continue
            if self.source_root is None:
                continue
            path = self.source_root / rel
            if not path.resolve().is_relative_to(self.source_root):
                self.issue("source_path_escape", "源码路径指向 source-root 外", file=relative)
            elif not path.is_file():
                self.issue("source_missing", "source-root 缺少冻结文件", file=relative)
            elif sha256(path.read_bytes()) != expected.lower():
                self.issue("source_hash_mismatch", "源码字节与冻结 hash 不一致", file=relative)
            else:
                result["verified_files"] += 1
        if self.source_root is None:
            self.issue("source_hashes_unknown", "未提供 --source-root，未核验冻结源码字节", severity="unknown")
        new = self.findings[start:]
        result["status"] = "failed" if any(i["severity"] == "error" for i in new) else "unknown" if new else "verified"
        return result

    def check_metadata(self, row, jobs):
        job = jobs.get(row.get("job_key"))
        if job is None:
            self.issue("unexpected_job", "记录不属于 manifest/jobs", row)
            return
        if any(row.get(field) != job.get(field) for field in JOB_FIELDS):
            self.issue("job_metadata_mismatch", "记录与 job 元数据不一致", row)

    def check_usage(self, row, manifest):
        start = len(self.findings)
        fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        known = all(integer(row.get(k)) for k in fields)
        if not known:
            self.issue("token_usage_unknown", "token 字段缺失或无效；总量保持 unknown", row, severity="unknown")
        elif row["total_tokens"] != row["prompt_tokens"] + row["completion_tokens"]:
            self.issue("token_total_mismatch", "total_tokens != prompt_tokens + completion_tokens", row)
        requests = row.get("model_requests")
        if not isinstance(requests, list) or not requests:
            self.issue("request_usage_unknown", "缺少 model_requests/request_log 用量证据", row, severity="unknown")
            return False
        complete = True
        indices = []
        for request in requests:
            if not isinstance(request, dict):
                complete = False
                continue
            self.check_model({**request, "run_id": row.get("run_id"), "job_key": row.get("job_key")}, manifest)
            index = request.get("request_index")
            if not integer(index) or index < 1:
                self.issue("request_index", "request_index 必须是正整数全局编号", row)
            else:
                indices.append(index)
            if not all(integer(request.get(k)) for k in fields[:2]):
                complete = False
            elif "total_tokens" in request and request["total_tokens"] != request["prompt_tokens"] + request["completion_tokens"]:
                self.issue("request_usage_mismatch", "请求自身 token 总数不一致", row)
        if indices != sorted(set(indices)):
            self.issue("request_index", "run 内请求编号须严格递增，但无须从 1 开始", row)
        if not integer(row.get("api_call_count")):
            self.issue("request_usage_unknown", "api_call_count 未记录", row, severity="unknown")
            complete = False
        elif len(requests) != row["api_call_count"]:
            self.issue("request_count_mismatch", "model_requests 数量与 api_call_count 不一致", row)
        if not complete:
            self.issue("request_usage_unknown", "请求 token 明细不全，不能补零或推算", row, severity="unknown")
        elif known and any(sum(request[k] for request in requests) != row[k] for k in fields[:2]):
            self.issue("request_usage_mismatch", "请求 prompt/completion 汇总与 run 不一致", row)
        return complete and known and not any(i["severity"] == "error" for i in self.findings[start:])

    def check_events(self, row):
        events = row.get("events")
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            self.issue("fault_events_unknown", "缺少有效通信事件", row, severity="unknown")
            return
        applied = [e for e in events if e.get("fault_applied") is True]
        fault = row.get("fault_type")
        expected = 0 if fault == "none" else 1
        if len(applied) != expected:
            self.issue("fault_event_count", "clean 必须零次注入，其他条件必须一次", row, observed=len(applied))
        if row.get("fault_applied") is not bool(applied):
            self.issue("fault_event_count", "run fault_applied 与事件不一致", row)
        for event in events:
            if type(event.get("fault_applied")) is not bool:
                self.issue("fault_event_flag", "事件 fault_applied 缺失或非布尔值", row)
            if event.get("fault_applied") is True:
                if event.get("step_index") != 4 or event.get("source_agent") != "Shopping Worker":
                    self.issue("fault_event_location", "故障不在 Worker step4", row)
                if event.get("fault_type") != fault:
                    self.issue("fault_event_type", "注入事件与 run 故障条件不一致", row)
            elif event.get("fault_type") != "none":
                self.issue("fault_event_type", "未注入事件应标记 none", row)
        if row.get("injection_step") != 4 or fault != row.get("fault"):
            self.issue("job_metadata_mismatch", "故障字段或注入位置与 job 不一致", row)
        if row.get("condition") != ("clean" if fault == "none" else fault):
            self.issue("job_metadata_mismatch", "condition 与 fault_type 不一致", row)

    def check_cart_identity(self, row, ledger):
        readbacks = [r for r in ledger if isinstance(r, dict)
                     and str(r.get("purpose", "")).startswith("reobserve_cart.")]
        if not readbacks:
            return
        creates = [r for r in ledger if isinstance(r, dict) and r.get("purpose") == "add_to_cart.create_cart"]
        if len(creates) != 1 or not valid_hash(creates[0].get("guest_cart_id_sha256")):
            self.issue("cart_identity_unknown", "复查缺少唯一且带 hash 的创建购物车凭据，不能认证恢复", row, severity="unknown")
            return
        created = creates[0]
        cart_hash = created["guest_cart_id_sha256"]
        payload = created.get("response_payload")
        if payload is None:
            self.issue("cart_identity_unknown", "创建购物车 response_payload 缺失，不能认证恢复", row, severity="unknown")
        elif not isinstance(payload, dict) or payload.get("guest_cart_id_sha256") != cart_hash:
            self.issue("cart_identity_mismatch", "创建购物车 payload 与 ledger cart hash 不一致", row)
        for receipt in readbacks + [r for r in ledger if isinstance(r, dict)
                                    and r.get("purpose") in ("add_to_cart.add_item", "verify_cart.items")]:
            value = receipt.get("guest_cart_id_sha256")
            if value is None:
                self.issue("cart_identity_unknown", "购物车操作缺少 cart hash，不能认证恢复", row, severity="unknown")
            elif value != cart_hash:
                self.issue("cart_identity_mismatch", "创建、写入与读回不是同一个 guest cart hash", row)

    def check_readback_payload(self, row, task, event, receipts, group):
        if task is None or len(receipts) != 2 or not all(isinstance(r, dict) for r in receipts):
            return
        page, items = (r.get("response_payload") for r in receipts)
        if page is None or items is None:
            self.issue("readback_payload_unknown", "复查 HTTP response_payload 缺失，不能认证恢复", row, severity="unknown")
            return
        if not isinstance(page, dict) or not page.get("product_id") or not page.get("sku") or not isinstance(items, list):
            self.issue("readback_payload_mismatch", "复查商品或购物车响应结构无效", row)
            return
        matched = next((item for item in items if isinstance(item, dict) and item.get("sku") == page["sku"]), None)
        quantity = matched.get("qty", 0) if matched else 0
        if type(quantity) is float and quantity.is_integer():
            quantity = int(quantity)
        # Mirror the frozen executor's read-only extraction from recorded bodies.
        expected = {"task_id": task["task_id"], "product_title": matched.get("name", "") if matched else "",
                    "product_id": page["product_id"], "sku": page["sku"], "requested_quantity": task["quantity"],
                    "observed_quantity": quantity,
                    "cart_verified": bool(matched and matched.get("name") == task["product_title"]
                                          and type(quantity) is int and quantity == task["quantity"])}
        fields = ["readback_response"] if group == "mitigation_events" else ["readback"]
        if group == "mitigation_events" and event.get("replacement_used") is True:
            fields.append("after")
        for field in fields:
            value = event.get(field)
            if (not isinstance(value, dict) or any(k not in value or payload_hash(value[k]) != payload_hash(v)
                                                   for k, v in expected.items())):
                self.issue("readback_payload_mismatch", "复查返回/替换内容与已记录商品及购物车 items 不一致", row, field=field)

    def check_readbacks(self, row, task):
        def needs_common(verdict):
            return (not isinstance(verdict, dict) or verdict.get("decision") != "accept"
                    or verdict.get("task_id") != row.get("task_id"))

        ledger = row.get("http_receipts")
        if not isinstance(ledger, list):
            self.issue("http_receipts_unknown", "HTTP ledger 缺失", row, severity="unknown")
            ledger = []
        for index, receipt in enumerate(ledger):
            if not isinstance(receipt, dict) or type(receipt.get("receipt_index")) is not int or receipt["receipt_index"] != index:
                self.issue("http_receipt_indices", "HTTP ledger receipt_index 与实际位置不一致", row)
        self.check_cart_identity(row, ledger)
        used = []
        for group, counter in (("mitigation_events", "additional_http_requests"),
                               ("common_recovery_events", "common_recovery_http_requests")):
            events = row.get(group)
            if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
                self.issue("readback_events_unknown", "复查事件缺失或无效", row, severity="unknown", group=group)
                events = []
            calls = sum(e.get("readback_called") is True for e in events)
            if group == "mitigation_events" and row.get("mitigation_mode") == "always_recheck" and calls != 1:
                self.issue("required_mitigation_readback", "完成的 always_recheck run 必须恰好调用一次新增复查", row)
            if group == "common_recovery_events":
                final = row.get("final_answer")
                verdict = final.get("verdict") if isinstance(final, dict) else None
                if (needs_common(verdict) or any(needs_common(e.get("before_verdict")) for e in events)) and calls != 1:
                    self.issue("required_common_readback", "最终或共有恢复前拒绝/未绑定，必须恰好调用一次共有恢复", row)
            if calls > 1 or group == "common_recovery_events" and len(events) > 1:
                self.issue("readback_budget", "一组复查超过一次预算", row, group=group)
            if group == "mitigation_events" and row.get("mitigation_mode") == "baseline" and calls:
                self.issue("readback_budget", "baseline 不允许新增接收端复查", row)
            linked = []
            for position, event in enumerate(events):
                if group == "mitigation_events" and task is not None:
                    expected_issues = list(check_evidence(task, event.get("before")).issues)
                    if "before" not in event or event.get("before_issues") != expected_issues:
                        self.issue("before_issues_mismatch", "before_issues 与 event.before 离线重算不一致", row)
                called = event.get("readback_called") is True
                if "readback_count" in event and (not integer(event["readback_count"]) or event["readback_count"] > 1):
                    self.issue("readback_budget", "事件累计复查数超过预算或无效", row)
                indices = event.get("readback_receipt_indices")
                valid = (isinstance(indices, list) and len(indices) <= 2
                         and all(type(i) is int and 0 <= i < len(ledger) for i in indices)
                         and indices == sorted(set(indices))
                         and (len(indices) < 2 or indices[1] == indices[0] + 1))
                if not valid or not called and indices:
                    self.issue("http_receipt_indices", "复查索引必须合法、连续且只能归属实际调用", row, group=group)
                    continue
                if called:
                    linked.extend(indices)
                    if not event.get("readback_error"):
                        self.check_readback_payload(row, task, event, [ledger[i] for i in indices], group)
                    for offset, index in enumerate(indices):
                        receipt = ledger[index]
                        purpose = ("reobserve_cart.product_page", "reobserve_cart.items")[offset]
                        if (not isinstance(receipt, dict) or receipt.get("request_method") != "GET"
                                or receipt.get("purpose") != purpose):
                            self.issue("http_readback_receipt", "复查只能关联对应的两个只读 GET", row)
                    if not event.get("readback_error") and event.get("replacement_used") is True:
                        if len(indices) != 2 or any(not isinstance(ledger[i], dict)
                                or not integer(ledger[i].get("status_code")) or not 200 <= ledger[i]["status_code"] < 300
                                or not valid_hash(ledger[i].get("response_sha256"))
                                or not ledger[i].get("request_url") for i in indices):
                            self.issue("http_readback_receipt", "使用的复查证据缺少两个成功且带 hash 的 GET", row)
                    if group == "common_recovery_events" and event.get("readback_receipts") != [ledger[i] for i in indices]:
                        self.issue("http_readback_receipt", "共有恢复内嵌 receipts 与 ledger 不一致", row)
                elif event.get("replacement_used") is True:
                    cached = event.get("cached_readback_event_index")
                    if not (type(cached) is int and 0 <= cached < position
                            and events[cached].get("readback_called") is True
                            and events[cached].get("readback_response") == event.get("after")):
                        self.issue("cached_readback_link", "缓存复查缺少合法先前来源", row)
            count = row.get(counter)
            if not integer(count):
                self.issue("http_usage_unknown", "HTTP 请求数未记录", row, severity="unknown", field=counter)
            elif count > 2 or count != len(linked) or count > 2 * calls:
                self.issue("http_budget", "HTTP 请求数与复查预算或 receipt 数不符", row, field=counter)
            used.extend(linked)
        if len(set(used)) != len(used):
            self.issue("http_receipt_indices", "同一 HTTP receipt 被多次或跨恢复组使用", row)
        readback_indices = {i for i, receipt in enumerate(ledger) if isinstance(receipt, dict)
                            and str(receipt.get("purpose", "")).startswith("reobserve_cart.")}
        if readback_indices != set(used):
            self.issue("http_receipt_indices", "存在未被复查事件认领的 HTTP readback", row)

    def check_attempts(self, attempts, errors, rows, jobs, manifest):
        starts, failures = defaultdict(list), defaultdict(list)
        for values, index, duplicate in ((attempts, starts, "duplicate_attempt"),
                                         (errors, failures, "duplicate_error_attempt")):
            for value in values:
                self.check_metadata(value, jobs)
                key, attempt = value.get("job_key"), value.get("attempt")
                if not integer(attempt) or attempt not in (1, 2):
                    self.issue("attempt_budget", "尝试编号必须为 1 或 2", value)
                    continue
                if attempt in index[key]:
                    self.issue(duplicate, "同 job 的尝试编号重复", value)
                index[key].append(attempt)
                if index is starts:
                    self.check_model(value, manifest)
        completed = Counter(r.get("job_key") for r in rows)
        inflight, blocked = [], []
        for key in set(starts) | set(failures) | set(completed):
            start, failure = starts[key], failures[key]
            if len(failure) + completed[key] > 2 or len(start) > 2:
                self.issue("attempt_budget", "错误尝试加完成记录超过两次", job_key=key)
            if not set(failure).issubset(start) or completed[key] and not start:
                self.issue("attempt_evidence_unknown", "错误或完成记录缺少 start journal 证据", severity="unknown", job_key=key)
            if start and sorted(set(start)) != list(range(1, max(start) + 1)):
                self.issue("attempt_sequence", "尝试编号不连续", job_key=key)
            if 2 in start and 1 not in failure:
                self.issue("attempt_evidence_unknown", "第二次 start 缺少第一次错误记录", severity="unknown", job_key=key)
            unresolved = sorted(set(start) - set(failure))
            if completed[key] and (not unresolved or start and max(start) in failure):
                self.issue("attempt_outcome_conflict", "完成记录没有未失败的尝试", job_key=key)
            if not completed[key] and unresolved:
                inflight.append(key)
            if not completed[key] and (len(failure) >= 2 or max(start or [0]) >= 2):
                blocked.append(key)
        return {"started_attempts": len(attempts), "error_attempts": len(errors),
                "unresolved_job_keys": sorted(inflight), "exhausted_or_second_attempt_job_keys": sorted(blocked)}

    def check_stale(self, row, by_run, reevaluated, task_map):
        source = row.get("stale_replay_source")
        if row.get("fault_type") != "stale_replay":
            if source:
                self.issue("unexpected_stale_source", "非 stale 条件携带旧消息来源", row)
            return
        if not isinstance(source, dict):
            self.issue("stale_source_missing", "stale replay 缺少真实来源", row)
            return
        if not all(isinstance(source.get(k), str) and source[k] for k in
                   ("source_run_id", "source_task_id", "source_message_id")):
            self.issue("stale_source_identity", "stale 来源标识缺失或类型错误", row)
            return
        payload = source.get("payload")
        if payload_hash(payload) != source.get("payload_sha256"):
            self.issue("stale_source_hash", "stale payload hash 不一致", row)
        candidates = by_run.get(source.get("source_run_id"), [])
        if len(candidates) != 1:
            self.issue("stale_source_missing_run", "来源 run 在本文件中缺失或不唯一", row)
        else:
            clean = candidates[0]
            evaluated = reevaluated.get(clean.get("run_id"), {})
            if (clean.get("fault_type") != "none" or clean.get("mitigation_mode") != "baseline"
                    or clean.get("final_task_success") is not True or evaluated.get("final_task_success") is not True
                    or clean.get("task_id") == row.get("task_id")
                    or clean.get("task_id") != source.get("source_task_id")
                    or clean.get("topology") != row.get("topology")
                    or clean.get("repeat_index") != row.get("repeat_index")
                    or clean.get("model") != source.get("model") or clean.get("provider") != source.get("provider")):
                self.issue("stale_source_eligibility", "来源必须为同拓扑/重复、不同任务的成功 baseline clean", row)
            events = clean.get("events") if isinstance(clean.get("events"), list) else []
            matches = [e for e in events if isinstance(e, dict) and e.get("message_id") == source.get("source_message_id")]
            if (len(matches) != 1 or matches[0].get("step_index") != 4
                    or matches[0].get("source_agent") != "Shopping Worker"
                    or matches[0].get("fault_applied") is not False):
                self.issue("stale_source_message", "来源 message_id 未指向原始 Worker step4 消息", row)
            elif payload != matches[0].get("original_message"):
                self.issue("stale_source_payload", "来源 payload 与 clean 原始消息不同，重算 hash 也不能替代", row)
            truth = clean.get("environment_state", {})
            task = task_map.get(clean.get("task_id"))
            if (not task or not isinstance(payload, dict) or not check_evidence(task, payload).valid
                    or not isinstance(truth, dict) or truth.get("cart_verified") is not True
                    or any(payload.get(k) != truth.get(k) for k in ("task_id", "product_title", "product_id", "sku", "observed_quantity"))):
                self.issue("stale_source_evidence", "原始旧消息未通过来源任务证据/环境一致性检查", row)
        for event in row.get("events") if isinstance(row.get("events"), list) else []:
            if isinstance(event, dict) and event.get("fault_applied") is True and event.get("delivered_message") != [payload]:
                self.issue("stale_delivery_mismatch", "实际注入 delivery 不是冻结旧消息", row)
        key = [row.get("task_id"), row.get("topology"), row.get("repeat_index")]
        name = "stale_sources/" + sha256(json.dumps(key).encode())[:20] + ".json"
        frozen = self.read_json(name, required=True)
        if self.inputs.get(name, {}).get("present") and frozen != source:
            self.issue("stale_frozen_file_mismatch", "stale_sources 冻结文件与嵌入来源不一致", row)

    def run(self):
        manifest = self.read_json("matrix_manifest.json", required=True)
        rows = self.read_jsonl("main_runs.jsonl", required=True)
        recorded_count = len(rows)
        valid_rows = []
        for row in rows:
            bad = [field for field in ("run_id", "job_key", "task_id")
                   if not isinstance(row.get(field), str) or not row[field]]
            if bad:
                for field in bad:
                    self.issue("invalid_" + field, "记录标识缺失或类型错误，保留记录数量但不分析", row)
                self.partial = True
            else:
                valid_rows.append(row)
        rows = valid_rows
        attempts = self.read_jsonl("run_attempts.jsonl", required=bool(rows))
        errors = self.read_jsonl("run_errors.jsonl")
        infra = self.read_jsonl("infrastructure_events.jsonl")
        repairs = self.read_jsonl("journal_repairs.jsonl")
        if repairs:
            self.issue("journal_repair_usage_unknown", "日志修复可能遗失尝试/token，保留 unknown", severity="unknown")
        tasks, jobs, expected = self.check_manifest(manifest)
        sources = self.check_sources(manifest)
        evaluator, version = load_evaluator()
        if manifest.get("version") != version:
            self.issue("evaluator_version_mismatch", "本地离线 evaluator 版本与 manifest 不一致")
        by_run, by_job = defaultdict(list), defaultdict(list)
        for row in rows:
            for field, index in (("run_id", by_run), ("job_key", by_job)):
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    self.issue("invalid_" + field, "标识缺失或非字符串", row)
                else:
                    index[value].append(row)
        for field, index in (("run_id", by_run), ("job_key", by_job)):
            for values in index.values():
                if len(values) > 1:
                    for row in values:
                        self.issue("duplicate_" + field, "结果记录标识重复，禁止静默去重", row)
        reevaluated, reconciled = {}, 0
        for row in rows:
            self.check_metadata(row, jobs)
            self.check_model(row, manifest)
            if row.get("common_recovery_enabled") is not True:
                self.issue("common_recovery_disabled", "该 run 未明确启用共有恢复", row)
            reconciled += self.check_usage(row, manifest)
            self.check_events(row)
            task = tasks.get(row.get("task_id"))
            self.check_readbacks(row, task)
            try:
                if task is None:
                    raise ValueError("unknown task")
                evaluated = evaluator(row, task)
                reevaluated[row.get("run_id")] = evaluated
                for key in DERIVED_FIELDS:
                    if key not in row or payload_hash(row[key]) != payload_hash(evaluated[key]):
                        self.issue("derived_field_mismatch", "离线 audit_outcome 派生字段不一致", row,
                                   field=key, recorded=row.get(key), reevaluated=evaluated[key])
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
                self.issue("evaluation_unavailable", "证据无法离线重评", row, error_type=type(exc).__name__)
        for row in rows:
            self.check_stale(row, by_run, reevaluated, tasks)
        stale_pairs = defaultdict(list)
        for row in rows:
            if row.get("fault_type") == "stale_replay":
                stale_pairs[row.get("pair_key")].append(row)
        for values in stale_pairs.values():
            if len({payload_hash(r.get("stale_replay_source")) for r in values}) > 1:
                for row in values:
                    self.issue("stale_pair_source_mismatch", "同 pair 各 arm 的旧消息来源不同", row)
        attempt_report = self.check_attempts(attempts, errors, rows, jobs, manifest)
        recorded_excluded = set()
        for event in infra:
            ids = event.get("completed_run_ids_at_recording")
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                self.issue("infrastructure_coverage_unknown", "基础设施事件缺少 completed run 排除名单", severity="unknown")
            else:
                recorded_excluded.update(ids)
        conservative_clean = {r.get("run_id") for r in rows if r.get("fault_type") == "none" and r.get("run_id")} if infra else set()
        excluded = recorded_excluded | conservative_clean
        if infra:
            self.issue("infrastructure_inflight_caveat", "记录时 completed 名单可能未覆盖当时 inflight；保守排除全部 clean 延迟，仍不保证覆盖其他受影响运行", severity="warning")
        # Re-reading only checks stability; no locks, repairs, or input writes.
        for relative, fingerprint in list(self.inputs.items()):
            path = self.root / relative
            if fingerprint["present"] and (not path.is_file() or sha256(path.read_bytes()) != fingerprint["sha256"]):
                self.issue("snapshot_changed", "审计期间输入字节变化；不能作为完整快照", severity="unknown", file=relative)
                self.partial = True
        invalid_ids = {i.get("run_id") for i in self.findings if (i["severity"] == "error"
                       or i["code"] in ("readback_payload_unknown", "cart_identity_unknown"))
                       and isinstance(i.get("run_id"), str)}
        usable = [reevaluated[r["run_id"]] for r in rows if r.get("run_id") in reevaluated
                  and r.get("run_id") not in invalid_ids and len(by_job.get(r.get("job_key"), [])) == 1
                  and r.get("job_key") in expected]
        success_clean = {(r["task_id"], r["topology"], r["repeat_index"]) for r in usable
                         if r["fault_type"] == "none" and r["mitigation_mode"] == "baseline" and r["final_task_success"]}
        for row in usable:
            if row["fault_type"] != "none" and (row["task_id"], row["topology"], row["repeat_index"]) not in success_clean:
                present = any(r.get("task_id") == row["task_id"] and r.get("topology") == row["topology"]
                              and r.get("repeat_index") == row["repeat_index"] and r.get("fault_type") == "none"
                              and r.get("mitigation_mode") == "baseline" for r in rows)
                self.issue("baseline_clean_not_admitted", "故障运行缺少通过审计的对应 baseline clean", row,
                           severity="error" if present else "unknown")
        unadmitted = {i.get("run_id") for i in self.findings if i["code"] == "baseline_clean_not_admitted"}
        usable = [r for r in usable if r["run_id"] not in unadmitted]
        missing = sorted(set(expected) - set(by_job))
        if missing:
            self.issue("missing_jobs", "快照仍有未完成 job，不能补为成功或失败", severity="unknown", count=len(missing))
        paired, pairing = build_pairs(usable, expected, excluded)
        known_tokens = sum(r["total_tokens"] for r in rows if integer(r.get("total_tokens")))
        error_tokens = sum(e["known_total_tokens"] for e in errors if integer(e.get("known_total_tokens")))
        errors_complete = all(e.get("usage_complete") is True and integer(e.get("known_total_tokens")) for e in errors)
        if not errors_complete:
            self.issue("error_usage_unknown", "错误尝试只有已知 token 下界，完整用量 unknown", severity="unknown")
        complete_usage = reconciled == len(rows) and not repairs and not self.partial
        usage_errors = any(i["code"] in ("token_total_mismatch", "request_usage_mismatch", "request_count_mismatch") for i in self.findings)
        levels = Counter(i["severity"] for i in self.findings)
        incomplete = bool(missing or self.partial or attempt_report["unresolved_job_keys"] or not expected)
        return {"schema_version": "shopping-offline-audit-v1", "status": "incomplete" if incomplete else "complete",
                "audit_status": "failed" if levels["error"] else "unknown" if levels["unknown"] else "passed",
                "result_dir": str(self.root), "input_files": self.inputs,
                "evaluator": {"version": version, "source_root": str(ROOT),
                              "sha256": {name: sha256((ROOT / name).read_bytes()) for name in CORE_SOURCES},
                              "ignored_field": "legacy_evaluation"},
                "source_hashes": sources, "findings": self.findings, "finding_counts": dict(levels),
                "matrix": {"expected_runs": len(expected), "manifest_jobs": len(jobs), "completed_runs": recorded_count,
                           "analysis_runs": len(usable), "missing_job_keys": missing,
                           "unexpected_job_keys": sorted(set(by_job) - set(expected))},
                "attempts": attempt_report,
                "usage": {"completed_known_total_tokens": known_tokens,
                          "completed_total_tokens": known_tokens if complete_usage and not usage_errors else None,
                          "request_reconciled_runs": reconciled, "error_known_total_tokens": error_tokens,
                          "error_total_tokens": error_tokens if errors_complete and not repairs else None,
                          "note": "仅对账已记录 prompt/completion；客户端可能将上游缺失 usage 写成 0，无法从现存日志识别该情况。"},
                "latency": {"recorded_excluded_run_ids": sorted(recorded_excluded),
                            "conservative_clean_excluded_run_ids": sorted(conservative_clean),
                            "effective_excluded_run_ids": sorted(excluded),
                            "possibly_uncovered_inflight": bool(infra), "infrastructure_events": len(infra)},
                "by_mode": grouped_counts(usable, "mitigation_mode", excluded),
                "by_condition": grouped_counts(usable, "condition", excluded),
                "by_topology": grouped_counts(usable, "topology", excluded),
                "by_mode_condition_topology": stratified_counts(usable, excluded),
                "pairing": pairing, "paired": paired}


def counts(rows, excluded):
    latency = [r["latency_ms"] for r in rows if r.get("run_id") not in excluded and number(r.get("latency_ms"))]
    result = {"runs": len(rows), "final_task_success": sum(r.get("final_task_success") is True for r in rows),
              "environment_verified": sum(isinstance(r.get("environment_state"), dict)
                                          and r["environment_state"].get("cart_verified") is True for r in rows),
              "environment_unknown": sum(not isinstance(r.get("environment_state"), dict)
                                         or type(r["environment_state"].get("cart_verified")) is not bool for r in rows),
              "latency_included_runs": len(latency), "latency_excluded_runs": sum(r.get("run_id") in excluded for r in rows),
              "mean_latency_ms": sum(latency) / len(latency) if latency else None}
    for key in ("mitigation_recovery_detected", "common_recovery_detected", "common_recovery_attempted", "m_propagated"):
        result[key] = sum(r.get(key) is True for r in rows)
    for key in ("additional_http_requests", "common_recovery_http_requests", "total_tokens"):
        result[key] = sum(r[key] for r in rows) if all(integer(r.get(key)) for r in rows) else None
    return result


def grouped_counts(rows, field, excluded):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {key: counts(value, excluded) for key, value in sorted(groups.items())}


def stratified_counts(rows, excluded):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mitigation_mode"], row["condition"], row["topology"])].append(row)
    return [{"mode": key[0], "condition": key[1], "topology": key[2], **counts(value, excluded)}
            for key, value in sorted(groups.items())]


def build_pairs(rows, expected, excluded):
    by_job = {r["job_key"]: r for r in rows}
    clean = {(r["task_id"], r["topology"], r["repeat_index"], r["mitigation_mode"]): r
             for r in rows if r["fault_type"] == "none"}
    pairs = {job["pair_key"]: job for job in expected.values()}
    records = []
    comparisons = {mode: {"complete_baseline_pairs": 0, "matched_clean_fault_pairs": 0,
                           "improvements": 0, "regressions": 0, "unchanged": 0,
                           "clean_success_regressions": 0} for mode in MODES if mode != "baseline"}
    for pair, job in sorted(pairs.items()):
        arms = {mode: by_job.get(pair + ":" + mode) for mode in MODES}
        record = {"pair_key": pair, "task_id": job["task_id"], "topology": job["topology"],
                  "condition": "clean" if job["fault"] == "none" else job["fault"],
                  "repeat_index": job["repeat_index"], "complete_three_arm": all(arms.values())}
        for mode, row in arms.items():
            for field in ("run_id", "final_task_success", "mitigation_recovery_detected", "common_recovery_detected",
                          "additional_http_requests", "common_recovery_http_requests", "total_tokens"):
                record[mode + "_" + field] = row.get(field) if row else None
            record[mode + "_environment_verified"] = row.get("environment_state", {}).get("cart_verified") if row else None
            record[mode + "_latency_ms"] = row.get("latency_ms") if row and row["run_id"] not in excluded else None
            record[mode + "_latency_excluded"] = row["run_id"] in excluded if row else None
            clean_row = clean.get((job["task_id"], job["topology"], job["repeat_index"], mode))
            record[mode + "_clean_success"] = clean_row["final_task_success"] if clean_row else None
        for mode, comparison in comparisons.items():
            baseline, other = arms["baseline"], arms[mode]
            complete = baseline is not None and other is not None
            matched = complete and job["fault"] != "none" and all(record[m + "_clean_success"] is True for m in ("baseline", mode))
            record[mode + "_matched_clean"] = matched
            delta = int(other["final_task_success"]) - int(baseline["final_task_success"]) if complete else None
            record[mode + "_success_delta"] = delta
            comparison["complete_baseline_pairs"] += complete
            comparison["matched_clean_fault_pairs"] += matched
            if matched:
                comparison["improvements" if delta == 1 else "regressions" if delta == -1 else "unchanged"] += 1
            if complete and job["fault"] == "none" and delta == -1:
                comparison["clean_success_regressions"] += 1
        records.append(record)
    return records, {"planned_three_arm_pairs": len(records),
                     "complete_three_arm_pairs": sum(r["complete_three_arm"] for r in records),
                     "comparisons": comparisons}


def audit_results(result_dir, source_root=None):
    """Audit a local snapshot without writing inputs or contacting any service."""
    return Auditor(result_dir, source_root).run()


def markdown(report):
    matrix = report["matrix"]
    lines = ["# Shopping 缓解结果离线审计", "",
             f"矩阵状态：**{report['status']}**；证据审计：**{report['audit_status']}**。",
             f"已记录 {matrix['completed_runs']} / 计划 {matrix['expected_runs']}；可分析 {matrix['analysis_runs']}；未完成 {len(matrix['missing_job_keys'])}。", "",
             "本轮仅在 Step 4 注入证据故障。environment_verified 统计 environment_state.cart_verified，表示购物车动作环境已验证；",
             "final_task_success 是 audit_outcome 的最终接受、证据契约/任务绑定/环境真值一致，不是原生 WebArena 官方任务分数。",
             "证据判定失败不等于购物车动作失败。缺失、未知、错误尝试与未完成 job 均不补成成功或失败。",
             "下表使用离线重评且无行级完整性错误的记录；复查响应或购物车身份未知的记录也不认证恢复、不纳入分析；全局审计失败时不得把表格当成已认证结果。",
             "共有恢复与新增缓解分别计数，不能将共有恢复收益归给新增策略。仅报告描述性配对计数，不作显著性或总体因果宣称。", ""]
    for label, groups in (("按组", report["by_mode"]), ("按条件", report["by_condition"]), ("按拓扑", report["by_topology"])):
        lines += ["## " + label, "", "| 分组 | Runs | environment_verified | final_task_success | 新增缓解恢复 | 共有恢复 | 新增 HTTP | 共有 HTTP |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, value in groups.items():
            lines.append(f"| {key} | {value['runs']} | {value['environment_verified']} | {value['final_task_success']} | {value['mitigation_recovery_detected']} | {value['common_recovery_detected']} | {value['additional_http_requests']} | {value['common_recovery_http_requests']} |")
        lines.append("")
    lines += ["## 完整配对", "", f"三组完整配对 {report['pairing']['complete_three_arm_pairs']} / {report['pairing']['planned_three_arm_pairs']}。",
              "matched-clean 仅纳入 baseline 与比较组各自对应 clean 均成功的故障配对；第三组缺失不补值。", "",
              "| 比较组 vs baseline | 完整两组配对 | matched-clean 故障对 | 改善 | 退化 | 不变 | clean退化 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, value in report["pairing"]["comparisons"].items():
        lines.append(f"| {mode} | {value['complete_baseline_pairs']} | {value['matched_clean_fault_pairs']} | {value['improvements']} | {value['regressions']} | {value['unchanged']} | {value['clean_success_regressions']} |")
    latency = report["latency"]
    lines += ["", "## 用量与延迟", "", f"完成记录已知 token：{report['usage']['completed_known_total_tokens']}；可对账完整总量：{report['usage']['completed_total_tokens'] if report['usage']['completed_total_tokens'] is not None else 'unknown'}。",
              f"错误尝试 {report['attempts']['error_attempts']}；已知 token 下界 {report['usage']['error_known_total_tokens']}，完整值 {report['usage']['error_total_tokens'] if report['usage']['error_total_tokens'] is not None else 'unknown'}。",
              report["usage"]["note"],
              f"基础设施 completed_run_ids_at_recording 显式排除 {len(latency['recorded_excluded_run_ids'])} 条；保守排除全部 clean 后共 {len(latency['effective_excluded_run_ids'])} 条，仅影响延迟，不删除结果。",
              "记录时名单可能未覆盖当时 inflight；保守 clean 排除也不保证覆盖其他受影响运行。未列入名单不等于基础设施未受影响。" if latency["possibly_uncovered_inflight"] else "未发现已记录基础设施事件，不等于证实不存在基础设施影响。",
              "", "## 源码与发现", "", f"冻结源码核验：{report['source_hashes']['status']}。--source-root 仅用于读文件比 hash，不从该目录执行代码。",
              "离线重用本地原始 audit_outcome；仅忽略非幂等 legacy_evaluation，其他派生字段逐项核对。",
              "HTTP hash 是已记录字节摘要，本审计不访问远端页面，也不将 hash 格式正确视为原始响应内容独立认证。", ""]
    for issue in report["findings"]:
        where = issue.get("run_id") or issue.get("job_key") or issue.get("file") or "全局"
        lines.append(f"- [{issue['severity']}] `{issue['code']}`：{issue['message']}（{where}）")
    if not report["findings"]:
        lines.append("未发现审计异常。")
    lines += ["", "详细 mode/condition/topology 联合分组、排除名单、输入字节 hash 与所有计划配对见 audit.json / paired.csv。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir
    if output.exists() or output.is_symlink():
        parser.error("output directory already exists; refusing overwrite")
    if output.resolve().is_relative_to(args.result_dir.resolve()):
        parser.error("output must be outside the read-only result directory")
    try:
        report = audit_results(args.result_dir, args.source_root)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "audit.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with (output / "audit.md").open("x", encoding="utf-8") as handle:
            handle.write(markdown(report))
        with (output / "paired.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["paired"][0]) if report["paired"] else ["pair_key", "complete_three_arm"])
            writer.writeheader()
            writer.writerows(report["paired"])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "audit_status": report["audit_status"],
                      "completed_runs": report["matrix"]["completed_runs"], "output_dir": str(output)}, ensure_ascii=False))
    return 1 if report["audit_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
