#!/usr/bin/env python3
"""Pure offline T5 audit. Never runs a runner, HTTP executor, model, or ledger.

Requires a NEW output directory outside the input snapshot. Exit codes: 0 for
a complete clean audit, 1 for findings/incompleteness, 2 for invalid input/CLI.
--no-strict permits report generation with exit 0; it never changes findings.
--source-root hashes frozen source files only, never executes their contents.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mas_faults import multimechanism_matrix as matrix
from mas_faults.shopping_mitigation import check_evidence

MODEL = "deepseek-flash"
PROVIDER = "deepseek"
MODEL_VERSION = "DeepSeek-V4.1-Flash"
IDENTITY_FIELDS = ("model", "provider", "model_version")
INFERENCE_PARAMETERS = {"temperature": 0, "max_tokens": 2048, "disable_thinking": True,
                        "socket_timeout_seconds": 90, "total_timeout_seconds": 120}
REQUEST_FIELDS = ("request_index", *IDENTITY_FIELDS, "response_model", "provider_request_id", "request_sent", "status",
                  "error_type", "exception_chain", "prompt_tokens", "completion_tokens", "total_tokens",
                  "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
JOB_FIELDS = ("job_key", "pair_key", "task_id", "topology", "condition", "boundary", "repeat_index", "arm")
DERIVED_FIELDS = ("environment_task_success", "decision_correct", "final_task_success", "task_score",
                  "evidence_acceptance_errors", "observed_M_consequence", "observed_A_symptom",
                  "recovery_detected", "recovery_type", "propagation_class")
ACTION_DERIVED_FIELDS = ("action_recovery_count", "action_prevention_count", "prevention_detected", "prevention_type")
SOURCE_CONDITIONS = {"cross_task_replay", "contract_consistent_identity_corruption"}
CORE_SOURCES = ("run_shopping_multimechanism.py", "src/mas_faults/shopping_multimechanism.py",
                "src/mas_faults/multimechanism_matrix.py", "src/mas_faults/multimechanism_faults.py",
                "src/mas_faults/shopping_action_protocol.py", "src/mas_faults/mitigation_protocol.py",
                "src/mas_faults/shopping_mitigation.py", "src/mas_faults/llm_client.py",
                "src/mas_faults/webarena_shopping_real.py")
LIMITATIONS = [
    "model_calls 是客户端请求尝试数，不是实际付费 API 调用数；request_sent 仅记录是否进入网络发送。",
    "deepseek-flash 为 API 名，DeepSeek-V4.1-Flash 为冻结文档版本声明；二者均不是 immutable 权重证据。",
    "evaluate_trial 重算仅检查一致性，不是对所有运行时主张的独立验证。",
    "HTTP/模型收据是记录证据，不是后端真实性的独立证明；不从 fault 标签补造暴露或成功。",
    "本审计只作描述性配对统计，不作统计显著性或跨域推广；重复编号不是模型随机 seed。",
    "缺少协议字段本身不构成语义失败；历史错误接受与环境成功、最终证据成功分列。",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def integer(value):
    return type(value) is int and value >= 0


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def load_evaluator():
    # Select only local pure definitions. Importing the module would load AutoGen
    # and the live executor. A supplied source-root is deliberately never executed.
    path = ROOT / "src/mas_faults/shopping_multimechanism.py"
    data = path.read_bytes()
    tree = ast.parse(data, filename=str(path))
    names = {"validate_action", "semantic_success", "evidence_state_digest",
             "verify_action_protocol_event", "verify_action_consequences", "evaluate_trial"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("unsupported evaluator definitions")
    namespace = {"copy": copy, "json": json, "check_evidence": check_evidence,
                 "config_digest": matrix.config_digest}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["evaluate_trial"], {"file": str(path), "sha256": sha256(data),
        "independent_runtime_validation": False, "kind": "runtime_evaluator_consistency",
        "limitations": LIMITATIONS}


def runner_exposure_summary(rows, errors):
    """Reconcile the runner's reporting schema, not the stricter case verdicts.

    A structurally recorded injection and an audited valid exposure are separate
    claims. Bad traces still receive findings from Auditor.exposure.
    """
    counts = dict.fromkeys(("clean_runs", "planned_fault_runs", "injection_recorded_runs",
        "exposed_fault_runs", "unreached_fault_runs", "prior_contract_failure_runs",
        "unknown_fault_exposure_runs", "error_attempts_exposure_unknown"), 0)
    for row in rows:
        if row.get("condition") == "clean":
            counts["clean_runs"] += 1
            continue
        counts["planned_fault_runs"] += 1
        events = row.get("fault_events")
        recorded = False
        if isinstance(events, list) and len(events) == 1 and isinstance(events[0], dict):
            event = events[0]
            delivered = event.get("delivered_sha256")
            hashes = [event.get("original_sha256"), *(delivered if isinstance(delivered, list) else [])]
            recorded = (event.get("condition") == row.get("condition")
                and event.get("boundary") == matrix.CELLS.get(row.get("condition"))
                and event.get("boundary") is not None
                and event.get("applied") is not False and event.get("reached") is not False
                and isinstance(delivered, list) and type(event.get("delivered_count")) is int
                and event["delivered_count"] == len(delivered)
                and all(isinstance(h, str) and valid_hash(h.lower()) for h in hashes))
        counts["injection_recorded_runs"] += int(recorded)
        if row.get("action_contract_valid") is False:
            category = "prior_contract_failure_runs"
        elif events == []:
            category = "unreached_fault_runs"
        elif recorded and row.get("action_contract_valid") is True:
            category = "exposed_fault_runs"
        else:
            category = "unknown_fault_exposure_runs"
        counts[category] += 1
    counts["error_attempts_exposure_unknown"] = sum(r.get("condition") != "clean" for r in errors)
    return counts


class Auditor:
    def __init__(self, result_dir, source_root=None):
        self.root = Path(result_dir).resolve()
        if not self.root.is_dir():
            raise ValueError("result directory must exist")
        self.source_root = Path(source_root).resolve() if source_root is not None else None
        self.findings, self.inputs = [], {}
        self.evaluate, self.evaluation = load_evaluator()

    def issue(self, code, row=None, *, severity="error", **details):
        finding = {"code": code, "severity": severity, **details}
        if row is not None:
            finding.update(run_id=row.get("run_id"), job_key=row.get("job_key"), attempt_id=row.get("attempt_id"))
        self.findings.append(finding)

    def check_summary_exposure(self, reported, rows, errors, field):
        expected = runner_exposure_summary(rows, errors)
        if not isinstance(reported, dict):
            self.issue("summary_count_mismatch", field=field)
            return
        for key in sorted(set(expected) | set(reported)):
            if key not in expected or type(reported.get(key)) is not int or reported[key] != expected[key]:
                self.issue("summary_count_mismatch", field=field + "." + key)

    def read_bytes(self, relative, *, required=False):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            self.issue("input_path_escape")
            return None
        if not path.is_file():
            self.inputs[str(relative)] = {"present": False}
            if required:
                self.issue("missing_input", severity="unknown", file=str(relative))
            return None
        data = path.read_bytes()
        self.inputs[str(relative)] = {"present": True, "sha256": sha256(data), "bytes": len(data)}
        return data

    def read_json(self, relative, *, required=False):
        data = self.read_bytes(relative, required=required)
        if data is None:
            return {}
        try:
            value = strict_json(data)
            if not isinstance(value, dict):
                raise ValueError("object required")
            return value
        except (ValueError, UnicodeDecodeError):
            self.issue("invalid_json", file=str(relative))
            return {}

    def read_jsonl(self, relative, *, required=False):
        data = self.read_bytes(relative, required=required)
        if data is None:
            return []
        result = []
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = strict_json(line)
                if not isinstance(value, dict):
                    raise ValueError("object required")
                result.append(value)
            except (ValueError, UnicodeDecodeError):
                torn = index == len(lines) - 1 and not line.endswith(b"\n")
                self.issue("torn_jsonl_tail" if torn else "invalid_jsonl", file=str(relative), line=index + 1)
        return result

    def check_model(self, row):
        if row.get("model") != MODEL:
            self.issue("model_mismatch", row)
        if row.get("provider") != PROVIDER:
            self.issue("provider_mismatch", row)
        if row.get("model_version") != MODEL_VERSION:
            self.issue("model_version_mismatch", row)

    def manifest(self):
        wrapped = self.read_json("matrix_manifest.json", required=True)
        config = wrapped.get("config")
        if not isinstance(config, dict):
            raise ValueError("matrix_manifest requires {config, config_digest}; flat legacy schema is unsupported")
        digest = matrix.config_digest(config)
        if wrapped.get("config_digest") != digest:
            self.issue("config_digest_mismatch")
        self.check_model(config)
        settings = config.get("inference_settings")
        if not isinstance(settings, dict):
            self.issue("inference_settings_unknown", severity="unknown")
        else:
            self.check_model(settings)
            if settings.get("api_base_url") not in (
                    "https://api.deepseek.com", "https://api.deepseek.com/",
                    "https://api.deepseek.com/v1", "https://api.deepseek.com/v1/"):
                self.issue("invalid_inference_endpoint")
            for field, expected in INFERENCE_PARAMETERS.items():
                if type(settings.get(field)) is not type(expected) or settings[field] != expected:
                    self.issue("inference_parameter_mismatch", field=field)
        if config.get("version") != matrix.VERSION or config.get("runner_schema") != 1:
            self.issue("manifest_schema")
        try:
            full = matrix.build_jobs(config["tasks"], config["repetitions"])
            expected = matrix.select_shard(full, config["shard_index"], config["shard_count"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("unsupported frozen task/shard schema") from None
        jobs = config.get("jobs")
        if not isinstance(jobs, list) or any(not isinstance(j, dict) for j in jobs):
            raise ValueError("frozen jobs must be objects")
        if jobs != expected:
            self.issue("manifest_jobs_mismatch")
        if config.get("planned_runs") != len(full) or config.get("shard_runs") != len(expected):
            self.issue("planned_count_mismatch")
        keys = [j.get("job_key") for j in jobs]
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("frozen job_key must be a string")
        if len(set(keys)) != len(keys):
            self.issue("duplicate_manifest_job_key")
        if config.get("max_attempts_per_job") != 2:
            self.issue("attempt_budget")
        return config, digest, {j["job_key"]: j for j in expected}, len(full)

    def provenance(self, config):
        hashes = config.get("source_hashes")
        verified = 0
        if not isinstance(hashes, dict) or not hashes:
            self.issue("source_hashes_unknown", severity="unknown")
            hashes = {}
        for name in CORE_SOURCES:
            if name not in hashes:
                self.issue("core_source_hash_missing", severity="unknown", file=name)
        for name, expected in hashes.items():
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or not valid_hash(expected):
                self.issue("invalid_source_hash")
                continue
            if self.source_root is None:
                continue
            path = (self.source_root / rel).resolve()
            if not path.is_relative_to(self.source_root) or not path.is_file():
                self.issue("source_missing", file=name)
            elif sha256(path.read_bytes()) != expected:
                self.issue("source_hash_mismatch", file=name)
            else:
                verified += 1
        if self.source_root is None:
            self.issue("source_hashes_unverified", severity="unknown")
        local = "src/mas_faults/shopping_multimechanism.py"
        if local in hashes and hashes[local] != self.evaluation["sha256"]:
            self.issue("evaluator_source_mismatch")
        self.evaluation["helper_sha256"] = {}
        for helper in ("src/mas_faults/shopping_mitigation.py", "src/mas_faults/multimechanism_matrix.py"):
            actual = sha256((ROOT / helper).read_bytes())
            self.evaluation["helper_sha256"][helper] = actual
            if helper in hashes and hashes[helper] != actual:
                self.issue("evaluator_source_mismatch", file=helper)
        return {"verified_source_files": verified, "frozen_source_files": len(hashes)}

    def metadata(self, row, jobs, digest):
        self.check_model(row)
        if row.get("config_digest") != digest:
            self.issue("config_digest_mismatch", row)
        job = jobs.get(row.get("job_key"))
        if job is None:
            self.issue("unexpected_job", row)
        elif any(row.get(k) != job[k] for k in JOB_FIELDS):
            self.issue("job_metadata_mismatch", row)
        if type(row.get("attempt")) is not int or row["attempt"] not in (1, 2):
            self.issue("attempt_budget", row)
        if not isinstance(row.get("attempt_id"), str) or not row["attempt_id"]:
            self.issue("attempt_id_missing", row)

    def attempts(self, starts, rows, errors, jobs, digest):
        indexed, terminal, ids = {}, {}, set()
        per_job = defaultdict(list)
        for start in starts:
            self.metadata(start, jobs, digest)
            identity = (start.get("job_key"), start.get("attempt"))
            if identity in indexed:
                self.issue("duplicate_attempt_start", start)
            if start.get("attempt_id") in ids:
                self.issue("duplicate_attempt_id", start)
            indexed[identity] = start
            ids.add(start.get("attempt_id"))
            per_job[start.get("job_key")].append(start.get("attempt"))
        for numbers in per_job.values():
            if any(type(n) is not int for n in numbers) or sorted(numbers) != list(range(1, len(numbers) + 1)):
                self.issue("noncontiguous_attempts")
        for row in [*rows, *errors]:
            self.metadata(row, jobs, digest)
            identity = (row.get("job_key"), row.get("attempt"))
            if identity in terminal:
                self.issue("duplicate_terminal_attempt", row)
            terminal[identity] = row
            start = indexed.get(identity)
            if start is None:
                self.issue("missing_attempt_start", row)
            elif any(row.get(k) != start.get(k) for k in ("attempt_id", "cross_task_source", "ledger_path")):
                self.issue("attempt_binding_mismatch", row)
        run_ids, completed = set(), {}
        for row in rows:
            run_id, key = row.get("run_id"), row.get("job_key")
            if not isinstance(run_id, str) or not run_id:
                self.issue("run_id_missing", row)
            elif run_id in run_ids:
                self.issue("duplicate_run_id", row)
            run_ids.add(run_id)
            if key in completed:
                self.issue("duplicate_completed_job", row)
            completed[key] = row.get("attempt")
        for key, numbers in per_job.items():
            if key in completed and integer(completed[key]) and any(integer(n) and n > completed[key] for n in numbers):
                self.issue("attempt_after_completion")
        unfinished = [s for identity, s in indexed.items() if identity not in terminal]
        for row in unfinished:
            self.issue("unfinished_attempt", row, severity="unknown")
        return unfinished

    def usage(self, row, *, failed=False):
        requests = row.get("model_requests")
        if not isinstance(requests, list) or any(not isinstance(r, dict) for r in requests):
            self.issue("request_usage_unknown", row, severity="unknown")
            return {"known_total_tokens": 0, "total_tokens": None}
        known_prompt = known_completion = 0
        complete = row.get("usage_complete") is True
        indices = []
        for request in requests:
            self.check_model(request)
            if not failed and request.get("status") != "success":
                self.issue("completed_attempt_request_error", row, request_index=request.get("request_index"))
            unsent_guard = (failed and row.get("status") in ("timeout", "infra_error")
                and request.get("status") == "error" and request.get("error_type") == "DeepSeekPeakWindowError"
                and request.get("exception_chain") == ["DeepSeekPeakWindowError"]
                and request.get("response_model") is None and request.get("provider_request_id") == ""
                and all(type(request.get(k)) is int and request[k] == 0 for k in
                        ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"))
                and (request.get("total_tokens") is None or type(request["total_tokens"]) is int and request["total_tokens"] == 0))
            if type(request.get("request_sent")) is not bool or not (request["request_sent"] or unsent_guard):
                self.issue("request_sent_mismatch", row, request_index=request.get("request_index"))
            if (request.get("status") not in ("success", "error") or "response_model" not in request
                    or not (request["response_model"] == MODEL
                            or request["response_model"] is None and request.get("status") == "error"
                            and failed and row.get("status") in ("timeout", "infra_error"))):
                self.issue("response_model_mismatch", row, request_index=request.get("request_index"))
            if request.get("status") == "error" and request.get("response_model") is None:
                chain = request.get("exception_chain")
                if not (isinstance(request.get("error_type"), str) and request["error_type"]
                        and isinstance(chain, list) and chain and chain[0] == request["error_type"]
                        and all(isinstance(name, str) and name for name in chain)):
                    self.issue("request_error_identity_unknown", row, request_index=request.get("request_index"))
            indices.append(request.get("request_index"))
            for key in ("prompt_tokens", "completion_tokens"):
                if not integer(request.get(key)):
                    complete = False
            if integer(request.get("prompt_tokens")):
                known_prompt += request["prompt_tokens"]
            if integer(request.get("completion_tokens")):
                known_completion += request["completion_tokens"]
            if "total_tokens" in request and all(integer(request.get(k)) for k in ("prompt_tokens", "completion_tokens")):
                if request["total_tokens"] != request["prompt_tokens"] + request["completion_tokens"]:
                    self.issue("request_usage_mismatch", row)
        if any(not integer(i) or i < 1 for i in indices) or indices != sorted(set(indices)):
            self.issue("request_index_mismatch", row)
        if row.get("model_calls") != len(requests):
            # Interrupted attempts legitimately have no call-count field.
            self.issue("request_count_mismatch" if "model_calls" in row else "request_count_unknown",
                       row, severity="error" if "model_calls" in row else "unknown")
            complete = False
        known = {"prompt_tokens": known_prompt, "completion_tokens": known_completion,
                 "total_tokens": known_prompt + known_completion}
        all_known = all(integer(r.get(k)) for r in requests for k in ("prompt_tokens", "completion_tokens"))
        for prefix in ("", "known_"):
            if all(integer(row.get(prefix + k)) for k in known):
                if row[prefix + "total_tokens"] != row[prefix + "prompt_tokens"] + row[prefix + "completion_tokens"]:
                    self.issue("token_total_mismatch", row)
                if all_known and any(row[prefix + k] != v for k, v in known.items()):
                    self.issue("request_usage_mismatch", row)
            elif prefix == "" and row.get("usage_complete") is True:
                self.issue("token_usage_unknown", row, severity="unknown")
                complete = False
        nested = row.get("token_usage")
        if nested is not None and (not isinstance(nested, dict) or all_known and nested != known):
            self.issue("nested_token_usage_mismatch", row)
        log = row.get("llm_request_log")
        if log is not None:
            if not isinstance(log, list) or len(log) != len(requests) or any(
                not isinstance(a, dict) or any(k not in a or type(a[k]) is not type(v) or a[k] != v
                    for k, v in b.items()) for a, b in zip(log, requests)):
                self.issue("request_logs_mismatch", row)
        if not complete:
            self.issue("attempt_usage_unknown", row, severity="unknown")
        return {"known_total_tokens": known["total_tokens"], "total_tokens": known["total_tokens"] if complete else None,
                "model_calls": row.get("model_calls"), "usage_complete": row.get("usage_complete"),
                "requests": [{k: r.get(k) for k in REQUEST_FIELDS} for r in requests]}

    def exposure(self, row):
        events, condition = row.get("fault_events"), row.get("condition")
        if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
            self.issue("fault_exposure_unknown", row, severity="unknown")
            return "unknown"
        if condition == "clean":
            if events:
                self.issue("clean_fault_event", row)
            return "clean"
        if row.get("action_contract_valid") is False and not events:
            return "prior_contract_failure"
        if not events:
            return "unreached"
        if len(events) != 1:
            self.issue("fault_event_count", row)
            return "unknown"
        event = events[0]
        expected_count = 0 if condition in {"request_non_delivery", "acknowledgement_loss"} else (
            2 if condition in {"duplicate_action_delivery", "same_session_reordering"} else 1)
        if (event.get("condition") != condition or event.get("boundary") != matrix.CELLS.get(condition)
                or event.get("applied") is False or event.get("reached") is False
                or event.get("delivered_count") != expected_count
                or not valid_hash(event.get("original_sha256"))
                or not isinstance(event.get("delivered_sha256"), list)
                or len(event["delivered_sha256"]) != expected_count
                or any(not valid_hash(h) for h in event["delivered_sha256"])):
            self.issue("fault_event_mismatch", row)
            return "unknown"
        if row.get("action_contract_valid") is False:
            return "prior_contract_failure"
        if row.get("action_contract_valid") is not True:
            self.issue("fault_exposure_unknown", row, severity="unknown")
            return "unknown"
        return "exposed"

    def local_replay(self, row):
        condition = row.get("condition")
        if condition not in {"same_session_reordering", "stale_judgment_replay"}:
            return
        for event in row.get("fault_events", []):
            if not isinstance(event, dict):
                continue
            candidates = []
            if condition == "same_session_reordering":
                candidates = [obj for obj in objects(row.get("events", [])) if
                              obj.get("evidence_id") == event.get("source_evidence_id") and "payload" in obj]
            else:
                for judgment in row.get("judgments", []):
                    if isinstance(judgment, dict) and judgment.get("judgment_id") == event.get("source_judgment_id"):
                        candidates.append({k: judgment.get(k) for k in ("judgment_id", "evidence_ids", "verdict")})
            if not candidates or not any(matrix.config_digest(obj) == event.get("source_sha256") for obj in candidates):
                self.issue("local_replay_source_unproven", row)

    def privacy(self, value, row=None):
        for obj in objects(value):
            for key, item in obj.items():
                if key.lower() in {"guest_cart_id", "cart_token", "guest_cart_token", "api_key", "authorization", "cookie"}:
                    if item not in (None, "", "[REDACTED]", "[redacted]"):
                        self.issue("private_token_field", row)
                if not isinstance(item, str):
                    continue
                # Scan nested JSON/prose too; findings never echo sensitive strings.
                for token in re.findall(r"/guest-carts/([^/\s\"?]+)", unquote(item)):
                    if token.lower() != "[redacted]":
                        self.issue("private_cart_url", row)
                if item.startswith(("http://", "https://")):
                    try:
                        parsed = urlsplit(item)
                        if parsed.username is not None or parsed.password is not None:
                            self.issue("url_credentials", row)
                        if any(k.lower() in {"token", "access_token", "api_key", "cart_token"}
                               and v.lower() != "[redacted]" for k, v in parse_qsl(parsed.query)):
                            self.issue("private_url_query", row)
                    except ValueError:
                        self.issue("invalid_url", row)

    def references(self, indices, receipts, row, *, allow_empty=False):
        valid = (isinstance(indices, list) and (bool(indices) or allow_empty)
                 and all(integer(i) and i < len(receipts) for i in indices))
        if not valid or len(set(indices)) != len(indices):
            self.issue("receipt_reference_invalid", row)
            return []
        return [receipts[i] for i in indices]

    def observation(self, payload, selected, row):
        """Reconstruct semantic fields from sanitized identity + cart receipts."""
        identities = [r.get("response_payload") for r in selected if r.get("purpose") == "reobserve_cart.product_page"]
        carts = [r.get("response_payload") for r in selected if r.get("purpose") == "reobserve_cart.items"]
        if len(identities) != 1 or len(carts) != 1 or not isinstance(identities[0], dict) or not isinstance(carts[0], list):
            self.issue("observation_receipts_unknown", row, severity="unknown")
            return False
        identity, items = identities[0], carts[0]
        matches = [item for item in items if isinstance(item, dict) and item.get("sku") == identity.get("sku")]
        if len(matches) > 1 or not isinstance(payload, dict):
            self.issue("environment_receipt_mismatch", row)
            return False
        item = matches[0] if matches else {}
        quantity = item.get("qty", 0)
        if type(quantity) is float and quantity.is_integer():
            quantity = int(quantity)
        task = row["task"]
        expected = {"task_id": task["task_id"], "product_title": item.get("name", ""),
            "product_id": identity.get("product_id"), "sku": identity.get("sku"),
            "requested_quantity": task["quantity"], "observed_quantity": quantity,
            "cart_verified": bool(item and item.get("name") == task["product_title"] and
                                  type(quantity) is int and quantity == task["quantity"])}
        if any(type(payload.get(k)) is not type(v) or payload.get(k) != v for k, v in expected.items()):
            self.issue("environment_receipt_mismatch", row)
            return False
        return True

    def trace(self, row):
        receipts = row.get("http_receipts")
        if not isinstance(receipts, list) or any(not isinstance(r, dict) for r in receipts):
            self.issue("http_receipts_unknown", row, severity="unknown")
            receipts = []
        for index, receipt in enumerate(receipts):
            if type(receipt.get("receipt_index")) is not int or receipt["receipt_index"] != index:
                self.issue("receipt_index_mismatch", row)
        cart_hashes = {r.get("guest_cart_id_sha256") for r in receipts if r.get("guest_cart_id_sha256") is not None}
        cart_ok = len(cart_hashes) == 1 and all(valid_hash(h) for h in cart_hashes)
        if len(cart_hashes) > 1:
            self.issue("cart_identity_mismatch", row)
        elif receipts and not cart_ok:
            self.issue("cart_identity_unknown", row, severity="unknown")
        events = row.get("events")
        if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
            self.issue("events_unknown", row, severity="unknown")
            events = []
        model_events = [e for e in events if "extra_model_call" in e]
        extra = sum(e.get("extra_model_call") is True for e in model_events)
        if len(model_events) != row.get("model_calls"):
            self.issue("model_event_count_mismatch", row)
        if [e.get("event_id") for e in events] != list(range(len(events))):
            self.issue("event_index_mismatch", row)
        if model_events and model_events[-1].get("output") != row.get("final_verdict"):
            self.issue("final_decision_event_mismatch", row)
        evaluation_indices = row.get("evaluation_receipt_indices")
        evaluation = self.references(evaluation_indices, receipts, row, allow_empty=True)
        evaluation_set = set(evaluation_indices) if evaluation and isinstance(evaluation_indices, list) else set()
        environment = row.get("environment_state")
        unavailable = isinstance(environment, dict) and environment.get("status") == "observation_unavailable"
        if evaluation:
            if evaluation_indices != list(range(min(evaluation_indices), len(receipts))):
                self.issue("evaluation_not_final_receipts", row)
            successful_reads = all(r.get("request_method") == "GET" and r.get("status_code") == 200
                                   and valid_hash(r.get("response_sha256")) for r in evaluation)
            if unavailable and not successful_reads and all(r.get("request_method") == "GET" for r in evaluation):
                self.issue("evaluation_observation_unavailable", row, severity="unknown")
            elif not successful_reads:
                self.issue("evaluation_receipt_invalid", row)
            if not unavailable or successful_reads:
                self.observation(environment, evaluation, row)
        elif isinstance(environment, dict) and environment.get("cart_verified") is True:
            self.issue("evaluation_receipts_missing", row)
        else:
            self.issue("environment_unverified", row, severity="unknown")
        request_log = row.get("model_requests", [])
        completed_times = [timestamp(r.get("completed_at")) for r in request_log if isinstance(r, dict)]
        if not completed_times or any(t is None for t in completed_times):
            self.issue("decision_timing_unknown", row, severity="unknown")
        else:
            for r in evaluation:
                observed = timestamp(r.get("timestamp"))
                if observed is None:
                    self.issue("evaluation_timing_unknown", row, severity="unknown")
                elif observed < max(completed_times):
                    self.issue("evaluation_before_decision", row)
        for obj in objects([events, row.get("judgments"), row.get("final_evidence"), row.get("recovery_events")]):
            if "http_receipt_indices" in obj:
                refs = obj["http_receipt_indices"]
                self.references(refs, receipts, row, allow_empty=True)
                if isinstance(refs, list) and any(integer(i) and i in evaluation_set for i in refs):
                    self.issue("evaluation_receipt_leak", row)
        recoveries = row.get("recovery_events", [])
        if not isinstance(recoveries, list) or any(not isinstance(e, dict) for e in recoveries):
            self.issue("recovery_schema", row)
            recoveries = []
        mitigation_refs, common_refs = set(), set()
        readback_recovery_count = 0
        used_refs = set()
        state_digest = self.evaluate.__globals__["evidence_state_digest"]
        for event in recoveries:
            refs = event.get("receipt_indices")
            selected = self.references(refs, receipts, row, allow_empty=event.get("verified") is False)
            indices = {r["receipt_index"] for r in selected if integer(r.get("receipt_index"))}
            if indices & used_refs:
                self.issue("recovery_receipt_reuse", row)
            used_refs.update(indices)
            if indices & evaluation_set:
                self.issue("evaluation_receipt_leak", row)
            target = common_refs if event.get("common") is True else mitigation_refs
            target.update(i for i in indices if receipts[i].get("request_method") == "GET")
            good_receipts = bool(selected) and all(r.get("request_method") == "GET" and
                r.get("status_code") == 200 and valid_hash(r.get("response_sha256")) and
                str(r.get("purpose", "")).startswith("reobserve_cart.") for r in selected)
            if selected and not good_receipts:
                self.issue("recovery_receipt_invalid", row)
            before, after = event.get("before"), event.get("after")
            before_hash, after_hash = state_digest(before), state_digest(after)
            hashes_match = event.get("before_sha256") == before_hash and event.get("after_sha256") == after_hash
            if not hashes_match:
                self.issue("recovery_payload_hash_mismatch", row)
            payload = after.get("payload") if isinstance(after, dict) else None
            payload_valid = isinstance(payload, dict) and check_evidence(row["task"], payload, require_success=False).valid
            observation_valid = self.observation(payload, selected, row) if good_receipts else False
            used = event.get("replacement_used") is True
            consumers = [i for i, e in enumerate(model_events) if any(obj == after for obj in objects(e.get("input")))]
            actually_in_input = bool(consumers)
            timing_ok = False
            if consumers and isinstance(request_log, list) and consumers[0] < len(request_log):
                consumer_time = timestamp(request_log[consumers[0]].get("started_at"))
                times = [timestamp(r.get("timestamp")) for r in selected]
                if consumer_time is not None and times and all(t is not None for t in times):
                    timing_ok = max(times) <= consumer_time
                    if not timing_ok:
                        self.issue("recovery_after_decision", row)
                else:
                    self.issue("recovery_timing_unknown", row, severity="unknown")
            trigger_ok = True
            if event.get("common") is True:
                prior = [e.get("output", {}) for e in model_events[:consumers[0]]] if consumers else []
                trigger_ok = any(isinstance(v, dict) and (v.get("decision") == "reject" or
                    "decision" in v and v.get("task_id") != row["task_id"]) for v in prior)
                if not trigger_ok:
                    self.issue("common_recovery_trigger_unproven", row, severity="unknown")
            if used and not actually_in_input:
                self.issue("replacement_use_unproven", row, severity="unknown")
            readback_recovery_count += int(good_receipts and payload_valid and observation_valid and used and
                actually_in_input and timing_ok and cart_ok and trigger_ok and hashes_match
                and event.get("verified") is True and before_hash != after_hash)
        if sum(e.get("common") is True for e in recoveries) > 1:
            self.issue("common_recovery_cap", row)
        detections = row.get("detection_events", [])
        detections = detections if isinstance(detections, list) else []
        replays = [e for e in detections if isinstance(e, dict) and e.get("kind") == "missing_action_receipt"]
        writes = [r for r in receipts if r.get("request_method") in {"POST", "PUT", "PATCH", "DELETE"}]
        state_writes = [r for r in writes if r.get("purpose") in {
            "add_to_cart.add_item", "add_quantity.add_item", "set_quantity.set_item"}]
        ledger = row.get("action_ledger_events", [])
        ledger = ledger if isinstance(ledger, list) else []
        confirmed = [e for e in ledger if isinstance(e, dict) and
                     e.get("event") in {"confirmed", "observed_delivery_confirmed"}]
        delivery_ids, confirmed_write_refs = set(), set()
        for event in confirmed:
            details = event.get("details", {})
            if not isinstance(details, dict):
                self.issue("ledger_confirmation_schema", row)
                continue
            delivery = details.get("delivery_id")
            if not isinstance(delivery, str) or not delivery:
                self.issue("ledger_delivery_id_unknown", row, severity="unknown")
            elif delivery in delivery_ids:
                self.issue("duplicate_ledger_delivery_id", row)
            delivery_ids.add(delivery)
            payload = details.get("receipt")
            if not isinstance(payload, dict):
                self.issue("ledger_confirmation_receipts_unknown", row, severity="unknown")
                continue
            selected = self.references(payload.get("http_receipt_indices"), receipts, row)
            links = {r["receipt_index"] for r in selected if r in state_writes}
            if not links:
                self.issue("ledger_confirmation_write_unproven", row, severity="unknown")
            if links & confirmed_write_refs:
                self.issue("ledger_confirmation_receipt_reuse", row)
            confirmed_write_refs.update(links)
        adjustment_writes = [r for r in state_writes if r.get("purpose", "").startswith(("add_quantity.", "set_quantity."))]
        for replay in replays:
            action = replay.get("action_id")
            action_events = [e.get("event") for e in ledger if isinstance(e, dict) and e.get("action_id") == action]
            fault_events = row.get("fault_events", [])
            nondelivery = any(e.get("condition") == "request_non_delivery" and e.get("delivered_count") == 0
                             for e in fault_events if isinstance(e, dict))
            action_inputs = [e.get("input") for e in events if e.get("role") == "ActionExecutor"]
            delivered = any(obj.get("action_id") == action for obj in objects(action_inputs))
            executed_writes = [r for r in state_writes if r.get("purpose", "").startswith(("add_quantity.", "set_quantity."))]
            if (not nondelivery or not delivered or not executed_writes or not action_events
                    or action_events[:3] != ["registered", "claimed", "execute_entered"]
                    or action_events.count("execute_entered") != 1 or "execution_outcome_unknown" in action_events):
                self.issue("replay_authorization_unproven", row, severity="unknown")
        actual = {"get": len(mitigation_refs), "model_call": extra, "replay": len(replays)}
        limits = {"get": 4, "model_call": 3, "replay": 1}
        budget = row.get("budget", {})
        if not isinstance(budget, dict):
            budget = {}
        if budget.get("limits") != limits:
            self.issue("budget_limits_mismatch", row)
        used = budget.get("used", {})
        remaining = budget.get("remaining", {})
        for kind, count in actual.items():
            if count > limits[kind]:
                self.issue({"get": "mitigation_get_cap", "model_call": "mitigation_model_cap", "replay": "action_replay_cap"}[kind], row)
            if not isinstance(used, dict) or used.get(kind) != count:
                self.issue("budget_receipt_mismatch" if kind == "get" else "budget_event_mismatch", row)
            if not isinstance(remaining, dict) or remaining.get(kind) != limits[kind] - count:
                self.issue("budget_remaining_mismatch", row)
        return {"mitigation_gets": actual["get"], "common_recovery_gets": len(common_refs),
                "base_gets": sum(r.get("request_method") == "GET" for r in receipts) - len(mitigation_refs | common_refs | evaluation_set),
                "evaluation_gets": len(evaluation), "mitigation_model_calls": extra,
                "base_and_common_model_calls": len(model_events) - extra, "action_replays": len(replays),
                "write_requests": len(writes), "state_write_requests": len(state_writes),
                "ledger_confirmed_deliveries": len(confirmed),
                "duplicate_adjustment_write_receipts": max(0, len(adjustment_writes) - 1),
                "actual_backend_write_count": None,
                "successful_state_write_receipts": sum(type(r.get("status_code")) is int and 200 <= r["status_code"] < 300 for r in state_writes),
                "ledger_duplicate_suppressions": sum(isinstance(e, dict) and e.get("event") == "confirmed_receipt_replayed" for e in ledger),
                "receipt_backed_recovery": bool(readback_recovery_count),
                "receipt_backed_readback_recovery_count": readback_recovery_count}

    def action_event_evidence(self, row, event):
        """Independent v1 ledger/HTTP check; does not call the runtime verifier.

        This proves recorded response recovery/prevention, not task success or
        backend exactly-once execution. The whole action history is inspected,
        including entries omitted from the event's claimed provenance links.
        """
        problems = []

        def require(condition, reason):
            if not condition:
                problems.append(reason)

        kinds = {"request_redelivery": ("recovery", "pending", 0, "missing_action_receipt"),
                 "acknowledgement_retrieval": ("recovery", "confirmed", 1, "confirmation_retrieved"),
                 "duplicate_prevention": ("prevention", "confirmed", 1, "duplicate_prevented")}
        if not isinstance(event, dict) or event.get("kind") not in kinds:
            return ["event_schema"]
        kind = event["kind"]
        outcome, before, before_count, trigger = kinds[kind]
        require(type(event.get("schema_version")) is int and event["schema_version"] == 1, "schema_version")
        require(row.get("arm") in {"action_protocol", "combined"}, "strategy")
        require(event.get("outcome") == outcome and event.get("response_used") is True, "outcome_or_use")
        require(row.get("action_outcome_unknown") is not True, "unknown_action")
        state = row.get("action_ledger_state")
        if not isinstance(state, dict) or not isinstance(state.get("receipt"), dict) or not isinstance(state.get("params"), dict):
            return problems + ["ledger_state_missing"]
        require(state.get("state") == "confirmed" and type(state.get("execution_count")) is int
                and state["execution_count"] == 1, "ledger_terminal_state")
        require(event.get("before_state") == before and type(event.get("before_execution_count")) is int
                and event["before_execution_count"] == before_count, "before_state")
        require(event.get("after_state") == "confirmed" and type(event.get("after_execution_count")) is int
                and event["after_execution_count"] == 1, "after_state")
        params, response, task = state["params"], state["receipt"], row["task"]
        for key in ("task_id", "session_id", "action_id"):
            require(isinstance(event.get(key), str) and bool(event[key]) and event[key] == state.get(key)
                    and event[key] == params.get(key), "binding_" + key)
        require(event.get("task_id") == task.get("task_id") == response.get("task_id"), "task_binding")
        require(type(response.get("cart_verified")) is bool, "response_contract")
        require(matrix.config_digest(params) == state.get("params_sha256"), "params_digest")
        operation = "add_quantity" if task["quantity"] > task["initial_quantity"] else "set_quantity"
        quantity = task["quantity"] - task["initial_quantity"] if operation == "add_quantity" else task["quantity"]
        require(params.get("operation") == operation and type(params.get("quantity")) is int
                and params["quantity"] == quantity, "operation_binding")
        require(valid_hash(event.get("receipt_sha256")) and matrix.config_digest(response) == event["receipt_sha256"], "receipt_digest")
        history = row.get("action_ledger_events")
        if not isinstance(history, list) or any(not isinstance(e, dict) or not integer(e.get("event_id")) for e in history):
            return problems + ["ledger_schema"]
        ids = [e["event_id"] for e in history]
        require(ids == sorted(set(ids)), "ledger_id_uniqueness_order")
        links = event.get("ledger_event_ids")
        if not isinstance(links, list) or not links or any(not integer(i) for i in links):
            return problems + ["ledger_links"]
        require(links == sorted(set(links)) and set(links) <= set(ids), "ledger_links")
        linked = [e for e in history if e["event_id"] in links]
        action = event.get("action_id")
        require(all(e.get("action_id") == action for e in linked), "ledger_action_binding")
        action_history = [e for e in history if e.get("action_id") == action]
        require(not any(e.get("event") == "execution_outcome_unknown" for e in action_history), "ledger_unknown")
        ordered = []
        for name in ("registered", "claimed", "execute_entered", "confirmed"):
            entries = [e for e in action_history if e.get("event") == name]
            if len(entries) != 1 or entries[0]["event_id"] not in links:
                return problems + ["ledger_" + name]
            ordered.append(entries[0])
        registration, claim, entry, confirmation = ordered
        require([e["event_id"] for e in ordered] == sorted(e["event_id"] for e in ordered), "ledger_transition_order")
        details = confirmation.get("details", {})
        require(matrix.config_digest(details.get("receipt")) == event.get("receipt_sha256"), "confirmation_receipt")
        delivery = details.get("delivery_id")
        require(isinstance(delivery, str) and bool(delivery) and
                all(e.get("details", {}).get("delivery_id") == delivery and e.get("details", {}).get("mode") == "guarded"
                    for e in (claim, entry)), "guarded_delivery_binding")
        require(registration.get("details", {}).get("params_sha256") == state.get("params_sha256"), "registration_digest")
        if kind == "duplicate_prevention":
            require(any(e.get("event") == "confirmed_receipt_replayed" and e["event_id"] > confirmation["event_id"]
                        for e in linked), "suppression_receipt")
        require(any(isinstance(e, dict) and e.get("kind") == trigger and e.get("action_id") == action
                    for e in row.get("detection_events", [])), "trigger")
        executors = [e for e in row.get("events", []) if isinstance(e, dict) and e.get("role") == "ActionExecutor"]
        require(any(obj == params for e in executors for obj in objects(e.get("input"))), "delivered_request")
        require(any(isinstance(ack, dict) and ack.get("action_id") == action and
                    matrix.config_digest(ack.get("receipt")) == event.get("receipt_sha256")
                    for e in executors for ack in e.get("output", [])), "response_consumed")
        receipts, indices = row.get("http_receipts"), event.get("receipt_indices")
        if (not isinstance(receipts, list) or any(not isinstance(r, dict) for r in receipts)
                or not isinstance(indices, list) or not indices or any(not integer(i) or i >= len(receipts) for i in indices)):
            return problems + ["http_links"]
        require(indices == sorted(set(indices)) and indices == response.get("http_receipt_indices"), "http_links")
        require(event.get("new_http_receipt_indices") == (indices if kind == "request_redelivery" else []), "new_http_links")
        require(not set(indices) & set(row.get("evaluation_receipt_indices", [])), "evaluation_isolation")
        traces = [receipts[i] for i in indices]
        require(all(r.get("receipt_index") == i for r, i in zip(traces, indices)), "http_index_binding")
        require(all(type(r.get("status_code")) is int and 200 <= r["status_code"] < 300 and not r.get("error_type")
                    and valid_hash(r.get("response_sha256")) for r in traces), "http_success_proof")
        writes = [r for r in traces if r.get("request_method") in {"POST", "PUT", "PATCH", "DELETE"}]
        all_writes = [r for r in receipts if r.get("purpose") in {"add_quantity.add_item", "set_quantity.set_item"}
                      and r.get("request_method") in {"POST", "PUT"}]
        if len(writes) != 1:
            return problems + ["single_write"]
        write = writes[0]
        require(len(all_writes) == 1 and all_writes[0] == write, "single_action_execution")
        expected_method, suffix = ("POST", ".add_item") if operation == "add_quantity" else ("PUT", ".set_item")
        item = (write.get("request_payload") or {}).get("cartItem", {})
        require(write.get("request_method") == expected_method and write.get("purpose") == operation + suffix
                and item.get("sku") == response.get("sku") and type(item.get("qty")) is int
                and item["qty"] == params.get("quantity"), "write_binding")
        cart = write.get("guest_cart_id_sha256")
        require(valid_hash(cart) and all(r.get("guest_cart_id_sha256") == cart for r in traces), "cart_binding")
        readbacks = [r for r in traces if r.get("request_method") == "GET" and r["receipt_index"] > write["receipt_index"]
                     and r.get("purpose") == operation + ".readback_items"]
        if len(readbacks) != 1 or not isinstance(readbacks[0].get("response_payload"), list):
            return problems + ["post_write_readback"]
        matches = [i for i in readbacks[0]["response_payload"] if isinstance(i, dict) and i.get("sku") == response.get("sku")]
        require(len(matches) == 1 and matches[0].get("name") == response.get("product_title")
                and type(matches[0].get("qty")) in {int, float} and matches[0]["qty"] == response.get("observed_quantity"), "readback_state")
        return problems

    def action_protocol(self, row):
        result = {"receipt_backed_action_recovery_count": 0, "receipt_backed_action_prevention_count": 0,
                  "action_event_checks": []}
        events = row.get("action_protocol_events")
        if events is None:
            result.update(receipt_backed_action_recovery_count=None, receipt_backed_action_prevention_count=None)
            self.issue("action_protocol_trace_unknown", row, severity="unknown")
            return result
        if not isinstance(events, list):
            self.issue("action_protocol_event_invalid", row, reasons=["events_schema"])
            return result
        seen = set()
        for index, event in enumerate(events):
            try:
                problems = self.action_event_evidence(row, event)
                identity = matrix.config_digest({k: event.get(k) for k in ("kind", "action_id", "ledger_event_ids")})
                if identity in seen:
                    problems.append("duplicate")
                    self.issue("action_protocol_duplicate", row, event_index=index)
                seen.add(identity)
                if type(event.get("event_id")) is not int or event["event_id"] != index:
                    problems.append("event_id")
            except (KeyError, TypeError, ValueError, AttributeError):
                problems = ["event_schema"]
            valid = not problems
            result["action_event_checks"].append({"event_index": index, "verified": valid, "reasons": problems})
            if not valid:
                self.issue("action_protocol_event_invalid", row, event_index=index, reasons=problems,
                           severity="error" if not isinstance(event, dict) or event.get("verified") is True else "unknown")
            if valid:
                key = "receipt_backed_action_prevention_count" if event["outcome"] == "prevention" else "receipt_backed_action_recovery_count"
                result[key] += 1
        return result

    def cross_source(self, row, rows, tasks, digest):
        if row.get("condition") not in SOURCE_CONDITIONS:
            return None
        start = len(self.findings)
        source = row.get("cross_task_source")
        if not isinstance(source, dict) or not isinstance(source.get("file"), str):
            self.issue("cross_source_missing", row, severity="unknown")
            return False
        data = self.read_bytes(source["file"], required=True)
        if data is None:
            return False
        if sha256(data) != source.get("file_sha256"):
            self.issue("cross_source_file_hash_mismatch", row)
        try:
            frozen = strict_json(data)
        except (ValueError, UnicodeDecodeError):
            self.issue("cross_source_schema", row)
            return False
        if not isinstance(frozen, dict) or frozen != {k: v for k, v in source.items() if k not in {"file", "file_sha256"}}:
            self.issue("cross_source_frozen_mismatch", row)
        target = [row.get("task_id"), row.get("topology"), row.get("repeat_index")]
        expected_file = "cross_task_sources/" + matrix.config_digest(target) + ".json"
        if source.get("target") != target or source.get("config_digest") != digest or source["file"] != expected_file:
            self.issue("cross_source_target_mismatch", row)
        envelope = source.get("envelope")
        if matrix.config_digest(envelope) != source.get("envelope_sha256"):
            self.issue("cross_source_envelope_hash_mismatch", row)
        candidates = [r for r in rows if r.get("run_id") == source.get("source_run_id")
                      and r.get("job_key") == source.get("source_job_key")]
        valid = len(candidates) == 1
        if valid:
            candidate = candidates[0]
            st, tt = tasks.get(candidate.get("task_id"), {}), tasks.get(row.get("task_id"), {})
            payload = envelope.get("payload") if isinstance(envelope, dict) else None
            truth = candidate.get("environment_state")
            valid = (candidate.get("condition") == "clean" and candidate.get("arm") == "baseline"
                and candidate.get("config_digest") == digest and candidate.get("task_id") == source.get("source_task_id")
                and st and tt and st["task_id"] != tt["task_id"] and st["product_title"] != tt["product_title"]
                and urlsplit(st["product_url"]).path != urlsplit(tt["product_url"]).path
                and isinstance(envelope, dict) and envelope == candidate.get("source_evidence")
                and envelope.get("task_id") == st["task_id"] and envelope.get("source") == "Worker"
                and all(envelope.get(k) for k in ("session_id", "entity_id", "action_id", "evidence_id"))
                and type(envelope.get("version")) is int and isinstance(payload, dict) and isinstance(truth, dict)
                and check_evidence(st, payload).valid and all(payload.get(k) == truth.get(k)
                    for k in ("product_id", "sku", "observed_quantity", "cart_verified")))
            try:
                valid = valid and self.evaluate(candidate)["final_task_success"] is True
            except (KeyError, TypeError, ValueError):
                valid = False
            if valid:
                for clean in rows:
                    p = clean.get("source_evidence")
                    if (clean.get("task_id") == row.get("task_id") and clean.get("condition") == "clean"
                            and clean.get("arm") == "baseline" and isinstance(p, dict) and isinstance(p.get("payload"), dict)):
                        if any(p["payload"].get(k) == payload.get(k) for k in ("sku", "product_id")):
                            valid = False
        if not valid:
            self.issue("cross_source_not_actual_eligible_clean", row)
        for event in row.get("fault_events", []):
            if not isinstance(event, dict):
                continue
            if (event.get("source_sha256") != source.get("envelope_sha256") or not isinstance(envelope, dict)
                    or event.get("source_evidence_id") != envelope.get("evidence_id")):
                self.issue("cross_source_event_mismatch", row)
        return valid and len(self.findings) == start

    def pair_results(self, cases):
        groups = defaultdict(list)
        for case in cases:
            groups[case["pair_key"]].append(case)
        complete = {key: {r["arm"]: r for r in group} for key, group in groups.items()
                    if len(group) == 7 and {r["arm"] for r in group} == set(matrix.ARMS)}
        clean = {(r["task_id"], r["topology"], r["repeat_index"]): group for group in complete.values()
                 for r in [group["baseline"]] if r["condition"] == "clean"}
        metrics = ("environment_task_success", "decision_correct", "final_task_success", "evidence_acceptance_errors")
        subsets = ("all_samples", "common_clean_success", "all_fault_arms_exposed", "common_clean_success_exposed")
        comparisons = {subset: {arm: {metric: {"improvements": 0, "regressions": 0, "unchanged": 0,
            "unknown": 0, "representative_case_ids": {"improvement": [], "regression": []}}
            for metric in metrics} for arm in matrix.ARMS[1:]} for subset in subsets}
        pairs = []
        for key, group in complete.items():
            baseline = group["baseline"]
            clean_group = clean.get((baseline["task_id"], baseline["topology"], baseline["repeat_index"]), {})
            common = len(clean_group) == 7 and all(r["final_task_success"] is True for r in clean_group.values())
            exposed = baseline["condition"] != "clean" and all(r["exposure"] == "exposed" for r in group.values())
            for arm in matrix.ARMS[1:]:
                other = group[arm]
                pair = {"pair_key": key, "task_id": baseline["task_id"], "topology": baseline["topology"],
                    "condition": baseline["condition"], "repeat_index": baseline["repeat_index"], "arm": arm,
                    "baseline_run_id": baseline["run_id"], "strategy_run_id": other["run_id"],
                    "common_clean_success": common, "all_fault_arms_exposed": exposed}
                for metric in metrics:
                    a, b = baseline.get(metric), other.get(metric)
                    if a is None or b is None:
                        change = "unknown"
                    elif a == b:
                        change = "unchanged"
                    else:
                        improved = b < a if metric == "evidence_acceptance_errors" else b > a
                        change = "improvement" if improved else "regression"
                    pair[metric + "_change"] = change
                    for subset, include in zip(subsets, (True, common, exposed, common and exposed)):
                        if not include:
                            continue
                        stats = comparisons[subset][arm][metric]
                        stats[change + "s" if change in {"improvement", "regression"} else change] += 1
                        if change in {"improvement", "regression"} and len(stats["representative_case_ids"][change]) < 3:
                            stats["representative_case_ids"][change].append([baseline["run_id"], other["run_id"]])
                a, b = baseline["total_tokens"], other["total_tokens"]
                pair["token_delta"] = None if a is None or b is None else b - a
                pairs.append(pair)
        return pairs, comparisons, {"complete_seven_arm_pairs": len(complete),
            "incomplete_pair_keys": [key for key in groups if key not in complete],
            "common_clean_success_groups": sum(all(r["final_task_success"] is True for r in group.values()) for group in clean.values())}

    def audit(self):
        config, digest, jobs, planned = self.manifest()
        limitations = [*LIMITATIONS,
            f"冻结计划：{len(config['tasks'])} 个任务实例，每个任务/拓扑/条件/策略 {config['repetitions']} 次重复；"
            "这是计划规模，实际完成量与覆盖率另列。"]
        if len(config["tasks"]) <= 3 and config["repetitions"] == 1:
            limitations.append("当前小规模单次重复设计仅提供机制描述，不足以支持总体推断。")
        self.evaluation["limitations"] = limitations
        self.privacy(config)
        provenance = self.provenance(config)
        rows = self.read_jsonl("main_runs.jsonl", required=True)
        starts = self.read_jsonl("run_attempts.jsonl", required=True)
        errors = self.read_jsonl("run_errors.jsonl")
        blocked = self.read_jsonl("blocked_cells.jsonl")
        for cell in blocked:
            if cell.get("config_digest") != digest or cell.get("pair_key") not in {j["pair_key"] for j in jobs.values()}:
                self.issue("blocked_cell_mismatch")
        unfinished = self.attempts(starts, rows, errors, jobs, digest)
        cases, usages, attempt_cases = [], [], []
        tasks = {t["task_id"]: t for t in config["tasks"]}
        for row in rows:
            self.privacy(row, row)
            if row.get("common_recovery_enabled") is not True:
                self.issue("common_recovery_disabled", row)
            if row.get("task") != tasks.get(row.get("task_id")):
                self.issue("task_mismatch", row)
            usage = self.usage(row)
            usages.append(usage)
            exposure = self.exposure(row)
            self.local_replay(row)
            try:
                trace = self.trace(row)
            except (KeyError, TypeError, ValueError, AttributeError):
                self.issue("trace_schema", row)
                trace = {}
            action_trace = self.action_protocol(row)
            source_verified = self.cross_source(row, rows, tasks, digest)
            try:
                evaluated = self.evaluate(row)
                fields = DERIVED_FIELDS + (ACTION_DERIVED_FIELDS if "action_protocol_events" in row
                    or any(k in row for k in ACTION_DERIVED_FIELDS) else ())
                for field in fields:
                    if field not in row or type(row[field]) is not type(evaluated[field]) or row[field] != evaluated[field]:
                        self.issue("evaluation_mismatch", row, field=field)
                for index, (saved, checked) in enumerate(zip(row.get("action_protocol_events", []),
                                                            evaluated.get("action_protocol_events", []))):
                    if type(saved.get("verified")) is not bool or saved["verified"] != checked["verified"]:
                        self.issue("action_protocol_verification_mismatch", row, event_index=index)
            except (KeyError, TypeError, ValueError, AttributeError):
                self.issue("evaluation_schema", row)
                evaluated = {}
            cases.append({**{k: row.get(k) for k in (*JOB_FIELDS, *IDENTITY_FIELDS, "run_id", "attempt_id")},
                          **{k: evaluated.get(k) for k in DERIVED_FIELDS + ACTION_DERIVED_FIELDS}, "exposure": exposure, **usage,
                          **trace, **action_trace, "cross_source_verified": source_verified,
                          "action_outcome_unknown": row.get("action_outcome_unknown"),
                          "final_commit_allowed": row.get("final_commit_allowed"),
                          "status": row.get("status", "completed")})
            attempt_cases.append({**{k: row.get(k) for k in (*JOB_FIELDS, *IDENTITY_FIELDS, "run_id", "attempt_id", "attempt")},
                                  "terminal_kind": "completed", **usage})
        for row in errors:
            self.privacy(row, row)
            usage = self.usage(row, failed=True)
            usages.append(usage)
            attempt_cases.append({**{k: row.get(k) for k in (*JOB_FIELDS, *IDENTITY_FIELDS, "run_id", "attempt_id", "attempt", "status", "error_type")},
                                  "terminal_kind": "error", **usage})
        # Request indices are process-local and can reset on resume. They are
        # checked within attempts, not (incorrectly) required globally unique.
        usages.extend({"known_total_tokens": 0, "total_tokens": None} for _ in unfinished)
        attempt_cases.extend({**{k: row.get(k) for k in (*JOB_FIELDS, *IDENTITY_FIELDS, "attempt_id", "attempt")},
                              "terminal_kind": "unfinished", "known_total_tokens": 0,
                              "total_tokens": None} for row in unfinished)
        missing = [key for key in jobs if key not in {r.get("job_key") for r in rows}]
        unknown = sum(u["total_tokens"] is None for u in usages)
        known = sum(u["known_total_tokens"] for u in usages)
        tokens = {"known_total_tokens": known, "total_tokens": None if unknown else known,
                  "unknown_attempts": unknown, "completed_known_tokens": sum(u["known_total_tokens"] for u in usages[:len(rows)]),
                  "error_attempts_known_tokens": sum(u["known_total_tokens"] for u in usages[len(rows):])}
        exposure_counts = Counter(c["exposure"] for c in cases)
        pairs, comparisons, pairing = self.pair_results(cases)
        original_summary = self.read_json("summary.json")
        if "exposure" in original_summary:
            self.check_summary_exposure(original_summary["exposure"], rows, errors, "exposure")
        summary_expected = {"planned_runs": len(jobs), "completed_runs": len(rows), "error_attempts": len(errors),
            "complete_seven_arm_pairs": pairing["complete_seven_arm_pairs"],
            "total_tokens_completed": sum(u["known_total_tokens"] for u in usages[:len(rows)]),
            "known_tokens_error_attempts": sum(u["known_total_tokens"] for u in usages[len(rows):len(rows) + len(errors)]),
            "usage_incomplete_attempts": sum(r.get("usage_complete") is not True for r in rows + errors)}
        for key, value in summary_expected.items():
            if key in original_summary and original_summary[key] != value:
                self.issue("summary_count_mismatch", field=key)
        if original_summary.get("status") == "complete" and (missing or unfinished):
            self.issue("summary_count_mismatch", field="status")
        by_arm = {}
        action_count_fields = ("receipt_backed_action_recovery_count", "receipt_backed_action_prevention_count",
                               "receipt_backed_readback_recovery_count")
        for arm in matrix.ARMS:
            values = [r for r in cases if r["arm"] == arm]
            by_arm[arm] = {"runs": len(values), **{k: sum(r.get(k) is True for r in values) for k in
                ("environment_task_success", "decision_correct", "final_task_success")},
                "evidence_acceptance_errors": sum(r.get("evidence_acceptance_errors") or 0 for r in values),
                "known_total_tokens": sum(r["known_total_tokens"] for r in values),
                "exposure": dict(Counter(r["exposure"] for r in values)),
                "error_attempts": sum(r.get("arm") == arm for r in errors),
                **{k: sum(r.get(k) or 0 for r in values) for k in action_count_fields},
                "action_trace_unknown_runs": sum(r.get("receipt_backed_action_recovery_count") is None for r in values)}
            reported_arm = original_summary.get("by_arm", {}).get(arm, {})
            for key, value in by_arm[arm].items():
                if key == "exposure":
                    if key in reported_arm:
                        self.check_summary_exposure(reported_arm[key],
                            [r for r in rows if r.get("arm") == arm],
                            [r for r in errors if r.get("arm") == arm], "by_arm." + arm + ".exposure")
                    continue
                if key in reported_arm and reported_arm[key] != value:
                    self.issue("summary_count_mismatch", field="by_arm." + arm + "." + key)
        cells = defaultdict(list)
        for case in cases:
            cells[(case["task_id"], case["topology"], case["condition"], case["repeat_index"])].append(case)
        by_cell = [{**dict(zip(("task_id", "topology", "condition", "repeat_index"), key)),
            "runs": len(values), "exposure": dict(Counter(r["exposure"] for r in values)),
            "propagation_classes": dict(Counter(r["propagation_class"] for r in values))}
            for key, values in cells.items()]
        starts_per_job = Counter(s.get("job_key") for s in starts)
        return {"audit_version": "shopping-multimechanism-offline-v1", "offline": True,
                **{k: config.get(k) for k in IDENTITY_FIELDS}, "model_identity_immutable": False,
                "status": "failed" if any(f["severity"] == "error" for f in self.findings) else
                    "incomplete" if missing or self.findings or config["shard_count"] != 1 else "complete",
                "coverage": {"planned_runs": planned, "shard_runs": len(jobs), "completed_runs": len(rows),
                    "planned_clean_runs": len(tasks) * 3 * 7 * config["repetitions"],
                    "planned_fault_runs": len(tasks) * 3 * 7 * 9 * config["repetitions"],
                    "completed_unique_jobs": len({r.get("job_key") for r in rows}),
                    "missing_job_keys": missing, "error_attempts": len(errors), "unfinished_attempts": len(unfinished),
                    "exhausted_job_keys": [key for key in missing if starts_per_job[key] >= 2],
                    "blocked_pair_keys": list(dict.fromkeys(c.get("pair_key") for c in blocked)),
                    "full_matrix_coverage": not missing and len(rows) == planned and config["shard_count"] == 1},
                "exposure": {"planned_fault_runs": sum(c["condition"] != "clean" for c in cases),
                    "exposed_fault_runs": exposure_counts["exposed"], **dict(exposure_counts)},
                "tokens": tokens, "evaluation": self.evaluation, "provenance": provenance,
                "inputs": self.inputs, "findings": self.findings, "cases": cases, "pairs": pairs,
                "attempt_cases": attempt_cases,
                "comparisons": comparisons, "pairing": pairing, "by_arm": by_arm, "by_cell": by_cell,
                "action_protocol": {"verified_recoveries": sum(r.get("receipt_backed_action_recovery_count") or 0 for r in cases),
                    "verified_preventions": sum(r.get("receipt_backed_action_prevention_count") or 0 for r in cases),
                    "verified_readback_recoveries": sum(r.get("receipt_backed_readback_recovery_count") or 0 for r in cases),
                    "unknown_trace_runs": sum(r.get("receipt_backed_action_recovery_count") is None for r in cases),
                    "validation": "independent recorded ledger and HTTP receipt checks; not backend or task-success proof"},
                "outcomes": {"unresolved_completed_trials": sum(r.get("action_outcome_unknown") is True or
                    r.get("status") == "unresolved" or r.get("final_commit_allowed") is False for r in rows),
                    "infra_error_attempts": sum(r.get("status") == "infra_error" for r in errors),
                    "timeout_attempts": sum(r.get("status") == "timeout" for r in errors),
                    "error_attempts_exposure_unknown": sum(r.get("condition") != "clean" for r in errors)},
                "statistical_inference": {"performed": False,
                    "reason": "descriptive paired audit only; no significance or seed-independence claims"},
                "product_count_by_manifest_url": len({urlsplit(t["product_url"]).path for t in tasks.values()}),
                "task_count": len(tasks), "repetitions": config["repetitions"], "limitations": limitations}


def write_reports(output, report):
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    for name, key in (("cases", "cases"), ("pairs", "pairs"), ("attempts", "attempt_cases")):
        values = report[key]
        fields = list(dict.fromkeys(k for row in values for k in row)) or ["run_id"]
        with (output / f"{name}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()} for row in values)
    lines = ["# 多机制 Shopping 离线审计", "", f"状态：{report['status']}。",
             f"全矩阵计划 {report['coverage']['planned_runs']}；完成记录 {report['coverage']['completed_runs']}。",
             f"已知 tokens：{report['tokens']['known_total_tokens']}；未知尝试：{report['tokens']['unknown_attempts']}。",
             "未知用量不按零消耗解释。", "", *report["limitations"]]
    lines += ["", "## 全样本", "", "| 策略 | 完成 | 环境成功 | 决策正确 | 最终证据成功 | 历史错误接受 | 错误尝试 | 已知tokens |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, stats in report["by_arm"].items():
        lines.append(f"| {arm} | {stats['runs']} | {stats['environment_task_success']} | {stats['decision_correct']} | "
                     f"{stats['final_task_success']} | {stats['evidence_acceptance_errors']} | {stats['error_attempts']} | "
                     f"{stats['known_total_tokens']} |")
    lines += ["", "## 动作协议与读回复查", "", "动作响应恢复不等于任务成功；重复预防单列，不计作恢复。",
              "| 策略 | 动作响应恢复 | 重复预防 | 读回复查恢复 | 动作trace未知 |",
              "|---|---:|---:|---:|---:|"]
    for arm, stats in report["by_arm"].items():
        lines.append(f"| {arm} | {stats['receipt_backed_action_recovery_count']} | {stats['receipt_backed_action_prevention_count']} | "
                     f"{stats['receipt_backed_readback_recovery_count']} | {stats['action_trace_unknown_runs']} |")
    lines += ["", f"计划故障已完成记录：{report['exposure']['planned_fault_runs']}；实际暴露：{report['exposure']['exposed_fault_runs']}。",
              "未到达、先前动作契约失败与未知暴露分别保留，不加入实际暴露收益分母。",
              f"完整七组配对：{report['pairing']['complete_seven_arm_pairs']}；共同 clean 成功组：{report['pairing']['common_clean_success_groups']}。",
              "", "## 配对改善与退化", "", "| 子集 | 策略 | 指标 | 改善 | 退化 | 不变 | 未知 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for subset, arms in report["comparisons"].items():
        for arm, metrics in arms.items():
            for metric, stats in metrics.items():
                lines.append(f"| {subset} | {arm} | {metric} | {stats['improvements']} | {stats['regressions']} | "
                             f"{stats['unchanged']} | {stats['unknown']} |")
    lines += ["", "## 代表配对", "", "以下 ID 仅定位描述性案例，不代表显著性；完整记录见 pairs.csv。"]
    representatives = sorted(report["pairs"], key=lambda p: p["final_task_success_change"] == "unchanged")[:6]
    for pair in representatives:
        lines.append(f"- {pair['baseline_run_id']} / {pair['strategy_run_id']}：{pair['arm']}，"
                     f"{pair['condition']}；最终证据结果 {pair['final_task_success_change']}，"
                     f"环境结果 {pair['environment_task_success_change']}。")
    lines += ["", "## 审计边界", "", "预算按实际引用 GET 收据、extra_model_call 事件与动作重投事件核算；"
              "基础和共有恢复成本单列，成功写响应不等同于独立证明真实写入次数或 exactly-once。",
              "未验证项和完整 findings 见 summary.json；审计不会修复输入或补造缺失任务。"]
    with (output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        source, output = args.result_dir.resolve(), args.output_dir.resolve()
        if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must be new and separate from source")
        if args.source_root is not None and output.is_relative_to(args.source_root.resolve()):
            raise ValueError("output must be outside source-root")
        report = Auditor(source, args.source_root).audit()
        write_reports(output, report)
        return int(not args.no_strict and report["status"] != "complete")
    except (OSError, ValueError, TypeError, KeyError):
        print("Offline audit stopped: invalid input/schema or unsafe/existing output; source is unchanged.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
