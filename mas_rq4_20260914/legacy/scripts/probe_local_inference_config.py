"""Short JSON-decision microbenchmark: 2 warmups, 12 measured calls on ONE service.

Linux only. Never launches/stops services or allocates GPUs. Engine activation,
hardware support, cross-config selection and final clean gates remain external.
No retries, paid fallbacks, adaptive prompts, or automatic formal experiments.
This does not measure full-task throughput.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
import sys
import time
import urllib.request
import urllib.response
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_shopping_multimechanism as runner
from mas_faults.llm_client import OpenAICompatibleHTTPClient, build_completion_payload, get_llm_client

# This local-vLLM microbenchmark is not a Flash matrix entry point.
MODEL = "Qwen/Qwen3.8-27B"
PROVIDER = "modelscope_local"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode())


def strict_json_loads(value):
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique_object)


def checked_settings():
    if os.environ.get("LLM_PROVIDER") != PROVIDER or os.environ.get("LLM_MODEL") != MODEL:
        raise ValueError("explicit local Qwen model and provider required")
    base = runner.validate_loopback_url(os.environ.get("LLM_BASE_URL", ""))
    socket_timeout = int(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "60"))
    total_timeout = int(os.environ.get("LLM_TOTAL_REQUEST_TIMEOUT_SECONDS", str(socket_timeout)))
    max_tokens = int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else None
    settings = {"provider": PROVIDER, "model": MODEL, "api_base_url": runner.openai_api_base(base),
        "temperature": 0, "max_tokens": max_tokens,
        "disable_thinking": os.environ.get("LLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes"},
        "socket_timeout_seconds": socket_timeout, "total_timeout_seconds": total_timeout}
    url = urlsplit(settings["api_base_url"])
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.path != "/v1"
            or not url.port or not os.environ.get("LLM_API_KEY", "").strip()
            or not settings["disable_thinking"] or type(settings["max_tokens"]) is not int
            or not 1 <= settings["max_tokens"] <= 4096
            or min(socket_timeout, total_timeout) <= 0
            or max(settings["socket_timeout_seconds"], settings["total_timeout_seconds"]) > 600):
        raise ValueError("explicit IPv4-loopback Qwen, thinking off, bounded tokens/timeouts and local key required")
    return settings


def preflight_client(settings):
    if checked_settings() != settings:
        raise ValueError("local inference settings changed before preflight")
    client = get_llm_client()
    info = client.model_info
    if (info.provider != PROVIDER or info.model != MODEL or info.client_type != "http_openai_compatible"
            or runner.openai_api_base(runner.validate_loopback_url(info.base_url)) != settings["api_base_url"]):
        raise ValueError("client does not match the local Qwen model")
    request = urllib.request.Request(settings["api_base_url"] + "/models",
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}, method="GET")
    with runner.local_inference_transport(), urllib.request.urlopen(request, timeout=10) as response:
        served = json.load(response)
    if MODEL not in {item.get("id") for item in served.get("data", []) if isinstance(item, dict)}:
        raise ValueError("required model is absent from local inference server")
    return client, {"model": MODEL, "provider": PROVIDER, "timestamp_unix": time.time(),
                    "served_models_sha256": object_hash(served), "model_calls": 0}


VALUE_FLAGS = {"served-model-name", "host", "port", "tensor-parallel-size", "dtype", "max-model-len",
               "max-num-seqs", "max-num-batched-tokens", "gpu-memory-utilization", "reasoning-parser",
               "default-chat-template-kwargs", "generation-config", "compilation-config", "seed"}
BOOL_FLAGS = {"language-model-only", "enforce-eager", "disable-custom-all-reduce"}
ENV_ALLOWLIST = {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "VLLM_USE_MODELSCOPE",
                 "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK", "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR",
                 "CUDA_CACHE_PATH", "LD_PRELOAD"}


def _launch_flags(argv):
    try:
        start = argv.index("serve")
        if start < 1 or Path(argv[start - 1]).name != "vllm":
            raise ValueError()
        model = Path(argv[start + 1])
        if not model.is_absolute():
            raise ValueError()
        flags, index = {}, start + 2
        while index < len(argv):
            token = argv[index]
            key, equal, value = token.removeprefix("--").partition("=")
            if not token.startswith("--") or key in flags:
                raise ValueError()
            if key in BOOL_FLAGS and not equal:
                flags[key] = True
            elif key in VALUE_FLAGS:
                if not equal:
                    index += 1
                    value = argv[index]
                flags[key] = value
            else:
                raise ValueError()
            index += 1
        return model.resolve(), flags
    except (ValueError, IndexError):
        raise ValueError("unsupported or ambiguous service launch flags") from None


def _model_artifacts(model, manifest_path):
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("file_hashes")
    if manifest.get("model_id") != MODEL or not isinstance(entries, list) or not entries:
        raise ValueError("verified local model manifest required")
    hashes, weights = {}, {}
    for entry in entries:
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or name in hashes:
            raise ValueError("invalid artifact path")
        path = model / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(model):
            raise ValueError("model artifact unavailable")
        stat = path.stat()
        expected = entry.get("sha256")
        if (type(entry.get("size")) is not int or stat.st_size != entry["size"]
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("model artifact manifest mismatch")
        hashes[name] = expected
        if name.endswith(".safetensors"):
            weights[name] = {"manifest_sha256": expected, "size": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
        else:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("model/template artifact hash mismatch")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
                "generation_config.json", "model.safetensors.index.json"}
    if not required.issubset(hashes) or not weights:
        raise ValueError("model/tokenizer/template provenance incomplete")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if set(index.get("weight_map", {}).values()) != set(weights):
        raise ValueError("weight index and manifest disagree")
    return {"model_manifest_sha256": sha256(manifest_bytes), "artifact_hashes": hashes,
            "weight_inventory": weights, "weight_content_rehashed": False}


def capture_service(pid, settings, model_manifest, *, proc_root=Path("/proc")):
    """Read-only, allowlisted process config bound to the endpoint's LISTEN socket.

    Launch requests are observed, not proof of graph/custom-allreduce activation.
    Small artifacts are rehashed; weights use manifest hashes plus live stat data.
    """
    if type(pid) is not int or pid <= 0:
        raise ValueError("a positive API-server PID is required")
    proc_root = Path(proc_root)
    process = proc_root / str(pid)
    if process.stat().st_uid != os.getuid():
        raise ValueError("API server must belong to the current user")
    stat = (process / "stat").read_text()
    start_ticks = stat.rsplit(")", 1)[1].split()[19]
    argv = (process / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    model, flags = _launch_flags(argv)
    expected = {"served-model-name": MODEL, "host": "127.0.0.1",
                "port": str(urlsplit(settings["api_base_url"]).port), "tensor-parallel-size": "4",
                "dtype": "bfloat16", "max-model-len": "16384", "max-num-seqs": "4",
                "max-num-batched-tokens": "2048", "gpu-memory-utilization": "0.72",
                "language-model-only": True, "reasoning-parser": "qwen3", "generation-config": "vllm"}
    if any(flags.get(k) != v for k, v in expected.items()):
        raise ValueError("service invariants differ from the fixed BF16 TP4 configuration")
    if json.loads(flags.get("default-chat-template-kwargs", "null")) != {"enable_thinking": False}:
        raise ValueError("service thinking/template configuration differs")
    sockets = {link.readlink().as_posix() for link in (process / "fd").iterdir() if link.is_symlink()}
    wanted = "0100007F:" + format(urlsplit(settings["api_base_url"]).port, "04X")
    listeners = {fields[9] for fields in (line.split() for line in (proc_root / "net" / "tcp").read_text().splitlines()[1:])
                 if len(fields) > 9 and fields[1] == wanted and fields[3] == "0A"}
    if len(listeners) != 1:
        raise ValueError("configured endpoint must have one unambiguous LISTEN inode")
    listener_inode = next(iter(listeners))
    if f"socket:[{listener_inode}]" not in sockets:
        raise ValueError("PID does not own the unambiguous loopback listening socket")
    environ = {}
    for entry in (process / "environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key.decode(errors="replace") in ENV_ALLOWLIST:
            environ[key.decode()] = value.decode()
    if len(set(environ.get("CUDA_VISIBLE_DEVICES", "").split(","))) != 4:
        raise ValueError("four explicit service GPU IDs required; no GPU allocation is performed")
    artifacts = _model_artifacts(model, model_manifest)
    if (process / "stat").read_text().rsplit(")", 1)[1].split()[19] != start_ticks:
        raise ValueError("service changed during capture")
    return {"pid": pid, "start_ticks": start_ticks, "listener_inode": listener_inode, "api_base_url": settings["api_base_url"],
            "model_path": str(model), "flags": flags, "environment": environ,
            "engine_activation_verified": False, "startup_compile_seconds": None,
            "config_evidence": "proc_cmdline_environ_and_owned_listener", **artifacts}


def build_request_manifest(settings):
    """Deterministic legal decision cases; never generated from model outputs."""
    requests = []
    for number, (target, observed) in enumerate(((2, 2), (2, 1), (1, 3)), 1):
        case_id = f"config-case-{number}"
        task = {"task_id": case_id, "product_title": "Green Tea", "sku": "TEA-01", "quantity": target}
        evidence = {"task_id": case_id, "product_title": "Green Tea", "sku": "TEA-01", "observed_quantity": observed}
        prompt = ('Decide whether the cart observation satisfies the task. Accept only when task_id, product_title, '
                  'sku and quantity match. Return exactly one JSON object with task_id and decision '
                  '("accept" or "reject"); no other fields or commentary.\nInput: '
                  + json.dumps({"task": task, "observation": evidence}, sort_keys=True, separators=(",", ":")))
        payload = build_completion_payload(provider=settings["provider"], model=settings["model"], prompt=prompt,
                                           json_mode=True, disable_thinking=True)
        payload_utf8 = json.dumps(payload)
        requests.append({"case_id": case_id, "prompt": prompt, "prompt_sha256": sha256(prompt.encode()),
                         "payload_utf8": payload_utf8, "payload_sha256": sha256(payload_utf8.encode()),
                         "expected_decision": "accept" if target == observed else "reject"})
    return requests


class _RecordingHandler(urllib.request.BaseHandler):
    handler_order = 100

    def __init__(self, capture, endpoint):
        self.capture, self.endpoint = capture, endpoint

    def http_request(self, request):
        if request.full_url != self.endpoint or request.get_method() != "POST":
            raise ValueError("only the configured local completion endpoint is permitted")
        return request

    https_request = http_request

    def http_response(self, request, response):
        body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("completion response exceeds the bounded capture size")
        self.capture.append({"status_code": response.code, "payload_sha256": sha256(request.data),
                             "wire_response_sha256": sha256(body), "body": body})
        replay = urllib.response.addinfourl(io.BytesIO(body), response.info(), response.geturl(), response.code)
        replay.msg = getattr(response, "msg", "response")
        response.close()
        return replay

    https_response = http_response


@contextmanager
def recording_transport(capture, *, handlers=()):
    settings = checked_settings()
    old = urllib.request._opener
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), runner._NoRedirect(),
                                         _RecordingHandler(capture, settings["api_base_url"] + "/chat/completions"), *handlers)
    urllib.request.install_opener(opener)
    try:
        yield
    finally:
        urllib.request._opener = old


def _raw_usage(data):
    usage = data.get("usage") if isinstance(data, dict) else None
    complete = (isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] > 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"])
    return {"usage_complete": bool(complete),
            **{k: usage[k] if complete else None for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}


def execute_request(spec, *, transport_factory=None):
    """One actual-client request on a spawn worker's main thread, with no retry."""
    row = {k: spec[k] for k in ("request_id", "phase", "round", "mode")}
    case, capture, client = spec["case"], [], None
    row.update(case_id=case["case_id"], prompt_sha256=case["prompt_sha256"], payload_sha256=case["payload_sha256"],
               model=MODEL, provider=PROVIDER, status="error", passed=False, semantic_valid=False,
               model_calls=0, response_text=None, response_sha256=None, parsed=None, raw_usage=None, **_raw_usage(None))
    started = time.perf_counter()
    try:
        settings = checked_settings()
        if settings != spec["settings"] or sys.platform != "linux" and transport_factory is None:
            raise ValueError("Linux with unchanged explicit local settings required")
        if case not in build_request_manifest(settings):
            raise ValueError("request is not an exact frozen legal case")
        client = OpenAICompatibleHTTPClient(api_key=os.environ["LLM_API_KEY"], base_url=settings["api_base_url"],
                    model=MODEL, provider=PROVIDER, timeout_seconds=settings["socket_timeout_seconds"])
        with (transport_factory or recording_transport)(capture):
            content = client.complete(case["prompt"], json_mode=True)
        row.update(status="completed", response_text=content, response_sha256=sha256(content.encode()))
        if len(capture) != 1 or capture[0]["payload_sha256"] != case["payload_sha256"]:
            raise ValueError("request capture does not match manifest")
        data = strict_json_loads(capture[0]["body"])
        raw_usage = data.get("usage")
        row["raw_usage"] = ({k: raw_usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in raw_usage}
                            if isinstance(raw_usage, dict) else None)
        row.update(_raw_usage(data))
        row["wire_response_sha256"] = capture[0]["wire_response_sha256"]
        row["served_model"] = data.get("model")
        choice = data["choices"][0]
        row["finish_reason"] = choice.get("finish_reason")
        row["parsed"] = strict_json_loads(content)
        row["semantic_valid"] = row["parsed"] == {"task_id": case["case_id"], "decision": case["expected_decision"]}
        row["passed"] = bool(row["semantic_valid"] and row["usage_complete"] and row["served_model"] == MODEL
                             and len(data["choices"]) == 1 and choice.get("finish_reason") == "stop"
                             and choice.get("message", {}).get("content") == content
                             and capture[0]["status_code"] == 200)
    except Exception as exc:
        row.update(status="error", passed=False, error_type=type(exc).__name__)
    finally:
        row["model_calls"] = client.call_count if client is not None else 0
        row["latency_seconds"] = time.perf_counter() - started
        row["http_captures"] = [{k: v for k, v in item.items() if k != "body"} for item in capture]
        row["client_requests"] = client.request_log if client is not None else []
    return row


def _pool(max_workers):
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))


def _cost(rows):
    return sum(r["total_tokens"] for r in rows) if rows and all(r.get("usage_complete") is True for r in rows) else None


def _record_passes(row, case):
    """Independently check worker observations, not just a worker's pass flag."""
    try:
        captures, content = row.get("http_captures"), row.get("response_text")
        usage = _raw_usage({"usage": row.get("raw_usage")})
        return bool(row.get("passed") is True and row.get("status") == "completed" and not row.get("error_type")
                    and row.get("semantic_valid") is True and usage["usage_complete"]
                    and all(row.get(k) == v for k, v in usage.items())
                    and row.get("model") == row.get("served_model") == MODEL and row.get("provider") == PROVIDER
                    and type(row.get("model_calls")) is int and row["model_calls"] == 1 and row.get("finish_reason") == "stop"
                    and isinstance(content, str) and row.get("response_sha256") == sha256(content.encode())
                    and strict_json_loads(content) == row.get("parsed") == {"task_id": case["case_id"], "decision": case["expected_decision"]}
                    and isinstance(captures, list) and len(captures) == 1
                    and type(captures[0].get("status_code")) is int and captures[0]["status_code"] == 200
                    and captures[0].get("payload_sha256") == case["payload_sha256"]
                    and captures[0].get("wire_response_sha256") == row.get("wire_response_sha256")
                    and isinstance(row.get("wire_response_sha256"), str) and len(row["wire_response_sha256"]) == 64
                    and type(row.get("latency_seconds")) in {int, float} and math.isfinite(row["latency_seconds"])
                    and row["latency_seconds"] >= 0)
    except (ValueError, TypeError, AttributeError):
        return False


def run_benchmark(cases, settings, output, *, request_runner=None, pool_factory=None, clock=None, before_batch=None):
    if cases != build_request_manifest(settings):
        raise ValueError("only the exact three frozen legal cases are allowed")
    request_runner, pool_factory, clock = request_runner or execute_request, pool_factory or _pool, clock or time.perf_counter
    rows, batches, phase_errors = [], [], []
    schedule = [("warmup", 0, "serial", cases[:2]), ("measured", 1, "serial", cases),
                ("measured", 1, "parallel", cases), ("measured", 2, "parallel", cases), ("measured", 2, "serial", cases)]
    for phase, number, mode, selected in schedule:
        if before_batch is not None:
            try:
                before_batch()
            except Exception as exc:
                phase_errors.append({"phase": phase, "round": number, "mode": mode, "error_type": type(exc).__name__})
                break
        specs = [{"request_id": f"{phase}-{number}-{mode}-{c['case_id']}", "phase": phase, "round": number,
                  "mode": mode, "case": copy.deepcopy(c), "settings": settings} for c in selected]
        for spec in specs:
            runner.append_jsonl(Path(output) / "attempts.jsonl", spec)
        started, batch_rows = clock(), []
        with pool_factory(max_workers=2 if mode == "parallel" else 1) as pool:
            futures = {pool.submit(request_runner, spec): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    row = future.result()
                    if (not isinstance(row, dict) or any(row.get(k) != spec[k] for k in ("request_id", "phase", "round", "mode"))
                            or row.get("case_id") != spec["case"]["case_id"]
                            or any(row.get(k) != spec["case"][k] for k in ("prompt_sha256", "payload_sha256"))):
                        raise ValueError("worker result is not bound to its exact request")
                    usage = _raw_usage({"usage": row.get("raw_usage")})
                    if any(type(row.get(k)) is not type(v) or row.get(k) != v for k, v in usage.items()):
                        row.update(_raw_usage(None))
                    try:
                        row["semantic_valid"] = bool(row.get("semantic_valid") is True
                            and strict_json_loads(row.get("response_text")) == row.get("parsed")
                            == {"task_id": spec["case"]["case_id"], "decision": spec["case"]["expected_decision"]})
                    except (ValueError, TypeError):
                        row["semantic_valid"] = False
                    row["passed"] = _record_passes(row, spec["case"])
                except Exception as exc:
                    row = {k: spec[k] for k in ("request_id", "phase", "round", "mode")}
                    row.update(case_id=spec["case"]["case_id"], status="error", passed=False, error_type=type(exc).__name__,
                               model_calls=None, **_raw_usage(None))
                runner.append_jsonl(Path(output) / "requests.jsonl", row)
                batch_rows.append(row)
        wall = clock() - started
        passed = sum(r["passed"] for r in batch_rows)
        batch = {"phase": phase, "round": number, "mode": mode, "wall_seconds": wall,
                 "request_count": len(batch_rows), "passed_count": passed, "passed": passed == len(selected) and wall > 0,
                 "request_ids": [r["request_id"] for r in batch_rows], "total_tokens": _cost(batch_rows),
                 "successful_requests_per_second": passed / wall if wall > 0 else None,
                 "completion_tokens_per_second": sum(r["completion_tokens"] for r in batch_rows) / wall
                    if wall > 0 and all(r["usage_complete"] for r in batch_rows) else None}
        runner.append_jsonl(Path(output) / "batches.jsonl", batch)
        batches.append(batch)
        rows.extend(batch_rows)
        if phase == "warmup" and not batch["passed"]:
            break
    rounds = []
    for number in (1, 2):
        pair = {b["mode"]: b for b in batches if b["phase"] == "measured" and b["round"] == number}
        if len(pair) == 2:
            seq, par = pair["serial"]["wall_seconds"], pair["parallel"]["wall_seconds"]
            rounds.append({"round": number, "comparable": all(b["passed"] for b in pair.values()),
                           "raw_wall_ratio": seq / par if seq > 0 and par > 0 else None})
    measured = [r for r in rows if r["phase"] == "measured"]
    warmup = [r for r in rows if r["phase"] == "warmup"]
    comparisons = []
    for case in cases:
        selected = [r for r in measured if r.get("case_id") == case["case_id"]]
        comparisons.append({"case_id": case["case_id"], "measured_responses": len(selected),
                            "response_hashes": [r.get("response_sha256") for r in selected],
                            "text_identical": len(selected) == 4 and all(r.get("response_sha256") for r in selected)
                                and len({r["response_sha256"] for r in selected}) == 1,
                            "semantics_consistent": len(selected) == 4 and all(r.get("semantic_valid") is True
                                and r.get("parsed") == {"task_id": case["case_id"], "decision": case["expected_decision"]} for r in selected)})
    passed = len(measured) == 12 and len(warmup) == 2 and all(b["passed"] for b in batches)
    return {"passed": passed, "valid_speed_claim": passed and len(rounds) == 2 and all(
                r["comparable"] and r["raw_wall_ratio"] is not None and r["raw_wall_ratio"] > 1 for r in rounds),
            "speed_claim_scope": "serial_vs_width2_on_one_observed_service_config_only",
            "rows": rows, "batches": batches, "rounds": rounds, "phase_errors": phase_errors,
            "response_comparisons": comparisons, "cost_total_tokens": _cost(rows),
            "measured_total_tokens": _cost(measured), "warmup_total_tokens": _cost(warmup),
            "model_calls": sum(r["model_calls"] for r in rows) if all(type(r.get("model_calls")) is int for r in rows) else None}


def _reports(output, summary):
    runner._write_json(output / "summary.json", summary)
    fields = ("request_id", "phase", "round", "mode", "case_id", "status", "passed", "semantic_valid",
              "latency_seconds", "model_calls", "usage_complete", "prompt_tokens", "completion_tokens", "total_tokens",
              "prompt_sha256", "payload_sha256", "response_sha256", "error_type")
    with (output / "requests.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in summary.get("rows", []))
    lines = ["# 单服务推理配置有界探针", "", f"- 通过：{summary['passed']}",
             f"- 有效双路加速结论：{summary['valid_speed_claim']}",
             "- 仅比较当前同一配置的串行与双路；不比较不同配置，不启动或重启服务。",
             "- 短 JSON 判定微基准，不代表完整任务吞吐。",
             "- 最多十二个测量请求和两个预热请求；预热成本计入总量，不计入测量吞吐。",
             "- 批次时间包含工作进程启动与收尾；不是并发请求耗时之和。",
             "- 权重未重新逐字节哈希；记录既有清单哈希和当前文件大小、mtime、inode。",
             "- 图执行与 custom all-reduce 的实际启用、硬件兼容和最终 clean 验收由主任务核验。",
             "- 服务启动和编译耗时未知；不从请求延迟推断。文本哈希不同不等于语义判定不同。",
             "- 单元测试仅为准备工作；无自动正式实验。原始内容仅保存在服务器本目录。", ""]
    for row in summary.get("rounds", []):
        lines.append(f"- 第 {row['round']} 轮：可比较={row['comparable']}，原始耗时比（仅诊断）={row['raw_wall_ratio']}。")
    (output / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args, *, workflow_lock=None, platform_name=None, capture=None, preflight=None, benchmark=None):
    if (platform_name or sys.platform) != "linux":
        raise ValueError("real probe is Linux-only")
    with (workflow_lock or runner.locked_workflow)():
        output = Path(args.output_dir)
        if output.exists() or output.is_symlink():
            raise FileExistsError("a new output directory is required")
        settings = checked_settings()
        capture = capture or capture_service
        service = capture(args.service_pid, settings, args.model_manifest)
        cases = build_request_manifest(settings)
        sources = set(ROOT.glob("*.py")) | set((ROOT / "src").rglob("*.py")) | {Path(__file__).resolve()}
        hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()) for p in sorted(sources)}
        config = {"schema": 1, "service": service, "settings": settings, "requests": cases, "source_hashes": hashes,
                  "max_measured_calls": 12, "max_warmup_calls": 2, "max_workers": 2, "attempts_per_request": 1,
                  "python": sys.version.split()[0], "model": MODEL, "provider": PROVIDER}
        config_digest = object_hash(config)
        invariants = {"flags": {k: v for k, v in service["flags"].items()
                                 if k not in {"enforce-eager", "disable-custom-all-reduce", "compilation-config"}},
                      "artifact_hashes": service["artifact_hashes"], "settings": settings,
                      "requests": [c["payload_sha256"] for c in cases]}
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        runner._write_json(output / "manifest.json", {**config, "config_sha256": config_digest,
                                                     "invariants_sha256": object_hash(invariants)}, exclusive=True)
        runner.freeze_source_snapshot(output, hashes, root=ROOT, resume=False)
        summary = {"passed": False, "valid_speed_claim": False, "status": "error", "config_sha256": config_digest,
                   "invariants_sha256": object_hash(invariants), "formal_started": False, "candidate_accepted": False}
        def check_service():
            if capture(args.service_pid, settings, args.model_manifest) != service or checked_settings() != settings:
                raise ValueError("service or inference settings changed before batch")
        try:
            with runner.local_inference_transport():
                _, info = (preflight or preflight_client)(settings)
                runner._write_json(output / "preflight.json", info, exclusive=True)
                summary.update(benchmark(cases, settings, output) if benchmark is not None else
                               run_benchmark(cases, settings, output, before_batch=check_service))
            summary["status"] = "completed"
        except Exception as exc:
            summary.update(passed=False, valid_speed_claim=False, error_type=type(exc).__name__)
        try:
            summary["service_unchanged"] = capture(args.service_pid, settings, args.model_manifest) == service
            summary["source_unchanged"] = all(sha256((ROOT / name).read_bytes()) == value for name, value in hashes.items())
        except Exception as exc:
            summary.update(service_unchanged=False, source_unchanged=False, final_check_error=type(exc).__name__)
        if not summary["service_unchanged"] or not summary["source_unchanged"]:
            summary.update(passed=False, valid_speed_claim=False, status="provenance_changed")
        _reports(output, summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-pid", required=True, type=int, help="PID owning the existing API listening socket")
    parser.add_argument("--model-manifest", required=True, type=Path, help="server-local verified download manifest")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args)
        print(json.dumps({k: result[k] for k in ("status", "passed", "valid_speed_claim")}))
        return 0 if result["passed"] else 1
    except Exception as exc:
        print(f"Probe refused/stopped ({type(exc).__name__}); busy locks are never bypassed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
