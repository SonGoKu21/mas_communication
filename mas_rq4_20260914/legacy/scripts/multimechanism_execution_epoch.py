"""Append-only operational provenance for the two-to-four-lane transition.

The original matrix, Python snapshot and journal prefixes remain unchanged.
An epoch overrides scheduling only; it is not another scientific matrix.
Call preparation under both workflow and output locks with workers drained.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import run_shopping_multimechanism as runner

EPOCH = "parallel4_v1"
JOURNALS = ("run_attempts.jsonl", "main_runs.jsonl", "run_errors.jsonl")
DRIVERS = tuple("scripts/" + name + ".py" for name in (
    "run_multimechanism_parallel4", "multimechanism_execution_epoch", "probe_multimechanism_four_carts"))
POLICY = {
    "logical_lanes": 4, "max_inflight_attempts": 4, "start_method": "spawn",
    "lane_assignment": "sha256(pair_key_utf8)_integer_mod_4",
    "lane_order": "frozen_jobs_order_retry_before_next_job",
    "terminal_order": "durable_start_order", "clean_barrier": True,
    "timing_comparability": "stratify_by_execution_epoch_do_not_pool_two_and_four_lane_latency",
    "scientific_configuration": "unchanged_base_matrix_manifest",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path, *, optional=False):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("epoch input symlinks are forbidden")
    if optional and not path.exists():
        return b""
    if not path.is_file():
        raise ValueError("missing regular epoch input file")
    return path.read_bytes()


def _json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("non-finite JSON value")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _rows(data):
    if data and not data.endswith(b"\n"):
        raise ValueError("unfinished journal line")
    rows = [_json(line) for line in data.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("journal rows must be objects")
    return rows


def _gate(data):
    report = _json(data)
    flows = report.get("flows", [])
    if (report.get("passed") is not True or report.get("gate_type") != "four_cart_http_isolation"
            or report.get("max_workers") != 4 or report.get("model_calls") != 0
            or not isinstance(flows, list) or len(flows) != 4
            or any(f.get("passed") is not True or f.get("http_receipts_verified") is not True for f in flows)):
        raise ValueError("four-cart isolation gate did not pass")
    hashes = [f.get("cart_id_sha256") for f in flows]
    if (any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)
            or len(set(hashes)) != 4):
        raise ValueError("four-cart isolation identities are invalid")


def _base(output, config):
    data = _bytes(output / "matrix_manifest.json")
    value = _json(data)
    if value != {"config": config, "config_digest": runner.matrix.config_digest(config)}:
        raise ValueError("base matrix manifest changed")
    if config.get("parallel_execution", {}).get("logical_lanes") != 2:
        raise ValueError("epoch requires the original two-lane matrix")
    return data


def _journals(output, config, epoch_digest=None, legacy_count=None):
    values = {name: _rows(_bytes(output / name, optional=True)) for name in JOURNALS}
    starts = values["run_attempts.jsonl"]
    count = len(starts) if legacy_count is None else legacy_count
    jobs = {j["job_key"]: j for j in config["jobs"]}
    base_digest = runner.matrix.config_digest(config)
    indexed = {}
    for index, start in enumerate(starts):
        key, aid = start.get("job_key"), start.get("attempt_id")
        if key not in jobs or not aid or aid in indexed:
            raise ValueError("unknown job or duplicate attempt")
        if start.get("config_digest") != base_digest:
            raise ValueError("attempt matrix binding changed")
        new = index >= count
        if start.get("execution_epoch") != (epoch_digest if new else None):
            raise ValueError("attempt epoch binding changed")
        lane = int(sha(jobs[key]["pair_key"].encode()), 16) % (4 if new else 2)
        if type(start.get("lane")) is not int or start["lane"] != lane:
            raise ValueError("attempt lane binding changed")
        indexed[aid] = start
    ended = set()
    for name in JOURNALS[1:]:
        for row in values[name]:
            aid = row.get("attempt_id")
            if aid not in indexed or aid in ended:
                raise ValueError("unknown or duplicate terminal attempt")
            if any(row.get(k) != indexed[aid].get(k) for k in
                   ("job_key", "attempt", "config_digest", "lane", "execution_epoch")):
                raise ValueError("terminal epoch binding changed")
            ended.add(aid)
    return {"legacy_attempts": count, "epoch_attempts": len(starts) - count,
            "unfinished_attempts": len(starts) - len(ended),
            "completed_runs": len(values["main_runs.jsonl"]), "error_attempts": len(values["run_errors.jsonl"])}


def prepare_epoch(output, config, source_root, isolation_report_path=None):
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    directory = output / "execution_epochs" / EPOCH
    target = directory / "manifest.json"
    if target.exists():
        verify_epoch(output, config, source_root)
        return _json(_bytes(target))
    if directory.exists():
        raise ValueError("incomplete epoch freeze; inspect it before continuing")
    base = _base(output, config)
    if _journals(output, config)["unfinished_attempts"]:
        raise ValueError("unfinished attempts must drain before transition")
    if isolation_report_path is None:
        raise ValueError("four-cart isolation report is required")
    isolation = _bytes(isolation_report_path)
    _gate(isolation)
    sources = {name: _bytes(source_root / name) for name in DRIVERS}
    prefixes = {}
    for name in JOURNALS:
        data = _bytes(output / name, optional=True)
        prefixes[name] = {"bytes": len(data), "sha256": sha(data), "records": len(_rows(data))}
    frozen = {"schema": 1, "name": EPOCH, "base_config_digest": runner.matrix.config_digest(config),
              "base_manifest_sha256": sha(base), "parallel_execution": POLICY,
              "journal_prefixes": prefixes, "source_hashes": {n: sha(d) for n, d in sources.items()},
              "isolation_report_sha256": sha(isolation), "created_at_unix": time.time()}
    epoch = {"config": frozen, "epoch_digest": runner.matrix.config_digest(frozen)}
    directory.mkdir(parents=True)
    for name, data in {**{"source/" + n: d for n, d in sources.items()}, "isolation_report.json": isolation}.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            runner.os.fsync(handle.fileno())
        runner._sync_directory(path.parent)
    runner._write_json(target, epoch, exclusive=True)
    verify_epoch(output, config, source_root)
    return epoch


def _verify_frozen(output, config, source_root=None):
    output = Path(output).resolve()
    directory = output / "execution_epochs" / EPOCH
    epoch = _json(_bytes(directory / "manifest.json"))
    frozen, digest = epoch["config"], epoch["epoch_digest"]
    if (digest != runner.matrix.config_digest(frozen) or frozen.get("schema") != 1 or frozen.get("name") != EPOCH
            or frozen.get("parallel_execution") != POLICY
            or frozen.get("base_config_digest") != runner.matrix.config_digest(config)
            or frozen.get("base_manifest_sha256") != sha(_base(output, config))):
        raise ValueError("execution epoch configuration changed")
    if set(frozen.get("source_hashes", {})) != set(DRIVERS):
        raise ValueError("execution epoch source set changed")
    for name, expected in frozen["source_hashes"].items():
        if sha(_bytes(directory / "source" / name)) != expected:
            raise ValueError("execution epoch source snapshot changed")
        if source_root is not None and sha(_bytes(Path(source_root) / name)) != expected:
            raise ValueError("execution epoch driver changed")
    isolation = _bytes(directory / "isolation_report.json")
    if sha(isolation) != frozen["isolation_report_sha256"]:
        raise ValueError("execution epoch isolation report changed")
    _gate(isolation)
    if set(frozen.get("journal_prefixes", {})) != set(JOURNALS):
        raise ValueError("execution epoch journal prefix set changed")
    for name, prefix in frozen["journal_prefixes"].items():
        size = prefix.get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError("invalid journal prefix size")
        data = _bytes(output / name, optional=True)
        if len(data) < size or sha(data[:size]) != prefix["sha256"] or len(_rows(data[:size])) != prefix["records"]:
            raise ValueError("legacy journal prefix changed")
    return frozen, digest


def recover_epoch_journals(output, config, source_root):
    """Repair only an appended torn tail, after verifying the immutable prefix."""
    output = Path(output).resolve()
    _verify_frozen(output, config, source_root)
    # Validate all complete lines before any repair, including duplicate keys.
    for name in JOURNALS:
        data = _bytes(output / name, optional=True)
        complete = data if data.endswith(b"\n") else data[:data.rfind(b"\n") + 1]
        _rows(complete)
    runner.recover_torn_journals(output)
    return verify_epoch(output, config, source_root)


def verify_epoch(output, config, source_root=None):
    output = Path(output).resolve()
    frozen, digest = _verify_frozen(output, config, source_root)
    return {"valid": True, "epoch_digest": digest, "parallel_execution": POLICY,
            **_journals(output, config, digest, frozen["journal_prefixes"]["run_attempts.jsonl"]["records"])}
