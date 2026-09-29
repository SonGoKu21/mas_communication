"""Frozen, paired design for real multi-state mitigation experiments."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from urllib.parse import urlsplit

VERSION = "shopping-multimechanism-v1"
ARMS = ("baseline", "always_recheck", "guarded_recheck", "dependency",
        "independent", "action_protocol", "combined")
TOPOLOGIES = ("sequential", "flat", "hierarchical")
CELLS = {
    "clean": None,
    "request_non_delivery": "action_request",
    "acknowledgement_loss": "action_ack",
    "duplicate_action_delivery": "action_request",
    "valid_partial": "evidence_handoff",
    "same_session_reordering": "observation_handoff",
    "cross_task_replay": "evidence_handoff",
    "stale_judgment_replay": "judgment_handoff",
    "conflicting_observation": "observation_handoff",
    "contract_consistent_identity_corruption": "evidence_handoff",
}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_tasks(tasks):
    if not tasks:
        raise ValueError("nonempty task manifest required")
    ids = set()
    for task in tasks:
        for field in ("task_id", "product_title", "product_url"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"nonempty {field} required")
        if task["task_id"] in ids:
            raise ValueError("duplicate task_id")
        ids.add(task["task_id"])
        url = urlsplit(task["product_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("task URL must be an HTTP(S) URL without credentials")
        for key in ("initial_quantity", "quantity"):
            if type(task.get(key)) is not int or task[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if task["initial_quantity"] == task["quantity"]:
            raise ValueError("multi-state task must change quantity")


def build_jobs(tasks, repetitions=3):
    validate_tasks(tasks)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive repetition count required")
    jobs = []
    for ci, (condition, boundary) in enumerate(CELLS.items()):
        for ti, task in enumerate(tasks):
            for ai, topology in enumerate(TOPOLOGIES):
                for repeat in range(1, repetitions + 1):
                    pair = json.dumps([task["task_id"], topology, condition, boundary, repeat],
                                      separators=(",", ":"))
                    offset = (ci + ti + ai + repeat - 1) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        jobs.append({"pair_key": pair, "job_key": pair + ":" + arm,
                                     "task_id": task["task_id"], "topology": topology,
                                     "condition": condition, "boundary": boundary,
                                     "repeat_index": repeat, "arm": arm})
    return jobs


def select_shard(jobs, shard_index, shard_count):
    if (type(shard_count) is not int or shard_count < 1 or type(shard_index) is not int
            or not 0 <= shard_index < shard_count):
        raise ValueError("invalid shard selection")
    # Keep all strategy arms of a pair on the same runner.
    return [j for j in jobs if int(hashlib.sha256(j["pair_key"].encode()).hexdigest(), 16)
            % shard_count == shard_index]


def pending_jobs(jobs, completed, errors, digest, max_attempts=2):
    if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
        raise ValueError("maximum attempts must be 1 or 2")
    known = {j["job_key"] for j in jobs}
    done = set()
    attempts = Counter()
    for row in [*completed, *errors]:
        if row.get("config_digest") != digest:
            raise ValueError("resume configuration does not match frozen experiment")
        if row.get("job_key") not in known:
            raise ValueError("unknown job in resume records")
        attempts[row["job_key"]] += 1
    for row in completed:
        if row["job_key"] in done:
            raise ValueError("duplicate completed job")
        done.add(row["job_key"])
    return [j for j in jobs if j["job_key"] not in done and attempts[j["job_key"]] < max_attempts]
