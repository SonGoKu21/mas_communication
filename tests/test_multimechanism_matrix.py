import copy
import pytest

from mas_faults import multimechanism_matrix as matrix


def tasks(n=10):
    return [{"task_id": f"t{i}", "product_title": f"Product {i}",
             "product_url": f"http://localhost:17770/p{i}.html",
             "initial_quantity": 1, "quantity": 2} for i in range(n)]


def test_formal_matrix_is_6300_balanced_unique_jobs():
    jobs = matrix.build_jobs(tasks(), repetitions=3)
    assert len(jobs) == 6300
    assert len({j["job_key"] for j in jobs}) == 6300
    assert sum(j["condition"] == "clean" for j in jobs) == 630
    assert {j["arm"] for j in jobs} == set(matrix.ARMS)
    pairs = {}
    for job in jobs:
        pairs.setdefault(job["pair_key"], []).append(job)
    assert len(pairs) == 900
    assert all(len(p) == 7 for p in pairs.values())
    assert len({p[0]["arm"] for p in pairs.values()}) == 7


def test_cells_are_semantic_boundaries_not_every_step():
    jobs = matrix.build_jobs(tasks(1), repetitions=1)
    assert {j["boundary"] for j in jobs if j["condition"] == "acknowledgement_loss"} == {"action_ack"}
    assert {j["boundary"] for j in jobs if j["condition"] == "stale_judgment_replay"} == {"judgment_handoff"}
    assert {j["boundary"] for j in jobs if j["condition"] == "clean"} == {None}


@pytest.mark.parametrize("bad", [[], tasks(1) * 2,
    [{**tasks(1)[0], "initial_quantity": True}],
    [{**tasks(1)[0], "quantity": 0}],
    [{**tasks(1)[0], "quantity": 1}]])
def test_bad_task_manifests_fail_before_running(bad):
    with pytest.raises(ValueError):
        matrix.build_jobs(bad, repetitions=3)


def test_resume_uses_exact_config_and_never_duplicates_finished_jobs():
    jobs = matrix.build_jobs(tasks(1), repetitions=1)
    config = {"model": "Qwen/Qwen3.8-27B", "repetitions": 1}
    digest = matrix.config_digest(config)
    rows = [{"job_key": jobs[0]["job_key"], "config_digest": digest}]
    remaining = matrix.pending_jobs(jobs, rows, [], digest)
    assert len(remaining) == 209
    assert jobs[0] not in remaining
    with pytest.raises(ValueError, match="configuration"):
        matrix.pending_jobs(jobs, rows, [], "changed")
    with pytest.raises(ValueError, match="duplicate"):
        matrix.pending_jobs(jobs, rows * 2, [], digest)
    errors = [{"job_key": jobs[1]["job_key"], "config_digest": digest}] * 2
    assert len(matrix.pending_jobs(jobs, rows, errors, digest)) == 208


def test_shards_partition_complete_matrix():
    jobs = matrix.build_jobs(tasks(), repetitions=3)
    shards = [matrix.select_shard(jobs, i, 3) for i in range(3)]
    assert sum(map(len, shards)) == len(jobs)
    assert len({j["job_key"] for s in shards for j in s}) == len(jobs)
    for s in shards:
        counts = {}
        for j in s:
            counts[j["pair_key"]] = counts.get(j["pair_key"], 0) + 1
        assert set(counts.values()) == {7}


def test_frozen_config_hash_is_order_independent_and_sensitive():
    config = {"model": "q", "tasks": tasks(1)}
    assert matrix.config_digest(config) == matrix.config_digest(dict(reversed(list(config.items()))))
    changed = copy.deepcopy(config)
    changed["tasks"][0]["quantity"] = 3
    assert matrix.config_digest(config) != matrix.config_digest(changed)
