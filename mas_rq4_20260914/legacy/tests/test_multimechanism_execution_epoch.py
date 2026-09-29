import hashlib
import importlib
import json
from pathlib import Path

import pytest


def module():
    name = "scripts.multimechanism_execution_epoch"
    assert importlib.util.find_spec(name) is not None, "execution epoch is not implemented"
    return importlib.import_module(name)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path):
    from mas_faults.multimechanism_matrix import config_digest
    output, source = tmp_path / "result", tmp_path / "code"
    output.mkdir()
    source.mkdir()
    config = {"jobs": [{"job_key": "j", "pair_key": "pair"}],
              "parallel_execution": {"logical_lanes": 2, "max_inflight_attempts": 2}}
    write_json(output / "matrix_manifest.json", {"config": config, "config_digest": config_digest(config)})
    lane = int(hashlib.sha256(b"pair").hexdigest(), 16) % 2
    start = {"attempt_id": "a", "job_key": "j", "pair_key": "pair", "attempt": 1,
             "config_digest": config_digest(config), "lane": lane}
    write_json(output / "run_attempts.jsonl", start)
    write_json(output / "main_runs.jsonl", {**start, "run_id": "r"})
    for name in ("multimechanism_execution_epoch", "run_multimechanism_parallel4", "probe_multimechanism_four_carts"):
        p = source / "scripts" / (name + ".py")
        p.parent.mkdir(exist_ok=True)
        p.write_text("# frozen driver\n")
    report = tmp_path / "isolation.json"
    write_json(report, {"passed": True, "gate_type": "four_cart_http_isolation", "max_workers": 4,
                       "model_calls": 0, "flows": [{"passed": True, "http_receipts_verified": True,
                       "cart_id_sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(4)]})
    return output, config, source, report, start


def test_freeze_and_resume_preserve_old_bytes_and_bind_driver(tmp_path):
    m = module()
    out, config, source, report, start = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    epoch = m.prepare_epoch(out, config, source, report)
    assert all((out / name).read_bytes() == data for name, data in before.items())
    assert epoch["config"]["parallel_execution"]["max_inflight_attempts"] == 4
    assert m.prepare_epoch(out, config, source, None) == epoch
    audit = m.verify_epoch(out, config, source)
    assert audit["legacy_attempts"] == 1 and audit["epoch_attempts"] == 0
    assert audit["unfinished_attempts"] == 0
    assert audit["epoch_digest"] == epoch["epoch_digest"]


@pytest.mark.parametrize("target", ["prefix", "driver", "manifest", "snapshot", "isolation"])
def test_tampering_is_rejected(tmp_path, target):
    m = module()
    out, config, source, report, _ = fixture(tmp_path)
    m.prepare_epoch(out, config, source, report)
    ep = out / "execution_epochs" / "parallel4_v1"
    paths = {"prefix": out / "main_runs.jsonl", "driver": source / "scripts/run_multimechanism_parallel4.py",
             "manifest": out / "matrix_manifest.json", "snapshot": ep / "source/scripts/run_multimechanism_parallel4.py",
             "isolation": ep / "isolation_report.json"}
    paths[target].write_bytes(paths[target].read_bytes() + b" ")
    with pytest.raises(ValueError):
        m.verify_epoch(out, config, source)


def test_transition_refuses_inflight_and_failed_isolation(tmp_path):
    m = module()
    out, config, source, report, _ = fixture(tmp_path)
    (out / "main_runs.jsonl").unlink()
    with pytest.raises(ValueError, match="unfinished"):
        m.prepare_epoch(out, config, source, report)
    (out / "run_attempts.jsonl").unlink()
    bad = json.loads(report.read_text())
    bad["flows"][1]["cart_id_sha256"] = bad["flows"][0]["cart_id_sha256"]
    write_json(report, bad)
    with pytest.raises(ValueError, match="isolation"):
        m.prepare_epoch(out, config, source, report)


@pytest.mark.parametrize("mutation", ["missing_epoch", "wrong_epoch", "wrong_lane", "terminal_binding"])
def test_new_attempt_epoch_and_lane_must_match(tmp_path, mutation):
    m = module()
    out, config, source, report, start = fixture(tmp_path)
    epoch = m.prepare_epoch(out, config, source, report)
    new = {**start, "attempt_id": "b", "execution_epoch": epoch["epoch_digest"],
           "lane": int(hashlib.sha256(b"pair").hexdigest(), 16) % 4}
    if mutation == "missing_epoch":
        del new["execution_epoch"]
    elif mutation == "wrong_epoch":
        new["execution_epoch"] = "wrong"
    elif mutation == "wrong_lane":
        new["lane"] = (new["lane"] + 1) % 4
    with (out / "run_attempts.jsonl").open("a") as h:
        h.write(json.dumps(new) + "\n")
    if mutation == "terminal_binding":
        with (out / "run_errors.jsonl").open("a") as h:
            h.write(json.dumps({**new, "execution_epoch": "wrong"}) + "\n")
    with pytest.raises(ValueError):
        m.verify_epoch(out, config, source)


def test_reports_mixed_regimes_without_rewriting_records(tmp_path):
    m = module()
    out, config, source, report, start = fixture(tmp_path)
    epoch = m.prepare_epoch(out, config, source, report)
    new = {**start, "attempt_id": "b", "execution_epoch": epoch["epoch_digest"],
           "lane": int(hashlib.sha256(b"pair").hexdigest(), 16) % 4}
    with (out / "run_attempts.jsonl").open("a") as h:
        h.write(json.dumps(new) + "\n")
    assert m.verify_epoch(out, config, source)["unfinished_attempts"] == 1
    with (out / "run_errors.jsonl").open("a") as h:
        h.write(json.dumps(new) + "\n")
    result = m.verify_epoch(out, config, source)
    assert result["legacy_attempts"] == result["epoch_attempts"] == 1
    assert result["unfinished_attempts"] == 0


def test_torn_new_tail_recovery_preserves_old_prefix_and_unknown_attempt(tmp_path):
    m = module()
    assert hasattr(m, "recover_epoch_journals"), "protected epoch recovery is not implemented"
    out, config, source, report, start = fixture(tmp_path)
    epoch = m.prepare_epoch(out, config, source, report)
    before = (out / "main_runs.jsonl").read_bytes()
    new = {**start, "attempt_id": "b", "execution_epoch": epoch["epoch_digest"],
           "lane": int(hashlib.sha256(b"pair").hexdigest(), 16) % 4}
    with (out / "run_attempts.jsonl").open("a") as h:
        h.write(json.dumps(new) + "\n")
    with (out / "main_runs.jsonl").open("ab") as h:
        h.write(b'{"attempt_id": "b", "cut')
    m.recover_epoch_journals(out, config, source)
    assert (out / "main_runs.jsonl").read_bytes() == before
    assert m.verify_epoch(out, config, source)["unfinished_attempts"] == 1


def test_recovery_cannot_repair_damaged_legacy_prefix(tmp_path):
    m = module()
    assert hasattr(m, "recover_epoch_journals"), "protected epoch recovery is not implemented"
    out, config, source, report, _ = fixture(tmp_path)
    m.prepare_epoch(out, config, source, report)
    path = out / "main_runs.jsonl"
    damaged = path.read_bytes()[:-5]
    path.write_bytes(damaged)
    with pytest.raises(ValueError):
        m.recover_epoch_journals(out, config, source)
    assert path.read_bytes() == damaged
