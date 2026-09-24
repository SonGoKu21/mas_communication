"""Local fake-HTTP unit tests, not evidence of real four-cart isolation."""
import copy
import hashlib
import importlib
import json
import stat
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from mas_faults.shopping_action_protocol import MultiStateShoppingExecutor

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:17770"
TASKS = [
    {"task_id": f"{product}-{variant}", "product_title": product.title(),
     "product_url": BASE + "/" + product, "initial_quantity": initial, "quantity": target,
     "frozen_extra": {"variant": variant}}
    for product in ("tea", "rice", "soap")
    for variant, initial, target in (("up", 1, 3), ("down", 4, 2))
]


def test_four_cart_entrypoint_requires_manifest_and_new_output():
    assert (ROOT / "scripts/probe_multimechanism_four_carts.py").is_file(), "four-cart probe entrypoint missing"
    module = importlib.import_module("scripts.probe_multimechanism_four_carts")
    with pytest.raises(SystemExit):
        module.parse_args([])


@pytest.fixture
def probe():
    path = ROOT / "scripts/probe_multimechanism_four_carts.py"
    assert path.is_file(), "four-cart isolation probe is not implemented"
    return importlib.import_module("scripts.probe_multimechanism_four_carts")


@pytest.fixture
def args(tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"tasks": TASKS}))
    return SimpleNamespace(task_manifest=manifest, output_dir=tmp_path / "probe", base_url=BASE)


class UnitSession:
    def __init__(self, identity, shared=None):
        self.identity, self.shared = identity, shared
        self.quantity, self.product, self.closed = 0, "", False

    def response(self, url, data=None, text=None):
        body = text if text is not None else json.dumps(data)
        return SimpleNamespace(status_code=200, url=url, text=body, content=body.encode(),
                               json=lambda: data, raise_for_status=lambda: None)

    def item(self):
        quantity = self.shared.get(self.product, self.quantity) if self.shared is not None else self.quantity
        return {"item_id": 17, "sku": self.product.upper(), "name": self.product.title(), "qty": quantity}

    def get(self, url, **kwargs):
        if "/guest-carts/" in url:
            return self.response(url, [self.item()])
        self.product = urlsplit(url).path.rsplit("/", 1)[-1]
        return self.response(url, text=f'<input name="product" value="{self.product}">'
                             f'<div data-product-sku="{self.product.upper()}">')

    def post(self, url, **kwargs):
        if url.endswith("/guest-carts"):
            return self.response(url, str(self.identity))
        self.quantity += kwargs["json"]["cartItem"]["qty"]
        if self.shared is not None and self.quantity != 1 and self.quantity != 4:
            self.shared[self.product] = self.quantity
        return self.response(url, self.item())

    def put(self, url, **kwargs):
        self.quantity = kwargs["json"]["cartItem"]["qty"]
        if self.shared is not None:
            self.shared[self.product] = self.quantity
        return self.response(url, self.item())

    def close(self):
        self.closed = True


def factory(*, damage=None, concurrent=False):
    instances, sessions, trace = [], [], []
    mutex = Lock()
    shared = {} if damage == "cross_talk" else None
    barriers = {stage: Barrier(4) for stage in ("initialize", "update", "recheck")} if concurrent else {}

    class UnitExecutor(MultiStateShoppingExecutor):
        def __init__(self, identity):
            super().__init__(BASE)
            self.session.close()
            self.session = UnitSession(0 if damage == "same_cart" else identity, shared)
            if damage == "same_session" and sessions:
                self.session = sessions[0]
            elif damage == "close_error" and identity == 0:
                def failed_close():
                    self.session.closed = True
                    raise RuntimeError("private close failure")
                self.session.close = failed_close
            self.identity, self.reads = identity, 0
            sessions.append(self.session)

        def enter(self, stage, task):
            trace.append((stage + "_start", task["task_id"]))
            if stage in barriers:
                barriers[stage].wait(timeout=3)
            if damage == stage + "_error" and task["task_id"] == "tea-up":
                raise RuntimeError("private backend error")

        def corrupt(self, stage, task, value):
            if task["task_id"] == "tea-up":
                if damage == stage + "_evidence":
                    value["evidence"] = ""
                if damage in (stage + "_identity", "pair_identity"):
                    value["product_id"] = "other-product"
                    value["evidence"] = json.dumps({k: v for k, v in value.items() if k != "evidence"})
            return value

        def add_to_cart(self, task):
            self.enter("initialize", task)
            return self.corrupt("ack", task, super().add_to_cart(task))

        def add_quantity(self, task, quantity):
            self.enter("update", task)
            value = self.corrupt("update", task, super().add_quantity(task, quantity))
            trace.append(("update_done", task["task_id"]))
            return value

        def set_quantity(self, task, quantity):
            self.enter("update", task)
            value = self.corrupt("update", task, super().set_quantity(task, quantity))
            trace.append(("update_done", task["task_id"]))
            return value

        def reobserve_cart(self, task):
            stage = "initial" if self.reads == 0 else "recheck"
            self.reads += 1
            if stage == "recheck":
                self.enter(stage, task)
            value = self.corrupt(stage, task, super().reobserve_cart(task))
            if damage == "receipt" and task["task_id"] == "tea-up":
                self.http_receipts[0]["response_sha256"] = "invalid"
            trace.append(("initialize_done" if stage == "initial" else "recheck_done", task["task_id"]))
            return value

    def create(base_url):
        assert base_url == BASE
        with mutex:
            if damage == "factory_error" and len(instances) == 3:
                raise RuntimeError("private factory error")
            executor = UnitExecutor(len(instances))
            instances.append(executor)
            return executor
    return create, instances, sessions, trace


def test_selects_first_two_exact_pairs_in_manifest_order(probe, args):
    interleaved = [TASKS[i] for i in (0, 2, 4, 1, 3, 5)]
    args.task_manifest.write_text(json.dumps({"tasks": interleaved}))
    before = args.task_manifest.read_bytes()
    selected = probe.select_tasks(args.task_manifest, BASE)
    assert selected == TASKS[:4]
    selected[0]["frozen_extra"]["variant"] = "changed copy"
    assert args.task_manifest.read_bytes() == before


@pytest.mark.parametrize("damage", ["duplicate_id", "one_pair", "three_variants", "same_quantity",
                                    "title", "foreign_origin", "nonchanging_quantity"])
def test_invalid_frozen_pair_selection_refused(probe, args, damage):
    tasks = copy.deepcopy(TASKS)
    if damage == "duplicate_id":
        tasks[1]["task_id"] = tasks[0]["task_id"]
    elif damage == "one_pair":
        tasks = tasks[:2]
    elif damage == "three_variants":
        tasks.append({**tasks[0], "task_id": "tea-extra", "quantity": 7})
    elif damage == "same_quantity":
        tasks[1]["quantity"] = tasks[0]["quantity"]
    elif damage == "title":
        tasks[1]["product_title"] = "Other"
    elif damage == "foreign_origin":
        for task in tasks[:2]:
            task["product_url"] = "https://foreign.invalid/tea"
    else:
        tasks[0]["quantity"] = tasks[0]["initial_quantity"]
    args.task_manifest.write_text(json.dumps(tasks))
    with pytest.raises(ValueError):
        probe.select_tasks(args.task_manifest, BASE)


def test_four_cart_barriers_receipts_and_sessions(probe):
    create, instances, sessions, trace = factory(concurrent=True)
    tasks = copy.deepcopy(TASKS[:4])
    result = probe.http_isolation_gate(tasks, BASE, executor_factory=create)
    assert tasks == TASKS[:4]
    assert result["passed"] and result["max_workers"] == 4 and result["model_calls"] == 0
    assert len(instances) == len(sessions) == len({id(s) for s in sessions}) == 4
    assert all(s.closed for s in sessions)
    stages = [stage for stage, _ in trace]
    assert max(i for i, s in enumerate(stages) if s == "initialize_done") < stages.index("update_start")
    assert max(i for i, s in enumerate(stages) if s == "update_done") < stages.index("recheck_start")
    assert [flow["recheck"]["observed_quantity"] for flow in result["flows"]] == [3, 2, 3, 2]
    assert len({f["cart_id_sha256"] for f in result["flows"]}) == 4
    assert len({f["session_object_id"] for f in result["flows"]}) == 4
    assert all(f["http_receipts_verified"] for f in result["flows"])


@pytest.mark.parametrize("damage", ["same_cart", "same_session", "cross_talk", "receipt", "pair_identity",
                                    "ack_evidence", "initial_evidence", "update_evidence", "recheck_evidence",
                                    "ack_identity", "update_identity", "recheck_identity"])
def test_false_isolation_and_invalid_evidence_never_pass(probe, damage):
    create, _, sessions, _ = factory(damage=damage)
    result = probe.http_isolation_gate(TASKS[:4], BASE, executor_factory=create)
    assert not result["passed"]
    assert all(s.closed for s in sessions)
    assert result["errors"] or any(f["errors"] for f in result["flows"])


@pytest.mark.parametrize("damage", ["factory_error", "initialize_error", "update_error", "recheck_error", "close_error"])
def test_errors_retained_and_all_created_sessions_closed(probe, damage):
    create, _, sessions, trace = factory(damage=damage)
    result = probe.http_isolation_gate(TASKS[:4], BASE, executor_factory=create)
    assert not result["passed"] and all(s.closed for s in sessions)
    assert any(e["error_type"] == "RuntimeError" for f in result["flows"] for e in f["errors"])
    assert "private" not in json.dumps(result)
    if damage in ("factory_error", "initialize_error"):
        assert not any(stage == "update_start" for stage, _ in trace)


def test_busy_workflow_precedes_output_and_executor(probe, args):
    lock = lambda: probe.runner.locked_workflow(args.output_dir.parent / "workflow.lock")
    def forbidden(*a):
        pytest.fail("busy workflow reached HTTP executor")
    with lock(), pytest.raises(RuntimeError):
        probe.run_probe(args, workflow_lock=lock, executor_factory=forbidden)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("kind", ["existing", "symlink", "dangling_symlink"])
def test_new_output_only(probe, args, kind):
    target = args.output_dir.parent / "target"
    if kind != "dangling_symlink":
        target.mkdir()
    if kind == "existing":
        args.output_dir.mkdir()
    else:
        args.output_dir.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        probe.run_probe(args, workflow_lock=nullcontext)


def test_private_output_no_model_work_and_frozen_sources_untouched(probe, args, monkeypatch):
    from scripts import probe_multimechanism_parallel as shared_probe
    def forbidden(*a, **kw):
        pytest.fail("four-cart isolation must never reach model setup or preflight")
    for name in ("get_llm_client", "preflight_client", "inference_settings"):
        monkeypatch.setattr(probe.runner, name, forbidden)
    for name in ("execute_trial", "run_batches", "_real_client"):
        monkeypatch.setattr(shared_probe, name, forbidden)
    frozen = [ROOT / "scripts/probe_multimechanism_parallel.py", ROOT / "run_shopping_multimechanism.py"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    create, _, sessions, _ = factory()
    @contextmanager
    def idle():
        yield
        assert all(s.closed for s in sessions)
        assert (args.output_dir / "summary.json").is_file()
    report = probe.run_probe(args, workflow_lock=idle, executor_factory=create)
    assert report["passed"] and report["model_calls"] == 0 and not report["model_performance_verified"]
    assert stat.S_IMODE(args.output_dir.stat().st_mode) == 0o700
    files = list(args.output_dir.iterdir())
    assert {p.name for p in files} == {"four_cart_http_receipts.json", "summary.json"}
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in files)
    detail = json.loads((args.output_dir / "four_cart_http_receipts.json").read_text())
    assert detail["tasks"] == TASKS[:4] and len(detail["flows"]) == 4
    assert detail["task_manifest_sha256"] == hashlib.sha256(args.task_manifest.read_bytes()).hexdigest()
    compact = json.loads((args.output_dir / "summary.json").read_text())
    assert compact == report and "flows" not in compact and "http_receipts" not in compact
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen} == before


def test_failure_outputs_remain_private_and_no_success_claim(probe, args):
    create, _, _, _ = factory(damage="update_error")
    summary = probe.run_probe(args, workflow_lock=nullcontext, executor_factory=create)
    assert not summary["passed"] and summary["status"] == "failed"
    detail = json.loads((args.output_dir / "four_cart_http_receipts.json").read_text())
    assert any(f["errors"] for f in detail["flows"])


def test_cli_accepts_only_manifest_output_and_base(probe, args, monkeypatch, capsys):
    parsed = probe.parse_args(["--task-manifest", str(args.task_manifest), "--output-dir", str(args.output_dir),
                               "--base-url", BASE])
    assert vars(parsed) == vars(args)
    create, _, _, _ = factory()
    monkeypatch.setattr(probe, "make_executor", create)
    monkeypatch.setattr(probe.runner, "locked_workflow", lambda: nullcontext())
    assert probe.main(["--task-manifest", str(args.task_manifest), "--output-dir", str(args.output_dir),
                       "--base-url", BASE]) == 0
    output = capsys.readouterr().out
    assert "response_payload" not in output and json.loads(output)["passed"] is True
