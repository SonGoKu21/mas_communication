"""P1 admission-gated 432-job execution; only the parent writes journals.

Import and admission validation perform no model or backend requests.
Formal execution requires an explicitly eligible, hash-bound design.
"""
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import multiprocessing
import os
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'legacy'))
sys.path.insert(0, str(ROOT / 'legacy/src'))
import run_shopping_multimechanism as common
from scripts.run_multimechanism_parallel import _real_client, _executor, _bind_parent_lifetime
from p1_execution import STRATEGIES, execution_jobs
from mas_faults.deepseek_schedule import is_deepseek_offpeak


ARCHIVED_ORIGIN = 'http://127.0.0.1:17770'
BASE_URL = 'http://localhost:7770'
GATE_ADAPTERS = ('shopping_action_protocol.py', 'webarena_shopping_real.py', 'shopping_mitigation.py')
SCHEDULE_SEED = 20260927


def balanced_schedule(jobs, *, seed=SCHEDULE_SEED):
    """Build a candidate order once; admission and resume never reshuffle it."""
    if type(seed) is not int:
        raise ValueError('integer scheduling seed required')
    blocks = defaultdict(list)
    for job in jobs:
        blocks[job['pair_key']].append(job)
    if len(jobs) != 432 or len(blocks) != 108 or len({j['job_key'] for j in jobs}) != 432:
        raise ValueError('complete 108 four-strategy blocks required')
    phases = {'clean': [], 'fault': []}
    for key in sorted(blocks):
        block = blocks[key]
        if len(block) != 4 or {j['variant'] for j in block} != set(STRATEGIES):
            raise ValueError('each pair block requires the four distinct strategies')
        identity = ('task_id', 'topology', 'repeat_index', 'condition', 'product_cluster')
        if any(any(j[field] != block[0][field] for field in identity) for j in block):
            raise ValueError('pair block mixes task/topology/repeat/condition/product')
        phase = 'clean' if block[0]['condition'] == 'clean' else 'fault'
        phases[phase].append({j['variant']: j for j in block})
    if len(phases['clean']) != 36 or len(phases['fault']) != 72:
        raise ValueError('36 clean and 72 fault blocks required')
    result = []
    for phase in ('clean', 'fault'):
        ordered = phases[phase]
        random.Random(f'{seed}:{phase}').shuffle(ordered)
        for index, block in enumerate(ordered):
            offset = index % 4
            strategies = STRATEGIES[offset:] + STRATEGIES[:offset]
            result.extend(copy.deepcopy(block[strategy]) for strategy in strategies)
    return result


def mapped_product_url(value):
    """Only the declared origin changes; no redirect or product substitution."""
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ValueError('invalid product URL origin')
    parsed = urlsplit(value)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    if origin not in (ARCHIVED_ORIGIN, BASE_URL) or not parsed.path.startswith('/') or parsed.path == '/':
        raise ValueError('product URL origin must be archived17770 or canonical localhost7770')
    if parsed.query or parsed.fragment:
        raise ValueError('product URL query/fragment is outside the frozen task mapping')
    return BASE_URL + parsed.path


def validate_admission(design, http_gate):
    """Fail closed before settings, credentials, output, locks, workers or clients."""
    if design.get('execution_eligible') is not True:
        raise ValueError('design execution_eligible must be explicitly true')
    gate_path = Path(http_gate)
    started_path = gate_path.with_name('started.json')
    gate_bytes, started_bytes = gate_path.read_bytes(), started_path.read_bytes()
    hashes = dict(summary_sha256=hashlib.sha256(gate_bytes).hexdigest(),
                  started_sha256=hashlib.sha256(started_bytes).hexdigest())
    for field, expected in (('http_gate_sha256', hashes['summary_sha256']),
                            ('http_gate_started_sha256', hashes['started_sha256'])):
        if design.get(field) != expected:
            raise ValueError('HTTP gate hash mismatch or missing: ' + field)
    gate, started = json.loads(gate_bytes), json.loads(started_bytes)
    required = dict(status='passed', http_gate_only=True, model_calls=0, max_http_concurrency=4,
                    product_get_passed=6, product_get_total=6,
                    quantity_transition_passed=6, quantity_transition_total=6,
                    four_lane_isolation_status='passed', new_guest_carts=10,
                    all_ten_carts_distinct=True, all_responses_hashed=True, source_unchanged=True)
    if any(type(gate.get(k)) is not type(v) or gate.get(k) != v for k, v in required.items()):
        raise ValueError('HTTP gate failed, incomplete or outside the four-lane scope')
    mapping = {'from': ARCHIVED_ORIGIN, 'to': BASE_URL}
    if (gate.get('origin_mapping') != mapping or started.get('origin_mapping') != mapping
            or gate.get('denied_network_destinations') != [] or gate.get('retained_failures') != []):
        raise ValueError('HTTP gate origin or network scope mismatch')
    if (type(gate.get('http_attempts')) is not int or gate['http_attempts'] < 1
            or gate.get('http_receipt_count') != gate['http_attempts']
            or gate.get('http_status_counts') != {'200': gate['http_attempts']}):
        raise ValueError('HTTP gate receipts incomplete or unsuccessful')
    adapter_hashes = gate.get('source_hashes', {})
    if not isinstance(adapter_hashes, dict) or len(adapter_hashes) != 3:
        raise ValueError('HTTP gate adapter hash set incomplete')
    by_name = {Path(name).name: digest for name, digest in adapter_hashes.items()}
    if set(by_name) != set(GATE_ADAPTERS) or started.get('source_hashes') != adapter_hashes:
        raise ValueError('HTTP gate adapter hashes disagree')
    for name, digest in by_name.items():
        if hashlib.sha256((ROOT / 'legacy/src/mas_faults' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('HTTP gate adapter source changed: ' + name)
    normalized = copy.deepcopy(design)
    if normalized.get('base_url') not in (None, ARCHIVED_ORIGIN, BASE_URL):
        raise ValueError('design backend origin differs from HTTP gate')
    if normalized.get('model') not in (None, common.MODEL) or normalized.get('provider') not in (None, common.PROVIDER):
        raise ValueError('P1 requires the original deepseek-flash model/provider')
    normalized['base_url'] = BASE_URL
    common.matrix.validate_tasks(normalized['tasks'])
    for task in normalized['tasks']:
        task['product_url'] = mapped_product_url(task['product_url'])
        task['product_cluster'] = mapped_product_url(task['product_cluster'])
        if task['product_cluster'] != task['product_url']:
            raise ValueError('task product cluster differs from its frozen URL')
    admitted_tasks = []
    for pair in started.get('tasks', []):
        original, mapped = pair['original'], pair['mapped']
        expected = dict(original, product_url=mapped_product_url(original['product_url']),
                        product_cluster=mapped_product_url(original['product_cluster']))
        if mapped != expected:
            raise ValueError('HTTP gate task mapping changed path/title/quantity')
        admitted_tasks.append(mapped)
    if len(admitted_tasks) != 6 or normalized['tasks'] != admitted_tasks:
        raise ValueError('design tasks differ from the six HTTP-gated products/quantities')
    tasks = {t['task_id']: t for t in normalized['tasks']}
    for job in normalized['jobs']:
        if job['task_id'] not in tasks:
            raise ValueError('matrix task absent from frozen six products')
        job['product_cluster'] = mapped_product_url(job['product_cluster'])
        if job['product_cluster'] != tasks[job['task_id']]['product_cluster']:
            raise ValueError('job product cluster differs from frozen task')
    jobs = execution_jobs(normalized, common.matrix.CELLS)
    return normalized, jobs, dict(**hashes, origin_mapping=mapping,
        adapter_hashes=by_name, summary=gate, started=started)


def pending_after_clean_barrier(state, jobs):
    pending = state['pending']
    clean_pending = [j for j in pending if j['condition'] == 'clean']
    if clean_pending:
        return clean_pending
    clean_rows = [r for r in state['rows'] if r['condition'] == 'clean']
    expected = {j['job_key'] for j in jobs if j['condition'] == 'clean'}
    if pending and (len(clean_rows) != 144 or len(expected) != 144
                    or {r['job_key'] for r in clean_rows} != expected):
        raise RuntimeError('clean jobs exhausted or incomplete; no fault launch')
    return pending


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}


def attempt(spec):
    from p1_trial import run
    start, client, executor, before = spec['start'], None, None, None
    started = time.perf_counter()
    try:
        common.ensure_deepseek_offpeak(common.MODEL)
        with common.local_inference_transport():
            client = _real_client(spec['settings'])
            before = common.usage_snapshot(client)
            executor = _executor(spec['base_url'])
            source = start.get('cross_task_source')
            row = asyncio.run(run(spec['task'], start, executor, client,
                                  cross_task_evidence=source['envelope'] if source else None,
                                  ledger_path=Path(spec['output']) / start['ledger_path']))
            return 'row', dict(row, **common.completed_usage(client, before, row),
                               latency_ms=round(1000 * (time.perf_counter() - started), 3))
    except Exception as exc:
        usage = common.usage_since(client, before, complete=False) if before is not None else dict(
            known_total_tokens=0, total_tokens=None, usage_complete=False, request_sent=False)
        return 'error', dict(start, **usage, error_type=type(exc).__name__,
                             offpeak_blocked=common.is_peak_exception(exc),
                             partial_trial=common.partial_trial_context(exc))
    finally:
        if executor is not None:
            executor.session.close()


def summary(output, jobs):
    rows = common.read_jsonl(output / 'main_runs.jsonl')
    errors = common.read_jsonl(output / 'run_errors.jsonl')
    groups = defaultdict(Counter)
    for r in rows:
        key = (r['experiment'], r['variant'], r['topology'], r['condition'])
        groups[key].update(runs=1, success=int(r['final_task_success']),
                           actual_injections=len(r.get('fault_events', [])),
                           tokens=r.get('known_total_tokens', 0))
    result = dict(completed=len(rows), planned=len(jobs), errors=len(errors),
                  known_total_tokens=sum(r.get('known_total_tokens', 0) for r in rows + errors),
                  usage_unknown=sum(r.get('total_tokens') is None for r in rows + errors),
                  status='complete' if len(rows) == len(jobs) else 'in_progress',
                  timestamp_unix=time.time())
    common._write_json(output / 'summary.json', result)
    with (output / 'summary.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'variant', 'topology', 'condition', 'runs', 'success', 'actual_injections', 'known_tokens'])
        for key, values in sorted(groups.items()):
            writer.writerow([*key, values['runs'], values['success'], values['actual_injections'], values['tokens']])
    (output / 'summary.md').write_text('# RQ4 P1 执行进度\n\n' +
        f"完成 {len(rows)}/{len(jobs)}，错误尝试 {len(errors)}，已知token下限 {result['known_total_tokens']}。\n\n" +
        '此文件是运行进度；配对效应与任务聚类统计在完成后单独报告。\n')
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--http-gate', type=Path, required=True,
                        help='Passed canonical HTTP summary with sibling started.json, both hash-bound in design')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--reset-circuit-after-audit', help='Audited cause/resolution; never resets attempt budgets')
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError('positive max-jobs required')
    design_bytes = args.design.read_bytes()
    design, jobs, gate = validate_admission(json.loads(design_bytes), args.http_gate)
    settings = common.inference_settings()
    config = dict(tasks=design['tasks'], jobs=jobs, inference_settings=settings,
                  model=common.MODEL, provider=common.PROVIDER, model_version=common.MODEL_VERSION,
                  base_url=BASE_URL, source_hashes=source_hashes(), lanes=4,
                  budget=dict(http=32, model=16, seconds=1200, action_replays=0,
                              semantic_readbacks={'baseline': 0, 'check_only': 0,
                                                  'always_readback': 1, 'guarded_readback': 1}),
                  max_attempts_per_job=2, repetitions=3,
                  design_sha256=hashlib.sha256(design_bytes).hexdigest(), http_gate=gate)
    digest = common.matrix.config_digest(config)
    out = args.output
    if out.resolve().is_relative_to(ROOT):
        raise ValueError('result directory must be outside source tree')
    out.mkdir(parents=True, exist_ok=True)
    with common.locked_workflow(), common.locked_output(out):
        manifest = out / 'matrix_manifest.json'
        resume = manifest.exists()
        common.freeze_manifest(out, config, resume=resume)
        common.freeze_source_snapshot(out, config['source_hashes'], root=ROOT, resume=resume)
        common.recover_torn_journals(out)
        state = common.reconcile_attempts(out, jobs, digest)
        if args.reset_circuit_after_audit:
            common.append_jsonl(out / 'circuit_resets.jsonl', dict(reason=args.reset_circuit_after_audit,
                                timestamp_unix=time.time(), error_count=len(state['errors'])))
        if state['consecutive_error_attempts'] >= 3 and not args.reset_circuit_after_audit:
            raise RuntimeError('circuit open; inspect error journal')
        tasks = {t['task_id']: t for t in design['tasks']}
        (out / 'action_ledgers').mkdir(exist_ok=True)
        count, consecutive = 0, 0 if args.reset_circuit_after_audit else state['consecutive_error_attempts']
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_bind_parent_lifetime, initargs=(os.getpid(),)) as pool:
            while True:
                state = common.reconcile_attempts(out, jobs, digest)
                pending = state['pending']
                if not pending or (args.max_jobs is not None and count >= args.max_jobs) or not is_deepseek_offpeak():
                    break
                pending = pending_after_clean_barrier(state, jobs)
                cap = min(4, args.max_jobs - count) if args.max_jobs else 4
                active = []
                source_blocked = False
                for lane, job in enumerate(pending[:cap]):
                    if not is_deepseek_offpeak():
                        break
                    source = None
                    if job['condition'] in common.SOURCE_CONDITIONS:
                        try:
                            source = common.freeze_cross_task_source(state['rows'], job, design['tasks'], out, digest)
                        except Exception as exc:
                            common.append_jsonl(out / 'blocked_cells.jsonl', dict(job_key=job['job_key'],
                                                reason=type(exc).__name__, timestamp_unix=time.time()))
                            source_blocked = True
                            break
                    aid = uuid.uuid4().hex
                    start = dict(job, attempt=state['attempts'][job['job_key']] + 1,
                                 attempt_id=aid, config_digest=digest, lane=lane,
                                 model=common.MODEL, provider=common.PROVIDER, model_version=common.MODEL_VERSION,
                                 timestamp_unix=time.time(), ledger_path='action_ledgers/' + aid + '.sqlite',
                                 cross_task_source=source)
                    common.append_jsonl(out / 'run_attempts.jsonl', start)
                    spec = dict(start=start, settings=settings, task=tasks[job['task_id']],
                                base_url=config['base_url'], output=str(out))
                    try:
                        future = pool.submit(attempt, spec)
                    except Exception as exc:
                        future = Future()
                        future.set_exception(exc)
                        source_blocked = True
                    active.append((start, future))
                    if source_blocked:
                        break
                stop = source_blocked
                for start, future in active:
                    try:
                        kind, record = future.result()
                    except Exception as exc:
                        kind, record = 'error', dict(start, error_type=type(exc).__name__,
                                                     total_tokens=None, known_total_tokens=0, usage_complete=False)
                    if any(record.get(k) != v for k, v in start.items()):
                        raise ValueError('worker metadata changed')
                    common.append_jsonl(out / ('main_runs.jsonl' if kind == 'row' else 'run_errors.jsonl'), record)
                    consecutive = consecutive + 1 if kind == 'error' else 0
                    stop |= consecutive >= 3 or record.get('offpeak_blocked', False)
                    count += 1
                summary(out, jobs)
                if stop:
                    break
        result = summary(out, jobs)
        return 0 if result['completed'] == len(jobs) else 2


if __name__ == '__main__':
    raise SystemExit(main())
