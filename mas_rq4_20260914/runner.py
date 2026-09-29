"""RQ4 real execution. Four isolated processes; only parent writes journals."""
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'legacy'))
sys.path.insert(0, str(ROOT / 'legacy/src'))
import run_shopping_multimechanism as common
from scripts.run_multimechanism_parallel import _real_client, _executor, _bind_parent_lifetime
from trial import run
from mas_faults.deepseek_schedule import is_deepseek_offpeak


def execution_jobs(design):
    jobs = []
    for j in design['jobs']:
        path = j['variant'] if j['experiment'] == 'path_diagnostic' else None
        jobs.append(dict(j, arm='baseline' if path else j['variant'],
                         topology='sequential' if path else j['topology'], path_design=path,
                         boundary=common.matrix.CELLS[j['condition']]))
    return jobs


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}


def attempt(spec):
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
    (output / 'summary.md').write_text('# RQ4 执行进度\n\n' +
        f"完成 {len(rows)}/{len(jobs)}，错误尝试 {len(errors)}，已知token下限 {result['known_total_tokens']}。\n\n" +
        '此文件是运行进度；配对效应与任务聚类统计在完成后单独报告。\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--isolation', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--reset-circuit-after-audit', help='Audited cause/resolution; never resets attempt budgets')
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError('positive max-jobs required')
    design = json.loads(args.design.read_text())
    isolation = json.loads(args.isolation.read_text())
    if isolation.get('passed') is not True or isolation.get('max_workers') != 4:
        raise ValueError('verified four-cart gate required')
    jobs = execution_jobs(design)
    if len(jobs) != 1098 or len({j['job_key'] for j in jobs}) != 1098:
        raise ValueError('invalid matrix')
    settings = common.inference_settings()
    config = dict(tasks=design['tasks'], jobs=jobs, inference_settings=settings,
                  model=common.MODEL, provider=common.PROVIDER, model_version=common.MODEL_VERSION,
                  base_url='http://127.0.0.1:17770', source_hashes=source_hashes(), lanes=4,
                  budget=dict(http=32, model=16, seconds=1200, semantic_readbacks=1, action_replays=2),
                  max_attempts_per_job=2, repetitions=3,
                  design_sha256=hashlib.sha256(args.design.read_bytes()).hexdigest())
    config['isolation_sha256'] = hashlib.sha256(args.isolation.read_bytes()).hexdigest()
    digest = common.matrix.config_digest(config)
    out = args.output
    if out.resolve().is_relative_to(ROOT):
        raise ValueError('result directory must be outside source tree')
    out.mkdir(parents=True, exist_ok=True)
    with common.locked_workflow(), common.locked_output(out):
        manifest = out / 'matrix_manifest.json'
        expected = dict(config=config, config_digest=digest)
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError('frozen source/config mismatch; refuse resume')
        else:
            common._write_json(manifest, expected, exclusive=True)
            for name in config['source_hashes']:
                dest = out / 'source_snapshot' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as f:
                    f.write((ROOT / name).read_bytes())
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
                clean_pending = [j for j in pending if j['condition'] == 'clean']
                if clean_pending:
                    pending = clean_pending
                elif sum(r['condition'] == 'clean' for r in state['rows']) != 144:
                    raise RuntimeError('clean jobs exhausted or incomplete; no fault launch')
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
