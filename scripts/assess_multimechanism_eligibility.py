#!/usr/bin/env python3
"""Immutable offline eligibility sidecar; no runtime or auditor imports.

Usage: python assess_multimechanism_eligibility.py --audit AUDIT/summary.json
       --result-dir SNAPSHOT --output NEW/eligibility-v1.json
Exit 0: formally outcome-eligible artifact (cost may be incomplete).
Exit 1: derived ineligible report. Exit 2: invalid input or unsafe output.
The supplied offline audit remains the authority for semantic/source checks;
this helper verifies its byte bindings, formal design, and every raw attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

VERSION = 'shopping-multimechanism-eligibility-v1'
POLICY_VERSION = 'formal-statistical-eligibility-20260912-v1'
ARMS = ('baseline', 'always_recheck', 'guarded_recheck', 'dependency',
        'independent', 'action_protocol', 'combined')
CELLS = {'clean': None, 'request_non_delivery': 'action_request',
         'acknowledgement_loss': 'action_ack', 'duplicate_action_delivery': 'action_request',
         'valid_partial': 'evidence_handoff', 'same_session_reordering': 'observation_handoff',
         'cross_task_replay': 'evidence_handoff', 'stale_judgment_replay': 'judgment_handoff',
         'conflicting_observation': 'observation_handoff',
         'contract_consistent_identity_corruption': 'evidence_handoff'}
DIMENSIONS = ('task_id', 'topology', 'condition', 'boundary', 'repeat_index', 'arm')
JOB_FIELDS = ('job_key', 'pair_key', *DIMENSIONS)
MODEL = 'Qwen/Qwen3.8-27B'
PROVIDER = 'modelscope_local'
SOURCE_CONDITIONS = {'cross_task_replay', 'contract_consistent_identity_corruption'}
CORE_SOURCES = ('run_shopping_multimechanism.py', 'src/mas_faults/shopping_multimechanism.py',
                'src/mas_faults/multimechanism_matrix.py', 'src/mas_faults/multimechanism_faults.py',
                'src/mas_faults/shopping_action_protocol.py', 'src/mas_faults/mitigation_protocol.py',
                'src/mas_faults/shopping_mitigation.py', 'src/mas_faults/llm_client.py',
                'src/mas_faults/webarena_shopping_real.py')
REQUIRED = ('matrix_manifest.json', 'main_runs.jsonl', 'run_attempts.jsonl')
OUTCOME_FIELDS = ('final_task_success', 'environment_task_success', 'decision_correct',
                  'evidence_acceptance_errors')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('nonfinite JSON number')

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def local_path(root, name):
    rel = Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe input path')
    path = (root / rel).resolve()
    require(path.is_relative_to(root), 'input path escape')
    return path


def read_inputs(audit_path, result_dir):
    ap = Path(audit_path).resolve(strict=True)
    root = Path(result_dir).resolve(strict=True)
    require(root.is_dir() and ap.is_file(), 'local audit file and result directory required')
    audit_bytes = ap.read_bytes()
    audit = strict_json(audit_bytes)
    require(isinstance(audit, dict) and audit.get('audit_version') == 'shopping-multimechanism-offline-v1'
            and audit.get('offline') is True, 'unsupported audit schema')
    recorded = audit.get('inputs')
    require(isinstance(recorded, dict), 'audit input hashes required')
    for name in REQUIRED:
        require(isinstance(recorded.get(name), dict) and recorded[name].get('present') is True,
                'required input missing: ' + name)
    # Include optional files even when the auditor recorded their absence. A new
    # error/blocked/summary file must not silently disappear from the snapshot.
    require({'run_errors.jsonl', 'blocked_cells.jsonl', 'summary.json'} <= recorded.keys(),
            'optional input inventory missing')
    blobs, snapshot = {}, {ap: audit_bytes}
    for name, info in sorted(recorded.items()):
        require(isinstance(info, dict) and type(info.get('present')) is bool, 'invalid input hash record')
        path = local_path(root, name)
        if not info['present']:
            require(not path.exists(), 'input hash presence mismatch: ' + name)
            snapshot[path] = None
            continue
        require(path.is_file(), 'required input file missing: ' + name)
        data = path.read_bytes()
        require(info.get('sha256') == sha256(data) and type(info.get('bytes')) is int
                and info['bytes'] == len(data), 'input hash mismatch: ' + name)
        blobs[name] = data
        snapshot[path] = data
    return ap, root, audit, blobs, snapshot


def rows_from(blobs, name):
    result = []
    for line in blobs.get(name, b'').splitlines():
        if not line.strip():
            continue
        row = strict_json(line)
        require(isinstance(row, dict), 'JSONL object required: ' + name)
        result.append(row)
    return result


def verify_source(row, blobs, digest, tasks, completed):
    """Bind every source-dependent attempt, including failures, to frozen evidence."""
    source = row.get('cross_task_source')
    require(isinstance(source, dict) and isinstance(source.get('file'), str), 'source_binding_unknown')
    data = blobs.get(source['file'])
    require(data is not None and sha256(data) == source.get('file_sha256'), 'source_input_hash_mismatch')
    require(strict_json(data) == {k: v for k, v in source.items() if k not in ('file', 'file_sha256')},
            'source_binding_mismatch')
    target = [row['task_id'], row['topology'], row['repeat_index']]
    require(source.get('config_digest') == digest and source.get('target') == target
            and source['file'] == 'cross_task_sources/' + sha256(canonical(target).encode()) + '.json',
            'source_target_mismatch')
    envelope = source.get('envelope')
    require(isinstance(envelope, dict) and source.get('envelope_sha256') == sha256(canonical(envelope).encode()),
            'source_envelope_hash_mismatch')
    donor = completed.get(source.get('source_job_key'), {})
    source_task = tasks.get(donor.get('task_id'), {})
    target_task = tasks[row['task_id']]
    require(donor.get('run_id') == source.get('source_run_id') and donor.get('condition') == 'clean'
            and donor.get('arm') == 'baseline' and donor.get('config_digest') == digest
            and donor.get('final_task_success') is True and source_task
            and donor.get('task_id') == source.get('source_task_id') != row['task_id']
            and source_task.get('product_title') != target_task.get('product_title')
            and urlsplit(source_task['product_url']).path != urlsplit(target_task['product_url']).path
            and envelope == donor.get('source_evidence') and envelope.get('task_id') == donor['task_id']
            and envelope.get('source') == 'Worker' and type(envelope.get('version')) is int
            and all(isinstance(envelope.get(k), str) and envelope[k]
                    for k in ('session_id', 'entity_id', 'action_id', 'evidence_id')),
            'source_not_actual_eligible_clean')
    payload, truth = envelope.get('payload'), donor.get('environment_state')
    require(isinstance(payload, dict) and isinstance(truth, dict)
            and all(k in payload and payload[k] == truth.get(k)
                    for k in ('product_id', 'sku', 'observed_quantity', 'cart_verified')),
            'source_payload_mismatch')


def usage(row):
    require(type(row.get('usage_complete')) is bool, 'usage_flag_unknown')
    requests = row.get('model_requests')
    require(isinstance(requests, list) and all(isinstance(r, dict) for r in requests), 'missing_requests')
    require(type(row.get('model_calls')) is int and row['model_calls'] == len(requests), 'request_count_mismatch')
    indices = [r.get('request_index') for r in requests]
    require(all(integer(i) and i > 0 for i in indices) and indices == sorted(set(indices)), 'request_index_mismatch')
    if indices:
        require(indices == list(range(indices[0], indices[0] + len(indices))), 'request_index_gap')
    known = {'prompt_tokens': 0, 'completion_tokens': 0}
    complete = True
    for request in requests:
        require(request.get('model') == MODEL and request.get('provider') == PROVIDER, 'request_model_mismatch')
        status = request.get('status')
        require(status in (None, 'success', 'error'), 'unknown_request_status')
        chain = request.get('exception_chain')
        require(chain is None or isinstance(chain, list) and all(isinstance(t, str) and t for t in chain),
                'invalid_exception_chain')
        if status == 'error':
            require(isinstance(request.get('error_type'), str) and request['error_type'] and chain,
                    'request_error_identity_unknown')
            require(chain[0] == request['error_type'], 'request_error_chain_mismatch')
        if status == 'success':
            require(not chain and not request.get('error_type'), 'request_status_error_conflict')
        for field in known:
            value = request.get(field)
            require(value is None or integer(value), 'invalid_token_value')
            complete &= integer(value)
            known[field] += value if integer(value) else 0
        if any(request.get(k) is None for k in known):
            require(status == 'error', 'unknown_usage_without_recorded_request_error')
        total = request.get('total_tokens')
        require(total is None or integer(total), 'invalid_token_value')
        if all(integer(request.get(k)) for k in known):
            require(total is None or total == sum(request[k] for k in known), 'request_token_mismatch')
        elif integer(total):
            # Partial components plus an independently recorded total are not
            # enough to certify the component-based accounting contract.
            require(total >= sum(request[k] for k in known if integer(request.get(k))), 'request_token_mismatch')
    known['total_tokens'] = sum(known.values())
    for prefix in ('', 'known_'):
        for field, value in known.items():
            recorded = row.get(prefix + field)
            require(recorded is None or integer(recorded), 'invalid_token_value')
            if prefix == 'known_' or complete:
                require(recorded == value, 'attempt_token_mismatch')
            else:
                require(recorded is None or recorded == value, 'attempt_token_mismatch')
    nested = row.get('token_usage')
    require(nested is None or isinstance(nested, dict) and all(
        nested.get(k) == (known[k] if complete else row.get(k)) for k in known), 'nested_token_mismatch')
    log = row.get('llm_request_log')
    require(log is None or isinstance(log, list) and len(log) == len(requests) and all(
        isinstance(a, dict) and all(a.get(k) == v for k, v in b.items()) for a, b in zip(log, requests)),
        'request_logs_mismatch')
    return known['total_tokens'], known['total_tokens'] if complete else None


def assess(audit_path, result_dir):
    ap, root, audit, blobs, snapshot = read_inputs(audit_path, result_dir)
    manifest = strict_json(blobs['matrix_manifest.json'])
    require(isinstance(manifest, dict) and isinstance(manifest.get('config'), dict), 'invalid manifest')
    config = manifest['config']
    digest = sha256(canonical(config).encode())
    require(manifest.get('config_digest') == digest, 'manifest config hash mismatch')
    blockers = set()

    def check(condition, reason):
        if not condition:
            blockers.add(reason)
        return bool(condition)

    tasks, jobs = config.get('tasks'), config.get('jobs')
    require(isinstance(tasks, list) and tasks and all(isinstance(t, dict) for t in tasks), 'tasks required')
    require(isinstance(jobs, list) and jobs and all(isinstance(j, dict) for j in jobs), 'jobs required')
    task_map = {t.get('task_id'): t for t in tasks}
    require(all(isinstance(k, str) and k for k in task_map) and len(task_map) == len(tasks), 'duplicate/invalid task')
    planned, coordinates, pairs = {}, set(), defaultdict(set)
    for job in jobs:
        require(all(k in job for k in JOB_FIELDS), 'missing job dimensions')
        require(isinstance(job['job_key'], str) and job['job_key'] and isinstance(job['pair_key'], str)
                and job['pair_key'], 'invalid job identity')
        coord = tuple(job[k] for k in DIMENSIONS)
        require(all(isinstance(job[k], str) for k in ('task_id', 'topology', 'condition', 'arm'))
                and (job['boundary'] is None or isinstance(job['boundary'], str))
                and type(job['repeat_index']) is int, 'invalid job dimensions')
        check(job['job_key'] not in planned and coord not in coordinates, 'duplicate_planned_job')
        planned[job['job_key']] = job
        coordinates.add(coord)
        pairs[job['pair_key']].add(coord)
        expected_pair = json.dumps(list(coord[:-1]), separators=(',', ':'))
        check(job['pair_key'] == expected_pair and job['job_key'] == expected_pair + ':' + job['arm'],
              'noncanonical_job_identity')
    expected = set(itertools.product(task_map, ('sequential', 'flat', 'hierarchical'), CELLS, range(1, 4), ARMS))
    actual = {(j['task_id'], j['topology'], j['condition'], j['repeat_index'], j['arm']) for j in jobs}
    products = defaultdict(list)
    for task in tasks:
        require(isinstance(task.get('product_url'), str) and task['product_url'], 'task product required')
        products[task['product_url']].append((task.get('initial_quantity'), task.get('quantity')))
    valid_variants = all(len(v) == 2 and len(set(v)) == 2 and all(
        type(a) is int and type(b) is int and a > 0 and b > 0 and a != b for a, b in v) for v in products.values())
    formal = (len(tasks) == 10 and len(products) == 5 and valid_variants and len(jobs) == 6300
              and config.get('repetitions') == 3 and config.get('planned_runs') == 6300
              and config.get('shard_runs') == 6300 and config.get('shard_count') == 1
              and config.get('shard_index') == 0 and actual == expected
              and all(j['boundary'] == CELLS.get(j['condition']) for j in jobs))
    check(formal, 'not_full_formal_design')
    check(all(len(v) == 7 and len({c[:-1] for c in v}) == 1 and {c[-1] for c in v} == set(ARMS)
              for v in pairs.values()), 'invalid_pairing_plan')
    check(config.get('max_attempts_per_job') == 2, 'attempt_budget')
    check(config.get('version') == 'shopping-multimechanism-v1' and type(config.get('runner_schema')) is int
          and config['runner_schema'] == 1, 'unknown_manifest_schema')
    check(config.get('model') == MODEL and config.get('provider') == PROVIDER, 'config_model_mismatch')
    check(config.get('planned_runs') == len(jobs) == config.get('shard_runs'), 'plan_count_mismatch')
    check(audit.get('task_count') == len(tasks) and audit.get('repetitions') == config.get('repetitions'),
          'audit_design_mismatch')
    provenance = audit.get('provenance', {})
    hashes = config.get('source_hashes')
    valid_sources = (isinstance(hashes, dict) and set(CORE_SOURCES) <= hashes.keys()
        and all(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
                and isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
                for name, value in hashes.items()))
    check(valid_sources, 'invalid_source_inventory')
    check(isinstance(provenance, dict) and integer(provenance.get('frozen_source_files'))
          and valid_sources and provenance['frozen_source_files'] == len(hashes)
          and provenance.get('verified_source_files') == provenance['frozen_source_files'], 'source_verification_missing')
    rows = rows_from(blobs, 'main_runs.jsonl')
    starts = rows_from(blobs, 'run_attempts.jsonl')
    errors = rows_from(blobs, 'run_errors.jsonl')
    errors_absent = 'run_errors.jsonl' not in blobs
    if errors_absent:
        coverage = audit.get('coverage', {})
        exported_attempts = audit.get('attempt_cases')
        require(isinstance(coverage, dict) and type(coverage.get('error_attempts')) is int
                and coverage['error_attempts'] == 0 and isinstance(exported_attempts, list)
                and all(isinstance(r, dict) and r.get('terminal_kind') == 'completed' for r in exported_attempts),
                'required input missing: run_errors.jsonl has failed or unknown attempts')
    check(not rows_from(blobs, 'blocked_cells.jsonl'), 'blocked_cells_present')

    def index(values, field, label):
        result = {}
        for row in sorted(values, key=canonical):
            require(isinstance(row, dict), label + ' row must be object')
            key = row.get(field)
            if not check(isinstance(key, str) and key, 'invalid_' + label + '_id'):
                continue
            check(key not in result, 'duplicate_' + label)
            result.setdefault(key, row)
        return result

    start_map = index(starts, 'attempt_id', 'attempt_start')
    terminal_map = index(rows + errors, 'attempt_id', 'terminal_attempt')
    completed_map = index(rows, 'job_key', 'completed_job')
    index(rows, 'run_id', 'run')
    index([r for r in rows + errors if r.get('run_id') is not None], 'run_id', 'terminal_run')
    check(start_map.keys() == terminal_map.keys(), 'unfinished_or_unstarted_attempts')
    check(completed_map.keys() == planned.keys(), 'missing_or_unexpected_completed_jobs')
    error_ids = {r.get('attempt_id') for r in errors}
    ledger_ids, per_job, attempt_costs = set(), defaultdict(list), {}
    for aid, start in sorted(start_map.items()):
        row = terminal_map.get(aid)
        if row is None:
            continue
        try:
            job = planned.get(start.get('job_key'))
            require(job is not None, 'unexpected_job')
            for value in (start, row):
                require(all(k in value and type(value[k]) is type(job[k]) and value[k] == job[k]
                            for k in JOB_FIELDS), 'job_metadata_mismatch')
                require(value.get('config_digest') == digest and value.get('model') == MODEL
                        and value.get('provider') == PROVIDER, 'identity_config_model_mismatch')
                require(type(value.get('attempt')) is int and value['attempt'] in (1, 2), 'attempt_budget')
            require(all(row.get(k) == start.get(k) for k in ('attempt', 'attempt_id', 'cross_task_source', 'ledger_path')),
                    'attempt_binding_mismatch')
            require('cross_task_source' in start and 'cross_task_source' in row, 'source_binding_unknown')
            source = row['cross_task_source']
            if row['condition'] in SOURCE_CONDITIONS or source is not None:
                verify_source(row, blobs, digest, task_map, completed_map)
            ledger = start.get('ledger_path')
            require(isinstance(ledger, str) and ledger, 'retry_isolation_unknown')
            ledger = str(local_path(root, ledger))
            require(ledger not in ledger_ids, 'retry_isolation_mismatch')
            ledger_ids.add(ledger)
            failed = aid in error_ids
            require(row.get('status', 'completed') in ({'timeout', 'infra_error'} if failed else {'completed', 'unresolved'}),
                    'nonterminal_or_unknown_status')
            if failed:
                require(isinstance(row.get('error_type'), str) and row['error_type'], 'failure_identity_unknown')
            else:
                require(row.get('task') == task_map[job['task_id']], 'task_binding_mismatch')
            lower, exact = usage(row)
            request_details = [{k: r.get(k) for k in ('request_index', 'status', 'error_type', 'exception_chain',
                               'prompt_tokens', 'completion_tokens', 'total_tokens')} for r in row['model_requests']]
            value = dict(attempt_id=aid, attempt=row['attempt'], terminal_kind='error' if failed else 'completed',
                         status=row.get('status', 'completed'), error_type=row.get('error_type'),
                         exception_chain=row.get('exception_chain'), requests=request_details,
                         known_lower_bound_tokens=lower, exact_total_tokens=exact)
            per_job[job['job_key']].append(value)
            attempt_costs[aid] = value
        except (ValueError, KeyError, TypeError) as exc:
            blockers.add('invalid_attempt:' + aid + ':' + str(exc))
    for key, attempts in per_job.items():
        attempts.sort(key=lambda a: (a['attempt'], a['attempt_id']))
        numbers = [a['attempt'] for a in attempts]
        check(numbers == list(range(1, len(numbers) + 1)) and len(numbers) <= 2, 'noncontiguous_or_duplicate_attempts')
        check(attempts[-1]['terminal_kind'] == 'completed' and all(a['terminal_kind'] == 'error' for a in attempts[:-1]),
              'retry_after_completion_or_no_outcome')
    check(len(attempt_costs) == len(starts) == len(rows) + len(errors), 'attempt_reconciliation_incomplete')

    cases = audit.get('cases')
    exported = audit.get('attempt_cases')
    require(isinstance(cases, list) and isinstance(exported, list), 'audit case/attempt exports required')
    case_map = index(cases, 'job_key', 'audit_case')
    exported_map = index(exported, 'attempt_id', 'audit_attempt')
    check(case_map.keys() == completed_map.keys(), 'audit_case_coverage_mismatch')
    check(exported_map.keys() == terminal_map.keys(), 'audit_attempt_coverage_mismatch')
    for key, case in case_map.items():
        raw = completed_map.get(key, {})
        check(all(type(case.get(k)) is type(raw.get(k)) and case.get(k) == raw.get(k)
                  for k in (*JOB_FIELDS, 'run_id', 'attempt_id', *OUTCOME_FIELDS))
              and case.get('status', 'completed') == raw.get('status', 'completed'), 'audit_case_mismatch')
        check(all(type(case.get(k)) is bool for k in OUTCOME_FIELDS[:3])
              and integer(case.get('evidence_acceptance_errors')), 'outcome_unknown')
    for aid, exported_row in exported_map.items():
        raw = terminal_map.get(aid, {})
        cost = attempt_costs.get(aid)
        check(all(exported_row.get(k) == raw.get(k) for k in (*JOB_FIELDS, 'run_id', 'attempt'))
              and exported_row.get('terminal_kind') == ('error' if aid in error_ids else 'completed'),
              'audit_attempt_mismatch')
        if cost:
            expected_exact = cost['exact_total_tokens'] if raw.get('usage_complete') is True else None
            check(exported_row.get('known_total_tokens') == cost['known_lower_bound_tokens']
                  and exported_row.get('total_tokens') == expected_exact, 'audit_attempt_cost_mismatch')
    coverage = audit.get('coverage', {})
    expected_counts = dict(planned_runs=len(jobs), shard_runs=len(jobs), completed_runs=len(rows),
                           completed_unique_jobs=len(completed_map), error_attempts=len(errors), unfinished_attempts=0)
    check(isinstance(coverage, dict) and all(type(coverage.get(k)) is int and coverage[k] == v
          for k, v in expected_counts.items()) and coverage.get('full_matrix_coverage') is True
          and all(coverage.get(k) == [] for k in ('missing_job_keys', 'blocked_pair_keys', 'exhausted_job_keys')),
          'coverage_incomplete_or_mismatch')
    findings = audit.get('findings')
    require(isinstance(findings, list) and all(isinstance(f, dict) and isinstance(f.get('code'), str)
            and f.get('severity') in ('info', 'warning', 'unknown', 'error') for f in findings), 'invalid findings')
    candidates = []
    for finding in findings:
        aid = finding.get('attempt_id')
        row = terminal_map.get(aid, {})
        if (finding['code'] == 'attempt_usage_unknown' and finding['severity'] == 'unknown'
                and aid in error_ids and aid in attempt_costs
                and all(finding.get(k) == row.get(k) for k in ('run_id', 'job_key', 'attempt_id'))):
            candidates.append(dict(code=finding['code'], attempt_id=aid, job_key=row['job_key'],
                reason='verified_failed_attempt_cost_only', known_lower_bound_tokens=attempt_costs[aid]['known_lower_bound_tokens'],
                exact_total_tokens=attempt_costs[aid]['exact_total_tokens']))
        else:
            blockers.add('audit_finding:' + finding['code'])
    exception_ids = {f['attempt_id'] for f in candidates}
    check(len(exception_ids) == len(candidates), 'duplicate_cost_exception')
    for aid, cost in attempt_costs.items():
        raw = terminal_map[aid]
        if raw.get('usage_complete') is not True or cost['exact_total_tokens'] is None:
            check(aid in exception_ids, 'unexplained_unknown_usage')
    check(audit.get('status') == ('incomplete' if findings else 'complete'), 'audit_status_unexplained')
    if audit.get('blockers'):
        blockers.add('source_audit_blockers')
    structural = blockers - {'not_full_formal_design'}
    if errors_absent:
        # Absence is meaningful only after starts, terminal rows, audit exports,
        # identities and costs all reconcile; a successful retry cannot hide it.
        require(not structural, 'required input missing: run_errors.jsonl lacks complete zero-error reconciliation')
    exceptions = sorted(candidates, key=canonical) if not structural else []
    report_jobs = []
    for key, job in sorted(planned.items()):
        attempts = per_job.get(key, [])
        lower = sum(a['known_lower_bound_tokens'] for a in attempts)
        exact = lower if attempts and all(a['exact_total_tokens'] is not None for a in attempts) else None
        report_jobs.append(dict(job_key=key, arm=job['arm'], attempt_count=len(attempts), attempts=attempts,
                                known_lower_bound_tokens=lower, exact_total_tokens=exact))

    def costs(values):
        attempts = [a for j in values for a in j['attempts']]
        lower = sum(j['known_lower_bound_tokens'] for j in values)
        complete = not structural and bool(values) and all(j['exact_total_tokens'] is not None for j in values)
        return dict(known_lower_bound_tokens=lower, exact_total_tokens=lower if complete else None,
                    unknown_attempts=sum(a['exact_total_tokens'] is None for a in attempts),
                    attempt_count=len(attempts), verified=not structural)

    def outcomes_for(arm=None):
        completed = [r for r in completed_map.values() if arm is None or r.get('arm') == arm]
        failed = [r for r in errors if arm is None or r.get('arm') == arm]
        return dict(denominator=len(completed),
                    **{k: sum(r.get(k) is True for r in completed) for k in OUTCOME_FIELDS[:3]},
                    error_attempts=len(failed), retried_jobs=sum(
                        j['attempt_count'] > 1 for j in report_jobs if arm is None or j['arm'] == arm),
                    first_attempt_completed=sum(r.get('attempt') == 1 for r in completed),
                    first_attempt_successes=sum(r.get('attempt') == 1 and r.get('final_task_success') is True for r in completed),
                    error_status_counts=dict(sorted(Counter(str(r.get('status', 'unknown')) for r in failed).items())),
                    error_type_counts=dict(sorted(Counter(str(r.get('error_type', 'unknown')) for r in failed).items())))

    outcomes = outcomes_for()
    cost = costs(report_jobs)
    for path, data in snapshot.items():
        require((not path.exists()) if data is None else path.is_file() and path.read_bytes() == data,
                'input changed during assessment')
    return dict(version=VERSION, policy_version=POLICY_VERSION, offline=True,
                status='eligible' if not blockers else 'ineligible', outcome_eligible=not blockers,
                artifact_eligible=not blockers, cost_complete=cost['exact_total_tokens'] is not None,
                exact_cost_comparison_eligible=not blockers and cost['exact_total_tokens'] is not None,
                blockers=sorted(blockers), exceptions=exceptions, jobs=report_jobs, cost=cost, outcomes=outcomes,
                attempt_reconciliation=dict(recorded_starts=len(starts), recorded_terminals=len(rows) + len(errors),
                    verified_joined_attempts=len(attempt_costs),
                    unverified_attempt_ids=sorted(set(start_map).union(terminal_map) - attempt_costs.keys())),
                by_arm={arm: dict(costs([j for j in report_jobs if j['arm'] == arm]), outcomes=outcomes_for(arm)) for arm in ARMS},
                design=dict(matches_formal_design=formal, planned_jobs=len(jobs), tasks=len(tasks), products=len(products),
                            required_jobs=6300, required_tasks=10, required_products=5, variants_per_product=2,
                            topologies=3, cells=10, arms=7, repetitions=3),
                provenance=dict(audit_file=str(ap), audit_sha256=sha256(snapshot[ap]), result_dir=str(root),
                                input_hashes=audit['inputs'], config_digest=digest,
                                helper_sha256=sha256(Path(__file__).read_bytes())),
                source_audit_status=audit['status'], source_findings=sorted(findings, key=canonical),
                limitations=['Eligibility trusts the hash-bound offline audit for source, semantic, exposure and trace validation; '
                             'it does not independently prove runtime or backend truth.',
                             'Outcome estimand is the frozen at-most-two-attempt workflow, not first-attempt success.',
                             'Unknown cost forbids exact cost-effectiveness comparisons; differences of lower bounds are not savings.',
                             'No confidence intervals, significance tests, or T7 integration are performed.',
                             'Blocked reports retain diagnostic accounting only; do not use it for inference.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--result-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        root = args.result_dir.resolve()
        require(not output.exists() and not output.is_relative_to(root) and output != args.audit.resolve(),
                'output must be new and outside result directory')
        report = assess(args.audit, args.result_dir)
        # Exclusive creation prevents a race from overwriting an existing audit.
        with output.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return int(not report['artifact_eligible'])
    except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
        print('Eligibility stopped: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
