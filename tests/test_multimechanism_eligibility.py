"""Synthetic offline fixtures only, never evidence of a completed formal run."""
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'always_recheck', 'guarded_recheck', 'dependency',
        'independent', 'action_protocol', 'combined')
SOURCES = ('run_shopping_multimechanism.py', 'src/mas_faults/shopping_multimechanism.py',
           'src/mas_faults/multimechanism_matrix.py', 'src/mas_faults/multimechanism_faults.py',
           'src/mas_faults/shopping_action_protocol.py', 'src/mas_faults/mitigation_protocol.py',
           'src/mas_faults/shopping_mitigation.py', 'src/mas_faults/llm_client.py',
           'src/mas_faults/webarena_shopping_real.py')
CELLS = {'clean': None, 'request_non_delivery': 'action_request',
         'acknowledgement_loss': 'action_ack', 'duplicate_action_delivery': 'action_request',
         'valid_partial': 'evidence_handoff', 'same_session_reordering': 'observation_handoff',
         'cross_task_replay': 'evidence_handoff', 'stale_judgment_replay': 'judgment_handoff',
         'conflicting_observation': 'observation_handoff',
         'contract_consistent_identity_corruption': 'evidence_handoff'}


def module():
    path = ROOT / 'scripts/assess_multimechanism_eligibility.py'
    assert path.is_file(), 'eligibility sidecar is not implemented'
    spec = importlib.util.spec_from_file_location('eligibility', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def fixture(formal=False, unknown=False, source_condition=None):
    tasks = [{'task_id': f't-{p}-{v}', 'product_url': f'https://offline.invalid/p{p}',
              'product_title': f'Product {p}', 'initial_quantity': 1, 'quantity': v + 2}
             for p in range(5 if formal else 2 if source_condition else 1) for v in range(2)]
    jobs = []
    for task, top, (condition, boundary), rep in itertools.product(
            tasks, ('sequential', 'flat', 'hierarchical') if formal else ('sequential',),
            CELLS.items() if formal else [('clean', None), (source_condition, CELLS[source_condition])]
            if source_condition else [('clean', None)], range(1, 4) if formal else [1]):
        pair = json.dumps([task['task_id'], top, condition, boundary, rep], separators=(',', ':'))
        jobs.extend(dict(task_id=task['task_id'], topology=top, condition=condition, boundary=boundary,
                         repeat_index=rep, arm=arm, pair_key=pair, job_key=f'{pair}:{arm}') for arm in ARMS)
    config = dict(tasks=tasks, jobs=jobs, repetitions=3 if formal else 1, planned_runs=len(jobs),
                  shard_runs=len(jobs), shard_count=1, shard_index=0, max_attempts_per_job=2,
                  model='Qwen/Qwen3.8-27B', provider='modelscope_local',
                  version='shopping-multimechanism-v1', runner_schema=1,
                  source_hashes={name: 'a' * 64 for name in SOURCES})
    manifest = dict(config=config, config_digest=digest(canonical(config)))
    rows, starts, errors = [], [], []
    task_map = {t['task_id']: t for t in tasks}
    for i, job in enumerate(jobs):
        start = dict(job, attempt=2 if i == 0 else 1, attempt_id=f'a{i}',
                     ledger_path=f'action_ledgers/a{i}.sqlite3', cross_task_source=None,
                     config_digest=manifest['config_digest'], model=config['model'], provider=config['provider'])
        request = dict(request_index=1, model=config['model'], provider=config['provider'],
                       prompt_tokens=80, completion_tokens=20, total_tokens=100, status='success')
        rows.append(dict(start, run_id=f'r{i}', status='completed', model_requests=[request],
                         llm_request_log=[request], model_calls=1, usage_complete=True,
                         known_prompt_tokens=80, known_completion_tokens=20, known_total_tokens=100,
                         prompt_tokens=80, completion_tokens=20, total_tokens=100,
                         task=task_map[job['task_id']], final_task_success=True,
                         environment_task_success=True, decision_correct=True, evidence_acceptance_errors=0))
        starts.append(start)
    for row in rows:
        task = task_map[row['task_id']]
        payload = dict(product_id=task['product_url'], sku=task['product_url'],
                       observed_quantity=task['quantity'], cart_verified=True)
        row['environment_state'] = payload
        row['source_evidence'] = dict(task_id=row['task_id'], source='Worker', version=2, payload=payload,
            session_id=row['run_id'], entity_id=row['run_id'], action_id=row['run_id'], evidence_id=row['run_id'])
    donors = [r for r in rows if r['condition'] == 'clean' and r['arm'] == 'baseline']
    frozen_sources = {}
    for row, start in zip(rows, starts):
        if row['condition'] not in {'cross_task_replay', 'contract_consistent_identity_corruption'}:
            continue
        target = [row['task_id'], row['topology'], row['repeat_index']]
        name = 'cross_task_sources/' + digest(canonical(target)) + '.json'
        if name not in frozen_sources:
            donor = next(r for r in donors if r['task']['product_url'] != row['task']['product_url'])
            frozen = dict(config_digest=manifest['config_digest'], target=target,
                source_task_id=donor['task_id'], source_job_key=donor['job_key'], source_run_id=donor['run_id'],
                envelope=donor['source_evidence'], envelope_sha256=digest(canonical(donor['source_evidence'])))
            frozen_sources[name] = dict(frozen, file=name, file_sha256=digest(canonical(frozen)))
        row['cross_task_source'] = start['cross_task_source'] = frozen_sources[name]
    failed_start = dict(starts[0], attempt=1, attempt_id='failed', ledger_path='action_ledgers/failed.sqlite3')
    starts.append(failed_start)
    requests = [dict(rows[0]['model_requests'][0], prompt_tokens=20, completion_tokens=0, total_tokens=20),
                dict(rows[0]['model_requests'][0], request_index=2, prompt_tokens=None if unknown else 5,
                     completion_tokens=None if unknown else 5, total_tokens=None if unknown else 10,
                     status='error', error_type='TimeoutError', exception_chain=['TimeoutError'])]
    errors.append(dict(failed_start, status='timeout', error_type='WrapperError', model_calls=2,
                       usage_complete=False, model_requests=requests, llm_request_log=requests,
                       known_prompt_tokens=20 if unknown else 25, known_completion_tokens=0 if unknown else 5,
                       known_total_tokens=20 if unknown else 30, prompt_tokens=None if unknown else 25,
                       completion_tokens=None if unknown else 5, total_tokens=None if unknown else 30))
    cases = [dict(r) for r in rows]
    attempts = [dict(r, terminal_kind='completed') for r in rows]
    attempts.append(dict(errors[0], terminal_kind='error', total_tokens=None))
    finding = dict(code='attempt_usage_unknown', severity='unknown', job_key=jobs[0]['job_key'],
                   attempt_id='failed', run_id=None)
    audit = dict(audit_version='shopping-multimechanism-offline-v1', offline=True, status='incomplete',
                 findings=[finding], cases=cases, attempt_cases=attempts, inputs={},
                 task_count=len(tasks), repetitions=config['repetitions'],
                 provenance=dict(verified_source_files=9, frozen_source_files=9),
                 coverage=dict(planned_runs=len(jobs), shard_runs=len(jobs), completed_runs=len(rows),
                               completed_unique_jobs=len(rows), missing_job_keys=[], unfinished_attempts=0,
                               error_attempts=1, exhausted_job_keys=[], blocked_pair_keys=[], full_matrix_coverage=True))
    return audit, manifest, {'main_runs.jsonl': rows, 'run_attempts.jsonl': starts,
                             'run_errors.jsonl': errors, 'blocked_cells.jsonl': []}


def save(tmp_path, values):
    audit, manifest, logs = values
    root = tmp_path / 'results'
    root.mkdir(exist_ok=True)
    contents = {'matrix_manifest.json': canonical(manifest), 'summary.json': b'{}'}
    for row in logs['main_runs.jsonl']:
        source = row.get('cross_task_source')
        if isinstance(source, dict) and 'file' in source:
            contents.setdefault(source['file'], canonical({k: v for k, v in source.items()
                                                          if k not in ('file', 'file_sha256')}))
    contents.update({name: b''.join(canonical(row) + b'\n' for row in rows) for name, rows in logs.items()})
    for name, data in contents.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
        audit['inputs'][name] = dict(present=True, sha256=digest(data), bytes=len(data))
    ap = tmp_path / 'audit.json'
    ap.write_bytes(canonical(audit))
    return ap, root


def test_formal_null_replacement_preserves_outcomes_and_cost_lower_bound(tmp_path):
    m = module()
    known = m.assess(*save(tmp_path, fixture(formal=True)))
    unknown = m.assess(*save(tmp_path, fixture(formal=True, unknown=True)))
    assert known['outcome_eligible'] and unknown['outcome_eligible']
    assert known['artifact_eligible'] and unknown['artifact_eligible']
    assert known['outcomes'] == unknown['outcomes']
    assert known['outcomes']['denominator'] == 6300
    assert known['cost_complete'] and not unknown['cost_complete']
    job = next(j for j in unknown['jobs'] if j['attempt_count'] == 2)
    assert (job['known_lower_bound_tokens'], job['exact_total_tokens']) == (120, None)
    assert unknown['cost']['known_lower_bound_tokens'] == 630020
    assert unknown['cost']['exact_total_tokens'] is None
    assert len(unknown['exceptions']) == 1
    assert job['attempts'][0]['requests'][1]['exception_chain'] == ['TimeoutError']
    assert job['attempts'][0]['status'] == 'timeout'


def test_pilot_is_never_formal(tmp_path):
    report = module().assess(*save(tmp_path, fixture()))
    assert not report['artifact_eligible'] and not report['outcome_eligible']
    assert 'not_full_formal_design' in report['blockers']


@pytest.mark.parametrize('code', ['request_count_mismatch', 'request_count_unknown', 'trace_schema',
                                  'evaluation_mismatch', 'future_unknown_code', 'attempt_budget'])
def test_any_other_finding_blocks_even_info(tmp_path, code):
    values = fixture()
    values[0]['findings'].append(dict(code=code, severity='info'))
    report = module().assess(*save(tmp_path, values))
    assert any(b == f'audit_finding:{code}' for b in report['blockers'])
    assert not report['outcome_eligible']


@pytest.mark.parametrize('damage', ['count', 'index', 'model', 'ledger', 'binding', 'total',
                                   'logs', 'unknown_status', 'missing_requests', 'duplicate', 'unfinished'])
def test_raw_defects_fail_closed(tmp_path, damage):
    values = fixture(unknown=True)
    logs = values[2]
    row = logs['run_errors.jsonl'][0]
    if damage == 'count':
        row['model_calls'] = 3
    elif damage == 'index':
        row['model_requests'][1]['request_index'] = 1
    elif damage == 'model':
        row['model_requests'][0]['model'] = 'wrong'
    elif damage == 'ledger':
        row['ledger_path'] = logs['main_runs.jsonl'][0]['ledger_path']
        logs['run_attempts.jsonl'][-1]['ledger_path'] = row['ledger_path']
    elif damage == 'binding':
        row['config_digest'] = 'wrong'
    elif damage == 'total':
        row['known_total_tokens'] = 999
    elif damage == 'logs':
        row['llm_request_log'] = []
    elif damage == 'unknown_status':
        row['model_requests'][1]['status'] = 'future_state'
    elif damage == 'missing_requests':
        row.pop('model_requests')
    elif damage == 'duplicate':
        logs['run_errors.jsonl'].append(copy.deepcopy(row))
    else:
        logs['run_errors.jsonl'].clear()
    report = module().assess(*save(tmp_path, values))
    assert set(report['blockers']) - {'not_full_formal_design'}
    assert not report['artifact_eligible']
    assert not report['exceptions']


@pytest.mark.parametrize('name', ['main_runs.jsonl', 'run_errors.jsonl', 'run_attempts.jsonl'])
def test_missing_log_cannot_hide_behind_absent_hash(tmp_path, name):
    values = fixture()
    ap, root = save(tmp_path, values)
    (root / name).unlink()
    values[0]['inputs'][name] = {'present': False}
    ap.write_bytes(canonical(values[0]))
    with pytest.raises(ValueError, match='required input'):
        module().assess(ap, root)


def test_hash_mismatch_rejected_and_all_input_bytes_unchanged(tmp_path):
    ap, root = save(tmp_path, fixture())
    before = {p: p.read_bytes() for p in [ap, *root.iterdir()]}
    module().assess(ap, root)
    assert before == {p: p.read_bytes() for p in before}
    (root / 'run_errors.jsonl').write_bytes(b'[]\n')
    with pytest.raises(ValueError, match='hash'):
        module().assess(ap, root)


def test_reordering_is_summary_invariant(tmp_path):
    values = fixture(unknown=True)
    first = module().assess(*save(tmp_path, values))
    for key in ('cases', 'attempt_cases', 'findings'):
        values[0][key].reverse()
    values[1]['config']['jobs'].reverse()
    values[1]['config_digest'] = digest(canonical(values[1]['config']))
    for rows in values[2].values():
        rows.reverse()
        for row in rows:
            row['config_digest'] = values[1]['config_digest']
    for key in ('cases', 'attempt_cases'):
        for row in values[0][key]:
            row['config_digest'] = values[1]['config_digest']
    second = module().assess(*save(tmp_path, values))
    for key in ('jobs', 'outcomes', 'cost', 'by_arm', 'blockers', 'exceptions'):
        assert first[key] == second[key]


def test_cli_requires_new_separate_output(tmp_path):
    ap, root = save(tmp_path, fixture())
    m = module()
    assert m.main(['--audit', str(ap), '--result-dir', str(root), '--output', str(ap)]) == 2
    assert m.main(['--audit', str(ap), '--result-dir', str(root), '--output', str(root / 'new.json')]) == 2
    output = tmp_path / 'derived.json'
    assert m.main(['--audit', str(ap), '--result-dir', str(root), '--output', str(output)]) == 1
    assert json.loads(output.read_text())['version'] == 'shopping-multimechanism-eligibility-v1'


@pytest.mark.parametrize('damage', ['raw_outcome_type', 'raw_dimension_type', 'usage_flag_type',
                                   'missing_request_status', 'error_chain_identity', 'missing_source_file'])
def test_strict_identity_and_runtime_error_metadata(tmp_path, damage):
    values = fixture(unknown=True)
    raw = values[2]['run_errors.jsonl'][0]
    if damage == 'raw_outcome_type':
        values[2]['main_runs.jsonl'][0]['final_task_success'] = 1
    elif damage == 'raw_dimension_type':
        raw['repeat_index'] = True
    elif damage == 'usage_flag_type':
        raw['usage_complete'] = 0
    elif damage == 'missing_request_status':
        raw['model_requests'][1].pop('status')
    elif damage == 'error_chain_identity':
        raw['model_requests'][1]['exception_chain'] = ['DifferentError']
    else:
        source = dict(file='cross_task_sources/missing.json', file_sha256='a' * 64)
        raw['cross_task_source'] = source
        values[2]['run_attempts.jsonl'][-1]['cross_task_source'] = source
    report = module().assess(*save(tmp_path, values))
    assert set(report['blockers']) - {'not_full_formal_design'}
    assert not report['exceptions']


@pytest.mark.parametrize('damage', ['variants', 'product_count', 'topology', 'boundary', 'shard', 'repeat',
                                   'missing_case', 'duplicate_case', 'unfinished', 'unknown_finding'])
def test_full_size_labels_cannot_override_actual_formal_defects(tmp_path, damage):
    values = fixture(formal=True)
    audit, manifest, logs = values
    config = manifest['config']
    if damage == 'variants':
        config['tasks'][1]['quantity'] = config['tasks'][0]['quantity']
    elif damage == 'product_count':
        config['tasks'][0]['product_url'] += '/different'
    elif damage in ('topology', 'boundary'):
        config['jobs'][0][damage] = 'unknown'
    elif damage == 'shard':
        config['shard_count'] = 2
    elif damage == 'repeat':
        config['jobs'][0]['repeat_index'] = 4
    elif damage == 'missing_case':
        audit['cases'].pop()
    elif damage == 'duplicate_case':
        audit['cases'].append(copy.deepcopy(audit['cases'][0]))
    elif damage == 'unfinished':
        logs['main_runs.jsonl'].pop()
    else:
        audit['findings'].append(dict(code='new_code', severity='warning'))
    manifest['config_digest'] = digest(canonical(config))
    report = module().assess(*save(tmp_path, values))
    assert not report['artifact_eligible'] and not report['outcome_eligible']


def zero_error_fixture(formal=False):
    values = fixture(formal=formal)
    audit, _, logs = values
    logs['run_errors.jsonl'].clear()
    logs['run_attempts.jsonl'].pop()
    logs['run_attempts.jsonl'][0]['attempt'] = 1
    logs['main_runs.jsonl'][0]['attempt'] = 1
    audit['cases'][0]['attempt'] = 1
    audit['attempt_cases'].pop()
    audit['attempt_cases'][0]['attempt'] = 1
    audit['findings'] = []
    audit['status'] = 'complete'
    audit['coverage']['error_attempts'] = 0
    return values


def test_clean_formal_without_retries_or_exceptions(tmp_path):
    values = zero_error_fixture(formal=True)
    report = module().assess(*save(tmp_path, values))
    assert report['artifact_eligible'] and report['cost_complete']
    assert report['exceptions'] == [] and report['cost']['exact_total_tokens'] == 630000
    assert report['outcomes']['first_attempt_completed'] == 6300


def test_failed_run_id_cannot_duplicate_successful_run_id(tmp_path):
    values = fixture()
    failed = values[2]['run_errors.jsonl'][0]
    failed['run_id'] = values[2]['main_runs.jsonl'][0]['run_id']
    values[0]['attempt_cases'][-1]['run_id'] = failed['run_id']
    values[0]['findings'][0]['run_id'] = failed['run_id']
    report = module().assess(*save(tmp_path, values))
    assert 'duplicate_terminal_run' in report['blockers']


@pytest.mark.parametrize('damage', ['optional_presence', 'size', 'escape', 'duplicate_json', 'nonfinite'])
def test_invalid_input_inventory_and_json_fail_closed(tmp_path, damage):
    values = fixture()
    ap, root = save(tmp_path, values)
    if damage == 'optional_presence':
        values[0]['inputs']['blocked_cells.jsonl'] = {'present': False}
    elif damage == 'size':
        values[0]['inputs']['run_errors.jsonl']['bytes'] += 1
    elif damage == 'escape':
        values[0]['inputs']['../outside'] = {'present': False}
    elif damage == 'duplicate_json':
        ap.write_bytes(b'{"offline":true,"offline":true}')
    else:
        ap.write_bytes(b'{"offline":NaN}')
    if damage not in ('duplicate_json', 'nonfinite'):
        ap.write_bytes(canonical(values[0]))
    with pytest.raises(ValueError):
        module().assess(ap, root)


def test_concurrent_input_change_rejected(tmp_path, monkeypatch):
    ap, root = save(tmp_path, fixture())
    m = module()
    real_usage = m.usage

    def changed(row):
        (root / 'summary.json').write_bytes(b'{"changed":true}')
        return real_usage(row)

    monkeypatch.setattr(m, 'usage', changed)
    with pytest.raises(ValueError, match='input changed'):
        m.assess(ap, root)


def test_formal_cli_unknown_cost_is_outcome_eligible_not_exact_cost(tmp_path):
    ap, root = save(tmp_path, fixture(formal=True, unknown=True))
    output = tmp_path / 'eligibility-v1.json'
    assert module().main(['--audit', str(ap), '--result-dir', str(root), '--output', str(output)]) == 0
    report = json.loads(output.read_bytes())
    assert report['artifact_eligible'] and not report['exact_cost_comparison_eligible']
    assert report['provenance']['audit_sha256'] == digest(ap.read_bytes())
    assert report['by_arm']['baseline']['outcomes']['error_attempts'] == 1
    assert report['by_arm']['baseline']['outcomes']['retried_jobs'] == 1


@pytest.mark.parametrize('damage', ['version', 'schema', 'job_key', 'source_inventory', 'source_hash'])
def test_unknown_frozen_plan_or_source_attestation_is_blocked(tmp_path, damage):
    values = fixture()
    config = values[1]['config']
    if damage == 'version':
        config['version'] = 'future-version'
    elif damage == 'schema':
        config['runner_schema'] = True
    elif damage == 'job_key':
        config['jobs'][0]['job_key'] = 'noncanonical'
    elif damage == 'source_inventory':
        config['source_hashes'].clear()
    else:
        config['source_hashes'][SOURCES[0]] = 'invalid'
    values[1]['config_digest'] = digest(canonical(config))
    for rows in values[2].values():
        for row in rows:
            row['config_digest'] = values[1]['config_digest']
    for key in ('cases', 'attempt_cases'):
        for row in values[0][key]:
            row['config_digest'] = values[1]['config_digest']
    report = module().assess(*save(tmp_path, values))
    assert set(report['blockers']) - {'not_full_formal_design'}


def source_retry_fixture(condition):
    values = fixture(unknown=True, source_condition=condition)
    audit, _, logs = values
    logs['main_runs.jsonl'][0]['attempt'] = logs['run_attempts.jsonl'][0]['attempt'] = 1
    target = next(r for r in logs['main_runs.jsonl'] if r['condition'] == condition)
    target['attempt'] = 2
    next(r for r in logs['run_attempts.jsonl'] if r['attempt_id'] == target['attempt_id'])['attempt'] = 2
    fields = ('task_id', 'topology', 'condition', 'boundary', 'repeat_index', 'arm', 'job_key',
              'pair_key', 'cross_task_source')
    for failed in (logs['run_errors.jsonl'][0], logs['run_attempts.jsonl'][-1]):
        failed.update({k: copy.deepcopy(target[k]) for k in fields})
    audit['cases'] = [dict(r) for r in logs['main_runs.jsonl']]
    audit['attempt_cases'] = [dict(r, terminal_kind='completed') for r in logs['main_runs.jsonl']]
    audit['attempt_cases'].append(dict(logs['run_errors.jsonl'][0], terminal_kind='error', total_tokens=None))
    audit['findings'][0]['job_key'] = target['job_key']
    return values, target


@pytest.mark.parametrize('condition', ['cross_task_replay', 'contract_consistent_identity_corruption'])
@pytest.mark.parametrize('terminal', ['failed', 'completed'])
@pytest.mark.parametrize('missing', ['null', 'absent'])
def test_source_dependent_attempt_requires_source(tmp_path, condition, terminal, missing):
    values, target = source_retry_fixture(condition)
    logs = values[2]
    row = logs['run_errors.jsonl'][0] if terminal == 'failed' else target
    start = next(r for r in logs['run_attempts.jsonl'] if r['attempt_id'] == row['attempt_id'])
    for value in (row, start):
        if missing == 'null':
            value['cross_task_source'] = None
        else:
            value.pop('cross_task_source')
    report = module().assess(*save(tmp_path, values))
    assert not report['exceptions']
    assert any('source_binding_unknown' in b for b in report['blockers'])


@pytest.mark.parametrize('condition', ['cross_task_replay', 'contract_consistent_identity_corruption'])
def test_verified_failed_source_preserves_cost_only_exception(tmp_path, condition):
    values, _ = source_retry_fixture(condition)
    report = module().assess(*save(tmp_path, values))
    assert report['blockers'] == ['not_full_formal_design']
    assert len(report['exceptions']) == 1
    retried = next(j for j in report['jobs'] if j['attempt_count'] == 2)
    assert retried['known_lower_bound_tokens'] == 120 and retried['exact_total_tokens'] is None


@pytest.mark.parametrize('damage', ['envelope_hash', 'donor_run', 'donor_task', 'envelope', 'path'])
def test_failed_source_must_match_frozen_actual_clean_donor(tmp_path, damage):
    values, target = source_retry_fixture('cross_task_replay')
    original = target['cross_task_source']
    source = copy.deepcopy(original)
    if damage == 'envelope_hash':
        source['envelope_sha256'] = '0' * 64
    elif damage == 'donor_run':
        source['source_run_id'] = 'nonexistent'
    elif damage == 'donor_task':
        source['source_task_id'] = target['task_id']
    elif damage == 'envelope':
        source['envelope']['session_id'] = 'not-the-recorded-donor'
        source['envelope_sha256'] = digest(canonical(source['envelope']))
    else:
        source['file'] = 'cross_task_sources/noncanonical.json'
    frozen = {k: v for k, v in source.items() if k not in ('file', 'file_sha256')}
    source['file_sha256'] = digest(canonical(frozen))
    # Coherently replace all copies and rehash the frozen bytes: outer hashes
    # alone cannot detect a fabricated source reference.
    for rows in values[2].values():
        for row in rows:
            if row.get('cross_task_source') == original:
                row['cross_task_source'] = copy.deepcopy(source)
    report = module().assess(*save(tmp_path, values))
    assert not report['exceptions']
    assert any('source_' in b for b in report['blockers'])


def omit_error_log(ap, root, audit):
    (root / 'run_errors.jsonl').unlink()
    audit['inputs']['run_errors.jsonl'] = {'present': False}
    ap.write_bytes(canonical(audit))


def test_zero_errors_explicitly_absent_log_preserves_formal_eligibility(tmp_path):
    values = zero_error_fixture(formal=True)
    ap, root = save(tmp_path, values)
    m = module()
    present = m.assess(ap, root)
    omit_error_log(ap, root, values[0])
    before = {p: p.read_bytes() for p in [ap, *root.rglob('*')] if p.is_file()}
    absent = m.assess(ap, root)
    assert absent['artifact_eligible'] and absent['cost_complete']
    assert absent['exceptions'] == []
    for key in ('jobs', 'outcomes', 'cost', 'by_arm', 'blockers'):
        assert absent[key] == present[key]
    assert absent['provenance']['input_hashes']['run_errors.jsonl'] == {'present': False}
    assert not (root / 'run_errors.jsonl').exists()
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('damage', ['coverage_failure', 'exported_failure', 'unfinished', 'duplicate_start',
                                   'hidden_failed_retry', 'unknown_error_count'])
def test_absent_error_log_requires_zero_failures_and_full_reconciliation(tmp_path, damage):
    values = zero_error_fixture()
    audit, _, logs = values
    if damage == 'coverage_failure':
        audit['coverage']['error_attempts'] = 1
    elif damage == 'exported_failure':
        audit['attempt_cases'][0]['terminal_kind'] = 'error'
    elif damage == 'unfinished':
        logs['main_runs.jsonl'].pop()
    elif damage == 'duplicate_start':
        logs['run_attempts.jsonl'].append(copy.deepcopy(logs['run_attempts.jsonl'][0]))
    elif damage == 'unknown_error_count':
        audit['coverage'].pop('error_attempts')
    else:
        values = fixture()
        audit = values[0]
        audit['coverage']['error_attempts'] = 0
        audit['attempt_cases'].pop()
        audit['findings'] = []
        audit['status'] = 'complete'
    ap, root = save(tmp_path, values)
    omit_error_log(ap, root, audit)
    with pytest.raises(ValueError, match='required input'):
        module().assess(ap, root)


@pytest.mark.parametrize('damage', ['unrecorded', 'present_but_missing', 'absent_but_exists'])
def test_zero_errors_does_not_relax_inventory_hash_binding(tmp_path, damage):
    values = zero_error_fixture()
    ap, root = save(tmp_path, values)
    if damage == 'unrecorded':
        values[0]['inputs'].pop('run_errors.jsonl')
    elif damage == 'present_but_missing':
        (root / 'run_errors.jsonl').unlink()
    else:
        values[0]['inputs']['run_errors.jsonl'] = {'present': False}
    ap.write_bytes(canonical(values[0]))
    with pytest.raises(ValueError):
        module().assess(ap, root)
