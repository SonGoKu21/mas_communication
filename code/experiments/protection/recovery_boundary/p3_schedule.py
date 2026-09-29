"""Offline candidate ordering only; does not admit or execute experiments."""
import collections
import copy
import hashlib
import json
import random

DEFAULT_SEED = 20260928


def build_candidate(draft, *, seed=DEFAULT_SEED):
    if draft.get('execution_eligible') is not False or type(seed) is not int:
        raise ValueError('ineligible draft and integer frozen seed required')
    jobs = draft['jobs']
    if len(jobs) != 288 or len({job['job_key'] for job in jobs}) != 288:
        raise ValueError('288 unique jobs required')
    pairs = collections.defaultdict(list)
    for job in jobs:
        pairs[job['pair_key']].append(job)
    strata = {'clean': [], 'fault': []}
    for key, block in pairs.items():
        if len(block) != 2 or {job['arm'] for job in block} != {'baseline', 'combined'}:
            raise ValueError('each pair must contain exactly Baseline and Combined')
        coordinates = {tuple(job[field] for field in ('task_id', 'topology', 'repeat_index', 'exposure_scheme')) for job in block}
        if len(coordinates) != 1:
            raise ValueError('pair coordinates do not match')
        scope = block[0]['exposure_scheme']
        if scope not in ('clean', 'single_handoff', 'persistent_handoff', 'shared_workflow_evidence'):
            raise ValueError('unknown P3 exposure scope')
        strata['clean' if scope == 'clean' else 'fault'].append(key)
    if [len(strata[name]) for name in ('clean', 'fault')] != [36, 108]:
        raise ValueError('36 clean and 108 fault pair blocks required')
    ordered = []
    for index, stratum in enumerate(('clean', 'fault')):
        keys = sorted(strata[stratum])
        random.Random(seed + index).shuffle(keys)
        for position, key in enumerate(keys):
            arm_order = ('baseline', 'combined') if position % 2 == 0 else ('combined', 'baseline')
            arms = {job['arm']: job for job in pairs[key]}
            ordered.extend(copy.deepcopy(arms[arm]) for arm in arm_order)
    candidate = copy.deepcopy(draft)
    candidate.update(
        jobs=ordered, execution_eligible=False, status='offline_runtime_candidate_not_admitted',
        source_draft_sha256=hashlib.sha256(json.dumps(draft, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        schedule_policy=dict(unit='adjacent two-strategy pair block', strata=['clean', 'fault'],
                             clean_seed=seed, fault_seed=seed + 1,
                             first_strategy='alternate Baseline, Combined by shuffled block index within each stratum',
                             clean_blocks=36, fault_blocks=108,
                             position_counts_per_strategy=dict(clean=18, fault=54)))
    return candidate
