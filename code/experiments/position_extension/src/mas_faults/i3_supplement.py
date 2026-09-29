"""Reconstruct the exact historical carrier and inner-evidence mutation."""
import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from mas_faults.cross_benchmark_main_matrix import _carrier

@lru_cache(maxsize=None)
def source_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reconstruct_pair(reference):
    clean = reference['clean']
    rows = [r for r in source_rows(clean['source_jsonl']) if r.get('run_id') == clean['source_run_id']]
    if len(rows) != 1:
        raise ValueError('historical source run must resolve uniquely')
    current = _carrier(rows[0], benchmark=reference['benchmark'], source_jsonl=Path(clean['source_jsonl']))
    expected = deepcopy(current.envelope)
    historical = clean['original_message']
    if expected != historical and reference['benchmark'] == 'WebArena Reddit':
        new_summary = json.loads(expected['payload']['evidence_result']['evidence_summary'])
        old_text = historical['payload']['evidence_result']['evidence_summary']
        old_summary = json.loads(old_text)
        added = set(new_summary) - set(old_summary)
        if (added <= {'editor_verdict', 'pre_edit_state', 'verified_state'}
                and {k:new_summary[k] for k in old_summary if k in new_summary} == old_summary
                and historical['payload']['visible_evidence'] == [old_text]):
            expected['payload']['evidence_result']['evidence_summary'] = old_text
            expected['payload']['visible_evidence'] = [old_text]
    if (current.task_id != reference['task_id'] or expected != clean['original_message']
            or current.expected_answer != clean['expected_answer']):
        raise ValueError('historical carrier mismatch')
    current = replace(current, envelope=expected)
    inner = reference['historical_corrupted_evidence']
    if inner == current.envelope['payload']['evidence_result']:
        raise ValueError('historical corruption leaves task evidence unchanged')
    stale_task = str(inner.get('task_id', ''))
    if not stale_task or stale_task == current.task_id:
        raise ValueError('historical corruption must identify another task')
    stale_envelope = deepcopy(current.envelope)
    stale_envelope['payload']['evidence_result'] = deepcopy(inner)
    # Only this historical inner payload is consumed by the I3/I4 operator.
    # No task action or model prompt uses the other fields of this replay carrier.
    stale = replace(current, task_id=stale_task, envelope=stale_envelope)
    return current, stale
