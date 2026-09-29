"""Map the exact frozen candidate to durable execution records."""
import copy,hashlib,json
SCHEDULE_SHA256 = '53078ca7949c96337f8815850146d85b553a333453334f02f96a5dadf323b639'
TASKS_SHA256 = '738dc8d903f8417aaa7b19cb4929860d7f248894ebe9cd74e975e9762b23547c'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def execution_jobs(design):
    if digest(design['jobs']) != SCHEDULE_SHA256 or digest(design['tasks']) != TASKS_SHA256:
        raise ValueError('P3 jobs/tasks differ from frozen balanced 288-cell candidate')
    tasks={t['task_id']:t for t in design['tasks']}
    return [dict(copy.deepcopy(j), experiment='rq4_p3_scope_20260928',variant=j['arm'],
                 condition='clean' if j['exposure_scheme']=='clean' else 'contract_consistent_identity_corruption',
                 product_cluster=tasks[j['task_id']]['product_cluster'],
                 boundary=None if j['exposure_scheme']=='clean' else 'evidence_handoff')
            for j in design['jobs']]

def pending_after_clean_barrier(state,jobs):
    pending=state['pending']
    clean=[j for j in pending if j['condition']=='clean']
    if clean: return clean
    expected={j['job_key'] for j in jobs if j['condition']=='clean'}
    actual=[r['job_key'] for r in state['rows'] if r['condition']=='clean']
    if pending and (len(actual)!=72 or len(expected)!=72 or set(actual)!=expected):
        raise RuntimeError('72 clean rows required before any P3 fault')
    return pending
