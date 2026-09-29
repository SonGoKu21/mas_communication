"""Validate the complete predeclared P1 matrix before mapping execution roles."""
from itertools import product

STRATEGIES=('baseline','check_only','always_readback','guarded_readback')
CONDITIONS=('clean','valid_partial','contract_consistent_identity_corruption')

def execution_jobs(design,boundaries):
    tasks=design['tasks'];jobs=design['jobs']
    task_ids=[t['task_id'] for t in tasks]
    if len(tasks)!=6 or len(set(task_ids))!=6 or len({t['product_cluster'] for t in tasks})!=6:
        raise ValueError('six distinct frozen tasks and product clusters required')
    expected=set(product(task_ids,('sequential','flat'),(1,2,3),CONDITIONS,STRATEGIES))
    actual=[(j['task_id'],j['topology'],j['repeat_index'],j['condition'],j['variant']) for j in jobs]
    if len(jobs)!=432 or set(actual)!=expected or len({j['job_key'] for j in jobs})!=432:
        raise ValueError('incomplete or duplicate P1 factorial matrix')
    result=[]
    for job in jobs:
        if job['planned_exposures']!=int(job['condition']!='clean') or job['recovery_path_exposed'] is not False:
            raise ValueError('P1 requires one initial fault and no recovery exposure')
        result.append(dict(job,arm='baseline' if job['variant']=='baseline' else 'semantic_only',
                           p1_strategy=job['variant'],path_design=None,boundary=boundaries[job['condition']]))
    return result
