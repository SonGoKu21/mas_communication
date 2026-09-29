"""RQ4 design only; importing or enumerating jobs never calls a model."""
import hashlib
import json

ARMS = ('baseline', 'action_only', 'semantic_only', 'combined')
TOPOLOGIES = ('sequential', 'flat')
FAULTS = ('request_non_delivery', 'duplicate_action_delivery', 'valid_partial',
          'cross_task_replay', 'contract_consistent_identity_corruption')
PATHS = ('single_path', 'duplicate_forwarding', 'independent_observation')


def mechanisms(arm):
    if arm not in ARMS:
        raise ValueError('unknown RQ4 arm; legacy Combined is not interchangeable')
    return dict(action=arm in ('action_only', 'combined'),
                semantic=arm in ('semantic_only', 'combined'))


def build_jobs(tasks):
    if len(tasks) != 6:
        raise ValueError('six frozen tasks required')
    for field in ('task_id', 'product_cluster'):
        values = [t.get(field) for t in tasks]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != 6:
            raise ValueError('six distinct task IDs and product clusters required')
    jobs = []

    def add(experiment, selected, conditions, variants, topologies, exposures):
        for ci, condition in enumerate(conditions):
            for ti, task in enumerate(selected):
                for ai, topology in enumerate(topologies):
                    for repeat in range(1, 4):
                        count = 0 if condition == 'clean' else exposures
                        pair = json.dumps([experiment, task['task_id'], topology,
                                           condition, count, repeat], separators=(',', ':'))
                        offset = (ci + ti + ai + repeat - 1) % len(variants)
                        for variant in variants[offset:] + variants[:offset]:
                            key = hashlib.sha256((pair + ':' + variant).encode()).hexdigest()
                            jobs.append(dict(experiment=experiment, task_id=task['task_id'],
                                             product_cluster=task['product_cluster'],
                                             topology=topology, condition=condition,
                                             variant=variant, repeat_index=repeat,
                                             planned_exposures=count,
                                             recovery_path_exposed=experiment == 'stress',
                                             pair_key=pair, job_key=key))
    add('main', tasks, ('clean',) + FAULTS, ARMS, TOPOLOGIES, 1)
    add('stress', tasks[:3], (FAULTS[0], FAULTS[-1]), ARMS, TOPOLOGIES, 2)
    add('path_diagnostic', tasks[:5], (FAULTS[-1], 'cross_task_replay'),
        PATHS, ('fixed_path_diagnostic',), 1)
    return jobs
