"""Only historical mechanism switches; P3 scheduling lives in p3_schedule.py."""
ARMS = ('baseline', 'action_only', 'semantic_only', 'combined')


def mechanisms(arm):
    if arm not in ARMS:
        raise ValueError('unknown RQ4 arm; legacy Combined is not interchangeable')
    return dict(action=arm in ('action_only', 'combined'),
                semantic=arm in ('semantic_only', 'combined'))
