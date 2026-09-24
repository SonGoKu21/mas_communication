import pytest
from mas_faults.recovery_diagnostic import Exposure, build_jobs, WholeBudget, LimitReached


def test_ninety_unique_jobs():
    tasks = [{'task_id': str(i)} for i in range(3)]
    jobs = build_jobs(tasks)
    assert len(jobs) == len({j['job_key'] for j in jobs}) == 90
    assert sum(j['condition'] == 'clean' for j in jobs) == 18


def test_finite_loss_does_not_reset_on_recovery():
    f = Exposure('request_non_delivery', 2)
    assert f.deliver('evidence_handoff', {'x': 1}) == [{'x': 1}]
    assert f.deliver('action_request', {'x': 1}) == []
    assert f.deliver('action_request', {'x': 1}) == []
    assert f.deliver('action_request', {'x': 1}) == [{'x': 1}]
    assert len(f.events) == 2


def test_shared_observation_count_and_input_unchanged():
    f = Exposure('contract_consistent_identity_corruption', 2,
                 source={'task_id': 'other', 'evidence_id': 'real', 'payload': {'sku': 'b', 'product_id': '2'}})
    p = {'task_id': 't', 'sku': 'a', 'product_id': '1', 'evidence': '{}'}
    assert f.observe(p)['sku'] == 'b'
    assert f.observe(p)['sku'] == 'b'
    assert f.observe(p)['sku'] == 'a'
    assert p['sku'] == 'a'


def test_whole_budget_never_exceeds_limit():
    b = WholeBudget(http=2, model=1)
    b.consume('http'); b.consume('http')
    with pytest.raises(LimitReached):
        b.consume('http')
    assert b.used['http'] == 2


def test_budget_exhaustion_is_terminal_not_retryable(tmp_path, monkeypatch):
    import asyncio
    import mas_faults.recovery_diagnostic as module
    from test_shopping_multimechanism import UnitClient, UnitExecutor, task
    factory = module.WholeBudget
    monkeypatch.setattr(module, 'WholeBudget', lambda: factory(model=1))
    executor = UnitExecutor()
    executor._request = lambda *a, **kw: None
    row = asyncio.run(module.run_diagnostic(task(), dict(diagnostic=True, condition='clean',
                        exposure_count=0, arm='baseline', topology='sequential', job_key='unit'),
                        executor, UnitClient(), ledger_path=tmp_path / 'ledger.sqlite'))
    assert row['termination'] == 'budget_exhausted'
    assert row['outcome_class'] == 'budget_terminated'
    assert row['whole_budget']['used']['model'] == 1
    assert row['partial_trial']['events']


@pytest.mark.parametrize('arm', ['baseline', 'action_always', 'combined'])
@pytest.mark.parametrize('condition,count', [('clean', 0), ('request_non_delivery', 1),
                                           ('request_non_delivery', 2),
                                           ('contract_consistent_identity_corruption', 1),
                                           ('contract_consistent_identity_corruption', 2)])
@pytest.mark.parametrize('topology', ['sequential', 'flat'])
def test_diagnostic_real_autogen_unit_transport(arm, condition, count, topology, tmp_path):
    import asyncio
    from test_shopping_multimechanism import UnitClient, UnitExecutor, task, payload
    from mas_faults.recovery_diagnostic import run_diagnostic
    executor = UnitExecutor()
    executor._request = lambda *a, **kw: None
    row = asyncio.run(run_diagnostic(task(), dict(diagnostic=True, condition=condition,
                        exposure_count=count, arm=arm, topology=topology, job_key='unit'),
                        executor, UnitClient(), cross_task_evidence=dict(task_id='other', evidence_id='source',
                            payload={**payload(), 'sku': 'other', 'product_id': '99'}),
                        ledger_path=tmp_path / 'ledger.sqlite'))
    assert row['whole_budget']['used']['model'] > 0
    assert len(row['fault_events']) <= count
    if condition == 'request_non_delivery' and arm != 'baseline':
        assert len(row['fault_events']) == count
        assert row['environment_state']['observed_quantity'] == 2
        assert row['exposure_deliveries'][-1]['damaged'] is False
    if condition == 'contract_consistent_identity_corruption':
        assert row['environment_state']['sku'] == 'SKU'
        assert len(row['fault_events']) == count
