"""Whole-workflow accounting around the actual AutoGen/HTTP runtime."""
import copy
import uuid
from mas_faults.recovery_diagnostic import WholeBudget, LimitReached
from mas_faults.mitigation_protocol import BudgetExhausted
from mas_faults.diagnostic_cart_audit import score
from runtime import run_trial


async def run(task, job, executor, client, *, cross_task_evidence=None, ledger_path):
    budget = WholeBudget(http=32, model=16, seconds=1200)
    request = executor._request
    evaluation = False
    evaluation_calls = 0

    def limited_request(*args, **kwargs):
        nonlocal evaluation_calls
        if evaluation:
            evaluation_calls += 1
        else:
            budget.consume('http')
        return request(*args, **kwargs)

    def begin_evaluation():
        nonlocal evaluation
        evaluation = True

    class CountedClient:
        model_info = client.model_info
        request_log = client.request_log

        def complete(self, *args, **kwargs):
            budget.consume('model')
            return client.complete(*args, **kwargs)

    executor._request = limited_request
    executor.begin_evaluation = begin_evaluation
    try:
        row = await run_trial(task, job, executor, CountedClient(),
                              cross_task_evidence=cross_task_evidence, ledger_path=ledger_path)
    except (LimitReached, BudgetExhausted) as exc:
        partial = getattr(exc, 'partial_trial', {})
        row = dict(job, task=task, run_id='rq4-budget-' + uuid.uuid4().hex,
                   final_task_success=False, termination='budget_exhausted',
                   exhausted_resource=str(exc), partial_trial=partial,
                   fault_events=partial.get('fault_events', []),
                   exposure_deliveries=partial.get('exposure_deliveries', []),
                   semantic_contract_events=partial.get('semantic_contract_events', []))
    finally:
        executor._request = request
    row['whole_budget'] = dict(limits=budget.limits, used=budget.used,
                               seconds_limit=budget.seconds, evaluation_http_calls=evaluation_calls)
    row['http_receipts'] = copy.deepcopy(executor.http_receipts)
    status, reason = score(row)
    row['recorded_cart_audit'] = dict(status=status, reason=reason)
    row['original_final_task_success'] = row.get('final_task_success')
    row['final_task_success'] = row.get('final_task_success') is True and status == 'pass'
    row['task_score'] = int(row['final_task_success'])
    return row
