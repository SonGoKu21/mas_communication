"""Isolated finite-burst diagnostic; injector state is never sent to agents."""
import copy
import json
import time
from mas_faults.multimechanism_faults import SingleBoundaryFault


class LimitReached(RuntimeError):
    pass


class WholeBudget:
    def __init__(self, http=32, model=16, seconds=1200):
        self.limits = {'http': http, 'model': model}
        self.used = dict.fromkeys(self.limits, 0)
        self.started = time.monotonic()
        self.seconds = seconds

    def consume(self, kind):
        if time.monotonic() - self.started >= self.seconds or self.used[kind] >= self.limits[kind]:
            raise LimitReached(kind)
        self.used[kind] += 1


class Exposure:
    def __init__(self, condition, count, source=None):
        if condition not in ('clean', 'request_non_delivery', 'contract_consistent_identity_corruption'):
            raise ValueError('unsupported diagnostic fault')
        if type(count) is not int or count not in (0, 1, 2):
            raise ValueError('invalid exposure count')
        self.condition, self.count, self.source = condition, count, copy.deepcopy(source)
        self.events, self.deliveries = [], []
        self.seen = 0

    def deliver(self, boundary, message):
        if boundary != 'action_request' or self.condition != 'request_non_delivery':
            return [copy.deepcopy(message)]
        self.seen += 1
        damaged = self.seen <= self.count
        result = [] if damaged else [copy.deepcopy(message)]
        self.deliveries.append({'path': boundary, 'ordinal': self.seen, 'damaged': damaged})
        if damaged:
            operator = SingleBoundaryFault('request_non_delivery')
            operator.deliver(boundary, message)
            self.events.extend(operator.events)
        return result

    def observe(self, payload):
        if self.condition != 'contract_consistent_identity_corruption':
            return copy.deepcopy(payload)
        self.seen += 1
        damaged = self.seen <= self.count
        self.deliveries.append({'path': 'shared_observation', 'ordinal': self.seen, 'damaged': damaged})
        if not damaged:
            return copy.deepcopy(payload)
        operator = SingleBoundaryFault(self.condition, cross_task_evidence=self.source)
        delivered = operator.deliver('evidence_handoff', {'task_id': payload.get('task_id'), 'payload': payload})
        self.events.extend([{**e, 'boundary': 'shared_observation', 'ordinal': self.seen} for e in operator.events])
        return delivered[0]['payload']


def build_jobs(tasks):
    if len(tasks) != 3:
        raise ValueError('exactly three frozen tasks required')
    jobs = []
    cells = [('clean', 0), ('request_non_delivery', 1), ('request_non_delivery', 2),
             ('contract_consistent_identity_corruption', 1), ('contract_consistent_identity_corruption', 2)]
    arms = ('baseline', 'action_always', 'combined')
    for ci, (condition, count) in enumerate(cells):
        for ti, task in enumerate(tasks):
            for topology in ('sequential', 'flat'):
                pair = json.dumps([task['task_id'], topology, condition, count, 1], separators=(',', ':'))
                offset = (ci + ti) % 3
                for arm in arms[offset:] + arms[:offset]:
                    jobs.append(dict(task_id=task['task_id'], topology=topology, condition=condition,
                                     boundary=None if condition == 'clean' else ('action_request' if condition == 'request_non_delivery' else 'shared_observation'),
                                     repeat_index=1, arm=arm, pair_key=pair, job_key=pair + ':' + arm,
                                     diagnostic=True, exposure_count=count))
    return jobs


async def run_diagnostic(task, job, executor, client, *, cross_task_evidence=None, ledger_path):
    from mas_faults.shopping_multimechanism import run_trial
    budget = WholeBudget()
    exposure = Exposure(job['condition'], job['exposure_count'], cross_task_evidence)
    active = False
    evaluation = False
    request, observe = executor._request, executor.reobserve_cart
    evaluation_calls = 0

    def limited_request(*args, **kwargs):
        nonlocal evaluation_calls
        if evaluation:
            evaluation_calls += 1
        else:
            budget.consume('http')
        return request(*args, **kwargs)

    def observed(*args, **kwargs):
        payload = observe(*args, **kwargs)
        return exposure.observe(payload) if active and not evaluation else payload

    def activate():
        nonlocal active
        active = True

    def evaluate():
        nonlocal evaluation
        evaluation = True

    class CountedClient:
        model_info = client.model_info
        request_log = client.request_log

        def complete(self, *args, **kwargs):
            budget.consume('model')
            return client.complete(*args, **kwargs)

    executor._request, executor.reobserve_cart = limited_request, observed
    executor.diagnostic_exposure = exposure
    executor.activate_diagnostic = activate
    executor.begin_evaluation = evaluate
    try:
        row = await run_trial(task, job, executor, CountedClient(),
                              cross_task_evidence=cross_task_evidence, ledger_path=ledger_path)
    except LimitReached as exc:
        # Budget exhaustion is a terminal observation, never a retryable API error.
        row = dict(job, task=task, run_id='budget-' + job['job_key'], final_task_success=False,
                   termination='budget_exhausted', exhausted_resource=str(exc),
                   final_verdict=None, evaluation_status='not_available',
                   partial_trial=getattr(exc, 'partial_trial', None))
    finally:
        executor._request, executor.reobserve_cart = request, observe
    row['whole_budget'] = dict(limits=budget.limits, used=budget.used, seconds_limit=budget.seconds,
                               evaluation_http_calls=evaluation_calls,
                               deadline_policy='checked_before_every_request; in-flight request retains transport timeout')
    row['exposure_deliveries'] = exposure.deliveries
    row['fault_events'] = exposure.events
    row['http_receipts'] = copy.deepcopy(executor.http_receipts)
    from mas_faults.diagnostic_cart_audit import score
    cart_status, cart_reason = score(row)
    row['recorded_cart_audit'] = dict(status=cart_status, reason=cart_reason)
    row['original_final_task_success'] = row.get('final_task_success')
    row['final_task_success'] = row.get('final_task_success') is True and cart_status == 'pass'
    verdict = row.get('final_verdict') or {}
    row['outcome_class'] = ('task_success' if row['final_task_success'] else
                            'false_success_declaration' if verdict.get('decision') == 'accept' and cart_status == 'fail'
                            else 'safe_rejection' if verdict.get('decision') == 'reject'
                            else 'budget_terminated' if row.get('termination') == 'budget_exhausted'
                            else 'unsuccessful_or_unknown')
    return row
