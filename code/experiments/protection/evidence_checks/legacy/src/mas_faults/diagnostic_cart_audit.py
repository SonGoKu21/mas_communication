"""Offline receipt audit. Does not import the experiment evaluator."""
import argparse
import collections
import copy
import json
from pathlib import Path


def score(row):
    receipts = row.get('http_receipts', [])
    indices = row.get('evaluation_receipt_indices', [])
    def valid(r):
        return r.get('request_method') == 'GET' and r.get('status_code') == 200
    initial = [r for r in receipts if valid(r) and r.get('purpose') == 'reobserve_cart.items'
               and r.get('receipt_index') not in indices]
    final = [r for r in receipts if valid(r) and r.get('purpose') == 'reobserve_cart.items'
             and r.get('receipt_index') in indices]
    identity = [r for r in receipts if valid(r) and r.get('purpose') == 'add_to_cart.product_page']
    if not initial or len(final) != 1 or len(identity) != 1:
        return 'unknown', 'missing_or_ambiguous_receipts'
    before, after = initial[0], final[0]
    cart = before.get('guest_cart_id_sha256')
    if not cart or cart != after.get('guest_cart_id_sha256'):
        return 'unknown', 'cart_binding'
    sku = (identity[0].get('response_payload') or {}).get('sku')
    a, b = before.get('response_payload'), after.get('response_payload')
    if not sku or not isinstance(a, list) or not isinstance(b, list):
        return 'unknown', 'snapshot_unavailable'
    fields = ('item_id', 'sku', 'name', 'qty')
    for items in (a, b):
        if any(not isinstance(x, dict) or any(k not in x or x[k] is None for k in fields)
               or type(x['qty']) not in (int, float) for x in items):
            return 'unknown', 'snapshot_fields'
    target_a = [x for x in a if x['sku'] == sku]
    target_b = [x for x in b if x['sku'] == sku]
    if len(target_a) != 1 or target_a[0]['qty'] != row['task']['initial_quantity']:
        return 'unknown', 'initial_state_invalid'
    if len(target_b) != 1 or target_b[0]['qty'] != row['task']['quantity']:
        return 'fail', 'target_quantity_or_cardinality'
    if target_b[0]['name'] != target_a[0]['name'] or target_b[0]['item_id'] != target_a[0]['item_id']:
        return 'fail', 'target_identity_changed'
    def others(items):
        return collections.Counter(json.dumps({k: x[k] for k in fields}, sort_keys=True)
                                   for x in items if x['sku'] != sku)
    if others(a) != others(b):
        return 'fail', 'non_target_changed_or_added'
    return 'pass', 'recorded_cart_constraints_satisfied'


def self_test():
    item = dict(item_id=1, sku='a', name='A', qty=1)
    row = dict(task=dict(initial_quantity=1, quantity=2), evaluation_receipt_indices=[2], http_receipts=[
        dict(request_method='GET', status_code=200, purpose='add_to_cart.product_page', response_payload={'sku': 'a'}),
        dict(request_method='GET', status_code=200, purpose='reobserve_cart.items', receipt_index=1,
             guest_cart_id_sha256='x', response_payload=[item]),
        dict(request_method='GET', status_code=200, purpose='reobserve_cart.items', receipt_index=2,
             guest_cart_id_sha256='x', response_payload=[{**item, 'qty': 2}])])
    assert score(row)[0] == 'pass'
    for modification, expected in [
        (lambda r: r['http_receipts'][-1]['response_payload'].append(dict(item_id=2, sku='b', name='B', qty=1)), 'fail'),
        (lambda r: r['http_receipts'][-1]['response_payload'][0].update(qty=3), 'fail'),
        (lambda r: r['http_receipts'].pop(), 'unknown'),
        (lambda r: r['http_receipts'][-1].update(guest_cart_id_sha256='y'), 'unknown'),
        (lambda r: r['http_receipts'][-1].update(response_payload=None), 'unknown'),
        (lambda r: r['http_receipts'][-1]['response_payload'].append(dict(item_id=3, sku='a', name='A', qty=2)), 'fail')]:
        changed = copy.deepcopy(row)
        modification(changed)
        assert score(changed)[0] == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    self_test()
    if not args.input:
        print('7 fixture checks passed')
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    totals = collections.defaultdict(collections.Counter)
    reasons = collections.Counter()
    with open(args.input) as source, (out / 'audit_rows.jsonl').open('x') as dest:
        for line in source:
            row = json.loads(line)
            status, reason = score(row)
            group = row['arm'] + ('/clean' if row['condition'] == 'clean' else '/fault')
            totals[group][status] += 1
            totals[group]['original_success'] += bool(row['final_task_success'])
            totals[group]['success_after_audit'] += bool(row['final_task_success']) and status == 'pass'
            reasons[reason] += 1
            dest.write(json.dumps(dict(run_id=row['run_id'], group=group, status=status, reason=reason)) + '\n')
    summary = dict(groups=totals, reasons=reasons, scope='Recorded full cart items only; excludes price, inventory, orders and transient external side effects.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
