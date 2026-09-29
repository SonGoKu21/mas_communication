"""Freeze the pre-fault task selection and proposed cells, not execution eligibility."""
import argparse
import hashlib
import json
from pathlib import Path
from design import build_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.gate.read_text())
    if data.get('status') != 'passed' or data.get('product_count') != 6:
        raise ValueError('six-product HTTP gate required')
    tasks, seen = [], set()
    for task in data['tasks']:
        cluster = task['product_url']
        if cluster in seen:
            continue
        seen.add(cluster)
        tasks.append(dict(task, product_cluster=cluster))
    jobs = build_jobs(tasks)
    result = dict(status='http_admitted_pending_runtime_gate', tasks=tasks,
                  jobs=jobs, count=len(jobs),
                  selection_rule='first transition per product in successful gate order; before fault outcomes',
                  gate_sha256=hashlib.sha256(args.gate.read_bytes()).hexdigest(),
                  paid_calls=0, execution_eligible=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'design_manifest.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(tasks=len(tasks), jobs=len(jobs), execution_eligible=False)))


if __name__ == '__main__':
    main()
