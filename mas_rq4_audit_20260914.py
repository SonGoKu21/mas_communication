"""Read-only RQ4 checkpoint audit; output must be a new directory."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    def rows(name):
        p = root / name
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
    manifest = json.loads((root / 'matrix_manifest.json').read_text())
    config, digest = manifest['config'], manifest['config_digest']
    starts, done, errors = rows('run_attempts.jsonl'), rows('main_runs.jsonl'), rows('run_errors.jsonl')
    findings = []
    jobs = {j['job_key']: j for j in config['jobs']}
    if len(jobs) != 1098:
        findings.append('matrix_count')
    for name, records, field in [('starts', starts, 'attempt_id'), ('done', done, 'job_key'), ('run_ids', done, 'run_id')]:
        values = [r.get(field) for r in records]
        if None in values or len(set(values)) != len(values):
            findings.append(name + '_uniqueness')
    start_map = {s['attempt_id']: s for s in starts}
    terminal = set()
    for row in done + errors:
        aid = row.get('attempt_id')
        if aid in terminal:
            findings.append('duplicate_terminal')
        terminal.add(aid)
        start = start_map.get(aid)
        if start is None or any(row.get(k) != v for k, v in start.items()):
            findings.append('terminal_binding')
    for s in starts:
        if s.get('config_digest') != digest or s.get('attempt') not in (1, 2):
            findings.append('attempt_config')
        if any(s.get(k) != v for k, v in jobs[s['job_key']].items()):
            findings.append('job_binding')
    carts = []
    for row in done:
        if row.get('model') != 'deepseek-flash' or row.get('provider') != 'deepseek':
            findings.append('model_provider')
        if row['condition'] == 'clean' and row.get('fault_events'):
            findings.append('clean_fault_applied')
        if len(row.get('fault_events', [])) > row['planned_exposures']:
            findings.append('exposure_budget')
        if row.get('final_task_success') and row.get('recorded_cart_audit', {}).get('status') != 'pass':
            findings.append('success_without_cart_audit')
        if not row.get('common_recovery_enabled', row.get('termination') == 'budget_exhausted'):
            findings.append('common_recovery_missing')
        hashes = {r.get('guest_cart_id_sha256') for r in row.get('http_receipts', []) if r.get('guest_cart_id_sha256')}
        if len(hashes) != 1:
            findings.append('cart_binding')
        carts.extend(hashes)
    if len(carts) != len(set(carts)):
        findings.append('cart_reused_between_runs')
    for name, expected in config['source_hashes'].items():
        if hashlib.sha256((root / 'source_snapshot' / name).read_bytes()).hexdigest() != expected:
            findings.append('source_snapshot')
    return dict(completed=len(done), success=sum(r['final_task_success'] for r in done),
                clean=sum(r['condition'] == 'clean' for r in done), errors=len(errors),
                inflight=len(set(start_map) - terminal), findings=sorted(set(findings)),
                known_total_tokens=sum(r.get('known_total_tokens', 0) for r in done + errors),
                unknown_usage=sum(r.get('total_tokens') is None for r in done + errors))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    result = audit(a.results)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(bool(result['findings']))
