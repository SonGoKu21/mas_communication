"""Finite clean-to-fault continuation, no polling or automatic error retry."""
import json
import subprocess
import sys
from pathlib import Path

root = Path('/data3/hqn/mas/code/rq4_mechanism_1098_20260914')
out = Path('/data3/hqn/mas/results/rq4_flash1098_20260914_v1')
audit = out.with_name(out.name + '_clean144_audit')
cmd = [sys.executable, '-u', str(root / 'runner.py'),
       '--design', '/data3/hqn/mas/results/rq4_design1098_20260914_v1/design_manifest.json',
       '--isolation', '/data3/hqn/mas/results/rq4_four_cart_gate_20260914_v1/summary.json',
       '--output', str(out)]
checkpoint = json.loads(out.with_name(out.name + '_clean24_audit').joinpath('summary.json').read_text())
if checkpoint['completed'] != 24 or checkpoint['inflight'] or checkpoint['findings']:
    raise RuntimeError('first checkpoint not verified')
result = subprocess.run(cmd + ['--max-jobs', '120'], cwd=root)
if result.returncode not in (0, 2):
    raise SystemExit(result.returncode)
subprocess.run([sys.executable, '/data3/hqn/mas/code/mas_rq4_audit_20260914.py', str(out), str(audit)], check=True)
gate = json.loads((audit / 'summary.json').read_text())
if gate['clean'] != 144 or gate['completed'] != 144 or gate['inflight'] or gate['findings']:
    raise RuntimeError('clean barrier incomplete; leave records for scheduled resume')
print('CLEAN144 AUDITED; continuing fault and path diagnostics', flush=True)
raise SystemExit(subprocess.run(cmd, cwd=root).returncode)
