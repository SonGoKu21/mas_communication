"""Recover allowlisted MAS files from local snapshots, retaining provenance."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

DEST = Path(__file__).resolve().parent
ROOT = DEST.parent
selected: dict[str, str] = {}


def tree(source: str, target: str) -> None:
    for path in sorted((ROOT / source).rglob('*.py')):
        if '__pycache__' not in path.parts:
            selected[str(Path(target) / path.relative_to(ROOT / source))] = str(path.relative_to(ROOT))


def take(source: str, target: str | None = None) -> None:
    selected[target or Path(source).name] = source


# These are reconstruction priorities, not a claim of a historical frozen commit.
for source in ['src/mas_faults', '.remote_mas_stage/src/mas_faults',
               'remote_staging/src/mas_faults', 'staging_admin/src/mas_faults',
               '.remote_edit/mas/src/mas_faults']:
    tree(source, 'src/mas_faults')
for name in ['webarena_topologies', 'topology_experiment_controls']:
    take(f'.remote_mas_stage/{name}.py', f'src/mas_faults/llm/{name}.py')
take('.remote_mas_stage/confirmation_fix/shopping_strict_evaluator.py', 'src/mas_faults/llm/shopping_strict_evaluator.py')
take('.remote_mas_stage/confirmation_fix/webarena_shopping_confirmation.py', 'src/mas_faults/webarena_shopping_confirmation.py')
take('staging_admin/src/mas_faults/llm_client.py', 'src/mas_faults/llm_client.py')
take('remote_patch/admin_filter_state/webarena_admin_controlled.py', 'src/mas_faults/webarena_admin_controlled.py')
take('remote_patch/admin_filter_state/webarena_admin_tools.py', 'src/mas_faults/webarena_admin_tools.py')
take('.remote-mas-edit/src/mas_faults/cross_benchmark_main_confirmation.py', 'src/mas_faults/cross_benchmark_main_confirmation.py')
take('.remote-mas-edit/src/mas_faults/cross_benchmark_main_matrix.py', 'src/mas_faults/cross_benchmark_main_matrix.py')
take('remote_patch/cross_benchmark_main_runner.py', 'src/mas_faults/cross_benchmark_main_runner.py')
tree('.remote_edit/mas/netem_bridge/src/mas_faults', 'src/mas_faults')
for name in ['run_webarena_architecture_rq2', 'run_webarena_admin_main_confirmation',
             'prepare_webarena_admin_repeated_confirmation', 'analyze_unified_fse_results']:
    take(f'.remote_edit/mas/{name}.py')
take('summarize_rq2_cases.py')
take('staging_admin/run_webarena_admin_controlled_fault_matrix.py')
take('remote_staging/run_webarena_reddit_official_admission.py')
take('.remote_mas_stage/run_webarena_reddit_topology_rq2.py')
take('remote_patch/run_webarena_reddit_stateful_rq1.py')
take('.remote_edit/mas/bottom_up_bridge/run_bottom_up_bridge_experiment.py')
take('.remote_edit/mas/netem_bridge/run_netem_task_bridge_experiment.py')
tree('staging_admin/scripts/smoke', 'scripts/smoke')
take('remote_patch/scripts/smoke/webarena_admin_browser_worker.py', 'scripts/smoke/webarena_admin_browser_worker.py')

for name in ['benchmark_trace_contract', 'benchmark_result_summary', 'webarena_shopping_real',
             'webarena_task_selection', 'webarena_clean_baseline_summary',
             'webarena_topologies', 'theagentcompany_real']:
    take(f'.remote_edit/mas/tests/test_{name}.py', f'tests/test_{name}.py')
for name in ['application_fault_matrix', 'propagation_evaluator', 'consequence_axes']:
    take(f'.remote_mas_stage/tests/test_{name}.py', f'tests/test_{name}.py')
for name in ['webarena_admin_main_evaluator', 'webarena_admin_main_matrix',
             'webarena_admin_topologies', 'webarena_admin_fault_matrix', 'llm_client']:
    take(f'staging_admin/tests/test_{name}.py', f'tests/test_{name}.py')
take('remote_patch/admin_filter_state/test_webarena_admin_tools.py', 'tests/test_webarena_admin_tools.py')
take('.remote_mas_stage/confirmation_fix/test_shopping_strict_evaluator.py', 'tests/test_shopping_strict_evaluator.py')
take('remote_staging/tests/test_deepseek_schedule.py', 'tests/test_deepseek_schedule.py')
take('.remote_edit/mas/bottom_up_bridge/tests/test_bottom_up_bridge.py', 'tests/test_bottom_up_bridge.py')
take('.remote_edit/mas/docs/内网镜像与部署复盘.md', 'docs/历史内网部署复盘.md')
take('.remote_edit/mas/requirements.txt', 'provenance/historical_requirements.txt')

records = []
for target, source in sorted(selected.items()):
    raw = (ROOT / source).read_bytes()
    content = raw.decode('utf-8')
    changes = []
    if source.endswith('llm_client.py') and target.startswith('src/'):
        content, count = re.subn(r'^DEFAULT_API_KEY\s*=\s*[^\n]+',
                                'DEFAULT_API_KEY = None', content, flags=re.MULTILINE)
        if count:
            changes.append('Removed hardcoded default API credential; environment only.')
    if re.search(r'sk-[A-Za-z0-9_-]{20,}', content):
        raise ValueError(f'Credential-like token detected: {source}; not copied')
    dest = DEST / target
    if dest.exists() and dest.read_text() != content:
        raise FileExistsError(f'Refusing to overwrite changed file: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    records.append({'target': target, 'source': source,
                    'source_sha256': hashlib.sha256(raw).hexdigest(),
                    'restored_sha256': hashlib.sha256(content.encode()).hexdigest(),
                    'changes': changes})

missing = {}
for target, source in selected.items():
    if not target.endswith('.py'):
        continue
    syntax = ast.parse((DEST / target).read_text(), filename=target)
    for node in ast.walk(syntax):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for module in names:
            if module.startswith('mas_faults.'):
                relative = Path('src') / module.replace('.', '/')
                if not ((DEST / relative).with_suffix('.py').is_file() or
                        (DEST / relative).is_dir()):
                    missing.setdefault(module, []).append(target)

manifest = {'status': 'reconstructed_from_partial_local_snapshots',
            'historical_commit_verified': False, 'files': records,
            'missing_internal_modules': missing}
(DEST / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'files': len(records), 'missing_internal_modules': missing}, indent=2))
