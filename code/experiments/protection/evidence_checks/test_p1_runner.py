"""Offline runner admission, scheduling and provenance regression tests."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT/'fixtures'
GATE = ARTIFACTS / 'http_gate/canonical_localhost_attempt'
SPEC = importlib.util.find_spec('p1_runner')
if SPEC is not None:
    import p1_runner as runner
else:
    runner = None


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'P1 runner admission has not been implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.design = json.loads((ARTIFACTS / 'jobs_proposed.json').read_text())
        self.design['tasks'] = json.loads((ARTIFACTS / 'tasks_recovered.json').read_text())['tasks']
        self.design['execution_eligible'] = True  # Ephemeral fixture, never a launch artifact.
        for name in ('summary.json', 'started.json'):
            (self.directory / name).write_bytes((GATE / name).read_bytes())
        self.gate = self.directory / 'summary.json'
        self.design['http_gate_sha256'] = hashlib.sha256(self.gate.read_bytes()).hexdigest()
        self.design['http_gate_started_sha256'] = hashlib.sha256(
            (self.directory / 'started.json').read_bytes()).hexdigest()

    def admitted(self):
        return runner.validate_admission(self.design, self.gate)

    def test_only_explicit_boolean_eligibility_can_reach_configuration(self):
        for value in (False, None, 1, 'true', {}):
            with self.subTest(value=value):
                self.design['execution_eligible'] = value
                with self.assertRaisesRegex(ValueError, 'execution_eligible'):
                    self.admitted()
        del self.design['execution_eligible']
        with self.assertRaisesRegex(ValueError, 'execution_eligible'):
            self.admitted()

    def test_gate_digest_and_started_digest_are_bound_before_admission(self):
        for field in ('http_gate_sha256', 'http_gate_started_sha256'):
            with self.subTest(field=field):
                previous = self.design[field]
                self.design[field] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'hash'):
                    self.admitted()
                self.design[field] = previous

    def test_rehashed_failed_or_incomplete_gate_is_still_rejected(self):
        original = json.loads(self.gate.read_text())
        for field, value in [('status', 'failed'), ('quantity_transition_passed', 5),
                ('four_lane_isolation_status', 'failed'), ('all_ten_carts_distinct', False),
                ('max_http_concurrency', 5), ('model_calls', 1), ('source_unchanged', False),
                ('all_responses_hashed', False)]:
            with self.subTest(field=field):
                self.gate.write_text(json.dumps(dict(original, **{field: value})))
                self.design['http_gate_sha256'] = hashlib.sha256(self.gate.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, 'HTTP gate'):
                    self.admitted()

    def test_adapter_hash_drift_rejects_even_a_rehashed_pass_summary(self):
        gate = json.loads(self.gate.read_text())
        key = next(iter(gate['source_hashes']))
        gate['source_hashes'][key] = '0' * 64
        self.gate.write_text(json.dumps(gate))
        self.design['http_gate_sha256'] = hashlib.sha256(self.gate.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'adapter'):
            self.admitted()

    def test_normalizes_only_origin_and_keeps_complete_four_policy_matrix(self):
        original = copy.deepcopy(self.design)
        design, jobs, provenance = self.admitted()
        self.assertEqual(self.design, original)
        self.assertEqual(len(jobs), 432)
        self.assertEqual(sum(j['condition'] == 'clean' for j in jobs), 144)
        self.assertEqual(len({j['job_key'] for j in jobs}), 432)
        for before, after in zip(original['tasks'], design['tasks']):
            for key in ('task_id', 'product_title', 'initial_quantity', 'quantity', 'task_family'):
                self.assertEqual(after[key], before[key])
            self.assertEqual(after['product_url'], before['product_url'].replace(
                'http://127.0.0.1:17770/', 'http://localhost:7770/', 1))
        self.assertTrue(all(j['product_cluster'].startswith('http://localhost:7770/') for j in jobs))
        self.assertEqual(provenance['summary_sha256'], original['http_gate_sha256'])
        self.assertEqual({j['p1_strategy'] for j in jobs},
                         {'baseline', 'check_only', 'always_readback', 'guarded_readback'})

    def test_substitution_of_product_title_quantity_or_path_is_rejected(self):
        original = copy.deepcopy(self.design['tasks'][0])
        for key, value in [('product_title', 'replacement product'), ('quantity', 4),
                           ('product_url', 'http://127.0.0.1:17770/replacement.html')]:
            with self.subTest(field=key):
                self.design['tasks'][0] = dict(original, **{key: value})
                with self.assertRaisesRegex(ValueError, 'task|product'):
                    self.admitted()

    def test_noncanonical_origin_and_mismatched_job_cluster_are_rejected(self):
        self.design['tasks'][0]['product_url'] = 'http://127.0.0.1:7770/product.html'
        with self.assertRaisesRegex(ValueError, 'origin'):
            self.admitted()
        self.design['tasks'][0] = json.loads((ARTIFACTS / 'tasks_recovered.json').read_text())['tasks'][0]
        self.design['jobs'][0]['product_cluster'] = self.design['tasks'][1]['product_cluster']
        with self.assertRaisesRegex(ValueError, 'cluster'):
            self.admitted()

    def test_incomplete_or_duplicate_matrix_is_never_admitted(self):
        self.design['jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'matrix'):
            self.admitted()

    def test_draft_cli_stops_before_credentials_output_or_workers(self):
        self.design['execution_eligible'] = False
        path = self.directory / 'draft.json'
        path.write_text(json.dumps(self.design))
        output = self.directory / 'must-not-exist'
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLM_')}
        result = subprocess.run([sys.executable, str(ROOT / 'p1_runner.py'),
            '--design', str(path), '--http-gate', str(self.gate), '--output', str(output)],
            capture_output=True, text=True, env=env, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('execution_eligible', result.stderr)
        self.assertFalse(output.exists())

    def test_pending_clean_jobs_precede_all_faults(self):
        _, jobs, _ = self.admitted()
        selected = runner.pending_after_clean_barrier(dict(pending=jobs, rows=[]), jobs)
        self.assertEqual(len(selected), 144)
        self.assertTrue(all(j['condition'] == 'clean' for j in selected))

    def test_exhausted_clean_attempts_block_fault_launch(self):
        _, jobs, _ = self.admitted()
        clean = [dict(j, final_task_success=True) for j in jobs if j['condition'] == 'clean']
        faults = [j for j in jobs if j['condition'] != 'clean']
        with self.assertRaisesRegex(RuntimeError, 'clean'):
            runner.pending_after_clean_barrier(dict(pending=faults, rows=clean[:-1]), jobs)

    def test_clean_failure_is_completed_without_selective_success_rerun(self):
        _, jobs, _ = self.admitted()
        clean = [dict(j, final_task_success=False) for j in jobs if j['condition'] == 'clean']
        faults = [j for j in jobs if j['condition'] != 'clean']
        self.assertEqual(runner.pending_after_clean_barrier(dict(pending=faults, rows=clean), jobs), faults)
        job = jobs[0]
        start = dict(job, attempt=1, attempt_id='offline-attempt', config_digest='offline-digest')
        runner.common.append_jsonl(self.directory / 'run_attempts.jsonl', start)
        runner.common.append_jsonl(self.directory / 'main_runs.jsonl',
                                   dict(start, run_id='offline-run', final_task_success=False))
        state = runner.common.reconcile_attempts(self.directory, jobs, 'offline-digest')
        self.assertNotIn(job['job_key'], {j['job_key'] for j in state['pending']})

    def test_all_four_policies_reuse_same_frozen_donor(self):
        design, jobs, _ = self.admitted()
        target = [j for j in jobs if j['task_id'] == design['tasks'][0]['task_id']
                  and j['topology'] == 'flat' and j['repeat_index'] == 2
                  and j['condition'] == 'contract_consistent_identity_corruption']
        task = design['tasks'][1]
        payload = dict(task_id=task['task_id'], product_title=task['product_title'],
            product_id='offline-product', sku='offline-sku', requested_quantity=task['quantity'],
            observed_quantity=task['quantity'], cart_verified=True, evidence='offline fixture')
        envelope = dict(task_id=task['task_id'], source='Worker', payload=payload, version=1,
            evidence_id='offline-evidence', session_id='offline-session', entity_id='offline-entity',
            action_id='offline-action')
        source = dict(task_id=task['task_id'], condition='clean', arm='baseline',
            topology='flat', repeat_index=2, final_task_success=True, environment_task_success=True,
            config_digest='offline-digest', run_id='offline-run', job_key='offline-job',
            source_evidence=envelope, environment_state=payload)
        donors = [runner.common.freeze_cross_task_source([source], job, design['tasks'],
                  self.directory, 'offline-digest') for job in target]
        self.assertEqual(len(donors), 4)
        self.assertTrue(all(donor == donors[0] for donor in donors))
        self.assertEqual(len(list((self.directory / 'cross_task_sources').glob('*.json'))), 1)

    def test_usage_summary_preserves_actual_request_parameters(self):
        class Client:
            prompt_tokens = 2
            completion_tokens = 3
            call_count = 1
            request_log = [dict(request_sent=True, status='success',
                request_parameters={'model': 'deepseek-flash', 'temperature': 0, 'max_tokens': 2048},
                request_timeout_seconds=90, total_timeout_seconds=120,
                unrelated_private_field='must disappear')]
        usage = runner.common.usage_since(Client(), dict(prompt_tokens=0, completion_tokens=0,
                                         call_count=0, log_length=0), complete=True)
        record = usage['model_requests'][0]
        self.assertEqual(record['request_parameters']['max_tokens'], 2048)
        self.assertEqual(record['request_timeout_seconds'], 90)
        self.assertEqual(record['total_timeout_seconds'], 120)
        self.assertNotIn('unrelated_private_field', record)

    def paused_main(self):
        design_path = self.directory / 'eligible-test-fixture.json'
        design_path.write_text(json.dumps(self.design))
        output = self.directory / 'offline-output'
        original_lock = runner.common.locked_workflow
        env = {'LLM_PROVIDER': 'deepseek', 'LLM_MODEL': 'deepseek-flash',
            'LLM_BASE_URL': 'https://api.deepseek.com', 'LLM_API_KEY': 'offline-fixture',
            'LLM_REQUEST_TIMEOUT_SECONDS': '90', 'LLM_TOTAL_REQUEST_TIMEOUT_SECONDS': '120',
            'LLM_MAX_TOKENS': '2048', 'LLM_DISABLE_THINKING': 'true'}
        with patch.dict(os.environ, env), patch.object(runner, 'is_deepseek_offpeak', return_value=False), \
                patch.object(runner.common, 'locked_workflow',
                    side_effect=lambda: original_lock(self.directory / 'local-test.lock')), \
                patch.object(runner, '_real_client', side_effect=AssertionError('model forbidden')), \
                redirect_stdout(StringIO()):
            code = runner.main(['--design', str(design_path), '--http-gate', str(self.gate),
                                '--output', str(output), '--max-jobs', '1'])
        return code, output

    def test_admitted_pause_freezes_full_parameters_without_starting_any_attempt(self):
        code, output = self.paused_main()
        self.assertEqual(code, 2)
        manifest = json.loads((output / 'matrix_manifest.json').read_text())
        config = manifest['config']
        self.assertEqual(config['model'], 'deepseek-flash')
        self.assertEqual(config['provider'], 'deepseek')
        self.assertEqual(config['base_url'], 'http://localhost:7770')
        self.assertEqual(config['lanes'], 4)
        self.assertEqual((config['budget']['http'], config['budget']['model'],
                          config['budget']['seconds']), (32, 16, 1200))
        self.assertEqual(config['inference_settings']['temperature'], 0)
        self.assertEqual(config['inference_settings']['max_tokens'], 2048)
        self.assertIs(config['inference_settings']['disable_thinking'], True)
        self.assertEqual(config['max_attempts_per_job'], 2)
        self.assertEqual(len(config['jobs']), 432)
        self.assertEqual(manifest['config_digest'], runner.common.matrix.config_digest(config))
        self.assertFalse((output / 'run_attempts.jsonl').exists())
        for name, digest in config['source_hashes'].items():
            self.assertEqual(hashlib.sha256((output / 'source_snapshot' / name).read_bytes()).hexdigest(), digest)

    def test_corrupt_source_snapshot_blocks_resume_before_any_model_attempt(self):
        _, output = self.paused_main()
        snapshot = output / 'source_snapshot/p1_runner.py'
        snapshot.write_text(snapshot.read_text() + '\n# corrupt offline fixture\n')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.paused_main()
        self.assertFalse((output / 'run_attempts.jsonl').exists())

    def test_unmanifested_existing_results_are_never_adopted_as_p1(self):
        output = self.directory / 'offline-output'
        output.mkdir()
        (output / 'legacy-results.json').write_text('{"preserve": true}')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.paused_main()
        self.assertFalse((output / 'matrix_manifest.json').exists())
        self.assertEqual((output / 'legacy-results.json').read_text(), '{"preserve": true}')

    def test_frozen_schedule_preserves_108_complete_blocks_and_all_job_fields(self):
        original = copy.deepcopy(self.design['jobs'])
        result = runner.balanced_schedule(original)
        self.assertEqual(original, self.design['jobs'])
        self.assertEqual(len(result), 432)
        self.assertEqual({j['job_key']: j for j in result}, {j['job_key']: j for j in original})
        blocks = [result[i:i + 4] for i in range(0, len(result), 4)]
        self.assertEqual(len(blocks), 108)
        for block in blocks:
            self.assertEqual(len({j['pair_key'] for j in block}), 1)
            self.assertEqual({j['variant'] for j in block},
                             {'baseline', 'check_only', 'always_readback', 'guarded_readback'})
        self.assertTrue(all(j['condition'] == 'clean' for j in result[:144]))
        self.assertTrue(all(j['condition'] != 'clean' for j in result[144:]))

    def test_frozen_schedule_balances_each_strategy_in_each_phase_position(self):
        result = runner.balanced_schedule(self.design['jobs'])
        for phase, expected in ((result[:144], 9), (result[144:], 18)):
            for position in range(4):
                for strategy in ('baseline', 'check_only', 'always_readback', 'guarded_readback'):
                    self.assertEqual(sum(j['variant'] == strategy for j in phase[position::4]), expected)

    def test_schedule_seed_is_deterministic_and_independent_of_input_order(self):
        jobs = self.design['jobs']
        first = runner.balanced_schedule(jobs, seed=20260927)
        self.assertEqual(first, runner.balanced_schedule(jobs, seed=20260927))
        self.assertEqual(first, runner.balanced_schedule(list(reversed(jobs)), seed=20260927))
        self.assertNotEqual(first, runner.balanced_schedule(jobs, seed=20260928))

    def test_broken_pair_block_is_rejected_before_scheduling(self):
        self.design['jobs'][1]['pair_key'] = 'unpaired'
        with self.assertRaisesRegex(ValueError, 'block'):
            runner.balanced_schedule(self.design['jobs'])


if __name__ == '__main__':
    unittest.main()
