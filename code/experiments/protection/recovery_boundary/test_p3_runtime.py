"""Real AutoGen dispatch with in-memory transports; all sockets are forbidden."""
import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legacy.tests.test_shopping_multimechanism import UnitClient, UnitExecutor, task
from p3_runtime import run_trial
from p3_trial import run

LEGACY = Path(__file__).resolve().parent/'reference'
DONOR = {'task_id': 'other', 'evidence_id': 'frozen-donor',
         'payload': {'sku': 'DONOR-SKU', 'product_id': '99'}}


class MockExecutor(UnitExecutor):
    def _request(self, *args, **kwargs):
        return None

    def begin_evaluation(self):
        pass

    def reobserve_cart(self, current_task):
        for _ in range(2):
            self._request('GET', 'offline://cart')
        return super().reobserve_cart(current_task)


class ReadbackExecutor(MockExecutor):
    def __init__(self, first_readback, defect):
        super().__init__()
        self.observation_calls = 0
        self.first_readback = first_readback
        self.defect = defect

    def begin_evaluation(self):
        self.defect = None

    def reobserve_cart(self, current_task):
        self.observation_calls += 1
        observed = super().reobserve_cart(current_task)
        if self.observation_calls >= self.first_readback:
            if self.defect == 'identity':
                observed['sku'], observed['product_id'] = 'WRONG', '999'
                observed['evidence'] = json.dumps({k: v for k, v in observed.items() if k != 'evidence'})
            elif self.defect == 'unavailable':
                return {'task_id': current_task['task_id'], 'status': 'observation_unavailable'}
        return observed


class ReceiverClient(UnitClient):
    def __init__(self, stage):
        super().__init__()
        self.stage, self.judgment_calls = stage, 0

    def complete(self, prompt, *, json_mode=False):
        value = json.loads(super().complete(prompt, json_mode=json_mode))
        inputs = json.loads(prompt.split('\nInput: ', 1)[1])
        if self.stage == 'final' and 'observation' in inputs:
            value['payload'] = {'task_id': inputs['task']['task_id']}
        if 'prior_judgment' in inputs:
            self.judgment_calls += 1
            if self.stage == 'common' and self.judgment_calls == 3:
                value['decision'] = 'reject'
        return json.dumps(value)


def job(scheme='clean', arm='combined', topology='sequential'):
    return dict(exposure_scheme=scheme, arm=arm, topology=topology, repeat_index=1,
                execution_eligible=False)


def load_legacy_runtime():
    """Load immutable source under private names without editing its imports."""
    replacements = {}
    for name in ('design', 'contract', 'exposure'):
        spec = importlib.util.spec_from_file_location('_p3_reference_' + name, LEGACY / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        replacements[name] = module
    with patch.dict(sys.modules, replacements):
        spec = importlib.util.spec_from_file_location('_p3_reference_runtime', LEGACY / 'runtime.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


class P3RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('network forbidden in P3 tests'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def case(self, scheme='clean', arm='combined', topology='sequential', *,
             executor=None, client=None, runner=run_trial, current_task=None, selected_job=None):
        with tempfile.TemporaryDirectory() as directory:
            return asyncio.run(runner(current_task or task(), selected_job or job(scheme, arm, topology),
                                      executor or MockExecutor(), client or UnitClient(),
                                      cross_task_evidence=copy.deepcopy(DONOR),
                                      ledger_path=Path(directory) / 'ledger.sqlite'))

    def test_all_sixteen_scope_arm_topology_cells_use_real_autogen(self):
        for scheme in ('clean', 'single_handoff', 'persistent_handoff', 'shared_workflow_evidence'):
            for arm in ('baseline', 'combined'):
                for topology in ('sequential', 'flat'):
                    with self.subTest(scheme=scheme, arm=arm, topology=topology):
                        row = self.case(scheme, arm, topology)
                        self.assertFalse(row['execution_eligible'])
                        self.assertTrue(row['events'])
                        self.assertEqual(row['rq4_mechanisms'], {'action': arm == 'combined', 'semantic': arm == 'combined'})
                        self.assertEqual(row['environment_state']['sku'], 'SKU')
                        evaluation = row['exposure_deliveries'][-1]
                        self.assertEqual(evaluation['path'], 'evaluation_readback')
                        self.assertFalse(evaluation['damaged'])
                        self.assertEqual(evaluation['before_sha256'], evaluation['after_sha256'])
                        self.assertTrue(row['common_recovery_enabled'])
                        for event in row['events']:
                            encoded = json.dumps(event.get('input'))
                            for forbidden in ('exposure_scheme', 'execution_eligible', 'frozen-donor', 'fault_events'):
                                self.assertNotIn(forbidden, encoded)

    def test_persistent_third_delivery_remains_corrupted(self):
        row = self.case('persistent_handoff')
        self.assertEqual([e['path'] for e in row['fault_events']],
                         ['initial_evidence', 'recovery_evidence', 'recovery_evidence'])
        self.assertEqual([e['kind'] for e in row['recovery_events']], ['semantic_contract', 'common_recovery'])
        self.assertTrue(all(e['after']['payload']['sku'] == 'DONOR-SKU' for e in row['recovery_events']))
        self.assertTrue(all(not e['verified'] and not e['replacement_used'] for e in row['recovery_events']))
        self.assertFalse(row['final_task_success'])
        self.assertEqual(row['final_verdict']['decision'], 'reject')

    def test_flat_corruption_precedes_peer_consumption_and_eval_is_clean(self):
        for scheme, expected in [('persistent_handoff', 'SKU'), ('shared_workflow_evidence', 'DONOR-SKU')]:
            for arm in ('baseline', 'combined'):
                with self.subTest(scheme=scheme, arm=arm):
                    row = self.case(scheme, arm, 'flat')
                    peer = next(e for e in row['events'] if e['role'] == 'EvidencePeer')
                    self.assertEqual(peer['input']['observation']['payload']['sku'], expected)
                    self.assertEqual(row['environment_state']['sku'], 'SKU')

    def test_single_handoff_preserves_original_recovery_behavior(self):
        legacy = load_legacy_runtime()
        for topology in ('sequential', 'flat'):
            row = self.case('single_handoff', topology=topology)
            self.assertEqual(len(row['fault_events']), 1)
            self.assertEqual(row['recovery_events'][0]['after']['payload']['sku'], 'SKU')
            self.assertTrue(row['final_task_success'])
            for arm in ('baseline', 'combined'):
                import uuid
                import p3_runtime
                from types import SimpleNamespace
                fixed_uuid = SimpleNamespace(uuid4=lambda: uuid.UUID(int=123))
                old_job = dict(arm=arm, topology=topology, condition='contract_consistent_identity_corruption',
                               planned_exposures=1, recovery_path_exposed=False, path_design=None)
                with patch.object(legacy, 'uuid', fixed_uuid), patch.object(p3_runtime, 'uuid', fixed_uuid):
                    old = self.case(runner=legacy.run_trial, selected_job=old_job)
                    new = self.case('single_handoff', arm, topology)
                for field in ('events', 'judgments', 'final_evidence', 'final_verdict', 'budget',
                              'recovery_events', 'http_receipts', 'final_task_success'):
                    self.assertEqual(new[field], old[field], field)

    def test_same_donor_and_unexposed_action_boundaries_are_auditable(self):
        from mas_faults.multimechanism_matrix import config_digest
        for arm in ('baseline', 'combined'):
            row = self.case('shared_workflow_evidence', arm, 'flat')
            self.assertEqual(row['action_ledger_state']['execution_count'], 1)
            for event in row['exposure_deliveries']:
                if event['damaged']:
                    self.assertEqual(event['source_sha256'], config_digest(DONOR))
                    self.assertEqual(event['source_evidence_id'], DONOR['evidence_id'])
                    self.assertNotEqual(event['before_sha256'], event['after_sha256'])
                else:
                    self.assertEqual(event['before_sha256'], event['after_sha256'])
            for boundary in ('action_request', 'action_ack', 'observation_handoff', 'judgment_handoff'):
                audit = next(e for e in row['exposure_deliveries'] if e['path'] == boundary)
                self.assertFalse(audit['eligible'])
                self.assertFalse(audit['damaged'])

    def test_common_recovery_is_available_to_baseline_and_combined(self):
        for arm in ('baseline', 'combined'):
            row = self.case(arm=arm, client=ReceiverClient('common'))
            self.assertEqual([e['kind'] for e in row['recovery_events']], ['common_recovery'])
            self.assertTrue(row['final_task_success'])
            self.assertTrue(row['recovery_events'][0]['common'])

    def test_model_error_propagates_without_retry_and_retains_partial_state(self):
        class FailingClient(UnitClient):
            def complete(self, prompt, *, json_mode=False):
                if len(self.request_log) == 5:
                    raise RuntimeError('synthetic model transport failure')
                return super().complete(prompt, json_mode=json_mode)
        client = FailingClient()
        with self.assertRaises(RuntimeError) as caught:
            self.case('persistent_handoff', client=client, runner=run)
        self.assertEqual(len(client.request_log), 5)
        self.assertEqual(str(caught.exception), 'synthetic model transport failure')
        self.assertTrue(caught.exception.partial_trial['fault_events'])

    def test_model_budget_and_elapsed_budget_are_enforced_before_transport(self):
        import p3_trial
        from mas_faults.recovery_diagnostic import WholeBudget

        def almost_spent_budget(**kwargs):
            budget = WholeBudget(**kwargs)
            budget.used['model'] = 15
            return budget

        client = UnitClient()
        with patch.object(p3_trial, 'WholeBudget', almost_spent_budget):
            row = self.case(runner=run, client=client)
        self.assertEqual(row['termination'], 'budget_exhausted')
        self.assertEqual(row['whole_budget']['used']['model'], 16)
        self.assertEqual(len(client.request_log), 1)
        self.assertEqual(row['whole_budget']['evaluation_http_calls'], 0)

        def expired_budget(**kwargs):
            budget = WholeBudget(**kwargs)
            budget.started -= budget.seconds + 1
            return budget

        client = UnitClient()
        with patch.object(p3_trial, 'WholeBudget', expired_budget):
            row = self.case(runner=run, client=client)
        self.assertEqual(row['termination'], 'budget_exhausted')
        self.assertFalse(client.request_log)

    def test_final_and_common_rejected_readback_flags(self):
        for stage, kind, first in [('final', 'semantic_final_contract', 6), ('common', 'common_recovery', 5)]:
            for defect in ('identity', 'unavailable', None):
                with self.subTest(stage=stage, defect=defect):
                    row = self.case(topology='flat' if stage == 'final' else 'sequential',
                                    client=ReceiverClient(stage), executor=ReadbackExecutor(first, defect))
                    event = next(e for e in row['recovery_events'] if e['kind'] == kind)
                    self.assertTrue(event['receipt_indices'])
                    self.assertEqual(event['verified'], defect is None)
                    self.assertEqual(event['replacement_used'], defect is None)
                    self.assertEqual(row['final_task_success'], defect is None)

    def test_accounting_regression_fixture_exposes_original_defect(self):
        legacy = load_legacy_runtime()
        for stage, kind, first in [('final', 'semantic_final_contract', 6), ('common', 'common_recovery', 5)]:
            topology = 'flat' if stage == 'final' else 'sequential'
            old_job = dict(arm='combined', topology=topology, condition='clean', planned_exposures=0,
                           recovery_path_exposed=False, path_design=None)
            old = self.case(runner=legacy.run_trial, selected_job=old_job,
                            executor=ReadbackExecutor(first, 'identity'), client=ReceiverClient(stage))
            old_event = next(e for e in old['recovery_events'] if e['kind'] == kind)
            self.assertTrue(old_event['verified'])
            self.assertTrue(old_event['replacement_used'])
            self.assertIsNone(old['final_evidence'])
            new = self.case(topology=topology, executor=ReadbackExecutor(first, 'identity'), client=ReceiverClient(stage))
            new_event = next(e for e in new['recovery_events'] if e['kind'] == kind)
            self.assertFalse(new_event['verified'])
            self.assertFalse(new_event['replacement_used'])

    def test_clean_matches_legacy_events_prompts_action_and_budgets(self):
        legacy = load_legacy_runtime()
        for arm in ('baseline', 'combined'):
            for topology in ('sequential', 'flat'):
                for target in (2, 1):
                    with self.subTest(arm=arm, topology=topology, target=target):
                        current_task = {**task(), 'quantity': target, 'initial_quantity': 1 if target == 2 else 3}
                        old_job = dict(arm=arm, topology=topology, condition='clean', planned_exposures=0,
                                       recovery_path_exposed=False, path_design=None)
                        import uuid
                        import p3_runtime
                        from types import SimpleNamespace
                        fixed_uuid = SimpleNamespace(uuid4=lambda: uuid.UUID(int=123))
                        with patch.object(legacy, 'uuid', fixed_uuid), patch.object(p3_runtime, 'uuid', fixed_uuid):
                            old = self.case(runner=legacy.run_trial, current_task=current_task, selected_job=old_job)
                            new = self.case(arm=arm, topology=topology, current_task=current_task)
                        for field in ('events', 'judgments', 'final_evidence', 'final_verdict', 'environment_state',
                                      'budget', 'rq4_mechanisms', 'semantic_contract_events', 'recovery_events',
                                      'http_receipts', 'evaluation_receipt_indices', 'llm_request_log',
                                      'action_contract_valid', 'action_outcome_unknown', 'final_task_success'):
                            self.assertEqual(new[field], old[field], field)
                        self.assertEqual(new['action_ledger_state']['execution_count'], old['action_ledger_state']['execution_count'])

    def test_whole_budget_unchanged_and_evaluation_separate(self):
        for scheme in ('clean', 'persistent_handoff', 'shared_workflow_evidence'):
            row = self.case(scheme, topology='flat', runner=run)
            budget = row['whole_budget']
            self.assertEqual(budget['limits'], {'http': 32, 'model': 16})
            self.assertEqual(budget['seconds_limit'], 1200)
            self.assertEqual(budget['evaluation_http_calls'], 2)
            self.assertEqual(budget['used']['http'] + 2, len(row['http_receipts']))
            self.assertEqual(budget['used']['model'], len(row['llm_request_log']))

    def test_http_budget_exhaustion_retains_partial_exposures(self):
        class ExpensiveExecutor(MockExecutor):
            def reobserve_cart(self, current_task):
                for _ in range(5):
                    self._request('GET', 'offline://cart')
                return super().reobserve_cart(current_task)
        row = self.case('persistent_handoff', runner=run, executor=ExpensiveExecutor())
        self.assertEqual(row['termination'], 'budget_exhausted')
        self.assertFalse(row['execution_eligible'])
        self.assertFalse(row['final_task_success'])
        self.assertTrue(row['fault_events'])
        self.assertTrue(row['partial_trial']['events'])
        self.assertEqual(row['whole_budget']['used']['http'], 32)
        attempt = row['semantic_contract_events'][-1]
        self.assertTrue(attempt['readback_called'])
        self.assertEqual(attempt['readbacks_used'], 1)
        self.assertIsNone(attempt['accepted'])
        self.assertIsNone(attempt['after_issues'])
        self.assertEqual(attempt['outcome'], 'readback_error')
        self.assertEqual(attempt['error_type'], 'LimitReached')

    def test_callback_errors_preserve_attempt_and_original_exception(self):
        from p3_contract import ReceiverContract
        from mas_faults.mitigation_protocol import BudgetExhausted
        for failure in (BudgetExhausted('local budget'), RuntimeError('local transport')):
            with self.subTest(error=type(failure).__name__):
                receiver = ReceiverContract(task(), session_id='s', action_id='a', entity_id='e', minimum_version=2)

                def readback():
                    raise failure

                with self.assertRaises(type(failure)) as caught:
                    receiver.accept(None, readback)
                self.assertIs(caught.exception, failure)
                self.assertEqual(receiver.readbacks, 1)
                self.assertEqual(len(receiver.events), 1)
                event = receiver.events[0]
                self.assertTrue(event['readback_called'])
                self.assertTrue(event['before_issues'])
                self.assertIsNone(event['accepted'])
                self.assertIsNone(event['after_issues'])
                self.assertEqual(event['outcome'], 'readback_error')
                self.assertEqual(event['error_type'], type(failure).__name__)
                self.assertEqual(receiver.minimum_version, 2)

    def test_design_contains_only_preserved_mechanisms_no_legacy_builder(self):
        import inspect
        import p3_design
        self.assertFalse(hasattr(p3_design, 'build_jobs'))
        legacy = load_legacy_runtime()
        self.assertEqual(inspect.getsource(p3_design.mechanisms), inspect.getsource(legacy.mechanisms))

    def test_rejects_execution_eligibility_and_p1_strategy_before_calls(self):
        for change in ({'execution_eligible': True}, {'p1_strategy': 'check_only'},
                       {'path_design': 'duplicate_forwarding'}, {'arm': 'semantic_only'},
                       {'exposure_scheme': 'unknown'}):
            client = UnitClient()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.case(client=client, selected_job={**job(), **change})
            self.assertFalse(client.request_log)


if __name__ == '__main__':
    unittest.main()
