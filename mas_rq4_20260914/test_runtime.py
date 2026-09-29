import asyncio
import tempfile
import unittest
from pathlib import Path
from legacy.tests.test_shopping_multimechanism import UnitClient, UnitExecutor, task, payload
from runtime import run_trial, make_envelope
from design import ARMS, FAULTS


class RuntimeTests(unittest.TestCase):
    def run_case(self, arm, condition, topology='sequential', count=1, path=None):
        executor = UnitExecutor()
        executor.begin_evaluation = lambda: None
        t = {**task(), 'product_url': 'http://localhost/product'}
        job = dict(arm=arm, topology=topology, condition=condition,
                   planned_exposures=0 if condition == 'clean' else count,
                   recovery_path_exposed=count == 2, path_design=path)
        source = make_envelope({**payload(sku='OTHER'), 'task_id': 'other', 'product_id': '44'},
                               task_id='other', session_id='other', entity_id='other', version=1,
                               action_id='other', evidence_id='other', source='unit-fixture')
        with tempfile.TemporaryDirectory() as d:
            return asyncio.run(run_trial(t, job, executor, UnitClient(),
                                         cross_task_evidence=source, ledger_path=Path(d)/'ledger.sqlite'))

    def test_all_factorial_cells_real_autogen_unit_transports(self):
        for arm in ARMS:
            for condition in ('clean',) + FAULTS:
                for topology in ('sequential', 'flat'):
                    with self.subTest(arm=arm, condition=condition, topology=topology):
                        r = self.run_case(arm, condition, topology)
                        self.assertTrue(r['common_recovery_enabled'])
                        self.assertEqual(len(r['fault_events']), int(condition != 'clean'))
                        if condition == 'clean':
                            self.assertTrue(r['final_task_success'])
                        self.assertFalse(any(e['kind'] == 'independent_observation' for e in r['recovery_events']))

    def test_two_deliveries_expose_recovery(self):
        r = self.run_case('action_only', 'request_non_delivery', count=2)
        self.assertEqual(len(r['fault_events']), 2)
        self.assertTrue(r['final_task_success'])
        self.assertTrue(r['fault_events'][1]['recovery'])

    def test_semantic_recovery_is_not_implicitly_correct(self):
        r = self.run_case('semantic_only', 'contract_consistent_identity_corruption', count=2)
        self.assertEqual(len(r['fault_events']), 2)
        self.assertFalse(r['semantic_contract_events'][0]['accepted'])
        self.assertTrue(r['final_task_success'])

    def test_three_path_designs(self):
        for path in ('single_path', 'duplicate_forwarding', 'independent_observation'):
            r = self.run_case('baseline', 'contract_consistent_identity_corruption', path=path)
            self.assertEqual(len(r['fault_events']), 1)
            self.assertEqual(any(e['role'] == 'EvidencePeer' for e in r['events']), path != 'single_path')

    def test_exception_preserves_injection_trace(self):
        from mas_faults.mitigation_protocol import BudgetExhausted
        class Limited(UnitClient):
            def complete(self, prompt, **kwargs):
                if len(self.request_log) == 5:
                    raise BudgetExhausted('unit budget')
                return super().complete(prompt, **kwargs)
        executor = UnitExecutor()
        executor.begin_evaluation = lambda: None
        job = dict(arm='baseline', topology='sequential', condition='valid_partial',
                   planned_exposures=1, recovery_path_exposed=False)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(BudgetExhausted) as raised:
                asyncio.run(run_trial(task(), job, executor, Limited(), ledger_path=Path(d)/'x.sqlite'))
            self.assertEqual(len(raised.exception.partial_trial['fault_events']), 1)


if __name__ == '__main__':
    unittest.main()
