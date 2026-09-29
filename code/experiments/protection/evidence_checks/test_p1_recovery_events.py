"""Offline transport tests for rejected readback accounting, never real runs."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from legacy.tests.test_shopping_multimechanism import UnitClient, UnitExecutor, task
from p1_runtime import run_trial
from p1_contract import ReceiverContract
from mas_faults.mitigation_protocol import BudgetExhausted


class ReadbackExecutor(UnitExecutor):
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
                observed['sku'] = 'WRONG-IDENTITY'
                observed['product_id'] = '999'
                observed['evidence'] = json.dumps({k: v for k, v in observed.items() if k != 'evidence'})
            elif self.defect == 'unavailable':
                return {'task_id': current_task['task_id'], 'status': 'observation_unavailable'}
        return observed


class ReceiverClient(UnitClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.judgment_calls = 0

    def complete(self, prompt, *, json_mode=False):
        result = json.loads(super().complete(prompt, json_mode=json_mode))
        inputs = json.loads(prompt.split('\nInput: ', 1)[1])
        if self.stage == 'final' and 'observation' in inputs:
            # The ordinary Flat peer is defective; the current receiver was clean.
            result['payload'] = {'task_id': inputs['task']['task_id']}
        if 'prior_judgment' in inputs:
            self.judgment_calls += 1
            if self.stage == 'common' and self.judgment_calls == 3:
                result['decision'] = 'reject'
        return json.dumps(result)


class P1RecoveryEventTests(unittest.TestCase):
    def test_readback_exception_retains_attempt_and_reraises_same_exception(self):
        for strategy in ('always_readback', 'guarded_readback'):
            with self.subTest(strategy=strategy):
                receiver = ReceiverContract(task(), session_id='session', action_id='action',
                                            entity_id='cart', minimum_version=2, strategy=strategy)
                failure = BudgetExhausted('offline test budget exhausted')

                def failed_readback():
                    raise failure

                with self.assertRaises(BudgetExhausted) as caught:
                    receiver.accept(None, failed_readback)
                self.assertIs(caught.exception, failure)
                self.assertEqual(receiver.readbacks, 1)
                self.assertEqual(len(receiver.events), 1)
                event = receiver.events[0]
                self.assertTrue(event['readback_called'])
                self.assertEqual(event['readbacks_used'], 1)
                self.assertTrue(event['before_issues'])
                self.assertIsNone(event['after_issues'])
                self.assertIsNone(event['accepted'])
                self.assertEqual(event['error_type'], 'BudgetExhausted')
                self.assertEqual(event['outcome'], 'readback_error')

    def run_case(self, stage, defect):
        strategy = 'check_only' if stage == 'common' else 'guarded_readback'
        topology = 'flat' if stage == 'final' else 'sequential'
        condition = 'valid_partial' if stage == 'current' else 'clean'
        executor = ReadbackExecutor(6 if stage == 'final' else 5, defect)
        job = dict(arm='semantic_only', p1_strategy=strategy, topology=topology,
                   condition=condition, planned_exposures=int(condition != 'clean'),
                   recovery_path_exposed=False, path_design=None)
        with tempfile.TemporaryDirectory() as directory:
            return asyncio.run(run_trial({**task(), 'product_url': 'http://localhost/product'},
                                         job, executor, ReceiverClient(stage),
                                         ledger_path=Path(directory) / 'ledger.sqlite'))

    def test_rejected_readbacks_are_neither_adopted_nor_verified(self):
        kinds = {'current': 'semantic_contract', 'final': 'semantic_final_contract',
                 'common': 'common_recovery'}
        for stage, kind in kinds.items():
            for defect in ('identity', 'unavailable'):
                with self.subTest(stage=stage, defect=defect):
                    row = self.run_case(stage, defect)
                    recovery = next(event for event in row['recovery_events'] if event['kind'] == kind)
                    self.assertTrue(recovery['receipt_indices'])
                    self.assertFalse(recovery['replacement_used'])
                    self.assertFalse(recovery['verified'])
                    self.assertIsNone(row['final_evidence'])
                    self.assertFalse(row['recovery_detected'])

    def test_accepted_readbacks_keep_adoption_and_verification(self):
        for stage in ('current', 'final', 'common'):
            with self.subTest(stage=stage):
                row = self.run_case(stage, None)
                self.assertTrue(row['recovery_events'])
                for event in row['recovery_events']:
                    self.assertTrue(event['replacement_used'])
                    self.assertTrue(event['verified'])
                self.assertTrue(row['final_task_success'])


if __name__ == '__main__':
    unittest.main()
