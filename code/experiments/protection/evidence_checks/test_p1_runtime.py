import asyncio
import tempfile
import unittest
from pathlib import Path
from legacy.tests.test_shopping_multimechanism import UnitClient,UnitExecutor,task,payload
from p1_runtime import run_trial,make_envelope

class P1RuntimeTests(unittest.TestCase):
    def run_case(self,strategy,condition,topology):
        executor=UnitExecutor();executor.begin_evaluation=lambda:None
        job=dict(arm='baseline' if strategy=='baseline' else 'semantic_only',p1_strategy=strategy,
                 topology=topology,condition=condition,planned_exposures=int(condition!='clean'),
                 recovery_path_exposed=False,path_design=None)
        donor=make_envelope({**payload(sku='OTHER'),'task_id':'other','product_id':'44'},
                            task_id='other',session_id='other',entity_id='other',version=1,
                            action_id='other',evidence_id='other',source='unit-fixture')
        with tempfile.TemporaryDirectory() as directory:
            return asyncio.run(run_trial({**task(),'product_url':'http://localhost/product'},job,
                                         executor,UnitClient(),cross_task_evidence=donor,
                                         ledger_path=Path(directory)/'ledger.sqlite'))

    def test_twenty_four_cells_keep_common_recovery_and_exposure(self):
        for strategy in ('baseline','check_only','always_readback','guarded_readback'):
            for condition in ('clean','valid_partial','contract_consistent_identity_corruption'):
                for topology in ('sequential','flat'):
                    with self.subTest(strategy=strategy,condition=condition,topology=topology):
                        row=self.run_case(strategy,condition,topology)
                        self.assertTrue(row['common_recovery_enabled'])
                        self.assertEqual(len(row['fault_events']),int(condition!='clean'))
                        readbacks=sum(e['readback_called'] for e in row['semantic_contract_events'])
                        if strategy in ('baseline','check_only'):self.assertEqual(readbacks,0)
                        if strategy=='always_readback':self.assertEqual(readbacks,1)
                        if strategy=='guarded_readback':self.assertLessEqual(readbacks,1)
                        self.assertFalse(row['rq4_mechanisms']['action'])
                        if condition=='clean':self.assertTrue(row['final_task_success'])

    def test_check_only_rejection_does_not_force_recovery(self):
        row=self.run_case('check_only','valid_partial','flat')
        self.assertFalse(row['semantic_contract_events'][0]['accepted'])
        self.assertFalse(any(e['kind']=='common_recovery_trigger' for e in row['detection_events']))

    def test_invalid_strategy_arm_combination_fails_before_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                asyncio.run(run_trial(task(),dict(arm='combined',p1_strategy='check_only',topology='flat'),
                            UnitExecutor(),UnitClient(),ledger_path=Path(directory)/'ledger.sqlite'))

if __name__=='__main__':unittest.main()
