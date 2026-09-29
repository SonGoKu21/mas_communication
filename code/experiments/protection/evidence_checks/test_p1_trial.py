import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from legacy.tests.test_shopping_multimechanism import UnitClient,UnitExecutor,task
import p1_trial

class CountedExecutor(UnitExecutor):
    def __init__(self):
        super().__init__();self.transport_calls=0
    def _request(self,*args,**kwargs):self.transport_calls+=1
    def reobserve_cart(self,t):
        self._request('GET','product');self._request('GET','cart')
        return super().reobserve_cart(t)

class WholeWorkflowTests(unittest.TestCase):
    def job(self,strategy='always_readback'):
        return dict(arm='baseline' if strategy=='baseline' else 'semantic_only',p1_strategy=strategy,
                    topology='sequential',condition='clean',planned_exposures=0,
                    recovery_path_exposed=False,path_design=None)

    def test_real_trial_binds_p1_and_accounts_all_calls(self):
        client=UnitClient();executor=CountedExecutor();original=executor._request
        with tempfile.TemporaryDirectory() as directory:
            row=asyncio.run(p1_trial.run(task(),self.job(),executor,client,ledger_path=Path(directory)/'a.sqlite'))
        self.assertEqual(sum(e['readback_called'] for e in row['semantic_contract_events']),1)
        self.assertEqual(row['whole_budget']['limits'],{'http':32,'model':16})
        self.assertEqual(row['whole_budget']['seconds_limit'],1200)
        self.assertEqual(row['whole_budget']['used']['model'],len(client.request_log))
        self.assertEqual(row['whole_budget']['used']['http']+row['whole_budget']['evaluation_http_calls'],executor.transport_calls)
        self.assertEqual(executor._request,original)

    def test_http_budget_blocks_33rd_request(self):
        executor=CountedExecutor()
        async def exhaust(task,job,executor,client,**kwargs):
            for _ in range(33):executor._request('GET','unit-only')
            raise AssertionError('budget failed')
        with tempfile.TemporaryDirectory() as directory,patch.object(p1_trial,'run_trial',exhaust):
            row=asyncio.run(p1_trial.run(task(),self.job(),executor,UnitClient(),ledger_path=Path(directory)/'a.sqlite'))
        self.assertEqual(row['termination'],'budget_exhausted')
        self.assertEqual(executor.transport_calls,32)
        self.assertFalse(row['final_task_success'])

    def test_model_budget_blocks_17th_request(self):
        class Client(UnitClient):
            def complete(self,*args,**kwargs):self.request_log.append({});return '{}'
        client=Client()
        async def exhaust(task,job,executor,client,**kwargs):
            for _ in range(17):client.complete('unit-only')
            raise AssertionError('budget failed')
        with tempfile.TemporaryDirectory() as directory,patch.object(p1_trial,'run_trial',exhaust):
            row=asyncio.run(p1_trial.run(task(),self.job(),CountedExecutor(),client,ledger_path=Path(directory)/'a.sqlite'))
        self.assertEqual(row['termination'],'budget_exhausted')
        self.assertEqual(len(client.request_log),16)
        self.assertFalse(row['final_task_success'])

if __name__=='__main__':unittest.main()
