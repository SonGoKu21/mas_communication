"""Exercise the real historical interceptor with the isolated adapter."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest
import test_admin_semantic

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'server_source'/'src'))
from mas_faults.webarena_admin_main_matrix import MainCommunicationInterceptor, MainConditionCell, CONDITION_BY_NAME
try:
    import adapter
except ModuleNotFoundError:
    adapter=None

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(adapter,'isolated adapter implementation missing')
        helper=test_admin_semantic.SemanticTests();self.entry=helper.entry()

    def test_real_base_batch_and_registry_are_preserved(self):
        registry=deepcopy(CONDITION_BY_NAME)
        cells=adapter.make_conditions(MainConditionCell)
        cls=adapter.make_interceptor_class(MainCommunicationInterceptor,self.entry)
        interceptor=cls(cells['visible_table_substitution_step3'])
        original={'visible_evidence':[['fresh']], 'url':'current'}
        result=interceptor.intercept(3,original,context={'eligible':True})
        self.assertTrue(result.fault_applied)
        self.assertEqual(result.fault_id,'A6')
        self.assertEqual(result.delivery_count,1)
        self.assertEqual(result.original_message,original)
        self.assertEqual(result.delivered_messages[0]['visible_evidence'],self.entry['raw_table'])
        self.assertEqual(CONDITION_BY_NAME,registry)
        self.assertFalse(interceptor.intercept(3,original,context={'eligible':True}).fault_applied)

    def test_noop_batch_records_attempt_without_applied(self):
        cells=adapter.make_conditions(MainConditionCell)
        cls=adapter.make_interceptor_class(MainCommunicationInterceptor,self.entry)
        interceptor=cls(cells['visible_table_substitution_step4'])
        result=interceptor.intercept(4,{'payload':{'visible_evidence':self.entry['target_projected_table']}},context={'eligible':True})
        self.assertFalse(result.fault_applied)
        self.assertEqual(result.observed_runtime_effect,'ineffective_visible_table_substitution')
        self.assertTrue(result.fault_parameters['injection_opportunity_consumed'])
        self.assertEqual(result.fault_id,'none')

    def test_i4_empty_current_table_can_be_replaced(self):
        cells=adapter.make_conditions(MainConditionCell)
        cls=adapter.make_interceptor_class(MainCommunicationInterceptor,self.entry)
        got=cls(cells['visible_table_substitution_step4']).intercept(4,{'payload':{'visible_evidence':[]}},context={'eligible':True})
        self.assertTrue(got.fault_applied)

    def test_changed_runtime_projection_is_rejected_before_execution(self):
        namespace={'MainCommunicationInterceptor':MainCommunicationInterceptor}
        exec('async def run_admin_confirmation_task(*args, **kwargs):\n    raise AssertionError("must not execute")\n',namespace)
        module=types.SimpleNamespace(MainCommunicationInterceptor=MainCommunicationInterceptor, run_admin_confirmation_task=namespace['run_admin_confirmation_task'],project_visible_evidence=lambda task,table:[['wrong']])
        with self.assertRaisesRegex(ValueError,'projection'):
            adapter.bind_confirmation_runner(module,self.entry)

    def test_bound_runner_clones_globals_without_changing_shared_module(self):
        namespace={'MainCommunicationInterceptor':MainCommunicationInterceptor}
        exec('async def run_admin_confirmation_task(client, browser, evaluator, task, **kwargs):\n    return {"class": MainCommunicationInterceptor, "task": task}\n',namespace)
        module=types.SimpleNamespace(MainCommunicationInterceptor=MainCommunicationInterceptor, run_admin_confirmation_task=namespace['run_admin_confirmation_task'],project_visible_evidence=test_admin_semantic.sut.project_visible_evidence)
        runner=adapter.bind_confirmation_runner(module,self.entry)
        cell=adapter.make_conditions(MainConditionCell)['visible_table_substitution_step3']
        task={'task_id':'4','task_stratum':'order_payment_aggregation'}
        result=asyncio.run(runner(None,None,None,task,topology='flat',condition_cell=cell))
        self.assertIsNot(result['class'],MainCommunicationInterceptor)
        self.assertIs(module.MainCommunicationInterceptor,MainCommunicationInterceptor)
        self.assertIs(namespace['MainCommunicationInterceptor'],MainCommunicationInterceptor)
        with self.assertRaisesRegex(ValueError,'target'):
            asyncio.run(runner(None,None,None,{**task,'task_id':'107'},topology='flat',condition_cell=cell))

if __name__=='__main__':unittest.main()
