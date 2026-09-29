"""Offline real AutoGen workflow plus independent runner contract tests."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import types

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
import admin_semantic as pure
from adapter import bind_confirmation_runner,make_conditions
from mas_faults import webarena_admin_confirmation as confirmation
from mas_faults.webarena_admin_main_matrix import MainConditionCell
from test_webarena_admin_confirmation_workflow import SequenceClient,FakeBrowser,FakeEvaluator,_decision,_evidence
try:
    import run_admin_semantic as runner
except ModuleNotFoundError:
    runner=None

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner,'independent execution runner missing')
        self.manifest=json.loads((ROOT/'donor_manifest.json').read_text())

    def test_repeat_inference_policy_is_explicit_and_model_specific(self):
        d1=runner.repeat_policy('deepseek-v4-flash',1);d2=runner.repeat_policy('deepseek-v4-flash',2)
        q1=runner.repeat_policy('Qwen3.5-9B',1);q2=runner.repeat_policy('Qwen3.5-9B',2)
        self.assertEqual((d1['max_steps'],d1['request_timeout'],d1['total_timeout'],d1['schedule_seed'],d1['max_tokens']),(12,120,120,20260815,None))
        self.assertEqual((d2['max_steps'],d2['request_timeout'],d2['total_timeout'],d2['schedule_seed'],d2['max_tokens']),(16,60,180,20260824,None))
        self.assertEqual(q1['max_tokens'],768)
        self.assertIsNone(q2['max_tokens'])

    def test_pilot_and_formal_schedule_counts_and_nonhistorical_names(self):
        pilot=runner.build_schedule(self.manifest,phase='pilot')
        formal=runner.build_schedule(self.manifest,phase='formal')
        self.assertEqual(len(pilot),12)
        self.assertEqual(len(formal),360)
        self.assertEqual({j['task_id'] for j in pilot},{'4','107'})
        self.assertEqual({j['condition'] for j in formal},set(pure.CONDITIONS))
        self.assertEqual(len({j['job_key'] for j in formal}),360)

    def test_task_failure_and_noop_are_saved_once_without_retry(self):
        jobs=runner.build_schedule(self.manifest,phase='pilot')[:2]
        calls=[]
        async def execute(job):
            calls.append(job['job_key'])
            return {'events':[],'final_task_success':False,'error':None,'model':'deepseek-v4-flash'}
        with tempfile.TemporaryDirectory() as d:
            asyncio.run(runner.execute_jobs(jobs,execute,Path(d)))
            asyncio.run(runner.execute_jobs(jobs,execute,Path(d)))
            self.assertEqual(calls,[j['job_key'] for j in jobs])
            saved=[json.loads(s) for s in (Path(d)/'main_runs.jsonl').read_text().splitlines()]
            self.assertEqual(len(saved),2)
            self.assertTrue(all(r['table_exposure']=='not_exposed' for r in saved))

    def test_full_workflow_all_topologies_and_positions_without_reference_leak(self):
        cases=0
        for topology in pure.TOPOLOGIES:
            for condition in pure.CONDITIONS:
                with self.subTest(topology=topology,condition=condition),tempfile.TemporaryDirectory() as d:
                    entry=next(e for e in self.manifest['entries'] if e['target']=={'task_id':'4','task_stratum':'sales_ranking','topology':topology,'repeat_index':1})
                    task={'task_id':4,'task_stratum':'sales_ranking','sites':['shopping_admin'],'intent':'Get current data','eval':{'reference_answers':{'must_include':['HIDDEN_REFERENCE_SENTINEL']}}}
                    config=Path(d)/'4.json';config.write_text(json.dumps(task))
                    client=SequenceClient(['Read and finish.','{"tool":"read_visible_table","arguments":{}}','{"tool":"finish_with_evidence","arguments":{}}',_evidence(),_decision(),_decision()])
                    browser=FakeBrowser()
                    result=asyncio.run(bind_confirmation_runner(confirmation,entry)(client,browser,FakeEvaluator(score=0),task,original_config_file=config,sanitized_config_dir=Path(d)/'sanitized',topology=topology,condition_cell=make_conditions(MainConditionCell)[condition],run_index=1))
                    self.assertFalse(result['final_task_success'])
                    self.assertEqual(result['condition'],condition)
                    self.assertEqual(result['observed_A_symptom'],['A6_message_semantic_corruption'])
                    self.assertEqual(len([e for e in result['events'] if e.get('fault_applied')]),1)
                    self.assertTrue(all('HIDDEN_REFERENCE_SENTINEL' not in p for p in client.prompts))
                    self.assertEqual([x[1] for x in browser.calls if x[0]=='tool'],['read_visible_table','finish_with_evidence'])
                    self.assertFalse(result.get('strict_derivation_validation_errors'))
                    self.assertFalse(result.get('axis_evidence_validation_errors'))
                    audit=runner.audit_row(result,entry,condition)
                    self.assertEqual(audit['errors'],[])
                    self.assertEqual(audit['exposure'],'applied')
                    cases+=1
        self.assertEqual(cases,6)

if __name__=='__main__':unittest.main()
