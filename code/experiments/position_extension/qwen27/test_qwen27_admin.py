import asyncio,json,tempfile,unittest
from pathlib import Path
from run_qwen27_admin import build_schedule,audit_row,execute
class AdminTests(unittest.TestCase):
 def test_six_cells_exact_coverage_and_own_donor(self):
  manifest=json.loads((Path(__file__).parent/'donor_manifest_qwen27.json').read_text())
  jobs=build_schedule(manifest,'formal')
  self.assertEqual(len(jobs),1080);self.assertEqual(len({j['job_key'] for j in jobs}),1080)
  for job in jobs:self.assertEqual(job['entry']['donor']['model'],'Qwen3.8-27B')
  self.assertEqual({j['repeat_index'] for j in build_schedule(manifest,'pilot')},{1,2})
 def test_preinjection_failure_visible_and_not_success(self):
  row={'events':[],'error':'model JSON invalid','final_task_success':False}
  audit=audit_row(row,{},'non_delivery_step3')
  self.assertEqual(audit['exposure'],'not_exposed');self.assertEqual(audit['errors'],[])
 def test_failed_model_output_is_persisted_once_on_resume(self):
  manifest=json.loads((Path(__file__).parent/'donor_manifest_qwen27.json').read_text())
  jobs=build_schedule(manifest,'pilot')[:2];calls=[]
  async def one(job):
   calls.append(job['job_key'])
   return {'model':'Qwen3.8-27B','events':[],'error':'ControlledRunError: ValueError: tool request argument values must be strings','final_task_success':False}
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)
   asyncio.run(execute(jobs,one,out));asyncio.run(execute(jobs,one,out))
   saved=[json.loads(s) for s in (out/'main_runs.jsonl').read_text().splitlines()]
   self.assertEqual(calls,[j['job_key'] for j in jobs]);self.assertEqual(len(saved),2)
   self.assertTrue(all(r['execution_class']=='model_output_error' and r['final_task_success'] is False for r in saved))
 def test_omission_delivered_rejected(self):
  row={'events':[{'fault_applied':True,'abstract_step':3,'delivery_count':1,'delivered_messages':[{}]}]}
  self.assertIn('omission_not_dropped',audit_row(row,{},'non_delivery_step3')['errors'])
if __name__=='__main__':unittest.main()
