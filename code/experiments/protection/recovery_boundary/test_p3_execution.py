import copy,json,unittest
from pathlib import Path
from p3_execution import execution_jobs, pending_after_clean_barrier
ROOT=(Path(__file__).resolve().parent/'fixtures')
class ExecutionTests(unittest.TestCase):
 def setUp(self): self.d=json.loads((ROOT/'jobs_candidate_balanced.json').read_text())
 def test_full_mapping_preserves_original_roles_and_keys(self):
  jobs=execution_jobs(self.d)
  self.assertEqual(len(jobs),288)
  self.assertEqual([j['job_key'] for j in jobs],[j['job_key'] for j in self.d['jobs']])
  self.assertEqual({j['arm'] for j in jobs},{'baseline','combined'})
  self.assertEqual(sum(j['condition']=='clean' for j in jobs),72)
  self.assertTrue(all('p1_strategy' not in j for j in jobs))
 def test_missing_cell_rejected(self):
  self.d['jobs'].pop()
  with self.assertRaises(ValueError): execution_jobs(self.d)
 def test_duplicate_coordinate_rejected(self):
  self.d['jobs'][1]=dict(self.d['jobs'][0],job_key='different')
  with self.assertRaises(ValueError): execution_jobs(self.d)
 def test_unsupported_arm_rejected(self):
  self.d['jobs'][0]['arm']='guarded_readback'
  with self.assertRaises(ValueError): execution_jobs(self.d)
 def test_schedule_reorder_rejected(self):
  self.d['jobs'][0],self.d['jobs'][1]=self.d['jobs'][1],self.d['jobs'][0]
  with self.assertRaises(ValueError): execution_jobs(self.d)
 def test_clean_barrier_returns_only_clean(self):
  jobs=execution_jobs(self.d)
  self.assertEqual(len(pending_after_clean_barrier(dict(pending=jobs,rows=[]),jobs)),72)
 def test_fault_blocked_without_all_clean(self):
  jobs=execution_jobs(self.d)
  with self.assertRaises(RuntimeError): pending_after_clean_barrier(dict(pending=jobs[72:],rows=jobs[:71]),jobs)
 def test_all_clean_releases_fault(self):
  jobs=execution_jobs(self.d)
  self.assertEqual(pending_after_clean_barrier(dict(pending=jobs[72:],rows=jobs[:72]),jobs),jobs[72:])
