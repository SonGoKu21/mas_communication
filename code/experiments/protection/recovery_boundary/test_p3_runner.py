import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import p3_runner as runner
ROOT=(Path(__file__).resolve().parent/'fixtures')
class RunnerTests(unittest.TestCase):
 def setUp(self):
  self.d=json.loads((ROOT/'jobs_candidate_balanced.json').read_text());self.d['execution_eligible']=True
  self.gate=ROOT/'http_gate/summary.json'
  self.d['http_gate_sha256']=hashlib.sha256(self.gate.read_bytes()).hexdigest()
  self.d['http_gate_started_sha256']=hashlib.sha256(self.gate.with_name('started.json').read_bytes()).hexdigest()
 def test_real_admission_unchanged_tasks_and_original_arms(self):
  d,jobs,_=runner.validate_admission(self.d,self.gate)
  self.assertEqual(len(jobs),288)
  self.assertEqual(sum(j['condition']=='clean' for j in jobs),72)
  self.assertEqual({j['arm'] for j in jobs},{'baseline','combined'})
  self.assertTrue(all(t['product_url'].startswith('http://localhost:7770/') for t in d['tasks']))
 def test_candidate_not_admitted(self):
  self.d['execution_eligible']=False
  with self.assertRaises(ValueError):runner.validate_admission(self.d,self.gate)
 def test_gate_hash_change_rejected(self):
  self.d['http_gate_sha256']='0'*64
  with self.assertRaises(ValueError):runner.validate_admission(self.d,self.gate)
 def test_changed_task_rejected(self):
  self.d['tasks'][0]['quantity']=4
  with self.assertRaises(ValueError):runner.validate_admission(self.d,self.gate)
 def test_admission_requires_clean_audit(self):
  _,jobs,_=runner.validate_admission(self.d,self.gate)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileNotFoundError):runner.validate_fault_admission(Path(d),jobs,dict(pending=jobs[72:],rows=jobs[:72]),'x')
 def test_formal_wrapper_retains_partial_delivery_audit(self):
  start=dict(arm='combined',exposure_scheme='persistent_handoff')
  exc=RuntimeError('synthetic');exc.partial_trial={'exposure_deliveries':[{'path':'initial_evidence'}]}
  with patch.object(runner.common,'ensure_deepseek_offpeak',side_effect=exc):
   kind,record=runner.attempt({'start':start})
  self.assertEqual(kind,'error');self.assertEqual(record['partial_trial'],exc.partial_trial)
