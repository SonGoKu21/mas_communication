import json,os,unittest
from pathlib import Path
import qwen27_runtime as runtime
O=Path(__file__).resolve().parent
class RuntimeTests(unittest.TestCase):
 def setUp(self):
  self.refs=json.loads((O/'historical_carrier_reference_by_repeat.json').read_text())
  self.policy=json.loads((O/'frozen_paired_inference_policy.json').read_text())
 def test_repeat_specific_source_is_not_collapsed(self):
  refs=runtime.index_references(self.refs)
  a=refs[('SWE-bench Verified','astropy__astropy-14309',1)]
  b=refs[('SWE-bench Verified','astropy__astropy-14309',2)]
  self.assertNotEqual(a['clean']['original_message'],b['clean']['original_message'])
  self.assertEqual(len(refs),150)
  with self.assertRaises(ValueError):runtime.index_references(self.refs+self.refs[:1])
 def test_budget_lookup_rejects_unknown_task(self):
  policies=runtime.index_policy(self.policy)
  self.assertEqual(len(policies),300)
  with self.assertRaises(KeyError):runtime.environment(policies,'swe','unknown',1)
 def test_budget_change_unsets_token_cap(self):
  p=runtime.index_policy(self.policy)
  first=runtime.environment(p,'swe','django__django-10097',1)
  later=runtime.environment(p,'swe','django__django-10097',2)
  runtime.apply_environment(first);self.assertEqual(os.environ['LLM_MAX_TOKENS'],'4096')
  runtime.apply_environment(later);self.assertNotIn('LLM_MAX_TOKENS',os.environ)
  self.assertEqual(os.environ['LLM_REQUEST_TIMEOUT_SECONDS'],'60')
  self.assertNotEqual(runtime.client_key(first),runtime.client_key(later))
 def test_pilot_covers_repeat_source_and_budget_classes(self):
  for domain,bench in [('reddit','WebArena Reddit'),('swe','SWE-bench Verified'),('tac','TheAgentCompany')]:
   refs=[r for r in self.refs if r['benchmark']==bench]
   p=runtime.index_policy(self.policy)
   selected=runtime.pilot_representatives(domain,refs,p)
   self.assertEqual({rep for task,rep in selected},{1,2,3})
   if domain=='swe':self.assertTrue(any(t=='astropy__astropy-14309' and rep==1 for t,rep in selected))
 def test_all_six_conditions_share_budget(self):
  self.assertTrue(all(len(row['all_six_conditions'])==6 for row in self.policy['proposed']))
  self.assertEqual(self.policy['status'],'FROZEN_FOR_NEW_CONTEMPORANEOUS_PAIRED_BATCH')
if __name__=='__main__':unittest.main()
