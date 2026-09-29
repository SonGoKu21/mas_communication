import unittest
from core import select_unit,paired,overlap,jaccard,CONFIGS

def rows(values=None,applied=True):
 return {c:dict(success=(values or {}).get(c,True),applied=applied) for c in CONFIGS}
class CoreTests(unittest.TestCase):
 def test_missing_and_unknown_not_false(self):
  r=rows();r.pop(CONFIGS[0]);self.assertEqual(select_unit(r,rows(),'common_applied'),'missing_fault_configuration')
  r=rows({CONFIGS[0]:None});self.assertEqual(select_unit(r,rows(),'common_applied'),'unknown_fault_outcome')
 def test_applied_and_clean_sensitivities(self):
  r=rows(applied=False);self.assertEqual(select_unit(r,rows(),'all_known'),'included');self.assertEqual(select_unit(r,rows(),'common_applied'),'not_all_nine_applied')
  c=rows({CONFIGS[0]:False});self.assertEqual(select_unit(rows(),c,'clean_success_common_applied'),'not_all_nine_clean_success')
 def test_no_clean_unknown_silently_coerced(self):
  self.assertEqual(select_unit(rows(),rows({CONFIGS[0]:None}),'clean_success_common_applied'),'unknown_clean_outcome')
 def test_four_states_and_rescue_denominator(self):
  s=paired([False,False,True,True],[False,True,False,True]);self.assertEqual(s['four_states'],{'00':1,'01':1,'10':1,'11':1});self.assertEqual(s['rescued_among_baseline_failures_pct'],50);self.assertEqual(s['net_pp'],0)
 def test_overlap_on_same_baseline_failures(self):
  o=overlap([False,False,False,False,True],[True,True,False,False,True],[True,False,True,False,True]);self.assertEqual([o[k] for k in ('both','flat_only','model_only','neither')],[1,1,1,1]);self.assertAlmostEqual(o['rescue_jaccard'],1/3)
 def test_empty_union_na(self):
  self.assertIsNone(jaccard(set(),set())['jaccard']);self.assertIsNone(paired([True],[True])['rescued_among_baseline_failures_pct'])
