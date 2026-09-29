import unittest
from analyze import transition
class Tests(unittest.TestCase):
 def test_counts(self):
  r=transition([(True,True),(False,True),(True,False),(False,False)])
  self.assertEqual([r[k] for k in ['ss','rescued','regression','ff']],[1,1,1,1]);self.assertEqual(r['net_pp'],0);self.assertEqual(r['rescue_given_failure_pct'],50)
 def test_no_failures(self):self.assertIsNone(transition([(True,True)])['rescue_given_failure_pct'])
 def test_empty(self):self.assertIsNone(transition([])['net_pp'])
if __name__=='__main__':unittest.main()
