import copy
import json
import unittest
from pathlib import Path
from p1_execution import execution_jobs

class MatrixTests(unittest.TestCase):
    def setUp(self):
        root=(Path(__file__).resolve().parent/'fixtures')
        self.design=json.loads((root/'jobs_proposed.json').read_text())
        self.design['tasks']=json.loads((root/'tasks_recovered.json').read_text())['tasks']
        self.boundaries={'clean':'none','valid_partial':'evidence_handoff',
                         'contract_consistent_identity_corruption':'evidence_handoff'}

    def test_complete_factorial_maps_four_policies_without_action_guard(self):
        rows=execution_jobs(self.design,self.boundaries)
        self.assertEqual(len(rows),432)
        self.assertEqual(sum(r['condition']=='clean' for r in rows),144)
        for row in rows:
            self.assertEqual(row['arm'],'baseline' if row['variant']=='baseline' else 'semantic_only')
            self.assertEqual(row['p1_strategy'],row['variant'])

    def test_duplicate_cell_rejected_even_with_distinct_job_key(self):
        rows=self.design['jobs'];rows[-1]=dict(rows[0],job_key='different-key')
        with self.assertRaises(ValueError):execution_jobs(self.design,self.boundaries)

    def test_missing_and_extra_repeats_rejected(self):
        self.design['jobs'][0]['repeat_index']=4
        with self.assertRaises(ValueError):execution_jobs(self.design,self.boundaries)

    def test_continuous_fault_not_silently_included(self):
        self.design['jobs'][-1]['recovery_path_exposed']=True
        with self.assertRaises(ValueError):execution_jobs(self.design,self.boundaries)

    def test_input_draft_stays_unchanged(self):
        before=copy.deepcopy(self.design)
        execution_jobs(self.design,self.boundaries)
        self.assertEqual(before,self.design)

if __name__=='__main__':unittest.main()
