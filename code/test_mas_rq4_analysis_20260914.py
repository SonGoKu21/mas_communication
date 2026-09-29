import unittest
from mas_rq4_analysis_20260914 import factorial, diagnostic_pairs


class AnalysisTests(unittest.TestCase):
    def test_factorial_action_only_effect(self):
        rows = []
        for i in range(6):
            for arm, outcome in [('baseline', False), ('action_only', True),
                                 ('semantic_only', False), ('combined', True)]:
                common = dict(task_id=str(i), product_cluster=str(i), topology='sequential',
                              repeat_index=1, arm=arm, variant=arm, experiment='main')
                rows.extend([dict(common, condition='clean', final_task_success=True),
                             dict(common, condition='request_non_delivery', final_task_success=outcome)])
        result = {r['contrast']: r for r in factorial(rows, matched=True)}
        self.assertEqual(result['factorial_action']['estimate'], 1)
        self.assertEqual(result['factorial_semantic']['estimate'], 0)
        self.assertEqual(result['interaction']['estimate'], 0)
        self.assertEqual(result['interaction']['clusters'], 6)
        rows[0]['final_task_success'] = False
        self.assertTrue(all(r['clusters'] == 5 for r in factorial(rows, matched=True)))

    def test_incomplete_quartet_not_used(self):
        rows = [dict(task_id='t', topology='flat', condition='valid_partial', repeat_index=1,
                     arm='baseline', product_cluster='p', experiment='main', final_task_success=False)]
        self.assertEqual(factorial(rows), [])

    def test_paired_diagnostics(self):
        common = dict(task_id='t', product_cluster='p', topology='sequential',
                      repeat_index=1, variant='combined', condition='request_non_delivery')
        rows = [dict(common, experiment='main', final_task_success=True),
                dict(common, experiment='stress', final_task_success=False)]
        self.assertEqual(diagnostic_pairs(rows)[0]['estimate'], -1)


if __name__ == '__main__':
    unittest.main()
