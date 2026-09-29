import unittest
from design import build_jobs
from runner import execution_jobs


class RunnerTests(unittest.TestCase):
    def test_explicit_execution_mapping(self):
        tasks = [dict(task_id=str(i), product_cluster=str(i)) for i in range(6)]
        jobs = execution_jobs(dict(jobs=build_jobs(tasks)))
        self.assertEqual(len(jobs), 1098)
        self.assertEqual(len({j['job_key'] for j in jobs}), 1098)
        for job in jobs:
            if job['experiment'] == 'path_diagnostic':
                self.assertEqual(job['arm'], 'baseline')
                self.assertEqual(job['topology'], 'sequential')
                self.assertEqual(job['path_design'], job['variant'])
            else:
                self.assertEqual(job['arm'], job['variant'])
                self.assertIsNone(job['path_design'])


if __name__ == '__main__':
    unittest.main()
