import unittest
from collections import Counter
from design import build_jobs, mechanisms


class DesignTests(unittest.TestCase):
    def tasks(self):
        return [dict(task_id=f't{i}', product_cluster=f'p{i}') for i in range(6)]

    def test_counts_and_uniqueness(self):
        jobs = build_jobs(self.tasks())
        self.assertEqual(len(jobs), 1098)
        self.assertEqual(len({j['job_key'] for j in jobs}), 1098)
        self.assertEqual(Counter(j['experiment'] for j in jobs),
                         {'main': 864, 'stress': 144, 'path_diagnostic': 90})
        self.assertEqual(sum(j['condition'] == 'clean' for j in jobs), 144)

    def test_factorial(self):
        self.assertEqual(mechanisms('combined'), dict(action=True, semantic=True))
        self.assertEqual(mechanisms('baseline'), dict(action=False, semantic=False))
        self.assertEqual(mechanisms('action_only'), dict(action=True, semantic=False))
        self.assertEqual(mechanisms('semantic_only'), dict(action=False, semantic=True))
        with self.assertRaises(ValueError):
            mechanisms('independent')

    def test_reject_pseudo_breadth(self):
        tasks = self.tasks()
        tasks[-1]['product_cluster'] = 'p0'
        with self.assertRaises(ValueError):
            build_jobs(tasks)

    def test_paired_cells(self):
        jobs = build_jobs(self.tasks())
        pairs = Counter(j['pair_key'] for j in jobs)
        for job in jobs:
            self.assertEqual(pairs[job['pair_key']], 3 if job['experiment'] == 'path_diagnostic' else 4)
            if job['experiment'] == 'stress':
                self.assertEqual(job['planned_exposures'], 2)
                self.assertTrue(job['recovery_path_exposed'])


if __name__ == '__main__':
    unittest.main()
