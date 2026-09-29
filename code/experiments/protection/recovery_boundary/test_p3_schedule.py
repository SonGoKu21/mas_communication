import collections
import copy
import json
import unittest
from pathlib import Path
from p3_schedule import build_candidate


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.draft = json.loads(((Path(__file__).resolve().parent/'fixtures') / 'jobs_draft.json').read_text())

    def test_candidate_preserves_every_job_and_task_without_mutating_draft(self):
        old = copy.deepcopy(self.draft)
        candidate = build_candidate(self.draft)
        self.assertEqual(self.draft, old)
        self.assertFalse(candidate['execution_eligible'])
        self.assertEqual(candidate['tasks'], old['tasks'])
        self.assertEqual({j['job_key']: j for j in candidate['jobs']}, {j['job_key']: j for j in old['jobs']})
        self.assertEqual(len(candidate['jobs']), 288)
        self.assertEqual(candidate, build_candidate(self.draft))

    def test_blocks_remain_adjacent_and_each_position_is_balanced_by_stratum(self):
        candidate = build_candidate(self.draft)
        counts = collections.Counter()
        seen = set()
        for offset in range(0, len(candidate['jobs']), 2):
            block = candidate['jobs'][offset:offset + 2]
            self.assertEqual(block[0]['pair_key'], block[1]['pair_key'])
            self.assertNotIn(block[0]['pair_key'], seen)
            seen.add(block[0]['pair_key'])
            stratum = 'clean' if block[0]['exposure_scheme'] == 'clean' else 'fault'
            for position, j in enumerate(block):
                counts[stratum, position, j['arm']] += 1
        for arm in ('baseline', 'combined'):
            for position in (0, 1):
                self.assertEqual(counts['clean', position, arm], 18)
                self.assertEqual(counts['fault', position, arm], 54)
        self.assertEqual(len(seen), 144)
        self.assertTrue(all(j['exposure_scheme'] == 'clean' for j in candidate['jobs'][:72]))

    def test_seed_changes_only_order_and_invalid_pairs_are_rejected(self):
        first = build_candidate(self.draft)
        second = build_candidate(self.draft, seed=20260929)
        self.assertNotEqual([j['job_key'] for j in first['jobs']], [j['job_key'] for j in second['jobs']])
        self.assertEqual({j['job_key'] for j in first['jobs']}, {j['job_key'] for j in second['jobs']})
        for mutation in ('missing', 'duplicate', 'eligible'):
            bad = copy.deepcopy(self.draft)
            if mutation == 'missing':
                bad['jobs'].pop()
            elif mutation == 'duplicate':
                bad['jobs'].append(copy.deepcopy(bad['jobs'][0]))
            else:
                bad['execution_eligible'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_candidate(bad)
