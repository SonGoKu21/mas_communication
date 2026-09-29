import unittest
from deepseek_inference_policy import settings

class InferenceSettings(unittest.TestCase):
    def test_reddit_old_request_shape(self):
        a = settings('reddit', '27', 2)
        self.assertIsNone(a['environment']['LLM_MAX_TOKENS'])
        self.assertIsNone(a['environment']['LLM_DISABLE_THINKING'])
        self.assertEqual(a['source_rollout_line'], 73615)

    def test_reddit_expansion_all_repeats(self):
        for r in (1, 2, 3):
            self.assertEqual(settings('reddit', '29', r)['environment']['LLM_MAX_TOKENS'], '1024')

    def test_swe_tac_repeats(self):
        for domain, task in [('swe', 'pallets__flask-5014'), ('tac', 'sde-install-openjdk')]:
            self.assertEqual(settings(domain, task, 1)['environment']['LLM_MAX_TOKENS'], '768')
            self.assertIsNone(settings(domain, task, 2)['environment']['LLM_MAX_TOKENS'])
            self.assertEqual(settings(domain, task, 3)['environment']['LLM_TOTAL_REQUEST_TIMEOUT_SECONDS'], '180')

    def test_original_cohorts_keep_360_seconds(self):
        for domain, task in [('swe', 'django__django-10097'), ('tac', 'sde-install-go')]:
            for r in (1, 2, 3):
                a = settings(domain, task, r)
                self.assertEqual(a['environment']['LLM_REQUEST_TIMEOUT_SECONDS'], '360')
                self.assertEqual(a['max_attempts'], 3)

    def test_bad_repeat_rejected(self):
        with self.assertRaises(ValueError): settings('swe', 'anything', 4)

if __name__ == '__main__':
    unittest.main()
