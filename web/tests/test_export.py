import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('export', Path(__file__).parents[1] / 'tools/export.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ExportTests(unittest.TestCase):
    def test_secret(self):
        self.assertEqual(module.sanitize({'x': {'Authorization': 'Bearer secret'}})['x']['Authorization'], '[REDACTED]')

    def test_embedded(self):
        value = module.sanitize('{"password":"abc123", "total_tokens":22}')
        self.assertNotIn('abc123', value)
        self.assertIn('22', value)

    def test_strings(self):
        value = module.sanitize('http://user:pw@10.102.35.120/a?token=abc /home/hqn/key.txt person@example.com Bearer abcdef')
        for bad in ['10.102.35.120','hqn','example.com','abcdef','token=']:
            self.assertNotIn(bad,value)

    def test_id(self):
        self.assertNotEqual(module.catalog_id('a','b','x'), module.catalog_id('c','b','x'))

    def test_no_invented_events(self):
        self.assertEqual(module.detail({'final_task_success': True}, {})['events'], [])

    def test_unknown(self):
        item = module.metadata({}, 's','d','m','f',0)
        self.assertIsNone(item['endpoint'])
        self.assertIsNone(item['fault_applied'])

    def test_model_identity(self):
        a=module.metadata({'run_id':'same'},'extension','Reddit','model-a','f',1)
        b=module.metadata({'run_id':'same'},'extension','Reddit','model-b','f',2)
        self.assertNotEqual(a['id'],b['id'])

    def test_source_evidence_id(self):
        a=module.metadata({'run_id':'normalized','source_run_id':'original'},'rq123','d','m','f',1)
        self.assertEqual(a['source_evidence_run_id'],'original')
        self.assertEqual(a['source_run_id'],'normalized')

    def test_fixture_dimensions(self):
        a=module.metadata({'level':3,'deadline_ms':5000,'payload_size_bytes':89},'bridge','d','No LLM','f',1)
        self.assertEqual(a['deadline_ms'],5000)
        self.assertEqual(a['severity'],3)

    def test_decision_endpoint(self):
        d=module.detail({'decision_correct':True,'environment_task_success':False}, {})
        self.assertTrue(d['evidence']['decision_correct'])
        self.assertFalse(d['evidence']['environment_task_success'])

if __name__ == '__main__': unittest.main()
