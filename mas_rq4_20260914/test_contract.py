import copy
import json
import unittest
from contract import ReceiverContract


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(task_id='t', product_title='Product', quantity=2)
        self.payload = dict(task_id='t', product_title='Product', requested_quantity=2,
                            observed_quantity=2, cart_verified=True, product_id='1', sku='s')
        self.payload['evidence'] = json.dumps(self.payload)
        self.message = dict(task_id='t', session_id='session', action_id='action',
                            entity_id='cart', version=2, evidence_id='e2', source='Worker',
                            payload=self.payload)

    def policy(self, **kwargs):
        return ReceiverContract(self.task, session_id='session', action_id='action',
                                entity_id='cart', minimum_version=2, **kwargs)

    def test_correct(self):
        self.assertEqual(self.policy().issues(self.message), ())

    def test_partial_and_binding(self):
        message = copy.deepcopy(self.message)
        message['payload'].pop('sku')
        message['session_id'] = 'other'
        issues = self.policy().issues(message)
        self.assertIn('missing:sku', issues)
        self.assertIn('binding:session_id', issues)

    def test_stale_and_boolean_version(self):
        for version in (1, True):
            message = dict(self.message, version=version)
            self.assertTrue(self.policy().issues(message))

    def test_no_hidden_identity_oracle(self):
        message = copy.deepcopy(self.message)
        message['payload']['sku'] = 'wrong'
        message['payload']['product_id'] = '999'
        message['payload']['evidence'] = json.dumps({k: v for k, v in message['payload'].items() if k != 'evidence'})
        self.assertEqual(self.policy().issues(message), ())
        policy = self.policy(known_identity={'sku': 's', 'product_id': '1'},
                             identity_provenance='task_input')
        self.assertIn('identity:sku', policy.issues(message))
        with self.assertRaises(ValueError):
            self.policy(known_identity={'sku': 's'})

    def test_recheck_is_checked_and_bounded(self):
        bad = dict(self.message, session_id='other')
        policy = self.policy()
        calls = []
        def readback():
            calls.append(1)
            return bad
        self.assertIsNone(policy.accept(bad, readback))
        self.assertEqual(len(calls), 1)
        self.assertFalse(policy.events[-1]['accepted'])
        self.assertIsNone(policy.accept(bad, readback))
        self.assertEqual(len(calls), 1)

    def test_higher_version_advances_watermark(self):
        policy = self.policy()
        policy.accept(dict(self.message, version=4), lambda: None)
        self.assertIn('stale:version', policy.issues(self.message))


if __name__ == '__main__':
    unittest.main()
