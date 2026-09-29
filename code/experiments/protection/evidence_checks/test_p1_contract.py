import copy
import json
import unittest
from p1_contract import ReceiverContract

class ReadbackPolicyTests(unittest.TestCase):
    def setUp(self):
        self.task={'task_id':'t','product_title':'Product','quantity':2}
        payload=dict(task_id='t',product_title='Product',requested_quantity=2,
                     observed_quantity=2,cart_verified=True,product_id='1',sku='s')
        payload['evidence']=json.dumps(payload)
        self.good=dict(task_id='t',session_id='session',action_id='action',entity_id='cart',
                       version=2,evidence_id='e2',source='Worker',payload=payload)
        self.bad=copy.deepcopy(self.good);self.bad['payload'].pop('sku')
        self.calls=[]

    def policy(self,strategy):
        return ReceiverContract(self.task,session_id='session',action_id='action',
                                entity_id='cart',minimum_version=2,strategy=strategy)

    def readback(self):
        self.calls.append(1)
        return copy.deepcopy(self.good)

    def test_check_only_rejects_without_observation(self):
        p=self.policy('check_only')
        self.assertIsNone(p.accept(self.bad,self.readback))
        self.assertEqual(self.calls,[])
        self.assertFalse(p.events[-1]['accepted'])

    def test_guarded_skips_clean_then_repairs(self):
        p=self.policy('guarded_readback')
        self.assertEqual(p.accept(self.good,self.readback),self.good)
        self.assertEqual(self.calls,[])
        self.assertEqual(p.accept(self.bad,self.readback),self.good)
        self.assertEqual(len(self.calls),1)

    def test_always_reads_at_first_receiver_even_clean(self):
        p=self.policy('always_readback')
        self.assertEqual(p.accept(self.good,self.readback),self.good)
        self.assertEqual(len(self.calls),1)
        self.assertIsNone(p.accept(self.bad,self.readback))
        self.assertEqual(len(self.calls),1)

    def test_guarded_failed_repair_stays_rejected_and_bounded(self):
        p=self.policy('guarded_readback')
        def bad_readback():self.calls.append(1);return self.bad
        self.assertIsNone(p.accept(self.bad,bad_readback))
        self.assertIsNone(p.accept(self.bad,bad_readback))
        self.assertEqual(len(self.calls),1)

    def test_baseline_preserves_message_without_check_or_readback(self):
        p=self.policy('baseline')
        self.assertEqual(p.accept(self.bad,self.readback),self.bad)
        self.assertEqual(self.calls,[])
        self.assertEqual(p.events,[])

    def test_no_hidden_identity_oracle(self):
        altered=copy.deepcopy(self.good)
        altered['payload']['sku']='other';altered['payload']['product_id']='999'
        altered['payload']['evidence']=json.dumps({k:v for k,v in altered['payload'].items() if k!='evidence'})
        for strategy in ('check_only','guarded_readback','always_readback'):
            self.assertEqual(self.policy(strategy).issues(altered),())

    def test_input_is_not_mutated(self):
        before=copy.deepcopy(self.bad)
        self.policy('always_readback').accept(self.bad,self.readback)
        self.assertEqual(before,self.bad)

    def test_policy_names_are_strict(self):
        with self.assertRaises(ValueError):self.policy('combined')

if __name__=='__main__':unittest.main()
