import copy
import unittest
from exposure import ScopedIdentityExposure


class ExposureTest(unittest.TestCase):
    def setUp(self):
        self.current = dict(task_id='target',evidence_id='e1',session_id='s',action_id='a',
                            payload=dict(sku='target-sku',product_id='1',requested_quantity=2,
                                         observed_quantity=2,cart_verified=True,evidence='original'))
        self.source = dict(task_id='donor',evidence_id='donor-e1',payload=dict(sku='donor-sku',product_id='2'))

    def make(self, scheme):
        return ScopedIdentityExposure(scheme, cross_task_evidence=self.source)

    def test_single_only_initial_once(self):
        f=self.make('single_handoff')
        self.assertEqual(f.deliver('initial_evidence',self.current)[0]['payload']['sku'],'donor-sku')
        self.assertEqual(f.deliver('initial_evidence',self.current),[self.current])
        self.assertEqual(f.deliver('recovery_evidence',self.current),[self.current])
        self.assertEqual(len(f.events),1)

    def test_persistent_has_no_two_attempt_cap(self):
        f=self.make('persistent_handoff')
        for path in ['initial_evidence']+['recovery_evidence']*5:
            self.assertEqual(f.deliver(path,self.current)[0]['payload']['sku'],'donor-sku')
        self.assertEqual(len(f.events),6)
        self.assertEqual(f.deliver('flat_observation',self.current),[self.current])

    def test_shared_exposes_all_workflow_paths_but_never_evaluation(self):
        f=self.make('shared_workflow_evidence')
        for path in ('initial_evidence','recovery_evidence','flat_observation'):
            self.assertEqual(f.deliver(path,self.current)[0]['payload']['sku'],'donor-sku')
        self.assertEqual(f.deliver('evaluation_readback',self.current),[self.current])
        self.assertFalse(f.deliveries[-1]['eligible'])

    def test_clean_and_source_immutability(self):
        old=copy.deepcopy(self.current)
        for scheme in ('clean','single_handoff','persistent_handoff','shared_workflow_evidence'):
            f=self.make(scheme)
            value=f.deliver('initial_evidence',self.current)[0]
            value['payload']['sku']='modified-by-consumer'
            self.assertEqual(self.current,old)
            self.assertEqual(self.source['payload']['sku'],'donor-sku')
            if scheme=='clean':self.assertEqual(f.events,[])

    def test_unknown_paths_fail_closed(self):
        with self.assertRaises(ValueError):self.make('unknown')
        with self.assertRaises(ValueError):self.make('single_handoff').deliver('typo',self.current)


if __name__=='__main__':unittest.main()
