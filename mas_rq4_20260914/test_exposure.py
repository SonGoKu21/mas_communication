import unittest
from exposure import FiniteExposure


class ExposureTests(unittest.TestCase):
    def test_two_losses_then_delivery(self):
        fault = FiniteExposure('request_non_delivery', count=2)
        self.assertEqual(fault.deliver('other', {'x': 1}), [{'x': 1}])
        self.assertEqual(fault.deliver('action_request', {'x': 1}), [])
        self.assertEqual(fault.deliver('action_request', {'x': 1}, recovery=True), [])
        self.assertEqual(fault.deliver('action_request', {'x': 1}, recovery=True), [{'x': 1}])
        self.assertEqual(len(fault.events), 2)
        self.assertEqual([d['damaged'] for d in fault.deliveries], [True, True, False])

    def test_no_phantom_second_injection(self):
        fault = FiniteExposure('request_non_delivery', count=2)
        fault.deliver('action_request', {})
        self.assertEqual(len(fault.events), 1)

    def test_duplicate_is_independent_copy(self):
        fault = FiniteExposure('duplicate_action_delivery', count=1)
        delivered = fault.deliver('action_request', {'nested': {'quantity': 1}})
        delivered[0]['nested']['quantity'] = 8
        self.assertEqual(delivered[1]['nested']['quantity'], 1)

    def test_recovery_exemption_is_explicit(self):
        fault = FiniteExposure('request_non_delivery', count=1, expose_recovery=False)
        self.assertEqual(fault.deliver('action_request', {}, recovery=True), [{}])
        self.assertEqual(len(fault.events), 0)


if __name__ == '__main__':
    unittest.main()
