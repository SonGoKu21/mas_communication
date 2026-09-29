from __future__ import annotations

import unittest

from mas_faults.benchmark_trace_contract import (
    CANONICAL_PROPAGATION_CLASSES,
    derive_propagation_class,
    normalize_run_record,
)


class PropagationClassTests(unittest.TestCase):
    def test_fault_exposed_at_a_without_m_consequence_is_not_recovery(self) -> None:
        label = derive_propagation_class(
            fault_applied=True,
            observed_a_symptom=["A1_message_latency"],
            observed_m_consequence=["none"],
            recovery_detected=False,
            final_task_success=True,
        )

        self.assertEqual(label, "exposed_at_A_only")

    def test_m_consequence_then_success_needs_recovery_evidence(self) -> None:
        label = derive_propagation_class(
            fault_applied=True,
            observed_a_symptom=["A8_message_truncation"],
            observed_m_consequence=["M3_incomplete_information_aggregation"],
            recovery_detected=True,
            final_task_success=True,
        )

        self.assertEqual(label, "detected_and_recovered")

    def test_silent_m_propagation_survives_final_success(self) -> None:
        label = derive_propagation_class(
            fault_applied=True,
            observed_a_symptom=["A12_timing_or_session_mismatch"],
            observed_m_consequence=["M5_stale_context_acceptance"],
            recovery_detected=False,
            final_task_success=True,
        )

        self.assertEqual(label, "silent_propagation_to_M")

    def test_m_consequence_with_final_failure_is_not_downgraded(self) -> None:
        label = derive_propagation_class(
            fault_applied=True,
            observed_a_symptom=["A5_message_omission"],
            observed_m_consequence=["M3_incomplete_information_aggregation", "M2_task_timeout_or_failure"],
            recovery_detected=False,
            final_task_success=False,
        )

        self.assertEqual(label, "propagated_to_M_final_failure")


class RecordNormalizationTests(unittest.TestCase):
    def test_normalization_keeps_final_success_separate_from_m_propagation(self) -> None:
        record = normalize_run_record(
            {
                "run_id": "r-1",
                "trace_id": "t-1",
                "benchmark": "WebArena-Verified-Shopping",
                "scenario": "shopping_cart_verification",
                "task_id": "shopping-001",
                "condition": "a12_stale_replay",
                "fault_applied": True,
                "observed_A_symptom": "A12_timing_or_session_mismatch",
                "observed_M_consequence": ["M5_stale_context_acceptance"],
                "recovery_detected": False,
                "final_task_success": True,
            }
        )

        self.assertEqual(record["observed_A_symptom"], ["A12_timing_or_session_mismatch"])
        self.assertEqual(record["propagation_class"], "silent_propagation_to_M")
        self.assertIn(record["propagation_class"], CANONICAL_PROPAGATION_CLASSES)

    def test_normalization_adds_shared_trace_fields_without_erasing_adapter_data(self) -> None:
        record = normalize_run_record(
            {
                "run_id": "r-2",
                "trace_id": "t-2",
                "benchmark": "SWE-bench Verified",
                "instance_id": "sympy__sympy-13091",
                "condition": "clean",
                "fault_applied": False,
                "observed_A_symptom": ["none"],
                "observed_M_consequence": ["none"],
                "recovery_detected": False,
                "final_task_success": True,
                "official_test_passed": True,
            }
        )

        self.assertEqual(record["task_id"], "sympy__sympy-13091")
        self.assertEqual(record["fault_type"], "clean")
        self.assertEqual(record["first_divergence"], "none")
        self.assertEqual(record["task_score"], 1.0)
        self.assertTrue(record["official_test_passed"])


if __name__ == "__main__":
    unittest.main()
