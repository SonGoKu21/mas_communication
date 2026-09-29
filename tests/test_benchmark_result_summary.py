from mas_faults.benchmark_result_summary import summarize_runs


def test_summary_keeps_m_propagation_when_final_task_succeeds():
    rows = [
        {
            "condition": "clean", "fault_applied": False, "observed_A_symptom": ["none"],
            "observed_M_consequence": ["none"], "recovery_detected": False,
            "final_task_success": True, "propagation_class": "clean", "latency_ms": 10, "total_tokens": 5,
        },
        {
            "condition": "a12_stale_replay", "fault_applied": True, "observed_A_symptom": ["A12_timing_or_session_mismatch"],
            "observed_M_consequence": ["M5_stale_context_acceptance"], "recovery_detected": False,
            "final_task_success": True, "propagation_class": "silent_propagation_to_M", "latency_ms": 20, "total_tokens": 7,
        },
        {
            "condition": "a5_omission", "fault_applied": True, "observed_A_symptom": ["A5_message_omission"],
            "observed_M_consequence": ["M3_incomplete_information_aggregation"], "recovery_detected": True,
            "final_task_success": True, "propagation_class": "detected_and_recovered", "latency_ms": 30, "total_tokens": 9,
        },
    ]

    summary, by_condition = summarize_runs(rows)

    assert summary["runs"] == 3
    assert summary["m_propagation_count"] == 2
    assert summary["final_success_count"] == 3
    assert summary["transition_counts"]["silent_propagation_to_M"] == 1
    assert by_condition["a12_stale_replay"]["m_consequence_count"] == 1
    assert by_condition["a5_omission"]["recovery_count"] == 1
