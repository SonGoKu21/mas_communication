from mas_faults.webarena_clean_baseline_summary import summarize_clean_runs


def test_summary_admits_only_tasks_that_pass_every_clean_repeat():
    rows = [
        {"task_id": "stable", "condition": "clean", "final_task_success": True, "error": None},
        {"task_id": "stable", "condition": "clean", "final_task_success": True, "error": None},
        {"task_id": "unstable", "condition": "clean", "final_task_success": True, "error": None},
        {"task_id": "unstable", "condition": "clean", "final_task_success": False, "error": None},
    ]

    summary = summarize_clean_runs(rows, expected_repeats=2)

    assert summary["stable_task_ids"] == ["stable"]
    assert summary["stable_tasks"] == 1
    assert summary["tasks"][1]["admission_reason"] == "clean_baseline_not_stable"
