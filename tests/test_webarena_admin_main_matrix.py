from collections import Counter

from mas_faults.webarena_admin_main_matrix import (
    CONDITION_BY_NAME,
    MAIN_CONDITION_CELLS,
    MAIN_TASK_IDS,
    MAIN_TOPOLOGIES,
    MainCommunicationInterceptor,
    build_main_matrix_jobs,
)


EXPECTED_CELLS = [
    ("clean", "clean", None),
    ("timeliness_moderate_step4", "timeliness", 4),
    ("timeliness_deadline_step4", "timeliness", 4),
    ("non_delivery_step2", "non_delivery", 2),
    ("non_delivery_step4", "non_delivery", 4),
    ("semantic_corruption_step2", "semantic_corruption", 2),
    ("semantic_corruption_step4", "semantic_corruption", 4),
    ("malformed_message_step4", "unparseable", 4),
    ("valid_partial_message_step4", "partial", 4),
    ("duplicate_delivery_step2", "duplication", 2),
    ("same_session_reordering_step3", "ordering_freshness", 3),
    ("stale_replay_step4", "ordering_freshness", 4),
    ("contract_key_drift_step4", "contract_drift", 4),
    ("contract_type_drift_step4", "contract_drift", 4),
]


def test_main_matrix_contract_is_frozen() -> None:
    assert MAIN_TOPOLOGIES == ("sequential", "flat", "hierarchical")
    assert MAIN_TASK_IDS == (4, 107, 187, 199, 288)
    assert [
        (cell.condition, cell.fault_family, cell.injection_step)
        for cell in MAIN_CONDITION_CELLS
    ] == EXPECTED_CELLS


def test_main_matrix_contains_exactly_630_unique_jobs() -> None:
    tasks = [{"task_id": task_id} for task_id in MAIN_TASK_IDS]
    jobs = build_main_matrix_jobs(tasks, repetitions=3)

    assert len(jobs) == 630
    assert len({job.job_key for job in jobs}) == 630
    assert {job.matrix_run_index for job in jobs} == set(range(1, 631))

    quotas = Counter(
        (job.topology, job.task["task_id"], job.condition_cell.condition)
        for job in jobs
    )
    assert set(quotas.values()) == {3}


def test_custom_holdout_matrix_contains_exactly_525_unique_jobs() -> None:
    task_ids = (
        0, 1, 2, 3, 4,
        107, 108, 109, 110, 111,
        183, 184, 185, 186, 187,
        198, 199, 200, 201, 202,
        288, 289, 290, 291, 292,
    )
    condition_names = (
        "clean",
        "timeliness_deadline_step4",
        "non_delivery_step2",
        "non_delivery_step4",
        "semantic_corruption_step4",
        "valid_partial_message_step4",
        "stale_replay_step4",
    )
    tasks = [{"task_id": task_id} for task_id in task_ids]
    jobs = build_main_matrix_jobs(
        tasks,
        repetitions=1,
        task_ids=task_ids,
        condition_cells=tuple(CONDITION_BY_NAME[name] for name in condition_names),
    )

    assert len(jobs) == 525
    assert len({job.job_key for job in jobs}) == 525
    assert {job.matrix_run_index for job in jobs} == set(range(1, 526))
    quotas = Counter(
        (job.topology, job.task["task_id"], job.condition_cell.condition)
        for job in jobs
    )
    assert set(quotas.values()) == {1}


def test_non_delivery_uses_a5_as_representative_and_preserves_cause() -> None:
    cells = {cell.condition: cell for cell in MAIN_CONDITION_CELLS}

    for name in ("non_delivery_step2", "non_delivery_step4"):
        assert cells[name].fault_id == "A5"
        assert cells[name].fault_cause == "message_omission"
        assert cells[name].fault_type == "omission"


def test_clean_cell_uses_same_path_without_fault() -> None:
    clean = MAIN_CONDITION_CELLS[0]

    assert clean.fault_id == "none"
    assert clean.fault_type == "clean"
    assert clean.fault_applied is False
    assert clean.parameters == {}


def _message() -> dict:
    return {
        "message_id": "current-message",
        "task_id": "107",
        "source_session": "current-session",
        "state_version": 3,
        "payload": {
            "evidence_result": {
                "candidate_answer": "May: 8 orders",
                "evidence_summary": "current evidence",
                "evidence_row_indices": [1],
            },
            "visible_evidence": [["Interval", "Orders"], ["5/2022", "8"]],
            "structured_task_evidence": {"row_count": 1},
        },
    }


def _stale_message() -> dict:
    stale = _message()
    stale["message_id"] = "stale-message"
    stale["task_id"] = "288"
    stale["source_session"] = "stale-session"
    stale["payload"]["evidence_result"]["candidate_answer"] = "Samantha Jones"
    stale["payload"]["evidence_result"]["evidence_summary"] = "stale evidence"
    return stale


def test_interceptor_applies_at_most_once_at_the_frozen_step() -> None:
    cell = next(
        item for item in MAIN_CONDITION_CELLS if item.condition == "non_delivery_step2"
    )
    interceptor = MainCommunicationInterceptor(cell)
    request = {
        "tool": "set_range_filter",
        "arguments": {"field": "Quantity", "from_value": "1", "to_value": "3"},
    }

    wrong_step = interceptor.intercept(3, {"state_version": 1})
    injected = interceptor.intercept(2, request, context={"eligible": True})
    later = interceptor.intercept(2, request, context={"eligible": True})

    assert wrong_step.fault_applied is False
    assert injected.fault_applied is True
    assert injected.delivered_messages == ()
    assert later.fault_applied is False
    assert later.delivered_messages == (request,)


def test_step2_semantic_corruption_keeps_the_tool_contract_valid() -> None:
    cell = next(
        item
        for item in MAIN_CONDITION_CELLS
        if item.condition == "semantic_corruption_step2"
    )
    request = {
        "tool": "set_range_filter",
        "arguments": {"field": "Quantity", "from_value": "1", "to_value": "3"},
    }

    delivery = MainCommunicationInterceptor(cell).intercept(
        2, request, context={"eligible": True}
    )

    assert delivery.fault_applied is True
    assert delivery.delivered_messages == (
        {
            "tool": "set_range_filter",
            "arguments": {
                "field": "Quantity",
                "from_value": "11",
                "to_value": "13",
            },
        },
    )


def test_step2_duplicate_delivers_the_same_request_twice() -> None:
    cell = next(
        item
        for item in MAIN_CONDITION_CELLS
        if item.condition == "duplicate_delivery_step2"
    )
    request = {"tool": "open_sales_orders", "arguments": {}}

    delivery = MainCommunicationInterceptor(cell).intercept(
        2, request, context={"eligible": True}
    )

    assert delivery.delivery_count == 2
    assert delivery.delivered_messages == (request, request)


def test_step3_reordering_delivers_newer_state_then_older_state() -> None:
    cell = next(
        item
        for item in MAIN_CONDITION_CELLS
        if item.condition == "same_session_reordering_step3"
    )
    older = {"state_version": 1, "visible_evidence": []}
    newer = {"state_version": 2, "visible_evidence": [["ID"], ["299"]]}

    delivery = MainCommunicationInterceptor(cell).intercept(
        3,
        newer,
        context={"eligible": True, "older_message": older},
    )

    assert delivery.delivered_messages == (newer, older)
    assert delivery.observed_runtime_effect == "newer_state_delivered_before_older_state"


def test_a7_is_unparseable_but_a8_is_valid_and_partial() -> None:
    by_name = {cell.condition: cell for cell in MAIN_CONDITION_CELLS}

    malformed = MainCommunicationInterceptor(
        by_name["malformed_message_step4"]
    ).intercept(4, _message(), context={"eligible": True})
    partial = MainCommunicationInterceptor(
        by_name["valid_partial_message_step4"]
    ).intercept(4, _message(), context={"eligible": True})

    assert isinstance(malformed.delivered_messages[0], str)
    assert malformed.parseable is False
    delivered_partial = partial.delivered_messages[0]
    assert isinstance(delivered_partial, dict)
    assert partial.parseable is True
    assert "candidate_answer" in delivered_partial["payload"]["evidence_result"]
    assert "evidence_row_indices" not in delivered_partial["payload"]["evidence_result"]


def test_a10_same_session_and_a12_cross_session_have_distinct_bindings() -> None:
    by_name = {cell.condition: cell for cell in MAIN_CONDITION_CELLS}
    current = _message()
    older = _message()
    older["state_version"] = 2

    reordered = MainCommunicationInterceptor(
        by_name["same_session_reordering_step3"]
    ).intercept(3, current, context={"eligible": True, "older_message": older})
    stale = MainCommunicationInterceptor(
        by_name["stale_replay_step4"], stale_message=_stale_message()
    ).intercept(4, current, context={"eligible": True})

    assert reordered.delivered_messages[-1]["source_session"] == "current-session"
    assert stale.delivered_messages[0]["source_session"] == "stale-session"
    assert stale.delivered_messages[0]["task_id"] == "288"


def test_contract_key_and_type_drift_remain_valid_json_but_differ() -> None:
    by_name = {cell.condition: cell for cell in MAIN_CONDITION_CELLS}
    key_drift = MainCommunicationInterceptor(
        by_name["contract_key_drift_step4"]
    ).intercept(4, _message(), context={"eligible": True})
    type_drift = MainCommunicationInterceptor(
        by_name["contract_type_drift_step4"]
    ).intercept(4, _message(), context={"eligible": True})

    key_message = key_drift.delivered_messages[0]
    type_message = type_drift.delivered_messages[0]
    assert "task_binding" in key_message and "task_id" not in key_message
    assert "candidate_response" in key_message["payload"]["evidence_result"]
    assert isinstance(type_message["state_version"], str)
    assert isinstance(
        type_message["payload"]["evidence_result"]["candidate_answer"], list
    )


def test_moderate_delay_and_deadline_have_different_delivery_outcomes() -> None:
    by_name = {cell.condition: cell for cell in MAIN_CONDITION_CELLS}
    sleeps: list[float] = []
    moderate = MainCommunicationInterceptor(
        by_name["timeliness_moderate_step4"], sleep_fn=sleeps.append
    ).intercept(4, _message(), context={"eligible": True})
    deadline = MainCommunicationInterceptor(
        by_name["timeliness_deadline_step4"], sleep_fn=sleeps.append
    ).intercept(4, _message(), context={"eligible": True})

    assert sleeps == [0.25, 0.5]
    assert moderate.delivery_count == 1
    assert deadline.delivery_count == 0
    assert deadline.observed_a_symptom == "A2_message_timeout"
