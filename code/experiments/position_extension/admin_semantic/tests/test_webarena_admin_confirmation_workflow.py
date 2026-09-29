import asyncio
import json

import pytest

from mas_faults.webarena_admin_confirmation import (
    project_visible_evidence,
    run_admin_confirmation_task,
)
from mas_faults.webarena_admin_controlled import ControlledRunError
from mas_faults.webarena_admin_main_matrix import CONDITION_BY_NAME


class SequenceClient:
    def __init__(self, responses):
        from mas_faults.llm_client import ModelInfo

        self.responses = list(responses)
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.request_log = []
        self.prompts = []
        self.model_info = ModelInfo("fake", "http://fake", "fake-model", "fake")

    def complete_with_metadata(self, prompt, *, json_mode=False, metadata=None):
        self.prompts.append(prompt)
        self.call_count += 1
        self.prompt_tokens += len(prompt.split())
        response = self.responses.pop(0)
        self.completion_tokens += len(response.split())
        self.request_log.append(
            {
                "provider_request_id": f"fake-{self.call_count}",
                "agent_role": (metadata or {}).get("agent_role", ""),
            }
        )
        return response

    def complete(self, prompt, *, json_mode=False):
        return self.complete_with_metadata(prompt, json_mode=json_mode, metadata={})


class FakeBrowser:
    def __init__(self):
        self.calls = []
        self.version = 0

    def reset(self, config_file):
        self.calls.append(("reset", config_file))
        return self._state([])

    def tool(self, name, arguments):
        self.calls.append(("tool", name, arguments))
        self.version += 1
        evidence = []
        if name == "read_visible_table":
            evidence = [["ID", "Status"], ["299", "Pending"]]
        return self._state(evidence, name=name, arguments=arguments)

    def _state(self, evidence, *, name="reset", arguments=None):
        return {
            "url": "http://admin/sales/order/",
            "title": "Orders",
            "available_tools": [
                "open_customers",
                "set_range_filter",
                "read_visible_table",
                "finish_with_evidence",
            ],
            "tool_status": "ok",
            "tool_name": name,
            "tool_arguments": arguments or {},
            "tool_subtrace": [],
            "visible_evidence": evidence,
            "state_version": self.version,
        }


class FakeEvaluator:
    def __init__(self, score=1.0):
        self.score = score
        self.calls = []

    def evaluate(self, config_file, answer):
        self.calls.append((config_file, answer))
        return {"score": self.score, "answer": answer}


def _task():
    return {
        "task_id": 199,
        "task_stratum": "order_state_lookup",
        "sites": ["shopping_admin"],
        "intent": "Get the newest pending order ID",
        "eval": {"reference_answers": {"must_include": ["299"]}},
    }


def _decision(answer="299"):
    return json.dumps(
        {"decision": "accept", "answer": answer, "reason": "supported"}
    )


@pytest.mark.parametrize(
    ("task_stratum", "header", "expected_header"),
    [
        (
            "inventory_attribute_lookup",
            ["SKU", "Name", "Quantity", "Price"],
            ["SKU", "Name", "Quantity"],
        ),
        (
            "order_state_lookup",
            ["ID", "Purchase Date", "Bill-to Name", "Ship-to Name", "Status"],
            ["ID", "Purchase Date", "Bill-to Name", "Ship-to Name", "Status"],
        ),
        (
            "customer_cancellation_aggregation",
            ["ID", "Purchase Date", "Bill-to Name", "Grand Total", "Status"],
            ["ID", "Purchase Date", "Bill-to Name", "Grand Total", "Status"],
        ),
        (
            "sales_product_ranking",
            ["Interval", "Product", "Price", "Order Quantity"],
            ["Interval", "Product", "Order Quantity"],
        ),
        (
            "sales_report_aggregation",
            ["Interval", "Product", "Price", "Order Quantity", "Orders"],
            ["Interval", "Product", "Order Quantity", "Orders"],
        ),
        (
            "order_payment_aggregation",
            ["ID", "Purchase Date", "Bill-to Name", "Grand Total", "Status"],
            ["ID", "Purchase Date", "Grand Total", "Status"],
        ),
        (
            "customer_order_aggregation",
            ["ID", "Purchase Date", "Bill-to Name", "Grand Total", "Status"],
            ["ID", "Purchase Date", "Bill-to Name", "Status"],
        ),
        (
            "customer_contact_lookup",
            ["Options", "Name", "Email", "Group", "Phone", "ZIP"],
            ["Name", "Email", "Phone"],
        ),
    ],
)
def test_project_visible_evidence_keeps_all_task_relevant_visible_columns(
    task_stratum,
    header,
    expected_header,
) -> None:
    evidence = [header, [f"v{index}" for index in range(len(header))]]

    projected = project_visible_evidence(
        {"task_stratum": task_stratum},
        evidence,
    )

    assert projected[0] == expected_header


def test_payment_projection_accepts_the_real_magento_base_total_column() -> None:
    evidence = [
        [
            "ID",
            "Purchase Date",
            "Bill-to Name",
            "Grand Total (Base)",
            "Grand Total (Purchased)",
            "Status",
        ],
        ["299", "May 31", "Sarah", "$219.40", "$219.40", "Pending"],
    ]

    projected = project_visible_evidence(
        {"task_stratum": "order_payment_aggregation"},
        evidence,
    )

    assert projected[0] == [
        "ID",
        "Purchase Date",
        "Grand Total (Base)",
        "Status",
    ]


def _evidence(answer="299", indices=None):
    return json.dumps(
        {
            "candidate_answer": answer,
            "evidence_summary": "visible row",
            "evidence_row_indices": [1] if indices is None else indices,
        }
    )


def test_step2_omission_occurs_before_browser_execution_and_can_naturally_repeat(
    tmp_path,
) -> None:
    config = tmp_path / "199.json"
    config.write_text(json.dumps(_task()), encoding="utf-8")
    client = SequenceClient(
        [
            "Filter and read.",
            '{"tool":"set_range_filter","arguments":{"field":"ID","from_value":"299","to_value":"299"}}',
            '{"tool":"set_range_filter","arguments":{"field":"ID","from_value":"299","to_value":"299"}}',
            '{"tool":"read_visible_table","arguments":{}}',
            '{"tool":"finish_with_evidence","arguments":{}}',
            _evidence(),
            _decision(),
            _decision(),
        ]
    )
    browser = FakeBrowser()

    record = asyncio.run(
        run_admin_confirmation_task(
            client,
            browser,
            FakeEvaluator(),
            _task(),
            original_config_file=config,
            sanitized_config_dir=tmp_path / "sanitized",
            topology="sequential",
            condition_cell=CONDITION_BY_NAME["non_delivery_step2"],
            run_index=1,
            max_steps=5,
        )
    )

    filtered_calls = [call for call in browser.calls if call[0] == "tool" and call[1] == "set_range_filter"]
    assert len(filtered_calls) == 1
    fault = next(event for event in record["events"] if event["fault_applied"])
    assert fault["abstract_step"] == 2
    assert fault["delivered_messages"] == []
    assert record["fault_applied"] is True
    assert record["final_task_success"] is True
    assert record["model"] == "fake-model"
    assert record["provider"] == "fake"


def test_step2_omission_can_target_read_action_when_no_filter_is_used(tmp_path) -> None:
    config = tmp_path / "199.json"
    config.write_text(json.dumps(_task()), encoding="utf-8")
    client = SequenceClient(
        [
            "Open and read.",
            '{"tool":"open_customers","arguments":{}}',
            '{"tool":"read_visible_table","arguments":{}}',
            '{"tool":"read_visible_table","arguments":{}}',
            '{"tool":"finish_with_evidence","arguments":{}}',
            _evidence(),
            _decision(),
            _decision(),
        ]
    )
    browser = FakeBrowser()

    record = asyncio.run(
        run_admin_confirmation_task(
            client,
            browser,
            FakeEvaluator(),
            _task(),
            original_config_file=config,
            sanitized_config_dir=tmp_path / "sanitized",
            topology="sequential",
            condition_cell=CONDITION_BY_NAME["non_delivery_step2"],
            run_index=1,
            max_steps=5,
        )
    )

    read_calls = [
        call
        for call in browser.calls
        if call[0] == "tool" and call[1] == "read_visible_table"
    ]
    assert len(read_calls) == 1
    fault = next(event for event in record["events"] if event["fault_applied"])
    assert fault["abstract_step"] == 2
    assert fault["original_message"]["tool"] == "read_visible_table"
    assert fault["delivered_messages"] == []


def test_step3_reordering_makes_the_older_observation_the_consumed_state(tmp_path) -> None:
    config = tmp_path / "199.json"
    config.write_text(json.dumps(_task()), encoding="utf-8")
    client = SequenceClient(
        [
            "Filter and read.",
            '{"tool":"set_range_filter","arguments":{"field":"ID","from_value":"299","to_value":"299"}}',
            '{"tool":"read_visible_table","arguments":{}}',
            '{"tool":"finish_with_evidence","arguments":{}}',
            _evidence("N/A", []),
            _decision("N/A"),
            _decision("N/A"),
        ]
    )

    record = asyncio.run(
        run_admin_confirmation_task(
            client,
            FakeBrowser(),
            FakeEvaluator(score=0.0),
            _task(),
            original_config_file=config,
            sanitized_config_dir=tmp_path / "sanitized",
            topology="sequential",
            condition_cell=CONDITION_BY_NAME["same_session_reordering_step3"],
            run_index=1,
            max_steps=4,
        )
    )

    fault = next(event for event in record["events"] if event["fault_applied"])
    assert fault["abstract_step"] == 3
    assert [item["state_version"] for item in fault["delivered_messages"]] == [2, 1]
    assert record["accepted_visible_evidence"] == []
    assert record["final_task_success"] is False


def test_step2_duplicate_executes_the_same_valid_tool_request_twice(tmp_path) -> None:
    config = tmp_path / "199.json"
    config.write_text(json.dumps(_task()), encoding="utf-8")
    client = SequenceClient(
        [
            "Filter and read.",
            '{"tool":"set_range_filter","arguments":{"field":"ID","from_value":"299","to_value":"299"}}',
            '{"tool":"read_visible_table","arguments":{}}',
            '{"tool":"finish_with_evidence","arguments":{}}',
            _evidence(),
            _decision(),
            _decision(),
        ]
    )
    browser = FakeBrowser()

    record = asyncio.run(
        run_admin_confirmation_task(
            client,
            browser,
            FakeEvaluator(),
            _task(),
            original_config_file=config,
            sanitized_config_dir=tmp_path / "sanitized",
            topology="flat",
            condition_cell=CONDITION_BY_NAME["duplicate_delivery_step2"],
            run_index=1,
            max_steps=4,
        )
    )

    filtered_calls = [call for call in browser.calls if call[0] == "tool" and call[1] == "set_range_filter"]
    assert len(filtered_calls) == 2
    fault = next(event for event in record["events"] if event["fault_applied"])
    assert fault["delivery_count"] == 2
    assert record["topology"] == "flat"
    assert record["used_edges"] == [
        ["Evidence Worker", "Verifier"],
        ["Evidence Worker", "Coordinator"],
        ["Verifier", "Coordinator"],
    ]
    direct_events = [
        event
        for event in record["events"]
        if event["source_agent"] == "Evidence Worker"
        and event["target_agent"] == "Coordinator"
    ]
    assert len(direct_events) == 1
    assert direct_events[0]["fault_applied"] is False
    assert direct_events[0]["observed_runtime_effect"] == "clean_delivery"


def test_invalid_tool_request_preserves_raw_llm_output_in_step2_trace(tmp_path) -> None:
    config = tmp_path / "199.json"
    config.write_text(json.dumps(_task()), encoding="utf-8")
    client = SequenceClient(
        [
            "Filter and read.",
            '{"tool":"set_range_filter","arguments":{"field":"ID"}}',
        ]
    )

    with pytest.raises(ControlledRunError) as captured:
        asyncio.run(
            run_admin_confirmation_task(
                client,
                FakeBrowser(),
                FakeEvaluator(),
                _task(),
                original_config_file=config,
                sanitized_config_dir=tmp_path / "sanitized",
                topology="sequential",
                condition_cell=CONDITION_BY_NAME["clean"],
                run_index=1,
                max_steps=1,
            )
        )

    event = captured.value.events[-1]
    assert event["abstract_step"] == 2
    assert event["original_message"] == (
        '{"tool":"set_range_filter","arguments":{"field":"ID"}}'
    )
    assert event["observed_runtime_effect"] == "tool_request_parse_failure"
