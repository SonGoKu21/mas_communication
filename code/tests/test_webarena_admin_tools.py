from __future__ import annotations

import pytest

from mas_faults.webarena_admin_tools import (
    parse_admin_tool_request,
    tool_catalog_text,
)


def test_parse_strict_json_tool_request_with_exact_arguments():
    request = parse_admin_tool_request(
        '{"tool":"set_text_filter","arguments":{"field":"Quantity","value":"0"}}'
    )

    assert request.name == "set_text_filter"
    assert request.arguments == {"field": "Quantity", "value": "0"}


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"tool":"add_product","arguments":{}}\n```',
        '```json\n{"tool":"open_catalog_products","arguments":{"answer":"secret"}}\n```',
        '```json\n{"tool":"set_text_filter","arguments":{"field":"Quantity"}}\n```',
        '```json\n{"tool":"set_text_filter","arguments":{"field":"Quantity","value":"0","extra":"x"}}\n```',
        '```json\n{"tool":"read_visible_table","arguments":{}}\n```',
        '```json\n{"tool":"read_visible_table","arguments":{}}\n```\n```json\n{"tool":"finish_with_evidence","arguments":{}}\n```',
    ],
)
def test_rejects_mutating_unknown_ambiguous_or_invalid_tool_requests(raw):
    with pytest.raises(ValueError, match="tool request"):
        parse_admin_tool_request(raw)


def test_catalog_lists_only_read_only_tools_and_their_exact_arguments():
    catalog = tool_catalog_text()

    assert "open_catalog_products()" in catalog
    assert "open_customers()" in catalog
    assert "open_sales_orders()" in catalog
    assert "open_sales_orders_report()" in catalog
    assert "open_search_terms()" in catalog
    assert "set_text_filter(field, value)" in catalog
    assert "field: Quantity|SKU|Name|Email|Phone" in catalog
    assert "set_range_filter(field, from_value, to_value)" in catalog
    assert "set_date_range(from_date, to_date, period)" in catalog
    assert "show_report()" in catalog
    assert "read_table_head(max_rows)" in catalog
    assert "MM/DD/YYYY" in catalog
    assert "Day|Month|Year" in catalog
    assert "all argument values are JSON strings" in catalog
    assert "Year=yearly, Month=monthly, Day=daily" in catalog
    assert "Status: Canceled|Closed|Complete" in catalog
    assert "Pending|Pending Payment|Processing" in catalog
    assert "finish_with_evidence()" in catalog
    assert "add_product" not in catalog.lower()
    assert "delete" not in catalog.lower()
    assert "save" not in catalog.lower()
    assert "individual order records" in catalog
    assert "use for customer order counts" in catalog
    assert "aggregate order counts by time period" in catalog
    assert "aggregate product sales ranking" in catalog
    assert "individual customer records" in catalog
    assert "does not contain order counts" in catalog
    assert "call open_filters() first" in catalog


def test_sales_orders_report_is_a_read_only_zero_argument_tool():
    request = parse_admin_tool_request(
        '{"tool":"open_sales_orders_report","arguments":{}}'
    )

    assert request.name == "open_sales_orders_report"
    assert request.arguments == {}


def test_search_terms_is_a_read_only_zero_argument_tool():
    request = parse_admin_tool_request(
        '{"tool":"open_search_terms","arguments":{}}'
    )

    assert request.name == "open_search_terms"
    assert request.arguments == {}


def test_catalog_can_be_restricted_to_tools_available_on_the_current_page():
    catalog = tool_catalog_text(
        available_tools=(
            "set_date_range",
            "show_report",
            "read_visible_table",
            "finish_with_evidence",
        )
    )

    assert "set_date_range(from_date, to_date, period)" in catalog
    assert "show_report()" in catalog
    assert "apply_filters()" not in catalog
    assert "open_filters()" not in catalog


@pytest.mark.parametrize(
    "raw",
    [
        '{"tool":"set_date_range","arguments":{"from_date":"2022-01-01","to_date":"12/31/2022","period":"Year"}}',
        '{"tool":"set_date_range","arguments":{"from_date":"01/01/2022","to_date":"12/31/2022","period":"day"}}',
    ],
)
def test_date_range_rejects_values_outside_the_frozen_magento_contract(raw):
    with pytest.raises(ValueError, match="date|period"):
        parse_admin_tool_request(raw)


def test_range_filter_requires_exact_three_argument_contract():
    request = parse_admin_tool_request(
        '{"tool":"set_range_filter","arguments":{"field":"Quantity","from_value":"1","to_value":"3"}}'
    )

    assert request.arguments == {
        "field": "Quantity",
        "from_value": "1",
        "to_value": "3",
    }


def test_status_filter_uses_exact_magento_option_labels():
    request = parse_admin_tool_request(
        '{"tool":"set_select_filter","arguments":{"field":"Status","value":"Canceled"}}'
    )
    assert request.arguments == {"field": "Status", "value": "Canceled"}

    with pytest.raises(ValueError, match="Status"):
        parse_admin_tool_request(
            '{"tool":"set_select_filter","arguments":{"field":"Status","value":"Cancelled"}}'
        )


def test_table_head_limit_is_a_small_string_encoded_integer():
    request = parse_admin_tool_request(
        '{"tool":"read_table_head","arguments":{"max_rows":"5"}}'
    )
    assert request.arguments == {"max_rows": "5"}

    with pytest.raises(ValueError, match="max_rows"):
        parse_admin_tool_request(
            '{"tool":"read_table_head","arguments":{"max_rows":"200"}}'
        )
