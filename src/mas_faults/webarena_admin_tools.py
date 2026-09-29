"""Strict high-level read-only tool protocol for WebArena Shopping Admin."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass


TOOL_ARGUMENTS: dict[str, frozenset[str]] = {
    "open_catalog_products": frozenset(),
    "open_customers": frozenset(),
    "open_sales_orders": frozenset(),
    "open_sales_orders_report": frozenset(),
    "open_bestsellers_report": frozenset(),
    "open_search_terms": frozenset(),
    "reveal_data_grid": frozenset(),
    "open_filters": frozenset(),
    "set_text_filter": frozenset({"field", "value"}),
    "set_range_filter": frozenset({"field", "from_value", "to_value"}),
    "set_select_filter": frozenset({"field", "value"}),
    "set_date_range": frozenset({"from_date", "to_date", "period"}),
    "apply_filters": frozenset(),
    "show_report": frozenset(),
    "read_visible_table": frozenset(),
    "read_table_head": frozenset({"max_rows"}),
    "finish_with_evidence": frozenset(),
}
STATUS_FILTER_VALUES = frozenset(
    {
        "Canceled",
        "Closed",
        "Complete",
        "Suspected Fraud",
        "On Hold",
        "Payment Review",
        "PayPal Canceled Reversal",
        "PayPal Reversed",
        "Pending",
        "Pending Payment",
        "Pending PayPal",
        "Processing",
    }
)


@dataclass(frozen=True)
class AdminToolRequest:
    name: str
    arguments: dict[str, str]


def validate_admin_tool_call(name: object, arguments: object) -> AdminToolRequest:
    """Validate an already decoded tool call against the read-only contract."""
    if not isinstance(name, str) or name not in TOOL_ARGUMENTS:
        raise ValueError("tool request names an unsupported tool")
    if not isinstance(arguments, dict) or set(arguments) != TOOL_ARGUMENTS[name]:
        raise ValueError("tool request arguments do not match the tool contract")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in arguments.items()):
        raise ValueError("tool request argument values must be strings")
    if name == "set_date_range":
        for key in ("from_date", "to_date"):
            value = arguments[key]
            try:
                parsed = datetime.strptime(value, "%m/%d/%Y")
            except ValueError as exc:
                raise ValueError(f"{key} must use MM/DD/YYYY") from exc
            if parsed.strftime("%m/%d/%Y") != value:
                raise ValueError(f"{key} must use zero-padded MM/DD/YYYY")
        if arguments["period"] not in {"Day", "Month", "Year"}:
            raise ValueError("period must be exactly Day, Month, or Year")
    if name == "set_range_filter" and arguments["field"] not in {
        "Quantity",
        "Price",
        "ID",
    }:
        raise ValueError("range filter field must be exactly Quantity, Price, or ID")
    if name == "set_select_filter":
        if arguments["field"] != "Status":
            raise ValueError("select filter field must be exactly Status")
        if arguments["value"] not in STATUS_FILTER_VALUES:
            raise ValueError("Status value must use an exact Magento option label")
    if name == "read_table_head":
        try:
            max_rows = int(arguments["max_rows"])
        except ValueError as exc:
            raise ValueError("max_rows must be an integer string") from exc
        if str(max_rows) != arguments["max_rows"] or not 1 <= max_rows <= 20:
            raise ValueError("max_rows must be a canonical integer string from 1 to 20")
    return AdminToolRequest(name=name, arguments=dict(arguments))


def parse_admin_tool_request(raw: str) -> AdminToolRequest:
    """Parse one strict JSON request and enforce the read-only contract."""
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("tool request contains invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        raise ValueError("tool request must contain only tool and arguments")
    return validate_admin_tool_call(payload["tool"], payload["arguments"])


def tool_catalog_text(
    *, available_tools: tuple[str, ...] | list[str] | None = None
) -> str:
    """Return the exact signatures available to the Tool Navigator."""
    lines = [
        "Contract: all argument values are JSON strings.",
        "Date aggregation: Year=yearly, Month=monthly, Day=daily.",
    ]
    names = list(TOOL_ARGUMENTS) if available_tools is None else list(available_tools)
    unknown = [name for name in names if name not in TOOL_ARGUMENTS]
    if unknown:
        raise ValueError(f"unknown available tools: {unknown}")
    for name in names:
        arguments = TOOL_ARGUMENTS[name]
        ordered = {
            "set_text_filter": ("field", "value"),
            "set_range_filter": ("field", "from_value", "to_value"),
            "set_select_filter": ("field", "value"),
            "set_date_range": ("from_date", "to_date", "period"),
            "read_table_head": ("max_rows",),
        }.get(name, tuple(sorted(arguments)))
        suffix = ""
        if name == "open_sales_orders":
            suffix = (
                " [individual order records: IDs, dates, customers, totals, and status; "
                "use for customer order counts]"
            )
        elif name == "open_customers":
            suffix = (
                " [individual customer records: name, email, and phone; "
                "does not contain order counts]"
            )
        elif name == "open_sales_orders_report":
            suffix = " [aggregate order counts by time period: day, month, or year]"
        elif name == "open_bestsellers_report":
            suffix = " [aggregate product sales ranking over a date range]"
        elif name == "open_catalog_products":
            suffix = " [individual product records: SKU, name, price, and quantity]"
        elif name == "open_search_terms":
            suffix = " [read-only search queries ranked by their Uses count]"
        elif name == "set_date_range":
            suffix = " [dates: MM/DD/YYYY; period: Day|Month|Year]"
        elif name == "set_range_filter":
            suffix = (
                " [call open_filters() first; field: Quantity|Price|ID; "
                "inclusive bounds]"
            )
        elif name == "set_text_filter":
            suffix = (
                " [call open_filters() first; field: Quantity|SKU|Name|Email|Phone; "
                "exact visible-grid filter]"
            )
        elif name == "set_select_filter":
            suffix = (
                " [call open_filters() first; Status: "
                "Canceled|Closed|Complete|Pending|Pending Payment|Processing; "
                "use exact Magento label]"
            )
        elif name == "read_table_head":
            suffix = (
                " [max_rows: number of data rows, excluding the header; "
                "string integer 1..20; use for newest/most recent]"
            )
        lines.append(f"- {name}({', '.join(ordered)}){suffix}")
    return "\n".join(lines)
