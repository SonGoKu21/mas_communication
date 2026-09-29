from __future__ import annotations
"""Unmodified historical Admin column projection; extracted without runtime imports."""

TASK_EVIDENCE_COLUMNS = {
    "sales_ranking": ("Interval", "Product", "Order Quantity"),
    "sales_product_ranking": ("Interval", "Product", "Order Quantity"),
    "sales_report_aggregation": (
        "Interval",
        "Product",
        "Order Quantity",
        "Orders",
    ),
    "temporal_sales_aggregation": ("Interval", "Orders"),
    "inventory_attribute_lookup": ("SKU", "Name", "Quantity"),
    "order_payment_aggregation": (
        "ID",
        "Purchase Date",
        "Grand Total",
        "Grand Total (Base)",
        "Status",
    ),
    "customer_order_aggregation": (
        "ID",
        "Purchase Date",
        "Bill-to Name",
        "Status",
    ),
    "customer_contact_lookup": ("Name", "Email", "Phone"),
    "order_state_lookup": (
        "ID",
        "Purchase Date",
        "Bill-to Name",
        "Ship-to Name",
        "Status",
    ),
    "customer_cancellation_aggregation": (
        "ID",
        "Purchase Date",
        "Bill-to Name",
        "Grand Total",
        "Status",
    ),
}

def project_visible_evidence(
    task: dict[str, Any], visible_evidence: list[list[str]]
) -> list[list[str]]:
    """Keep task-relevant columns without selecting rows or an answer."""
    if not visible_evidence:
        return []
    header = visible_evidence[0]
    requested = TASK_EVIDENCE_COLUMNS.get(str(task.get("task_stratum")), ())
    indices = [header.index(name) for name in requested if name in header]
    if not indices:
        return visible_evidence
    return [
        [row[index] if index < len(row) else "" for index in indices]
        for row in visible_evidence
    ]
