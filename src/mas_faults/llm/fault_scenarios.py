from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_FAULTS = {
    "none",
    "omission",
    "delay",
    "timeout",
    "channel_interruption",
    "malformed_json",
    "truncation",
    "valid_partial",
    "duplicate_request",
    "reordering",
    "stale_replay",
    "endpoint_unavailable",
    "partition_equivalent",
    "message_corruption",
    "schema_mismatch",
    "runtime_state_corruption",
    "prompt_injection",
    "contract_violation",
}


@dataclass(frozen=True)
class FaultScenario:
    fault: str = "none"
    severity: str = "medium"
    delay_ms: int = 1000
    timeout_ms: int = 500
    source_agent: str | None = None
    target_agent: str | None = None

    @property
    def enabled(self) -> bool:
        return self.fault not in {"", "none"}


def validate_fault(fault: str) -> str:
    if fault not in SUPPORTED_FAULTS:
        supported = ", ".join(sorted(SUPPORTED_FAULTS))
        raise ValueError(f"unsupported fault={fault!r}; supported: {supported}")
    return fault


def a_layer_symptom_for_fault(fault: str) -> str:
    return {
        "none": "none",
        "omission": "A5",
        "delay": "A1",
        "timeout": "A2",
        "channel_interruption": "A3",
        "malformed_json": "A7",
        "truncation": "A8",
        "valid_partial": "A8",
        "duplicate_request": "A9",
        "reordering": "A10",
        "stale_replay": "A12",
        "endpoint_unavailable": "A4",
        "partition_equivalent": "A4/A5",
        "message_corruption": "A6",
        "schema_mismatch": "A11",
        "runtime_state_corruption": "A13",
        "prompt_injection": "A14",
        "contract_violation": "A15",
    }.get(fault, "unknown")
