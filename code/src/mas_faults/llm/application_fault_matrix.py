from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicationFaultSpec:
    code: str
    name: str
    label: str
    steps: tuple[int, ...]


APPLICATION_FAULTS = (
    ApplicationFaultSpec("A1", "delay", "Message latency", (2, 3, 4)),
    ApplicationFaultSpec("A2", "timeout", "Message timeout", (2, 3, 4)),
    ApplicationFaultSpec("A3", "channel_interruption", "Channel interruption", (2, 3, 4)),
    ApplicationFaultSpec("A4", "endpoint_unavailable", "Endpoint unavailable", (2, 3, 4)),
    ApplicationFaultSpec("A5", "omission", "Message omission", (2, 3, 4)),
    ApplicationFaultSpec("A6", "message_corruption", "Semantic message corruption", (2, 3, 4)),
    ApplicationFaultSpec("A7", "malformed_json", "Malformed message", (2, 3, 4)),
    ApplicationFaultSpec("A8", "truncation", "Message truncation", (2, 3, 4)),
    ApplicationFaultSpec("A9", "duplicate_request", "Duplicate execution request", (2,)),
    ApplicationFaultSpec("A10", "reordering", "Runtime message reordering", (3, 4)),
    ApplicationFaultSpec("A11", "schema_mismatch", "Schema mismatch", (2, 3, 4)),
    ApplicationFaultSpec("A12", "stale_replay", "Timing or session mismatch", (2, 3, 4)),
    ApplicationFaultSpec("A13", "runtime_state_corruption", "Runtime communication-state corruption", (2, 3, 4)),
    ApplicationFaultSpec("A14", "prompt_injection", "Instruction injection through message channel", (3, 4)),
    ApplicationFaultSpec("A15", "contract_violation", "Output contract violation", (2, 3, 4)),
)


def fault_names_for_steps(steps: tuple[int, ...]) -> tuple[str, ...]:
    requested = set(steps)
    return tuple(spec.name for spec in APPLICATION_FAULTS if requested.intersection(spec.steps))


def steps_for_fault(name: str) -> tuple[int, ...]:
    for spec in APPLICATION_FAULTS:
        if spec.name == name:
            return spec.steps
    raise ValueError(f"unsupported application fault: {name}")
