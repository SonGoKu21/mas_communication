from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyDefinition:
    name: str
    architecture_taxonomy: str
    agent_roles: tuple[str, ...]
    message_path: tuple[tuple[str, str], ...]


SUPPORTED_TOPOLOGIES = ("sequential", "flat", "hierarchical", "team", "hybrid")


TOPOLOGIES = {
    "sequential": TopologyDefinition(
        name="sequential",
        architecture_taxonomy="sequential",
        agent_roles=("Coordinator", "Shopping Worker", "Verifier"),
        message_path=(
            ("Coordinator", "Shopping Worker"),
            ("Shopping Worker", "WebArena Shopping"),
            ("WebArena Shopping", "Shopping Worker"),
            ("Shopping Worker", "Verifier"),
            ("Verifier", "Coordinator"),
        ),
    ),
    "flat": TopologyDefinition(
        name="flat",
        architecture_taxonomy="flat",
        agent_roles=("Shopping Worker", "Verifier", "Coordinator"),
        message_path=(
            ("Shopping Worker", "WebArena Shopping"),
            ("WebArena Shopping", "Shopping Worker"),
            ("Shopping Worker", "Verifier"),
            ("Shopping Worker", "Coordinator"),
            ("Verifier", "Coordinator"),
        ),
    ),
    "hierarchical": TopologyDefinition(
        name="hierarchical",
        architecture_taxonomy="hierarchical",
        agent_roles=("Supervisor", "Shopping Worker"),
        message_path=(
            ("Supervisor", "Shopping Worker"),
            ("Shopping Worker", "WebArena Shopping"),
            ("WebArena Shopping", "Shopping Worker"),
            ("Shopping Worker", "Supervisor"),
        ),
    ),
    "team": TopologyDefinition(
        name="team",
        architecture_taxonomy="team",
        agent_roles=("Execution Team", "Verification Team", "Team Lead"),
        message_path=(
            ("Execution Team", "WebArena Shopping"),
            ("WebArena Shopping", "Execution Team"),
            ("Execution Team", "Verification Team"),
            ("Verification Team", "Team Lead"),
        ),
    ),
    "hybrid": TopologyDefinition(
        name="hybrid",
        architecture_taxonomy="hybrid",
        agent_roles=("Supervisor", "Shopping Worker", "Verifier"),
        message_path=(
            ("Supervisor", "Shopping Worker"),
            ("Shopping Worker", "WebArena Shopping"),
            ("WebArena Shopping", "Shopping Worker"),
            ("Shopping Worker", "Verifier"),
            ("Shopping Worker", "Supervisor"),
            ("Verifier", "Supervisor"),
        ),
    ),
}


def get_topology(name: str) -> TopologyDefinition:
    try:
        return TOPOLOGIES[name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_TOPOLOGIES)
        raise ValueError(f"unsupported topology={name!r}; supported: {supported}") from exc
