from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedditTopology:
    name: str
    agent_roles: tuple[str, ...]
    message_path: tuple[tuple[str, str], ...]


REDDIT_TOPOLOGIES = {
    "flat": RedditTopology("flat", ("Reader", "Verifier", "Coordinator"), (("Reader", "Verifier"), ("Reader", "Coordinator"), ("Verifier", "Coordinator"))),
    "hierarchical": RedditTopology("hierarchical", ("Reader", "Supervisor"), (("Reader", "Supervisor"),)),
    "team": RedditTopology("team", ("Reader", "Verification Team", "Team Lead"), (("Reader", "Verification Team"), ("Verification Team", "Team Lead"))),
    "hybrid": RedditTopology("hybrid", ("Reader", "Verifier", "Supervisor"), (("Reader", "Verifier"), ("Reader", "Supervisor"), ("Verifier", "Supervisor"))),
}


def get_reddit_topology(name: str) -> RedditTopology:
    try:
        return REDDIT_TOPOLOGIES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported Reddit topology: {name}") from exc
