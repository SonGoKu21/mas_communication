"""Shared off-peak guard for DeepSeek API experiment batches."""

from __future__ import annotations

import os
from datetime import datetime, time
from zoneinfo import ZoneInfo


BEIJING = ZoneInfo("Asia/Shanghai")
PEAK_WINDOWS = (
    (time(9, 0), time(12, 0)),
    (time(14, 0), time(18, 0)),
)
TRUTHY = {"1", "true", "yes", "on"}


class DeepSeekPeakWindowError(RuntimeError):
    """Raised before a new DeepSeek API job starts during a peak window."""


def is_deepseek_offpeak(now: datetime | None = None) -> bool:
    current = now or datetime.now(BEIJING)
    if current.tzinfo is None:
        current = current.replace(tzinfo=BEIJING)
    else:
        current = current.astimezone(BEIJING)
    wall_time = current.time().replace(tzinfo=None)
    return not any(start <= wall_time < end for start, end in PEAK_WINDOWS)


def _env_enabled() -> bool:
    return os.getenv("MAS_DEEPSEEK_OFFPEAK_ONLY", "").strip().lower() in TRUTHY


def ensure_deepseek_offpeak(
    model: str,
    *,
    now: datetime | None = None,
    enabled: bool | None = None,
) -> None:
    active = _env_enabled() if enabled is None else enabled
    if not active or "deepseek" not in model.lower() or is_deepseek_offpeak(now):
        return
    current = (now or datetime.now(BEIJING)).astimezone(BEIJING)
    raise DeepSeekPeakWindowError(
        "refusing to start a new DeepSeek API job during the Beijing peak "
        f"window at {current.isoformat(timespec='seconds')}"
    )
