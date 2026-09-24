"""Shared off-peak guard for DeepSeek API experiment batches."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


BEIJING = ZoneInfo("Asia/Shanghai")
PEAK_WINDOWS = (
    (time(9, 0), time(12, 0)),
    (time(14, 0), time(18, 0)),
)


class DeepSeekPeakWindowError(RuntimeError):
    """Raised before a DeepSeek job or paid request during a peak window."""


def is_deepseek_offpeak(now: datetime | None = None) -> bool:
    current = now or datetime.now(BEIJING)
    if current.tzinfo is None:
        current = current.replace(tzinfo=BEIJING)
    else:
        current = current.astimezone(BEIJING)
    if current.weekday() >= 5:
        return True
    wall_time = current.time().replace(tzinfo=None)
    return not any(start <= wall_time < end for start, end in PEAK_WINDOWS)


def ensure_deepseek_offpeak(
    model: str,
    *,
    now: datetime | None = None,
    enabled: bool | None = None,
) -> None:
    # Retain the old argument for callers, but it can no longer disable safety.
    if "deepseek" not in model.lower() or is_deepseek_offpeak(now):
        return
    current = now or datetime.now(BEIJING)
    current = current.replace(tzinfo=BEIJING) if current.tzinfo is None else current.astimezone(BEIJING)
    raise DeepSeekPeakWindowError(
        "refusing to start a DeepSeek API job or request during the Beijing peak "
        f"window at {current.isoformat(timespec='seconds')}"
    )
