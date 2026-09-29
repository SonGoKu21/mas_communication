from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mas_faults.deepseek_schedule import (
    DeepSeekPeakWindowError,
    ensure_deepseek_offpeak,
    is_deepseek_offpeak,
)


BEIJING = ZoneInfo("Asia/Shanghai")


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (8, 59, True),
        (9, 0, False),
        (11, 59, False),
        (12, 0, True),
        (13, 59, True),
        (14, 0, False),
        (17, 59, False),
        (18, 0, True),
    ],
)
def test_offpeak_boundaries(hour: int, minute: int, expected: bool) -> None:
    now = datetime(2026, 8, 17, hour, minute, tzinfo=BEIJING)

    assert is_deepseek_offpeak(now) is expected


def test_guard_rejects_deepseek_job_in_peak_window() -> None:
    now = datetime(2026, 8, 17, 10, 0, tzinfo=BEIJING)

    with pytest.raises(DeepSeekPeakWindowError):
        ensure_deepseek_offpeak("deepseek-v4-flash", now=now, enabled=True)


def test_guard_does_not_restrict_non_deepseek_model() -> None:
    now = datetime(2026, 8, 17, 10, 0, tzinfo=BEIJING)

    ensure_deepseek_offpeak("qwen3-8b", now=now, enabled=True)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_guard_cannot_be_disabled_for_deepseek(monkeypatch, enabled) -> None:
    monkeypatch.delenv("MAS_DEEPSEEK_OFFPEAK_ONLY", raising=False)
    now = datetime(2026, 8, 17, 10, 0, tzinfo=BEIJING)

    monkeypatch.setenv("MAS_DEEPSEEK_OFFPEAK_ONLY", "0")
    with pytest.raises(DeepSeekPeakWindowError):
        ensure_deepseek_offpeak("deepseek-v4-flash", now=now, enabled=enabled)


@pytest.mark.parametrize("day", [12, 13])
@pytest.mark.parametrize("hour", [0, 9, 11, 14, 17, 23])
def test_beijing_weekend_is_offpeak_all_day(day, hour):
    assert is_deepseek_offpeak(datetime(2026, 9, day, hour, tzinfo=BEIJING))


def test_weekend_is_determined_after_timezone_conversion():
    utc = ZoneInfo("UTC")
    assert is_deepseek_offpeak(datetime(2026, 9, 12, 2, tzinfo=utc))
    assert not is_deepseek_offpeak(datetime(2026, 9, 14, 2, tzinfo=utc))
