from __future__ import annotations

from typing import Tuple


def rate_deadline_due(last_deadline_s: float, now_s: float, rate_hz: float) -> Tuple[bool, float]:
    rate = max(1.0, float(rate_hz))
    period = 1.0 / rate
    last = float(last_deadline_s)
    now = float(now_s)
    if last <= 0.0:
        return True, now
    elapsed = now - last
    if elapsed + 1e-9 < period:
        return False, last
    if elapsed > max(1.0, period * 4.0):
        return True, now
    periods = max(1, int((elapsed + 1e-9) / period))
    return True, last + periods * period
