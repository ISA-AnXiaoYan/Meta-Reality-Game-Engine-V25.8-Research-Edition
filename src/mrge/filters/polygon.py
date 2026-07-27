# SPDX-License-Identifier: AGPL-3.0-only
"""Polygon contract checks and stale handling for candidate preview paths."""

from collections.abc import Iterable


def validate_polygon(points: Iterable[Iterable[float]], max_points: int = 128) -> dict[str, object]:
    normalized = [[float(value) for value in point] for point in points]
    if len(normalized) < 3:
        return {"gate_state": "fail", "reason": "too_few_points", "polygon": normalized, "authority": "candidate"}
    if len(normalized) > max_points:
        return {"gate_state": "fail", "reason": "too_many_points", "polygon": normalized[:max_points], "authority": "candidate"}
    if any(len(point) != 2 for point in normalized):
        return {"gate_state": "fail", "reason": "invalid_dimension", "polygon": normalized, "authority": "candidate"}
    return {"gate_state": "pass", "reason": "valid", "polygon": normalized, "authority": "candidate"}


def stale_polygon(age_frames: int, max_age_frames: int) -> dict[str, object]:
    if age_frames < 0 or max_age_frames < 0:
        raise ValueError("polygon ages must be >= 0")
    if age_frames > max_age_frames:
        return {"gate_state": "fail_open", "reason": "stale_polygon", "age_frames": age_frames, "authority": "candidate"}
    return {"gate_state": "pass", "reason": "fresh_polygon", "age_frames": age_frames, "authority": "candidate"}
