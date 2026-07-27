# SPDX-License-Identifier: Apache-2.0
"""Compatibility source copy of deterministic replay fault injection."""

from copy import deepcopy
from typing import Any, Iterable

SUPPORTED_FAULTS = {"missing_frame", "duplicate_frame", "out_of_order", "stale_polygon", "empty_event"}


def inject(rows: Iterable[dict[str, Any]], faults: Iterable[str]) -> list[dict[str, Any]]:
    result = [deepcopy(row) for row in rows]
    selected = list(faults)
    unknown = sorted(set(selected) - SUPPORTED_FAULTS)
    if unknown:
        raise ValueError(f"unsupported faults: {', '.join(unknown)}")
    if "missing_frame" in selected and result:
        result = [row for row in result if row.get("frame", {}).get("frame_index") != 1]
    if "duplicate_frame" in selected and result:
        result.insert(1, deepcopy(result[0]))
    if "out_of_order" in selected and len(result) >= 2:
        result[0], result[1] = result[1], result[0]
    if "stale_polygon" in selected:
        for row in result:
            row.setdefault("diagnostics", {})["polygon_age_frames"] = 999
    if "empty_event" in selected:
        for row in result:
            row.setdefault("frame", {}).setdefault("payload", {})["event_pixels"] = []
    return result
