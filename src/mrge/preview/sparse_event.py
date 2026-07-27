# SPDX-License-Identifier: AGPL-3.0-only
"""Materialize sparse Event points without constructing a full raster."""

from collections.abc import Iterable, Mapping
from typing import Any


def build_sparse_preview(events: Iterable[Mapping[str, Any]], footprint: int = 4, color: str = "blue") -> dict[str, object]:
    if footprint < 1:
        raise ValueError("footprint must be >= 1")
    points = [{"x": int(event["x"]), "y": int(event["y"]), "polarity": int(event.get("polarity", 1))} for event in events]
    return {
        "schema_version": "mrge.sparse_event_preview.v1",
        "points": points,
        "footprint": footprint,
        "color": color,
        "preview_only": True,
        "authority_write": False,
    }
