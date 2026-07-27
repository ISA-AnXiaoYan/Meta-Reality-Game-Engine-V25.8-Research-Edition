# SPDX-License-Identifier: AGPL-3.0-only
"""Clean-room event filters with explicit reject reasons."""

from collections.abc import Iterable, Mapping
from typing import Any


def filter_hot_pixels(events: Iterable[Mapping[str, Any]], defect_pixels: set[tuple[int, int]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    rejected = 0
    for event in events:
        point = (int(event["x"]), int(event["y"]))
        if point in defect_pixels:
            rejected += 1
            continue
        kept.append(dict(event))
    return kept, {"input_count": len(kept) + rejected, "kept_count": len(kept), "hot_pixel_reject_count": rejected}


def cluster_events(events: Iterable[Mapping[str, Any]], radius: int = 1) -> list[list[dict[str, Any]]]:
    """Group nearby points deterministically; this is not a hit decision."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    clusters: list[list[dict[str, Any]]] = []
    for event in events:
        point = dict(event)
        x, y = int(point["x"]), int(point["y"])
        target = next((cluster for cluster in clusters if any(abs(x - int(item["x"])) <= radius and abs(y - int(item["y"])) <= radius for item in cluster)), None)
        if target is None:
            clusters.append([point])
        else:
            target.append(point)
    return clusters
