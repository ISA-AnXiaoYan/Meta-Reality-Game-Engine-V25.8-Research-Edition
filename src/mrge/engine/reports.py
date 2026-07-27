# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic, research-only reports derived from replay facts."""

from collections.abc import Iterable
from typing import Any


def build_reports(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    facts = list(rows)
    frames = [row.get("frame", row) for row in facts]
    cameras = sorted({frame.get("camera_id", "unknown") for frame in frames})
    sync = {
        "schema_version": "mrge.sync_report.v1",
        "frame_count": len(frames),
        "cameras": cameras,
        "terminal_state": "replay_analyzed",
        "official_result": False,
    }
    polygon = {
        "schema_version": "mrge.polygon_report.v1",
        "records": [{"camera_id": f.get("camera_id"), "frame_index": f.get("frame_index"), "gate_state": "unknown", "authority": "candidate"} for f in frames],
    }
    world = {
        "schema_version": "mrge.world_state_report.v1",
        "persons": [],
        "authority": "research-only",
    }
    hit = {
        "schema_version": "mrge.hit_candidate_report.v1",
        "candidates": [],
        "official_result": False,
    }
    bev = {
        "schema_version": "mrge.bev_static_report.v1",
        "points": [],
        "authority": "research-only",
    }
    return {"sync_report": sync, "polygon_report": polygon, "world_state_report": world, "hit_candidate_report": hit, "bev_static_report": bev}
