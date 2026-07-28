#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hit_judge_server.py

Bridge the existing event-camera bullet events into the current MVS/YOLO human
region system.

Inputs
------
1) UDP BulletEvent packets from event-camera processes, usually port 5003.
2) sync_ipc/human_result_{camera}.jsonl written by the MVS YOLO process.
3) sync_ipc/event_trigger_{camera}.csv for MVS sync-index -> event timestamp.
4) event->MVS homographies from hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json.

Output
------
sync_ipc/hit_candidate_{camera}.jsonl and sync_ipc/hit_candidate_all.jsonl.

The server does not make a final game-rule decision.  It emits scored hit
candidates so thresholds can be tuned without touching the event/MVS pipelines.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import select
import socket
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from hit_protocol import (
        BulletEvent,
        HitCandidate,
        ScoreBreakdown,
        MATCH_POINT_IN_BBOX,
        MATCH_POINT_NEAR_BBOX,
        MATCH_POINT_ONLY,
        MATCH_SEGMENT_INTERSECT,
        EVENT_TERMINATE,
        EVENT_TURN,
    )
except Exception:
    # Allow running from rentijiance/.
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from hit_protocol import (  # type: ignore
        BulletEvent,
        HitCandidate,
        ScoreBreakdown,
        MATCH_POINT_IN_BBOX,
        MATCH_POINT_NEAR_BBOX,
        MATCH_POINT_ONLY,
        MATCH_SEGMENT_INTERSECT,
        EVENT_TERMINATE,
        EVENT_TURN,
    )


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _candidate_config_paths(config_arg: str) -> List[Path]:
    raw = Path(config_arg).expanduser()
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    out: List[Path] = []
    if raw.is_absolute():
        out.append(raw)
    else:
        out.extend([cwd / raw, script_dir / raw])
        if cwd.name == "rentijiance":
            out.extend([cwd.parent / raw.name, cwd.parent / raw])
        if script_dir.name == "rentijiance":
            out.extend([script_dir.parent / raw.name, script_dir.parent / raw])
    seen = set()
    uniq = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def _load_json_config(config_arg: str) -> Tuple[Dict[str, Any], str]:
    errors = []
    for p in _candidate_config_paths(config_arg):
        if not p.exists():
            continue
        try:
            return _load_json(str(p)), str(p.resolve())
        except Exception as exc:
            errors.append(f"{p}: {exc}")
    raise RuntimeError("could not load config: " + "; ".join(errors))


def _routes_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    routes = cfg.get("routes")
    if isinstance(routes, list) and routes:
        return [r for r in routes if isinstance(r, dict) and _as_bool(r.get("enabled", True), True)]
    active = str(cfg.get("active_profile", "") or "")
    prof = (cfg.get("profiles") or {}).get(active) if isinstance(cfg.get("profiles"), dict) else None
    if isinstance(prof, dict):
        routes = prof.get("routes") or []
        return [r for r in routes if isinstance(r, dict) and _as_bool(r.get("enabled", True), True)]
    return []


def _sync_path(sync_cfg: Dict[str, Any], key: str, camera: str, default_template: str) -> str:
    root = str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
    template = str(sync_cfg.get(key, default_template) or default_template)
    p = template.format(root=root, camera=str(camera), cam=str(camera), id=str(camera))
    return os.path.abspath(os.path.expanduser(p))


def _event_serial_for_camera(sync_cfg: Dict[str, Any], camera: str) -> str:
    d = sync_cfg.get("event_serials") if isinstance(sync_cfg, dict) else {}
    return str((d or {}).get(str(camera), "") or "") if isinstance(d, dict) else ""


def _homography_for_camera(cfg: Dict[str, Any], cam_id: str) -> np.ndarray:
    overlay_base = cfg.get("event_overlay") if isinstance(cfg.get("event_overlay"), dict) else {}
    per_route = overlay_base.get("per_route") if isinstance(overlay_base.get("per_route"), dict) else {}
    rcfg = dict(overlay_base)
    rcfg.update(per_route.get(str(cam_id), {}) if isinstance(per_route, dict) else {})
    hs = rcfg.get("homographies_event_to_mvs") if isinstance(rcfg.get("homographies_event_to_mvs"), dict) else {}
    active = str(rcfg.get("active_homography", "") or "")
    H = None
    if active and active in hs:
        H = hs.get(active)
    elif "default" in hs:
        H = hs.get("default")
    elif hs:
        H = next(iter(hs.values()))
    if H is not None:
        arr = np.array(H, dtype=np.float64)
        if arr.shape == (3, 3):
            return arr
    # Fallback for old center_offset overlay.  This is not as accurate as calibrated homography.
    scale = float(rcfg.get("scale", 1.0) or 1.0)
    x = float(rcfg.get("x", 0.0) or 0.0)
    y = float(rcfg.get("y", 0.0) or 0.0)
    return np.array([[scale, 0.0, x], [0.0, scale, y], [0.0, 0.0, 1.0]], dtype=np.float64)


def _apply_H(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    v = H @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    if abs(float(v[2])) < 1e-9:
        return float(x), float(y)
    return float(v[0] / v[2]), float(v[1] / v[2])


def _expand_bbox_xyxy(box: Iterable[float], px: float, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    vals = list(box)
    x1, y1, x2, y2 = [float(v) for v in vals[:4]]
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(1.0, float(vals[2]))
        y2 = y1 + max(1.0, float(vals[3]))
    return (
        max(0.0, x1 - px),
        max(0.0, y1 - px),
        min(float(max(1, frame_w)), x2 + px),
        min(float(max(1, frame_h)), y2 + px),
    )


def _point_in_rect(p: Tuple[float, float], r: Tuple[float, float, float, float]) -> bool:
    x, y = p
    x1, y1, x2, y2 = r
    return x1 <= x <= x2 and y1 <= y <= y2


def _dist_to_rect(p: Tuple[float, float], r: Tuple[float, float, float, float]) -> float:
    x, y = p
    x1, y1, x2, y2 = r
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _segments_intersect(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    def on_seg(p, q, r):
        return min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6 and min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
    o1 = orient(a, b, c); o2 = orient(a, b, d); o3 = orient(c, d, a); o4 = orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-6 and on_seg(a, c, b): return True
    if abs(o2) < 1e-6 and on_seg(a, d, b): return True
    if abs(o3) < 1e-6 and on_seg(c, a, d): return True
    if abs(o4) < 1e-6 and on_seg(c, b, d): return True
    return False


def _segment_intersects_rect(a: Tuple[float, float], b: Tuple[float, float], r: Tuple[float, float, float, float]) -> bool:
    if _point_in_rect(a, r) or _point_in_rect(b, r):
        return True
    x1, y1, x2, y2 = r
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    return any(_segments_intersect(a, b, c, d) for c, d in edges)


def _point_in_polygon(p: Tuple[float, float], poly: Iterable[Iterable[float]]) -> bool:
    pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
    if len(pts) < 3:
        return False
    x, y = p
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)):
            denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
            x_cross = (xj - xi) * (y - yi) / denom + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _segment_intersects_polygon(a: Tuple[float, float], b: Tuple[float, float], poly: Iterable[Iterable[float]]) -> bool:
    pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
    if len(pts) < 3:
        return False
    if _point_in_polygon(a, pts) or _point_in_polygon(b, pts):
        return True
    for i in range(len(pts)):
        c = pts[i]
        d = pts[(i + 1) % len(pts)]
        if _segments_intersect(a, b, c, d):
            return True
    return False



def _point_segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def _dist_to_polygon(p: Tuple[float, float], poly: Iterable[Iterable[float]]) -> float:
    pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
    if len(pts) < 3:
        return 1e9
    if _point_in_polygon(p, pts):
        return 0.0
    return min(_point_segment_distance(p, pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))


def _segment_intersection_point(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], float]]:
    """Return (intersection_xy, t_on_ab) for two finite segments when possible."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denom = rx * sy - ry * sx
    qpx, qpy = cx - ax, cy - ay
    if abs(denom) < 1e-9:
        return None
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        t2 = max(0.0, min(1.0, float(t)))
        return ((ax + t2 * rx, ay + t2 * ry), t2)
    return None


def _segment_polygon_first_intersection(
    a: Tuple[float, float],
    b: Tuple[float, float],
    poly: Iterable[Iterable[float]],
) -> Optional[Tuple[Tuple[float, float], float]]:
    pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
    if len(pts) < 3:
        return None
    best: Optional[Tuple[Tuple[float, float], float]] = None
    for i in range(len(pts)):
        got = _segment_intersection_point(a, b, pts[i], pts[(i + 1) % len(pts)])
        if got is None:
            continue
        if best is None or got[1] < best[1]:
            best = got
    return best


def _angle_between_vectors_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return float(math.degrees(math.acos(c)))


def _person_projected_impact(
    person: Dict[str, Any],
    event_pt: Tuple[float, float],
    approach_pt: Tuple[float, float],
    approach_valid: bool,
    extend_px: float,
    max_last_point_dist_px: float,
    min_recent_len_px: float,
) -> Optional[Dict[str, Any]]:
    """
    Only create a pending impact, not a final hit.

    With the upstream human digital mask enabled, a real bullet usually loses
    events when it reaches the body.  Therefore the evidence we can still trust
    is: an external fast track approaches the current body mask boundary.  The
    final hit must then be confirmed by later behaviour: disappear near boundary
    or bounce near boundary.
    """
    if not approach_valid:
        return None
    vx = float(event_pt[0]) - float(approach_pt[0])
    vy = float(event_pt[1]) - float(approach_pt[1])
    recent_len = math.hypot(vx, vy)
    if recent_len < float(min_recent_len_px):
        return None
    ux, uy = vx / max(1e-9, recent_len), vy / max(1e-9, recent_len)
    projected_end = (float(event_pt[0]) + ux * float(extend_px), float(event_pt[1]) + uy * float(extend_px))

    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    if isinstance(polys, dict):
        polys = list(polys.values())
    if not isinstance(polys, list):
        return None

    best: Optional[Dict[str, Any]] = None
    for poly in polys:
        if not isinstance(poly, list):
            continue
        pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
        if len(pts) < 3:
            continue
        approach_inside = _point_in_polygon(approach_pt, pts)
        event_inside = _point_in_polygon(event_pt, pts)
        # A track segment that is already born inside the body is exactly the
        # human-motion false-positive class we are trying to suppress.
        if approach_inside:
            continue
        approach_dist = _dist_to_polygon(approach_pt, pts)
        last_dist = _dist_to_polygon(event_pt, pts)
        if last_dist > float(max_last_point_dist_px):
            continue
        got = _segment_polygon_first_intersection(event_pt, projected_end, pts)
        if got is None and event_inside:
            got = (event_pt, 0.0)
        if got is None:
            continue
        impact_xy, t = got
        projection_len = float(t) * float(extend_px)
        detail = {
            "method": "pending_projected_impact",
            "space_score": 0.53,
            "evidence_level": "pending",
            "event_inside_mask": bool(event_inside),
            "approach_inside_mask": bool(approach_inside),
            "last_point_to_mask_dist_px": round(float(last_dist), 3),
            "impact_approach_to_mask_dist_px": round(float(approach_dist), 3),
            "projected_hit_xy": [round(float(impact_xy[0]), 3), round(float(impact_xy[1]), 3)],
            "projected_end_xy": [round(float(projected_end[0]), 3), round(float(projected_end[1]), 3)],
            "projection_len_px": round(float(projection_len), 3),
            "incoming_dir_xy": [round(float(ux), 6), round(float(uy), 6)],
            "recent_approach_len_px": round(float(recent_len), 3),
        }
        if best is None or float(detail["last_point_to_mask_dist_px"]) < float(best["last_point_to_mask_dist_px"]):
            best = detail
    return best

def _person_mask_hit(person: Dict[str, Any], event_pt: Tuple[float, float], approach_pt: Tuple[float, float], approach_valid: bool) -> Tuple[Optional[str], float, str]:
    """Try precise segmentation-mask matching before bbox fallback.

    Evidence is intentionally layered:
      strong        : impact point inside mask
      medium_strong : approach segment crosses mask

    Point-in-mask is checked before segment intersection because it is the
    strongest evidence that the impact endpoint itself lies on the body.
    """
    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    if isinstance(polys, dict):
        polys = list(polys.values())
    if polys and isinstance(polys, list):
        # Human results usually store a list of polygons: [[[x,y],...], ...]
        for poly in polys:
            if not isinstance(poly, list):
                continue
            if _point_in_polygon(event_pt, poly):
                return "point_in_mask", 0.72, "strong"
            if approach_valid and _segment_intersects_polygon(approach_pt, event_pt, poly):
                return "segment_intersect_mask", 0.62, "medium_strong"
    return None, 0.0, "none"


def _person_mask_contact_pending_detail(
    person: Dict[str, Any],
    event_pt: Tuple[float, float],
    approach_pt: Tuple[float, float],
    approach_valid: bool,
    mask_method: Optional[str],
    mask_space: float,
    mask_evidence: str,
    min_recent_len_px: float,
) -> Optional[Dict[str, Any]]:
    """Convert a precise mask contact into a behaviour pending impact.

    This does NOT emit a final hit.  It only creates a pending impact when the
    event track touches the human mask.  Final confirmation still depends on
    later behaviour: continued straight movement => near miss; disappearance /
    slowdown / bounce after contact => hit.

    v13: be deliberately more permissive for strong mask contact.  A real hit
    can lose events immediately after entering the body, so the available recent
    segment may be short or the previous point may already be inside the mask.
    In those cases, point_in_mask / segment_intersect_mask should still create
    a pending impact instead of falling through to no_geometry_match.  The
    behaviour state machine remains responsible for rejecting clear pass-through
    tracks.
    """
    if not mask_method:
        return None

    mask_method_s = str(mask_method)
    strong_method = mask_method_s in {"point_in_mask", "segment_intersect_mask"}
    if not strong_method:
        return None

    vx = float(event_pt[0]) - float(approach_pt[0]) if approach_valid else 0.0
    vy = float(event_pt[1]) - float(approach_pt[1]) if approach_valid else 0.0
    recent_len = math.hypot(vx, vy)
    if approach_valid and recent_len > 1e-9:
        ux, uy = vx / recent_len, vy / recent_len
    else:
        ux, uy = 0.0, 0.0

    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    if isinstance(polys, dict):
        polys = list(polys.values())
    if not isinstance(polys, list):
        polys = []

    # Segment-contact evidence still needs a usable segment.  Point-in-mask is
    # allowed even with a short/absent recent segment because the hit may have
    # already been absorbed at the body boundary.
    if mask_method_s == "segment_intersect_mask" and (not approach_valid or recent_len < float(min_recent_len_px)):
        return None

    best: Optional[Dict[str, Any]] = None
    for poly in polys:
        if not isinstance(poly, list):
            continue
        pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
        if len(pts) < 3:
            continue

        approach_inside = bool(approach_valid and _point_in_polygon(approach_pt, pts))
        event_inside = _point_in_polygon(event_pt, pts)
        intersects = bool(approach_valid and _segment_intersects_polygon(approach_pt, event_pt, pts))

        if mask_method_s == "point_in_mask" and not event_inside:
            continue
        if mask_method_s == "segment_intersect_mask" and not intersects:
            continue

        approach_dist = _dist_to_polygon(approach_pt, pts) if approach_valid else -1.0
        last_dist = _dist_to_polygon(event_pt, pts)
        # Use event point as impact point for direct contact.  This is a pending
        # contact only; it does not by itself emit a final hit.
        detail = {
            "method": "pending_projected_impact",
            "space_score": max(0.60 if mask_method_s == "point_in_mask" else 0.56, float(mask_space or 0.0)),
            "evidence_level": "pending_mask_contact",
            "pending_contact_source_method": mask_method_s,
            "pending_contact_source_score": round(float(mask_space or 0.0), 3),
            "pending_contact_source_evidence": str(mask_evidence or ""),
            "event_inside_mask": bool(event_inside),
            "approach_inside_mask": bool(approach_inside),
            "segment_intersects_mask": bool(intersects),
            "last_point_to_mask_dist_px": round(float(last_dist), 3),
            "impact_approach_to_mask_dist_px": round(float(approach_dist), 3),
            "projected_hit_xy": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "projected_end_xy": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "projection_len_px": 0.0,
            "incoming_dir_xy": [round(float(ux), 6), round(float(uy), 6)],
            "recent_approach_len_px": round(float(recent_len), 3),
            "pending_created_from_direct_mask_contact": True,
            "pending_direct_mask_contact_relaxed": bool(recent_len < float(min_recent_len_px) or approach_inside or not approach_valid),
        }
        if best is None or float(detail["last_point_to_mask_dist_px"]) < float(best["last_point_to_mask_dist_px"]):
            best = detail

    # If masks were unavailable but _person_mask_hit already established a
    # strong point-in-mask contact, still create a conservative pending contact.
    if best is None and mask_method_s == "point_in_mask":
        best = {
            "method": "pending_projected_impact",
            "space_score": max(0.60, float(mask_space or 0.0)),
            "evidence_level": "pending_mask_contact",
            "pending_contact_source_method": mask_method_s,
            "pending_contact_source_score": round(float(mask_space or 0.0), 3),
            "pending_contact_source_evidence": str(mask_evidence or ""),
            "event_inside_mask": True,
            "approach_inside_mask": bool(False),
            "segment_intersects_mask": bool(False),
            "last_point_to_mask_dist_px": 0.0,
            "impact_approach_to_mask_dist_px": -1.0,
            "projected_hit_xy": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "projected_end_xy": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "projection_len_px": 0.0,
            "incoming_dir_xy": [round(float(ux), 6), round(float(uy), 6)],
            "recent_approach_len_px": round(float(recent_len), 3),
            "pending_created_from_direct_mask_contact": True,
            "pending_direct_mask_contact_relaxed": True,
        }
    return best

class JsonlTailer:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.fp = None
        self.offset = 0
        self.last_try = 0.0
        self.inode = None
        self.last_error = "not opened"

    def poll(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self.fp is None:
            now = time.time()
            if now - self.last_try < 0.5:
                return out
            self.last_try = now
            try:
                st = os.stat(self.path)
                self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
                self.inode = (st.st_dev, st.st_ino)
                self.offset = 0
            except Exception as exc:
                self.last_error = f"open failed: {exc}"
                return out
        try:
            st = os.stat(self.path)
            key = (st.st_dev, st.st_ino)
            if key != self.inode or st.st_size < self.offset:
                try: self.fp.close()
                except Exception: pass
                self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
                self.inode = key
                self.offset = 0
            self.fp.seek(self.offset)
            while True:
                line = self.fp.readline()
                if not line:
                    break
                self.offset = self.fp.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"poll failed: {exc}"
            try:
                if self.fp is not None:
                    self.fp.close()
            except Exception:
                pass
            self.fp = None
        return out


class TriggerCsvTailer:
    """Incrementally read event_trigger_{camera}.csv.

    It keeps both sync_index -> event_ts_us and a timestamp-sorted view so the
    hit judge can answer the important question:

        given a bullet impact timestamp, which 40 ms MVS trigger period did it
        happen in?

    This avoids using UDP receive time or YOLO write time for hit matching.
    """
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.fp = None
        self.offset = 0
        self.last_try = 0.0
        self.header: Optional[List[str]] = None
        self.map: Dict[int, int] = {}
        self._ts_items: List[Tuple[int, int]] = []  # (event_ts_us, sync_index)
        self._dirty = True
        self.inode = None
        self.last_error = "not opened"

    def poll(self) -> None:
        if self.fp is None:
            now = time.time()
            if now - self.last_try < 0.5:
                return
            self.last_try = now
            try:
                st = os.stat(self.path)
                self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
                self.inode = (st.st_dev, st.st_ino)
                self.offset = 0
                self.header = None
                self.map.clear()
                self._dirty = True
            except Exception as exc:
                self.last_error = f"open failed: {exc}"
                return
        try:
            st = os.stat(self.path)
            key = (st.st_dev, st.st_ino)
            if key != self.inode or st.st_size < self.offset:
                try:
                    if self.fp is not None:
                        self.fp.close()
                except Exception:
                    pass
                self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
                self.inode = key
                self.offset = 0
                self.header = None
                self.map.clear()
                self._dirty = True
            self.fp.seek(self.offset)
            while True:
                line = self.fp.readline()
                if not line:
                    break
                self.offset = self.fp.tell()
                line = line.strip()
                if not line:
                    continue
                parts = next(csv.reader([line]))
                if self.header is None:
                    self.header = parts
                    continue
                d = {self.header[i]: parts[i] for i in range(min(len(self.header), len(parts)))}
                try:
                    si = int(float(d.get("sync_index", "")))
                    ts = int(float(d.get("event_ts_us", "")))
                    # Some trigger CSVs contain helper rows with sync_index=-1.
                    # They are useful for diagnostics but not for MVS-frame anchoring.
                    if si >= 0 and ts > 0:
                        old = self.map.get(si)
                        self.map[si] = ts
                        if old != ts:
                            self._dirty = True
                except Exception:
                    continue
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"poll failed: {exc}"
            try:
                if self.fp is not None: self.fp.close()
            except Exception:
                pass
            self.fp = None

    def _ensure_sorted(self) -> None:
        if not self._dirty:
            return
        self._ts_items = sorted((int(ts), int(si)) for si, ts in self.map.items() if int(si) >= 0 and int(ts) > 0)
        self._dirty = False

    def find_period(self, event_ts_us: int, expected_period_us: int = 40000) -> Optional[Dict[str, Any]]:
        """Return the MVS trigger period containing event_ts_us.

        Returns None if the timestamp is earlier than the first known MVS trigger.
        If the next trigger is not in the CSV yet, it estimates the next boundary
        from expected_period_us.  The pending-event queue will still wait for the
        needed human_result frame before judging.
        """
        self._ensure_sorted()
        if not self._ts_items:
            return None
        ts_list = [x[0] for x in self._ts_items]
        i = bisect.bisect_right(ts_list, int(event_ts_us)) - 1
        if i < 0:
            return None
        anchor_ts, sync_index = self._ts_items[i]
        if i + 1 < len(self._ts_items):
            next_ts, next_sync = self._ts_items[i + 1]
            estimated_next = False
        else:
            next_ts = int(anchor_ts) + int(expected_period_us)
            next_sync = int(sync_index) + 1
            estimated_next = True
        period_us = max(1, int(next_ts) - int(anchor_ts))
        offset_us = int(event_ts_us) - int(anchor_ts)
        return {
            "sync_index": int(sync_index),
            "anchor_ts_us": int(anchor_ts),
            "next_sync_index": int(next_sync),
            "next_ts_us": int(next_ts),
            "period_us": int(period_us),
            "offset_us": int(offset_us),
            "offset_ms": float(offset_us) / 1000.0,
            "estimated_next": bool(estimated_next),
        }


@dataclass
class PersonEntry:
    camera: str
    camera_sn: str
    event_serial: str
    sync_index: int
    event_ts_us: Optional[int]
    frame_w: int
    frame_h: int
    persons: List[Dict[str, Any]]
    raw: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = field(default_factory=time.time)



@dataclass
class PendingBulletEvent:
    ev: BulletEvent
    cam: str
    first_wall: float = field(default_factory=time.time)
    tries: int = 0


@dataclass
class PendingImpact:
    cam: str
    bullet_id: int
    stable_id: int
    base_ev: BulletEvent
    base_candidate: Dict[str, Any]
    period: Optional[Dict[str, Any]]
    selection: Dict[str, Any]
    geometry: Dict[str, Any]
    dir_in: Tuple[float, float]
    impact_xy: Tuple[float, float]
    last_xy: Tuple[float, float]
    created_wall: float = field(default_factory=time.time)
    last_update_wall: float = field(default_factory=time.time)
    event_ts_us: int = 0
    last_event_ts_us: int = 0
    max_forward_progress_px: float = 0.0
    max_move_from_pending_px: float = 0.0
    seen_followup_events: int = 0
    absorb_evidence: str = ""
    emitted: bool = False


@dataclass
class RecentTracklet:
    """A sparse, judged bullet packet kept for front/rear occlusion reasoning.

    Human-mask hard filtering deliberately removes events inside the body region.
    A true pass-through may therefore appear as two short tracklets: one before
    the mask and one after the mask, sometimes with different bullet ids.  This
    cache lets hit_judge conservatively stitch those sparse tracklets without
    re-enabling body-region bullet detection.
    """
    cam: str
    bullet_id: int
    seq_id: int
    event_ts_us: int
    event_type: str
    terminate_reason: str
    event_pt: Tuple[float, float]
    approach_pt: Tuple[float, float]
    dir_xy: Tuple[float, float]
    speed_px_per_ms: float
    track_point_count: int
    track_path_len_px: float
    approach_len_px: float
    track_straightness: float
    track_line_error_px: float
    weak: bool = False
    weak_reason: str = ""


class HitJudgeServer:
    def __init__(self, cfg: Dict[str, Any], cfg_path: str, args: argparse.Namespace):
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.args = args
        self.sync_cfg = cfg.get("sync") if isinstance(cfg.get("sync"), dict) else {}
        self.hit_cfg = cfg.get("hit_judge") if isinstance(cfg.get("hit_judge"), dict) else {}
        self.routes = _routes_from_config(cfg)
        self.cam_ids = [str(r.get("id", "") or "").strip() for r in self.routes if str(r.get("id", "") or "").strip()]
        wanted = {x.strip() for x in str(args.cams or "").split(",") if x.strip()}
        if wanted:
            self.cam_ids = [c for c in self.cam_ids if c in wanted]
        self.serial_to_cam: Dict[str, str] = {}
        self.cam_to_serial: Dict[str, str] = {}
        for c in self.cam_ids:
            es = _event_serial_for_camera(self.sync_cfg, c)
            if es:
                self.serial_to_cam[es] = c
                self.cam_to_serial[c] = es
        self.H: Dict[str, np.ndarray] = {c: _homography_for_camera(cfg, c) for c in self.cam_ids}
        self.human_tailers: Dict[str, JsonlTailer] = {c: JsonlTailer(_sync_path(self.sync_cfg, "human_jsonl_template", c, "{root}/human_result_{camera}.jsonl")) for c in self.cam_ids}
        self.trigger_tailers: Dict[str, TriggerCsvTailer] = {c: TriggerCsvTailer(_sync_path(self.sync_cfg, "event_trigger_csv_template", c, "{root}/event_trigger_{camera}.csv")) for c in self.cam_ids}
        self.history: Dict[str, Deque[PersonEntry]] = {c: deque(maxlen=int(self.hit_cfg.get("person_history_frames", 128) or 128)) for c in self.cam_ids}
        self.pending_without_ts: Dict[str, Deque[PersonEntry]] = {c: deque(maxlen=256) for c in self.cam_ids}
        root = str(self.sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.hit_template = str(self.hit_cfg.get("hit_jsonl_template", "{root}/hit_candidate_{camera}.jsonl") or "{root}/hit_candidate_{camera}.jsonl")
        self.all_path = os.path.abspath(str(self.hit_cfg.get("hit_all_jsonl", "{root}/hit_candidate_all.jsonl") or "{root}/hit_candidate_all.jsonl").format(root=self.root))
        self.out_files: Dict[str, Any] = {}
        for c in self.cam_ids:
            p = self.hit_template.format(root=self.root, camera=c, cam=c, id=c)
            if not os.path.isabs(p): p = os.path.abspath(p)
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            self.out_files[c] = open(p, "a", encoding="utf-8", buffering=1)
        self.all_fp = open(self.all_path, "a", encoding="utf-8", buffering=1)

        # Per-impact diagnostic logs.  These are written for every impact-like
        # turn/terminate event, including rejected and timeout cases, so a run can
        # be audited statistically instead of only looking at visible candidates.
        self.write_debug_jsonl = _as_bool(self.hit_cfg.get("write_debug_jsonl", True), True)
        self.debug_template = str(self.hit_cfg.get("hit_debug_jsonl_template", "{root}/hit_judge_debug_{camera}.jsonl") or "{root}/hit_judge_debug_{camera}.jsonl")
        self.debug_all_path = os.path.abspath(str(self.hit_cfg.get("hit_debug_all_jsonl", "{root}/hit_judge_debug_all.jsonl") or "{root}/hit_judge_debug_all.jsonl").format(root=self.root))
        self.debug_files: Dict[str, Any] = {}
        if self.write_debug_jsonl:
            for c in self.cam_ids:
                dp = self.debug_template.format(root=self.root, camera=c, cam=c, id=c)
                if not os.path.isabs(dp): dp = os.path.abspath(dp)
                Path(dp).parent.mkdir(parents=True, exist_ok=True)
                self.debug_files[c] = open(dp, "a", encoding="utf-8", buffering=1)
            Path(self.debug_all_path).parent.mkdir(parents=True, exist_ok=True)
            self.debug_all_fp = open(self.debug_all_path, "a", encoding="utf-8", buffering=1)
        else:
            self.debug_all_fp = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((args.bind_ip, int(args.port)))
        self.sock.setblocking(False)
        self.seen_keys: Deque[Tuple[str, int, int]] = deque(maxlen=4096)
        self.seen_set = set()
        self.pending_events: Deque[PendingBulletEvent] = deque(maxlen=int(self.hit_cfg.get("pending_event_queue", 2048) or 2048))
        self.last_print = time.time()
        self.recv_count = 0
        self.cand_count = 0
        self.no_human_count = 0
        self.no_cam_count = 0
        self.nonimpact_count = 0
        self.wait_count = 0
        self.drop_count = 0
        self.max_dt_ms = float(self.hit_cfg.get("max_dt_ms", args.max_dt_ms))
        self.bbox_expand_px = float(self.hit_cfg.get("bbox_expand_px", args.bbox_expand_px))
        self.near_px = float(self.hit_cfg.get("near_bbox_px", args.near_bbox_px))
        self.min_score = float(self.hit_cfg.get("min_score_to_log", args.min_score_to_log))

        # Impact-timestamp-driven matching.  This is the more precise path:
        # bullet turn/terminate timestamp -> event_trigger period -> MVS human frame.
        self.impact_sync_enable = _as_bool(self.hit_cfg.get("impact_sync_enable", True), True)
        self.impact_event_types = set(self.hit_cfg.get("impact_event_types", [EVENT_TURN, EVENT_TERMINATE]) or [EVENT_TURN, EVENT_TERMINATE])
        self.impact_terminate_reasons = set(self.hit_cfg.get("impact_terminate_reasons", []) or [])
        self.impact_min_turn_angle_deg = float(self.hit_cfg.get("impact_min_turn_angle_deg", 0.0) or 0.0)
        self.impact_person_strategy = str(self.hit_cfg.get("impact_person_strategy", "nearest") or "nearest").lower()
        self.impact_pending_timeout_ms = float(self.hit_cfg.get("impact_pending_timeout_ms", 1500.0) or 1500.0)
        self.impact_max_offset_ms = float(self.hit_cfg.get("impact_max_offset_ms", 45.0) or 45.0)
        self.impact_use_mask = _as_bool(self.hit_cfg.get("impact_use_mask", True), True)
        sync_period_ms = float(self.sync_cfg.get("mvs_soft_trigger_period_ms", 40.0) or 40.0)
        self.expected_period_us = int(float(self.hit_cfg.get("impact_anchor_period_ms", sync_period_ms) or sync_period_ms) * 1000.0)
        self.impact_midpoint_ms = float(self.hit_cfg.get("impact_midpoint_ms", self.expected_period_us / 2000.0) or (self.expected_period_us / 2000.0))

        # Robustness controls for real-world sparse YOLO results and imperfect logs.
        self.short_grace_ms = float(self.hit_cfg.get("short_grace_ms", 120.0) or 120.0)
        self.human_search_radius_sync = int(self.hit_cfg.get("human_search_radius_sync", 2) or 2)
        self.trigger_period_min_ms = float(self.hit_cfg.get("trigger_period_min_ms", 25.0) or 25.0)
        self.trigger_period_max_ms = float(self.hit_cfg.get("trigger_period_max_ms", 60.0) or 60.0)
        self.emit_candidate_on_bad_trigger_period = _as_bool(self.hit_cfg.get("emit_candidate_on_bad_trigger_period", False), False)
        self.trajectory_segment_enable = _as_bool(self.hit_cfg.get("trajectory_segment_enable", True), True)
        self.trajectory_segment_lookback_ms = float(self.hit_cfg.get("trajectory_segment_lookback_ms", 12.0) or 12.0)

        # 连续直线轨迹质量门控：过滤人身上冒出来的伪 bullet
        self.track_quality_enable = _as_bool(self.hit_cfg.get("track_quality_enable", True), True)
        self.min_track_point_count = int(self.hit_cfg.get("min_track_point_count", 5) or 5)
        self.min_track_path_len_px = float(self.hit_cfg.get("min_track_path_len_px", 80.0) or 80.0)
        self.min_track_duration_ms = float(self.hit_cfg.get("min_track_duration_ms", 15.0) or 15.0)
        self.min_track_straightness = float(self.hit_cfg.get("min_track_straightness", 0.85) or 0.85)
        self.max_track_line_error_px = float(self.hit_cfg.get("max_track_line_error_px", 12.0) or 12.0)
        # 兼容两套配置命名：旧补丁里有时叫 min_track_speed_px_per_ms / min_approach_len_px，
        # 当前主配置里叫 min_hit_avg_speed_px_per_ms / min_hit_approach_len_px。
        self.min_hit_avg_speed_px_per_ms = float(
            self.hit_cfg.get(
                "min_hit_avg_speed_px_per_ms",
                self.hit_cfg.get("min_track_speed_px_per_ms", self.hit_cfg.get("min_track_speed", 0.6)),
            ) or 0.6
        )
        self.min_hit_approach_len_px = float(
            self.hit_cfg.get(
                "min_hit_approach_len_px",
                self.hit_cfg.get("min_approach_len_px", 45.0),
            ) or 45.0
        )
        self.reject_terminate_reasons_for_hit = set(self.hit_cfg.get("reject_terminate_reasons_for_hit", ["static", "probation", "probation_offset"]) or ["static", "probation", "probation_offset"])

        # 最终 HIT 只认人体 mask，bbox 只作为 debug 参考
        self.final_hit_mask_only = _as_bool(self.hit_cfg.get("final_hit_mask_only", True), True)
        self.allow_bbox_final_hit = _as_bool(self.hit_cfg.get("allow_bbox_final_hit", False), False)
        self.require_trajectory_for_hit = _as_bool(self.hit_cfg.get("require_trajectory_for_hit", True), True)
        self.min_hit_trajectory_len_px = float(self.hit_cfg.get("min_hit_trajectory_len_px", 35.0) or 35.0)
        self.fallback_penalty_base = float(self.hit_cfg.get("human_fallback_penalty", 0.12) or 0.12)
        self.fallback_penalty_per_sync = float(self.hit_cfg.get("human_fallback_penalty_per_sync", 0.05) or 0.05)
        self.estimated_next_trigger_penalty = float(self.hit_cfg.get("estimated_next_trigger_penalty", 0.03) or 0.03)

        # Final hit candidates are intentionally stricter than debug matching.
        # Nearby/fallback human frames are useful for diagnosis, but allowing them
        # to emit final candidates can turn pre-shot event-camera noise on the body
        # into false hits.  Default to exact person sync for candidates.
        self.allow_human_fallback_for_hit = _as_bool(self.hit_cfg.get("allow_human_fallback_for_hit", False), False)
        self.max_person_frame_delta_sync_for_hit = int(self.hit_cfg.get("max_person_frame_delta_sync_for_hit", 0) or 0)

        # v6: impact-behaviour state machine for the upstream human digital mask.
        # A projected ray touching the current person mask is only a pending impact.
        # Final candidates are emitted only if the track disappears near the body
        # boundary (hit_absorb) or turns/bounces near that boundary (hit_bounce).
        self.enable_impact_behavior_hit = _as_bool(self.hit_cfg.get("enable_impact_behavior_hit", True), True)
        self.impact_behavior_pending_window_ms = float(self.hit_cfg.get("impact_behavior_pending_window_ms", 70.0) or 70.0)
        self.impact_boundary_dist_px = float(self.hit_cfg.get("impact_boundary_dist_px", 60.0) or 60.0)
        self.impact_project_extend_px = float(self.hit_cfg.get("impact_project_extend_px", 140.0) or 140.0)
        self.impact_min_recent_len_px = float(self.hit_cfg.get("impact_min_recent_len_px", 25.0) or 25.0)
        self.impact_bounce_min_angle_deg = float(self.hit_cfg.get("impact_bounce_min_angle_deg", 40.0) or 40.0)
        self.impact_bounce_max_boundary_dist_px = float(self.hit_cfg.get("impact_bounce_max_boundary_dist_px", 85.0) or 85.0)
        self.impact_near_miss_continue_len_px = float(self.hit_cfg.get("impact_near_miss_continue_len_px", 90.0) or 90.0)
        self.impact_near_miss_max_angle_deg = float(self.hit_cfg.get("impact_near_miss_max_angle_deg", 22.0) or 22.0)
        # v7: projected/mask contact is never a final hit by itself.  It only
        # creates a pending impact.  A final candidate is emitted later only if
        # the track disappears near the body boundary after the observation
        # window, or if it bounces/turns near that boundary.
        self.impact_behavior_direct_mask_final_hit = _as_bool(self.hit_cfg.get("impact_behavior_direct_mask_final_hit", False), False)
        self.impact_absorb_min_pending_age_ms = float(self.hit_cfg.get("impact_absorb_min_pending_age_ms", self.impact_behavior_pending_window_ms) or self.impact_behavior_pending_window_ms)
        self.impact_absorb_require_no_followup = _as_bool(self.hit_cfg.get("impact_absorb_require_no_followup", True), True)
        self.impact_absorb_max_followup_progress_px = float(self.hit_cfg.get("impact_absorb_max_followup_progress_px", 35.0) or 35.0)
        self.impact_update_pending_before_human = _as_bool(self.hit_cfg.get("impact_update_pending_before_human", True), True)
        # P19 C++ shot_bounce_link is context-only by default: it may update/confirm
        # an existing pending impact, but it cannot create a new final HIT by itself.
        self.shot_bounce_link_context_only = _as_bool(self.hit_cfg.get("shot_bounce_link_context_only", True), True)
        # v10: direct mask hits stay strict, but the behaviour state machine may
        # use a nearby held human frame when YOLO did not write the exact sync.
        # This is only used to create pending_impact / hit_absorb / hit_bounce;
        # point_in_mask and bbox methods are still not allowed to emit final hits.
        self.impact_behavior_allow_human_fallback = _as_bool(self.hit_cfg.get("impact_behavior_allow_human_fallback", True), True)
        self.impact_behavior_max_person_frame_delta_sync = int(
            self.hit_cfg.get("impact_behavior_max_person_frame_delta_sync", self.human_search_radius_sync) or self.human_search_radius_sync
        )
        self.impact_behavior_min_pending_score = float(self.hit_cfg.get("impact_behavior_min_pending_score", 0.45) or 0.45)
        # v13: direct mask contact is strong geometry.  It should create pending
        # even when the recent segment is short, because real hits may disappear
        # immediately after touching the body.  It still does not directly emit
        # a final hit; behaviour must confirm absorb/bounce or reject clear pass-through.
        self.impact_behavior_mask_contact_min_pending_score = float(self.hit_cfg.get("impact_behavior_mask_contact_min_pending_score", 0.28) or 0.28)
        self.impact_near_miss_min_followup_events_for_strong = int(self.hit_cfg.get("impact_near_miss_min_followup_events_for_strong", 3) or 3)
        self.impact_near_miss_min_move_px_for_strong = float(
            self.hit_cfg.get(
                "impact_near_miss_min_move_px_for_strong",
                max(float(self.impact_near_miss_continue_len_px) * 1.25, float(self.impact_near_miss_continue_len_px) + 25.0),
            )
            or max(float(self.impact_near_miss_continue_len_px) * 1.25, float(self.impact_near_miss_continue_len_px) + 25.0)
        )
        # v14: strong mask contact should only become near_miss when pass-through
        # evidence is stronger than the normal body-proximity continuation gate.
        # Real hits often leave a few residual events after impact; those should
        # not be treated as straight pass-through unless the track continues far,
        # stable, and for multiple follow-up events.
        self.impact_near_miss_continue_len_px_for_strong = float(
            self.hit_cfg.get(
                "impact_near_miss_continue_len_px_for_strong",
                max(float(self.impact_near_miss_continue_len_px) * 1.35, float(self.impact_near_miss_continue_len_px) + 35.0),
            )
            or max(float(self.impact_near_miss_continue_len_px) * 1.35, float(self.impact_near_miss_continue_len_px) + 35.0)
        )
        self.impact_near_miss_max_angle_deg_for_strong = float(
            self.hit_cfg.get("impact_near_miss_max_angle_deg_for_strong", min(float(self.impact_near_miss_max_angle_deg), 14.0))
            or min(float(self.impact_near_miss_max_angle_deg), 14.0)
        )
        # v12: strong mask contact is not a final hit by itself, but it must be
        # allowed to create pending_impact.  Behaviour then decides: continuing
        # straight past the projected body boundary => near_miss; disappearing,
        # slowing, or bouncing after mask contact => hit.
        self.impact_behavior_mask_contact_pending = _as_bool(self.hit_cfg.get("impact_behavior_mask_contact_pending", True), True)
        self.impact_mask_contact_pending_min_recent_len_px = float(self.hit_cfg.get("impact_mask_contact_pending_min_recent_len_px", 6.0) or 6.0)
        self.impact_absorb_relax_external_gate_on_strong_geometry = _as_bool(
            self.hit_cfg.get("impact_absorb_relax_external_gate_on_strong_geometry", False), False
        )
        self.impact_absorb_strong_geometry_methods = set(
            self.hit_cfg.get(
                "impact_absorb_strong_geometry_methods",
                ["point_in_mask", "segment_intersect_mask", "pending_mask_contact"],
            )
            or ["point_in_mask", "segment_intersect_mask", "pending_mask_contact"]
        )
        # Projection-only evidence means the current bullet ray would hit the body
        # if extended.  It is useful for creating pending_impact, but should not be
        # treated as direct body contact by default; otherwise a nearby obstacle
        # bounce/turn can be mistaken for a human hit.
        self.impact_projected_impact_is_strong_geometry = _as_bool(
            self.hit_cfg.get("impact_projected_impact_is_strong_geometry", False), False
        )
        if not bool(self.impact_projected_impact_is_strong_geometry):
            self.impact_absorb_strong_geometry_methods.discard("pending_projected_impact")
        self.impact_direct_body_contact_max_gap_px = float(
            self.hit_cfg.get("impact_direct_body_contact_max_gap_px", 18.0) or 18.0
        )
        self.impact_allow_projected_only_bounce = _as_bool(
            self.hit_cfg.get("impact_allow_projected_only_bounce", False), False
        )
        self.impact_allow_projected_only_absorb = _as_bool(
            self.hit_cfg.get("impact_allow_projected_only_absorb", False), False
        )
        # v11: final hit_absorb must look like an external incoming bullet, not
        # a short body-edge motion fragment.  These gates are checked only when
        # converting a pending impact into hit_absorb.  They do not affect
        # hit_bounce, and they do not re-enable direct point/segment mask hits.
        self.impact_absorb_min_track_path_len_px = float(self.hit_cfg.get("impact_absorb_min_track_path_len_px", 120.0) or 120.0)
        self.impact_absorb_min_approach_len_px = float(self.hit_cfg.get("impact_absorb_min_approach_len_px", 100.0) or 100.0)
        self.impact_absorb_min_approach_to_mask_dist_px = float(self.hit_cfg.get("impact_absorb_min_approach_to_mask_dist_px", 60.0) or 60.0)
        self.impact_absorb_min_avg_speed_px_per_ms = float(self.hit_cfg.get("impact_absorb_min_avg_speed_px_per_ms", 4.0) or 4.0)

        # v17: human mask is treated as an occlusion/judgement zone.  Events inside
        # the mask may still be hard-filtered upstream to suppress waving/person
        # motion, but final hit/near-miss decisions are based on external tracklets:
        #   front external tracklet -> mask boundary -> rear straight tracklet = near_miss
        #   front external tracklet -> true boundary contact -> no rear tracklet = hit_absorb
        #   large turn outside mask boundary = obstacle_bounce_near_person, never HIT
        self.mask_occlusion_judge_enable = _as_bool(self.hit_cfg.get("mask_occlusion_judge_enable", True), True)
        self.mask_occlusion_rear_search_ms = float(self.hit_cfg.get("mask_occlusion_rear_search_ms", 220.0) or 220.0)
        self.mask_occlusion_rear_min_dt_ms = float(self.hit_cfg.get("mask_occlusion_rear_min_dt_ms", 1.0) or 1.0)
        self.mask_occlusion_rear_min_progress_px = float(self.hit_cfg.get("mask_occlusion_rear_min_progress_px", 55.0) or 55.0)
        self.mask_occlusion_rear_max_lateral_error_px = float(self.hit_cfg.get("mask_occlusion_rear_max_lateral_error_px", 32.0) or 32.0)
        self.mask_occlusion_rear_max_angle_deg = float(self.hit_cfg.get("mask_occlusion_rear_max_angle_deg", 16.0) or 16.0)
        self.mask_occlusion_rear_min_speed_ratio = float(self.hit_cfg.get("mask_occlusion_rear_min_speed_ratio", 0.45) or 0.45)
        self.mask_occlusion_rear_max_speed_ratio = float(self.hit_cfg.get("mask_occlusion_rear_max_speed_ratio", 2.20) or 2.20)
        self.mask_occlusion_rear_min_track_path_len_px = float(self.hit_cfg.get("mask_occlusion_rear_min_track_path_len_px", 45.0) or 45.0)
        self.mask_occlusion_rear_min_approach_len_px = float(self.hit_cfg.get("mask_occlusion_rear_min_approach_len_px", 25.0) or 25.0)
        self.mask_occlusion_allow_cross_bullet_stitch = _as_bool(self.hit_cfg.get("mask_occlusion_allow_cross_bullet_stitch", True), True)
        self.mask_occlusion_bounce_requires_true_mask_contact = _as_bool(self.hit_cfg.get("mask_occlusion_bounce_requires_true_mask_contact", True), True)
        self.mask_occlusion_recent_tracklet_cache_ms = float(self.hit_cfg.get("mask_occlusion_recent_tracklet_cache_ms", 2500.0) or 2500.0)
        # v18: do not lower the normal hit track gate for person-motion safety.
        # Instead, keep a separate "weak rear fragment" cache for tracklets that
        # are too short to create a HIT, but are still useful as pass-through /
        # continuation evidence for an already pending external impact.  These
        # fragments can only convert a pending hit to near_miss; they can never
        # create hit_absorb/hit_bounce by themselves.
        self.mask_occlusion_weak_rear_tracklet_enable = _as_bool(self.hit_cfg.get("mask_occlusion_weak_rear_tracklet_enable", True), True)
        self.mask_occlusion_rear_allow_weak_tracklet = _as_bool(self.hit_cfg.get("mask_occlusion_rear_allow_weak_tracklet", True), True)
        self.mask_occlusion_weak_rear_min_track_point_count = int(self.hit_cfg.get("mask_occlusion_weak_rear_min_track_point_count", 6) or 6)
        self.mask_occlusion_weak_rear_min_track_path_len_px = float(self.hit_cfg.get("mask_occlusion_weak_rear_min_track_path_len_px", 35.0) or 35.0)
        self.mask_occlusion_weak_rear_min_duration_ms = float(self.hit_cfg.get("mask_occlusion_weak_rear_min_duration_ms", 20.0) or 20.0)
        self.mask_occlusion_weak_rear_min_straightness = float(self.hit_cfg.get("mask_occlusion_weak_rear_min_straightness", 0.85) or 0.85)
        self.mask_occlusion_weak_rear_max_line_error_px = float(self.hit_cfg.get("mask_occlusion_weak_rear_max_line_error_px", 12.0) or 12.0)
        self.mask_occlusion_weak_rear_min_speed_px_per_ms = float(self.hit_cfg.get("mask_occlusion_weak_rear_min_speed_px_per_ms", 0.6) or 0.6)
        self.mask_occlusion_weak_rear_min_approach_len_px = float(self.hit_cfg.get("mask_occlusion_weak_rear_min_approach_len_px", 30.0) or 30.0)
        allowed_weak_reasons = self.hit_cfg.get("mask_occlusion_weak_rear_allowed_reject_reasons", [
            "track_path_len_too_short",
            "approach_len_too_short",
            "track_speed_too_low",
        ])
        if isinstance(allowed_weak_reasons, (list, tuple, set)):
            self.mask_occlusion_weak_rear_allowed_reject_reasons = {str(x) for x in allowed_weak_reasons}
        else:
            self.mask_occlusion_weak_rear_allowed_reject_reasons = {
                "track_path_len_too_short",
                "approach_len_too_short",
                "track_speed_too_low",
            }
        self.pending_impacts: Dict[Tuple[str, int, int], PendingImpact] = {}
        self.recent_tracklets: Dict[str, Deque[RecentTracklet]] = defaultdict(lambda: deque(maxlen=4096))
        self.emitted_impact_keys: Deque[Tuple[str, int, int]] = deque(maxlen=4096)
        self.emitted_impact_set = set()
        print(f"[hit_judge] config={cfg_path}", flush=True)
        print(f"[hit_judge] listen udp {args.bind_ip}:{args.port}; cams={self.cam_ids}", flush=True)
        print(f"[hit_judge] outputs: {self.hit_template}; all={self.all_path}", flush=True)

    def close(self) -> None:
        try: self.sock.close()
        except Exception: pass
        for fp in self.out_files.values():
            try: fp.close()
            except Exception: pass
        try: self.all_fp.close()
        except Exception: pass
        for fp in getattr(self, "debug_files", {}).values():
            try: fp.close()
            except Exception: pass
        try:
            if getattr(self, "debug_all_fp", None) is not None:
                self.debug_all_fp.close()
        except Exception: pass

    def poll_inputs(self) -> None:
        for c, t in self.trigger_tailers.items():
            t.poll()
        for c, tailer in self.human_tailers.items():
            for rec in tailer.poll():
                self._add_human_record(c, rec)
        self._resolve_pending_timestamps()

    def _add_human_record(self, cam: str, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict) or not rec.get("valid", True):
            return
        si = rec.get("sync_index_hint", None)
        try:
            sync_index = int(si)
        except Exception:
            meta = rec.get("mvs_meta") if isinstance(rec.get("mvs_meta"), dict) else {}
            try: sync_index = int(meta.get("program_sync_index"))
            except Exception: sync_index = -1
        event_ts = self.trigger_tailers.get(cam).map.get(sync_index) if sync_index >= 0 and cam in self.trigger_tailers else None
        entry = PersonEntry(
            camera=cam,
            camera_sn=str(rec.get("mvs_serial", "") or ""),
            event_serial=str(rec.get("event_serial", "") or self.cam_to_serial.get(cam, "")),
            sync_index=sync_index,
            event_ts_us=event_ts,
            frame_w=int(rec.get("frame_w", 0) or 0),
            frame_h=int(rec.get("frame_h", 0) or 0),
            persons=list(rec.get("persons") or []),
            raw=rec,
        )
        if event_ts is None and sync_index >= 0:
            self.pending_without_ts[cam].append(entry)
        else:
            self.history[cam].append(entry)

    def _resolve_pending_timestamps(self) -> None:
        for cam, q in self.pending_without_ts.items():
            if not q:
                continue
            trig = self.trigger_tailers.get(cam)
            if trig is None:
                continue
            keep: Deque[PersonEntry] = deque(maxlen=q.maxlen or 256)
            while q:
                e = q.popleft()
                ts = trig.map.get(e.sync_index)
                if ts is None:
                    # Keep recent unresolved records only.
                    if time.time() - e.wall_time < 3.0:
                        keep.append(e)
                    continue
                e.event_ts_us = int(ts)
                self.history[cam].append(e)
            self.pending_without_ts[cam] = keep

    def _camera_for_event(self, ev: BulletEvent) -> Optional[str]:
        if ev.camera_alias and ev.camera_alias in self.cam_ids:
            return ev.camera_alias
        if ev.camera_sn in self.serial_to_cam:
            return self.serial_to_cam[ev.camera_sn]
        # Last fallback: allow alias passed in camera_sn.
        if ev.camera_sn in self.cam_ids:
            return ev.camera_sn
        return None

    def _mark_seen(self, key: Tuple[str, int, int]) -> bool:
        if key in self.seen_set:
            return False
        if self.seen_keys.maxlen is not None and len(self.seen_keys) >= self.seen_keys.maxlen:
            try:
                old = self.seen_keys.popleft()
                self.seen_set.discard(old)
            except Exception:
                pass
        self.seen_keys.append(key)
        self.seen_set.add(key)
        return True

    def receive_udp(self) -> None:
        while True:
            r, _, _ = select.select([self.sock], [], [], 0)
            if not r:
                return
            try:
                data, _addr = self.sock.recvfrom(65535)
                ev = BulletEvent.from_json(data)
            except Exception as exc:
                print(f"[hit_judge][WARN] bad udp packet: {exc}", flush=True)
                continue
            key = (str(ev.camera_sn), int(ev.bullet_id), int(ev.seq_id))
            if not self._mark_seen(key):
                continue
            self.recv_count += 1
            cam = self._camera_for_event(ev)
            if cam is None:
                self.no_cam_count += 1
                continue
            if self.impact_sync_enable and not self._is_impact_event(ev):
                self.nonimpact_count += 1
                continue
            self.pending_events.append(PendingBulletEvent(ev=ev, cam=cam))

    def _is_shot_bounce_link_event(self, ev: BulletEvent) -> bool:
        """Return True for C++ P19 physical-shot bounce-link context events.

        These events are useful follow-up context for an already pending body-mask
        impact, but they must not create a new HIT by themselves.  This prevents a
        foreground obstacle bounce near a person from being upgraded to a hit just
        because the link segment crosses/approaches the human mask.
        """
        try:
            link_type = str(getattr(ev, "shot_link_type", "") or "").lower()
            link_reason = str(getattr(ev, "shot_link_reason", "") or "").lower()
        except Exception:
            return False
        return link_type == "bounce" or "shot_bounce_link" in link_reason

    def _is_impact_event(self, ev: BulletEvent) -> bool:
        """Only turn/terminate-like events should trigger hit judgment.

        Normal bullet track points are visual data.  Hit judgment should be driven
        by physically meaningful moments: turn/bounce, disappearance, terminate.
        """
        if str(ev.event_type) not in self.impact_event_types:
            return False
        if ev.event_type == EVENT_TURN and float(ev.turn_angle_deg) < self.impact_min_turn_angle_deg:
            return False
        if ev.event_type == EVENT_TERMINATE and self.impact_terminate_reasons:
            return str(ev.terminate_reason) in self.impact_terminate_reasons
        return True

    def _event_track_metrics(self, ev: BulletEvent) -> Dict[str, float]:
        """Safely extract trajectory-quality fields from a BulletEvent."""
        def _as_float(name: str, default: float = 0.0) -> float:
            try:
                value = getattr(ev, name, default)
                if value is None or value == "":
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        def _as_int(name: str, default: int = 0) -> int:
            try:
                value = getattr(ev, name, default)
                if value is None or value == "":
                    return int(default)
                return int(value)
            except Exception:
                return int(default)

        return {
            "track_point_count": float(_as_int("track_point_count", 0)),
            "track_path_len_px": _as_float("track_path_len_px", 0.0),
            "track_duration_ms": _as_float("track_duration_ms", 0.0),
            "track_displacement_px": _as_float("track_displacement_px", 0.0),
            "track_straightness": _as_float("track_straightness", 0.0),
            "track_line_error_px": _as_float("track_line_error_px", 999.0),
            "approach_len_px": _as_float("approach_len_px", 0.0),
            "avg_speed_px_per_ms": _as_float("avg_speed_px_per_ms", 0.0),
        }

    def _track_quality_ok(self, ev: BulletEvent, metrics: Optional[Dict[str, float]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        连续轨迹质量门控。

        这一步在几何判定前执行：没有足够长、足够直、足够快的真实轨迹，
        就不允许产生 hit_candidate。这样可以过滤人体边缘、衣服、反光等局部噪声。
        """
        m = metrics or self._event_track_metrics(ev)
        detail = {
            "track_quality_enable": bool(self.track_quality_enable),
            "track_point_count": int(m.get("track_point_count", 0) or 0),
            "track_path_len_px": round(float(m.get("track_path_len_px", 0.0) or 0.0), 3),
            "track_duration_ms": round(float(m.get("track_duration_ms", 0.0) or 0.0), 3),
            "track_displacement_px": round(float(m.get("track_displacement_px", 0.0) or 0.0), 3),
            "track_straightness": round(float(m.get("track_straightness", 0.0) or 0.0), 3),
            "track_line_error_px": round(float(m.get("track_line_error_px", 999.0) if m.get("track_line_error_px", 999.0) is not None else 999.0), 3),
            "avg_speed_px_per_ms": round(float(m.get("avg_speed_px_per_ms", 0.0) or 0.0), 3),
            "approach_len_px": round(float(m.get("approach_len_px", 0.0) or 0.0), 3),
            "min_track_point_count": int(self.min_track_point_count),
            "min_track_path_len_px": float(self.min_track_path_len_px),
            "min_track_duration_ms": float(self.min_track_duration_ms),
            "min_track_straightness": float(self.min_track_straightness),
            "max_track_line_error_px": float(self.max_track_line_error_px),
            "min_hit_avg_speed_px_per_ms": float(self.min_hit_avg_speed_px_per_ms),
            "min_hit_approach_len_px": float(self.min_hit_approach_len_px),
        }
        if not self.track_quality_enable:
            return True, "", detail

        if ev.event_type == EVENT_TERMINATE and str(ev.terminate_reason) in self.reject_terminate_reasons_for_hit:
            return False, f"terminate_reason_rejected:{ev.terminate_reason}", detail
        if int(m.get("track_point_count", 0) or 0) < int(self.min_track_point_count):
            return False, "track_point_count_too_small", detail
        if float(m.get("track_path_len_px", 0.0) or 0.0) < float(self.min_track_path_len_px):
            return False, "track_path_len_too_short", detail
        if float(m.get("track_duration_ms", 0.0) or 0.0) < float(self.min_track_duration_ms):
            return False, "track_duration_too_short", detail
        if float(m.get("track_straightness", 0.0) or 0.0) < float(self.min_track_straightness):
            return False, "track_not_straight", detail
        if float(m.get("track_line_error_px", 999.0) if m.get("track_line_error_px", 999.0) is not None else 999.0) > float(self.max_track_line_error_px):
            return False, "track_line_error_too_large", detail
        if float(m.get("avg_speed_px_per_ms", 0.0) or 0.0) < float(self.min_hit_avg_speed_px_per_ms):
            return False, "track_speed_too_low", detail
        if float(m.get("approach_len_px", 0.0) or 0.0) < float(self.min_hit_approach_len_px):
            return False, "approach_len_too_short", detail
        return True, "", detail

    def _person_entries_by_sync(self, cam: str, sync_index: int, require_persons: bool = True) -> List[PersonEntry]:
        out: List[PersonEntry] = []
        for e in reversed(self.history.get(cam, [])):
            if int(e.sync_index) == int(sync_index):
                if (not require_persons) or bool(e.persons):
                    out.append(e)
        return out

    def _person_entry_by_sync(self, cam: str, sync_index: int) -> Optional[PersonEntry]:
        entries = self._person_entries_by_sync(cam, sync_index, require_persons=False)
        return entries[0] if entries else None

    def _nearby_person_entries(self, cam: str, target_sync: int, radius: int) -> Tuple[List[PersonEntry], Optional[int]]:
        """Return entries from the nearest available sync within +/- radius."""
        hist = [e for e in self.history.get(cam, []) if e.persons and int(e.sync_index) >= 0]
        if not hist:
            return [], None
        best_delta: Optional[int] = None
        best_sync: Optional[int] = None
        for e in hist:
            delta = int(e.sync_index) - int(target_sync)
            if abs(delta) > int(radius):
                continue
            if best_delta is None or abs(delta) < abs(best_delta) or (abs(delta) == abs(best_delta) and e.wall_time > 0):
                best_delta = delta
                best_sync = int(e.sync_index)
        if best_sync is None:
            return [], None
        return self._person_entries_by_sync(cam, best_sync, require_persons=True), best_sync

    def _nearest_person_entry(self, cam: str, event_ts: int) -> Optional[PersonEntry]:
        # Kept as a fallback for impact_sync_enable=false.
        hist = [e for e in self.history.get(cam, []) if e.event_ts_us is not None]
        if not hist:
            return None
        best = min(hist, key=lambda e: abs(int(e.event_ts_us or 0) - int(event_ts)))
        if abs(int(best.event_ts_us or 0) - int(event_ts)) / 1000.0 > self.max_dt_ms:
            return None
        return best

    def _impact_period_for_event(self, cam: str, event_ts_us: int) -> Optional[Dict[str, Any]]:
        trig = self.trigger_tailers.get(cam)
        if trig is None:
            return None
        period = trig.find_period(int(event_ts_us), expected_period_us=self.expected_period_us)
        if period is None:
            return None
        period_ms = float(period.get("period_us", 0)) / 1000.0
        period["trigger_period_ms"] = period_ms
        period["trigger_period_ok"] = bool(self.trigger_period_min_ms <= period_ms <= self.trigger_period_max_ms)
        period["trigger_period_min_ms"] = float(self.trigger_period_min_ms)
        period["trigger_period_max_ms"] = float(self.trigger_period_max_ms)
        # Reject clearly out-of-period events.  A small margin protects against
        # minor trigger jitter, not against assigning an event to the wrong MVS frame.
        if float(period.get("offset_ms", 0.0)) < -1.0:
            return None
        if float(period.get("offset_ms", 0.0)) > self.impact_max_offset_ms:
            return None
        return period

    def _entries_for_impact(self, cam: str, period: Dict[str, Any], age_ms: float = 0.0, force_final: bool = False) -> Tuple[List[PersonEntry], str, bool, Dict[str, Any]]:
        """Choose human frame(s) for a bullet impact timestamp with robust fallback.

        The first choice follows impact_person_strategy.  If that frame is not
        available, this function waits only short_grace_ms before using an
        adjacent or nearby human frame.  pending_timeout_ms is reserved for cases
        where no useful human frame exists yet.
        """
        k = int(period["sync_index"])
        k_next = int(period.get("next_sync_index", k + 1))
        offset_ms = float(period.get("offset_ms", 0.0))
        strategy = self.impact_person_strategy
        want_next = offset_ms >= self.impact_midpoint_ms

        if strategy in {"period", "anchor", "current"}:
            primary_syncs = [k]
            secondary_syncs = [k_next]
            target_sync = k
            base_strategy = "period"
        elif strategy in {"union", "both"}:
            primary_syncs = [k, k_next]
            secondary_syncs = []
            target_sync = k_next if want_next else k
            base_strategy = "union"
        else:
            target_sync = k_next if want_next else k
            primary_syncs = [target_sync]
            secondary_syncs = [k if want_next else k_next]
            base_strategy = "nearest_next" if want_next else "nearest_current"

        info: Dict[str, Any] = {
            "impact_sync_index": k,
            "target_person_sync_index": int(target_sync),
            "target_person_sync_indices": [int(x) for x in primary_syncs],
            "human_fallback_used": False,
            "human_fallback_reason": "",
            "person_frame_delta_sync": None,
            "searched_sync_range": [-int(self.human_search_radius_sync), int(self.human_search_radius_sync)],
            "short_grace_ms": float(self.short_grace_ms),
            "age_ms": float(age_ms),
        }

        primary_entries: List[PersonEntry] = []
        for si in primary_syncs:
            primary_entries.extend(self._person_entries_by_sync(cam, int(si), require_persons=True))
        if primary_entries:
            info["used_person_sync_index"] = int(primary_entries[0].sync_index)
            info["person_frame_delta_sync"] = int(primary_entries[0].sync_index) - int(target_sync)
            return primary_entries, base_strategy, False, info

        # Give the desired sync a short grace period so the common case stays strict,
        # but do not wait the full pending timeout before using an already available
        # adjacent human frame.
        if (age_ms < self.short_grace_ms) and (not force_final):
            return [], f"{base_strategy}_short_wait", True, info

        secondary_entries: List[PersonEntry] = []
        for si in secondary_syncs:
            secondary_entries.extend(self._person_entries_by_sync(cam, int(si), require_persons=True))
        if secondary_entries:
            used = int(secondary_entries[0].sync_index)
            info.update({
                "used_person_sync_index": used,
                "person_frame_delta_sync": used - int(target_sync),
                "human_fallback_used": True,
                "human_fallback_reason": "primary_missing_used_adjacent",
            })
            return secondary_entries, f"{base_strategy}_fallback_adjacent", False, info

        nearby_entries, nearby_sync = self._nearby_person_entries(cam, int(target_sync), int(self.human_search_radius_sync))
        if nearby_entries and nearby_sync is not None:
            info.update({
                "used_person_sync_index": int(nearby_sync),
                "person_frame_delta_sync": int(nearby_sync) - int(target_sync),
                "human_fallback_used": True,
                "human_fallback_reason": "primary_missing_used_nearby",
            })
            return nearby_entries, f"{base_strategy}_fallback_nearby", False, info

        if not force_final:
            return [], f"{base_strategy}_wait_human", True, info
        info["human_fallback_reason"] = "no_human_result_near_sync"
        return [], f"{base_strategy}_human_timeout", False, info

    def _debug_base(self, ev: BulletEvent, cam: Optional[str], status: str, reject_reason: Optional[str]) -> Dict[str, Any]:
        return {
            "source": "hit_judge_server",
            "status": str(status),
            "reject_reason": reject_reason,
            "camera": cam or ev.camera_alias or "",
            "camera_sn": str(ev.camera_sn),
            "camera_alias": str(ev.camera_alias),
            "event_type": str(ev.event_type),
            "terminate_reason": str(ev.terminate_reason),
            "bullet_id": int(ev.bullet_id),
            "seq_id": int(ev.seq_id),
            "event_ts_us": int(ev.timestamp_us),
            "event_xy": [float(ev.event_x), float(ev.event_y)],
            "approach_xy": [float(ev.approach_x), float(ev.approach_y)],
            "approach_valid": bool(ev.approach_valid),
            "approach_len_px": float(getattr(ev, "approach_len_px", 0.0) or 0.0),
            "turn_angle_deg": float(ev.turn_angle_deg),
            "avg_speed_px_per_ms": float(ev.avg_speed_px_per_ms),
            "track_point_count": int(getattr(ev, "track_point_count", 0) or 0),
            "track_path_len_px": round(float(getattr(ev, "track_path_len_px", 0.0) or 0.0), 3),
            "track_duration_ms": round(float(getattr(ev, "track_duration_ms", 0.0) or 0.0), 3),
            "track_displacement_px": round(float(getattr(ev, "track_displacement_px", 0.0) or 0.0), 3),
            "track_straightness": round(float(getattr(ev, "track_straightness", 0.0) or 0.0), 3),
            "track_line_error_px": round(float(getattr(ev, "track_line_error_px", 999.0) if getattr(ev, "track_line_error_px", 999.0) is not None else 999.0), 3),
            "shot_link_type": str(getattr(ev, "shot_link_type", "") or ""),
            "shot_link_reason": str(getattr(ev, "shot_link_reason", "") or ""),
            "shot_link_parent_bullet_id": int(getattr(ev, "shot_link_parent_bullet_id", -1) or -1),
            "shot_link_parent_segment_index": int(getattr(ev, "shot_link_parent_segment_index", -1) or -1),
            "shot_link_gap_ms": round(float(getattr(ev, "shot_link_gap_ms", 0.0) or 0.0), 3),
            "shot_link_gap_dist_px": round(float(getattr(ev, "shot_link_gap_dist_px", 0.0) or 0.0), 3),
            "shot_link_angle_deg": round(float(getattr(ev, "shot_link_angle_deg", 0.0) or 0.0), 3),
            "event_human_mask_stats": dict(getattr(ev, "event_human_mask_stats", {}) or {}),
            "judge_wall_time": time.time(),
        }

    def _emit_debug(self, cam: Optional[str], ev: BulletEvent, status: str, reject_reason: Optional[str] = None,
                    period: Optional[Dict[str, Any]] = None, selection: Optional[Dict[str, Any]] = None,
                    geometry: Optional[Dict[str, Any]] = None, candidate: Optional[Dict[str, Any]] = None,
                    age_ms: Optional[float] = None) -> None:
        if not getattr(self, "write_debug_jsonl", False):
            return
        d = self._debug_base(ev, cam, status, reject_reason)
        if age_ms is not None:
            d["pending_age_ms"] = round(float(age_ms), 3)
        if period is not None:
            d.update({
                "impact_sync_index": int(period.get("sync_index", -1)),
                "impact_anchor_ts_us": int(period.get("anchor_ts_us", 0)),
                "impact_next_sync_index": int(period.get("next_sync_index", -1)),
                "impact_next_ts_us": int(period.get("next_ts_us", 0)),
                "impact_offset_ms": round(float(period.get("offset_ms", 0.0)), 3),
                "trigger_period_ms": round(float(period.get("trigger_period_ms", float(period.get("period_us", 0)) / 1000.0)), 3),
                "trigger_period_ok": bool(period.get("trigger_period_ok", False)),
                "trigger_period_min_ms": float(period.get("trigger_period_min_ms", self.trigger_period_min_ms)),
                "trigger_period_max_ms": float(period.get("trigger_period_max_ms", self.trigger_period_max_ms)),
                "impact_next_boundary_estimated": bool(period.get("estimated_next", False)),
            })
        if selection:
            d.update(selection)
        if geometry:
            d.update(geometry)
        if candidate:
            d["candidate"] = candidate
            d["candidate_score"] = candidate.get("total_score")
            d["geometry_mode"] = candidate.get("match_method")
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            if cam and cam in self.debug_files:
                self.debug_files[cam].write(line)
            if getattr(self, "debug_all_fp", None) is not None:
                self.debug_all_fp.write(line)
        except Exception as exc:
            print(f"[hit_judge][WARN] debug write failed: {exc}", flush=True)

    def _mark_emitted_impact_key(self, key: Tuple[str, int, int]) -> bool:
        if key in self.emitted_impact_set:
            return False
        if self.emitted_impact_keys.maxlen is not None and len(self.emitted_impact_keys) >= self.emitted_impact_keys.maxlen:
            try:
                old = self.emitted_impact_keys.popleft()
                self.emitted_impact_set.discard(old)
            except Exception:
                pass
        self.emitted_impact_keys.append(key)
        self.emitted_impact_set.add(key)
        return True

    def _candidate_to_output_dict(
        self,
        cand: HitCandidate,
        entry: PersonEntry,
        ev: BulletEvent,
        cam: str,
        period: Optional[Dict[str, Any]],
        selection_info: Dict[str, Any],
        strategy_used: str,
        meta: Dict[str, Any],
        track_metrics: Dict[str, float],
        use_segment: bool,
        approach_len_px: float,
    ) -> Dict[str, Any]:
        d = cand.to_dict()
        d["source"] = "hit_judge_server"
        d["judge_mode"] = "impact_sync" if self.impact_sync_enable else "nearest_time_fallback"
        d["human_match_strategy"] = strategy_used
        d["human_fallback_used"] = bool(selection_info.get("human_fallback_used", False))
        d["human_fallback_reason"] = str(selection_info.get("human_fallback_reason", ""))
        d["target_person_sync_index"] = selection_info.get("target_person_sync_index")
        d["used_person_sync_index"] = int(entry.sync_index)
        d["person_frame_delta_sync"] = int(meta.get("person_frame_delta_sync", int(entry.sync_index) - int(selection_info.get("target_person_sync_index", entry.sync_index))))
        d["evidence_level"] = meta.get("evidence_level", "")
        d["geometry_mode"] = meta.get("geometry_mode", cand.match_method)
        d["track_quality_enable"] = bool(self.track_quality_enable)
        d["final_hit_mask_only"] = bool(self.final_hit_mask_only)
        d["track_point_count"] = int(track_metrics.get("track_point_count", 0) or 0)
        d["track_path_len_px"] = round(float(track_metrics.get("track_path_len_px", 0.0) or 0.0), 3)
        d["track_duration_ms"] = round(float(track_metrics.get("track_duration_ms", 0.0) or 0.0), 3)
        d["track_displacement_px"] = round(float(track_metrics.get("track_displacement_px", 0.0) or 0.0), 3)
        d["track_straightness"] = round(float(track_metrics.get("track_straightness", 0.0) or 0.0), 3)
        d["track_line_error_px"] = round(float(track_metrics.get("track_line_error_px", 999.0) if track_metrics.get("track_line_error_px", 999.0) is not None else 999.0), 3)
        d["trajectory_required_for_hit"] = bool(self.require_trajectory_for_hit)
        d["trajectory_segment_used"] = bool(use_segment)
        d["approach_len_px"] = round(float(approach_len_px), 3)
        d["min_hit_trajectory_len_px"] = round(float(self.min_hit_trajectory_len_px), 3)
        d["event_human_mask_stats"] = dict(getattr(ev, "event_human_mask_stats", {}) or {})
        if period is not None:
            d["impact_event_ts_us"] = int(ev.timestamp_us)
            d["impact_sync_index"] = int(period["sync_index"])
            d["impact_anchor_ts_us"] = int(period["anchor_ts_us"])
            d["impact_next_sync_index"] = int(period["next_sync_index"])
            d["impact_next_ts_us"] = int(period["next_ts_us"])
            d["impact_offset_ms"] = round(float(period["offset_ms"]), 3)
            d["impact_period_ms"] = round(float(period["period_us"]) / 1000.0, 3)
            d["trigger_period_ms"] = round(float(period.get("trigger_period_ms", float(period["period_us"]) / 1000.0)), 3)
            d["trigger_period_ok"] = bool(period.get("trigger_period_ok", False))
            d["impact_next_boundary_estimated"] = bool(period.get("estimated_next", False))
        d["mvs_sync_index"] = int(entry.sync_index)
        d["person_frame_sync_index"] = int(entry.sync_index)
        d["event_to_mvs_homography"] = "configured"
        if meta:
            for k, v in meta.items():
                if k.startswith("impact_") or k.startswith("projected_") or k in {
                    "last_point_to_mask_dist_px", "incoming_dir_xy", "recent_approach_len_px",
                    "pending_impact", "event_inside_mask", "approach_inside_mask",
                    "impact_approach_to_mask_dist_px",
                }:
                    d[k] = v
        return d

    def _write_candidate_dict(self, cam: str, d: Dict[str, Any]) -> None:
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.out_files[cam].write(line)
        self.all_fp.write(line)
        self.cand_count += 1

    def _register_pending_impact(
        self,
        cam: str,
        ev: BulletEvent,
        cand: HitCandidate,
        entry: PersonEntry,
        meta: Dict[str, Any],
        period: Optional[Dict[str, Any]],
        selection_info: Dict[str, Any],
        geometry_summary: Dict[str, Any],
        output_dict: Dict[str, Any],
        age_ms: float,
    ) -> str:
        key = (str(cam), int(ev.bullet_id), int(cand.stable_id))
        if key in self.emitted_impact_set:
            self._emit_debug(cam, ev, "impact_duplicate_ignored", "impact_already_emitted", period=period, selection=selection_info, geometry=geometry_summary, candidate=output_dict, age_ms=age_ms)
            return "no_candidate"
        dir_xy = meta.get("incoming_dir_xy") or [0.0, 0.0]
        impact_xy = meta.get("projected_hit_xy") or list(cand.event_point)
        pending = PendingImpact(
            cam=str(cam),
            bullet_id=int(ev.bullet_id),
            stable_id=int(cand.stable_id),
            base_ev=ev,
            base_candidate=dict(output_dict),
            period=period,
            selection=dict(selection_info),
            geometry=dict(geometry_summary),
            dir_in=(float(dir_xy[0]), float(dir_xy[1])),
            impact_xy=(float(impact_xy[0]), float(impact_xy[1])),
            last_xy=(float(cand.event_point[0]), float(cand.event_point[1])),
            event_ts_us=int(ev.timestamp_us),
            last_event_ts_us=int(ev.timestamp_us),
        )
        pending_direct_ok, pending_direct_detail = self._pending_direct_body_contact_detail(pending)
        pending.geometry.update(pending_direct_detail)
        pending.base_candidate.update({
            "impact_direct_body_contact": bool(pending_direct_ok),
            "impact_projected_only_guard": bool(not pending_direct_ok),
        })
        self.pending_impacts[key] = pending
        dbg_candidate = dict(output_dict)
        dbg_candidate["impact_behavior_state"] = "pending"
        dbg_candidate.update({
            "impact_direct_body_contact": bool(pending_direct_ok),
            "impact_projected_only_guard": bool(not pending_direct_ok),
        })
        geom_for_debug = dict(geometry_summary)
        geom_for_debug.update(pending_direct_detail)
        self._emit_debug(cam, ev, "pending_impact", None, period=period, selection=selection_info, geometry=geom_for_debug, candidate=dbg_candidate, age_ms=age_ms)
        return "pending_impact"

    def _finalize_pending_impact(
        self,
        key: Tuple[str, int, int],
        method: str,
        reason: str,
        ev: Optional[BulletEvent] = None,
        extra_geometry: Optional[Dict[str, Any]] = None,
    ) -> str:
        pending = self.pending_impacts.pop(key, None)
        if pending is None:
            return "no_candidate"
        if not self._mark_emitted_impact_key(key):
            return "no_candidate"
        now = time.time()
        d = dict(pending.base_candidate)
        d["match_method"] = str(method)
        d["geometry_mode"] = str(method)
        d["evidence_level"] = "impact_behavior"
        d["impact_behavior_result"] = str(method)
        d["impact_behavior_reason"] = str(reason)
        d["impact_behavior_pending_age_ms"] = round((now - pending.created_wall) * 1000.0, 3)
        d["judge_wall_time"] = now
        self._write_candidate_dict(pending.cam, d)
        geom = dict(pending.geometry)
        geom["impact_behavior_result"] = str(method)
        geom["impact_behavior_reason"] = str(reason)
        if extra_geometry:
            geom.update(extra_geometry)
        dbg_ev = ev or pending.base_ev
        self._emit_debug(pending.cam, dbg_ev, "emitted_candidate", None, period=pending.period, selection=pending.selection, geometry=geom, candidate=d, age_ms=(now - pending.created_wall) * 1000.0)
        print(f"[hit_judge][{pending.cam}] behavior candidate bullet={pending.bullet_id} person={pending.stable_id} method={method} reason={reason} score={d.get('total_score','?')}", flush=True)
        return "emitted"

    def _remember_tracklet_event(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        use_segment: bool,
        track_metrics: Dict[str, float],
        weak: bool = False,
        weak_reason: str = "",
    ) -> None:
        """Cache a sparse external bullet packet for mask-occlusion stitching.

        This cache is intentionally filled only after the normal track-quality
        gate passes.  It must not become a new bullet detector for person motion.
        """
        if not bool(getattr(self, "mask_occlusion_judge_enable", True)):
            return
        if not bool(use_segment):
            return
        vx = float(event_pt[0]) - float(approach_pt[0])
        vy = float(event_pt[1]) - float(approach_pt[1])
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            return
        dq = self.recent_tracklets[str(cam)]
        cutoff_us = int(ev.timestamp_us) - int(float(self.mask_occlusion_recent_tracklet_cache_ms) * 1000.0)
        while dq and int(dq[0].event_ts_us) < cutoff_us:
            dq.popleft()
        speed = float(getattr(ev, "avg_speed_px_per_ms", 0.0) or 0.0)
        if speed <= 0.0:
            speed = float(track_metrics.get("avg_speed_px_per_ms", 0.0) or 0.0)
        dq.append(RecentTracklet(
            cam=str(cam),
            bullet_id=int(ev.bullet_id),
            seq_id=int(ev.seq_id),
            event_ts_us=int(ev.timestamp_us),
            event_type=str(ev.event_type),
            terminate_reason=str(ev.terminate_reason),
            event_pt=(float(event_pt[0]), float(event_pt[1])),
            approach_pt=(float(approach_pt[0]), float(approach_pt[1])),
            dir_xy=(float(vx / norm), float(vy / norm)),
            speed_px_per_ms=float(speed),
            track_point_count=int(track_metrics.get("track_point_count", 0) or 0),
            track_path_len_px=float(track_metrics.get("track_path_len_px", 0.0) or 0.0),
            approach_len_px=float(track_metrics.get("approach_len_px", 0.0) or getattr(ev, "approach_len_px", 0.0) or 0.0),
            track_straightness=float(track_metrics.get("track_straightness", 0.0) or 0.0),
            track_line_error_px=float(track_metrics.get("track_line_error_px", 999.0) if track_metrics.get("track_line_error_px", 999.0) is not None else 999.0),
            weak=bool(weak),
            weak_reason=str(weak_reason or ""),
        ))


    def _weak_rear_tracklet_ok(
        self,
        ev: BulletEvent,
        track_metrics: Dict[str, float],
        reject_reason: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Return whether a rejected packet is safe to cache as rear evidence only.

        This deliberately does not relax the normal hit gate.  A weak rear tracklet
        can only help prove pass-through / near-miss for an existing pending impact.
        It never creates a new pending impact or a final HIT.
        """
        reason = str(reject_reason or "")
        detail: Dict[str, Any] = {
            "mask_occlusion_weak_rear_tracklet_enable": bool(getattr(self, "mask_occlusion_weak_rear_tracklet_enable", True)),
            "mask_occlusion_weak_rear_allowed_reject_reasons": sorted(getattr(self, "mask_occlusion_weak_rear_allowed_reject_reasons", set()) or []),
            "mask_occlusion_weak_rear_reject_reason": reason,
            "mask_occlusion_weak_rear_cached": False,
        }
        if not bool(getattr(self, "mask_occlusion_weak_rear_tracklet_enable", True)):
            detail["mask_occlusion_weak_rear_skip_reason"] = "disabled"
            return False, detail
        if reason not in (getattr(self, "mask_occlusion_weak_rear_allowed_reject_reasons", set()) or set()):
            detail["mask_occlusion_weak_rear_skip_reason"] = "reject_reason_not_allowed"
            return False, detail
        try:
            pc = int(track_metrics.get("track_point_count", 0) or 0)
            path = float(track_metrics.get("track_path_len_px", 0.0) or 0.0)
            dur = float(track_metrics.get("track_duration_ms", 0.0) or 0.0)
            straight = float(track_metrics.get("track_straightness", 0.0) or 0.0)
            line_err = float(track_metrics.get("track_line_error_px", 999.0) if track_metrics.get("track_line_error_px", 999.0) is not None else 999.0)
            speed = float(track_metrics.get("avg_speed_px_per_ms", 0.0) or 0.0)
            app_len = float(track_metrics.get("approach_len_px", 0.0) or 0.0)
        except Exception:
            detail["mask_occlusion_weak_rear_skip_reason"] = "bad_metrics"
            return False, detail
        detail.update({
            "mask_occlusion_weak_rear_min_track_point_count": int(getattr(self, "mask_occlusion_weak_rear_min_track_point_count", 6)),
            "mask_occlusion_weak_rear_min_track_path_len_px": round(float(getattr(self, "mask_occlusion_weak_rear_min_track_path_len_px", 35.0)), 3),
            "mask_occlusion_weak_rear_min_duration_ms": round(float(getattr(self, "mask_occlusion_weak_rear_min_duration_ms", 20.0)), 3),
            "mask_occlusion_weak_rear_min_straightness": round(float(getattr(self, "mask_occlusion_weak_rear_min_straightness", 0.85)), 3),
            "mask_occlusion_weak_rear_max_line_error_px": round(float(getattr(self, "mask_occlusion_weak_rear_max_line_error_px", 12.0)), 3),
            "mask_occlusion_weak_rear_min_speed_px_per_ms": round(float(getattr(self, "mask_occlusion_weak_rear_min_speed_px_per_ms", 0.6)), 3),
            "mask_occlusion_weak_rear_min_approach_len_px": round(float(getattr(self, "mask_occlusion_weak_rear_min_approach_len_px", 30.0)), 3),
        })
        checks = [
            (pc >= int(getattr(self, "mask_occlusion_weak_rear_min_track_point_count", 6)), "weak_point_count_too_small"),
            (path >= float(getattr(self, "mask_occlusion_weak_rear_min_track_path_len_px", 35.0)), "weak_path_len_too_short"),
            (dur >= float(getattr(self, "mask_occlusion_weak_rear_min_duration_ms", 20.0)), "weak_duration_too_short"),
            (straight >= float(getattr(self, "mask_occlusion_weak_rear_min_straightness", 0.85)), "weak_track_not_straight"),
            (line_err <= float(getattr(self, "mask_occlusion_weak_rear_max_line_error_px", 12.0)), "weak_line_error_too_large"),
            (speed >= float(getattr(self, "mask_occlusion_weak_rear_min_speed_px_per_ms", 0.6)), "weak_speed_too_low"),
            (app_len >= float(getattr(self, "mask_occlusion_weak_rear_min_approach_len_px", 30.0)), "weak_approach_len_too_short"),
        ]
        for ok, why in checks:
            if not ok:
                detail["mask_occlusion_weak_rear_skip_reason"] = why
                return False, detail
        detail["mask_occlusion_weak_rear_cached"] = True
        detail["mask_occlusion_weak_rear_skip_reason"] = ""
        return True, detail

    def _remember_weak_rear_tracklet_event(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        use_segment: bool,
        track_metrics: Dict[str, float],
        reject_reason: str,
    ) -> Dict[str, Any]:
        ok, detail = self._weak_rear_tracklet_ok(ev, track_metrics, reject_reason)
        if ok and bool(use_segment):
            self._remember_tracklet_event(
                cam, ev, event_pt, approach_pt, use_segment, track_metrics,
                weak=True, weak_reason=str(reject_reason or ""),
            )
        elif ok and not bool(use_segment):
            detail["mask_occlusion_weak_rear_cached"] = False
            detail["mask_occlusion_weak_rear_skip_reason"] = "no_valid_segment"
        return detail

    @staticmethod
    def _cross2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])

    def _rear_tracklet_match_for_pending(self, pending: PendingImpact) -> Optional[Dict[str, Any]]:
        """Find a rear tracklet that proves straight pass-through / near miss.

        The rear tracklet can have a different bullet_id because the hard mask may
        split one physical bullet into two event-camera tracks.  It must appear
        after the pending boundary contact, lie past the projected body boundary,
        stay close to the same line, and keep a compatible direction/speed.
        """
        if not bool(getattr(self, "mask_occlusion_judge_enable", True)):
            return None
        dq = self.recent_tracklets.get(str(pending.cam))
        if not dq:
            return None
        base_speed = 0.0
        try:
            base_speed = float((pending.base_candidate or {}).get("avg_speed_px_per_ms", 0.0) or 0.0)
        except Exception:
            base_speed = 0.0
        if base_speed <= 0.0:
            base_speed = float(getattr(pending.base_ev, "avg_speed_px_per_ms", 0.0) or 0.0)
        best: Optional[Dict[str, Any]] = None
        for rt in reversed(dq):
            dt_ms = (int(rt.event_ts_us) - int(pending.event_ts_us)) / 1000.0
            if dt_ms < float(self.mask_occlusion_rear_min_dt_ms):
                continue
            if dt_ms > float(self.mask_occlusion_rear_search_ms):
                # Older entries will only be older because the deque is time ordered.
                break
            if bool(getattr(rt, "weak", False)) and not bool(getattr(self, "mask_occlusion_rear_allow_weak_tracklet", True)):
                continue
            if int(rt.bullet_id) != int(pending.bullet_id) and not bool(self.mask_occlusion_allow_cross_bullet_stitch):
                continue
            rel = (float(rt.event_pt[0]) - float(pending.impact_xy[0]),
                   float(rt.event_pt[1]) - float(pending.impact_xy[1]))
            progress = rel[0] * float(pending.dir_in[0]) + rel[1] * float(pending.dir_in[1])
            lateral = abs(self._cross2(pending.dir_in, rel))
            if progress < float(self.mask_occlusion_rear_min_progress_px):
                continue
            if lateral > float(self.mask_occlusion_rear_max_lateral_error_px):
                continue
            angle = _angle_between_vectors_deg(pending.dir_in, rt.dir_xy)
            if angle > float(self.mask_occlusion_rear_max_angle_deg):
                continue
            if rt.track_path_len_px < float(self.mask_occlusion_rear_min_track_path_len_px):
                continue
            if rt.approach_len_px < float(self.mask_occlusion_rear_min_approach_len_px):
                continue
            if base_speed > 1e-6 and rt.speed_px_per_ms > 0.0:
                speed_ratio = float(rt.speed_px_per_ms) / max(1e-6, float(base_speed))
            else:
                speed_ratio = 1.0
            if speed_ratio < float(self.mask_occlusion_rear_min_speed_ratio) or speed_ratio > float(self.mask_occlusion_rear_max_speed_ratio):
                continue
            detail = {
                "mask_occlusion_rear_match": True,
                "mask_occlusion_rear_reason": "rear_tracklet_straight_through_mask",
                "mask_occlusion_rear_bullet_id": int(rt.bullet_id),
                "mask_occlusion_rear_seq_id": int(rt.seq_id),
                "mask_occlusion_rear_dt_ms": round(float(dt_ms), 3),
                "mask_occlusion_rear_progress_px": round(float(progress), 3),
                "mask_occlusion_rear_lateral_error_px": round(float(lateral), 3),
                "mask_occlusion_rear_angle_deg": round(float(angle), 3),
                "mask_occlusion_rear_speed_ratio": round(float(speed_ratio), 3),
                "mask_occlusion_rear_event_xy": [round(float(rt.event_pt[0]), 3), round(float(rt.event_pt[1]), 3)],
                "mask_occlusion_rear_approach_xy": [round(float(rt.approach_pt[0]), 3), round(float(rt.approach_pt[1]), 3)],
                "mask_occlusion_rear_cross_bullet": bool(int(rt.bullet_id) != int(pending.bullet_id)),
                "mask_occlusion_rear_weak_tracklet": bool(getattr(rt, "weak", False)),
                "mask_occlusion_rear_weak_reason": str(getattr(rt, "weak_reason", "") or ""),
                "mask_occlusion_rear_track_path_len_px": round(float(rt.track_path_len_px), 3),
                "mask_occlusion_rear_approach_len_px": round(float(rt.approach_len_px), 3),
                "mask_occlusion_rear_search_ms": round(float(self.mask_occlusion_rear_search_ms), 3),
                "mask_occlusion_rear_min_progress_px": round(float(self.mask_occlusion_rear_min_progress_px), 3),
                "mask_occlusion_rear_max_lateral_error_px": round(float(self.mask_occlusion_rear_max_lateral_error_px), 3),
                "mask_occlusion_rear_max_angle_deg": round(float(self.mask_occlusion_rear_max_angle_deg), 3),
            }
            rank = (float(angle), float(lateral), -float(progress), float(dt_ms))
            if best is None or rank < best.get("_rank", (999.0, 9999.0, 0.0, 9999.0)):
                detail["_rank"] = rank
                best = detail
        if best is not None:
            best.pop("_rank", None)
        return best

    def _pending_bounce_has_true_mask_contact(self, direct_detail: Dict[str, Any]) -> bool:
        if not bool(getattr(self, "mask_occlusion_bounce_requires_true_mask_contact", True)):
            return bool(direct_detail.get("impact_direct_body_contact", False))
        return bool(
            direct_detail.get("impact_direct_body_contact_by_method", False)
            or direct_detail.get("impact_direct_body_contact_by_flags", False)
        )

    def _pending_keys_for_bullet(self, cam: str, bullet_id: int) -> List[Tuple[str, int, int]]:
        return [k for k in list(self.pending_impacts.keys()) if k[0] == str(cam) and k[1] == int(bullet_id)]

    def _pending_geometry_methods(self, pending: PendingImpact) -> set:
        geom = pending.geometry if isinstance(pending.geometry, dict) else {}
        methods = {
            str(geom.get("geometry_best_method", "") or ""),
            str(geom.get("direct_mask_method_ignored_for_behavior_hit", "") or ""),
            str(geom.get("pending_contact_source_method", "") or ""),
        }
        if bool(geom.get("pending_created_from_direct_mask_contact", False)):
            methods.add("pending_mask_contact")
        if bool(geom.get("event_inside_mask", False)) or bool(geom.get("segment_intersects_mask", False)):
            methods.add("pending_mask_contact")
        return {m for m in methods if m}

    def _pending_has_strong_geometry_contact(self, pending: PendingImpact) -> bool:
        methods = self._pending_geometry_methods(pending)
        strong = set(getattr(self, "impact_absorb_strong_geometry_methods", set()) or set())
        return bool(methods.intersection(strong))

    def _pending_direct_body_contact_detail(self, pending: PendingImpact) -> Tuple[bool, Dict[str, Any]]:
        """Return whether the pending impact has real body-contact evidence.

        projected-only pending means: the event point was near the mask and the
        future ray would intersect it.  That is not enough to finalize hit_absorb
        or hit_bounce, because a foreground obstacle can produce the same geometry.
        Direct contact requires point/segment mask evidence or a very small gap.
        """
        geom = pending.geometry if isinstance(pending.geometry, dict) else {}
        methods = self._pending_geometry_methods(pending)
        direct_methods = {"point_in_mask", "segment_intersect_mask", "pending_mask_contact"}
        direct_by_method = bool(methods.intersection(direct_methods))
        direct_by_flags = bool(
            geom.get("pending_created_from_direct_mask_contact", False)
            or geom.get("event_inside_mask", False)
            or geom.get("segment_intersects_mask", False)
        )

        def _f(key: str, default: float = 1e9) -> float:
            try:
                v = geom.get(key, default)
                if v is None:
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        last_gap = _f("last_point_to_mask_dist_px")
        proj_len = _f("projection_len_px")
        max_gap = float(getattr(self, "impact_direct_body_contact_max_gap_px", 18.0))
        # A tiny outside gap is treated as contact to tolerate segmentation
        # boundary/homography error, but larger projection-only gaps remain guarded.
        direct_by_tiny_gap = bool(last_gap <= max_gap and proj_len <= max_gap)
        direct = bool(direct_by_method or direct_by_flags or direct_by_tiny_gap)
        detail = {
            "impact_direct_body_contact": bool(direct),
            "impact_direct_body_contact_methods": sorted(methods),
            "impact_direct_body_contact_by_method": bool(direct_by_method),
            "impact_direct_body_contact_by_flags": bool(direct_by_flags),
            "impact_direct_body_contact_by_tiny_gap": bool(direct_by_tiny_gap),
            "impact_direct_body_contact_max_gap_px": round(float(max_gap), 3),
            "impact_direct_body_contact_last_gap_px": None if last_gap >= 1e8 else round(float(last_gap), 3),
            "impact_direct_body_contact_projection_len_px": None if proj_len >= 1e8 else round(float(proj_len), 3),
            "impact_projected_only_guard": bool(not direct),
        }
        return direct, detail

    def _pending_has_direct_body_contact(self, pending: PendingImpact) -> bool:
        ok, _detail = self._pending_direct_body_contact_detail(pending)
        return bool(ok)

    def _update_pending_impacts_for_event(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        use_segment: bool,
        period: Optional[Dict[str, Any]],
        selection_info: Dict[str, Any],
        age_ms: float,
    ) -> Optional[str]:
        """Update an existing pending impact with a later event from the same bullet.

        v7 important change: this function is intentionally callable before human
        matching.  Once a pending impact has been created, subsequent events from
        the same bullet are enough to prove near-miss / bounce behaviour; they do
        not need a fresh current human frame.  This prevents a pending impact from
        timing out into a false hit_absorb just because later packets hit
        human_timeout.
        """
        keys = self._pending_keys_for_bullet(cam, int(ev.bullet_id))
        if not keys:
            return None

        vx = float(event_pt[0]) - float(approach_pt[0])
        vy = float(event_pt[1]) - float(approach_pt[1])
        if not use_segment or math.hypot(vx, vy) <= 1e-6:
            first = self.pending_impacts.get(keys[0])
            if first is not None:
                vx = float(event_pt[0]) - float(first.last_xy[0])
                vy = float(event_pt[1]) - float(first.last_xy[1])
        vnorm = math.hypot(vx, vy)
        if vnorm <= 1e-6:
            return None
        curr_dir = (vx / vnorm, vy / vnorm)

        any_result: Optional[str] = None
        for key in keys:
            pending = self.pending_impacts.get(key)
            if pending is None:
                continue
            # Ignore the exact event that created pending_impact.  Absorb must
            # never be emitted from the creator event itself.
            if int(ev.timestamp_us) <= int(pending.event_ts_us):
                continue

            angle = _angle_between_vectors_deg(pending.dir_in, curr_dir)
            impact_dist = math.hypot(float(event_pt[0]) - pending.impact_xy[0], float(event_pt[1]) - pending.impact_xy[1])
            progress = ((float(event_pt[0]) - pending.impact_xy[0]) * pending.dir_in[0] +
                        (float(event_pt[1]) - pending.impact_xy[1]) * pending.dir_in[1])
            move_from_pending = math.hypot(float(event_pt[0]) - float(pending.last_xy[0]),
                                           float(event_pt[1]) - float(pending.last_xy[1]))
            pending.seen_followup_events += 1
            pending.last_event_ts_us = max(int(pending.last_event_ts_us), int(ev.timestamp_us))
            pending.max_forward_progress_px = max(float(pending.max_forward_progress_px), float(progress))
            pending.max_move_from_pending_px = max(float(pending.max_move_from_pending_px), float(move_from_pending))
            extra = {
                "impact_behavior_angle_deg": round(float(angle), 3),
                "impact_behavior_point_dist_px": round(float(impact_dist), 3),
                "impact_behavior_forward_progress_px": round(float(progress), 3),
                "impact_behavior_current_xy": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
                "impact_behavior_followup_events": int(pending.seen_followup_events),
                "impact_behavior_max_forward_progress_px": round(float(pending.max_forward_progress_px), 3),
                "impact_behavior_max_move_from_pending_px": round(float(pending.max_move_from_pending_px), 3),
                "shot_bounce_link_followup": bool(self._is_shot_bounce_link_event(ev)),
                "shot_link_type": str(getattr(ev, "shot_link_type", "") or ""),
                "shot_link_reason": str(getattr(ev, "shot_link_reason", "") or ""),
                "shot_link_gap_ms": round(float(getattr(ev, "shot_link_gap_ms", 0.0) or 0.0), 3),
                "shot_link_gap_dist_px": round(float(getattr(ev, "shot_link_gap_dist_px", 0.0) or 0.0), 3),
                "shot_link_angle_deg": round(float(getattr(ev, "shot_link_angle_deg", 0.0) or 0.0), 3),
            }

            # If it keeps moving in almost the same direction beyond the predicted
            # body boundary, this is a near miss / pass-through, not an absorbed hit.
            # v14: after strong mask contact, require stricter pass-through proof.
            # A real hit can still leave a few residual events; do not convert those
            # to near_miss unless the continuation is long, stable, and repeated.
            strong_contact = self._pending_has_strong_geometry_contact(pending)
            if strong_contact:
                explicit_straight_now = (
                    angle <= float(self.impact_near_miss_max_angle_deg_for_strong)
                    and progress >= float(self.impact_near_miss_continue_len_px_for_strong)
                    and int(pending.seen_followup_events) >= int(self.impact_near_miss_min_followup_events_for_strong)
                    and float(pending.max_move_from_pending_px) >= float(self.impact_near_miss_min_move_px_for_strong)
                )
            else:
                explicit_straight_now = (
                    angle <= float(self.impact_near_miss_max_angle_deg)
                    and progress >= float(self.impact_near_miss_continue_len_px)
                )
            rear_match = self._rear_tracklet_match_for_pending(pending)
            if rear_match is not None:
                removed = self.pending_impacts.pop(key, None)
                if removed is not None:
                    geom = dict(removed.geometry)
                    geom.update(extra)
                    geom.update(rear_match)
                    self._emit_debug(cam, ev, "near_miss", "rear_tracklet_straight_through_mask", period=period or removed.period, selection=selection_info or removed.selection, geometry=geom, candidate=removed.base_candidate, age_ms=age_ms)
                any_result = "near_miss"
                continue
            if explicit_straight_now:
                removed = self.pending_impacts.pop(key, None)
                if removed is not None:
                    geom = dict(removed.geometry)
                    geom.update(extra)
                    geom.update({
                        "impact_behavior_explicit_straight_continue": True,
                        "impact_behavior_strong_geometry_contact": bool(strong_contact),
                        "impact_near_miss_continue_len_px_for_strong": round(float(getattr(self, "impact_near_miss_continue_len_px_for_strong", self.impact_near_miss_continue_len_px)), 3),
                        "impact_near_miss_min_followup_events_for_strong": int(getattr(self, "impact_near_miss_min_followup_events_for_strong", 0)),
                        "impact_near_miss_min_move_px_for_strong": round(float(getattr(self, "impact_near_miss_min_move_px_for_strong", self.impact_near_miss_continue_len_px)), 3),
                        "impact_near_miss_continue_len_px_for_strong": round(float(getattr(self, "impact_near_miss_continue_len_px_for_strong", self.impact_near_miss_continue_len_px)), 3),
                        "impact_near_miss_max_angle_deg_for_strong": round(float(getattr(self, "impact_near_miss_max_angle_deg_for_strong", self.impact_near_miss_max_angle_deg)), 3),
                    })
                    self._emit_debug(cam, ev, "near_miss", "continued_straight_past_projected_mask", period=period or removed.period, selection=selection_info or removed.selection, geometry=geom, candidate=removed.base_candidate, age_ms=age_ms)
                any_result = "near_miss"
                continue
            elif strong_contact and progress >= float(self.impact_near_miss_continue_len_px):
                pending.absorb_evidence = pending.absorb_evidence or "strong_mask_contact_ambiguous_continuation_not_clear_pass_through"

            # Boundary turn / bounce: a large direction change near the predicted impact point.
            # Guard projection-only pending impacts: a foreground obstacle can turn
            # near the human mask without touching the body.
            if angle >= float(self.impact_bounce_min_angle_deg) and impact_dist <= float(self.impact_bounce_max_boundary_dist_px):
                direct_contact, direct_detail = self._pending_direct_body_contact_detail(pending)
                extra.update(direct_detail)
                if (not direct_contact) and (not bool(self.impact_allow_projected_only_bounce)):
                    removed = self.pending_impacts.pop(key, None)
                    if removed is not None:
                        geom = dict(removed.geometry)
                        geom.update(extra)
                        geom.update({
                            "mask_occlusion_obstacle_guard": True,
                            "mask_occlusion_obstacle_reason": "turn_point_outside_mask_no_direct_body_contact",
                        })
                        self._emit_debug(
                            cam, ev, "obstacle_bounce_near_person",
                            "projected_only_turn_near_mask_obstacle_guard",
                            period=period or removed.period,
                            selection=selection_info or removed.selection,
                            geometry=geom,
                            candidate=removed.base_candidate,
                            age_ms=age_ms,
                        )
                    any_result = "obstacle_bounce_near_person"
                    continue
                if not self._pending_bounce_has_true_mask_contact(direct_detail):
                    removed = self.pending_impacts.pop(key, None)
                    if removed is not None:
                        geom = dict(removed.geometry)
                        geom.update(extra)
                        geom.update({
                            "mask_occlusion_obstacle_guard": True,
                            "mask_occlusion_obstacle_reason": "bounce_requires_true_mask_contact_tiny_gap_only",
                            "mask_occlusion_bounce_requires_true_mask_contact": bool(self.mask_occlusion_bounce_requires_true_mask_contact),
                        })
                        self._emit_debug(
                            cam, ev, "obstacle_bounce_near_person",
                            "turn_near_mask_but_no_true_mask_contact",
                            period=period or removed.period,
                            selection=selection_info or removed.selection,
                            geometry=geom,
                            candidate=removed.base_candidate,
                            age_ms=age_ms,
                        )
                    any_result = "obstacle_bounce_near_person"
                    continue
                # Body-motion guard for hit_bounce: a waving arm/edge can create
                # a direct mask-contact turn, so bounce also needs to look like an
                # external incoming projectile.  This reuses the same gate as
                # absorb: enough track length, enough approach length, approach
                # origin outside the body mask, and sufficient speed.
                ok_bounce_gate, bounce_reject_reason, bounce_gate_detail = self._hit_absorb_external_approach_ok(pending)
                extra.update(bounce_gate_detail)
                if not ok_bounce_gate:
                    removed = self.pending_impacts.pop(key, None)
                    if removed is not None:
                        geom = dict(removed.geometry)
                        geom.update(extra)
                        self._emit_debug(
                            cam, ev, "impact_bounce_rejected",
                            f"bounce_external_approach_gate_failed:{bounce_reject_reason}",
                            period=period or removed.period,
                            selection=selection_info or removed.selection,
                            geometry=geom,
                            candidate=removed.base_candidate,
                            age_ms=age_ms,
                        )
                    any_result = "impact_bounce_rejected"
                    continue
                any_result = self._finalize_pending_impact(key, "hit_bounce", "large_turn_near_mask_boundary", ev=ev, extra_geometry=extra)
                continue

            # A real non-probe termination near the predicted boundary is absorb
            # evidence, but v7 no longer emits hit_absorb immediately.  It waits
            # until the observation window expires so a later continuation can
            # still convert the pending impact to near_miss.
            if ev.event_type == EVENT_TERMINATE and str(ev.terminate_reason) != "track_probe" and impact_dist <= float(self.impact_bounce_max_boundary_dist_px):
                pending.absorb_evidence = f"terminate_near_mask_boundary:{ev.terminate_reason}"
                geom = dict(pending.geometry)
                geom.update(extra)
                self._emit_debug(cam, ev, "pending_absorb_evidence", pending.absorb_evidence, period=period or pending.period, selection=selection_info or pending.selection, geometry=geom, candidate=pending.base_candidate, age_ms=age_ms)

            pending.last_xy = (float(event_pt[0]), float(event_pt[1]))
            pending.last_update_wall = time.time()
        return any_result

    def _hit_absorb_external_approach_ok(self, pending: PendingImpact) -> Tuple[bool, str, Dict[str, Any]]:
        """Return whether a pending impact is allowed to become hit_absorb.

        v11 safety gate: with human digital masking, body movement can still
        create short line fragments near the mask boundary.  A real absorbed hit
        should have an external incoming track: long enough path, long enough
        approach segment, approach point clearly outside the mask boundary, and
        sufficient speed.
        """
        d = pending.base_candidate if isinstance(pending.base_candidate, dict) else {}
        geom = pending.geometry if isinstance(pending.geometry, dict) else {}

        def _f(obj: Dict[str, Any], key: str, default: float = 0.0) -> float:
            try:
                v = obj.get(key, default)
                if v is None:
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        track_path = _f(d, "track_path_len_px")
        approach_len = _f(d, "approach_len_px", _f(d, "recent_approach_len_px"))
        speed = _f(d, "avg_speed_px_per_ms", float(getattr(pending.base_ev, "avg_speed_px_per_ms", 0.0) or 0.0))
        if speed <= 0.0:
            speed = float(getattr(pending.base_ev, "avg_speed_px_per_ms", 0.0) or 0.0)
        approach_to_mask = _f(d, "impact_approach_to_mask_dist_px", _f(geom, "impact_approach_to_mask_dist_px", -1.0))

        strong_geometry_contact = self._pending_has_strong_geometry_contact(pending)
        direct_body_contact, direct_body_detail = self._pending_direct_body_contact_detail(pending)
        relax_external_gate = bool(self.impact_absorb_relax_external_gate_on_strong_geometry) and bool(strong_geometry_contact)
        detail = {
            "impact_absorb_external_gate": True,
            "impact_absorb_strong_geometry_contact": bool(strong_geometry_contact),
            "impact_absorb_direct_body_contact": bool(direct_body_contact),
            "impact_allow_projected_only_absorb": bool(self.impact_allow_projected_only_absorb),
            "impact_absorb_relax_external_gate_on_strong_geometry": bool(relax_external_gate),
            "impact_absorb_track_path_len_px": round(float(track_path), 3),
            "impact_absorb_min_track_path_len_px": round(float(self.impact_absorb_min_track_path_len_px), 3),
            "impact_absorb_approach_len_px": round(float(approach_len), 3),
            "impact_absorb_min_approach_len_px": round(float(self.impact_absorb_min_approach_len_px), 3),
            "impact_absorb_approach_to_mask_dist_px": round(float(approach_to_mask), 3),
            "impact_absorb_min_approach_to_mask_dist_px": round(float(self.impact_absorb_min_approach_to_mask_dist_px), 3),
            "impact_absorb_avg_speed_px_per_ms": round(float(speed), 3),
            "impact_absorb_min_avg_speed_px_per_ms": round(float(self.impact_absorb_min_avg_speed_px_per_ms), 3),
        }
        detail.update(direct_body_detail)
        if (not direct_body_contact) and (not bool(self.impact_allow_projected_only_absorb)):
            return False, "projected_only_no_direct_body_contact_obstacle_guard", detail

        if track_path < float(self.impact_absorb_min_track_path_len_px):
            if relax_external_gate:
                detail["impact_absorb_relaxed_track_path_too_short"] = True
            else:
                return False, "absorb_track_path_too_short", detail
        if approach_len < float(self.impact_absorb_min_approach_len_px):
            if relax_external_gate:
                detail["impact_absorb_relaxed_approach_len_too_short"] = True
            else:
                return False, "absorb_approach_len_too_short", detail
        if approach_to_mask >= 0.0 and approach_to_mask < float(self.impact_absorb_min_approach_to_mask_dist_px):
            if relax_external_gate:
                detail["impact_absorb_relaxed_approach_origin_too_close_to_mask"] = True
            else:
                return False, "absorb_approach_origin_too_close_to_mask", detail
        if speed < float(self.impact_absorb_min_avg_speed_px_per_ms):
            if relax_external_gate:
                detail["impact_absorb_relaxed_speed_too_low"] = True
            else:
                return False, "absorb_speed_too_low", detail
        return True, "", detail

    def _process_pending_impacts(self) -> None:
        if not self.pending_impacts:
            return
        now = time.time()
        for key, pending in list(self.pending_impacts.items()):
            age_ms = (now - pending.created_wall) * 1000.0
            if age_ms < float(self.impact_absorb_min_pending_age_ms):
                continue
            # If any later event from the same bullet moved far enough beyond the
            # projected body boundary, force near_miss here as a safety net.  This
            # catches cases where the continuation did not have a fresh human frame.
            strong_contact = self._pending_has_strong_geometry_contact(pending)
            if strong_contact:
                explicit_straight_continue = (
                    float(pending.max_forward_progress_px) >= float(self.impact_near_miss_continue_len_px_for_strong)
                    and int(pending.seen_followup_events) >= int(self.impact_near_miss_min_followup_events_for_strong)
                    and float(pending.max_move_from_pending_px) >= float(self.impact_near_miss_min_move_px_for_strong)
                )
            else:
                explicit_straight_continue = (
                    float(pending.max_forward_progress_px) >= float(self.impact_near_miss_continue_len_px)
                )
            rear_match = self._rear_tracklet_match_for_pending(pending)
            if rear_match is not None:
                removed = self.pending_impacts.pop(key, None)
                if removed is not None:
                    geom = dict(removed.geometry)
                    geom.update({
                        "impact_behavior_timeout_ms": round(float(age_ms), 3),
                        "impact_behavior_followup_events": int(removed.seen_followup_events),
                        "impact_behavior_max_forward_progress_px": round(float(removed.max_forward_progress_px), 3),
                        "impact_behavior_max_move_from_pending_px": round(float(removed.max_move_from_pending_px), 3),
                        "impact_behavior_strong_geometry_contact": bool(strong_contact),
                    })
                    geom.update(rear_match)
                    self._emit_debug(removed.cam, removed.base_ev, "near_miss", "rear_tracklet_straight_through_mask", period=removed.period, selection=removed.selection, geometry=geom, candidate=removed.base_candidate, age_ms=age_ms)
                continue
            if explicit_straight_continue:
                removed = self.pending_impacts.pop(key, None)
                if removed is not None:
                    geom = dict(removed.geometry)
                    geom.update({
                        "impact_behavior_timeout_ms": round(float(age_ms), 3),
                        "impact_behavior_followup_events": int(removed.seen_followup_events),
                        "impact_behavior_max_forward_progress_px": round(float(removed.max_forward_progress_px), 3),
                        "impact_behavior_max_move_from_pending_px": round(float(removed.max_move_from_pending_px), 3),
                        "impact_behavior_explicit_straight_continue": True,
                        "impact_behavior_strong_geometry_contact": bool(strong_contact),
                    })
                    self._emit_debug(removed.cam, removed.base_ev, "near_miss", "continued_after_pending_timeout", period=removed.period, selection=removed.selection, geometry=geom, candidate=removed.base_candidate, age_ms=age_ms)
                continue
            elif strong_contact and float(pending.max_forward_progress_px) >= float(self.impact_near_miss_continue_len_px):
                # For strong mask contact, a single small/ambiguous follow-up beyond
                # the projection is not enough to prove pass-through.  Treat it as
                # absorb evidence unless the explicit straight-continuation gate above
                # is satisfied.
                pending.absorb_evidence = pending.absorb_evidence or "strong_mask_contact_timeout_no_clear_pass_through"
            if bool(self.impact_absorb_require_no_followup) and int(pending.seen_followup_events) > 0 and float(pending.max_forward_progress_px) > float(self.impact_absorb_max_followup_progress_px):
                # Strong mask contact should not be rejected merely because there
                # were a few follow-up events.  Only the explicit straight
                # continuation gate above is allowed to convert it to near_miss.
                # Otherwise keep it pending until timeout and let absorb/bounce
                # confirmation decide.
                if self._pending_has_strong_geometry_contact(pending) and float(pending.max_forward_progress_px) < float(self.impact_near_miss_continue_len_px):
                    pending.absorb_evidence = pending.absorb_evidence or "strong_mask_contact_with_limited_followup"
                else:
                    removed = self.pending_impacts.pop(key, None)
                    if removed is not None:
                        geom = dict(removed.geometry)
                        geom.update({
                            "impact_behavior_timeout_ms": round(float(age_ms), 3),
                            "impact_behavior_followup_events": int(removed.seen_followup_events),
                            "impact_behavior_max_forward_progress_px": round(float(removed.max_forward_progress_px), 3),
                            "impact_behavior_max_move_from_pending_px": round(float(removed.max_move_from_pending_px), 3),
                        })
                        self._emit_debug(removed.cam, removed.base_ev, "near_miss", "followup_after_pending_not_absorbed", period=removed.period, selection=removed.selection, geometry=geom, candidate=removed.base_candidate, age_ms=age_ms)
                    continue
            # No meaningful continuation after the observation window: with the
            # human digital mask enabled, this is the absorbed-hit signature, but
            # v11 requires the pending track to look like an external incoming
            # bullet.  Short fragments born near the human mask boundary are
            # treated as human-motion false positives, not hits.
            reason = pending.absorb_evidence or "disappeared_near_mask_boundary"
            extra = {
                "impact_behavior_timeout_ms": round(float(age_ms), 3),
                "impact_behavior_followup_events": int(pending.seen_followup_events),
                "impact_behavior_max_forward_progress_px": round(float(pending.max_forward_progress_px), 3),
                "impact_behavior_max_move_from_pending_px": round(float(pending.max_move_from_pending_px), 3),
            }
            ok_absorb, reject_reason, gate_detail = self._hit_absorb_external_approach_ok(pending)
            extra.update(gate_detail)
            if not ok_absorb:
                removed = self.pending_impacts.pop(key, None)
                if removed is not None:
                    geom = dict(removed.geometry)
                    geom.update(extra)
                    self._emit_debug(removed.cam, removed.base_ev, "impact_absorb_rejected", reject_reason, period=removed.period, selection=removed.selection, geometry=geom, candidate=removed.base_candidate, age_ms=age_ms)
                continue
            self._finalize_pending_impact(key, "hit_absorb", reason, ev=pending.base_ev, extra_geometry=extra)

    def process_pending_events(self) -> None:
        if not self.pending_events:
            return
        keep: Deque[PendingBulletEvent] = deque(maxlen=self.pending_events.maxlen or 2048)
        now = time.time()
        while self.pending_events:
            pe = self.pending_events.popleft()
            pe.tries += 1
            age_ms = (now - pe.first_wall) * 1000.0
            force_final = age_ms > self.impact_pending_timeout_ms
            result = self.judge_event(pe.ev, pe.cam, age_ms=age_ms, force_final=force_final)
            if result.startswith("wait"):
                if not force_final:
                    keep.append(pe)
                    self.wait_count += 1
                else:
                    # judge_event(force_final=True) should have emitted a debug line.
                    self.drop_count += 1
            elif result == "no_human":
                self.no_human_count += 1
            elif result == "no_cam":
                self.no_cam_count += 1
            elif result in {"drop", "bad_trigger", "no_trigger"}:
                self.drop_count += 1
        self.pending_events = keep

    def judge_event(self, ev: BulletEvent, cam: Optional[str] = None, age_ms: float = 0.0, force_final: bool = False) -> str:
        cam = cam or self._camera_for_event(ev)
        if cam is None:
            self._emit_debug(None, ev, "no_camera", "camera_not_mapped", age_ms=age_ms)
            return "no_cam"
        H = self.H.get(cam)
        if H is None:
            self._emit_debug(cam, ev, "no_homography", "homography_missing", age_ms=age_ms)
            return "drop"

        track_metrics = self._event_track_metrics(ev)
        track_point_count = int(track_metrics.get("track_point_count", 0) or 0)
        track_path_len_px = float(track_metrics.get("track_path_len_px", 0.0) or 0.0)
        track_duration_ms = float(track_metrics.get("track_duration_ms", 0.0) or 0.0)
        track_displacement_px = float(track_metrics.get("track_displacement_px", 0.0) or 0.0)
        track_straightness = float(track_metrics.get("track_straightness", 0.0) or 0.0)
        track_line_error_px = float(track_metrics.get("track_line_error_px", 999.0) if track_metrics.get("track_line_error_px", 999.0) is not None else 999.0)
        approach_len_px = float(track_metrics.get("approach_len_px", 0.0) or 0.0)

        # Compute mapped trajectory before the track-quality gate.  Existing
        # pending impacts must still observe later events from the same bullet
        # even if those later events end as unknown/probation or have short/dirty
        # metrics.  Otherwise a real pass-through can be misclassified as an
        # absorbed hit simply because the continuation was rejected before it
        # updated the pending state.
        event_pt = _apply_H(H, float(ev.event_x), float(ev.event_y))
        raw_approach_valid = bool(ev.approach_valid) and approach_len_px >= float(self.min_hit_trajectory_len_px)
        use_segment = bool(self.trajectory_segment_enable and raw_approach_valid)
        approach_pt = _apply_H(H, float(ev.approach_x), float(ev.approach_y)) if use_segment else event_pt

        early_pending_result: Optional[str] = None
        if bool(self.enable_impact_behavior_hit) and bool(self.impact_update_pending_before_human):
            early_pending_result = self._update_pending_impacts_for_event(
                cam, ev, event_pt, approach_pt, bool(use_segment), None, {}, age_ms
            )
            if early_pending_result == "emitted":
                return "emitted"
            if early_pending_result in {"near_miss", "obstacle_bounce_near_person", "impact_bounce_rejected"}:
                return "no_candidate"

        is_shot_bounce_link = self._is_shot_bounce_link_event(ev)
        if bool(is_shot_bounce_link) and bool(getattr(self, "shot_bounce_link_context_only", True)):
            # Do not let a P19 link event create a fresh pending impact/final HIT.
            # It has already had a chance above to update an existing pending impact.
            self._emit_debug(
                cam,
                ev,
                "shot_bounce_link_context_only",
                "shot_bounce_link_does_not_create_new_hit",
                geometry={
                    "shot_bounce_link_context_only": True,
                    "pending_update_checked_before_context_gate": bool(self.enable_impact_behavior_hit and self.impact_update_pending_before_human),
                    "track_point_count": int(track_point_count),
                    "track_path_len_px": round(float(track_path_len_px), 3),
                    "track_duration_ms": round(float(track_duration_ms), 3),
                    "track_straightness": round(float(track_straightness), 3),
                    "track_line_error_px": round(float(track_line_error_px), 3),
                },
                age_ms=age_ms,
            )
            return "no_candidate"

        track_ok, track_reason, track_detail = self._track_quality_ok(ev, track_metrics)
        if not track_ok:
            weak_rear_detail = self._remember_weak_rear_tracklet_event(
                cam, ev, event_pt, approach_pt, bool(use_segment), track_metrics, str(track_reason)
            )
            self._emit_debug(
                cam,
                ev,
                "no_valid_track",
                track_reason,
                geometry={
                    **track_detail,
                    **weak_rear_detail,
                    "pending_update_checked_before_track_gate": bool(self.enable_impact_behavior_hit and self.impact_update_pending_before_human),
                },
                age_ms=age_ms,
            )
            return "no_candidate"

        self._remember_tracklet_event(cam, ev, event_pt, approach_pt, bool(use_segment), track_metrics)

        period: Optional[Dict[str, Any]] = None
        strategy_used = "nearest_time_fallback"
        selection_info: Dict[str, Any] = {}
        if self.impact_sync_enable:
            period = self._impact_period_for_event(cam, int(ev.timestamp_us))
            if period is None:
                if force_final:
                    self._emit_debug(cam, ev, "no_trigger_period", "no_trigger_period_for_impact_ts", age_ms=age_ms)
                    return "no_trigger"
                return "wait_trigger"
            if not bool(period.get("trigger_period_ok", False)):
                if not self.emit_candidate_on_bad_trigger_period:
                    self._emit_debug(cam, ev, "bad_trigger_period", "bad_trigger_period", period=period, age_ms=age_ms)
                    return "bad_trigger"
            entries, strategy_used, should_wait, selection_info = self._entries_for_impact(cam, period, age_ms=age_ms, force_final=force_final)
            if should_wait and not force_final:
                return "wait_human"
            if not entries:
                status = "human_timeout" if force_final else "no_human_result"
                self._emit_debug(cam, ev, status, "no_human_result_near_sync", period=period, selection=selection_info, age_ms=age_ms)
                return "no_human"

            # For final candidates, do not use adjacent/nearby person frames unless
            # explicitly enabled.  This prevents an event that happened before/after
            # a person frame from being attached to the wrong human mask.
            frame_delta = selection_info.get("person_frame_delta_sync", None)
            try:
                frame_delta_i = int(frame_delta) if frame_delta is not None else 999999
            except Exception:
                frame_delta_i = 999999
            fallback_used = bool(selection_info.get("human_fallback_used", False))
            behavior_only_mode = bool(self.enable_impact_behavior_hit) and not bool(self.impact_behavior_direct_mask_final_hit)
            effective_allow_fallback = bool(self.allow_human_fallback_for_hit)
            effective_max_delta_sync = int(self.max_person_frame_delta_sync_for_hit)
            effective_sync_policy = "direct_final_hit"
            if behavior_only_mode:
                effective_allow_fallback = bool(self.impact_behavior_allow_human_fallback)
                effective_max_delta_sync = int(self.impact_behavior_max_person_frame_delta_sync)
                effective_sync_policy = "impact_behavior_only"
                selection_info["impact_behavior_fallback_policy"] = effective_sync_policy
                selection_info["impact_behavior_allow_human_fallback"] = bool(effective_allow_fallback)
                selection_info["impact_behavior_max_person_frame_delta_sync"] = int(effective_max_delta_sync)
            if fallback_used and not bool(effective_allow_fallback):
                sync_detail = {
                    **track_detail,
                    "sync_policy": str(effective_sync_policy),
                    "allow_human_fallback_for_hit": bool(self.allow_human_fallback_for_hit),
                    "max_person_frame_delta_sync_for_hit": int(self.max_person_frame_delta_sync_for_hit),
                    "impact_behavior_allow_human_fallback": bool(self.impact_behavior_allow_human_fallback),
                    "impact_behavior_max_person_frame_delta_sync": int(self.impact_behavior_max_person_frame_delta_sync),
                }
                self._emit_debug(
                    cam,
                    ev,
                    "no_valid_human_sync",
                    "human_fallback_rejected_for_final_hit",
                    period=period,
                    selection=selection_info,
                    geometry=sync_detail,
                    age_ms=age_ms,
                )
                return "no_candidate"
            if abs(frame_delta_i) > int(effective_max_delta_sync):
                sync_detail = {
                    **track_detail,
                    "sync_policy": str(effective_sync_policy),
                    "allow_human_fallback_for_hit": bool(self.allow_human_fallback_for_hit),
                    "max_person_frame_delta_sync_for_hit": int(self.max_person_frame_delta_sync_for_hit),
                    "impact_behavior_allow_human_fallback": bool(self.impact_behavior_allow_human_fallback),
                    "impact_behavior_max_person_frame_delta_sync": int(self.impact_behavior_max_person_frame_delta_sync),
                    "effective_max_person_frame_delta_sync": int(effective_max_delta_sync),
                }
                self._emit_debug(
                    cam,
                    ev,
                    "no_valid_human_sync",
                    "person_frame_delta_sync_too_large",
                    period=period,
                    selection=selection_info,
                    geometry=sync_detail,
                    age_ms=age_ms,
                )
                return "no_candidate"
        else:
            entry = self._nearest_person_entry(cam, int(ev.timestamp_us))
            if entry is None:
                self._emit_debug(cam, ev, "no_human_result", "nearest_time_no_human", age_ms=age_ms)
                return "no_human"
            entries = [entry]
            selection_info = {
                "target_person_sync_index": int(entry.sync_index),
                "used_person_sync_index": int(entry.sync_index),
                "human_fallback_used": False,
                "human_fallback_reason": "",
                "person_frame_delta_sync": 0,
            }

        geometry_summary: Dict[str, Any] = {
            "event_point_mvs": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "approach_point_mvs": [round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],
            "approach_len_px": round(float(approach_len_px), 3),
            "approach_valid_raw": bool(ev.approach_valid),
            "trajectory_segment_used": bool(use_segment),
            "trajectory_required_for_hit": bool(self.require_trajectory_for_hit),
            "min_hit_trajectory_len_px": round(float(self.min_hit_trajectory_len_px), 3),
            "geometry_tested_persons": 0,
            "geometry_best_method": None,
            "geometry_best_score": 0.0,
            "evidence_level": "none",
            **track_detail,
        }

        if bool(self.require_trajectory_for_hit) and not use_segment:
            self._emit_debug(
                cam,
                ev,
                "no_valid_trajectory",
                "missing_or_short_trajectory",
                period=period,
                selection=selection_info,
                geometry=geometry_summary,
                age_ms=age_ms,
            )
            return "no_candidate"

        if bool(self.enable_impact_behavior_hit) and not bool(self.impact_update_pending_before_human):
            pending_result = self._update_pending_impacts_for_event(
                cam, ev, event_pt, approach_pt, bool(use_segment), period, selection_info, age_ms
            )
            if pending_result == "emitted":
                return "emitted"
            if pending_result in {"near_miss", "obstacle_bounce_near_person", "impact_bounce_rejected"}:
                return "no_candidate"

        candidates_meta: List[Tuple[HitCandidate, PersonEntry, Dict[str, Any]]] = []
        for entry in entries:
            if entry is None or not entry.persons:
                continue
            for person in entry.persons:
                if not isinstance(person, dict):
                    continue
                geometry_summary["geometry_tested_persons"] += 1
                box = person.get("bbox_xyxy") or person.get("bbox")
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                rect = _expand_bbox_xyxy(box, self.bbox_expand_px, entry.frame_w, entry.frame_h)
                method = None
                space = 0.0
                evidence_level = "none"
                projected_detail: Optional[Dict[str, Any]] = None
                mask_method = None
                mask_space = 0.0
                mask_evidence = "none"
                if self.impact_use_mask:
                    mask_method, mask_space, mask_evidence = _person_mask_hit(person, event_pt, approach_pt, bool(use_segment))
                    # In v7 impact-behaviour mode, direct mask contact is only a
                    # diagnostic hint.  It must not emit a final candidate by
                    # itself, because a pass-through / near-miss can also put a
                    # point or segment on the mask.
                    if (not bool(self.enable_impact_behavior_hit)) or bool(self.impact_behavior_direct_mask_final_hit):
                        method, space, evidence_level = mask_method, mask_space, mask_evidence
                    elif mask_method is not None and float(mask_space) > float(geometry_summary.get("geometry_best_score", 0.0)):
                        geometry_summary["geometry_best_method"] = str(mask_method)
                        geometry_summary["geometry_best_score"] = round(float(mask_space), 3)
                        geometry_summary["evidence_level"] = str(mask_evidence)
                        geometry_summary["direct_mask_method_ignored_for_behavior_hit"] = str(mask_method)
                if bool(self.enable_impact_behavior_hit) and self.impact_use_mask:
                    projected_detail = _person_projected_impact(
                        person,
                        event_pt,
                        approach_pt,
                        bool(use_segment),
                        float(self.impact_project_extend_px),
                        float(self.impact_boundary_dist_px),
                        float(self.impact_min_recent_len_px),
                    )
                    if projected_detail is not None:
                        method = str(projected_detail.get("method", "pending_projected_impact"))
                        space = float(projected_detail.get("space_score", 0.53) or 0.53)
                        evidence_level = str(projected_detail.get("evidence_level", "pending") or "pending")
                    elif bool(self.impact_behavior_mask_contact_pending) and mask_method is not None:
                        projected_detail = _person_mask_contact_pending_detail(
                            person,
                            event_pt,
                            approach_pt,
                            bool(use_segment),
                            mask_method,
                            float(mask_space),
                            str(mask_evidence),
                            float(self.impact_mask_contact_pending_min_recent_len_px),
                        )
                        if projected_detail is not None:
                            method = str(projected_detail.get("method", "pending_projected_impact"))
                            space = float(projected_detail.get("space_score", 0.56) or 0.56)
                            evidence_level = str(projected_detail.get("evidence_level", "pending_mask_contact") or "pending_mask_contact")
                if method is None:
                    if _point_in_rect(event_pt, rect):
                        method = MATCH_POINT_IN_BBOX
                        space = 0.50
                        evidence_level = "medium"
                    elif use_segment and _segment_intersects_rect(approach_pt, event_pt, rect):
                        method = MATCH_SEGMENT_INTERSECT
                        space = 0.40
                        evidence_level = "weak"
                    else:
                        d_rect = _dist_to_rect(event_pt, rect)
                        if d_rect <= self.near_px:
                            method = MATCH_POINT_NEAR_BBOX
                            space = max(0.20, 0.34 * (1.0 - d_rect / max(1.0, self.near_px)))
                            evidence_level = "weak"
                if method is None:
                    continue
                if space > float(geometry_summary.get("geometry_best_score", 0.0)):
                    geometry_summary["geometry_best_method"] = str(method)
                    geometry_summary["geometry_best_score"] = round(float(space), 3)
                    geometry_summary["evidence_level"] = evidence_level
                    if projected_detail:
                        geometry_summary.update(projected_detail)
                if bool(self.final_hit_mask_only) and not bool(self.allow_bbox_final_hit):
                    if bool(self.enable_impact_behavior_hit) and not bool(self.impact_behavior_direct_mask_final_hit):
                        allowed_mask_methods = {"pending_projected_impact"}
                    else:
                        allowed_mask_methods = {"point_in_mask", "segment_intersect_mask"}
                        if bool(self.enable_impact_behavior_hit):
                            allowed_mask_methods.add("pending_projected_impact")
                    if str(method) not in allowed_mask_methods:
                        geometry_summary["non_behavior_method_ignored_for_final_hit"] = str(method)
                        continue
                if period is not None:
                    # This dt is not UDP/wall time.  It is the offset between the
                    # impact timestamp and the selected MVS frame's trigger time.
                    dt_ms = abs(int(entry.event_ts_us or 0) - int(ev.timestamp_us)) / 1000.0 if entry.event_ts_us else abs(float(period.get("offset_ms", 0.0)))
                    # Penalize within the 40 ms period gently.  The frame was selected
                    # by sync period, so large offsets are expected and not fatal.
                    time_score = max(0.0, 0.24 * (1.0 - min(dt_ms, self.expected_period_us / 1000.0) / max(1.0, self.expected_period_us / 1000.0)))
                else:
                    dt_ms = abs(int(entry.event_ts_us or 0) - int(ev.timestamp_us)) / 1000.0
                    time_score = max(0.0, 0.22 * (1.0 - dt_ms / max(1.0, self.max_dt_ms)))
                # Impact-like events are now the only events judged, so event score
                # reflects reliability: terminate/disappear/end > turn.
                event_score = 0.14 if ev.event_type == EVENT_TERMINATE else 0.11
                speed_score = min(0.08, max(0.0, float(ev.avg_speed_px_per_ms) / 3.0 * 0.08))
                delta_sync = int(entry.sync_index) - int(selection_info.get("target_person_sync_index", entry.sync_index))
                loss_penalty = 0.0
                if bool(selection_info.get("human_fallback_used", False)):
                    loss_penalty -= self.fallback_penalty_base + abs(delta_sync) * self.fallback_penalty_per_sync
                if period is not None and bool(period.get("estimated_next", False)):
                    loss_penalty -= self.estimated_next_trigger_penalty
                if period is not None and not bool(period.get("trigger_period_ok", True)):
                    loss_penalty -= 0.25
                score = ScoreBreakdown(time_score=time_score, space_score=space, event_score=event_score, speed_score=speed_score, loss_penalty=loss_penalty)
                bbox = rect
                try:
                    raw_box = list(box)
                    if len(raw_box) >= 4:
                        x1, y1, x2, y2 = [float(v) for v in raw_box[:4]]
                        if x2 <= x1 or y2 <= y1:
                            x2, y2 = x1 + float(raw_box[2]), y1 + float(raw_box[3])
                        bbox = (int(round(x1)), int(round(y1)), int(round(x2-x1)), int(round(y2-y1)))
                except Exception:
                    bbox = (int(rect[0]), int(rect[1]), int(rect[2]-rect[0]), int(rect[3]-rect[1]))
                cand = HitCandidate(
                    camera_sn=str(ev.camera_sn),
                    camera_alias=cam,
                    bullet_id=int(ev.bullet_id),
                    seq_id=int(ev.seq_id),
                    stable_id=int(person.get("stable_id", person.get("person_id", 0)) or 0),
                    event_ts=int(ev.timestamp_us),
                    person_ts=int(entry.event_ts_us or 0),
                    dt_ms=float(dt_ms),
                    event_type=str(ev.event_type),
                    terminate_reason=str(ev.terminate_reason),
                    turn_angle_deg=float(ev.turn_angle_deg),
                    event_point=(float(event_pt[0]), float(event_pt[1])),
                    approach_point=(float(approach_pt[0]), float(approach_pt[1])),
                    approach_valid=bool(use_segment),
                    match_method=str(method),
                    track_point_count=int(track_point_count),
                    track_path_len_px=float(track_path_len_px),
                    track_duration_ms=float(track_duration_ms),
                    track_displacement_px=float(track_displacement_px),
                    track_straightness=float(track_straightness),
                    track_line_error_px=float(track_line_error_px),
                    person_bbox=tuple(int(v) for v in bbox),
                    person_centroid=tuple(person.get("foot_pixel") or person.get("centroid") or (0.0, 0.0)),
                    person_confirmed=True,
                    person_held=False,
                    score_breakdown=score,
                )
                pending_behavior_method = bool(projected_detail) and str(method) == "pending_projected_impact"
                pending_mask_contact_method = bool(pending_behavior_method) and bool(
                    projected_detail.get("pending_created_from_direct_mask_contact", False)
                    or str(projected_detail.get("evidence_level", "")) == "pending_mask_contact"
                )
                if pending_behavior_method:
                    required_score = float(self.impact_behavior_mask_contact_min_pending_score) if pending_mask_contact_method else float(self.impact_behavior_min_pending_score)
                else:
                    required_score = float(self.min_score)
                if cand.total_score >= required_score:
                    meta = {
                        "evidence_level": evidence_level,
                        "geometry_mode": str(method),
                        "used_person_sync_index": int(entry.sync_index),
                        "person_frame_delta_sync": int(delta_sync),
                        "required_score": round(float(required_score), 3),
                    }
                    if projected_detail:
                        meta.update(projected_detail)
                        meta["pending_impact"] = True
                    candidates_meta.append((cand, entry, meta))
                elif pending_behavior_method:
                    geometry_summary["pending_projected_impact_score_rejected"] = {
                        "total_score": round(float(cand.total_score), 3),
                        "required_score": round(float(required_score), 3),
                        "person_frame_delta_sync": int(delta_sync),
                    }
        candidates_meta.sort(key=lambda x: x[0].total_score, reverse=True)
        if not candidates_meta:
            self._emit_debug(cam, ev, "no_geometry_match", "no_person_geometry_match", period=period, selection=selection_info, geometry=geometry_summary, age_ms=age_ms)
            return "no_candidate"
        best, best_entry, best_meta = candidates_meta[0]
        d = self._candidate_to_output_dict(
            best,
            best_entry,
            ev,
            cam,
            period,
            selection_info,
            strategy_used,
            best_meta,
            track_metrics,
            bool(use_segment),
            float(approach_len_px),
        )

        if bool(best_meta.get("pending_impact", False)) and str(best.match_method) == "pending_projected_impact":
            return self._register_pending_impact(
                cam, ev, best, best_entry, best_meta, period, selection_info, geometry_summary, d, age_ms
            )

        self._write_candidate_dict(cam, d)
        dbg_candidate = dict(d)
        self._emit_debug(cam, ev, "emitted_candidate", None, period=period, selection=selection_info, geometry=geometry_summary, candidate=dbg_candidate, age_ms=age_ms)
        print(f"[hit_judge][{cam}] impact candidate bullet={best.bullet_id} person={best.stable_id} score={best.total_score:.3f} offset={d.get('impact_offset_ms','?')}ms dt={best.dt_ms:.1f}ms method={best.match_method} strategy={strategy_used} fallback={d.get('human_fallback_used')}", flush=True)
        return "emitted"

    def run(self) -> None:
        try:
            while True:
                self.poll_inputs()
                self.receive_udp()
                self.process_pending_events()
                self._process_pending_impacts()
                now = time.time()
                if now - self.last_print >= 2.0:
                    hist_info = " ".join([f"{c}:H{len(self.history.get(c, []))}/P{len(self.pending_without_ts.get(c, []))}/T{len(self.trigger_tailers.get(c).map if self.trigger_tailers.get(c) else {})}" for c in self.cam_ids])
                    print(f"[hit_judge][perf] recv={self.recv_count} cand={self.cand_count} wait={self.wait_count} drop={self.drop_count} nonimpact={self.nonimpact_count} no_human={self.no_human_count} no_cam={self.no_cam_count} pending={len(self.pending_events)} pending_impacts={len(self.pending_impacts)} {hist_info}", flush=True)
                    self.recv_count = self.cand_count = self.no_human_count = self.no_cam_count = self.nonimpact_count = self.wait_count = self.drop_count = 0
                    self.last_print = now
                time.sleep(0.002)
        finally:
            self.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UDP hit judge bridge for MVS human regions + event bullet events")
    p.add_argument("--config", default="hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json")
    p.add_argument("--bind-ip", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5003)
    p.add_argument("--cams", default="", help="Comma-separated camera ids. Empty means all enabled routes.")
    p.add_argument("--max-dt-ms", type=float, default=80.0)
    p.add_argument("--bbox-expand-px", type=float, default=35.0)
    p.add_argument("--near-bbox-px", type=float, default=20.0)
    p.add_argument("--min-score-to-log", type=float, default=0.38)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg, cfg_path = _load_json_config(args.config)
    os.chdir(os.path.dirname(cfg_path) or os.getcwd())
    server = HitJudgeServer(cfg, cfg_path, args)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
