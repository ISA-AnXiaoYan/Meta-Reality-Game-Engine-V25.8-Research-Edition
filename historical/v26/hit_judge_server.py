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
   V9.1-R prefers mvs_soft_trigger_event_ts_us, which is the MVS soft-trigger
   instant converted into the event-camera timestamp domain; event_ts_us is the
   observed hardware edge and is only a fallback/diagnostic anchor.
4) event->MVS homographies from hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json.

Output
------
sync_ipc/hit_candidate_{camera}.jsonl and sync_ipc/hit_candidate_all.jsonl.

The server does not make a final game-rule decision.  It emits scored hit
candidates so thresholds can be tuned without touching the event/MVS pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import csv
import json
import math
import os
import select
import socket
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from preview_gpu_shm import PreviewTrajectoryWriter
except Exception:
    PreviewTrajectoryWriter = None

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
        EVENT_POINT,
        REASON_VISUAL_POINT,
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
        EVENT_POINT,
        REASON_VISUAL_POINT,
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


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


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


def _person_mask_polygons_as_points(person: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    if isinstance(polys, dict):
        polys = list(polys.values())
    out: List[List[Tuple[float, float]]] = []
    if not isinstance(polys, list):
        return out
    for poly in polys:
        if not isinstance(poly, list):
            continue
        pts: List[Tuple[float, float]] = []
        for q in poly:
            if isinstance(q, (list, tuple)) and len(q) >= 2:
                try:
                    pts.append((float(q[0]), float(q[1])))
                except Exception:
                    continue
        if len(pts) >= 3:
            out.append(pts)
    return out


def _point_in_any_polygon(pt: Tuple[float, float], polys: List[List[Tuple[float, float]]]) -> bool:
    return any(_point_in_polygon(pt, poly) for poly in polys)


def _segment_intersects_any_polygon(a: Tuple[float, float], b: Tuple[float, float], polys: List[List[Tuple[float, float]]]) -> bool:
    return any(_segment_intersects_polygon(a, b, poly) for poly in polys)


def _point_in_expanded_bbox(pt: Tuple[float, float], bbox: Any, expand_px: float) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x, y = float(pt[0]), float(pt[1])
        e = max(0.0, float(expand_px))
    except Exception:
        return False
    return (x1 - e) <= x <= (x2 + e) and (y1 - e) <= y <= (y2 + e)


def _path_len_px(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return float(total)


def _person_track_mask_hit(
    person: Dict[str, Any],
    track_points: List[Tuple[int, float, float, int]],
    event_pt: Tuple[float, float],
    approach_pt: Tuple[float, float],
    approach_valid: bool,
    *,
    project_enable: bool,
    project_extend_px: float,
    project_max_last_dist_px: float,
) -> Tuple[Optional[str], float, str, Dict[str, Any]]:
    """Evaluate the whole stable visual-point track against a person mask.

    The point stream often shows a long bullet trajectory while the latest
    point-to-previous-point segment misses the current mask.  Recall-first final
    hit needs the whole stable track, not just the last tiny segment.
    """
    method, score, evidence = _person_mask_hit(person, event_pt, approach_pt, approach_valid)
    if method:
        return method, float(score), str(evidence), {
            "track_mask_method": str(method),
            "track_mask_source": "latest_segment",
            "track_mask_hit_point": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
        }

    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    if isinstance(polys, dict):
        polys = list(polys.values())
    if not isinstance(polys, list):
        return None, 0.0, "none", {}

    points: List[Tuple[int, float, float, int]] = []
    for row in track_points or []:
        try:
            idx, x, y, ts_us = row
            points.append((int(idx), float(x), float(y), int(ts_us)))
        except Exception:
            continue
    if len(points) < 2:
        return None, 0.0, "none", {}

    best: Optional[Tuple[str, float, str, Dict[str, Any]]] = None
    for poly in polys:
        if not isinstance(poly, list):
            continue
        pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
        if len(pts) < 3:
            continue

        for idx, x, y, ts_us in reversed(points):
            if _point_in_polygon((x, y), pts):
                return "track_point_inside_mask", 0.74, "strong_track_point", {
                    "track_mask_method": "track_point_inside_mask",
                    "track_mask_source": "stable_track",
                    "track_mask_point_index": int(idx),
                    "track_mask_point_ts_us": int(ts_us),
                    "track_mask_hit_point": [round(float(x), 3), round(float(y), 3)],
                }

        for a, b in zip(reversed(points[:-1]), reversed(points[1:])):
            a_pt = (float(a[1]), float(a[2]))
            b_pt = (float(b[1]), float(b[2]))
            if not _segment_intersects_polygon(a_pt, b_pt, pts):
                continue
            hit_xy = _segment_polygon_first_intersection(a_pt, b_pt, pts)
            if hit_xy is not None:
                hit_point = hit_xy[0]
            else:
                hit_point = b_pt
            detail = {
                "track_mask_method": "track_polyline_cross_mask",
                "track_mask_source": "stable_track",
                "track_mask_segment_start_index": int(a[0]),
                "track_mask_segment_end_index": int(b[0]),
                "track_mask_hit_point": [round(float(hit_point[0]), 3), round(float(hit_point[1]), 3)],
                "track_mask_segment_start": [round(float(a_pt[0]), 3), round(float(a_pt[1]), 3)],
                "track_mask_segment_end": [round(float(b_pt[0]), 3), round(float(b_pt[1]), 3)],
            }
            return "track_polyline_cross_mask", 0.66, "medium_strong_track_polyline", detail

        if not bool(project_enable):
            continue
        last = points[-1]
        prev = None
        for cand in reversed(points[:-1]):
            if math.hypot(float(last[1]) - float(cand[1]), float(last[2]) - float(cand[2])) > 1e-6:
                prev = cand
                break
        if prev is None:
            continue
        last_pt = (float(last[1]), float(last[2]))
        prev_pt = (float(prev[1]), float(prev[2]))
        if _point_in_polygon(last_pt, pts):
            continue
        last_dist = _dist_to_polygon(last_pt, pts)
        if last_dist > float(project_max_last_dist_px):
            continue
        vx = last_pt[0] - prev_pt[0]
        vy = last_pt[1] - prev_pt[1]
        dist = math.hypot(vx, vy)
        if dist <= 1e-6:
            continue
        ux, uy = vx / dist, vy / dist
        projected_end = (last_pt[0] + ux * float(project_extend_px), last_pt[1] + uy * float(project_extend_px))
        got = _segment_polygon_first_intersection(last_pt, projected_end, pts)
        if got is None:
            continue
        hit_point, t = got
        detail = {
            "track_mask_method": "track_projected_inbound_mask",
            "track_mask_source": "stable_track_projection",
            "track_mask_point_index": int(last[0]),
            "track_mask_point_ts_us": int(last[3]),
            "track_mask_hit_point": [round(float(hit_point[0]), 3), round(float(hit_point[1]), 3)],
            "track_mask_projected_end": [round(float(projected_end[0]), 3), round(float(projected_end[1]), 3)],
            "track_mask_last_point_to_mask_dist_px": round(float(last_dist), 3),
            "track_mask_projection_len_px": round(float(t) * float(project_extend_px), 3),
            "track_mask_incoming_dir_xy": [round(float(ux), 6), round(float(uy), 6)],
        }
        candidate = ("track_projected_inbound_mask", 0.58, "projected_track", detail)
        if best is None:
            best = candidate

    if best is not None:
        return best
    return None, 0.0, "none", {}


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


class OverlayBroadcaster:
    """Optional lightweight WebSocket broadcaster for UE5 LiveOverlay clients.

    hit_judge remains usable without the third-party websockets package: when
    websockets is unavailable, JSONL output still works and this broadcaster is
    simply disabled.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, queue_max: int = 4096):
        self.host = str(host or "0.0.0.0")
        self.port = int(port)
        self.queue_max = int(queue_max)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue] = None
        self.stop_event: Optional[asyncio.Event] = None
        self.thread: Optional[threading.Thread] = None
        self.clients = set()
        self.started = False
        self.available = False
        self.drop_count = 0

    def start(self) -> bool:
        if self.started:
            return self.available
        self.started = True
        try:
            import websockets  # noqa: F401
        except Exception as exc:
            print(f"[overlay_ws][WARN] disabled: import websockets failed: {exc}", flush=True)
            return False
        self.available = True
        self.thread = threading.Thread(target=self._thread_main, daemon=True, name="OverlayWebSocketBroadcaster")
        self.thread.start()
        print(f"[overlay_ws] listen ws://{self.host}:{self.port}", flush=True)
        return True

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.queue = asyncio.Queue(maxsize=max(1, self.queue_max))
        self.stop_event = asyncio.Event()
        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as exc:
            print(f"[overlay_ws][ERROR] server stopped: {exc}", flush=True)
        finally:
            try:
                for ws in list(self.clients):
                    self.loop.run_until_complete(ws.close())
            except Exception:
                pass
            self.clients.clear()
            self.available = False
            try:
                self.loop.close()
            except Exception:
                pass

    async def _run_server(self) -> None:
        import websockets
        async with websockets.serve(self._handler, self.host, self.port, max_size=2_000_000):
            sender_task = asyncio.create_task(self._send_loop())
            try:
                assert self.stop_event is not None
                await self.stop_event.wait()
            finally:
                sender_task.cancel()
                try:
                    await sender_task
                except BaseException:
                    pass

    async def _handler(self, websocket) -> None:
        self.clients.add(websocket)
        print(f"[overlay_ws] client connected; clients={len(self.clients)}", flush=True)
        try:
            async for _msg in websocket:
                # UE currently only consumes server-pushed overlay messages.
                pass
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[overlay_ws] client disconnected; clients={len(self.clients)}", flush=True)

    async def _send_loop(self) -> None:
        assert self.queue is not None
        while True:
            msg = await self.queue.get()
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    def broadcast(self, obj: Dict[str, Any]) -> None:
        if not self.available or self.loop is None or self.queue is None:
            return
        try:
            msg = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            print(f"[overlay_ws][WARN] json encode failed: {exc}", flush=True)
            return

        def _put() -> None:
            assert self.queue is not None
            try:
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                        self.drop_count += 1
                    except Exception:
                        pass
                self.queue.put_nowait(msg)
            except Exception:
                self.drop_count += 1

        try:
            self.loop.call_soon_threadsafe(_put)
        except Exception:
            self.drop_count += 1


    def close(self) -> None:
        if self.loop is not None and self.stop_event is not None:
            try:
                self.loop.call_soon_threadsafe(self.stop_event.set)
            except Exception:
                pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)


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
        self._ts_items: List[Tuple[int, int]] = []  # (trigger anchor in event time, sync_index)
        self.sync_source_counts: Dict[str, int] = {}
        self.anchor_source_counts: Dict[str, int] = {}
        self.sync_source_by_index: Dict[int, str] = {}
        self.anchor_source_by_index: Dict[int, str] = {}
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
                self.sync_source_counts.clear()
                self.anchor_source_counts.clear()
                self.sync_source_by_index.clear()
                self.anchor_source_by_index.clear()
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
                self.sync_source_counts.clear()
                self.anchor_source_counts.clear()
                self.sync_source_by_index.clear()
                self.anchor_source_by_index.clear()
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
                    mapped_raw = str(d.get("mapped_mvs_sync_index", "") or "").strip()
                    if mapped_raw:
                        si = int(float(mapped_raw))
                        sync_source = "mapped_mvs_sync_index"
                    else:
                        si = int(float(d.get("sync_index", "")))
                        sync_source = "event_local_sync_index"
                    anchor_source = "event_edge_ts_us"
                    ts = 0
                    mvs_anchor_raw = str(d.get("mvs_soft_trigger_event_ts_us", "") or "").strip()
                    if mvs_anchor_raw:
                        try:
                            ts = int(float(mvs_anchor_raw))
                            anchor_source = "mvs_soft_trigger_event_ts_us"
                        except Exception:
                            ts = 0
                    if ts <= 0:
                        ts = int(float(d.get("event_ts_us", "")))
                        anchor_source = "event_edge_ts_us"
                    # Some trigger CSVs contain helper rows with sync_index=-1.
                    # They are useful for diagnostics but not for MVS-frame anchoring.
                    if si >= 0 and ts > 0:
                        old = self.map.get(si)
                        self.map[si] = ts
                        self.sync_source_counts[sync_source] = int(self.sync_source_counts.get(sync_source, 0) or 0) + 1
                        self.anchor_source_counts[anchor_source] = int(self.anchor_source_counts.get(anchor_source, 0) or 0) + 1
                        self.sync_source_by_index[int(si)] = str(sync_source)
                        self.anchor_source_by_index[int(si)] = str(anchor_source)
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
            "sync_source": str(self.sync_source_by_index.get(int(sync_index)) or ("mapped_mvs_sync_index" if int(self.sync_source_counts.get("mapped_mvs_sync_index", 0) or 0) > 0 else "event_local_sync_index")),
            "anchor_source": str(self.anchor_source_by_index.get(int(sync_index)) or ("mvs_soft_trigger_event_ts_us" if int(self.anchor_source_counts.get("mvs_soft_trigger_event_ts_us", 0) or 0) > 0 else "event_edge_ts_us")),
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


@dataclass
class LiveBulletPointState:
    cam: str
    camera_sn: str
    bullet_id: int
    point_index: int
    point_ts_us: int
    mvs_point: Tuple[float, float]
    raw_point: Tuple[float, float]
    first_point_ts_us: int = 0
    track_point_count: int = 1
    track_path_len_px: float = 0.0
    recent_mvs_points: List[Tuple[int, float, float, int]] = field(default_factory=list)
    receive_wall: float = field(default_factory=time.time)


@dataclass
class PendingVisualPointSegment:
    cam: str
    ev: BulletEvent
    prev_state: LiveBulletPointState
    curr_mvs: Tuple[float, float]
    age_ms: float = 0.0
    created_wall: float = field(default_factory=time.time)
    tries: int = 0


@dataclass
class SemanticPathPoint:
    """One filtered display-path point after event->MVS projection.

    These points are the semantic source of truth for UE5 markers.  Raw
    EVENT_TURN/EVENT_TERMINATE packets are still logged for audit, but formal
    turn/terminal markers are derived from this continuous path.
    """
    cam: str
    camera_sn: str
    bullet_id: int
    seq_id: int
    point_index: int
    ts_us: int
    x: float
    y: float
    raw_x: float
    raw_y: float
    receive_wall: float = field(default_factory=time.time)


@dataclass
class SemanticPathState:
    cam: str
    camera_sn: str
    bullet_id: int
    points: Deque[SemanticPathPoint]
    last_update_wall: float = field(default_factory=time.time)
    last_turn_wall: float = 0.0
    last_turn_ts_us: int = 0
    last_turn_xy: Optional[Tuple[float, float]] = None
    emitted_turn_keys: set = field(default_factory=set)
    terminal_sent_for_index: Optional[int] = None
    terminal_sent_ts_us: int = 0
    terminal_generation: int = 0


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
        self.latest_frame_size_by_cam: Dict[str, Tuple[int, int]] = {}
        root = str(self.sync_cfg.get("root", "./sync_ipc") or "./sync_ipc")
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        control_arg = str(getattr(args, "control_json", "") or "")
        if not control_arg:
            control_arg = os.path.join(self.root, "hit_judge_control.json")
        control_path = Path(os.path.expanduser(control_arg))
        if not control_path.is_absolute():
            control_path = Path(self.root) / control_path
        self.control_path = str(control_path.resolve())
        self.control_mtime = 0.0
        self.control_seq = 0
        self.control_error = ""
        self.hit_detection_enabled = True
        self.hit_candidate_output_enabled = True
        self.strong_gate_enabled = _as_bool(self.hit_cfg.get("strong_gate_enabled", True), True)
        self.strong_reasoning_enabled = _as_bool(self.hit_cfg.get("strong_reasoning_enabled", True), True)
        self.strong_reasoning_shadow_only = _as_bool(self.hit_cfg.get("strong_reasoning_shadow_only", True), True)
        self.final_hit_authority = self._normalize_final_hit_authority(self.hit_cfg.get("final_hit_authority", "strong_gate"))
        if str(getattr(self, "final_hit_authority", "strong_gate")) == "strong_reasoning":
            self.strong_reasoning_shadow_only = False
        self.pseudo_hit_enable = _as_bool(self.hit_cfg.get("pseudo_hit_enable", True), True)
        self.pseudo_hit_dedup_ms = float(self.hit_cfg.get("pseudo_hit_dedup_ms", 120.0) or 120.0)
        self.human_noise_filter_enable = _as_bool(self.hit_cfg.get("human_noise_filter_enable", True), True)
        self.human_noise_filter_mode = self._normalize_human_noise_filter_mode(self.hit_cfg.get("human_noise_filter_mode", "soft"))
        self.human_noise_require_inbound_for_low_quality = _as_bool(self.hit_cfg.get("human_noise_require_inbound_for_low_quality", True), True)
        self.human_noise_min_pre_hit_outside_points = int(self.hit_cfg.get("human_noise_min_pre_hit_outside_points", 3) or 3)
        self.human_noise_min_pre_hit_path_len_px = float(self.hit_cfg.get("human_noise_min_pre_hit_path_len_px", 30.0) or 30.0)
        self.human_noise_short_track_max_points = int(self.hit_cfg.get("human_noise_short_track_max_points", 5) or 5)
        self.human_noise_short_track_max_path_len_px = float(self.hit_cfg.get("human_noise_short_track_max_path_len_px", 35.0) or 35.0)
        self.human_noise_short_track_max_duration_ms = float(self.hit_cfg.get("human_noise_short_track_max_duration_ms", 20.0) or 20.0)
        self.human_noise_soft_threshold = float(self.hit_cfg.get("human_noise_soft_threshold", 0.62) or 0.62)
        self.human_noise_hard_threshold = float(self.hit_cfg.get("human_noise_hard_threshold", 0.88) or 0.88)
        self.human_noise_track_start_bbox_filter_enable = _as_bool(self.hit_cfg.get("human_noise_track_start_bbox_filter_enable", True), True)
        self.human_noise_track_start_bbox_expand_px = float(self.hit_cfg.get("human_noise_track_start_bbox_expand_px", 24.0) or 24.0)
        self.human_noise_track_start_bbox_policy = self._normalize_track_start_bbox_policy(
            self.hit_cfg.get("human_noise_track_start_bbox_policy", "soft")
        )
        self._poll_control_json(force=True)
        self.hit_template = str(self.hit_cfg.get("hit_jsonl_template", "{root}/hit_candidate_{camera}.jsonl") or "{root}/hit_candidate_{camera}.jsonl")
        self.all_path = os.path.abspath(str(self.hit_cfg.get("hit_all_jsonl", "{root}/hit_candidate_all.jsonl") or "{root}/hit_candidate_all.jsonl").format(root=self.root))
        self.out_files: Dict[str, Any] = {}
        for c in self.cam_ids:
            p = self.hit_template.format(root=self.root, camera=c, cam=c, id=c)
            if not os.path.isabs(p): p = os.path.abspath(p)
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            self.out_files[c] = open(p, "a", encoding="utf-8", buffering=1)
        self.all_fp = open(self.all_path, "a", encoding="utf-8", buffering=1)
        self.hit_shadow_enable = _as_bool(self.hit_cfg.get("hit_shadow_enable", True), True)
        self.hit_shadow_template = str(
            self.hit_cfg.get(
                "hit_shadow_jsonl_template",
                "{root}/hit_candidate_shadow_{chain}_{camera}.jsonl",
            )
            or "{root}/hit_candidate_shadow_{chain}_{camera}.jsonl"
        )
        self.hit_shadow_all_template = str(
            self.hit_cfg.get(
                "hit_shadow_all_jsonl_template",
                "{root}/hit_candidate_shadow_{chain}_all.jsonl",
            )
            or "{root}/hit_candidate_shadow_{chain}_all.jsonl"
        )
        self.hit_shadow_files: Dict[str, Dict[str, Any]] = {}
        self.hit_shadow_all_files: Dict[str, Any] = {}
        if self.hit_shadow_enable:
            for chain in ("strong_gate", "strong_reasoning"):
                self.hit_shadow_files[chain] = {}
                for c in self.cam_ids:
                    sp = self.hit_shadow_template.format(root=self.root, camera=c, cam=c, id=c, chain=chain)
                    if not os.path.isabs(sp):
                        sp = os.path.abspath(sp)
                    Path(sp).parent.mkdir(parents=True, exist_ok=True)
                    self.hit_shadow_files[chain][c] = open(sp, "a", encoding="utf-8", buffering=1)
                sap = self.hit_shadow_all_template.format(root=self.root, chain=chain)
                if not os.path.isabs(sap):
                    sap = os.path.abspath(sap)
                Path(sap).parent.mkdir(parents=True, exist_ok=True)
                self.hit_shadow_all_files[chain] = open(sap, "a", encoding="utf-8", buffering=1)
        self.pseudo_hit_template = str(self.hit_cfg.get("pseudo_hit_jsonl_template", "{root}/pseudo_hit_{camera}.jsonl") or "{root}/pseudo_hit_{camera}.jsonl")
        self.pseudo_hit_all_path = os.path.abspath(str(self.hit_cfg.get("pseudo_hit_all_jsonl", "{root}/pseudo_hit_all.jsonl") or "{root}/pseudo_hit_all.jsonl").format(root=self.root))
        self.pseudo_hit_files: Dict[str, Any] = {}
        self.pseudo_hit_all_fp = None
        if bool(getattr(self, "pseudo_hit_enable", True)):
            for c in self.cam_ids:
                pp = self.pseudo_hit_template.format(root=self.root, camera=c, cam=c, id=c)
                if not os.path.isabs(pp): pp = os.path.abspath(pp)
                Path(pp).parent.mkdir(parents=True, exist_ok=True)
                self.pseudo_hit_files[c] = open(pp, "a", encoding="utf-8", buffering=1)
            Path(self.pseudo_hit_all_path).parent.mkdir(parents=True, exist_ok=True)
            self.pseudo_hit_all_fp = open(self.pseudo_hit_all_path, "a", encoding="utf-8", buffering=1)
        self.pseudo_hit_total = 0
        self.pseudo_hit_by_camera: Dict[str, int] = defaultdict(int)
        self.pseudo_hit_reason_counts: Dict[str, int] = defaultdict(int)
        self.pseudo_hit_duplicate_suppressed = 0
        self.pseudo_hit_promoted_to_final_count = 0
        self.pseudo_hit_human_noise_suspect = 0
        self.pseudo_hit_human_noise_suppressed = 0
        self.pseudo_hit_human_noise_reason_counts: Dict[str, int] = defaultdict(int)
        self.strong_gate_output_suppressed = 0
        self.strong_reasoning_output_suppressed = 0
        self.final_hit_cross_camera_dedup_enable = _as_bool(self.hit_cfg.get("final_hit_cross_camera_dedup_enable", True), True)
        self.final_hit_cross_camera_dedup_ms = float(self.hit_cfg.get("final_hit_cross_camera_dedup_ms", 700.0) or 700.0)
        self.final_hit_duplicate_suppressed = 0
        self.final_hit_same_track_episode_dedup_enable = _as_bool(self.hit_cfg.get("final_hit_same_track_episode_dedup_enable", True), True)
        self.final_hit_same_track_episode_dedup_ms = float(self.hit_cfg.get("final_hit_same_track_episode_dedup_ms", 1500.0) or 1500.0)
        self.same_track_episode_duplicate_suppressed = 0
        self.final_hit_seen_keys: Deque[Tuple[Any, ...]] = deque(maxlen=4096)
        self.final_hit_seen_set = set()
        self.final_hit_episode_seen: Dict[Tuple[Any, ...], float] = {}
        self.pseudo_hit_seen_keys: Deque[Tuple[Any, ...]] = deque(maxlen=8192)
        self.pseudo_hit_seen_set = set()
        self.latest_pseudo_hit: Dict[str, Any] = {}

        # Full visual bullet stream for UE overlay/Niagara.  Unlike hit_candidate,
        # this is emitted before human matching and therefore does not depend on
        # any person box/mask being present.
        self.visual_stream_enable = _as_bool(self.hit_cfg.get("visual_bullet_stream_enable", True), True)
        self.visual_template = str(self.hit_cfg.get("visual_bullet_jsonl_template", "{root}/overlay_bullet_event_{camera}.jsonl") or "{root}/overlay_bullet_event_{camera}.jsonl")
        self.visual_all_path = os.path.abspath(str(self.hit_cfg.get("visual_bullet_all_jsonl", "{root}/overlay_bullet_event_all.jsonl") or "{root}/overlay_bullet_event_all.jsonl").format(root=self.root))
        self.visual_files: Dict[str, Any] = {}
        self.visual_all_fp = None
        if self.visual_stream_enable:
            for c in self.cam_ids:
                vp = self.visual_template.format(root=self.root, camera=c, cam=c, id=c)
                if not os.path.isabs(vp): vp = os.path.abspath(vp)
                Path(vp).parent.mkdir(parents=True, exist_ok=True)
                self.visual_files[c] = open(vp, "a", encoding="utf-8", buffering=1)
            Path(self.visual_all_path).parent.mkdir(parents=True, exist_ok=True)
            self.visual_all_fp = open(self.visual_all_path, "a", encoding="utf-8", buffering=1)
        self.visual_seen_keys: Deque[Tuple[str, int, int]] = deque(maxlen=8192)
        self.visual_seen_set = set()
        self.visual_count = 0

        # Real-time bullet point stream for UE5 Niagara Segment rendering.
        # Each message is one processed MVS pixel point. UE should connect previous
        # point -> current point for the same (pair_id, bullet_id).
        self.visual_point_stream_enable = _as_bool(self.hit_cfg.get("visual_bullet_point_stream_enable", True), True)
        self.visual_point_template = str(self.hit_cfg.get("visual_bullet_point_jsonl_template", "{root}/overlay_bullet_point_{camera}.jsonl") or "{root}/overlay_bullet_point_{camera}.jsonl")
        self.visual_point_all_path = os.path.abspath(str(self.hit_cfg.get("visual_bullet_point_all_jsonl", "{root}/overlay_bullet_point_all.jsonl") or "{root}/overlay_bullet_point_all.jsonl").format(root=self.root))
        self.visual_point_files: Dict[str, Any] = {}
        self.visual_point_all_fp = None
        if self.visual_point_stream_enable:
            for c in self.cam_ids:
                pp = self.visual_point_template.format(root=self.root, camera=c, cam=c, id=c)
                if not os.path.isabs(pp): pp = os.path.abspath(pp)
                Path(pp).parent.mkdir(parents=True, exist_ok=True)
                self.visual_point_files[c] = open(pp, "a", encoding="utf-8", buffering=1)
            Path(self.visual_point_all_path).parent.mkdir(parents=True, exist_ok=True)
            self.visual_point_all_fp = open(self.visual_point_all_path, "a", encoding="utf-8", buffering=1)
        self.visual_point_count = 0

        # Binary shared-memory stream for the independent PyOpenGL/CUDA preview renderer.
        # This avoids high-rate JSON tailing for bullet trajectory redraws.
        self.preview_trajectory_enable = _as_bool(
            self.hit_cfg.get("preview_trajectory_shm_enable", os.environ.get("PREVIEW_TRAJECTORY_SHM_ENABLE", "1")), True
        )
        self.preview_trajectory_writer = None
        self.cam_index_by_id = {str(c): int(i) for i, c in enumerate(self.cam_ids)}
        if self.preview_trajectory_enable and PreviewTrajectoryWriter is not None:
            try:
                traj_name = str(self.hit_cfg.get("preview_trajectory_shm_name", os.environ.get("PREVIEW_TRAJECTORY_SHM", "preview_trajectory")) or "preview_trajectory")
                traj_max = int(self.hit_cfg.get("preview_trajectory_max_segments", os.environ.get("PREVIEW_TRAJECTORY_MAX_SEGMENTS", "65536")) or 65536)
                self.preview_trajectory_writer = PreviewTrajectoryWriter(traj_name, max_segments=traj_max, create=True, flush=False)
                print(f"[hit_judge] preview trajectory shm enabled: {traj_name} max_segments={traj_max}", flush=True)
            except Exception as exc:
                print(f"[hit_judge][WARN] preview trajectory shm disabled: {type(exc).__name__}: {exc}", flush=True)
                self.preview_trajectory_writer = None
        elif self.preview_trajectory_enable:
            print("[hit_judge][WARN] preview_gpu_shm.PreviewTrajectoryWriter unavailable; trajectory shm disabled", flush=True)

        # 4ms-level point-segment hit judging.  This uses the same real-time point
        # stream as UE5 rendering: previous processed MVS point -> current processed
        # MVS point.  It does not wait for 40ms track_probe/terminate packets.
        self.visual_point_hit_judge_enable = _as_bool(self.hit_cfg.get("visual_point_hit_judge_enable", True), True)
        self.visual_point_hit_min_segment_len_px = float(self.hit_cfg.get("visual_point_hit_min_segment_len_px", 1.0) or 1.0)
        self.visual_point_hit_min_track_points = int(self.hit_cfg.get("visual_point_hit_min_track_points", 2) or 2)
        self.visual_point_hit_min_track_path_len_px = float(self.hit_cfg.get("visual_point_hit_min_track_path_len_px", 0.0) or 0.0)
        self.visual_point_hit_min_score = float(self.hit_cfg.get("visual_point_hit_min_score", self.hit_cfg.get("min_score_to_log", 0.38)) or self.hit_cfg.get("min_score_to_log", 0.38) or 0.38)
        self.visual_point_hit_event_score = float(self.hit_cfg.get("visual_point_hit_event_score", 0.12) or 0.12)
        self.visual_point_hit_allow_bbox = _as_bool(self.hit_cfg.get("visual_point_hit_allow_bbox", self.hit_cfg.get("allow_bbox_final_hit", False)), bool(self.hit_cfg.get("allow_bbox_final_hit", False)))
        self.visual_point_hit_allow_human_fallback = _as_bool(self.hit_cfg.get("visual_point_hit_allow_human_fallback", True), True)
        self.visual_point_hit_max_person_frame_delta_sync = int(self.hit_cfg.get("visual_point_hit_max_person_frame_delta_sync", self.hit_cfg.get("human_search_radius_sync", 2)) or self.hit_cfg.get("human_search_radius_sync", 2) or 2)
        self.visual_point_hit_emit_once_per_bullet = _as_bool(self.hit_cfg.get("visual_point_hit_emit_once_per_bullet", True), True)
        self.visual_point_hit_body_motion_reject_enable = _as_bool(self.hit_cfg.get("visual_point_hit_body_motion_reject_enable", False), False)
        self.visual_point_hit_verbose_debug = _as_bool(self.hit_cfg.get("visual_point_hit_verbose_debug", False), False)
        self.visual_point_hit_late_human_retry_ms = max(
            float(self.hit_cfg.get("visual_point_hit_late_human_retry_ms", 80.0) or 80.0),
            float(self.hit_cfg.get("strong_reasoning_late_human_retry_ms", 500.0) or 500.0),
        )
        self.visual_point_track_cache_max_points = int(self.hit_cfg.get("visual_point_track_cache_max_points", 192) or 192)
        self.strong_reasoning_track_mask_enable = _as_bool(self.hit_cfg.get("strong_reasoning_track_mask_enable", True), True)
        self.strong_reasoning_track_project_enable = _as_bool(self.hit_cfg.get("strong_reasoning_track_project_enable", True), True)
        self.strong_reasoning_track_project_extend_px = float(self.hit_cfg.get("strong_reasoning_track_project_extend_px", 100.0) or 100.0)
        self.strong_reasoning_track_project_max_last_dist_px = float(self.hit_cfg.get("strong_reasoning_track_project_max_last_dist_px", 40.0) or 40.0)
        self.projected_mask_final_mode = self._normalize_projected_mask_final_mode(self.hit_cfg.get("projected_mask_final_mode", "pending_only"))
        self.projected_mask_require_terminal_or_turn = _as_bool(self.hit_cfg.get("projected_mask_require_terminal_or_turn", True), True)
        self.projected_pending_count = 0
        self.projected_promoted_count = 0
        self.projected_suppressed_final_count = 0
        self.visual_point_pending_segments: Deque[PendingVisualPointSegment] = deque(
            maxlen=int(self.hit_cfg.get("visual_point_hit_late_human_queue", 1024) or 1024)
        )
        self.visual_point_pending_keys = set()
        self.visual_point_result_counts: Dict[str, int] = defaultdict(int)
        self.visual_point_stage_counts: Dict[str, int] = defaultdict(int)
        self.visual_point_stage_by_camera: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.hit_reject_counts: Dict[str, int] = defaultdict(int)
        self.latest_visual_point_reject: Dict[str, Any] = {}
        self.latest_visual_point_candidate: Dict[str, Any] = {}
        self.latest_visual_point_stage: Dict[str, Any] = {}
        self.visual_point_last_by_key: Dict[Tuple[str, str, int], LiveBulletPointState] = {}
        self.visual_point_hit_keys: Deque[Tuple[str, int, int]] = deque(maxlen=4096)
        self.visual_point_hit_set = set()

        # Semantic path resolver for UE5 and the future hit pipeline.  Formal
        # turn/terminal markers are derived from the same filtered display-path
        # points used for overlay_bullet_point, not from raw line-filter
        # EVENT_TURN/EVENT_TERMINATE packets.
        self.semantic_path_enable = _as_bool(self.hit_cfg.get("semantic_path_enable", True), True)
        self.semantic_emit_track_state = _as_bool(self.hit_cfg.get("semantic_emit_track_state", True), True)
        self.semantic_path_max_points = int(self.hit_cfg.get("semantic_path_max_points", 96) or 96)
        self.semantic_terminal_delay_ms = float(self.hit_cfg.get("semantic_terminal_delay_ms", 140.0) or 140.0)
        self.semantic_terminal_min_points = int(self.hit_cfg.get("semantic_terminal_min_points", 3) or 3)
        self.semantic_terminal_min_path_len_px = float(self.hit_cfg.get("semantic_terminal_min_path_len_px", 35.0) or 35.0)
        self.semantic_terminal_min_duration_ms = float(self.hit_cfg.get("semantic_terminal_min_duration_ms", 8.0) or 8.0)
        self.semantic_terminal_cleanup_ms = float(self.hit_cfg.get("semantic_terminal_cleanup_ms", 3000.0) or 3000.0)
        self.semantic_turn_enable = _as_bool(self.hit_cfg.get("semantic_turn_enable", True), True)
        self.semantic_turn_angle_deg = float(self.hit_cfg.get("semantic_turn_angle_deg", 40.0) or 40.0)
        self.semantic_turn_min_leg_px = float(self.hit_cfg.get("semantic_turn_min_leg_px", 35.0) or 35.0)
        self.semantic_turn_min_spacing_px = float(self.hit_cfg.get("semantic_turn_min_spacing_px", 45.0) or 45.0)
        self.semantic_turn_min_spacing_ms = float(self.hit_cfg.get("semantic_turn_min_spacing_ms", 55.0) or 55.0)
        self.semantic_turn_lookback_points = int(self.hit_cfg.get("semantic_turn_lookback_points", 16) or 16)
        self.semantic_turn_max_candidates_per_point = int(self.hit_cfg.get("semantic_turn_max_candidates_per_point", 8) or 8)
        self.semantic_paths: Dict[Tuple[str, str, int], SemanticPathState] = {}
        # Keep raw turn/terminate packets in JSONL by default, but do not let them
        # drive UE5 effects when semantic_path_enable is on.  This prevents raw
        # terminate/track_probe packets from being confused with confirmed terminal.
        default_raw_ws = False if self.semantic_path_enable else True
        self.semantic_raw_event_ws_enable = _as_bool(self.hit_cfg.get("semantic_raw_event_ws_enable", default_raw_ws), default_raw_ws)
        self.semantic_raw_marker_enable = _as_bool(self.hit_cfg.get("semantic_raw_marker_enable", default_raw_ws), default_raw_ws)

        # Optional WebSocket push channel for UE5.  JSONL remains the ground-truth
        # test output even when no UE client is connected.
        overlay_ws_enable = _as_bool(self.hit_cfg.get("overlay_ws_enable", False), False) or bool(getattr(args, "overlay_ws_enable", False))
        overlay_ws_host = str(getattr(args, "overlay_ws_host", None) or self.hit_cfg.get("overlay_ws_host", "0.0.0.0") or "0.0.0.0")
        overlay_ws_port = int(getattr(args, "overlay_ws_port", None) or self.hit_cfg.get("overlay_ws_port", 8765) or 8765)
        self.overlay_ws = OverlayBroadcaster(overlay_ws_host, overlay_ws_port) if overlay_ws_enable else None
        if self.overlay_ws is not None:
            self.overlay_ws.start()

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
        self.write_reject_debug_jsonl = _as_bool(self.hit_cfg.get("write_reject_debug_jsonl", True), True)
        self.reject_debug_template = str(
            self.hit_cfg.get("hit_reject_debug_jsonl_template", "{root}/hit_reject_debug_{camera}.jsonl")
            or "{root}/hit_reject_debug_{camera}.jsonl"
        )
        self.reject_debug_all_path = os.path.abspath(
            str(
                self.hit_cfg.get("hit_reject_debug_all_jsonl", "{root}/hit_reject_debug_all.jsonl")
                or "{root}/hit_reject_debug_all.jsonl"
            ).format(root=self.root)
        )
        self.reject_debug_files: Dict[str, Any] = {}
        if self.write_reject_debug_jsonl:
            for c in self.cam_ids:
                rp = self.reject_debug_template.format(root=self.root, camera=c, cam=c, id=c)
                if not os.path.isabs(rp): rp = os.path.abspath(rp)
                Path(rp).parent.mkdir(parents=True, exist_ok=True)
                self.reject_debug_files[c] = open(rp, "a", encoding="utf-8", buffering=1)
            Path(self.reject_debug_all_path).parent.mkdir(parents=True, exist_ok=True)
            self.reject_debug_all_fp = open(self.reject_debug_all_path, "a", encoding="utf-8", buffering=1)
        else:
            self.reject_debug_all_fp = None
        self.status_path = os.path.abspath(
            str(self.hit_cfg.get("hit_judge_status_json", "{root}/hit_judge_status.json") or "{root}/hit_judge_status.json").format(root=self.root)
        )
        self.status_interval_s = max(0.2, float(self.hit_cfg.get("hit_judge_status_interval_ms", 1000.0) or 1000.0) / 1000.0)
        self.last_status_write = 0.0
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
        self.recv_total = 0
        self.cand_total = 0
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

        # Body-motion false-positive guard.  This replaces the old expensive
        # event-side blocking mask wait: event cameras stay real-time, while the
        # low-rate hit judge rejects bullet-like tracks that originate inside the
        # current human mask.  A real hit must have a valid external approach.
        self.body_motion_reject_enable = _as_bool(self.hit_cfg.get("body_motion_reject_enable", True), True)
        self.body_motion_reject_if_approach_inside = _as_bool(self.hit_cfg.get("body_motion_reject_if_approach_inside", True), True)
        self.body_motion_reject_if_both_inside = _as_bool(self.hit_cfg.get("body_motion_reject_if_both_inside", True), True)
        self.body_motion_reject_event_inside_without_external_approach = _as_bool(
            self.hit_cfg.get("body_motion_reject_event_inside_without_external_approach", True), True
        )
        self.body_motion_min_external_approach_len_px = float(self.hit_cfg.get("body_motion_min_external_approach_len_px", self.min_hit_approach_len_px) or self.min_hit_approach_len_px)
        self.body_motion_use_bbox_fallback = _as_bool(self.hit_cfg.get("body_motion_use_bbox_fallback", False), False)

        # 最终 HIT 只认人体 mask，bbox 只作为 debug 参考
        self.final_hit_mask_only = _as_bool(self.hit_cfg.get("final_hit_mask_only", True), True)
        self.allow_bbox_final_hit = _as_bool(self.hit_cfg.get("allow_bbox_final_hit", False), False)
        self.require_trajectory_for_hit = _as_bool(self.hit_cfg.get("require_trajectory_for_hit", True), True)
        self.min_hit_trajectory_len_px = float(self.hit_cfg.get("min_hit_trajectory_len_px", 35.0) or 35.0)
        self.hit_judge_final_mode = str(self.hit_cfg.get("hit_judge_final_mode", "mask_trigger_person_sync") or "mask_trigger_person_sync").strip().lower()
        self.trajectory_only_min_track_points = int(self.hit_cfg.get("trajectory_only_min_track_points", self.min_track_point_count) or self.min_track_point_count)
        self.trajectory_only_min_track_path_len_px = float(self.hit_cfg.get("trajectory_only_min_track_path_len_px", self.min_track_path_len_px) or self.min_track_path_len_px)
        self.trajectory_only_min_duration_ms = float(self.hit_cfg.get("trajectory_only_min_duration_ms", self.min_track_duration_ms) or self.min_track_duration_ms)
        self.trajectory_only_min_straightness = float(self.hit_cfg.get("trajectory_only_min_straightness", self.min_track_straightness) or self.min_track_straightness)
        self.trajectory_only_max_line_error_px = float(self.hit_cfg.get("trajectory_only_max_line_error_px", self.max_track_line_error_px) or self.max_track_line_error_px)
        self.trajectory_only_min_speed_px_per_ms = float(self.hit_cfg.get("trajectory_only_min_speed_px_per_ms", self.min_hit_avg_speed_px_per_ms) or self.min_hit_avg_speed_px_per_ms)
        self.trajectory_only_min_approach_len_px = float(self.hit_cfg.get("trajectory_only_min_approach_len_px", self.min_hit_approach_len_px) or self.min_hit_approach_len_px)
        self.trajectory_only_min_point_segment_len_px = float(self.hit_cfg.get("trajectory_only_min_point_segment_len_px", self.visual_point_hit_min_segment_len_px) or self.visual_point_hit_min_segment_len_px)
        self.trajectory_only_allow_unknown_line_error = _as_bool(self.hit_cfg.get("trajectory_only_allow_unknown_line_error", True), True)
        self.trajectory_only_min_score = float(self.hit_cfg.get("trajectory_only_min_score", self.visual_point_hit_min_score) or self.visual_point_hit_min_score)
        self.evidence_gated_require_trigger = _as_bool(self.hit_cfg.get("evidence_gated_require_trigger", True), True)
        self.evidence_gated_require_person_sync = _as_bool(self.hit_cfg.get("evidence_gated_require_person_sync", True), True)
        self.evidence_gated_require_mask = _as_bool(self.hit_cfg.get("evidence_gated_require_mask", True), True)
        self.evidence_gated_require_mask_polygon = _as_bool(self.hit_cfg.get("evidence_gated_require_mask_polygon", True), True)
        self.evidence_gated_max_person_frame_delta_sync = int(
            self.hit_cfg.get("evidence_gated_max_person_frame_delta_sync", self.visual_point_hit_max_person_frame_delta_sync)
            or self.visual_point_hit_max_person_frame_delta_sync
        )
        self.evidence_gated_mask_max_age_ms = float(self.hit_cfg.get("evidence_gated_mask_max_age_ms", 220.0) or 220.0)
        self.evidence_gated_mask_max_sync_delta = int(self.hit_cfg.get("evidence_gated_mask_max_sync_delta", 2) or 2)
        self.evidence_gated_mask_cache_ttl_ms = float(self.hit_cfg.get("evidence_gated_mask_cache_ttl_ms", 25.0) or 25.0)
        self.evidence_gated_mask_status_json = os.path.abspath(
            str(self.hit_cfg.get("evidence_gated_mask_service_status_json", "{root}/event_mask_evidence_status.json") or "{root}/event_mask_evidence_status.json").format(root=self.root)
        )
        self.evidence_gated_mask_json_template = str(
            self.hit_cfg.get("evidence_gated_mask_evidence_json_template", "{root}/event_mask_evidence_{camera}.json")
            or "{root}/event_mask_evidence_{camera}.json"
        )
        self._mask_evidence_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
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

    def _broadcast_overlay(self, obj: Dict[str, Any]) -> None:
        ws = getattr(self, "overlay_ws", None)
        if ws is not None:
            ws.broadcast(obj)

    def _display_size_for_cam(self, cam: str) -> Tuple[int, int]:
        if cam in self.latest_frame_size_by_cam:
            return self.latest_frame_size_by_cam[cam]
        # Defaults match the MVS preview size used by the current UE overlay path.
        w = int(self.hit_cfg.get("visual_bullet_display_width", self.hit_cfg.get("display_width", 1280)) or 1280)
        h = int(self.hit_cfg.get("visual_bullet_display_height", self.hit_cfg.get("display_height", 720)) or 720)
        return max(1, w), max(1, h)

    def _emit_overlay_frame_state(self, cam: str, entry: "PersonEntry") -> None:
        if not bool(getattr(self, "visual_stream_enable", True)):
            return
        w = int(entry.frame_w or 0)
        h = int(entry.frame_h or 0)
        if w > 0 and h > 0:
            self.latest_frame_size_by_cam[str(cam)] = (w, h)
        else:
            w, h = self._display_size_for_cam(str(cam))
        msg = {
            "type": "overlay_frame_state",
            "schema_version": 1,
            "sync_run_id": str(self.hit_cfg.get("sync_run_id", "real_run_001") or "real_run_001"),
            "pair_id": str(cam),
            "trigger_index": int(entry.sync_index),
            "frame_event_ts_us": int(entry.event_ts_us or 0),
            "display_width": int(w),
            "display_height": int(h),
            "source": "hit_judge_human_result_stream",
        }
        self._broadcast_overlay(msg)

    def _mark_visual_seen(self, key: Tuple[str, int, int]) -> bool:
        if key in self.visual_seen_set:
            return False
        if self.visual_seen_keys.maxlen is not None and len(self.visual_seen_keys) >= self.visual_seen_keys.maxlen:
            try:
                old = self.visual_seen_keys.popleft()
                self.visual_seen_set.discard(old)
            except Exception:
                pass
        self.visual_seen_keys.append(key)
        self.visual_seen_set.add(key)
        return True

    def _project_track_points_mvs(
        self,
        H: np.ndarray,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
    ) -> List[List[float]]:
        """Project BulletEvent.track_points from event pixels into MVS pixels.

        Backward compatible: old senders do not carry track_points, so the visual
        stream falls back to approach->event two-point segment.
        """
        out: List[List[float]] = []
        seen = set()
        raw_points = getattr(ev, "track_points", None) or []
        if isinstance(raw_points, list):
            for item in raw_points:
                try:
                    if isinstance(item, dict):
                        ts = int(item.get("timestamp_us", item.get("ts", ev.timestamp_us)) or ev.timestamp_us)
                        x = float(item.get("x", item.get("event_x", 0.0)) or 0.0)
                        y = float(item.get("y", item.get("event_y", 0.0)) or 0.0)
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        ts = int(float(item[0]))
                        x = float(item[1])
                        y = float(item[2])
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        ts = int(ev.timestamp_us)
                        x = float(item[0])
                        y = float(item[1])
                    else:
                        continue
                    mx, my = _apply_H(H, x, y)
                    key = (ts, round(float(mx), 3), round(float(my), 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append([int(ts), round(float(mx), 3), round(float(my), 3)])
                except Exception:
                    continue
        if len(out) < 2:
            ts0 = int(ev.timestamp_us) - max(1000, int(round(float(getattr(ev, "track_duration_ms", 0.0) or 0.0) * 1000.0)))
            out = [
                [int(ts0), round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],
                [int(ev.timestamp_us), round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            ]
        return out

    def _write_visual_bullet_dict(self, cam: str, d: Dict[str, Any]) -> None:
        if not bool(getattr(self, "visual_stream_enable", True)):
            return
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        fp = self.visual_files.get(cam)
        if fp is not None:
            fp.write(line)
        if getattr(self, "visual_all_fp", None) is not None:
            self.visual_all_fp.write(line)
        self.visual_count += 1

    def _is_visual_point_event(self, ev: BulletEvent) -> bool:
        try:
            return str(getattr(ev, "event_type", "") or "") == EVENT_POINT or str(getattr(ev, "terminate_reason", "") or "") == REASON_VISUAL_POINT
        except Exception:
            return False

    def _write_visual_bullet_point_dict(self, cam: str, d: Dict[str, Any]) -> None:
        if not bool(getattr(self, "visual_point_stream_enable", True)):
            return
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        fp = self.visual_point_files.get(cam)
        if fp is not None:
            fp.write(line)
        if getattr(self, "visual_point_all_fp", None) is not None:
            self.visual_point_all_fp.write(line)
        self.visual_point_count += 1
        if getattr(self, "preview_trajectory_writer", None) is not None:
            try:
                self.preview_trajectory_writer.push_from_overlay_point(
                    str(cam), d, cam_index=int(self.cam_index_by_id.get(str(cam), 0)),
                    life_ms=float(
                        self.hit_cfg.get(
                            "preview_trajectory_life_ms",
                            os.environ.get("PREVIEW_TRAJECTORY_LIFE_MS", "2500.0"),
                        ) or 2500.0
                    ),
                )
            except Exception as exc:
                if int(getattr(self, "_preview_traj_warn_count", 0) or 0) < 10:
                    print(f"[hit_judge][WARN] trajectory shm write failed: {type(exc).__name__}: {exc}", flush=True)
                self._preview_traj_warn_count = int(getattr(self, "_preview_traj_warn_count", 0) or 0) + 1

    def _emit_visual_bullet_point(
        self,
        cam: str,
        ev: BulletEvent,
        H: Optional[np.ndarray] = None,
        age_ms: float = 0.0,
        curr_mvs: Optional[Tuple[float, float]] = None,
        prev_state: Optional[LiveBulletPointState] = None,
        receive_wall: Optional[float] = None,
    ) -> None:
        """Push one processed MVS bullet point to UE5/logs.

        One JSON message equals one trajectory sample point.  When prev_state is
        available, the message also includes the real point interval and MVS
        segment length, so UE/debug scripts can verify whether the stream is
        close to 4ms per point.
        """
        if not bool(getattr(self, "visual_point_stream_enable", True)):
            return
        display_w, display_h = self._display_size_for_cam(str(cam))
        if curr_mvs is None:
            if H is None:
                return
            x_mvs, y_mvs = _apply_H(H, float(ev.event_x), float(ev.event_y))
        else:
            x_mvs, y_mvs = curr_mvs
        point_index = int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1)
        status = str(getattr(ev, "point_status", "tracking") or "tracking")

        prev_x_mvs = x_mvs
        prev_y_mvs = y_mvs
        prev_valid = False
        prev_point_index = -1
        prev_point_ts_us = 0
        point_delta_ms: Optional[float] = None
        segment_len_mvs_px: Optional[float] = None
        receive_delta_ms: Optional[float] = None
        if prev_state is not None:
            prev_x_mvs, prev_y_mvs = prev_state.mvs_point
            prev_valid = True
            prev_point_index = int(prev_state.point_index)
            prev_point_ts_us = int(prev_state.point_ts_us)
            point_delta_ms = (int(ev.timestamp_us) - int(prev_state.point_ts_us)) / 1000.0
            segment_len_mvs_px = math.hypot(float(x_mvs) - float(prev_x_mvs), float(y_mvs) - float(prev_y_mvs))
            if receive_wall is not None:
                receive_delta_ms = (float(receive_wall) - float(prev_state.receive_wall)) * 1000.0

        msg = {
            "type": "overlay_bullet_point",
            "schema_version": 2,
            "sync_run_id": str(self.hit_cfg.get("sync_run_id", "real_run_001") or "real_run_001"),
            "pair_id": str(cam),
            "camera_sn": str(ev.camera_sn),
            "camera_alias": str(ev.camera_alias or cam),
            "bullet_id": int(ev.bullet_id),
            "bullet_seq_id": int(ev.seq_id),
            "point_index": int(point_index),
            "point_ts_us": int(ev.timestamp_us),
            "x": round(float(x_mvs), 3),
            "y": round(float(y_mvs), 3),
            "display_width": int(display_w),
            "display_height": int(display_h),
            "confidence": float(getattr(ev, "confidence", 1.0) or 1.0),
            "status": status,
            "state": "track",
            "semantic_state": "track",
            "prev_x": round(float(prev_x_mvs), 3),
            "prev_y": round(float(prev_y_mvs), 3),
            "prev_valid": bool(prev_valid),
            "prev_point_index": int(prev_point_index),
            "prev_point_ts_us": int(prev_point_ts_us),
            "point_delta_ms": None if point_delta_ms is None else round(float(point_delta_ms), 3),
            "receive_delta_ms": None if receive_delta_ms is None else round(float(receive_delta_ms), 3),
            "segment_len_mvs_px": None if segment_len_mvs_px is None else round(float(segment_len_mvs_px), 3),
            "event_x_raw": round(float(ev.event_x), 3),
            "event_y_raw": round(float(ev.event_y), 3),
            "source": "visual_bullet_point_stream",
            "reason": "realtime_bullet_point",
            "judge_age_ms": round(float(age_ms), 3),
            "wall_time": time.time(),
        }
        self._write_visual_bullet_point_dict(str(cam), msg)
        self._broadcast_overlay(msg)

    def _semantic_key(self, cam: str, camera_sn: str, bullet_id: int) -> Tuple[str, str, int]:
        return (str(cam), str(camera_sn), int(bullet_id))

    def _semantic_path_metrics(self, pts: List[SemanticPathPoint]) -> Dict[str, float]:
        if not pts:
            return {
                "semantic_point_count": 0,
                "semantic_path_len_px": 0.0,
                "semantic_duration_ms": 0.0,
            }
        path_len = 0.0
        for a, b in zip(pts, pts[1:]):
            path_len += math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))
        duration_ms = max(0.0, (int(pts[-1].ts_us) - int(pts[0].ts_us)) / 1000.0)
        disp = math.hypot(float(pts[-1].x) - float(pts[0].x), float(pts[-1].y) - float(pts[0].y)) if len(pts) >= 2 else 0.0
        straightness = disp / max(path_len, 1e-6) if path_len > 0.0 else 1.0
        return {
            "semantic_point_count": float(len(pts)),
            "semantic_path_len_px": float(path_len),
            "semantic_duration_ms": float(duration_ms),
            "semantic_displacement_px": float(disp),
            "semantic_straightness": float(straightness),
        }

    def _semantic_emit_marker(
        self,
        st: SemanticPathState,
        point: SemanticPathPoint,
        marker_type: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a confirmed semantic marker derived from display-path points."""
        if not bool(getattr(self, "visual_stream_enable", True)):
            return
        cam = str(st.cam)
        display_w, display_h = self._display_size_for_cam(cam)
        pts = list(st.points)
        metrics = self._semantic_path_metrics(pts)
        state_s = "terminal" if marker_type in {"end", "terminal"} else str(marker_type)
        marker_out = "end" if marker_type in {"terminal", "end"} else str(marker_type)
        msg = {
            "type": "overlay_bullet_event",
            "schema_version": 3,
            "sync_run_id": str(self.hit_cfg.get("sync_run_id", "real_run_001") or "real_run_001"),
            "pair_id": cam,
            "camera_sn": str(st.camera_sn),
            "camera_alias": cam,
            "bullet_id": int(st.bullet_id),
            "bullet_seq_id": int(point.seq_id),
            "bullet_ts_us": int(point.ts_us),
            "point_ts_us": int(point.ts_us),
            "point_index": int(point.point_index),
            "event_type": "semantic_" + state_s,
            "marker_type": marker_out,
            "state": state_s,
            "semantic_state": state_s,
            "terminate_reason": "semantic_terminal" if state_s == "terminal" else "",
            "x": round(float(point.x), 3),
            "y": round(float(point.y), 3),
            "event_point_mvs": [round(float(point.x), 3), round(float(point.y), 3)],
            "display_width": int(display_w),
            "display_height": int(display_h),
            "confirmed": True,
            "semantic_confirmed": True,
            "source": "bullet_path_semantic_resolver",
            "reason": str(reason),
            "wall_time": time.time(),
        }
        msg.update({k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in metrics.items()})
        if extra:
            msg.update(extra)
        self._write_visual_bullet_dict(cam, msg)
        self._broadcast_overlay(msg)

        marker_msg = {
            "type": "overlay_bullet_marker",
            "schema_version": 2,
            "sync_run_id": msg.get("sync_run_id"),
            "pair_id": cam,
            "camera_sn": str(st.camera_sn),
            "camera_alias": cam,
            "bullet_id": int(st.bullet_id),
            "bullet_seq_id": int(point.seq_id),
            "point_index": int(point.point_index),
            "marker_type": marker_out,
            "state": state_s,
            "semantic_state": state_s,
            "event_type": msg.get("event_type"),
            "terminate_reason": msg.get("terminate_reason", ""),
            "ts_us": int(point.ts_us),
            "point_ts_us": int(point.ts_us),
            "x": round(float(point.x), 3),
            "y": round(float(point.y), 3),
            "display_width": int(display_w),
            "display_height": int(display_h),
            "confirmed": True,
            "semantic_confirmed": True,
            "source": "bullet_path_semantic_resolver",
            "reason": str(reason),
            "wall_time": time.time(),
        }
        if extra:
            for k, v in extra.items():
                if k not in marker_msg:
                    marker_msg[k] = v
        self._broadcast_overlay(marker_msg)

    def _semantic_valid_terminal(self, st: SemanticPathState) -> Tuple[bool, str, Dict[str, Any]]:
        pts = list(st.points)
        metrics = self._semantic_path_metrics(pts)
        if len(pts) < int(getattr(self, "semantic_terminal_min_points", 3)):
            return False, "terminal_points_too_few", metrics
        if float(metrics.get("semantic_path_len_px", 0.0)) < float(getattr(self, "semantic_terminal_min_path_len_px", 35.0)):
            return False, "terminal_path_too_short", metrics
        if float(metrics.get("semantic_duration_ms", 0.0)) < float(getattr(self, "semantic_terminal_min_duration_ms", 8.0)):
            return False, "terminal_duration_too_short", metrics
        return True, "semantic_terminal_confirmed", metrics

    def _semantic_find_turn_candidate(self, st: SemanticPathState) -> Optional[Tuple[SemanticPathPoint, Dict[str, Any]]]:
        if not bool(getattr(self, "semantic_turn_enable", True)):
            return None
        pts = list(st.points)
        n = len(pts)
        if n < 5:
            return None
        max_candidates = max(1, int(getattr(self, "semantic_turn_max_candidates_per_point", 8)))
        start = max(1, n - 1 - max_candidates)
        min_leg = float(getattr(self, "semantic_turn_min_leg_px", 35.0))
        angle_thr = float(getattr(self, "semantic_turn_angle_deg", 40.0))
        now_pt = pts[-1]
        best: Optional[Tuple[SemanticPathPoint, Dict[str, Any]]] = None
        best_angle = 0.0
        for j in range(start, n - 1):
            pivot = pts[j]
            pkey = (int(pivot.point_index), round(float(pivot.x), 2), round(float(pivot.y), 2))
            if pkey in st.emitted_turn_keys:
                continue
            # Find an incoming anchor far enough before the pivot.
            prev_anchor = None
            in_len = 0.0
            for i in range(j - 1, -1, -1):
                cand = pts[i]
                d = math.hypot(float(pivot.x) - float(cand.x), float(pivot.y) - float(cand.y))
                if d >= min_leg:
                    prev_anchor = cand
                    in_len = d
                    break
            if prev_anchor is None:
                continue
            out_len = math.hypot(float(now_pt.x) - float(pivot.x), float(now_pt.y) - float(pivot.y))
            if out_len < min_leg:
                continue
            v1 = (float(pivot.x) - float(prev_anchor.x), float(pivot.y) - float(prev_anchor.y))
            v2 = (float(now_pt.x) - float(pivot.x), float(now_pt.y) - float(pivot.y))
            l1 = math.hypot(v1[0], v1[1])
            l2 = math.hypot(v2[0], v2[1])
            if l1 <= 1e-6 or l2 <= 1e-6:
                continue
            cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
            angle = math.degrees(math.acos(cosv))
            if angle < angle_thr:
                continue
            if st.last_turn_xy is not None:
                spacing_px = math.hypot(float(pivot.x) - float(st.last_turn_xy[0]), float(pivot.y) - float(st.last_turn_xy[1]))
                spacing_ms = abs(int(pivot.ts_us) - int(st.last_turn_ts_us)) / 1000.0 if st.last_turn_ts_us else 999999.0
                if spacing_px < float(getattr(self, "semantic_turn_min_spacing_px", 45.0)) and spacing_ms < float(getattr(self, "semantic_turn_min_spacing_ms", 55.0)):
                    continue
            if angle > best_angle:
                best_angle = angle
                best = (pivot, {
                    "turn_angle_deg": round(float(angle), 3),
                    "turn_in_len_px": round(float(in_len), 3),
                    "turn_out_len_px": round(float(out_len), 3),
                    "turn_prev_point_index": int(prev_anchor.point_index),
                    "turn_next_point_index": int(now_pt.point_index),
                    "turn_prev_xy": [round(float(prev_anchor.x), 3), round(float(prev_anchor.y), 3)],
                    "turn_next_xy": [round(float(now_pt.x), 3), round(float(now_pt.y), 3)],
                })
        return best

    def _semantic_process_point(
        self,
        cam: str,
        ev: BulletEvent,
        curr_mvs: Tuple[float, float],
        receive_wall: float,
    ) -> None:
        if not bool(getattr(self, "semantic_path_enable", True)):
            return
        key = self._semantic_key(cam, str(ev.camera_sn), int(ev.bullet_id))
        st = self.semantic_paths.get(key)
        if st is None:
            st = SemanticPathState(
                cam=str(cam),
                camera_sn=str(ev.camera_sn),
                bullet_id=int(ev.bullet_id),
                points=deque(maxlen=int(getattr(self, "semantic_path_max_points", 96))),
            )
            self.semantic_paths[key] = st
        point_index = int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1)
        # Ignore old/out-of-order samples in the semantic resolver.  They are still
        # logged as overlay_bullet_point, but should not move terminal/turn state.
        if st.points:
            last = st.points[-1]
            if point_index >= 0 and last.point_index >= 0 and point_index <= last.point_index:
                return
            if int(ev.timestamp_us) <= int(last.ts_us):
                return
        pt = SemanticPathPoint(
            cam=str(cam),
            camera_sn=str(ev.camera_sn),
            bullet_id=int(ev.bullet_id),
            seq_id=int(ev.seq_id),
            point_index=int(point_index),
            ts_us=int(ev.timestamp_us),
            x=float(curr_mvs[0]),
            y=float(curr_mvs[1]),
            raw_x=float(ev.event_x),
            raw_y=float(ev.event_y),
            receive_wall=float(receive_wall),
        )
        st.points.append(pt)
        st.last_update_wall = float(receive_wall)
        # A new point after a pending/previous terminal means the path continued;
        # allow a later terminal to be confirmed from the newer last point.
        if st.terminal_sent_for_index is not None and point_index > int(st.terminal_sent_for_index):
            st.terminal_sent_for_index = None

        turn = self._semantic_find_turn_candidate(st)
        if turn is not None:
            pivot, extra = turn
            pkey = (int(pivot.point_index), round(float(pivot.x), 2), round(float(pivot.y), 2))
            st.emitted_turn_keys.add(pkey)
            st.last_turn_wall = float(receive_wall)
            st.last_turn_ts_us = int(pivot.ts_us)
            st.last_turn_xy = (float(pivot.x), float(pivot.y))
            self._semantic_emit_marker(st, pivot, "turn", "semantic_turn_from_display_path", extra=extra)

    def _flush_semantic_terminals(self, force: bool = False) -> None:
        if not bool(getattr(self, "semantic_path_enable", True)):
            return
        now = time.time()
        delay_s = float(getattr(self, "semantic_terminal_delay_ms", 140.0)) / 1000.0
        cleanup_s = float(getattr(self, "semantic_terminal_cleanup_ms", 3000.0)) / 1000.0
        remove_keys = []
        for key, st in list(self.semantic_paths.items()):
            if not st.points:
                remove_keys.append(key)
                continue
            last = st.points[-1]
            age_s = now - float(st.last_update_wall)
            if (force or age_s >= delay_s) and st.terminal_sent_for_index != int(last.point_index):
                ok, reason, metrics = self._semantic_valid_terminal(st)
                if ok:
                    extra = {
                        "terminal_delay_ms": round(float(age_s * 1000.0), 3),
                        "terminal_confirm_delay_ms": round(float(age_s * 1000.0), 3),
                    }
                    extra.update({k: round(float(v), 3) for k, v in metrics.items()})
                    self._semantic_emit_marker(st, last, "terminal", reason, extra=extra)
                    st.terminal_sent_for_index = int(last.point_index)
                    st.terminal_sent_ts_us = int(last.ts_us)
                    st.terminal_generation += 1
                else:
                    # Keep this as a JSONL audit event only; no UE5 marker is sent.
                    display_w, display_h = self._display_size_for_cam(str(st.cam))
                    msg = {
                        "type": "overlay_bullet_event",
                        "schema_version": 3,
                        "sync_run_id": str(self.hit_cfg.get("sync_run_id", "real_run_001") or "real_run_001"),
                        "pair_id": str(st.cam),
                        "camera_sn": str(st.camera_sn),
                        "camera_alias": str(st.cam),
                        "bullet_id": int(st.bullet_id),
                        "bullet_seq_id": int(last.seq_id),
                        "bullet_ts_us": int(last.ts_us),
                        "point_ts_us": int(last.ts_us),
                        "point_index": int(last.point_index),
                        "event_type": "semantic_terminal_suppressed",
                        "marker_type": "terminal_suppressed",
                        "state": "suppressed",
                        "semantic_state": "suppressed",
                        "x": round(float(last.x), 3),
                        "y": round(float(last.y), 3),
                        "display_width": int(display_w),
                        "display_height": int(display_h),
                        "confirmed": False,
                        "semantic_confirmed": False,
                        "ue5_marker_suppressed": True,
                        "ue5_marker_suppression_reason": str(reason),
                        "source": "bullet_path_semantic_resolver",
                        "reason": str(reason),
                        "wall_time": now,
                    }
                    msg.update({k: round(float(v), 3) for k, v in metrics.items()})
                    self._write_visual_bullet_dict(str(st.cam), msg)
                    st.terminal_sent_for_index = int(last.point_index)
            if st.terminal_sent_for_index is not None and age_s >= cleanup_s:
                remove_keys.append(key)
        for key in remove_keys:
            self.semantic_paths.pop(key, None)

    def _mark_visual_point_hit_emitted(self, cam: str, bullet_id: int, stable_id: int) -> bool:
        stable_key = 0 if bool(getattr(self, "visual_point_hit_emit_once_per_bullet", True)) else int(stable_id)
        key = (str(cam), int(bullet_id), int(stable_key))
        if key in self.visual_point_hit_set:
            return False
        if self.visual_point_hit_keys.maxlen is not None and len(self.visual_point_hit_keys) >= self.visual_point_hit_keys.maxlen:
            try:
                old = self.visual_point_hit_keys.popleft()
                self.visual_point_hit_set.discard(old)
            except Exception:
                pass
        self.visual_point_hit_keys.append(key)
        self.visual_point_hit_set.add(key)
        return True

    def _visual_point_pending_key(self, cam: str, ev: BulletEvent) -> Tuple[str, str, int, int, int]:
        point_index = int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1)
        return (str(cam), str(ev.camera_sn), int(ev.bullet_id), int(point_index), int(ev.timestamp_us))

    def _record_visual_point_stage(
        self,
        stage: str,
        *,
        cam: Optional[str] = None,
        ev: Optional[BulletEvent] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = str(stage or "unknown")
        self.visual_point_stage_counts[key] += 1
        cam_s = str(cam or "")
        if not cam_s and ev is not None:
            cam_s = str(getattr(ev, "camera_alias", "") or getattr(ev, "camera_sn", "") or "")
        if cam_s:
            self.visual_point_stage_by_camera[cam_s][key] += 1
        if ev is not None:
            item = {
                "stage": key,
                "camera": cam_s,
                "camera_sn": str(getattr(ev, "camera_sn", "")),
                "bullet_id": int(getattr(ev, "bullet_id", -1) or -1),
                "seq_id": int(getattr(ev, "seq_id", -1) or -1),
                "point_index": int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1),
                "event_ts_us": int(getattr(ev, "timestamp_us", 0) or 0),
                "wall_time": time.time(),
            }
            if detail:
                item.update(detail)
            self.latest_visual_point_stage = item

    def _record_visual_point_result(self, result: str, *, cam: Optional[str] = None, ev: Optional[BulletEvent] = None) -> None:
        key = str(result or "unknown")
        self.visual_point_result_counts[key] += 1
        self._record_visual_point_stage(f"result:{key}", cam=cam, ev=ev)
        if key.startswith("point_") and key not in {"point_emitted", "point_only", "point_pending_human"}:
            self.hit_reject_counts[key] += 1
            if ev is not None:
                self.latest_visual_point_reject = {
                    "result": key,
                    "camera": str(cam or getattr(ev, "camera_alias", "") or ""),
                    "camera_sn": str(getattr(ev, "camera_sn", "")),
                    "bullet_id": int(getattr(ev, "bullet_id", -1) or -1),
                    "seq_id": int(getattr(ev, "seq_id", -1) or -1),
                    "point_index": int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1),
                    "event_ts_us": int(getattr(ev, "timestamp_us", 0) or 0),
                    "wall_time": time.time(),
                }

    def _emit_visual_point_reject(
        self,
        cam: str,
        ev: BulletEvent,
        status: str,
        reject_reason: str,
        *,
        period: Optional[Dict[str, Any]] = None,
        selection: Optional[Dict[str, Any]] = None,
        geometry: Optional[Dict[str, Any]] = None,
        age_ms: Optional[float] = None,
    ) -> None:
        status_s = str(status or reject_reason or "point_rejected")
        reason_s = str(reject_reason or status_s)
        self._record_visual_point_result(status_s, cam=cam, ev=ev)
        self._emit_debug(cam, ev, status_s, reason_s, period=period, selection=selection, geometry=geometry, age_ms=age_ms)
        if not bool(getattr(self, "write_reject_debug_jsonl", False)):
            return
        try:
            d = self._debug_base(ev, cam, status_s, reason_s)
            d["reject_stage"] = "visual_point_segment"
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
                })
            if selection:
                d.update(selection)
            if geometry:
                d.update(geometry)
            line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
            if cam and cam in getattr(self, "reject_debug_files", {}):
                self.reject_debug_files[cam].write(line)
            if getattr(self, "reject_debug_all_fp", None) is not None:
                self.reject_debug_all_fp.write(line)
        except Exception as exc:
            print(f"[hit_judge][WARN] reject debug write failed: {exc}", flush=True)

    def _register_pending_visual_point_segment(
        self,
        cam: str,
        ev: BulletEvent,
        prev_state: LiveBulletPointState,
        curr_mvs: Tuple[float, float],
        age_ms: float,
        *,
        period: Optional[Dict[str, Any]] = None,
        selection: Optional[Dict[str, Any]] = None,
        geometry: Optional[Dict[str, Any]] = None,
    ) -> str:
        if float(getattr(self, "visual_point_hit_late_human_retry_ms", 0.0) or 0.0) <= 0.0:
            self._emit_visual_point_reject(
                cam, ev, "point_no_human", "no_human_result_near_point_ts",
                period=period, selection=selection, geometry=geometry, age_ms=age_ms,
            )
            return "point_no_human"
        key = self._visual_point_pending_key(cam, ev)
        if key not in self.visual_point_pending_keys:
            maxlen = self.visual_point_pending_segments.maxlen
            if maxlen is not None and len(self.visual_point_pending_segments) >= int(maxlen):
                old = self.visual_point_pending_segments.popleft()
                self.visual_point_pending_keys.discard(self._visual_point_pending_key(old.cam, old.ev))
            self.visual_point_pending_segments.append(
                PendingVisualPointSegment(cam=str(cam), ev=ev, prev_state=prev_state, curr_mvs=curr_mvs, age_ms=float(age_ms))
            )
            self.visual_point_pending_keys.add(key)
        self._record_visual_point_result("point_pending_human", cam=cam, ev=ev)
        return "point_pending_human"

    def _process_pending_visual_point_segments(self) -> None:
        if not getattr(self, "visual_point_pending_segments", None):
            return
        retry_ms = max(0.0, float(getattr(self, "visual_point_hit_late_human_retry_ms", 80.0) or 0.0))
        now = time.time()
        keep: Deque[PendingVisualPointSegment] = deque(maxlen=self.visual_point_pending_segments.maxlen)
        while self.visual_point_pending_segments:
            item = self.visual_point_pending_segments.popleft()
            key = self._visual_point_pending_key(item.cam, item.ev)
            age_ms = (now - float(item.created_wall)) * 1000.0
            if age_ms < retry_ms:
                item.tries += 1
                result = self._judge_visual_point_segment(
                    item.cam,
                    item.ev,
                    item.prev_state,
                    item.curr_mvs,
                    age_ms=float(age_ms),
                    allow_late_human_pending=False,
                    log_no_human_reject=False,
                )
                if result == "point_no_human":
                    keep.append(item)
                    continue
                self.visual_point_pending_keys.discard(key)
                continue
            self.visual_point_pending_keys.discard(key)
            self._emit_visual_point_reject(
                item.cam,
                item.ev,
                "point_no_human_timeout",
                "late_human_retry_timeout",
                geometry={"late_human_retry_ms": round(float(retry_ms), 3), "late_human_age_ms": round(float(age_ms), 3), "late_human_tries": int(item.tries)},
                age_ms=float(age_ms),
            )
        self.visual_point_pending_segments = keep

    def _trajectory_only_final_enabled(self) -> bool:
        mode = str(getattr(self, "hit_judge_final_mode", "") or "").strip().lower().replace("-", "_")
        return mode in {"trajectory_only", "track_only", "ballistic_only", "raw_trajectory_only"}

    def _evidence_gated_final_enabled(self) -> bool:
        mode = str(getattr(self, "hit_judge_final_mode", "") or "").strip().lower().replace("-", "_")
        return mode in {"evidence_gated", "trajectory_evidence_gated", "track_evidence_gated", "raw_evidence_gated"}

    def _mask_evidence_path(self, cam: str) -> str:
        p = self.evidence_gated_mask_json_template.format(root=self.root, camera=cam, cam=cam, id=cam)
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        return p

    def _read_mask_evidence_json(self, cam: str) -> Dict[str, Any]:
        now = time.time()
        cache = getattr(self, "_mask_evidence_cache", {}).get(str(cam))
        ttl_s = max(0.0, float(getattr(self, "evidence_gated_mask_cache_ttl_ms", 25.0) or 25.0)) / 1000.0
        if cache is not None and now - float(cache[0]) <= ttl_s:
            return dict(cache[1])
        path = self._mask_evidence_path(str(cam))
        obj: Dict[str, Any] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                obj = loaded
        except Exception as exc:
            obj = {"_read_error": f"{type(exc).__name__}: {exc}"}
        try:
            mtime = os.path.getmtime(path)
            obj["_file_exists"] = True
            obj["_file_age_ms"] = max(0.0, (time.time() - mtime) * 1000.0)
        except Exception:
            obj["_file_exists"] = False
            obj["_file_age_ms"] = -1.0
        obj["_path"] = path
        self._mask_evidence_cache[str(cam)] = (now, dict(obj))
        return obj

    def _mask_evidence_gate_ok(self, cam: str, entry: Optional[PersonEntry], entry_sync_hint: int = -1) -> Tuple[bool, str, Dict[str, Any]]:
        detail: Dict[str, Any] = {
            "require_mask": bool(getattr(self, "evidence_gated_require_mask", True)),
            "require_polygon": bool(getattr(self, "evidence_gated_require_mask_polygon", True)),
            "max_age_ms": float(getattr(self, "evidence_gated_mask_max_age_ms", 220.0)),
            "max_sync_delta": int(getattr(self, "evidence_gated_mask_max_sync_delta", 2)),
        }
        if not bool(getattr(self, "evidence_gated_require_mask", True)):
            detail["mask_evidence_required"] = False
            return True, "", detail
        evidence = self._read_mask_evidence_json(cam)
        detail.update({
            "mask_evidence_required": True,
            "mask_evidence_path": str(evidence.get("_path", "")),
            "mask_evidence_file_exists": bool(evidence.get("_file_exists", False)),
            "mask_evidence_file_age_ms": round(float(evidence.get("_file_age_ms", -1.0) or -1.0), 3),
            "mask_evidence_fresh": bool(evidence.get("fresh", False)),
            "mask_evidence_age_ms": round(float(evidence.get("age_ms", evidence.get("_file_age_ms", -1.0)) or -1.0), 3),
            "mask_evidence_sync_index": _safe_int(evidence.get("sync_index"), -1),
            "mask_evidence_person_count": _safe_int(evidence.get("person_count"), 0),
            "mask_evidence_polygon_count": _safe_int(evidence.get("polygon_count"), 0),
            "mask_evidence_bbox_count": _safe_int(evidence.get("bbox_count"), 0),
        })
        if not bool(evidence.get("_file_exists", False)):
            detail["reject_reason"] = "mask_evidence_file_missing"
            return False, "mask_evidence_file_missing", detail
        if str(evidence.get("_read_error", "") or ""):
            detail["reject_reason"] = "mask_evidence_read_error"
            detail["mask_evidence_read_error"] = str(evidence.get("_read_error"))
            return False, "mask_evidence_read_error", detail
        age_ms = float(evidence.get("age_ms", evidence.get("_file_age_ms", -1.0)) or -1.0)
        if age_ms < 0.0 or age_ms > float(getattr(self, "evidence_gated_mask_max_age_ms", 220.0)):
            detail["reject_reason"] = "mask_evidence_stale"
            return False, "mask_evidence_stale", detail
        if not bool(evidence.get("fresh", False)):
            detail["reject_reason"] = "mask_evidence_not_fresh"
            return False, "mask_evidence_not_fresh", detail
        if bool(getattr(self, "evidence_gated_require_mask_polygon", True)) and _safe_int(evidence.get("polygon_count"), 0) <= 0:
            detail["reject_reason"] = "mask_evidence_no_polygon"
            return False, "mask_evidence_no_polygon", detail
        entry_sync = _safe_int(getattr(entry, "sync_index", -1), -1) if entry is not None else _safe_int(entry_sync_hint, -1)
        if entry_sync >= 0:
            ev_sync = _safe_int(evidence.get("sync_index"), -1)
            detail["person_entry_sync_index"] = int(entry_sync)
            if ev_sync >= 0 and entry_sync >= 0:
                delta = int(ev_sync - entry_sync)
                detail["mask_person_sync_delta"] = int(delta)
                if abs(delta) > int(getattr(self, "evidence_gated_mask_max_sync_delta", 2)):
                    detail["reject_reason"] = "mask_evidence_sync_delta_too_large"
                    return False, "mask_evidence_sync_delta_too_large", detail
        detail["reject_reason"] = ""
        return True, "", detail

    def _final_evidence_gate_ok(
        self,
        cam: str,
        entry: Optional[PersonEntry],
        period: Optional[Dict[str, Any]],
        selection_info: Dict[str, Any],
        geometry: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        detail: Dict[str, Any] = {
            "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
            "trigger_evidence_required": bool(getattr(self, "evidence_gated_require_trigger", True)),
            "person_sync_evidence_required": bool(getattr(self, "evidence_gated_require_person_sync", True)),
            "mask_evidence_required": bool(getattr(self, "evidence_gated_require_mask", True)),
            "geometry_mode": str(geometry.get("geometry_best_method", geometry.get("geometry_mode", "")) or ""),
            "evidence_level": str(geometry.get("evidence_level", "") or ""),
        }
        if bool(getattr(self, "evidence_gated_require_trigger", True)):
            if period is None:
                detail["reject_reason"] = "evidence_gate_no_trigger"
                return False, "evidence_gate_no_trigger", detail
            if not bool(period.get("trigger_period_ok", True)) and not bool(getattr(self, "emit_candidate_on_bad_trigger_period", False)):
                detail["reject_reason"] = "evidence_gate_bad_trigger"
                return False, "evidence_gate_bad_trigger", detail
        detail["trigger_evidence_used"] = period is not None
        if bool(getattr(self, "evidence_gated_require_person_sync", True)):
            used_sync = _safe_int(getattr(entry, "sync_index", -1), -1) if entry is not None else _safe_int(selection_info.get("used_person_sync_index"), -1)
            if used_sync < 0:
                detail["reject_reason"] = "evidence_gate_no_person_sync"
                return False, "evidence_gate_no_person_sync", detail
            try:
                frame_delta_i = int(selection_info.get("person_frame_delta_sync", 0))
            except Exception:
                frame_delta_i = 999999
            detail["person_frame_delta_sync"] = int(frame_delta_i)
            detail["target_person_sync_index"] = selection_info.get("target_person_sync_index")
            detail["used_person_sync_index"] = int(used_sync)
            if abs(frame_delta_i) > int(getattr(self, "evidence_gated_max_person_frame_delta_sync", 0)):
                detail["reject_reason"] = "evidence_gate_person_sync_delta_too_large"
                return False, "evidence_gate_person_sync_delta_too_large", detail
        detail["person_sync_evidence_used"] = bool(entry is not None or _safe_int(selection_info.get("used_person_sync_index"), -1) >= 0)
        mask_ok, mask_reason, mask_detail = self._mask_evidence_gate_ok(cam, entry, _safe_int(selection_info.get("used_person_sync_index"), -1))
        detail["mask_evidence"] = mask_detail
        if not mask_ok:
            if str(mask_reason) == "mask_evidence_sync_delta_too_large":
                detail["mask_evidence_used"] = True
                detail["mask_evidence_quality"] = "degraded_sync_delta"
                detail["mask_evidence_degraded"] = True
                detail["mask_evidence_degrade_reason"] = str(mask_reason)
                detail["reject_reason"] = ""
                return True, "", detail
            detail["reject_reason"] = str(mask_reason)
            return False, str(mask_reason), detail
        detail["mask_evidence_used"] = bool(getattr(self, "evidence_gated_require_mask", True))
        detail["reject_reason"] = ""
        return True, "", detail

    def _trajectory_only_quality(
        self,
        track_metrics: Dict[str, float],
        *,
        min_segment_len_px: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        def _f(key: str, default: float = 0.0) -> float:
            try:
                value = track_metrics.get(key, default)
                if value is None or value == "":
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        def _i(key: str, default: int = 0) -> int:
            try:
                value = track_metrics.get(key, default)
                if value is None or value == "":
                    return int(default)
                return int(float(value))
            except Exception:
                return int(default)

        point_count = _i("track_point_count", 0)
        path_len = _f("track_path_len_px", 0.0)
        duration_ms = _f("track_duration_ms", 0.0)
        straightness = _f("track_straightness", 0.0)
        line_error = _f("track_line_error_px", 999.0)
        speed = _f("avg_speed_px_per_ms", 0.0)
        segment_len = _f("approach_len_px", 0.0)
        line_error_unknown = (not math.isfinite(line_error)) or line_error >= 900.0
        detail: Dict[str, Any] = {
            "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
            "trajectory_only_min_track_points": int(getattr(self, "trajectory_only_min_track_points", 0)),
            "trajectory_only_min_track_path_len_px": round(float(getattr(self, "trajectory_only_min_track_path_len_px", 0.0)), 3),
            "trajectory_only_min_duration_ms": round(float(getattr(self, "trajectory_only_min_duration_ms", 0.0)), 3),
            "trajectory_only_min_straightness": round(float(getattr(self, "trajectory_only_min_straightness", 0.0)), 3),
            "trajectory_only_max_line_error_px": round(float(getattr(self, "trajectory_only_max_line_error_px", 0.0)), 3),
            "trajectory_only_min_speed_px_per_ms": round(float(getattr(self, "trajectory_only_min_speed_px_per_ms", 0.0)), 3),
            "trajectory_only_min_segment_len_px": round(float(min_segment_len_px), 3),
            "trajectory_only_allow_unknown_line_error": bool(getattr(self, "trajectory_only_allow_unknown_line_error", True)),
            "track_point_count": int(point_count),
            "track_path_len_px": round(float(path_len), 3),
            "track_duration_ms": round(float(duration_ms), 3),
            "track_straightness": round(float(straightness), 3),
            "track_line_error_px": round(float(line_error), 3) if math.isfinite(line_error) else 999.0,
            "track_line_error_unknown": bool(line_error_unknown),
            "avg_speed_px_per_ms": round(float(speed), 3),
            "approach_len_px": round(float(segment_len), 3),
        }
        checks = [
            (point_count >= int(getattr(self, "trajectory_only_min_track_points", 0)), "trajectory_only_track_points_too_small"),
            (path_len >= float(getattr(self, "trajectory_only_min_track_path_len_px", 0.0)), "trajectory_only_path_len_too_short"),
            (duration_ms >= float(getattr(self, "trajectory_only_min_duration_ms", 0.0)), "trajectory_only_duration_too_short"),
            (straightness >= float(getattr(self, "trajectory_only_min_straightness", 0.0)), "trajectory_only_track_not_straight"),
            (speed >= float(getattr(self, "trajectory_only_min_speed_px_per_ms", 0.0)), "trajectory_only_speed_too_low"),
            (segment_len >= float(min_segment_len_px), "trajectory_only_segment_too_short"),
        ]
        for ok, reason in checks:
            if not ok:
                detail["trajectory_only_reject_reason"] = reason
                return False, reason, detail
        if line_error_unknown:
            if not bool(getattr(self, "trajectory_only_allow_unknown_line_error", True)):
                detail["trajectory_only_reject_reason"] = "trajectory_only_line_error_unknown"
                return False, "trajectory_only_line_error_unknown", detail
        elif line_error > float(getattr(self, "trajectory_only_max_line_error_px", 0.0)):
            detail["trajectory_only_reject_reason"] = "trajectory_only_line_error_too_large"
            return False, "trajectory_only_line_error_too_large", detail
        detail["trajectory_only_reject_reason"] = ""
        return True, "", detail

    def _emit_trajectory_only_hit(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        track_metrics: Dict[str, float],
        *,
        judge_source: str,
        use_segment: bool,
        min_segment_len_px: float,
        point_index: int = -1,
        prev_point_index: int = -1,
        point_delta_ms: float = 0.0,
        segment_len_mvs_px: float = 0.0,
        age_ms: float = 0.0,
    ) -> str:
        quality_ok, quality_reason, quality_detail = self._trajectory_only_quality(
            track_metrics,
            min_segment_len_px=float(min_segment_len_px),
        )
        geometry = {
            "judge_source": str(judge_source),
            "event_point_mvs": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "approach_point_mvs": [round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],
            "trajectory_segment_used": bool(use_segment),
            "mask_evidence_used": False,
            "trigger_evidence_used": False,
            "person_sync_evidence_used": False,
            **quality_detail,
        }
        if int(point_index) >= 0:
            geometry.update({
                "point_index": int(point_index),
                "prev_point_index": int(prev_point_index),
                "point_delta_ms": round(float(point_delta_ms), 3),
                "segment_len_mvs_px": round(float(segment_len_mvs_px), 3),
            })
        if not quality_ok:
            if str(judge_source) == "visual_point_segment":
                self._emit_visual_point_reject(
                    cam,
                    ev,
                    f"point_{quality_reason}",
                    str(quality_reason),
                    geometry=geometry,
                    age_ms=age_ms,
                )
                return f"point_{quality_reason}"
            self._emit_debug(cam, ev, "no_valid_trajectory", str(quality_reason), geometry=geometry, age_ms=age_ms)
            return "no_candidate"

        speed = float(track_metrics.get("avg_speed_px_per_ms", 0.0) or 0.0)
        path_len = float(track_metrics.get("track_path_len_px", 0.0) or 0.0)
        min_path = max(1e-3, float(getattr(self, "trajectory_only_min_track_path_len_px", 1.0) or 1.0))
        min_speed = max(1e-3, float(getattr(self, "trajectory_only_min_speed_px_per_ms", 1.0) or 1.0))
        event_score = 0.18 if str(ev.event_type) == EVENT_POINT else (0.20 if str(ev.event_type) == EVENT_TERMINATE else 0.16)
        score = ScoreBreakdown(
            time_score=0.0,
            space_score=min(0.45, max(0.0, path_len / min_path) * 0.45),
            event_score=float(event_score),
            speed_score=min(0.17, max(0.0, speed / min_speed) * 0.17),
            loss_penalty=0.0,
        )
        required_score = float(getattr(self, "trajectory_only_min_score", self.visual_point_hit_min_score) or self.visual_point_hit_min_score)
        if score.total < required_score:
            geometry.update({
                "trajectory_only_total_score": round(float(score.total), 3),
                "trajectory_only_required_score": round(float(required_score), 3),
            })
            if str(judge_source) == "visual_point_segment":
                self._emit_visual_point_reject(
                    cam,
                    ev,
                    "point_trajectory_only_score_too_low",
                    "trajectory_only_score_too_low",
                    geometry=geometry,
                    age_ms=age_ms,
                )
                return "point_trajectory_only_score_too_low"
            self._emit_debug(cam, ev, "no_valid_trajectory", "trajectory_only_score_too_low", geometry=geometry, age_ms=age_ms)
            return "no_candidate"

        if bool(getattr(self, "visual_point_hit_emit_once_per_bullet", True)):
            if not self._mark_visual_point_hit_emitted(cam, int(ev.bullet_id), 0):
                if str(judge_source) == "visual_point_segment":
                    self._emit_visual_point_reject(
                        cam,
                        ev,
                        "point_duplicate_hit_ignored",
                        "trajectory_only_hit_already_emitted_for_bullet",
                        geometry=geometry,
                        age_ms=age_ms,
                    )
                    return "point_duplicate_hit_ignored"
                self._emit_debug(cam, ev, "duplicate_hit_ignored", "trajectory_only_hit_already_emitted_for_bullet", geometry=geometry, age_ms=age_ms)
                return "no_candidate"

        cand = HitCandidate(
            camera_sn=str(ev.camera_sn),
            camera_alias=str(cam),
            bullet_id=int(ev.bullet_id),
            seq_id=int(ev.seq_id),
            stable_id=0,
            global_player_id=None,
            event_ts=int(ev.timestamp_us),
            person_ts=0,
            dt_ms=-1.0,
            event_type=str(ev.event_type),
            terminate_reason=str(ev.terminate_reason),
            turn_angle_deg=float(ev.turn_angle_deg),
            event_point=(float(event_pt[0]), float(event_pt[1])),
            approach_point=(float(approach_pt[0]), float(approach_pt[1])),
            approach_valid=bool(use_segment),
            match_method="trajectory_only",
            track_point_count=int(track_metrics.get("track_point_count", 0) or 0),
            track_path_len_px=float(track_metrics.get("track_path_len_px", 0.0) or 0.0),
            track_duration_ms=float(track_metrics.get("track_duration_ms", 0.0) or 0.0),
            track_displacement_px=float(track_metrics.get("track_displacement_px", 0.0) or 0.0),
            track_straightness=float(track_metrics.get("track_straightness", 0.0) or 0.0),
            track_line_error_px=float(track_metrics.get("track_line_error_px", 999.0) if track_metrics.get("track_line_error_px", 999.0) is not None else 999.0),
            person_bbox=(0, 0, 0, 0),
            person_centroid=(0.0, 0.0),
            person_confirmed=False,
            person_held=False,
            score_breakdown=score,
        )
        d = cand.to_dict()
        d.update({
            "source": "hit_judge_server",
            "judge_mode": "trajectory_only",
            "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
            "judge_source": str(judge_source),
            "human_match_strategy": "not_used",
            "human_fallback_used": False,
            "human_fallback_reason": "",
            "target_person_sync_index": None,
            "used_person_sync_index": None,
            "person_frame_delta_sync": None,
            "mvs_sync_index": -1,
            "person_frame_sync_index": -1,
            "impact_sync_index": -1,
            "impact_sync_source": "not_used",
            "human_evidence_used": False,
            "mask_evidence_used": False,
            "trigger_evidence_used": False,
            "person_sync_evidence_used": False,
            "evidence_level": "ballistic_track",
            "geometry_mode": "trajectory_only",
            "track_quality_enable": bool(self.track_quality_enable),
            "final_hit_mask_only": False,
            "configured_final_hit_mask_only": bool(self.final_hit_mask_only),
            "trajectory_required_for_hit": True,
            "trajectory_segment_used": bool(use_segment),
            "approach_len_px": round(float(track_metrics.get("approach_len_px", 0.0) or 0.0), 3),
            "min_hit_trajectory_len_px": round(float(getattr(self, "trajectory_only_min_track_path_len_px", 0.0)), 3),
            "trajectory_only_min_score": round(float(required_score), 3),
            "trajectory_only_quality": quality_detail,
            "event_human_mask_stats": dict(getattr(ev, "event_human_mask_stats", {}) or {}),
        })
        if int(point_index) >= 0:
            d.update({
                "realtime_point_hit": True,
                "point_index": int(point_index),
                "prev_point_index": int(prev_point_index),
                "point_delta_ms": round(float(point_delta_ms), 3),
                "segment_len_mvs_px": round(float(segment_len_mvs_px), 3),
            })
        self._write_candidate_dict(cam, d)
        if str(judge_source) == "visual_point_segment":
            self._record_visual_point_result("point_emitted", cam=cam, ev=ev)
        self.latest_visual_point_candidate = {
            "camera": str(cam),
            "camera_sn": str(ev.camera_sn),
            "bullet_id": int(ev.bullet_id),
            "stable_id": 0,
            "seq_id": int(ev.seq_id),
            "point_index": int(point_index),
            "total_score": round(float(cand.total_score), 3),
            "match_method": "trajectory_only",
            "judge_source": str(judge_source),
            "wall_time": time.time(),
        }
        self._emit_debug(cam, ev, "emitted_candidate", None, geometry=geometry, candidate=dict(d), age_ms=age_ms)
        print(f"[hit_judge][{cam}] trajectory-only candidate bullet={ev.bullet_id} score={cand.total_score:.3f} points={int(track_metrics.get('track_point_count', 0) or 0)} path={float(track_metrics.get('track_path_len_px', 0.0) or 0.0):.1f}px source={judge_source}", flush=True)
        return "point_emitted" if str(judge_source) == "visual_point_segment" else "emitted"

    def _write_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - float(getattr(self, "last_status_write", 0.0) or 0.0) < float(getattr(self, "status_interval_s", 1.0) or 1.0):
            return
        self.last_status_write = now
        try:
            obj = {
                "schema_version": 1,
                "source": "hit_judge_server",
                "timestamp_us": int(now * 1_000_000),
                "status": "running",
                "enabled": bool(getattr(self, "hit_detection_enabled", True)),
                "hit_detection_enabled": bool(getattr(self, "hit_detection_enabled", True)),
                "hit_candidate_output_enabled": bool(getattr(self, "hit_candidate_output_enabled", True)),
                "strong_gate_enabled": bool(getattr(self, "strong_gate_enabled", True)),
                "strong_reasoning_enabled": bool(getattr(self, "strong_reasoning_enabled", True)),
                "strong_reasoning_shadow_only": bool(getattr(self, "strong_reasoning_shadow_only", True)),
                "final_hit_authority": str(getattr(self, "final_hit_authority", "strong_gate") or "strong_gate"),
                "pseudo_hit_enable": bool(getattr(self, "pseudo_hit_enable", True)),
                "control": {
                    "path": str(getattr(self, "control_path", "")),
                    "seq": int(getattr(self, "control_seq", 0) or 0),
                    "error": str(getattr(self, "control_error", "") or ""),
                },
                "hit_recv_total": int(getattr(self, "recv_total", 0) or 0),
                "hit_candidate_total": int(getattr(self, "cand_total", 0) or 0),
                "pseudo_hit_total": int(getattr(self, "pseudo_hit_total", 0) or 0),
                "pseudo_hit_by_camera": dict(getattr(self, "pseudo_hit_by_camera", {}) or {}),
                "pseudo_hit_reason_total": dict(getattr(self, "pseudo_hit_reason_counts", {}) or {}),
                "pseudo_hit_duplicate_suppressed": int(getattr(self, "pseudo_hit_duplicate_suppressed", 0) or 0),
                "pseudo_hit_promoted_to_final_count": int(getattr(self, "pseudo_hit_promoted_to_final_count", 0) or 0),
                "pseudo_hit_human_noise_suspect": int(getattr(self, "pseudo_hit_human_noise_suspect", 0) or 0),
                "pseudo_hit_human_noise_suppressed": int(getattr(self, "pseudo_hit_human_noise_suppressed", 0) or 0),
                "pseudo_hit_human_noise_reason_total": dict(getattr(self, "pseudo_hit_human_noise_reason_counts", {}) or {}),
                "human_noise_filter": {
                    "enable": bool(getattr(self, "human_noise_filter_enable", True)),
                    "mode": str(getattr(self, "human_noise_filter_mode", "soft") or "soft"),
                    "require_inbound_for_low_quality": bool(getattr(self, "human_noise_require_inbound_for_low_quality", True)),
                    "soft_threshold": round(float(getattr(self, "human_noise_soft_threshold", 0.62) or 0.62), 3),
                    "hard_threshold": round(float(getattr(self, "human_noise_hard_threshold", 0.88) or 0.88), 3),
                    "min_pre_hit_outside_points": int(getattr(self, "human_noise_min_pre_hit_outside_points", 3) or 3),
                    "min_pre_hit_path_len_px": round(float(getattr(self, "human_noise_min_pre_hit_path_len_px", 30.0) or 30.0), 3),
                    "short_track_max_points": int(getattr(self, "human_noise_short_track_max_points", 5) or 5),
                    "short_track_max_path_len_px": round(float(getattr(self, "human_noise_short_track_max_path_len_px", 35.0) or 35.0), 3),
                    "short_track_max_duration_ms": round(float(getattr(self, "human_noise_short_track_max_duration_ms", 20.0) or 20.0), 3),
                    "track_start_bbox_filter_enable": bool(getattr(self, "human_noise_track_start_bbox_filter_enable", True)),
                    "track_start_bbox_expand_px": round(float(getattr(self, "human_noise_track_start_bbox_expand_px", 24.0) or 24.0), 3),
                    "track_start_bbox_policy": str(getattr(self, "human_noise_track_start_bbox_policy", "soft") or "soft"),
                },
                "strong_gate_output_suppressed": int(getattr(self, "strong_gate_output_suppressed", 0) or 0),
                "strong_reasoning_output_suppressed": int(getattr(self, "strong_reasoning_output_suppressed", 0) or 0),
                "hit_shadow_enable": bool(getattr(self, "hit_shadow_enable", True)),
                "final_hit_cross_camera_dedup_enable": bool(getattr(self, "final_hit_cross_camera_dedup_enable", True)),
                "final_hit_cross_camera_dedup_ms": round(float(getattr(self, "final_hit_cross_camera_dedup_ms", 700.0) or 700.0), 3),
                "final_hit_duplicate_suppressed": int(getattr(self, "final_hit_duplicate_suppressed", 0) or 0),
                "final_hit_same_track_episode_dedup_enable": bool(getattr(self, "final_hit_same_track_episode_dedup_enable", True)),
                "final_hit_same_track_episode_dedup_ms": round(float(getattr(self, "final_hit_same_track_episode_dedup_ms", 1500.0) or 1500.0), 3),
                "same_track_episode_duplicate_suppressed": int(getattr(self, "same_track_episode_duplicate_suppressed", 0) or 0),
                "projected_mask_policy": {
                    "final_mode": str(getattr(self, "projected_mask_final_mode", "pending_only") or "pending_only"),
                    "require_terminal_or_turn": bool(getattr(self, "projected_mask_require_terminal_or_turn", True)),
                    "project_enable": bool(getattr(self, "strong_reasoning_track_project_enable", True)),
                    "project_extend_px": round(float(getattr(self, "strong_reasoning_track_project_extend_px", 100.0) or 100.0), 3),
                    "project_max_last_dist_px": round(float(getattr(self, "strong_reasoning_track_project_max_last_dist_px", 40.0) or 40.0), 3),
                    "pending_count": int(getattr(self, "projected_pending_count", 0) or 0),
                    "promoted_count": int(getattr(self, "projected_promoted_count", 0) or 0),
                    "suppressed_final_count": int(getattr(self, "projected_suppressed_final_count", 0) or 0),
                },
                "latest_pseudo_hit": dict(getattr(self, "latest_pseudo_hit", {}) or {}),
                "hit_reject_total_by_reason": dict(getattr(self, "hit_reject_counts", {}) or {}),
                "visual_point_result_total_by_reason": dict(getattr(self, "visual_point_result_counts", {}) or {}),
                "visual_point_stage_total": dict(getattr(self, "visual_point_stage_counts", {}) or {}),
                "visual_point_stage_by_camera": {
                    str(cam): dict(counts)
                    for cam, counts in (getattr(self, "visual_point_stage_by_camera", {}) or {}).items()
                },
                "visual_point_active_track_keys": int(len(getattr(self, "visual_point_last_by_key", {}) or {})),
                "visual_point_pending_human": int(len(getattr(self, "visual_point_pending_segments", []) or [])),
                "visual_point_late_human_retry_ms": float(getattr(self, "visual_point_hit_late_human_retry_ms", 0.0) or 0.0),
                "latest_visual_point_stage": dict(getattr(self, "latest_visual_point_stage", {}) or {}),
                "latest_visual_point_reject": dict(getattr(self, "latest_visual_point_reject", {}) or {}),
                "latest_visual_point_candidate": dict(getattr(self, "latest_visual_point_candidate", {}) or {}),
                "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
                "trajectory_only_final_enabled": bool(self._trajectory_only_final_enabled()),
                "evidence_gated_final_enabled": bool(self._evidence_gated_final_enabled()),
                "evidence_gated_config": {
                    "require_trigger": bool(getattr(self, "evidence_gated_require_trigger", True)),
                    "require_person_sync": bool(getattr(self, "evidence_gated_require_person_sync", True)),
                    "require_mask": bool(getattr(self, "evidence_gated_require_mask", True)),
                    "require_mask_polygon": bool(getattr(self, "evidence_gated_require_mask_polygon", True)),
                    "max_person_frame_delta_sync": int(getattr(self, "evidence_gated_max_person_frame_delta_sync", 0)),
                    "mask_max_age_ms": float(getattr(self, "evidence_gated_mask_max_age_ms", 0.0)),
                    "mask_max_sync_delta": int(getattr(self, "evidence_gated_mask_max_sync_delta", 0)),
                    "mask_status_json": str(getattr(self, "evidence_gated_mask_status_json", "")),
                    "mask_json_template": str(getattr(self, "evidence_gated_mask_json_template", "")),
                },
                "trajectory_only_thresholds": {
                    "min_track_points": int(getattr(self, "trajectory_only_min_track_points", 0)),
                    "min_track_path_len_px": float(getattr(self, "trajectory_only_min_track_path_len_px", 0.0)),
                    "min_duration_ms": float(getattr(self, "trajectory_only_min_duration_ms", 0.0)),
                    "min_straightness": float(getattr(self, "trajectory_only_min_straightness", 0.0)),
                    "max_line_error_px": float(getattr(self, "trajectory_only_max_line_error_px", 0.0)),
                    "min_speed_px_per_ms": float(getattr(self, "trajectory_only_min_speed_px_per_ms", 0.0)),
                    "min_approach_len_px": float(getattr(self, "trajectory_only_min_approach_len_px", 0.0)),
                    "min_point_segment_len_px": float(getattr(self, "trajectory_only_min_point_segment_len_px", 0.0)),
                    "allow_unknown_line_error": bool(getattr(self, "trajectory_only_allow_unknown_line_error", True)),
                    "min_score": float(getattr(self, "trajectory_only_min_score", 0.0)),
                },
                "pending_impacts": int(len(getattr(self, "pending_impacts", []) or [])),
                "history_depth_by_camera": {str(c): int(len(self.history.get(c, []))) for c in self.cam_ids},
                "debug_outputs": {
                    "hit_debug_all_jsonl": str(getattr(self, "debug_all_path", "")),
                    "hit_reject_debug_all_jsonl": str(getattr(self, "reject_debug_all_path", "")),
                },
            }
            path = Path(self.status_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[hit_judge][WARN] status write failed: {exc}", flush=True)

    def _poll_control_json(self, force: bool = False) -> None:
        path_s = str(getattr(self, "control_path", "") or "")
        if not path_s:
            return
        try:
            path = Path(path_s)
            if not path.exists():
                if force:
                    self.control_error = ""
                return
            mtime = float(path.stat().st_mtime)
            if not force and mtime <= float(getattr(self, "control_mtime", 0.0) or 0.0):
                return
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(obj, dict):
                raise ValueError("control json root is not an object")
            self.control_mtime = mtime
            self.control_seq = _safe_int(obj.get("seq"), int(getattr(self, "control_seq", 0) or 0))
            if "hit_detection_enabled" in obj:
                self.hit_detection_enabled = _as_bool(obj.get("hit_detection_enabled"), True)
            elif "enabled" in obj:
                self.hit_detection_enabled = _as_bool(obj.get("enabled"), True)
            if "hit_candidate_output_enabled" in obj:
                self.hit_candidate_output_enabled = _as_bool(obj.get("hit_candidate_output_enabled"), bool(getattr(self, "hit_detection_enabled", True)))
            else:
                self.hit_candidate_output_enabled = bool(getattr(self, "hit_detection_enabled", True))
            if "strong_gate_enabled" in obj:
                self.strong_gate_enabled = _as_bool(obj.get("strong_gate_enabled"), True)
            if "strong_reasoning_enabled" in obj:
                self.strong_reasoning_enabled = _as_bool(obj.get("strong_reasoning_enabled"), True)
            if "strong_reasoning_shadow_only" in obj:
                self.strong_reasoning_shadow_only = _as_bool(obj.get("strong_reasoning_shadow_only"), True)
            if "final_hit_authority" in obj:
                self.final_hit_authority = self._normalize_final_hit_authority(obj.get("final_hit_authority"))
            if str(getattr(self, "final_hit_authority", "strong_gate")) == "strong_reasoning":
                self.strong_reasoning_shadow_only = False
            if "pseudo_hit_enable" in obj:
                self.pseudo_hit_enable = _as_bool(obj.get("pseudo_hit_enable"), True)
            if "projected_mask_final_mode" in obj:
                self.projected_mask_final_mode = self._normalize_projected_mask_final_mode(obj.get("projected_mask_final_mode"))
            if "projected_mask_require_terminal_or_turn" in obj:
                self.projected_mask_require_terminal_or_turn = _as_bool(obj.get("projected_mask_require_terminal_or_turn"), True)
            if "strong_reasoning_track_project_enable" in obj:
                self.strong_reasoning_track_project_enable = _as_bool(obj.get("strong_reasoning_track_project_enable"), True)
            if "human_noise_filter_enable" in obj:
                self.human_noise_filter_enable = _as_bool(obj.get("human_noise_filter_enable"), True)
            if "human_noise_filter_mode" in obj:
                self.human_noise_filter_mode = self._normalize_human_noise_filter_mode(obj.get("human_noise_filter_mode"))
            if "human_noise_require_inbound_for_low_quality" in obj:
                self.human_noise_require_inbound_for_low_quality = _as_bool(obj.get("human_noise_require_inbound_for_low_quality"), True)
            if "human_noise_track_start_bbox_filter_enable" in obj:
                self.human_noise_track_start_bbox_filter_enable = _as_bool(obj.get("human_noise_track_start_bbox_filter_enable"), True)
            if "human_noise_track_start_bbox_policy" in obj:
                self.human_noise_track_start_bbox_policy = self._normalize_track_start_bbox_policy(obj.get("human_noise_track_start_bbox_policy"))
            for attr, key in (
                ("human_noise_soft_threshold", "human_noise_soft_threshold"),
                ("human_noise_hard_threshold", "human_noise_hard_threshold"),
                ("human_noise_min_pre_hit_path_len_px", "human_noise_min_pre_hit_path_len_px"),
                ("human_noise_short_track_max_path_len_px", "human_noise_short_track_max_path_len_px"),
                ("human_noise_short_track_max_duration_ms", "human_noise_short_track_max_duration_ms"),
                ("human_noise_track_start_bbox_expand_px", "human_noise_track_start_bbox_expand_px"),
                ("final_hit_cross_camera_dedup_ms", "final_hit_cross_camera_dedup_ms"),
                ("final_hit_same_track_episode_dedup_ms", "final_hit_same_track_episode_dedup_ms"),
                ("strong_reasoning_track_project_extend_px", "strong_reasoning_track_project_extend_px"),
                ("strong_reasoning_track_project_max_last_dist_px", "strong_reasoning_track_project_max_last_dist_px"),
            ):
                if key in obj:
                    try:
                        setattr(self, attr, float(obj.get(key)))
                    except Exception:
                        pass
            if "final_hit_cross_camera_dedup_enable" in obj:
                self.final_hit_cross_camera_dedup_enable = _as_bool(obj.get("final_hit_cross_camera_dedup_enable"), True)
            if "final_hit_same_track_episode_dedup_enable" in obj:
                self.final_hit_same_track_episode_dedup_enable = _as_bool(obj.get("final_hit_same_track_episode_dedup_enable"), True)
            for attr, key in (
                ("human_noise_min_pre_hit_outside_points", "human_noise_min_pre_hit_outside_points"),
                ("human_noise_short_track_max_points", "human_noise_short_track_max_points"),
            ):
                if key in obj:
                    try:
                        setattr(self, attr, int(obj.get(key)))
                    except Exception:
                        pass
            self.control_error = ""
        except Exception as exc:
            self.control_error = f"{type(exc).__name__}: {exc}"
            print(f"[hit_judge][WARN] control read failed: {self.control_error}", flush=True)

    def _normalize_final_hit_authority(self, value: Any) -> str:
        text = str(value or "strong_gate").strip().lower()
        if text in {"gate", "old", "legacy", "strong_gate"}:
            return "strong_gate"
        if text in {"reasoning", "new", "strong_reasoning"}:
            return "strong_reasoning"
        if text in {"both", "compare", "both_shadow_compare"}:
            return "both_shadow_compare"
        return "strong_gate"

    def _normalize_human_noise_filter_mode(self, value: Any) -> str:
        text = str(value or "soft").strip().lower()
        if text in {"off", "disable", "disabled", "none", "0"}:
            return "off"
        if text in {"shadow", "observe", "dryrun", "dry_run"}:
            return "shadow"
        if text in {"hard", "strict"}:
            return "hard"
        return "soft"

    def _normalize_track_start_bbox_policy(self, value: Any) -> str:
        text = str(value or "soft").strip().lower()
        if text in {"hard", "strict", "block", "deny"}:
            return "hard"
        if text in {"off", "disable", "disabled", "ignore", "none", "0"}:
            return "off"
        return "soft"

    def _normalize_projected_mask_final_mode(self, value: Any) -> str:
        text = str(value or "pending_only").strip().lower()
        if text in {"final", "allow", "allowed", "final_allowed", "promote"}:
            return "final_allowed"
        if text in {"shadow", "shadow_only", "observe", "dryrun", "dry_run"}:
            return "shadow_only"
        return "pending_only"

    def _strong_gate_final_output_allowed(self) -> bool:
        if not bool(getattr(self, "hit_detection_enabled", True)):
            return False
        if not bool(getattr(self, "strong_gate_enabled", True)):
            return False
        return str(getattr(self, "final_hit_authority", "strong_gate")) in {"strong_gate", "both_shadow_compare"}

    def _strong_reasoning_final_output_allowed(self) -> bool:
        if not bool(getattr(self, "hit_detection_enabled", True)):
            return False
        if not bool(getattr(self, "strong_reasoning_enabled", True)):
            return False
        return str(getattr(self, "final_hit_authority", "strong_gate")) == "strong_reasoning"

    def _handle_visual_bullet_point_event(self, cam: str, ev: BulletEvent, H: np.ndarray, age_ms: float = 0.0) -> str:
        """Project/send one real-time point and judge prev->current segment.

        This is the new high-frequency path.  The visual point is always emitted
        first for UE5.  If a previous point for the same (pair_id, camera_sn,
        bullet_id) exists, that segment is immediately tested against the current
        human/mask state instead of waiting for 40ms track_probe events.
        """
        now = time.time()
        curr_mvs = _apply_H(H, float(ev.event_x), float(ev.event_y))
        key = (str(cam), str(ev.camera_sn), int(ev.bullet_id))
        prev_state = self.visual_point_last_by_key.get(key)
        self._record_visual_point_stage(
            "point_projected",
            cam=cam,
            ev=ev,
            detail={"has_prev_point": bool(prev_state is not None)},
        )

        # Always send the real-time point to UE5/logs.  The message includes
        # point_delta_ms and segment_len_mvs_px when a previous point exists.
        self._emit_visual_bullet_point(
            cam,
            ev,
            H=H,
            age_ms=age_ms,
            curr_mvs=curr_mvs,
            prev_state=prev_state,
            receive_wall=now,
        )
        self._semantic_process_point(cam, ev, curr_mvs, receive_wall=now)

        point_index = int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1)
        result = "point_only"
        segment_ok = False
        if prev_state is not None:
            # Ignore out-of-order samples for judgement, but keep UE logs visible.
            if point_index >= 0 and prev_state.point_index >= 0 and point_index <= prev_state.point_index:
                result = "point_out_of_order"
                self._record_visual_point_stage("point_out_of_order", cam=cam, ev=ev)
            elif int(ev.timestamp_us) <= int(prev_state.point_ts_us):
                result = "point_ts_out_of_order"
                self._record_visual_point_stage("point_ts_out_of_order", cam=cam, ev=ev)
            else:
                seg_len = math.hypot(float(curr_mvs[0]) - float(prev_state.mvs_point[0]), float(curr_mvs[1]) - float(prev_state.mvs_point[1]))
                if seg_len >= float(getattr(self, "visual_point_hit_min_segment_len_px", 1.0)):
                    segment_ok = True
                    self._record_visual_point_stage(
                        "segment_attempt",
                        cam=cam,
                        ev=ev,
                        detail={"segment_len_mvs_px": round(float(seg_len), 3)},
                    )
                    result = self._judge_visual_point_segment(cam, ev, prev_state, curr_mvs, age_ms=age_ms)
                else:
                    result = "point_segment_too_short"
                    self._record_visual_point_stage(
                        "segment_too_short",
                        cam=cam,
                        ev=ev,
                        detail={"segment_len_mvs_px": round(float(seg_len), 3)},
                    )
        else:
            self._record_visual_point_stage("waiting_prev_point", cam=cam, ev=ev)

        # Update the cache after judging so the current point becomes the next prev.
        if prev_state is None or point_index < 0 or prev_state.point_index < 0 or point_index >= prev_state.point_index:
            next_first_ts_us = int(getattr(prev_state, "first_point_ts_us", 0) or getattr(prev_state, "point_ts_us", int(ev.timestamp_us))) if prev_state is not None else int(ev.timestamp_us)
            next_track_points = int(getattr(prev_state, "track_point_count", 0) or 0) + 1 if prev_state is not None else 1
            next_track_path = float(getattr(prev_state, "track_path_len_px", 0.0) or 0.0)
            recent_points = list(getattr(prev_state, "recent_mvs_points", []) or []) if prev_state is not None else []
            if prev_state is not None:
                next_track_path += math.hypot(float(curr_mvs[0]) - float(prev_state.mvs_point[0]), float(curr_mvs[1]) - float(prev_state.mvs_point[1]))
            recent_points.append((int(point_index), float(curr_mvs[0]), float(curr_mvs[1]), int(ev.timestamp_us)))
            max_recent = max(8, int(getattr(self, "visual_point_track_cache_max_points", 192) or 192))
            if len(recent_points) > max_recent:
                recent_points = recent_points[-max_recent:]
            self.visual_point_last_by_key[key] = LiveBulletPointState(
                cam=str(cam),
                camera_sn=str(ev.camera_sn),
                bullet_id=int(ev.bullet_id),
                point_index=int(point_index),
                point_ts_us=int(ev.timestamp_us),
                mvs_point=(float(curr_mvs[0]), float(curr_mvs[1])),
                raw_point=(float(ev.event_x), float(ev.event_y)),
                first_point_ts_us=int(next_first_ts_us),
                track_point_count=int(next_track_points),
                track_path_len_px=float(next_track_path),
                recent_mvs_points=recent_points,
                receive_wall=float(now),
            )
        if (not segment_ok) and result not in {"point_emitted", "point_pending_human"}:
            self._record_visual_point_result(str(result), cam=cam, ev=ev)
        self._record_visual_point_stage(f"handled:{result}", cam=cam, ev=ev)
        return result if segment_ok else result

    def _judge_visual_point_segment(
        self,
        cam: str,
        ev: BulletEvent,
        prev_state: LiveBulletPointState,
        curr_mvs: Tuple[float, float],
        age_ms: float = 0.0,
        allow_late_human_pending: bool = True,
        log_no_human_reject: bool = True,
    ) -> str:
        """Run hit judgement on one high-frequency segment: prev MVS -> curr MVS."""
        if not bool(getattr(self, "visual_point_hit_judge_enable", True)):
            self._record_visual_point_result("point_hit_judge_disabled", cam=cam, ev=ev)
            return "point_hit_judge_disabled"
        if not bool(getattr(self, "hit_detection_enabled", True)):
            self._record_visual_point_result("hit_detection_disabled", cam=cam, ev=ev)
            return "hit_detection_disabled"

        prev_pt = (float(prev_state.mvs_point[0]), float(prev_state.mvs_point[1]))
        curr_pt = (float(curr_mvs[0]), float(curr_mvs[1]))
        seg_len = math.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
        dt_point_ms = max(0.0, (int(ev.timestamp_us) - int(prev_state.point_ts_us)) / 1000.0)
        point_index = int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1)

        total_track_points = max(int(getattr(ev, "track_point_count", 0) or 0), point_index + 1 if point_index >= 0 else 0)
        cached_track_points = int(getattr(prev_state, "track_point_count", 1) or 1) + 1
        cached_track_path = float(getattr(prev_state, "track_path_len_px", 0.0) or 0.0) + float(seg_len)
        first_point_ts_us = int(getattr(prev_state, "first_point_ts_us", 0) or getattr(prev_state, "point_ts_us", int(ev.timestamp_us)))
        cached_track_duration_ms = max(0.0, (int(ev.timestamp_us) - int(first_point_ts_us)) / 1000.0)
        total_track_points = max(int(total_track_points), int(cached_track_points))
        total_track_path = max(float(getattr(ev, "track_path_len_px", 0.0) or 0.0), float(cached_track_path))
        if total_track_path <= 0.0:
            total_track_path = seg_len
        if total_track_points < int(getattr(self, "visual_point_hit_min_track_points", 2)):
            self._emit_visual_point_reject(
                cam, ev, "point_track_points_too_small", "track_points_below_visual_point_gate",
                geometry={"track_point_count": int(total_track_points), "point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                age_ms=age_ms,
            )
            return "point_track_points_too_small"
        if total_track_path < float(getattr(self, "visual_point_hit_min_track_path_len_px", 0.0)):
            self._emit_visual_point_reject(
                cam, ev, "point_track_path_too_short", "track_path_below_visual_point_gate",
                geometry={"track_path_len_px": round(float(total_track_path), 3), "point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                age_ms=age_ms,
            )
            return "point_track_path_too_short"

        track_metrics = {
            "track_point_count": int(total_track_points),
            "track_path_len_px": float(total_track_path),
            "track_duration_ms": max(float(dt_point_ms), float(cached_track_duration_ms), float(getattr(ev, "track_duration_ms", 0.0) or 0.0)),
            "track_displacement_px": float(getattr(ev, "track_displacement_px", 0.0) or seg_len),
            "track_straightness": float(getattr(ev, "track_straightness", 1.0) or 1.0),
            "track_line_error_px": float(getattr(ev, "track_line_error_px", 999.0) if getattr(ev, "track_line_error_px", 999.0) is not None else 999.0),
            "approach_len_px": float(seg_len),
            "avg_speed_px_per_ms": float(getattr(ev, "avg_speed_px_per_ms", 0.0) or ((seg_len / max(dt_point_ms, 1e-3)) if dt_point_ms > 0.0 else 0.0)),
        }
        if self._trajectory_only_final_enabled():
            return self._emit_trajectory_only_hit(
                cam,
                ev,
                curr_pt,
                prev_pt,
                track_metrics,
                judge_source="visual_point_segment",
                use_segment=True,
                min_segment_len_px=float(getattr(self, "trajectory_only_min_point_segment_len_px", self.visual_point_hit_min_segment_len_px)),
                point_index=int(point_index),
                prev_point_index=int(prev_state.point_index),
                point_delta_ms=float(dt_point_ms),
                segment_len_mvs_px=float(seg_len),
                age_ms=age_ms,
            )

        if self._evidence_gated_final_enabled() and not self._strong_reasoning_final_output_allowed():
            quality_ok, quality_reason, quality_detail = self._trajectory_only_quality(
                track_metrics,
                min_segment_len_px=float(getattr(self, "trajectory_only_min_point_segment_len_px", self.visual_point_hit_min_segment_len_px)),
            )
            if not quality_ok:
                self._emit_visual_point_reject(
                    cam,
                    ev,
                    f"point_{quality_reason}",
                    str(quality_reason),
                    geometry={
                        "judge_source": "visual_point_segment",
                        "evidence_gate": "trajectory_quality",
                        "event_point_mvs": [round(float(curr_pt[0]), 3), round(float(curr_pt[1]), 3)],
                        "approach_point_mvs": [round(float(prev_pt[0]), 3), round(float(prev_pt[1]), 3)],
                        "point_index": int(point_index),
                        "prev_point_index": int(prev_state.point_index),
                        "point_delta_ms": round(float(dt_point_ms), 3),
                        "segment_len_mvs_px": round(float(seg_len), 3),
                        **quality_detail,
                    },
                    age_ms=age_ms,
                )
                return f"point_{quality_reason}"

        period: Optional[Dict[str, Any]] = None
        strategy_used = "visual_point_nearest_time"
        selection_info: Dict[str, Any] = {}
        if self.impact_sync_enable:
            period = self._impact_period_for_event(cam, int(ev.timestamp_us))
            if period is None:
                if self._strong_reasoning_final_output_allowed():
                    entry = self._nearest_person_entry(cam, int(ev.timestamp_us))
                    if entry is not None:
                        entries = [entry]
                        strategy_used = "strong_reasoning_no_trigger_nearest_time"
                        selection_info = {
                            "target_person_sync_index": int(entry.sync_index),
                            "used_person_sync_index": int(entry.sync_index),
                            "human_fallback_used": True,
                            "human_fallback_reason": "no_trigger_period_for_point_ts",
                            "person_frame_delta_sync": 0,
                            "trigger_evidence_degraded": True,
                            "trigger_evidence_degrade_reason": "no_trigger_period_for_point_ts",
                        }
                    else:
                        self._emit_visual_point_reject(
                            cam, ev, "point_no_trigger", "no_trigger_period_for_point_ts",
                            geometry={"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                            age_ms=age_ms,
                        )
                        return "point_no_trigger"
                else:
                    self._emit_visual_point_reject(
                        cam, ev, "point_no_trigger", "no_trigger_period_for_point_ts",
                        geometry={"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                        age_ms=age_ms,
                    )
                    return "point_no_trigger"
            if period is not None and not bool(period.get("trigger_period_ok", False)) and not self.emit_candidate_on_bad_trigger_period and not self._strong_reasoning_final_output_allowed():
                self._emit_visual_point_reject(
                    cam, ev, "point_bad_trigger", "bad_trigger_period_for_point_ts",
                    period=period,
                    geometry={"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                    age_ms=age_ms,
                )
                return "point_bad_trigger"
            if period is not None:
                entries, strategy_used, should_wait, selection_info = self._entries_for_impact(cam, period, age_ms=age_ms, force_final=True)
                if not bool(period.get("trigger_period_ok", False)) and self._strong_reasoning_final_output_allowed():
                    selection_info = dict(selection_info or {})
                    selection_info["trigger_evidence_degraded"] = True
                    selection_info["trigger_evidence_degrade_reason"] = "bad_trigger_period_for_point_ts"
            if not entries:
                geom = {"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)}
                if allow_late_human_pending:
                    return self._register_pending_visual_point_segment(
                        cam, ev, prev_state, curr_mvs, age_ms,
                        period=period, selection=selection_info, geometry=geom,
                    )
                if log_no_human_reject:
                    self._emit_visual_point_reject(
                        cam, ev, "point_no_human", "no_human_result_near_point_ts",
                        period=period, selection=selection_info, geometry=geom, age_ms=age_ms,
                    )
                return "point_no_human"
            try:
                frame_delta_i = int(selection_info.get("person_frame_delta_sync", 999999))
            except Exception:
                frame_delta_i = 999999
            if bool(selection_info.get("human_fallback_used", False)) and not bool(getattr(self, "visual_point_hit_allow_human_fallback", True)):
                self._emit_visual_point_reject(
                    cam, ev, "point_human_fallback_rejected", "visual_point_human_fallback_disabled",
                    period=period, selection=selection_info,
                    geometry={"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                    age_ms=age_ms,
                )
                return "point_human_fallback_rejected"
            if abs(frame_delta_i) > int(getattr(self, "visual_point_hit_max_person_frame_delta_sync", 2)):
                self._emit_visual_point_reject(
                    cam, ev, "point_person_frame_delta_too_large", "visual_point_person_frame_delta_gate",
                    period=period, selection=selection_info,
                    geometry={"person_frame_delta_sync": int(frame_delta_i), "point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                    age_ms=age_ms,
                )
                return "point_person_frame_delta_too_large"
        else:
            entry = self._nearest_person_entry(cam, int(ev.timestamp_us))
            if entry is None:
                if allow_late_human_pending:
                    return self._register_pending_visual_point_segment(
                        cam, ev, prev_state, curr_mvs, age_ms,
                        geometry={"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                    )
                if log_no_human_reject:
                    self._emit_visual_point_reject(
                        cam, ev, "point_no_human", "nearest_time_no_human",
                        geometry={"point_delta_ms": round(dt_point_ms, 3), "segment_len_mvs_px": round(seg_len, 3)},
                        age_ms=age_ms,
                    )
                return "point_no_human"
            entries = [entry]
            selection_info = {
                "target_person_sync_index": int(entry.sync_index),
                "used_person_sync_index": int(entry.sync_index),
                "human_fallback_used": False,
                "human_fallback_reason": "",
                "person_frame_delta_sync": 0,
            }

        track_metrics = {
            "track_point_count": int(total_track_points),
            "track_path_len_px": float(total_track_path),
            "track_duration_ms": max(float(dt_point_ms), float(getattr(ev, "track_duration_ms", 0.0) or 0.0)),
            "track_displacement_px": float(seg_len),
            "track_straightness": float(getattr(ev, "track_straightness", 1.0) or 1.0),
            "track_line_error_px": float(getattr(ev, "track_line_error_px", 0.0) or 0.0),
            "approach_len_px": float(seg_len),
            "avg_speed_px_per_ms": (seg_len / max(dt_point_ms, 1e-3)) if dt_point_ms > 0.0 else float(getattr(ev, "avg_speed_px_per_ms", 0.0) or 0.0),
        }
        track_points_mvs = list(getattr(prev_state, "recent_mvs_points", []) or [])
        curr_track_point = (int(point_index), float(curr_pt[0]), float(curr_pt[1]), int(ev.timestamp_us))
        if not track_points_mvs or int(track_points_mvs[-1][0]) != int(point_index) or int(track_points_mvs[-1][3]) != int(ev.timestamp_us):
            track_points_mvs.append(curr_track_point)
        max_recent = max(8, int(getattr(self, "visual_point_track_cache_max_points", 192) or 192))
        if len(track_points_mvs) > max_recent:
            track_points_mvs = track_points_mvs[-max_recent:]
        track_metrics["recent_mvs_points"] = list(track_points_mvs)

        geometry_summary: Dict[str, Any] = {
            "judge_source": "visual_point_segment",
            "point_index": int(point_index),
            "prev_point_index": int(prev_state.point_index),
            "point_delta_ms": round(float(dt_point_ms), 3),
            "segment_len_mvs_px": round(float(seg_len), 3),
            "segment_start_mvs": [round(float(prev_pt[0]), 3), round(float(prev_pt[1]), 3)],
            "segment_end_mvs": [round(float(curr_pt[0]), 3), round(float(curr_pt[1]), 3)],
            "geometry_tested_persons": 0,
            "geometry_best_method": None,
            "geometry_best_score": 0.0,
            "evidence_level": "none",
        }

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

                if bool(getattr(self, "visual_point_hit_body_motion_reject_enable", False)) and not self._strong_reasoning_final_output_allowed():
                    body_reject_detail = self._person_body_motion_reject_detail(person, curr_pt, prev_pt, True, seg_len)
                    if body_reject_detail is not None:
                        if float(geometry_summary.get("geometry_best_score", 0.0)) <= 0.0:
                            geometry_summary["geometry_best_method"] = body_reject_detail.get("body_motion_reject_reason")
                            geometry_summary["evidence_level"] = "reject"
                        continue

                method = None
                space = 0.0
                evidence_level = "none"
                track_mask_detail: Dict[str, Any] = {}
                if self.impact_use_mask:
                    if bool(getattr(self, "strong_reasoning_track_mask_enable", True)):
                        method, space, evidence_level, track_mask_detail = _person_track_mask_hit(
                            person,
                            track_points_mvs,
                            curr_pt,
                            prev_pt,
                            True,
                            project_enable=bool(getattr(self, "strong_reasoning_track_project_enable", True)),
                            project_extend_px=float(getattr(self, "strong_reasoning_track_project_extend_px", 100.0) or 100.0),
                            project_max_last_dist_px=float(getattr(self, "strong_reasoning_track_project_max_last_dist_px", 40.0) or 40.0),
                        )
                    else:
                        method, space, evidence_level = _person_mask_hit(person, curr_pt, prev_pt, True)
                    if method in {
                        "point_in_mask",
                        "segment_intersect_mask",
                        "track_point_inside_mask",
                        "track_polyline_cross_mask",
                        "track_projected_inbound_mask",
                    }:
                        body_detail = self._person_body_motion_reject_detail(person, curr_pt, prev_pt, True, seg_len)
                        external_ok = body_detail is None or self._strong_reasoning_final_output_allowed()
                        if external_ok:
                            if method in {"point_in_mask", "track_point_inside_mask"}:
                                space = max(float(space), 0.78)
                            elif method in {"segment_intersect_mask", "track_polyline_cross_mask"}:
                                space = max(float(space), 0.68)
                            else:
                                space = max(float(space), 0.58)
                            if str(method) == "track_projected_inbound_mask":
                                evidence_level = "projected_track_mask_entry"
                            else:
                                evidence_level = "strong_external_mask_entry"

                if method is None and (bool(getattr(self, "visual_point_hit_allow_bbox", False)) or not bool(self.final_hit_mask_only)):
                    rect = _expand_bbox_xyxy(box, self.bbox_expand_px, entry.frame_w, entry.frame_h)
                    if _point_in_rect(curr_pt, rect):
                        method = MATCH_POINT_IN_BBOX
                        space = 0.50
                        evidence_level = "medium"
                    elif _segment_intersects_rect(prev_pt, curr_pt, rect):
                        method = MATCH_SEGMENT_INTERSECT
                        space = 0.40
                        evidence_level = "weak"
                    else:
                        d_rect = _dist_to_rect(curr_pt, rect)
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
                    if track_mask_detail:
                        geometry_summary["track_mask"] = dict(track_mask_detail)

                if period is not None:
                    dt_ms = abs(int(entry.event_ts_us or 0) - int(ev.timestamp_us)) / 1000.0 if entry.event_ts_us else abs(float(period.get("offset_ms", 0.0)))
                    time_score = max(0.0, 0.24 * (1.0 - min(dt_ms, self.expected_period_us / 1000.0) / max(1.0, self.expected_period_us / 1000.0)))
                else:
                    dt_ms = abs(int(entry.event_ts_us or 0) - int(ev.timestamp_us)) / 1000.0
                    time_score = max(0.0, 0.22 * (1.0 - dt_ms / max(1.0, self.max_dt_ms)))
                speed_score = min(0.08, max(0.0, float(track_metrics.get("avg_speed_px_per_ms", 0.0)) / 3.0 * 0.08))
                loss_penalty = 0.0
                if bool(selection_info.get("human_fallback_used", False)):
                    try:
                        delta_sync = int(entry.sync_index) - int(selection_info.get("target_person_sync_index", entry.sync_index))
                    except Exception:
                        delta_sync = 0
                    loss_penalty -= self.fallback_penalty_base + abs(delta_sync) * self.fallback_penalty_per_sync
                if period is not None and bool(period.get("estimated_next", False)):
                    loss_penalty -= self.estimated_next_trigger_penalty
                if period is not None and not bool(period.get("trigger_period_ok", True)):
                    loss_penalty -= 0.25

                score = ScoreBreakdown(
                    time_score=float(time_score),
                    space_score=float(space),
                    event_score=float(getattr(self, "visual_point_hit_event_score", 0.12)),
                    speed_score=float(speed_score),
                    loss_penalty=float(loss_penalty),
                )
                try:
                    raw_box = list(box)
                    x1, y1, x2, y2 = [float(v) for v in raw_box[:4]]
                    if x2 <= x1 or y2 <= y1:
                        x2, y2 = x1 + float(raw_box[2]), y1 + float(raw_box[3])
                    bbox = (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))
                except Exception:
                    bbox = (0, 0, 0, 0)
                cand = HitCandidate(
                    camera_sn=str(ev.camera_sn),
                    camera_alias=cam,
                    bullet_id=int(ev.bullet_id),
                    seq_id=int(ev.seq_id),
                    stable_id=int(person.get("stable_id", person.get("person_id", 0)) or 0),
                    event_ts=int(ev.timestamp_us),
                    person_ts=int(entry.event_ts_us or 0),
                    dt_ms=float(dt_ms),
                    event_type=str(EVENT_POINT),
                    terminate_reason=str(REASON_VISUAL_POINT),
                    turn_angle_deg=0.0,
                    event_point=(float(curr_pt[0]), float(curr_pt[1])),
                    approach_point=(float(prev_pt[0]), float(prev_pt[1])),
                    approach_valid=True,
                    match_method=str(method),
                    track_point_count=int(track_metrics.get("track_point_count", 0) or 0),
                    track_path_len_px=float(track_metrics.get("track_path_len_px", 0.0) or 0.0),
                    track_duration_ms=float(track_metrics.get("track_duration_ms", 0.0) or 0.0),
                    track_displacement_px=float(track_metrics.get("track_displacement_px", 0.0) or 0.0),
                    track_straightness=float(track_metrics.get("track_straightness", 0.0) or 0.0),
                    track_line_error_px=float(track_metrics.get("track_line_error_px", 0.0) or 0.0),
                    person_bbox=tuple(int(v) for v in bbox),
                    person_centroid=tuple(person.get("foot_pixel") or person.get("centroid") or (0.0, 0.0)),
                    person_confirmed=True,
                    person_held=False,
                    score_breakdown=score,
                )
                required_score = float(getattr(self, "visual_point_hit_min_score", self.min_score))
                if cand.total_score >= required_score:
                    meta = {
                        "evidence_level": evidence_level,
                        "geometry_mode": str(method),
                        "used_person_sync_index": int(entry.sync_index),
                        "person_frame_delta_sync": int(selection_info.get("person_frame_delta_sync", 0) or 0),
                        "required_score": round(float(required_score), 3),
                        "external_mask_entry_boost": bool(evidence_level == "strong_external_mask_entry"),
                        "track_mask_detail": dict(track_mask_detail),
                    }
                    candidates_meta.append((cand, entry, meta))

        candidates_meta.sort(key=lambda x: x[0].total_score, reverse=True)
        if not candidates_meta:
            self._emit_visual_point_reject(
                cam, ev, "point_no_candidate", "no_person_geometry_match_for_point_segment",
                period=period, selection=selection_info, geometry=geometry_summary, age_ms=age_ms,
            )
            return "point_no_candidate"

        best, best_entry, best_meta = candidates_meta[0]
        track_mask_detail = dict(best_meta.get("track_mask_detail", {}) or {})
        override_hit_xy = self._coerce_xy(track_mask_detail.get("track_mask_hit_point")) or curr_pt
        override_approach_xy = (
            self._coerce_xy(track_mask_detail.get("track_mask_segment_start"))
            or prev_pt
        )
        pseudo_mask_override = {
            "person_id": int(best.stable_id),
            "mask_method": str(best.match_method),
            "mask_space": float(best.score_breakdown.space_score),
            "mask_evidence": str(best_meta.get("evidence_level", "")),
            "hit_point_mvs": [float(override_hit_xy[0]), float(override_hit_xy[1])],
            "approach_point_mvs": [float(override_approach_xy[0]), float(override_approach_xy[1])],
            "track_mask_detail": track_mask_detail,
        }
        pseudo_emitted = self._try_emit_pseudo_hit_for_event(
            cam,
            ev,
            override_hit_xy,
            override_approach_xy,
            True,
            [best_entry],
            period,
            selection_info,
            strategy_used,
            track_metrics,
            age_ms,
            "visual_point_stable_track",
            mask_contact_override=pseudo_mask_override,
        )
        if self._strong_reasoning_final_output_allowed():
            if pseudo_emitted > 0:
                self._record_visual_point_result("point_pseudo_hit_emitted", cam=cam, ev=ev)
                self.latest_visual_point_candidate = {
                    "camera": str(cam),
                    "bullet_id": int(getattr(ev, "bullet_id", 0) or 0),
                    "seq_id": int(getattr(ev, "seq_id", 0) or 0),
                    "point_index": int(point_index),
                    "result": "point_pseudo_hit_emitted",
                    "judge_chain": "strong_reasoning",
                    "match_method": str(best.match_method),
                    "track_path_len_px": round(float(track_metrics.get("track_path_len_px", 0.0) or 0.0), 3),
                    "track_point_count": int(track_metrics.get("track_point_count", 0) or 0),
                    "updated_wall_time": time.time(),
                }
                return "point_emitted"
            self._emit_visual_point_reject(
                cam,
                ev,
                "point_pseudo_hit_not_emitted",
                "strong_reasoning_no_pseudo_hit",
                period=period,
                selection=selection_info,
                geometry={**geometry_summary, **best_meta},
                age_ms=age_ms,
            )
            return "point_pseudo_hit_not_emitted"
        evidence_gate_detail: Dict[str, Any] = {}
        if self._evidence_gated_final_enabled():
            gate_geometry = {**geometry_summary, **best_meta}
            gate_ok, gate_reason, evidence_gate_detail = self._final_evidence_gate_ok(
                cam, best_entry, period, selection_info, gate_geometry
            )
            if not gate_ok:
                self._emit_visual_point_reject(
                    cam,
                    ev,
                    "point_evidence_gate_rejected",
                    str(gate_reason),
                    period=period,
                    selection=selection_info,
                    geometry={**geometry_summary, "evidence_gate": evidence_gate_detail},
                    age_ms=age_ms,
                )
                return "point_evidence_gate_rejected"
        if bool(getattr(self, "visual_point_hit_emit_once_per_bullet", True)):
            if not self._mark_visual_point_hit_emitted(cam, int(best.bullet_id), int(best.stable_id)):
                self._emit_visual_point_reject(
                    cam, ev, "point_duplicate_hit_ignored", "visual_point_hit_already_emitted_for_bullet",
                    period=period, selection=selection_info, geometry=geometry_summary, age_ms=age_ms,
                )
                return "point_duplicate_hit_ignored"
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
            True,
            float(seg_len),
        )
        d.update({
            "source": "hit_judge_server",
            "judge_source": "visual_point_segment",
            "realtime_point_hit": True,
            "event_type": str(EVENT_POINT),
            "terminate_reason": str(REASON_VISUAL_POINT),
            "point_index": int(point_index),
            "prev_point_index": int(prev_state.point_index),
            "point_delta_ms": round(float(dt_point_ms), 3),
            "segment_len_mvs_px": round(float(seg_len), 3),
            "segment_start_mvs": [round(float(prev_pt[0]), 3), round(float(prev_pt[1]), 3)],
            "segment_end_mvs": [round(float(curr_pt[0]), 3), round(float(curr_pt[1]), 3)],
            "visual_point_hit_min_score": round(float(getattr(self, "visual_point_hit_min_score", self.min_score)), 3),
            "visual_point_hit_allow_bbox": bool(getattr(self, "visual_point_hit_allow_bbox", False)),
        })
        if self._evidence_gated_final_enabled():
            d.update({
                "judge_mode": "evidence_gated",
                "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
                "trajectory_evidence_used": True,
                "trigger_evidence_used": bool(period is not None),
                "person_sync_evidence_used": True,
                "mask_evidence_used": True,
                "evidence_gate": evidence_gate_detail,
            })
        self._write_candidate_dict(cam, d)
        self._record_visual_point_result("point_emitted", cam=cam, ev=ev)
        self.latest_visual_point_candidate = {
            "camera": str(cam),
            "camera_sn": str(ev.camera_sn),
            "bullet_id": int(best.bullet_id),
            "stable_id": int(best.stable_id),
            "seq_id": int(ev.seq_id),
            "point_index": int(point_index),
            "total_score": round(float(best.total_score), 3),
            "match_method": str(best.match_method),
            "wall_time": time.time(),
        }
        self._emit_debug(cam, ev, "emitted_candidate", None, period=period, selection=selection_info, geometry=geometry_summary, candidate=dict(d), age_ms=age_ms)
        print(f"[hit_judge][{cam}] point-segment candidate bullet={best.bullet_id} person={best.stable_id} score={best.total_score:.3f} dt={best.dt_ms:.1f}ms point_dt={dt_point_ms:.1f}ms seg={seg_len:.1f}px method={best.match_method} strategy={strategy_used}", flush=True)
        return "point_emitted"

    def _emit_visual_bullet_event(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        track_points_mvs: List[List[float]],
        age_ms: float = 0.0,
    ) -> None:
        if not bool(getattr(self, "visual_stream_enable", True)):
            return
        key = (str(ev.camera_sn), int(ev.bullet_id), int(ev.seq_id))
        if not self._mark_visual_seen(key):
            return
        display_w, display_h = self._display_size_for_cam(str(cam))
        marker_type = "event"
        if str(ev.event_type) == EVENT_TURN:
            marker_type = "turn"
        elif str(ev.event_type) == EVENT_TERMINATE:
            reason_s = str(getattr(ev, "terminate_reason", "") or "")
            marker_type = "track_probe" if reason_s == "track_probe" else "end"

        msg = {
            "type": "overlay_bullet_event",
            "schema_version": 2,
            "sync_run_id": str(self.hit_cfg.get("sync_run_id", "real_run_001") or "real_run_001"),
            "pair_id": str(cam),
            "camera_sn": str(ev.camera_sn),
            "camera_alias": str(ev.camera_alias or cam),
            "bullet_id": int(ev.bullet_id),
            "bullet_seq_id": int(ev.seq_id),
            "bullet_ts_us": int(ev.timestamp_us),
            "event_type": str(ev.event_type),
            "marker_type": marker_type,
            "terminate_reason": str(ev.terminate_reason),
            "turn_angle_deg": round(float(getattr(ev, "turn_angle_deg", 0.0) or 0.0), 3),
            "confidence": float(ev.confidence),
            "display_width": int(display_w),
            "display_height": int(display_h),
            "event_point_mvs": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
            "approach_point_mvs": [round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],
            "x": round(float(event_pt[0]), 3),
            "y": round(float(event_pt[1]), 3),
            "approach_x": round(float(approach_pt[0]), 3),
            "approach_y": round(float(approach_pt[1]), 3),
            "event_x_raw": round(float(ev.event_x), 3),
            "event_y_raw": round(float(ev.event_y), 3),
            "approach_x_raw": round(float(getattr(ev, "approach_x", ev.event_x)), 3),
            "approach_y_raw": round(float(getattr(ev, "approach_y", ev.event_y)), 3),
            "track_points": track_points_mvs,
            "track_points_count": int(len(track_points_mvs)),
            "track_point_count": int(getattr(ev, "track_point_count", 0) or 0),
            "track_path_len_px": round(float(getattr(ev, "track_path_len_px", 0.0) or 0.0), 3),
            "track_duration_ms": round(float(getattr(ev, "track_duration_ms", 0.0) or 0.0), 3),
            "track_straightness": round(float(getattr(ev, "track_straightness", 0.0) or 0.0), 4),
            "track_line_error_px": round(float(getattr(ev, "track_line_error_px", 999.0) or 999.0), 3),
            "avg_speed_px_per_ms": round(float(getattr(ev, "avg_speed_px_per_ms", 0.0) or 0.0), 4),
            "approach_valid": bool(getattr(ev, "approach_valid", False)),
            "approach_len_px": round(float(getattr(ev, "approach_len_px", 0.0) or 0.0), 3),
            "segment_index": int(getattr(ev, "segment_index", 0) or 0),
            "hit": False,
            "source": "visual_bullet_stream",
            "reason": "visual_bullet_track",
            "judge_age_ms": round(float(age_ms), 3),
            "wall_time": time.time(),
        }
        msg["raw_event_type"] = str(ev.event_type)
        msg["raw_marker_type"] = marker_type
        if bool(getattr(self, "semantic_path_enable", True)):
            msg["ue5_marker_suppressed"] = True
            msg["ue5_marker_suppression_reason"] = "raw_event_replaced_by_semantic_path_resolver"
            if marker_type == "end":
                msg["marker_type"] = "raw_end"
            elif marker_type == "turn":
                msg["marker_type"] = "raw_turn"
            elif marker_type == "track_probe":
                msg["marker_type"] = "raw_track_probe"
        self._write_visual_bullet_dict(str(cam), msg)
        if bool(getattr(self, "semantic_raw_event_ws_enable", False)):
            self._broadcast_overlay(msg)

        if not bool(getattr(self, "semantic_raw_marker_enable", False)):
            return

        # UE5 marker-only message.  overlay_bullet_event keeps the full debug/audit
        # payload; overlay_bullet_marker is the compact real-time trigger for UE5
        # effects such as turn sparks, end markers and track probes.
        marker_msg = {
            "type": "overlay_bullet_marker",
            "schema_version": 1,
            "sync_run_id": msg.get("sync_run_id"),
            "pair_id": str(cam),
            "camera_sn": str(ev.camera_sn),
            "camera_alias": str(ev.camera_alias or cam),
            "bullet_id": int(ev.bullet_id),
            "bullet_seq_id": int(ev.seq_id),
            "marker_type": marker_type,
            "event_type": str(ev.event_type),
            "terminate_reason": str(ev.terminate_reason),
            "turn_angle_deg": round(float(getattr(ev, "turn_angle_deg", 0.0) or 0.0), 3),
            "ts_us": int(ev.timestamp_us),
            "x": msg["x"],
            "y": msg["y"],
            "approach_x": msg["approach_x"],
            "approach_y": msg["approach_y"],
            "display_width": int(display_w),
            "display_height": int(display_h),
            "confidence": float(ev.confidence),
            "segment_index": int(getattr(ev, "segment_index", 0) or 0),
            "track_points_count": int(len(track_points_mvs)),
            "track_point_count": int(getattr(ev, "track_point_count", 0) or 0),
            "track_path_len_px": round(float(getattr(ev, "track_path_len_px", 0.0) or 0.0), 3),
            "source": "hit_judge_server",
            "reason": "ue5_bullet_marker",
            "wall_time": time.time(),
        }
        self._broadcast_overlay(marker_msg)


    def close(self) -> None:
        try:
            self._flush_semantic_terminals(force=True)
        except Exception:
            pass
        try: self.sock.close()
        except Exception: pass
        for fp in self.out_files.values():
            try: fp.close()
            except Exception: pass
        try: self.all_fp.close()
        except Exception: pass
        for chain_files in getattr(self, "hit_shadow_files", {}).values():
            for fp in chain_files.values():
                try: fp.close()
                except Exception: pass
        for fp in getattr(self, "hit_shadow_all_files", {}).values():
            try: fp.close()
            except Exception: pass
        for fp in getattr(self, "pseudo_hit_files", {}).values():
            try: fp.close()
            except Exception: pass
        try:
            if getattr(self, "pseudo_hit_all_fp", None) is not None:
                self.pseudo_hit_all_fp.close()
        except Exception: pass
        for fp in getattr(self, "visual_files", {}).values():
            try: fp.close()
            except Exception: pass
        try:
            if getattr(self, "visual_all_fp", None) is not None:
                self.visual_all_fp.close()
        except Exception: pass
        for fp in getattr(self, "visual_point_files", {}).values():
            try: fp.close()
            except Exception: pass
        try:
            if getattr(self, "visual_point_all_fp", None) is not None:
                self.visual_point_all_fp.close()
        except Exception: pass
        try:
            if getattr(self, "overlay_ws", None) is not None:
                self.overlay_ws.close()
        except Exception: pass
        for fp in getattr(self, "debug_files", {}).values():
            try: fp.close()
            except Exception: pass
        try:
            if getattr(self, "debug_all_fp", None) is not None:
                self.debug_all_fp.close()
        except Exception: pass
        for fp in getattr(self, "reject_debug_files", {}).values():
            try: fp.close()
            except Exception: pass
        try:
            if getattr(self, "reject_debug_all_fp", None) is not None:
                self.reject_debug_all_fp.close()
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
        if int(entry.frame_w or 0) > 0 and int(entry.frame_h or 0) > 0:
            self.latest_frame_size_by_cam[str(cam)] = (int(entry.frame_w), int(entry.frame_h))
        if event_ts is None and sync_index >= 0:
            self.pending_without_ts[cam].append(entry)
        else:
            self.history[cam].append(entry)
            self._emit_overlay_frame_state(cam, entry)

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
                self._emit_overlay_frame_state(cam, e)
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
            is_visual_point = self._is_visual_point_event(ev)

            if is_visual_point:
                key = (
                    "vpoint",
                    str(ev.camera_sn),
                    str(getattr(ev, "camera_alias", "") or ""),
                    int(ev.bullet_id),
                    int(getattr(ev, "point_index", -1) if getattr(ev, "point_index", -1) is not None else -1),
                    int(getattr(ev, "timestamp_us", 0) or 0),
                )
            else:
                key = (
                    "event",
                    str(ev.camera_sn),
                    int(ev.bullet_id),
                    int(ev.seq_id),
                    str(getattr(ev, "event_type", "") or ""),
                    str(getattr(ev, "terminate_reason", "") or ""),
                )

            if not self._mark_seen(key):
                if is_visual_point:
                    self._record_visual_point_stage(
                        "udp_duplicate_drop",
                        cam=str(getattr(ev, "camera_alias", "") or ""),
                        ev=ev,
                    )
                if is_visual_point and int(getattr(ev, "seq_id", 0) or 0) % 20 == 0:
                    print(
                        f"[VPOINT_DUP_DROP] cam_sn={ev.camera_sn} alias={getattr(ev, 'camera_alias', '')} "
                        f"bullet={ev.bullet_id} seq={ev.seq_id} idx={getattr(ev, 'point_index', -1)} "
                        f"ts={getattr(ev, 'timestamp_us', 0)}",
                        flush=True,
                    )
                continue

            self.recv_count += 1
            self.recv_total += 1
            cam = self._camera_for_event(ev)
            if is_visual_point:
                self._record_visual_point_stage(
                    "udp_recv",
                    cam=cam or str(getattr(ev, "camera_alias", "") or ""),
                    ev=ev,
                )
            if cam is None:
                if is_visual_point:
                    self._record_visual_point_stage(
                        "no_camera_route",
                        cam=str(getattr(ev, "camera_alias", "") or ""),
                        ev=ev,
                    )
                self.no_cam_count += 1
                continue

            # Real-time visual point samples take the high-frequency path.
            # They are projected to MVS, broadcast as overlay_bullet_point for UE5,
            # and the previous->current MVS segment is also judged immediately.
            # This makes point-segment hit detection run at the bullet point update
            # rate instead of waiting for 40ms track_probe/turn/terminate events.
            if self._is_visual_point_event(ev):
                H = self.H.get(cam)
                if H is None:
                    self._record_visual_point_stage("no_homography", cam=cam, ev=ev)
                    self.drop_count += 1
                    continue
                result = self._handle_visual_bullet_point_event(cam, ev, H, age_ms=0.0)
                self._record_visual_point_stage(f"point_handle_result:{result}", cam=cam, ev=ev)
                if result in {"point_no_trigger", "point_bad_trigger"}:
                    self.drop_count += 1
                elif result == "point_no_human":
                    self.no_human_count += 1
                elif result == "point_pending_human":
                    self.wait_count += 1
                elif result == "point_emitted":
                    pass
                continue

            # UE5 marker stream must not depend on hit-judge gates.  Turn/end/probe
            # events are visual events even when they do not produce a hit candidate.
            # Emit once here; judge_event() may call the same helper later, but
            # _mark_visual_seen() prevents duplicates.
            if str(getattr(ev, "event_type", "") or "") in {EVENT_TURN, EVENT_TERMINATE}:
                H_vis = self.H.get(cam)
                if H_vis is not None:
                    try:
                        event_pt_vis = _apply_H(H_vis, float(ev.event_x), float(ev.event_y))
                        if bool(getattr(ev, "approach_valid", False)):
                            approach_pt_vis = _apply_H(H_vis, float(getattr(ev, "approach_x", ev.event_x)), float(getattr(ev, "approach_y", ev.event_y)))
                        else:
                            approach_pt_vis = event_pt_vis
                        track_points_mvs_vis = self._project_track_points_mvs(H_vis, ev, event_pt_vis, approach_pt_vis)
                        self._emit_visual_bullet_event(cam, ev, event_pt_vis, approach_pt_vis, track_points_mvs_vis, age_ms=0.0)
                    except Exception as exc:
                        self._emit_debug(cam, ev, "visual_marker_emit_failed", str(exc), age_ms=0.0)

            if not bool(getattr(self, "hit_detection_enabled", True)):
                self.hit_reject_counts["hit_detection_disabled"] += 1
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

    def _person_body_motion_reject_detail(
        self,
        person: Dict[str, Any],
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        approach_valid: bool,
        approach_len_px: float,
    ) -> Optional[Dict[str, Any]]:
        """Return reject detail when a candidate looks born from human motion.

        This gate is intentionally evaluated in hit_judge, not in the event-camera
        process, so human mask usage cannot slow down the high-frequency event loop.
        The rule is conservative: a final human hit must have a usable approach
        segment from outside the current mask.  Tracks born inside the mask are the
        common false-positive class caused by turning bodies, clothes and limbs.
        """
        if not bool(getattr(self, 'body_motion_reject_enable', True)):
            return None

        polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
        if isinstance(polys, dict):
            polys = list(polys.values())
        checks: List[Dict[str, Any]] = []
        if isinstance(polys, list):
            for poly in polys:
                if not isinstance(poly, list):
                    continue
                pts = [(float(q[0]), float(q[1])) for q in poly if isinstance(q, (list, tuple)) and len(q) >= 2]
                if len(pts) < 3:
                    continue
                event_inside = bool(_point_in_polygon(event_pt, pts))
                approach_inside = bool(approach_valid and _point_in_polygon(approach_pt, pts))
                segment_cross = bool(approach_valid and _segment_intersects_polygon(approach_pt, event_pt, pts))
                event_dist = float(_dist_to_polygon(event_pt, pts))
                approach_dist = float(_dist_to_polygon(approach_pt, pts)) if approach_valid else -1.0
                checks.append({
                    "source": "mask_polygon",
                    "event_inside_mask": event_inside,
                    "approach_inside_mask": approach_inside,
                    "segment_intersects_mask": segment_cross,
                    "event_to_mask_dist_px": round(event_dist, 3),
                    "approach_to_mask_dist_px": round(approach_dist, 3),
                })

        if not checks and bool(getattr(self, 'body_motion_use_bbox_fallback', False)):
            box = person.get("bbox_xyxy") or person.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                    if x2 <= x1 or y2 <= y1:
                        x2, y2 = x1 + float(box[2]), y1 + float(box[3])
                    event_inside = (x1 <= float(event_pt[0]) <= x2 and y1 <= float(event_pt[1]) <= y2)
                    approach_inside = bool(approach_valid and x1 <= float(approach_pt[0]) <= x2 and y1 <= float(approach_pt[1]) <= y2)
                    checks.append({
                        "source": "bbox_fallback",
                        "event_inside_mask": bool(event_inside),
                        "approach_inside_mask": bool(approach_inside),
                        "segment_intersects_mask": False,
                        "event_to_mask_dist_px": 0.0 if event_inside else -1.0,
                        "approach_to_mask_dist_px": 0.0 if approach_inside else -1.0,
                    })
                except Exception:
                    pass

        if not checks:
            return None

        any_event_inside = any(bool(c.get("event_inside_mask")) for c in checks)
        any_approach_inside = any(bool(c.get("approach_inside_mask")) for c in checks)
        any_segment_cross = any(bool(c.get("segment_intersects_mask")) for c in checks)
        min_external_len = float(getattr(self, 'body_motion_min_external_approach_len_px', 45.0) or 45.0)

        reason = None
        if any_approach_inside and bool(getattr(self, 'body_motion_reject_if_approach_inside', True)):
            reason = "body_motion_origin_inside_mask"
        elif any_event_inside and any_approach_inside and bool(getattr(self, 'body_motion_reject_if_both_inside', True)):
            reason = "body_motion_both_points_inside_mask"
        elif any_event_inside and bool(getattr(self, 'body_motion_reject_event_inside_without_external_approach', True)):
            if (not bool(approach_valid)) or float(approach_len_px) < min_external_len or not any_segment_cross:
                reason = "body_motion_no_external_approach"

        if reason is None:
            return None
        return {
            "body_motion_reject": True,
            "body_motion_reject_reason": reason,
            "body_motion_checks": checks[:4],
            "body_motion_min_external_approach_len_px": round(min_external_len, 3),
            "body_motion_approach_len_px": round(float(approach_len_px), 3),
            "body_motion_approach_valid": bool(approach_valid),
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
                "impact_sync_source": str(period.get("sync_source", "")),
                "impact_anchor_source": str(period.get("anchor_source", "")),
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

    def _pseudo_hit_runtime_enabled(self) -> bool:
        return bool(getattr(self, "hit_detection_enabled", True)) and bool(getattr(self, "strong_reasoning_enabled", True)) and bool(getattr(self, "pseudo_hit_enable", True))

    def _pseudo_hit_reason_for_event(self, ev: BulletEvent) -> Optional[str]:
        event_type = str(getattr(ev, "event_type", "") or "")
        if event_type == EVENT_TURN:
            return "turn_point_inside_mask"
        if event_type == EVENT_TERMINATE:
            reason = str(getattr(ev, "terminate_reason", "") or "")
            if reason in {"track_probe"}:
                return None
            return "terminal_point_inside_mask"
        if event_type == EVENT_POINT or str(getattr(ev, "terminate_reason", "") or "") == REASON_VISUAL_POINT:
            return "visual_point_stable_track_inside_mask"
        return None

    def _mark_pseudo_hit_key(self, key: Tuple[Any, ...]) -> bool:
        if key in self.pseudo_hit_seen_set:
            self.pseudo_hit_duplicate_suppressed += 1
            return False
        if self.pseudo_hit_seen_keys.maxlen is not None and len(self.pseudo_hit_seen_keys) >= self.pseudo_hit_seen_keys.maxlen:
            try:
                old = self.pseudo_hit_seen_keys.popleft()
                self.pseudo_hit_seen_set.discard(old)
            except Exception:
                pass
        self.pseudo_hit_seen_keys.append(key)
        self.pseudo_hit_seen_set.add(key)
        return True

    def _pseudo_hit_track_key(self, ev: BulletEvent) -> str:
        global_id = str(getattr(ev, "global_bullet_id", "") or "").strip()
        if global_id:
            return f"global:{global_id}"
        stable_id = _safe_int(getattr(ev, "stable_track_id", -1), -1)
        if stable_id >= 0:
            return f"stable:{stable_id}"
        display_id = _safe_int(getattr(ev, "display_bullet_id", -1), -1)
        if display_id >= 0:
            return f"display:{display_id}"
        return f"bullet:{_safe_int(getattr(ev, 'bullet_id', 0), 0)}"

    def _pseudo_hit_person_id(self, person: Dict[str, Any]) -> int:
        for key in ("global_person_id", "global_id", "person_global_id", "stable_id", "person_id", "track_id"):
            val = person.get(key)
            try:
                if val is not None:
                    return int(val)
            except Exception:
                continue
        return 0

    def _person_identity_evidence(
        self,
        cam: str,
        entry: PersonEntry,
        person: Dict[str, Any],
        local_id: int,
        source: str,
    ) -> Dict[str, Any]:
        bbox = person.get("bbox_xyxy") or person.get("bbox") or []
        centroid = person.get("foot_pixel") or person.get("centroid") or person.get("person_centroid") or []
        evidence: Dict[str, Any] = {
            "source": str(source),
            "camera": str(cam),
            "camera_local_id": int(local_id),
            "person_sync_index": int(entry.sync_index),
            "person_event_ts_us": int(entry.event_ts_us or 0),
            "person_wall_time": float(getattr(entry, "wall_time", 0.0) or 0.0),
            "frame_w": int(entry.frame_w or 0),
            "frame_h": int(entry.frame_h or 0),
            "bbox_xyxy": list(bbox) if isinstance(bbox, (list, tuple)) else [],
            "centroid": list(centroid) if isinstance(centroid, (list, tuple)) else [],
            "confidence": _safe_float(person.get("confidence"), 0.0),
            "stable_id_is_global": False,
        }
        for key in ("local_xy", "local_xy_raw", "fused_xy", "anchor_pixel_raw", "anchor_pixel_corrected", "foot_pixel"):
            value = person.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    evidence[key] = [float(value[0]), float(value[1])]
                except Exception:
                    evidence[key] = list(value)
        phys = person.get("physical_coord")
        if isinstance(phys, dict):
            evidence["physical_coord"] = dict(phys)
        for key in ("global_id", "global_person_id", "person_global_id", "target_global_id"):
            gid = _safe_int(person.get(key), -1)
            if gid >= 0:
                evidence["target_global_runtime_id"] = int(gid)
                evidence["target_global_runtime_reason"] = f"person_field_{key}"
                break
        player_id = str(person.get("global_player_id") or person.get("target_global_player_id") or "")
        if player_id:
            evidence["target_global_player_id"] = player_id
        return evidence

    def _pseudo_hit_entries_best_effort(
        self,
        cam: str,
        ev: BulletEvent,
        age_ms: float,
        force_final: bool,
    ) -> Tuple[List[PersonEntry], Optional[Dict[str, Any]], str, Dict[str, Any]]:
        period: Optional[Dict[str, Any]] = None
        strategy_used = "nearest_time_fallback"
        selection_info: Dict[str, Any] = {}
        if self.impact_sync_enable:
            period = self._impact_period_for_event(cam, int(getattr(ev, "timestamp_us", 0) or 0))
            if period is None:
                return [], None, "no_trigger_period", {"pseudo_hit_entries_reason": "no_trigger_period"}
            entries, strategy_used, _should_wait, selection_info = self._entries_for_impact(
                cam, period, age_ms=age_ms, force_final=True if force_final else False
            )
            return list(entries or []), period, strategy_used, dict(selection_info or {})
        entry = self._nearest_person_entry(cam, int(getattr(ev, "timestamp_us", 0) or 0))
        if entry is None:
            return [], None, strategy_used, {"pseudo_hit_entries_reason": "nearest_time_no_human"}
        selection_info = {
            "target_person_sync_index": int(entry.sync_index),
            "used_person_sync_index": int(entry.sync_index),
            "human_fallback_used": False,
            "human_fallback_reason": "",
            "person_frame_delta_sync": 0,
        }
        return [entry], None, strategy_used, selection_info

    def _pseudo_hit_human_noise_detail(
        self,
        person: Dict[str, Any],
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        approach_valid: bool,
        track_metrics: Dict[str, Any],
        mask_method: str,
        mask_evidence: str,
    ) -> Dict[str, Any]:
        points: List[Tuple[float, float]] = []
        raw_points = track_metrics.get("recent_mvs_points") if isinstance(track_metrics, dict) else None
        if isinstance(raw_points, list):
            for row in raw_points:
                try:
                    if isinstance(row, (list, tuple)) and len(row) >= 3:
                        points.append((float(row[1]), float(row[2])))
                except Exception:
                    continue
        if len(points) < 2 and approach_valid:
            points = [(float(approach_pt[0]), float(approach_pt[1])), (float(event_pt[0]), float(event_pt[1]))]
        elif len(points) < 1:
            points = [(float(event_pt[0]), float(event_pt[1]))]

        track_start_pt = points[0] if points else (float(event_pt[0]), float(event_pt[1]))
        track_start_bbox_expand_px = float(getattr(self, "human_noise_track_start_bbox_expand_px", 24.0) or 24.0)
        track_start_bbox = person.get("bbox_xyxy") or person.get("bbox") or person.get("person_bbox")
        track_start_inside_bbox = _point_in_expanded_bbox(track_start_pt, track_start_bbox, track_start_bbox_expand_px)
        track_start_bbox_policy = self._normalize_track_start_bbox_policy(
            getattr(self, "human_noise_track_start_bbox_policy", "soft")
        )

        polys = _person_mask_polygons_as_points(person)
        if not polys:
            reasons = ["track_start_inside_expanded_bbox"] if track_start_inside_bbox else ["no_mask_polygon"]
            start_score = 0.0
            if track_start_inside_bbox and track_start_bbox_policy != "off":
                start_score = 1.0 if track_start_bbox_policy == "hard" else 0.5
            return {
                "human_noise_filter_enable": bool(getattr(self, "human_noise_filter_enable", True)),
                "human_noise_filter_mode": str(getattr(self, "human_noise_filter_mode", "soft")),
                "human_noise_track_start_bbox_filter_enable": bool(getattr(self, "human_noise_track_start_bbox_filter_enable", True)),
                "human_noise_track_start_bbox_expand_px": round(float(track_start_bbox_expand_px), 3),
                "human_noise_track_start_bbox_policy": str(track_start_bbox_policy),
                "human_noise_track_start_inside_expanded_bbox": bool(track_start_inside_bbox),
                "human_noise_track_start_point": [round(float(track_start_pt[0]), 3), round(float(track_start_pt[1]), 3)],
                "human_noise_score": round(float(start_score), 3),
                "human_noise_suspect": bool(track_start_inside_bbox),
                "human_noise_clear": bool(track_start_inside_bbox and track_start_bbox_policy == "hard"),
                "human_noise_reasons": reasons,
                "inbound_evidence": False,
                "inbound_evidence_quality": "unknown_no_mask_polygon",
            }

        inside_flags = [_point_in_any_polygon(pt, polys) for pt in points]
        born_inside = bool(inside_flags[0]) if inside_flags else False
        latest_inside = bool(inside_flags[-1]) if inside_flags else _point_in_any_polygon(event_pt, polys)
        outside_count = int(sum(1 for item in inside_flags if not item))
        inside_count = int(sum(1 for item in inside_flags if item))

        outside_to_inside = False
        segment_cross = False
        first_inside_idx = -1
        for idx, inside in enumerate(inside_flags):
            if inside:
                first_inside_idx = idx
                break
        for idx in range(1, len(points)):
            if (not inside_flags[idx - 1]) and inside_flags[idx]:
                outside_to_inside = True
            if (not inside_flags[idx - 1] or not inside_flags[idx]) and _segment_intersects_any_polygon(points[idx - 1], points[idx], polys):
                segment_cross = True
        if approach_valid:
            approach_inside = _point_in_any_polygon(approach_pt, polys)
            event_inside = _point_in_any_polygon(event_pt, polys)
            if (not approach_inside) and event_inside:
                outside_to_inside = True
            if (not approach_inside or not event_inside) and _segment_intersects_any_polygon(approach_pt, event_pt, polys):
                segment_cross = True
        else:
            approach_inside = False
            event_inside = latest_inside

        if first_inside_idx >= 0:
            lead_points = points[: first_inside_idx + 1]
            pre_hit_outside_points = int(sum(1 for item in inside_flags[:first_inside_idx] if not item))
        else:
            lead_points = points
            pre_hit_outside_points = int(outside_count)
        pre_hit_path_len = _path_len_px(lead_points)

        inbound_by_method = str(mask_method or "") in {"segment_intersect_mask", "track_polyline_cross_mask", "track_projected_inbound_mask"}
        enough_pre_hit = (
            int(pre_hit_outside_points) >= int(getattr(self, "human_noise_min_pre_hit_outside_points", 3))
            and float(pre_hit_path_len) >= float(getattr(self, "human_noise_min_pre_hit_path_len_px", 30.0))
        )
        inbound = bool(outside_to_inside or segment_cross or inbound_by_method or enough_pre_hit)

        point_count = int(float(track_metrics.get("track_point_count", len(points)) or len(points))) if isinstance(track_metrics, dict) else len(points)
        path_len = float(track_metrics.get("track_path_len_px", _path_len_px(points)) or _path_len_px(points)) if isinstance(track_metrics, dict) else _path_len_px(points)
        duration_ms = float(track_metrics.get("track_duration_ms", 0.0) or 0.0) if isinstance(track_metrics, dict) else 0.0
        straightness = float(track_metrics.get("track_straightness", 1.0) or 1.0) if isinstance(track_metrics, dict) else 1.0
        speed = float(track_metrics.get("avg_speed_px_per_ms", 0.0) or 0.0) if isinstance(track_metrics, dict) else 0.0
        short_track = (
            point_count <= int(getattr(self, "human_noise_short_track_max_points", 5))
            and path_len <= float(getattr(self, "human_noise_short_track_max_path_len_px", 35.0))
        )
        short_duration = duration_ms > 0.0 and duration_ms <= float(getattr(self, "human_noise_short_track_max_duration_ms", 20.0))

        reasons: List[str] = []
        score = 0.0
        if born_inside and not inbound:
            score += 0.35
            reasons.append("born_inside_mask_without_inbound")
        if not inbound and outside_count <= 0:
            score += 0.18
            reasons.append("no_outside_track_points")
        if not inbound and pre_hit_outside_points < int(getattr(self, "human_noise_min_pre_hit_outside_points", 3)):
            score += 0.14
            reasons.append("pre_hit_outside_points_too_few")
        if not inbound and pre_hit_path_len < float(getattr(self, "human_noise_min_pre_hit_path_len_px", 30.0)):
            score += 0.14
            reasons.append("pre_hit_path_len_too_short")
        if not inbound and short_track:
            score += 0.18
            reasons.append("short_track_inside_mask")
        if not inbound and short_duration:
            score += 0.06
            reasons.append("short_duration_inside_mask")
        if not inbound and str(mask_method or "") in {"point_in_mask", "track_point_inside_mask"}:
            score += 0.08
            reasons.append("point_only_mask_contact")
        if not inbound and straightness > 0.0 and straightness < 0.35:
            score += 0.05
            reasons.append("low_direction_consistency")
        if track_start_inside_bbox and track_start_bbox_policy != "off":
            score += 0.50 if track_start_bbox_policy == "hard" else 0.28
            reasons.append("track_start_inside_expanded_bbox")
        if inbound:
            score -= 0.35
            reasons.append("has_inbound_evidence")
        if enough_pre_hit:
            score -= 0.18
            reasons.append("enough_pre_hit_outside_path")
        if path_len >= max(80.0, float(getattr(self, "human_noise_min_pre_hit_path_len_px", 30.0)) * 2.0) and point_count >= 5:
            score -= 0.10
            reasons.append("long_stable_track")

        score = max(0.0, min(1.0, float(score)))
        clear_noise = bool(
            born_inside
            and not inbound
            and outside_count <= 0
            and short_track
        )
        suspect = bool(score >= float(getattr(self, "human_noise_soft_threshold", 0.62)) or clear_noise)
        quality = "inbound" if inbound else ("clear_noise" if clear_noise else ("suspect" if suspect else "weak_or_unknown"))
        return {
            "human_noise_filter_enable": bool(getattr(self, "human_noise_filter_enable", True)),
            "human_noise_filter_mode": str(getattr(self, "human_noise_filter_mode", "soft")),
            "human_noise_track_start_bbox_filter_enable": bool(getattr(self, "human_noise_track_start_bbox_filter_enable", True)),
            "human_noise_track_start_bbox_expand_px": round(float(track_start_bbox_expand_px), 3),
            "human_noise_track_start_bbox_policy": str(track_start_bbox_policy),
            "human_noise_track_start_inside_expanded_bbox": bool(track_start_inside_bbox),
            "human_noise_track_start_point": [round(float(track_start_pt[0]), 3), round(float(track_start_pt[1]), 3)],
            "human_noise_score": round(float(score), 3),
            "human_noise_suspect": bool(suspect),
            "human_noise_clear": bool(clear_noise),
            "human_noise_reasons": reasons,
            "inbound_evidence": bool(inbound),
            "inbound_evidence_quality": quality,
            "born_inside_mask": bool(born_inside),
            "latest_inside_mask": bool(latest_inside),
            "approach_inside_mask": bool(approach_inside),
            "event_inside_mask": bool(event_inside),
            "outside_to_inside_transition": bool(outside_to_inside),
            "segment_cross_mask": bool(segment_cross),
            "pre_hit_outside_points": int(pre_hit_outside_points),
            "pre_hit_path_len_px": round(float(pre_hit_path_len), 3),
            "track_inside_point_count": int(inside_count),
            "track_outside_point_count": int(outside_count),
            "human_noise_short_track": bool(short_track),
            "human_noise_track_point_count": int(point_count),
            "human_noise_track_path_len_px": round(float(path_len), 3),
            "human_noise_track_duration_ms": round(float(duration_ms), 3),
            "human_noise_track_straightness": round(float(straightness), 3),
            "human_noise_track_speed_px_per_ms": round(float(speed), 3),
            "mask_evidence_for_noise": str(mask_evidence or ""),
        }

    def _human_noise_positive_contact_evidence(self, d: Dict[str, Any]) -> bool:
        reason = str(d.get("pseudo_hit_reason", "") or "")
        method = str(d.get("mask_method", "") or "")
        event_type = str(d.get("event_type", "") or "")
        point_count = _safe_int(d.get("track_point_count"), 0)
        try:
            path_len = float(d.get("track_path_len_px", 0.0) or 0.0)
        except Exception:
            path_len = 0.0
        min_path = float(getattr(self, "visual_point_hit_min_track_path_len_px", 20.0) or 20.0)
        terminal_or_turn = event_type in {EVENT_TERMINATE, EVENT_TURN}
        stable_visual = reason in {
            "visual_point_stable_track_inside_mask",
            "visual_point_stable_segment_cross_mask",
        }
        keypoint_inside = method in {
            "point_in_mask",
            "segment_intersect_mask",
            "track_point_inside_mask",
            "track_polyline_cross_mask",
        }
        enough_track_shape = point_count >= 3 or path_len >= min_path
        return bool((terminal_or_turn or stable_visual) and keypoint_inside and enough_track_shape)

    def _human_noise_should_suppress_final(self, d: Dict[str, Any]) -> Tuple[bool, str]:
        if not bool(getattr(self, "human_noise_filter_enable", True)):
            return False, "filter_disabled"
        mode = self._normalize_human_noise_filter_mode(getattr(self, "human_noise_filter_mode", "soft"))
        if mode in {"off", "shadow"}:
            return False, f"mode_{mode}"
        try:
            score = float(d.get("human_noise_score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        clear_noise = bool(d.get("human_noise_clear", False))
        inbound = bool(d.get("inbound_evidence", False))
        require_inbound = bool(getattr(self, "human_noise_require_inbound_for_low_quality", True))
        start_bbox_filter = bool(getattr(self, "human_noise_track_start_bbox_filter_enable", True))
        start_inside_bbox = bool(d.get("human_noise_track_start_inside_expanded_bbox", False))
        start_policy = self._normalize_track_start_bbox_policy(getattr(self, "human_noise_track_start_bbox_policy", "soft"))
        positive_contact = self._human_noise_positive_contact_evidence(d)
        if start_bbox_filter and start_inside_bbox and start_policy == "hard":
            return True, "track_start_inside_expanded_bbox"
        if mode == "hard":
            if score >= float(getattr(self, "human_noise_soft_threshold", 0.62)):
                return True, "hard_score_threshold"
            if require_inbound and (not inbound) and bool(d.get("human_noise_suspect", False)):
                return True, "hard_require_inbound"
        else:
            if score >= float(getattr(self, "human_noise_hard_threshold", 0.88)) and not positive_contact:
                return True, "soft_high_score_threshold"
            if require_inbound and clear_noise and not positive_contact:
                return True, "soft_clear_noise_without_inbound"
        return False, "pass"

    def _write_pseudo_hit_dict(self, cam: str, d: Dict[str, Any]) -> None:
        d.setdefault("schema_version", 1)
        d.setdefault("source", "hit_judge_server")
        d.setdefault("judge_chain", "strong_reasoning")
        d.setdefault("shadow_only", bool(getattr(self, "strong_reasoning_shadow_only", True)))
        d.setdefault("final_hit_authority", str(getattr(self, "final_hit_authority", "strong_gate") or "strong_gate"))
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        fp = getattr(self, "pseudo_hit_files", {}).get(str(cam))
        if fp is not None:
            fp.write(line)
        all_fp = getattr(self, "pseudo_hit_all_fp", None)
        if all_fp is not None:
            all_fp.write(line)
        self.pseudo_hit_total += 1
        self.pseudo_hit_by_camera[str(cam)] += 1
        self.pseudo_hit_reason_counts[str(d.get("pseudo_hit_reason", ""))] += 1
        if bool(d.get("human_noise_suspect", False)):
            self.pseudo_hit_human_noise_suspect += 1
            reasons = d.get("human_noise_reasons", [])
            if isinstance(reasons, list):
                for reason in reasons[:6]:
                    self.pseudo_hit_human_noise_reason_counts[str(reason)] += 1
            elif reasons:
                self.pseudo_hit_human_noise_reason_counts[str(reasons)] += 1
        self.latest_pseudo_hit = dict(d)

    def _is_projected_mask_candidate(self, d: Dict[str, Any]) -> bool:
        return (
            str(d.get("pseudo_hit_reason", "") or "") == "visual_point_stable_projected_mask"
            or str(d.get("mask_method", "") or "") == "track_projected_inbound_mask"
            or str(d.get("mask_evidence", "") or "") == "projected_track_mask_entry"
        )

    def _projected_mask_final_suppress_reason(self, d: Dict[str, Any]) -> str:
        if not self._is_projected_mask_candidate(d):
            return ""
        mode = self._normalize_projected_mask_final_mode(getattr(self, "projected_mask_final_mode", "pending_only"))
        if mode == "pending_only":
            return "projected_mask_pending_only"
        if mode == "shadow_only":
            return "projected_mask_shadow_only"
        if bool(getattr(self, "projected_mask_require_terminal_or_turn", True)):
            event_type = str(d.get("event_type", "") or "")
            terminate_reason = str(d.get("terminate_reason", "") or "")
            if event_type == EVENT_POINT or terminate_reason == REASON_VISUAL_POINT:
                return "projected_mask_requires_terminal_or_turn"
        return ""

    def _write_projected_mask_shadow_candidate(self, cam: str, d: Dict[str, Any], reason: str) -> None:
        shadow = dict(d)
        shadow.update({
            "type": "hit_candidate",
            "judge_chain": "strong_reasoning",
            "judge_mode": "strong_reasoning_pseudo_hit",
            "match_method": "pseudo_hit_projected_mask_pending",
            "source": "hit_judge_server",
            "final_hit_projected_mask_policy": str(getattr(self, "projected_mask_final_mode", "pending_only") or "pending_only"),
            "final_hit_projected_mask_suppress_reason": str(reason or ""),
            "total_score": float(shadow.get("confidence", shadow.get("space_score", 0.0)) or 0.0),
            "score_breakdown": {
                "time_score": 0.0,
                "space_score": float(shadow.get("space_score", shadow.get("confidence", 0.0)) or 0.0),
                "event_score": 0.0,
                "speed_score": 0.0,
                "loss_penalty": 0.0,
            },
        })
        self._write_candidate_shadow_dict(str(cam), shadow, True, str(reason or "projected_mask_pending_only"))

    def _promote_pseudo_hit_to_final_if_enabled(self, cam: str, d: Dict[str, Any]) -> None:
        if not bool(getattr(self, "hit_candidate_output_enabled", True)):
            return
        if not self._strong_reasoning_final_output_allowed():
            return
        if self._is_projected_mask_candidate(d):
            projected_reason = self._projected_mask_final_suppress_reason(d)
            if projected_reason:
                self.projected_pending_count += 1
                self.projected_suppressed_final_count += 1
                self.hit_reject_counts[projected_reason] += 1
                self.strong_reasoning_output_suppressed += 1
                self._write_projected_mask_shadow_candidate(cam, d, projected_reason)
                return
            self.projected_promoted_count += 1
        suppress, suppress_reason = self._human_noise_should_suppress_final(d)
        if suppress:
            self.pseudo_hit_human_noise_suppressed += 1
            self.hit_reject_counts[f"human_noise_{suppress_reason}"] += 1
            self.strong_reasoning_output_suppressed += 1
            shadow = dict(d)
            shadow.update({
                "type": "hit_candidate",
                "judge_chain": "strong_reasoning",
                "judge_mode": "strong_reasoning_pseudo_hit",
                "match_method": "pseudo_hit_inside_mask",
                "source": "hit_judge_server",
                "human_noise_shadow_written": True,
                "human_noise_shadow_reason": str(suppress_reason),
            })
            self._write_candidate_shadow_dict(str(cam), shadow, True, f"human_noise_{suppress_reason}")
            return
        base_space_score = float(d.get("space_score", d.get("confidence", 0.72)) or 0.72)
        quality_multiplier = 1.0
        if (
            self._normalize_track_start_bbox_policy(getattr(self, "human_noise_track_start_bbox_policy", "soft")) == "soft"
            and bool(d.get("human_noise_track_start_inside_expanded_bbox", False))
        ):
            quality_multiplier = 0.82 if bool(d.get("inbound_evidence", False)) else 0.72
        adjusted_space_score = max(0.0, min(1.0, base_space_score * quality_multiplier))
        cand = dict(d)
        cand.update({
            "type": "hit_candidate",
            "judge_chain": "strong_reasoning",
            "judge_mode": "strong_reasoning_pseudo_hit",
            "match_method": "pseudo_hit_inside_mask",
            "source": "hit_judge_server",
            "confidence": round(float(adjusted_space_score), 3),
            "space_score": round(float(adjusted_space_score), 3),
            "total_score": round(float(adjusted_space_score), 3),
            "human_noise_quality_multiplier": round(float(quality_multiplier), 3),
            "score_breakdown": {
                "time_score": 0.0,
                "space_score": round(float(adjusted_space_score), 3),
                "event_score": 0.14 if str(cand.get("event_type", "")) == EVENT_TERMINATE else 0.11,
                "speed_score": 0.0,
                "loss_penalty": 0.0,
                "human_noise_penalty": round(float(base_space_score - adjusted_space_score), 3),
            },
        })
        self.pseudo_hit_promoted_to_final_count += 1
        self._write_candidate_dict(cam, cand)

    def _try_emit_pseudo_hit_for_event(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        use_segment: bool,
        entries: List[PersonEntry],
        period: Optional[Dict[str, Any]],
        selection_info: Dict[str, Any],
        strategy_used: str,
        track_metrics: Dict[str, float],
        age_ms: float,
        source: str,
        mask_contact_override: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not self._pseudo_hit_runtime_enabled():
            return 0
        pseudo_reason = self._pseudo_hit_reason_for_event(ev)
        if not pseudo_reason:
            return 0
        if not entries:
            return 0
        emitted = 0
        track_key = self._pseudo_hit_track_key(ev)
        bucket_us = max(1, int(float(getattr(self, "pseudo_hit_dedup_ms", 120.0) or 120.0) * 1000.0))
        time_bucket = int(int(getattr(ev, "timestamp_us", 0) or 0) // bucket_us)
        for entry in entries:
            if entry is None or not entry.persons:
                continue
            for person in entry.persons:
                if not isinstance(person, dict):
                    continue
                person_id = self._pseudo_hit_person_id(person)
                override = mask_contact_override if isinstance(mask_contact_override, dict) else {}
                override_person_id = _safe_int(override.get("person_id"), -999999) if override else -999999
                if override and override_person_id != -999999 and int(person_id) != int(override_person_id):
                    continue
                if override:
                    mask_method = str(override.get("mask_method") or "")
                    mask_space = float(override.get("mask_space", 0.0) or 0.0)
                    mask_evidence = str(override.get("mask_evidence", "") or "")
                    hit_xy = self._coerce_xy(override.get("hit_point_mvs"))
                    approach_xy = self._coerce_xy(override.get("approach_point_mvs"))
                    if hit_xy is not None:
                        event_pt = hit_xy
                    if approach_xy is not None:
                        approach_pt = approach_xy
                else:
                    mask_method, mask_space, mask_evidence = _person_mask_hit(person, event_pt, approach_pt, bool(use_segment))
                mask_method_s = str(mask_method or "")
                if mask_method_s not in {
                    "point_in_mask",
                    "segment_intersect_mask",
                    "track_point_inside_mask",
                    "track_polyline_cross_mask",
                    "track_projected_inbound_mask",
                }:
                    continue
                if mask_method_s == "segment_intersect_mask" and not bool(use_segment):
                    continue
                pseudo_reason_for_person = str(pseudo_reason)
                if str(getattr(ev, "event_type", "") or "") == EVENT_POINT:
                    if mask_method_s in {"point_in_mask", "track_point_inside_mask"}:
                        pseudo_reason_for_person = "visual_point_stable_track_inside_mask"
                    elif mask_method_s in {"segment_intersect_mask", "track_polyline_cross_mask"}:
                        pseudo_reason_for_person = "visual_point_stable_segment_cross_mask"
                    else:
                        pseudo_reason_for_person = "visual_point_stable_projected_mask"
                key = (
                    str(cam),
                    str(track_key),
                    str(getattr(ev, "event_type", "")),
                    str(pseudo_reason_for_person),
                    int(person_id),
                    int(entry.sync_index),
                    int(time_bucket),
                )
                if not self._mark_pseudo_hit_key(key):
                    continue
                bbox = person.get("bbox_xyxy") or person.get("bbox") or []
                centroid = person.get("foot_pixel") or person.get("centroid") or []
                payload: Dict[str, Any] = {
                    "type": "pseudo_hit",
                    "pseudo_hit_reason": str(pseudo_reason_for_person),
                    "pseudo_hit_source": str(source),
                    "camera": str(cam),
                    "camera_sn": str(getattr(ev, "camera_sn", "") or ""),
                    "camera_alias": str(getattr(ev, "camera_alias", "") or cam),
                    "bullet_id": int(getattr(ev, "bullet_id", 0) or 0),
                    "stable_track_id": _safe_int(getattr(ev, "stable_track_id", -1), -1),
                    "display_bullet_id": _safe_int(getattr(ev, "display_bullet_id", -1), -1),
                    "global_bullet_id": str(getattr(ev, "global_bullet_id", "") or ""),
                    "track_key": str(track_key),
                    "seq_id": int(getattr(ev, "seq_id", 0) or 0),
                    "event_type": str(getattr(ev, "event_type", "") or ""),
                    "event_ts": int(getattr(ev, "timestamp_us", 0) or 0),
                    "terminate_reason": str(getattr(ev, "terminate_reason", "") or ""),
                    "turn_angle_deg": round(float(getattr(ev, "turn_angle_deg", 0.0) or 0.0), 3),
                    "hit_point_mvs": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
                    "event_point": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
                    "approach_point": [round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],
                    "approach_valid": bool(use_segment),
                    "mask_method": str(mask_method),
                    "mask_evidence": str(mask_evidence),
                    "track_mask_detail": dict(override.get("track_mask_detail", {}) or {}) if override else {},
                    "space_score": round(float(mask_space), 3),
                    "confidence": round(float(mask_space), 3),
                    "stable_id": int(person_id),
                    "person_id": int(person_id),
                    "target_camera": str(cam),
                    "target_camera_local_id": int(person_id),
                    "target_person_sync_index": int(entry.sync_index),
                    "target_person_event_ts_us": int(entry.event_ts_us or 0),
                    "target_identity_evidence": self._person_identity_evidence(
                        cam, entry, person, int(person_id), "hit_judge_pseudo_hit"
                    ),
                    "person_bbox": list(bbox) if isinstance(bbox, (list, tuple)) else [],
                    "person_centroid": list(centroid) if isinstance(centroid, (list, tuple)) else [],
                    "target_local_xy": list(person.get("local_xy")) if isinstance(person.get("local_xy"), (list, tuple)) else [],
                    "target_physical_coord": dict(person.get("physical_coord")) if isinstance(person.get("physical_coord"), dict) else {},
                    "person_ts": int(entry.event_ts_us or 0),
                    "mvs_sync_index": int(entry.sync_index),
                    "person_frame_sync_index": int(entry.sync_index),
                    "human_match_strategy": str(strategy_used),
                    "human_fallback_used": bool(selection_info.get("human_fallback_used", False)),
                    "human_fallback_reason": str(selection_info.get("human_fallback_reason", "") or ""),
                    "person_frame_delta_sync": _safe_int(selection_info.get("person_frame_delta_sync"), 0),
                    "final_hit_eligible": bool(self._strong_reasoning_final_output_allowed()),
                    "event_human_mask_stats": dict(getattr(ev, "event_human_mask_stats", {}) or {}),
                    "track_point_count": int(track_metrics.get("track_point_count", 0) or 0),
                    "track_path_len_px": round(float(track_metrics.get("track_path_len_px", 0.0) or 0.0), 3),
                    "track_duration_ms": round(float(track_metrics.get("track_duration_ms", 0.0) or 0.0), 3),
                    "track_displacement_px": round(float(track_metrics.get("track_displacement_px", 0.0) or 0.0), 3),
                    "track_straightness": round(float(track_metrics.get("track_straightness", 0.0) or 0.0), 3),
                    "track_line_error_px": round(float(track_metrics.get("track_line_error_px", 999.0) if track_metrics.get("track_line_error_px", 999.0) is not None else 999.0), 3),
                    "wall_time": time.time(),
                }
                noise_detail = self._pseudo_hit_human_noise_detail(
                    person,
                    event_pt,
                    approach_pt,
                    bool(use_segment),
                    track_metrics,
                    str(mask_method),
                    str(mask_evidence),
                )
                suppress, suppress_reason = self._human_noise_should_suppress_final(noise_detail)
                payload.update(noise_detail)
                payload["final_hit_suppressed_by_human_noise"] = bool(suppress)
                payload["final_hit_human_noise_suppress_reason"] = str(suppress_reason)
                payload["final_hit_eligible"] = bool(self._strong_reasoning_final_output_allowed() and not suppress)
                if period is not None:
                    payload.update({
                        "impact_event_ts_us": int(getattr(ev, "timestamp_us", 0) or 0),
                        "impact_sync_index": int(period["sync_index"]),
                        "impact_anchor_ts_us": int(period["anchor_ts_us"]),
                        "impact_next_sync_index": int(period["next_sync_index"]),
                        "impact_next_ts_us": int(period["next_ts_us"]),
                        "impact_offset_ms": round(float(period.get("offset_ms", 0.0) or 0.0), 3),
                        "impact_period_ms": round(float(period.get("period_us", 0.0) or 0.0) / 1000.0, 3),
                        "impact_sync_source": str(period.get("sync_source", "")),
                        "impact_anchor_source": str(period.get("anchor_source", "")),
                        "trigger_period_ok": bool(period.get("trigger_period_ok", False)),
                    })
                self._write_pseudo_hit_dict(cam, payload)
                self._promote_pseudo_hit_to_final_if_enabled(cam, payload)
                emitted += 1
        return emitted

    def _try_emit_pseudo_hit_best_effort(
        self,
        cam: str,
        ev: BulletEvent,
        event_pt: Tuple[float, float],
        approach_pt: Tuple[float, float],
        use_segment: bool,
        track_metrics: Dict[str, float],
        age_ms: float,
        force_final: bool,
        source: str,
    ) -> int:
        if not self._pseudo_hit_runtime_enabled():
            return 0
        entries, period, strategy_used, selection_info = self._pseudo_hit_entries_best_effort(cam, ev, age_ms, force_final)
        return self._try_emit_pseudo_hit_for_event(
            cam,
            ev,
            event_pt,
            approach_pt,
            bool(use_segment),
            entries,
            period,
            selection_info,
            strategy_used,
            track_metrics,
            age_ms,
            source,
        )

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
        d["judge_chain"] = "strong_gate"
        d["final_hit_authority"] = str(getattr(self, "final_hit_authority", "strong_gate") or "strong_gate")
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
            d["impact_sync_source"] = str(period.get("sync_source", ""))
            d["impact_anchor_source"] = str(period.get("anchor_source", ""))
            d["trigger_period_ms"] = round(float(period.get("trigger_period_ms", float(period["period_us"]) / 1000.0)), 3)
            d["trigger_period_ok"] = bool(period.get("trigger_period_ok", False))
            d["impact_next_boundary_estimated"] = bool(period.get("estimated_next", False))
        d["mvs_sync_index"] = int(entry.sync_index)
        d["person_frame_sync_index"] = int(entry.sync_index)
        d["target_camera"] = str(cam)
        d["target_camera_local_id"] = int(cand.stable_id)
        d["target_person_sync_index"] = int(entry.sync_index)
        d["target_person_event_ts_us"] = int(entry.event_ts_us or 0)
        d["target_identity_evidence"] = self._person_identity_evidence(
            cam,
            entry,
            {
                "bbox_xyxy": list(cand.person_bbox),
                "centroid": list(cand.person_centroid),
                "confidence": 1.0 if bool(cand.person_confirmed) else 0.0,
            },
            int(cand.stable_id),
            "hit_judge_final_candidate",
        )
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
        if not bool(getattr(self, "hit_candidate_output_enabled", True)):
            self.hit_reject_counts["hit_candidate_output_disabled"] += 1
            self._write_candidate_shadow_dict(str(cam), d, True, "hit_candidate_output_disabled")
            return
        chain = str(d.get("judge_chain", "strong_gate") or "strong_gate")
        if chain == "strong_gate" and not self._strong_gate_final_output_allowed():
            self.strong_gate_output_suppressed += 1
            self.hit_reject_counts["strong_gate_output_suppressed"] += 1
            self._write_candidate_shadow_dict(str(cam), d, True, "strong_gate_output_suppressed")
            return
        if chain == "strong_reasoning" and not self._strong_reasoning_final_output_allowed():
            self.strong_reasoning_output_suppressed += 1
            self.hit_reject_counts["strong_reasoning_output_suppressed"] += 1
            self._write_candidate_shadow_dict(str(cam), d, True, "strong_reasoning_output_suppressed")
            return
        if not self._mark_final_hit_output_key(str(cam), d):
            suppress_reason = str(d.pop("_final_hit_suppress_reason", "final_hit_duplicate_suppressed") or "final_hit_duplicate_suppressed")
            self._write_candidate_shadow_dict(str(cam), d, True, suppress_reason)
            return
        d.setdefault("final_hit_authority", str(getattr(self, "final_hit_authority", "strong_gate") or "strong_gate"))
        self._write_candidate_shadow_dict(str(cam), d, False, "")
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.out_files[cam].write(line)
        self.all_fp.write(line)
        self.cand_count += 1
        self.cand_total += 1
        self._broadcast_hit_candidate(str(cam), d)

    def _coerce_xy(self, value: Any) -> Optional[Tuple[float, float]]:
        try:
            if isinstance(value, dict):
                return (float(value.get("x")), float(value.get("y")))
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return (float(value[0]), float(value[1]))
        except Exception:
            return None
        return None

    def _candidate_hit_xy(self, d: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        # Candidate dictionaries from different code paths use slightly different
        # field names.  All candidates here are already in MVS pixel coordinates.
        for key in (
            "projected_hit_xy",
            "hit_point_mvs",
            "event_point",
            "segment_end_mvs",
            "projected_end_xy",
            "impact_xy",
            "impact_point_mvs",
        ):
            xy = self._coerce_xy(d.get(key))
            if xy is not None:
                return xy
        return None

    def _candidate_event_time_us(self, d: Dict[str, Any]) -> int:
        for key in ("event_ts", "impact_event_ts_us", "event_ts_us", "point_ts_us"):
            val = d.get(key)
            try:
                if val is not None and val != "":
                    return int(float(val))
            except Exception:
                continue
        try:
            return int(float(d.get("judge_wall_time", time.time())) * 1_000_000)
        except Exception:
            return int(time.time() * 1_000_000)

    def _mark_final_hit_output_key(self, cam: str, d: Dict[str, Any]) -> bool:
        if not bool(getattr(self, "final_hit_cross_camera_dedup_enable", True)):
            return True
        window_s = max(0.001, float(getattr(self, "final_hit_cross_camera_dedup_ms", 700.0) or 700.0) / 1000.0)
        episode_window_s = max(0.0, float(getattr(self, "final_hit_same_track_episode_dedup_ms", 1500.0) or 1500.0) / 1000.0)
        event_time_s = self._candidate_event_time_us(d) / 1_000_000.0
        identity = self._final_hit_dedup_identity(str(cam), d)
        if bool(getattr(self, "final_hit_same_track_episode_dedup_enable", True)) and episode_window_s > 0.0:
            expire_after = max(episode_window_s * 4.0, 8.0)
            for key, last_t in list(getattr(self, "final_hit_episode_seen", {}).items()):
                try:
                    if event_time_s - float(last_t) > expire_after:
                        self.final_hit_episode_seen.pop(key, None)
                except Exception:
                    self.final_hit_episode_seen.pop(key, None)
        for key in list(self.final_hit_seen_keys):
            try:
                old_t = float(key[1])
            except Exception:
                continue
            if event_time_s - old_t > max(window_s * 4.0, 1.0):
                self.final_hit_seen_set.discard(key)
        while self.final_hit_seen_keys:
            try:
                old_t = float(self.final_hit_seen_keys[0][1])
            except Exception:
                break
            if event_time_s - old_t <= max(window_s * 4.0, 1.0):
                break
            old = self.final_hit_seen_keys.popleft()
            self.final_hit_seen_set.discard(old)
        for key in list(self.final_hit_seen_keys):
            try:
                if key[0] == identity and abs(event_time_s - float(key[1])) <= window_s:
                    self.final_hit_duplicate_suppressed += 1
                    self.hit_reject_counts["final_hit_duplicate_suppressed"] += 1
                    return False
            except Exception:
                continue
        if (
            bool(getattr(self, "final_hit_same_track_episode_dedup_enable", True))
            and episode_window_s > 0.0
            and isinstance(identity, tuple)
            and len(identity) >= 1
            and identity[0] == "target_track"
        ):
            try:
                last_t = self.final_hit_episode_seen.get(identity)
                if last_t is not None and abs(event_time_s - float(last_t)) <= episode_window_s:
                    self.same_track_episode_duplicate_suppressed += 1
                    self.hit_reject_counts["same_track_episode_duplicate_suppressed"] += 1
                    d["_final_hit_suppress_reason"] = "same_track_episode_duplicate_suppressed"
                    return False
            except Exception:
                pass
        key = (identity, round(float(event_time_s), 6))
        if key in self.final_hit_seen_set:
            self.final_hit_duplicate_suppressed += 1
            self.hit_reject_counts["final_hit_duplicate_suppressed"] += 1
            return False
        if self.final_hit_seen_keys.maxlen is not None and len(self.final_hit_seen_keys) >= self.final_hit_seen_keys.maxlen:
            try:
                old = self.final_hit_seen_keys.popleft()
                self.final_hit_seen_set.discard(old)
            except Exception:
                pass
        self.final_hit_seen_keys.append(key)
        self.final_hit_seen_set.add(key)
        if (
            bool(getattr(self, "final_hit_same_track_episode_dedup_enable", True))
            and episode_window_s > 0.0
            and isinstance(identity, tuple)
            and len(identity) >= 1
            and identity[0] == "target_track"
        ):
            self.final_hit_episode_seen[identity] = float(event_time_s)
        return True

    def _final_hit_dedup_identity(self, cam: str, d: Dict[str, Any]) -> Tuple[Any, ...]:
        target = (
            d.get("target_global_id")
            or d.get("target_global_id_int")
            or d.get("victim_global_id")
            or d.get("target_person_id")
            or d.get("target_camera_local_id")
            or d.get("person_id")
            or d.get("stable_id")
            or "unknown"
        )
        track = (
            d.get("global_bullet_id")
            or d.get("track_key")
            or d.get("stable_track_id")
            or d.get("bullet_id")
            or d.get("display_bullet_id")
            or ""
        )
        if str(track).strip() not in {"", "-1", "0", "None", "none"}:
            return ("target_track", str(target), str(track))
        xy = self._candidate_hit_xy(d)
        if xy is not None:
            # 32px bucket is only a fallback when no track identity exists.
            return ("target_xy", str(target), str(cam), int(round(float(xy[0]) / 32.0)), int(round(float(xy[1]) / 32.0)))
        return ("target_time", str(target), str(cam))

    def _write_candidate_shadow_dict(self, cam: str, d: Dict[str, Any], shadow_only: bool, reason: str) -> None:
        if not bool(getattr(self, "hit_shadow_enable", True)):
            return
        chain = str(d.get("judge_chain", "strong_gate") or "strong_gate")
        if chain not in {"strong_gate", "strong_reasoning"}:
            chain = "strong_gate"
        rec = dict(d)
        rec.setdefault("type", "hit_candidate_shadow")
        rec["shadow_stream"] = chain
        rec["shadow_only"] = bool(shadow_only)
        rec["authoritative_output"] = not bool(shadow_only)
        rec["shadow_suppressed_reason"] = str(reason or "")
        rec.setdefault("final_hit_authority", str(getattr(self, "final_hit_authority", "strong_gate") or "strong_gate"))
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
        fp = getattr(self, "hit_shadow_files", {}).get(chain, {}).get(str(cam))
        if fp is not None:
            fp.write(line)
        all_fp = getattr(self, "hit_shadow_all_files", {}).get(chain)
        if all_fp is not None:
            all_fp.write(line)

    def _broadcast_hit_candidate(self, cam: str, d: Dict[str, Any]) -> None:
        """Broadcast hit candidates to UE5 over the overlay WebSocket.

        JSONL remains the audit output.  When Linux no longer renders OpenGL,
        UE5 needs the same hit point in real time.  Coordinates are MVS pixels.
        """
        hit_xy = self._candidate_hit_xy(d)
        if hit_xy is None:
            return
        display_w, display_h = self._display_size_for_cam(str(cam))
        x, y = float(hit_xy[0]), float(hit_xy[1])
        msg = dict(d)
        msg.update({
            "type": "overlay_hit_candidate",
            "schema_version": 1,
            "sync_run_id": str(self.hit_cfg.get("sync_run_id", "real_run_001") or "real_run_001"),
            "pair_id": str(cam),
            "marker_type": "hit",
            "state": "hit",
            "semantic_state": "hit",
            "hit": True,
            "hit_point_mvs": [round(x, 3), round(y, 3)],
            "x": round(x, 3),
            "y": round(y, 3),
            "display_width": int(display_w),
            "display_height": int(display_h),
            "source": "hit_judge_server",
            "reason": "ue5_hit_candidate",
            "wall_time": time.time(),
        })
        self._broadcast_overlay(msg)

        marker_msg = {
            "type": "overlay_bullet_marker",
            "schema_version": 1,
            "sync_run_id": msg.get("sync_run_id"),
            "pair_id": str(cam),
            "camera_sn": str(d.get("camera_sn", "")),
            "camera_alias": str(d.get("camera_alias", cam) or cam),
            "bullet_id": int(d.get("bullet_id", 0) or 0),
            "bullet_seq_id": int(d.get("seq_id", d.get("bullet_seq_id", 0)) or 0),
            "stable_id": int(d.get("stable_id", 0) or 0),
            "marker_type": "hit",
            "state": "hit",
            "semantic_state": "hit",
            "event_type": str(d.get("event_type", "")),
            "terminate_reason": str(d.get("terminate_reason", "")),
            "ts_us": int(d.get("event_ts", d.get("impact_event_ts_us", d.get("point_ts_us", 0))) or 0),
            "x": round(x, 3),
            "y": round(y, 3),
            "display_width": int(display_w),
            "display_height": int(display_h),
            "total_score": float(d.get("total_score", 0.0) or 0.0),
            "match_method": str(d.get("match_method", d.get("geometry_mode", "")) or ""),
            "evidence_level": str(d.get("evidence_level", "") or ""),
            "source": "hit_judge_server",
            "reason": "ue5_hit_marker",
            "wall_time": time.time(),
        }
        self._broadcast_overlay(marker_msg)

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
        evidence_gate_detail: Dict[str, Any] = {}
        if self._evidence_gated_final_enabled():
            gate_ok, gate_reason, evidence_gate_detail = self._final_evidence_gate_ok(
                pending.cam, None, pending.period, pending.selection, pending.geometry
            )
            if not gate_ok:
                dbg_ev = ev or pending.base_ev
                geom = dict(pending.geometry)
                geom["evidence_gate"] = evidence_gate_detail
                geom["impact_behavior_result"] = str(method)
                geom["impact_behavior_reason"] = str(reason)
                self._emit_debug(
                    pending.cam,
                    dbg_ev,
                    "evidence_gate_rejected",
                    str(gate_reason),
                    period=pending.period,
                    selection=pending.selection,
                    geometry=geom,
                    candidate=dict(pending.base_candidate),
                    age_ms=(time.time() - pending.created_wall) * 1000.0,
                )
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
        if self._evidence_gated_final_enabled():
            d.update({
                "judge_mode": "evidence_gated",
                "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
                "trajectory_evidence_used": True,
                "trigger_evidence_used": bool(pending.period is not None),
                "person_sync_evidence_used": True,
                "mask_evidence_used": True,
                "evidence_gate": evidence_gate_detail,
            })
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
        visual_approach_pt = _apply_H(H, float(ev.approach_x), float(ev.approach_y)) if bool(ev.approach_valid) else event_pt
        raw_approach_valid = bool(ev.approach_valid) and approach_len_px >= float(self.min_hit_trajectory_len_px)
        use_segment = bool(self.trajectory_segment_enable and raw_approach_valid)
        approach_pt = _apply_H(H, float(ev.approach_x), float(ev.approach_y)) if use_segment else event_pt

        # Full visual stream: emit before human/track gates, so UE can render all
        # mapped bullet tracks even when there is no person box or no hit candidate.
        track_points_mvs = self._project_track_points_mvs(H, ev, event_pt, visual_approach_pt)
        self._emit_visual_bullet_event(cam, ev, event_pt, visual_approach_pt, track_points_mvs, age_ms=age_ms)

        early_pending_result: Optional[str] = None
        if (not self._trajectory_only_final_enabled()) and bool(self.enable_impact_behavior_hit) and bool(self.impact_update_pending_before_human):
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

        self._try_emit_pseudo_hit_best_effort(
            cam,
            ev,
            event_pt,
            approach_pt,
            bool(use_segment),
            track_metrics,
            age_ms,
            force_final,
            source="pre_track_gate",
        )

        if self._trajectory_only_final_enabled():
            return self._emit_trajectory_only_hit(
                cam,
                ev,
                event_pt,
                approach_pt,
                track_metrics,
                judge_source="impact_event",
                use_segment=bool(use_segment),
                min_segment_len_px=float(getattr(self, "trajectory_only_min_approach_len_px", self.min_hit_approach_len_px)),
                age_ms=age_ms,
            )

        if self._evidence_gated_final_enabled():
            quality_ok, quality_reason, quality_detail = self._trajectory_only_quality(
                track_metrics,
                min_segment_len_px=float(getattr(self, "trajectory_only_min_approach_len_px", self.min_hit_approach_len_px)),
            )
            if not quality_ok:
                self._emit_debug(
                    cam,
                    ev,
                    "no_valid_trajectory",
                    str(quality_reason),
                    geometry={
                        "judge_source": "impact_event",
                        "evidence_gate": "trajectory_quality",
                        "event_point_mvs": [round(float(event_pt[0]), 3), round(float(event_pt[1]), 3)],
                        "approach_point_mvs": [round(float(approach_pt[0]), 3), round(float(approach_pt[1]), 3)],
                        **quality_detail,
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
            self._try_emit_pseudo_hit_for_event(
                cam,
                ev,
                event_pt,
                approach_pt,
                bool(use_segment),
                entries,
                period,
                selection_info,
                strategy_used,
                track_metrics,
                age_ms,
                source="impact_sync_entries",
            )

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
            self._try_emit_pseudo_hit_for_event(
                cam,
                ev,
                event_pt,
                approach_pt,
                bool(use_segment),
                entries,
                period,
                selection_info,
                strategy_used,
                track_metrics,
                age_ms,
                source="nearest_time_entries",
            )

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
                body_reject_detail = self._person_body_motion_reject_detail(
                    person, event_pt, approach_pt, bool(use_segment), float(approach_len_px)
                )
                if body_reject_detail is not None:
                    geometry_summary["body_motion_reject_last"] = body_reject_detail
                    geometry_summary["geometry_best_method"] = body_reject_detail.get("body_motion_reject_reason")
                    geometry_summary["evidence_level"] = "reject"
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

        evidence_gate_detail: Dict[str, Any] = {}
        if self._evidence_gated_final_enabled():
            gate_geometry = {**geometry_summary, **best_meta}
            gate_ok, gate_reason, evidence_gate_detail = self._final_evidence_gate_ok(
                cam, best_entry, period, selection_info, gate_geometry
            )
            if not gate_ok:
                self._emit_debug(
                    cam,
                    ev,
                    "evidence_gate_rejected",
                    str(gate_reason),
                    period=period,
                    selection=selection_info,
                    geometry={**geometry_summary, "evidence_gate": evidence_gate_detail},
                    candidate=dict(d),
                    age_ms=age_ms,
                )
                return "no_candidate"
            d.update({
                "judge_mode": "evidence_gated",
                "hit_judge_final_mode": str(getattr(self, "hit_judge_final_mode", "")),
                "trajectory_evidence_used": True,
                "trigger_evidence_used": bool(period is not None),
                "person_sync_evidence_used": True,
                "mask_evidence_used": True,
                "evidence_gate": evidence_gate_detail,
            })
        self._write_candidate_dict(cam, d)
        dbg_candidate = dict(d)
        self._emit_debug(cam, ev, "emitted_candidate", None, period=period, selection=selection_info, geometry=geometry_summary, candidate=dbg_candidate, age_ms=age_ms)
        print(f"[hit_judge][{cam}] impact candidate bullet={best.bullet_id} person={best.stable_id} score={best.total_score:.3f} offset={d.get('impact_offset_ms','?')}ms dt={best.dt_ms:.1f}ms method={best.match_method} strategy={strategy_used} fallback={d.get('human_fallback_used')}", flush=True)
        return "emitted"

    def run(self) -> None:
        try:
            while True:
                self._poll_control_json()
                self.poll_inputs()
                self.receive_udp()
                self._process_pending_visual_point_segments()
                self.process_pending_events()
                self._process_pending_impacts()
                self._flush_semantic_terminals()
                self._write_status()
                now = time.time()
                if now - self.last_print >= 2.0:
                    hist_info = " ".join([f"{c}:H{len(self.history.get(c, []))}/P{len(self.pending_without_ts.get(c, []))}/T{len(self.trigger_tailers.get(c).map if self.trigger_tailers.get(c) else {})}" for c in self.cam_ids])
                    vp_stage = getattr(self, "visual_point_stage_counts", {}) or {}
                    vp_info = (
                        f"vpoint_udp_total={int(vp_stage.get('udp_recv', 0) or 0)} "
                        f"vpoint_segment_total={int(vp_stage.get('segment_attempt', 0) or 0)} "
                        f"vpoint_wait_prev_total={int(vp_stage.get('waiting_prev_point', 0) or 0)} "
                        f"vpoint_result_emit_total={int(vp_stage.get('result:point_emitted', 0) or 0)}"
                    )
                    print(f"[hit_judge][perf] recv={self.recv_count} cand={self.cand_count} wait={self.wait_count} drop={self.drop_count} nonimpact={self.nonimpact_count} no_human={self.no_human_count} no_cam={self.no_cam_count} pending={len(self.pending_events)} pending_impacts={len(self.pending_impacts)} {vp_info} {hist_info}", flush=True)
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
    p.add_argument("--control-json", default="", help="Runtime control JSON for hit detection enable/output toggles")
    p.add_argument("--overlay-ws-enable", action="store_true", help="Broadcast overlay_frame_state/overlay_bullet_event to UE5 over WebSocket")
    p.add_argument("--overlay-ws-host", default=None, help="WebSocket bind host for UE5 overlay, default from config or 0.0.0.0")
    p.add_argument("--overlay-ws-port", type=int, default=None, help="WebSocket bind port for UE5 overlay, default from config or 8765")
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
