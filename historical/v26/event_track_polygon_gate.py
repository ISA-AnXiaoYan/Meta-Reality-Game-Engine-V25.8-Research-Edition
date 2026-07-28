"""V26-B shadow helpers for post-tracker human-polygon gating and ID evaluation.

The module is deliberately independent from the official C++ LineMotionFilter.
In shadow mode it never changes BulletEventSender input; it only evaluates the
same sanitized clusters through a polygon-gated branch and writes diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import statistics
import time
from typing import Any, Iterable

import numpy as np


def _camera_selected(spec: str, camera_alias: str) -> bool:
    wanted = {item.strip().upper() for item in str(spec or "").replace(";", ",").split(",") if item.strip()}
    return not wanted or "ALL" in wanted or "*" in wanted or str(camera_alias or "").upper() in wanted


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cluster_fields(clusters: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw_id/x/y/width/height arrays for structured arrays or dict rows."""
    if isinstance(clusters, np.ndarray) and clusters.dtype.fields is not None:
        fields = clusters.dtype.fields
        id_name = "id" if "id" in fields else "raw_id"
        return (
            np.asarray(clusters[id_name]),
            np.asarray(clusters["x"]),
            np.asarray(clusters["y"]),
            np.asarray(clusters["width"]),
            np.asarray(clusters["height"]),
        )
    rows = list(clusters or [])
    ids, xs, ys, ws, hs = [], [], [], [], []
    for row in rows:
        get = row.get if isinstance(row, dict) else lambda key, default=None: default
        ids.append(_as_int(get("id", get("raw_id", -1))))
        xs.append(_as_float(get("x", 0.0)))
        ys.append(_as_float(get("y", 0.0)))
        ws.append(max(0.0, _as_float(get("width", 0.0))))
        hs.append(max(0.0, _as_float(get("height", 0.0))))
    return tuple(np.asarray(items) for items in (ids, xs, ys, ws, hs))


def _subset_clusters(clusters: Any, keep: np.ndarray) -> Any:
    if isinstance(clusters, np.ndarray):
        return clusters[keep]
    return [row for row, item_keep in zip(list(clusters or []), keep.tolist()) if item_keep]


class RateLimitedJsonlWriter:
    def __init__(self, template: str, camera_alias: str, max_fps: float):
        raw = str(template or "").strip()
        try:
            self.path = raw.format(camera=camera_alias, cam=camera_alias, alias=camera_alias)
        except Exception:
            self.path = raw
        self.max_fps = max(0.0, float(max_fps or 0.0))
        self.next_write_s = 0.0
        self.fp = None
        self.error = ""

    def write(self, row: dict[str, Any]) -> bool:
        if not self.path:
            return False
        now = time.monotonic()
        if self.max_fps > 0.0 and now < self.next_write_s:
            return False
        if self.max_fps > 0.0:
            self.next_write_s = now + 1.0 / self.max_fps
        try:
            if self.fp is None:
                folder = os.path.dirname(os.path.abspath(self.path))
                if folder:
                    os.makedirs(folder, exist_ok=True)
                self.fp = open(self.path, "a", encoding="utf-8")
            self.fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.fp.flush()
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}:{exc}"
            return False


class EventTrackPolygonGate:
    """Cluster-level gate fed by the same-sync polygon mask provider.

    `mask_provider` must expose `get_current_mask(sync_index, slice_ts_us,
    current_sync_event_ts_us) -> (mask, info, reason)`. The gate owns HOLD;
    the provider remains strict-current-frame and never uses carry-last data.
    """

    def __init__(self, args: Any, width: int, height: int, camera_alias: str, mask_provider: Any):
        self.mode = str(getattr(args, "event_track_polygon_gate_mode", "off") or "off").strip().lower()
        self.camera_alias = str(camera_alias or "")
        self.enabled = self.mode in {"shadow", "active"} and _camera_selected(
            getattr(args, "event_track_polygon_gate_cameras", ""), self.camera_alias
        )
        self.width = int(width)
        self.height = int(height)
        self.mask_provider = mask_provider
        self.hold_enable = bool(getattr(args, "event_track_polygon_gate_hold_enable", True))
        self.hold_us = max(0, int(round(float(getattr(args, "event_track_polygon_gate_hold_ms", 30.0) or 0.0) * 1000.0)))
        self.hold_max_sync_gap = max(0, int(getattr(args, "event_track_polygon_gate_hold_max_sync_gap", 2) or 0))
        self._last_mask = None
        self._last_info: dict[str, Any] | None = None
        self._last_valid_ts = -1
        self._last_valid_sync = -1
        self._next_empty_probe_s = 0.0
        self._empty_probe_interval_s = max(0.2, 1.0 / max(0.1, float(getattr(args, "event_track_polygon_gate_log_max_fps", 5.0) or 5.0)))
        self.last_diag: dict[str, Any] = {"mode": self.mode, "enabled": self.enabled, "source": "NONE"}
        self.logger = RateLimitedJsonlWriter(
            getattr(args, "event_track_polygon_gate_log_jsonl", ""),
            self.camera_alias,
            getattr(args, "event_track_polygon_gate_log_max_fps", 5.0),
        )

    @staticmethod
    def _empty_diag(mode: str, enabled: bool, ts: int, sync_index: Any, count: int, reason: str) -> dict[str, Any]:
        return {
            "mode": str(mode),
            "enabled": bool(enabled),
            "slice_ts_us": int(ts or 0),
            "sync_index": None if sync_index is None else _as_int(sync_index),
            "source": "NONE",
            "reason": str(reason),
            "input_cluster_count": int(count),
            "gated_cluster_count": int(count),
            "filtered_cluster_count": 0,
            "filtered_cluster_ids": [],
            "mask_available": False,
            "mask_pixels": 0,
            "hold_age_us": 0,
            "hold_sync_gap": 0,
        }

    def apply(self, clusters: Any, ts: int, sync_index: Any, current_sync_event_ts_us: Any = None) -> tuple[Any, Any, dict[str, Any]]:
        try:
            count = int(len(clusters)) if clusters is not None else 0
        except Exception:
            count = 0
        diag = self._empty_diag(self.mode, self.enabled, ts, sync_index, count, "disabled" if not self.enabled else "init")
        if not self.enabled:
            self.last_diag = diag
            return clusters, clusters, diag
        if count <= 0:
            # Empty SpatterTracker output is common in a quiet scene. Probe the
            # same-sync Polygon source at a bounded rate so online evidence can
            # distinguish "no target" from "gate not receiving polygons".
            now_s = time.monotonic()
            if now_s >= self._next_empty_probe_s:
                self._next_empty_probe_s = now_s + self._empty_probe_interval_s
                mask, info, reason = self.mask_provider.get_current_mask(
                    current_sync_index=sync_index,
                    slice_ts_us=int(ts),
                    current_sync_event_ts_us=current_sync_event_ts_us,
                )
                if mask is not None and isinstance(info, dict) and int(info.get("mask_pixels", 0) or 0) > 0:
                    diag.update({
                        "source": "LIVE",
                        "reason": str(reason),
                        "mask_available": True,
                        "mask_pixels": int(info.get("mask_pixels", 0) or 0),
                        "mask_sync_index": info.get("sync_index"),
                        "mask_used_persons": int(info.get("used_persons", 0) or 0),
                    })
                else:
                    diag["reason"] = str(reason)
                self.last_diag = diag
                self.logger.write(diag)
                return clusters, clusters, diag
            cached = dict(self.last_diag)
            cached.update({
                "slice_ts_us": int(ts or 0),
                "sync_index": None if sync_index is None else _as_int(sync_index),
                "input_cluster_count": 0,
                "gated_cluster_count": 0,
                "filtered_cluster_count": 0,
            })
            return clusters, clusters, cached

        mask, info, reason = self.mask_provider.get_current_mask(
            current_sync_index=sync_index,
            slice_ts_us=int(ts),
            current_sync_event_ts_us=current_sync_event_ts_us,
        )
        source = "NONE"
        hold_age_us = 0
        hold_sync_gap = 0
        current_sync = _as_int(sync_index, -1)
        if mask is not None and isinstance(info, dict) and int(info.get("mask_pixels", 0) or 0) > 0:
            source = "LIVE"
            self._last_mask = mask
            self._last_info = dict(info)
            self._last_valid_ts = int(ts)
            self._last_valid_sync = current_sync
        elif self.hold_enable and self._last_mask is not None and self._last_info is not None:
            hold_age_us = max(0, int(ts) - int(self._last_valid_ts))
            hold_sync_gap = abs(current_sync - int(self._last_valid_sync)) if current_sync >= 0 and self._last_valid_sync >= 0 else 0
            if hold_age_us <= self.hold_us and hold_sync_gap <= self.hold_max_sync_gap:
                mask = self._last_mask
                info = dict(self._last_info)
                source = "HOLD"
                reason = f"hold:{reason}"
            else:
                reason = "hold_expired" if hold_age_us > self.hold_us else "hold_sync_gap"

        if mask is None or not isinstance(info, dict) or int(info.get("mask_pixels", 0) or 0) <= 0:
            diag.update({"reason": str(reason), "hold_age_us": hold_age_us, "hold_sync_gap": hold_sync_gap})
            self.last_diag = diag
            self.logger.write(diag)
            return clusters, clusters, diag

        try:
            raw_ids, xs, ys, ws, hs = _cluster_fields(clusters)
            cx = xs.astype(np.float64, copy=False) + 0.5 * ws.astype(np.float64, copy=False)
            cy = ys.astype(np.float64, copy=False) + 0.5 * hs.astype(np.float64, copy=False)
            ix = np.rint(cx).astype(np.int32, copy=False)
            iy = np.rint(cy).astype(np.int32, copy=False)
            valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
            inside = np.zeros(len(ix), dtype=bool)
            inside[valid] = np.asarray(mask)[iy[valid], ix[valid]] > 0
        except Exception as exc:
            diag.update({"reason": f"cluster_center_error:{type(exc).__name__}", "source": source})
            self.last_diag = diag
            self.logger.write(diag)
            return clusters, clusters, diag

        filtered_ids = [_as_int(value) for value in raw_ids[inside].tolist()]
        gated = _subset_clusters(clusters, ~inside)
        diag.update({
            "source": source,
            "reason": str(reason),
            "mask_available": True,
            "mask_pixels": int(info.get("mask_pixels", 0) or 0),
            "mask_sync_index": info.get("sync_index"),
            "mask_used_persons": int(info.get("used_persons", 0) or 0),
            "gated_cluster_count": int(len(gated)),
            "filtered_cluster_count": int(np.count_nonzero(inside)),
            "filtered_cluster_ids": filtered_ids[:64],
            "filtered_cluster_ids_truncated": bool(len(filtered_ids) > 64),
            "hold_age_us": int(hold_age_us),
            "hold_sync_gap": int(hold_sync_gap),
        })
        self.last_diag = diag
        self.logger.write(diag)
        primary = gated if self.mode == "active" else clusters
        return primary, gated, diag


@dataclass
class _BusinessV1State:
    bullet_id: int
    first_ts: int
    last_ts: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    observations: int = 0
    confirmed: bool = False
    cloud_suspect: bool = False
    history: list[tuple[float, float, int, float, float]] = field(default_factory=list)


class BusinessIdV1Shadow:
    """HotUI-v1-inspired, non-authoritative association for gated clusters."""

    def __init__(self, args: Any, camera_alias: str):
        self.enabled = bool(getattr(args, "event_business_id_v1_shadow_enable", False)) and _camera_selected(
            getattr(args, "event_business_id_v1_shadow_cameras", ""), camera_alias
        )
        self.camera_alias = str(camera_alias or "")
        self.confirm_min_observations = max(2, int(getattr(args, "event_business_id_v1_confirm_min_observations", 4) or 4))
        self.confirm_min_displacement_px = max(1.0, float(getattr(args, "event_business_id_v1_confirm_min_displacement_px", 20.0) or 20.0))
        self.confirm_min_speed_px_ms = max(0.01, float(getattr(args, "event_business_id_v1_confirm_min_speed_px_ms", 0.32) or 0.32))
        self.assoc_radius_px = max(2.0, float(getattr(args, "event_business_id_v1_assoc_radius_px", 28.0) or 28.0))
        self.max_gap_us = max(1_000, int(round(float(getattr(args, "event_business_id_v1_max_gap_ms", 120.0) or 120.0) * 1000.0)))
        self.collision_window_us = max(self.max_gap_us, int(round(float(getattr(args, "event_business_id_v1_collision_window_ms", 260.0) or 260.0) * 1000.0)))
        self.collision_max_gap_px = max(2.0, float(getattr(args, "event_business_id_v1_collision_max_gap_px", 95.0) or 95.0))
        self.states: dict[int, _BusinessV1State] = {}
        self.next_id = 1
        self.last_diag: dict[str, Any] = {"enabled": self.enabled}
        self.logger = RateLimitedJsonlWriter(
            getattr(args, "event_business_id_v1_shadow_log_jsonl", ""), camera_alias,
            getattr(args, "event_business_id_v1_shadow_log_max_fps", 5.0),
        )

    @staticmethod
    def _motion(state: _BusinessV1State) -> tuple[float, float, float]:
        if len(state.history) < 2:
            return 0.0, 0.0, 0.0
        first = state.history[0]
        last = state.history[-1]
        dx, dy = last[0] - first[0], last[1] - first[1]
        displacement = math.hypot(dx, dy)
        dt_ms = max(0.001, (last[2] - first[2]) / 1000.0)
        return displacement, displacement / dt_ms, dt_ms

    @staticmethod
    def _linearity(state: _BusinessV1State) -> float:
        if len(state.history) < 3:
            return 0.0
        first, last = state.history[0], state.history[-1]
        dx, dy = last[0] - first[0], last[1] - first[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return float("inf")
        residuals = [abs(dy * (p[0] - first[0]) - dx * (p[1] - first[1])) / norm for p in state.history[1:-1]]
        return float(statistics.median(residuals)) if residuals else 0.0

    def _expire(self, ts: int) -> None:
        self.states = {bid: state for bid, state in self.states.items() if int(ts) - state.last_ts <= self.collision_window_us}

    def _best_state(self, x: float, y: float, ts: int, used: set[int]) -> tuple[_BusinessV1State | None, str, float]:
        best: tuple[float, _BusinessV1State, str] | None = None
        for state in self.states.values():
            if state.bullet_id in used:
                continue
            dt_us = max(0, int(ts) - state.last_ts)
            if dt_us > self.collision_window_us:
                continue
            dt_ms = dt_us / 1000.0
            px = state.x + state.vx * dt_ms
            py = state.y + state.vy * dt_ms
            distance = math.hypot(x - px, y - py)
            normal = dt_us <= self.max_gap_us and distance <= self.assoc_radius_px
            collision = state.confirmed and dt_us <= self.collision_window_us and distance <= self.collision_max_gap_px
            if not normal and not collision:
                continue
            kind = "assoc" if normal else "collision"
            score = distance + (0.0 if normal else self.assoc_radius_px)
            if best is None or score < best[0]:
                best = (score, state, kind)
        return (best[1], best[2], best[0]) if best is not None else (None, "new", 0.0)

    def update(self, clusters: Any, ts: int, gate_diag: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.enabled:
            self.last_diag = {"enabled": False, "reason": "disabled", "slice_ts_us": int(ts)}
            return [], dict(self.last_diag)
        self._expire(ts)
        raw_ids, xs, ys, ws, hs = _cluster_fields(clusters)
        used: set[int] = set()
        assignments: list[dict[str, Any]] = []
        collision_links = 0
        for raw_id, x0, y0, w0, h0 in zip(raw_ids.tolist(), xs.tolist(), ys.tolist(), ws.tolist(), hs.tolist()):
            x, y, w, h = _as_float(x0), _as_float(y0), max(0.0, _as_float(w0)), max(0.0, _as_float(h0))
            cx, cy = x + 0.5 * w, y + 0.5 * h
            state, link_kind, link_cost = self._best_state(cx, cy, int(ts), used)
            if state is None:
                state = _BusinessV1State(self.next_id, int(ts), int(ts), cx, cy)
                self.states[state.bullet_id] = state
                self.next_id += 1
            else:
                dt_ms = max(0.001, (int(ts) - state.last_ts) / 1000.0)
                state.vx = (cx - state.x) / dt_ms
                state.vy = (cy - state.y) / dt_ms
            state.x, state.y, state.last_ts = cx, cy, int(ts)
            state.observations += 1
            state.history.append((cx, cy, int(ts), w, h))
            if len(state.history) > 32:
                state.history.pop(0)
            displacement, speed, _ = self._motion(state)
            residual = self._linearity(state)
            state.cloud_suspect = bool(state.observations >= self.confirm_min_observations and (speed < self.confirm_min_speed_px_ms or residual > self.assoc_radius_px * 0.65))
            if not state.confirmed and not state.cloud_suspect:
                state.confirmed = bool(
                    state.observations >= self.confirm_min_observations
                    and displacement >= self.confirm_min_displacement_px
                    and speed >= self.confirm_min_speed_px_ms
                )
            if link_kind == "collision":
                collision_links += 1
            used.add(state.bullet_id)
            assignments.append({
                "raw_id": _as_int(raw_id), "business_v1_id": state.bullet_id,
                "status": "confirmed" if state.confirmed else "candidate",
                "cloud_suspect": bool(state.cloud_suspect), "link_kind": link_kind,
                "link_cost_px": round(float(link_cost), 3), "observations": state.observations,
                "displacement_px": round(displacement, 3), "speed_px_ms": round(speed, 5),
                "line_residual_px": round(residual, 3) if math.isfinite(residual) else None,
                "cx": round(cx, 3), "cy": round(cy, 3), "width": round(w, 3), "height": round(h, 3),
            })
        confirmed = sum(1 for state in self.states.values() if state.confirmed)
        cloud = sum(1 for state in self.states.values() if state.cloud_suspect)
        diag = {
            "enabled": True, "slice_ts_us": int(ts), "input_cluster_count": int(len(assignments)),
            "active_ids": int(len(self.states)), "confirmed_ids": int(confirmed), "cloud_suspect_ids": int(cloud),
            "collision_links": int(collision_links), "gate_source": (gate_diag or {}).get("source", "NONE"),
            "gate_filtered_cluster_count": int((gate_diag or {}).get("filtered_cluster_count", 0) or 0),
            "assignments": assignments,
        }
        self.last_diag = diag
        self.logger.write(diag)
        return assignments, diag


@dataclass
class _BusinessV3State:
    status: str = "candidate"
    diagonal_seed: list[float] = field(default_factory=list)
    diagonal_baseline: float = 0.0
    outlier_streak: int = 0
    size_jump: bool = False
    last_ts: int = 0


class BusinessIdV3Shadow:
    """Behavior-state and diagonal-size evaluator for V26-B shadow only."""

    def __init__(self, args: Any, camera_alias: str):
        self.mode = str(getattr(args, "event_business_id_v3_mode", "off") or "off").strip().lower()
        self.enabled = self.mode == "shadow" and _camera_selected(
            getattr(args, "event_business_id_v3_cameras", ""), camera_alias
        )
        self.warmup = max(3, int(getattr(args, "event_business_id_v3_size_warmup_observations", 6) or 6))
        self.max_size_ratio = max(1.1, float(getattr(args, "event_business_id_v3_max_size_ratio", 3.0) or 3.0))
        self.outlier_consecutive = max(1, int(getattr(args, "event_business_id_v3_size_outlier_consecutive", 3) or 3))
        self.states: dict[int, _BusinessV3State] = {}
        self.logger = RateLimitedJsonlWriter(
            getattr(args, "event_business_id_v3_log_jsonl", ""), camera_alias,
            getattr(args, "event_business_id_v3_log_max_fps", 5.0),
        )
        self.last_diag: dict[str, Any] = {"enabled": self.enabled, "mode": self.mode}

    def update(self, assignments: Iterable[dict[str, Any]], ts: int, gate_diag: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled:
            self.last_diag = {"enabled": False, "mode": self.mode, "reason": "disabled", "slice_ts_us": int(ts)}
            return dict(self.last_diag)
        rows = []
        counts = {"candidate": 0, "probation": 0, "confirmed": 0, "cloud": 0, "size_jump": 0}
        for item in assignments:
            bid = _as_int(item.get("business_v1_id"), -1)
            if bid < 0:
                continue
            state = self.states.setdefault(bid, _BusinessV3State())
            diagonal = math.hypot(_as_float(item.get("width")), _as_float(item.get("height")))
            if diagonal > 0.0 and len(state.diagonal_seed) < self.warmup:
                state.diagonal_seed.append(diagonal)
                if len(state.diagonal_seed) >= self.warmup:
                    state.diagonal_baseline = float(statistics.median(state.diagonal_seed))
            ratio = diagonal / state.diagonal_baseline if state.diagonal_baseline > 1e-6 else 1.0
            outlier = state.diagonal_baseline > 1e-6 and ratio > self.max_size_ratio
            state.outlier_streak = state.outlier_streak + 1 if outlier else 0
            state.size_jump = state.outlier_streak >= self.outlier_consecutive
            if bool(item.get("cloud_suspect", False)):
                state.status = "cloud"
            elif bool(item.get("status") == "confirmed") and not state.size_jump:
                state.status = "confirmed"
            elif int(item.get("observations", 0) or 0) >= 2:
                state.status = "probation"
            else:
                state.status = "candidate"
            state.last_ts = int(ts)
            counts[state.status] = counts.get(state.status, 0) + 1
            if state.size_jump:
                counts["size_jump"] += 1
            rows.append({
                "business_v1_id": bid, "status": state.status, "size_metric": "bbox_diagonal_median_baseline",
                "diagonal_px": round(diagonal, 3), "baseline_diagonal_px": round(state.diagonal_baseline, 3),
                "size_ratio": round(ratio, 3), "outlier_streak": state.outlier_streak,
                "size_jump": bool(state.size_jump), "collision_allowed": bool(state.status == "confirmed"),
            })
        diag = {
            "enabled": True, "mode": "shadow", "slice_ts_us": int(ts), "counts": counts,
            "gate_source": (gate_diag or {}).get("source", "NONE"),
            "gate_filtered_cluster_count": int((gate_diag or {}).get("filtered_cluster_count", 0) or 0),
            "assignments": rows,
        }
        self.last_diag = diag
        self.logger.write(diag)
        return diag
