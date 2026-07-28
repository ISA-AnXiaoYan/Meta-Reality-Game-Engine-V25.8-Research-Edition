#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent PyOpenGL + CUDA preview renderer.

Design goals for the 8-camera fusion project:
  - Video stream updates at 40 FPS.
  - Bullet trajectory redraws at 250 FPS.
  - Video updates do not run at 250 FPS and do not repeatedly upload unchanged frames.
  - Preview consumes raw Bayer8 shared memory and performs Bayer->RGBA+resize on CUDA.
  - Bullet trajectories are consumed from a compact binary shm and rendered as GPU VBO lines.

This renderer is intentionally independent from fusion_renderer_gl.py.  It is a
preview-only consumer and must not back-pressure MVS, YOLO, fusion, or hit logic.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import glob
import csv
import hashlib
import json
import os
import re
import mmap
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QSurfaceFormat
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import QApplication, QPlainTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem, QFileDialog
except Exception as exc:  # pragma: no cover
    print(f"[pygl_cuda_preview][ERROR] PySide6 required: {exc}", file=sys.stderr, flush=True)
    raise

try:
    from OpenGL import GL
except Exception as exc:  # pragma: no cover
    print(f"[pygl_cuda_preview][ERROR] PyOpenGL required: {exc}", file=sys.stderr, flush=True)
    raise

from preview_gpu_shm import (
    RawPreviewFrameReader,
    PreviewTrajectoryReader,
    RAW_PIXEL_BAYER8,
    _R_FRAME_ID,
    _R_UPDATE_MONO_NS,
)


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_cams(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").replace(";", ",").split(",") if x.strip()]


def _shm_name_from_template(template: str, cam: str) -> str:
    return str(template or "mvs_raw_latest_{camera}").format(camera=cam, cam=cam, id=cam)


def _grid_layout(cams: List[str], layout: str) -> Tuple[int, int]:
    s = str(layout or "grid2x4").lower().strip()
    if s.startswith("grid") and "x" in s:
        try:
            a = s.replace("grid", "").split("x")
            rows = int(a[0]); cols = int(a[1])
            return max(1, rows), max(1, cols)
        except Exception:
            pass
    n = max(1, len(cams))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


CUDA_KERNEL = r'''
extern "C" __device__ unsigned char clamp_u8(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

extern "C" __device__ unsigned char src_at(const unsigned char* src, int w, int h, int stride, int x, int y) {
    x = x < 0 ? 0 : (x >= w ? w - 1 : x);
    y = y < 0 ? 0 : (y >= h ? h - 1 : y);
    return src[y * stride + x];
}

extern "C" __device__ int color_at(int x, int y, int pattern) {
    // return 0=R, 1=G, 2=B. pattern: 1=BG, 2=RG, 3=GB, 4=GR.
    int ey = (y & 1) == 0;
    int ex = (x & 1) == 0;
    if (pattern == 1) { // BG: B G / G R
        if (ey && ex) return 2;
        if (!ey && !ex) return 0;
        return 1;
    }
    if (pattern == 2) { // RG: R G / G B
        if (ey && ex) return 0;
        if (!ey && !ex) return 2;
        return 1;
    }
    if (pattern == 3) { // GB: G B / R G
        if (ey && !ex) return 2;
        if (!ey && ex) return 0;
        return 1;
    }
    // GR: G R / B G
    if (ey && !ex) return 0;
    if (!ey && ex) return 2;
    return 1;
}

extern "C" __device__ unsigned char channel_avg(const unsigned char* src, int w, int h, int stride, int x, int y, int pattern, int wanted) {
    int sum = 0;
    int cnt = 0;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
            if (color_at(xx, yy, pattern) == wanted) {
                sum += (int)src_at(src, w, h, stride, xx, yy);
                cnt += 1;
            }
        }
    }
    if (cnt <= 0) return src_at(src, w, h, stride, x, y);
    return clamp_u8(sum / cnt);
}

extern "C" __global__ void bayer_resize_to_rgba(
    const unsigned char* src,
    int src_w, int src_h, int src_stride,
    unsigned char* dst,
    int dst_w, int dst_h,
    int pattern
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;
    float sx_f = ((float)x + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;
    float sy_f = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int sx = (int)(sx_f + 0.5f);
    int sy = (int)(sy_f + 0.5f);
    unsigned char r = channel_avg(src, src_w, src_h, src_stride, sx, sy, pattern, 0);
    unsigned char g = channel_avg(src, src_w, src_h, src_stride, sx, sy, pattern, 1);
    unsigned char b = channel_avg(src, src_w, src_h, src_stride, sx, sy, pattern, 2);
    int off = (y * dst_w + x) * 4;
    dst[off + 0] = r;
    dst[off + 1] = g;
    dst[off + 2] = b;
    dst[off + 3] = 255;
}
'''


def _now_ms() -> float:
    return time.time() * 1000.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _safe_gid_int(v: Any, default: int = -1) -> int:
    if isinstance(v, str):
        raw = v.strip().upper()
        if raw.startswith("G") or raw.startswith("P"):
            raw = raw[1:]
        return _safe_int(raw, default)
    return _safe_int(v, default)


_MARKER_COLORS = {
    "hit": (1.0, 0.08, 0.02, 1.0),
    "turn": (1.0, 0.85, 0.10, 1.0),
    "end": (0.2, 0.8, 1.0, 1.0),
    "terminal": (0.2, 0.8, 1.0, 1.0),
    "track_probe": (0.7, 0.4, 1.0, 1.0),
    "raw_turn": (1.0, 0.55, 0.0, 0.8),
    "raw_end": (0.4, 0.9, 1.0, 0.8),
    "point": (1.0, 0.2, 0.05, 0.55),
    "event": (0.4, 1.0, 0.4, 0.75),
    "status": (0.7, 0.9, 1.0, 1.0),
}


def _marker_color(kind: str) -> Tuple[float, float, float, float]:
    return _MARKER_COLORS.get(str(kind or "event"), (1.0, 0.35, 0.10, 1.0))


class OverlayState:
    """Thread-safe small overlay state for the independent CUDA preview.

    Video is intentionally not affected by this state.  Overlay data is small
    and is drawn as OpenGL primitives / Qt text after the video atlas pass.
    """

    def __init__(self, cams: List[str], max_markers: int = 4096, max_text: int = 256):
        self.cams = list(cams)
        self.cam_to_index = {str(c): i for i, c in enumerate(self.cams)}
        self.lock = threading.RLock()
        self.frame_sizes: Dict[str, Tuple[int, int]] = {}
        self.markers: deque = deque(maxlen=int(max_markers))
        self.texts: deque = deque(maxlen=int(max_text))
        # Large, short-lived HIT prompts drawn independently from ordinary text.
        # These are fed by overlay_hit_candidate websocket messages and by
        # hit_candidate_*.jsonl fallback records.
        self.hit_prompts: deque = deque(maxlen=64)
        self.ws_connected = False
        self.ws_error = ""
        self.ws_rx_count = 0
        self.ws_last_ms = 0.0
        self.hit_jsonl_rx_count = 0
        self.last_hit_ms = 0.0
        self.human_jsonl_rx_count = 0
        self.last_human_ms = 0.0
        self.humans_by_cam: Dict[str, Dict[str, Any]] = {}
        # Lightweight diagnostic histories used only by the separate status window.
        # They do not affect video/trajectory/human/hit rendering paths.
        self.bullet_points: deque = deque(maxlen=1024)
        self.bullet_events: deque = deque(maxlen=512)
        self.hit_decisions: deque = deque(maxlen=512)

    def _cam_from_msg(self, msg: Dict[str, Any]) -> str:
        cam = msg.get("pair_id", msg.get("camera_alias", msg.get("cam", msg.get("camera", ""))))
        return str(cam or "").strip()

    def _display_size_from_msg(self, cam: str, msg: Dict[str, Any]) -> Tuple[float, float]:
        dw = _safe_float(msg.get("display_width", 0), 0.0)
        dh = _safe_float(msg.get("display_height", 0), 0.0)
        if dw <= 1 or dh <= 1:
            with self.lock:
                if cam in self.frame_sizes:
                    dw, dh = self.frame_sizes[cam]
        return max(float(dw), 1.0), max(float(dh), 1.0)

    def set_ws_status(self, connected: bool, err: str = "") -> None:
        with self.lock:
            self.ws_connected = bool(connected)
            self.ws_error = str(err or "")[:160]

    def _append_text_locked(self, text: str, cam: str = "", ttl_ms: float = 1800.0, level: str = "info") -> None:
        self.texts.append({
            "text": str(text)[:240],
            "cam": str(cam or ""),
            "ts_ms": _now_ms(),
            "ttl_ms": float(ttl_ms),
            "level": str(level or "info"),
        })

    def _append_marker_locked(self, cam: str, x: float, y: float, dw: float, dh: float,
                              kind: str, label: str = "", ttl_ms: float = 1500.0,
                              radius_px: float = 11.0, color: Optional[Tuple[float, float, float, float]] = None) -> None:
        if not cam or cam not in self.cam_to_index:
            return
        self.markers.append({
            "cam": str(cam),
            "x": float(x),
            "y": float(y),
            "display_w": max(float(dw), 1.0),
            "display_h": max(float(dh), 1.0),
            "kind": str(kind or "event"),
            "label": str(label or "")[:80],
            "ts_ms": _now_ms(),
            "ttl_ms": float(ttl_ms),
            "radius_px": float(radius_px),
            "color": tuple(color or _marker_color(kind)),
        })

    def _append_hit_prompt_locked(self, cam: str, x: float, y: float, dw: float, dh: float,
                                  label: str, ttl_ms: float = 3600.0,
                                  score: Optional[float] = None, stable_id: Any = None,
                                  bullet_id: Any = None, method: str = "") -> None:
        if not cam or cam not in self.cam_to_index:
            return
        prompt = {
            "cam": str(cam),
            "x": float(x),
            "y": float(y),
            "display_w": max(float(dw), 1.0),
            "display_h": max(float(dh), 1.0),
            "label": str(label or "HIT")[:160],
            "score": None if score is None else float(score),
            "stable_id": stable_id,
            "bullet_id": bullet_id,
            "method": str(method or "")[:80],
            "ts_ms": _now_ms(),
            "ttl_ms": float(ttl_ms),
        }
        self.hit_prompts.append(prompt)

    def update_from_overlay_msg(self, msg: Dict[str, Any], source: str = "ws") -> None:
        if not isinstance(msg, dict):
            return
        typ = str(msg.get("type", "") or "")
        cam = self._cam_from_msg(msg)
        now = _now_ms()
        with self.lock:
            if source == "ws":
                self.ws_rx_count += 1
                self.ws_last_ms = now
            else:
                self.hit_jsonl_rx_count += 1
                self.last_hit_ms = now

            if typ == "overlay_frame_state":
                dw = _safe_int(msg.get("display_width", 0), 0)
                dh = _safe_int(msg.get("display_height", 0), 0)
                if cam and dw > 0 and dh > 0:
                    self.frame_sizes[cam] = (dw, dh)
                return

            # Normalize hit_candidate JSONL records that may not carry an overlay type.
            if not typ and ("hit_point_mvs" in msg or "total_score" in msg or "stable_id" in msg):
                typ = "overlay_hit_candidate"

            if typ in {"overlay_hit_candidate", "hit_candidate"}:
                dw, dh = self._display_size_from_msg(cam, msg)
                xy = msg.get("hit_point_mvs") or msg.get("hit_xy") or msg.get("point_mvs")
                if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                    x, y = _safe_float(xy[0]), _safe_float(xy[1])
                else:
                    x, y = _safe_float(msg.get("x")), _safe_float(msg.get("y"))
                sid = msg.get("stable_id", None)
                bid = msg.get("bullet_id", msg.get("bullet_seq_id", None))
                score = _safe_float(msg.get("total_score", msg.get("score", 0.0)), 0.0)
                method = str(msg.get("match_method", msg.get("geometry_mode", "")) or "")
                label = "HIT 命中"
                if sid is not None:
                    label += f" id={_safe_int(sid)}"
                if score > 0:
                    label += f" s={score:.2f}"
                if bid is not None:
                    label += f" b={_safe_int(bid)}"
                self._append_marker_locked(cam, x, y, dw, dh, "hit", label, ttl_ms=4200.0, radius_px=20.0)
                self._append_hit_prompt_locked(cam, x, y, dw, dh, label, ttl_ms=4200.0,
                                               score=score, stable_id=sid, bullet_id=bid, method=method)
                self._append_text_locked(f"{cam} {label} px=({x:.0f},{y:.0f}) {method}", cam=cam, ttl_ms=4200.0, level="hit")
                self.hit_decisions.append({
                    "ts_ms": now,
                    "cam": cam,
                    "result": "HIT",
                    "reason": str(msg.get("reject_reason", msg.get("reason", "hit_candidate")) or "hit_candidate"),
                    "bullet_id": bid,
                    "stable_id": sid,
                    "score": score,
                    "method": method,
                    "event_ts_us": _safe_int(msg.get("impact_event_ts_us", msg.get("event_ts", msg.get("point_ts_us", 0))), 0),
                    "impact_offset_ms": _safe_float(msg.get("impact_offset_ms", msg.get("offset_ms", 0.0)), 0.0),
                    "person_frame_delta_sync": _safe_int(msg.get("person_frame_delta_sync", 999999), 999999),
                    "human_fallback_used": bool(msg.get("human_fallback_used", False)),
                    "mask_distance_px": _safe_float(msg.get("last_point_to_mask_dist_px", msg.get("impact_approach_to_mask_dist_px", -1.0)), -1.0),
                })
                return

            if typ in {"overlay_bullet_marker", "overlay_bullet_event"}:
                dw, dh = self._display_size_from_msg(cam, msg)
                x, y = _safe_float(msg.get("x")), _safe_float(msg.get("y"))
                kind = str(msg.get("marker_type", msg.get("state", "event")) or "event")
                ttl = 1800.0 if kind != "hit" else 2500.0
                label = kind
                bid = msg.get("bullet_id", None)
                if bid is not None:
                    label += f" #{_safe_int(bid)}"
                self._append_marker_locked(cam, x, y, dw, dh, kind, label, ttl_ms=ttl, radius_px=12.0)
                if kind == "hit":
                    score = _safe_float(msg.get("total_score", msg.get("score", 0.0)), 0.0)
                    sid = msg.get("stable_id", None)
                    bid2 = msg.get("bullet_id", None)
                    method = str(msg.get("match_method", msg.get("geometry_mode", "")) or "")
                    hit_label = "HIT 命中"
                    if sid is not None:
                        hit_label += f" id={_safe_int(sid)}"
                    if score > 0:
                        hit_label += f" s={score:.2f}"
                    self._append_hit_prompt_locked(cam, x, y, dw, dh, hit_label, ttl_ms=4200.0,
                                                   score=score, stable_id=sid, bullet_id=bid2, method=method)
                # Draw approach segment as another transient line endpoint marker when present.
                ax = msg.get("approach_x", None)
                ay = msg.get("approach_y", None)
                if ax is not None and ay is not None:
                    self._append_marker_locked(cam, _safe_float(ax), _safe_float(ay), dw, dh, "event", "approach", ttl_ms=ttl, radius_px=7.0, color=(0.25, 1.0, 0.25, 0.75))
                self.bullet_events.append({
                    "ts_ms": now,
                    "cam": cam,
                    "type": typ,
                    "kind": kind,
                    "bullet_id": bid,
                    "state": str(msg.get("state", msg.get("semantic_state", kind)) or kind),
                    "event_ts_us": _safe_int(msg.get("ts_us", msg.get("point_ts_us", msg.get("event_ts", 0))), 0),
                    "x": x,
                    "y": y,
                    "label": label,
                })
                self._append_text_locked(f"{cam} {label}", cam=cam, ttl_ms=ttl, level="marker")
                return

            if typ == "overlay_bullet_point":
                # The trajectory shm is the high-rate source of truth.  Keep a tiny
                # live point marker for debugging when websocket is enabled.
                dw, dh = self._display_size_from_msg(cam, msg)
                bx = _safe_float(msg.get("x"))
                by = _safe_float(msg.get("y"))
                self._append_marker_locked(cam, bx, by, dw, dh,
                                           "point", "", ttl_ms=350.0, radius_px=4.0, color=(1.0, 0.2, 0.05, 0.45))
                self.bullet_points.append({
                    "ts_ms": now,
                    "cam": cam,
                    "bullet_id": _safe_int(msg.get("bullet_id", 0), 0),
                    "point_index": _safe_int(msg.get("point_index", -1), -1),
                    "point_ts_us": _safe_int(msg.get("point_ts_us", msg.get("ts_us", 0)), 0),
                    "wall_time_us": int(_safe_float(msg.get("wall_time", 0.0), 0.0) * 1_000_000) if _safe_float(msg.get("wall_time", 0.0), 0.0) > 0 else 0,
                    "x": bx,
                    "y": by,
                    "display_w": dw,
                    "display_h": dh,
                    "point_delta_ms": _safe_float(msg.get("point_delta_ms", -1.0), -1.0),
                    "receive_delta_ms": _safe_float(msg.get("receive_delta_ms", -1.0), -1.0),
                    "segment_len_mvs_px": _safe_float(msg.get("segment_len_mvs_px", -1.0), -1.0),
                    "status": str(msg.get("status", msg.get("state", "track")) or "track"),
                })
                return

            if typ or msg.get("text") or msg.get("message"):
                txt = str(msg.get("text", msg.get("message", typ)) or typ)
                if txt:
                    self._append_text_locked(txt, cam=cam, ttl_ms=1600.0, level="info")

    def update_from_human_msg(self, msg: Dict[str, Any], source: str = "human_jsonl") -> None:
        """Store the latest YOLO person mask/bbox result for each camera.

        human_result_*.jsonl carries msg_type=person_regions records from
        hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py.  Keep only the
        newest record per camera so stale files from an old sync root cannot
        overwrite a current run.
        """
        if not isinstance(msg, dict):
            return
        if str(msg.get("msg_type", msg.get("type", "")) or "") != "person_regions":
            return
        cam = str(msg.get("camera", msg.get("camera_name", "")) or "").strip()
        if not cam or cam not in self.cam_to_index:
            return
        persons = msg.get("persons", [])
        if not isinstance(persons, list):
            persons = []
        frame_w = _safe_float(msg.get("frame_w", msg.get("width", 0)), 0.0)
        frame_h = _safe_float(msg.get("frame_h", msg.get("height", 0)), 0.0)
        if frame_w <= 1 or frame_h <= 1:
            meta = msg.get("mvs_meta", {})
            if isinstance(meta, dict):
                shape = meta.get("shape")
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    frame_h = _safe_float(shape[0], frame_h)
                    frame_w = _safe_float(shape[1], frame_w)
                frame_w = _safe_float(meta.get("nWidth", frame_w), frame_w)
                frame_h = _safe_float(meta.get("nHeight", frame_h), frame_h)
        frame_w = max(frame_w, 1.0)
        frame_h = max(frame_h, 1.0)
        record_wall_us = _safe_float(msg.get("record_wall_us", msg.get("capture_wall_us", 0)), 0.0)
        frame_id = _safe_int(msg.get("frame_id_local", msg.get("mvs_frame_num", -1)), -1)
        now = _now_ms()
        with self.lock:
            old = self.humans_by_cam.get(cam)
            if old is not None:
                old_wall = _safe_float(old.get("record_wall_us", 0.0), 0.0)
                old_frame = _safe_int(old.get("frame_id", -1), -1)
                # Ignore stale duplicates from an older sync root or old JSONL tail.
                if record_wall_us > 0 and old_wall > 0 and record_wall_us < old_wall:
                    return
                if record_wall_us <= 0 and frame_id >= 0 and old_frame >= 0 and frame_id < old_frame:
                    return
            clean_persons: List[Dict[str, Any]] = []
            for person in persons[:64]:
                if not isinstance(person, dict):
                    continue
                polys = person.get("mask_polygons", [])
                bbox = person.get("bbox_xyxy", None)
                clean_persons.append({
                    "stable_id": person.get("stable_id", person.get("person_id", "")),
                    "person_id": person.get("person_id", ""),
                    "confidence": _safe_float(person.get("confidence", 0.0), 0.0),
                    "bbox_xyxy": bbox if isinstance(bbox, (list, tuple)) else None,
                    "mask_polygons": polys if isinstance(polys, list) else [],
                    "foot_pixel": person.get("foot_pixel", person.get("centroid", None)),
                    "local_xy": person.get("local_xy", None),
                    "physical_coord": person.get("physical_coord", None),
                    "global_id_from_A": person.get("global_id_from_A", None),
                    "global_id": person.get("global_id", person.get("source_global_id", None)),
                    "global_player_id": person.get("global_player_id", person.get("source_global_player_id", "")),
                    "source_global_id": person.get("source_global_id", person.get("global_id", None)),
                    "source_global_player_id": person.get("source_global_player_id", person.get("global_player_id", "")),
                    "display_player_id": person.get("display_player_id", ""),
                    "display_slot_id": person.get("display_slot_id", person.get("target_display_slot_id", -1)),
                    "visibility": str(person.get("visibility", "") or ""),
                })
            capture_wall_us = _safe_float(msg.get("capture_wall_us", 0), 0.0)
            self.humans_by_cam[cam] = {
                "cam": cam,
                "ts_ms": now,
                "record_wall_us": record_wall_us,
                "capture_wall_us": capture_wall_us,
                "frame_id": frame_id,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "persons": clean_persons,
                "person_count": len(clean_persons),
                "projector": msg.get("projector", None),
                "yolo_batch_size": _safe_int(msg.get("yolo_batch_size", 0), 0),
                "yolo_predict_ms": _safe_float(msg.get("yolo_predict_ms", 0.0), 0.0),
                "yolo_total_batch_ms": _safe_float(msg.get("yolo_total_batch_ms", 0.0), 0.0),
                "yolo_capture_to_result_ms": max(0.0, (record_wall_us - capture_wall_us) / 1000.0) if record_wall_us > 0 and capture_wall_us > 0 else 0.0,
            }
            self.human_jsonl_rx_count += 1
            self.last_human_ms = now

    def snapshot(self, marker_ttl_ms: float = 2500.0, text_ttl_ms: float = 2500.0) -> Dict[str, Any]:
        now = _now_ms()
        with self.lock:
            markers = [m for m in self.markers if now - float(m.get("ts_ms", 0.0)) <= min(float(marker_ttl_ms), float(m.get("ttl_ms", marker_ttl_ms)))]
            texts = [t for t in self.texts if now - float(t.get("ts_ms", 0.0)) <= min(float(text_ttl_ms), float(t.get("ttl_ms", text_ttl_ms)))]
            hit_prompts = [p for p in self.hit_prompts if now - float(p.get("ts_ms", 0.0)) <= float(p.get("ttl_ms", 3600.0))]
            humans = dict(self.humans_by_cam)
            return {
                "markers": markers,
                "texts": texts,
                "hit_prompts": hit_prompts,
                "humans": humans,
                "frame_sizes": dict(self.frame_sizes),
                "ws_connected": self.ws_connected,
                "ws_error": self.ws_error,
                "ws_rx_count": self.ws_rx_count,
                "ws_age_ms": 0.0 if self.ws_last_ms <= 0 else now - self.ws_last_ms,
                "hit_jsonl_rx_count": self.hit_jsonl_rx_count,
                "human_jsonl_rx_count": self.human_jsonl_rx_count,
                "human_age_ms": 0.0 if self.last_human_ms <= 0 else now - self.last_human_ms,
                "bullet_points": list(self.bullet_points),
                "bullet_events": list(self.bullet_events),
                "hit_decisions": list(self.hit_decisions),
            }


# latest_frame_shm.py header constants, duplicated here to read metadata without copying frames.
_SF_MAGIC = 0x46524D32
_SF_HEADER_U32_COUNT = 32
_SF_HEADER_BYTES = _SF_HEADER_U32_COUNT * 4
_SF_ACTIVE_SLOT = 8
_SF_SLOT_FIELDS = [(10, 11, 12, 13), (14, 15, 16, 17)]


def _join_u64_u32(lo: int, hi: int) -> int:
    return (int(hi) << 32) | int(lo)


class SharedFrameHeaderReader:
    """Header-only reader for /dev/shm/mvs_latest_*.mmap.

    It never copies image payload, so using it for telemetry does not affect the
    YOLO/fusion frame path.
    """

    def __init__(self, name: str):
        self.name = str(name)
        self.path = self.name if self.name.startswith('/') else f"/dev/shm/{self.name}.mmap"
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, os.path.getsize(self.path), access=mmap.ACCESS_READ)
        self.header = np.ndarray((_SF_HEADER_U32_COUNT,), dtype=np.uint32, buffer=self.mm, offset=0)
        if int(self.header[0]) != _SF_MAGIC:
            raise RuntimeError(f"shared frame magic mismatch: {self.path}")

    def sample(self) -> Dict[str, Any]:
        slot = int(self.header[_SF_ACTIVE_SLOT])
        if slot not in (0, 1):
            return {"ok": False}
        ready_i, fid_i, lo_i, hi_i = _SF_SLOT_FIELDS[slot]
        ready = int(self.header[ready_i]) == 1
        fid = int(self.header[fid_i])
        ts_us = _join_u64_u32(int(self.header[lo_i]), int(self.header[hi_i]))
        return {"ok": bool(ready), "frame_id": fid, "timestamp_us": ts_us}

    def close(self) -> None:
        try:
            self.header = None
            self.mm.close()
            os.close(self.fd)
        except Exception:
            pass


class EventOverlayPreviewReader:
    """Best-effort reader for event overlay SharedFrameWriter + event_viz sidecar.

    The overlay frame is visualization-only.  This class keeps the latest texture
    payload and metadata, while all sync decisions are made in PreviewWidget.
    """

    @staticmethod
    def _split_templates(raw: str, default: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"[,;|]", str(raw or default)) if p.strip()]
        return parts or [default]

    @staticmethod
    def _resolve_sidecar_path(sync_root: str, sidecar_name: str) -> str:
        sidecar_s = str(sidecar_name or "")
        if os.path.isabs(sidecar_s):
            return sidecar_s
        normalized = sidecar_s.replace("\\", "/")
        if normalized.startswith("sync_ipc/"):
            return sidecar_s
        return os.path.join(str(sync_root or "sync_ipc"), sidecar_s)

    @staticmethod
    def _masked_sidecar_for_overlay(name: str, sidecar_template: str) -> str:
        if "event_overlay_masked_latest_" not in str(name):
            return sidecar_template
        sidecar_s = str(sidecar_template or "event_overlay_{camera}_latest.json")
        normalized = sidecar_s.replace("\\", "/")
        if "event_overlay_masked_" in normalized:
            return sidecar_s
        return sidecar_s.replace("event_overlay_", "event_overlay_masked_", 1)

    def __init__(self, cam: str, name_template: str, sync_root: str, sidecar_template: str = "event_overlay_{camera}_latest.json"):
        self.cam = str(cam)
        name_templates = self._split_templates(str(name_template or "event_overlay_latest_{camera}"), "event_overlay_latest_{camera}")
        sidecar_templates = self._split_templates(str(sidecar_template or "event_overlay_{camera}_latest.json"), "event_overlay_{camera}_latest.json")
        candidates: List[Tuple[str, str, str]] = []
        for idx, tmpl in enumerate(name_templates):
            name = str(tmpl).format(camera=cam, cam=cam, id=cam)
            path = name if name.startswith('/') else f"/dev/shm/{name}.mmap"
            if len(sidecar_templates) == len(name_templates):
                sidecar_tmpl = sidecar_templates[idx]
            else:
                sidecar_tmpl = sidecar_templates[min(idx, len(sidecar_templates) - 1)]
            sidecar_tmpl = self._masked_sidecar_for_overlay(name, sidecar_tmpl)
            sidecar_name = str(sidecar_tmpl).format(camera=cam, cam=cam, id=cam)
            candidates.append((name, path, self._resolve_sidecar_path(sync_root, sidecar_name)))
        self.candidate_paths = [path for _name, path, _sidecar in candidates]
        self.name = ""
        self.path = ""
        self.sidecar_path = ""
        self.fd = -1
        self.mm = None
        self.header = None
        last_error = ""
        for name, path, sidecar_path in candidates:
            if not os.path.exists(path):
                continue
            fd = -1
            mm_obj = None
            try:
                fd = os.open(path, os.O_RDONLY)
                mm_obj = mmap.mmap(fd, os.path.getsize(path), access=mmap.ACCESS_READ)
                header = np.ndarray((_SF_HEADER_U32_COUNT,), dtype=np.uint32, buffer=mm_obj, offset=0)
                if int(header[0]) != _SF_MAGIC:
                    raise RuntimeError(f"event overlay shared frame magic mismatch: {path}")
                self.name = name
                self.path = path
                self.sidecar_path = sidecar_path
                self.fd = fd
                self.mm = mm_obj
                self.header = header
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                try:
                    if mm_obj is not None:
                        mm_obj.close()
                except Exception:
                    pass
                try:
                    if fd >= 0:
                        os.close(fd)
                except Exception:
                    pass
        if self.header is None or self.mm is None or self.fd < 0:
            if last_error:
                raise RuntimeError(f"{last_error}; candidates={self.candidate_paths}")
            raise FileNotFoundError(", ".join(self.candidate_paths))
        self.width = int(self.header[2])
        self.height = int(self.header[3])
        self.channels = int(self.header[4])
        self.data_bytes = int(self.header[9])
        self._slots = [
            np.ndarray((self.height, self.width, self.channels), dtype=np.uint8, buffer=self.mm, offset=_SF_HEADER_BYTES + i * self.data_bytes)
            for i in range(2)
        ]
        self.last_frame_id = -1
        self.last_sidecar_mtime = 0.0
        self.last_meta: Dict[str, Any] = {}
        self.last_frame: Optional[np.ndarray] = None

    def _read_sidecar(self, force: bool = False) -> Dict[str, Any]:
        try:
            st = os.stat(self.sidecar_path)
            if not force and st.st_mtime == self.last_sidecar_mtime:
                return dict(self.last_meta)
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self.last_meta = obj
                self.last_sidecar_mtime = st.st_mtime
        except Exception:
            pass
        return dict(self.last_meta)

    def read_latest(self) -> Optional[Dict[str, Any]]:
        meta = self._read_sidecar(False)
        meta["_reader_name"] = self.name
        meta["_reader_path"] = self.path
        meta["_reader_sidecar_path"] = self.sidecar_path
        for _ in range(4):
            slot_a = int(self.header[_SF_ACTIVE_SLOT])
            if slot_a not in (0, 1):
                return None
            ready_i, fid_i, lo_i, hi_i = _SF_SLOT_FIELDS[slot_a]
            if int(self.header[ready_i]) != 1:
                return None
            fid = int(self.header[fid_i])
            ts_us = _join_u64_u32(int(self.header[lo_i]), int(self.header[hi_i]))
            if fid == self.last_frame_id and self.last_frame is not None:
                return {"frame": self.last_frame, "frame_id": fid, "timestamp_us": ts_us, "shape": (self.height, self.width, self.channels), "meta": meta, "new_frame": False}
            frame = self._slots[slot_a].copy()
            slot_b = int(self.header[_SF_ACTIVE_SLOT])
            if slot_a != slot_b:
                continue
            self.last_frame_id = fid
            self.last_frame = frame
            return {"frame": frame, "frame_id": fid, "timestamp_us": ts_us, "shape": (self.height, self.width, self.channels), "meta": meta, "new_frame": True}
        return None

    def close(self) -> None:
        try:
            self.header = None
            self._slots = []
            self.mm.close()
            os.close(self.fd)
        except Exception:
            pass


class LightweightLogTailer:
    def __init__(self, path: str):
        self.path = str(path)
        self.pos = 0
        self.inode = None

    def poll(self, max_lines: int = 512) -> List[str]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return []
        inode = (st.st_ino, st.st_size)
        if self.inode is None or inode[0] != self.inode[0] or st.st_size < self.pos:
            self.pos = 0
        self.inode = inode
        try:
            with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.pos)
                lines = f.readlines()
                self.pos = f.tell()
        except Exception:
            return []
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return [ln.rstrip('\n') for ln in lines]


class PreviewTelemetry:
    def __init__(self, args: argparse.Namespace, cams: List[str], overlay_state: OverlayState):
        self.args = args
        self.cams = list(cams)
        self.overlay_state = overlay_state
        self.sync_root = str(getattr(args, 'sync_root', 'sync_ipc') or 'sync_ipc')
        self.raw_template = str(getattr(args, 'raw_template', 'mvs_raw_latest_{camera}') or 'mvs_raw_latest_{camera}')
        self.mvs_template = str(getattr(args, 'mvs_shm_template', 'mvs_latest_{camera}') or 'mvs_latest_{camera}')
        self.raw_readers: Dict[str, RawPreviewFrameReader] = {}
        self.mvs_readers: Dict[str, SharedFrameHeaderReader] = {}
        self.prev_raw: Dict[str, Tuple[int, float]] = {}
        self.prev_mvs: Dict[str, Tuple[int, float]] = {}
        self.raw_fps: Dict[str, float] = {c: 0.0 for c in self.cams}
        self.mvs_fps: Dict[str, float] = {c: 0.0 for c in self.cams}
        self.raw_age_ms: Dict[str, float] = {c: -1.0 for c in self.cams}
        self.mvs_age_ms: Dict[str, float] = {c: -1.0 for c in self.cams}
        self.shm_missing: Dict[str, bool] = {c: True for c in self.cams}
        self.pending_q: Dict[str, int] = {c: -1 for c in self.cams}
        self.yolo = {
            "infer_fps": 0.0,
            "inferred_fps_total": 0.0,
            "published_fps_total": 0.0,
            "async_record_postprocess_enabled": False,
            "postprocess_queue_depth": 0,
            "postprocess_dropped_batches": 0,
            "polygon_contract_ok": None,
            "published_records_with_polygon_ratio": 1.0,
            "published_persons_with_polygon_ratio": 1.0,
            "batch_size": 0,
            "predict_ms": 0.0,
            "capture_to_result_ms": 0.0,
            "result_video_delta_frames": {},
            "workers": {},
            "worker_count": 0,
        }
        self.hit = {"recv_per_s": 0.0, "no_human_per_s": 0.0, "no_human_count": 0,
                    "eval_per_s": 0.0, "candidate_per_s": 0.0, "confirmed_per_s": 0.0,
                    "no_mask_per_s": 0.0, "stale_human_per_s": 0.0, "sync_bad_per_s": 0.0,
                    "outside_mask_per_s": 0.0, "no_cam_per_s": 0.0, "drop_per_s": 0.0,
                    "pending_count": 0, "pending_impacts": 0}
        self.trigger = {"state": "unknown", "fps": 0.0, "period_ms": -1.0,
                        "jitter_p50_ms": -1.0, "jitter_p95_ms": -1.0, "jitter_max_ms": -1.0,
                        "last_age_ms": -1.0, "ok_cams": 0,
                        "per_cam_sent": {}, "per_cam_ack": {}, "per_cam_fail": {},
                        "per_cam_trigger_to_mvs_frame_ms": {}, "per_cam_trigger_to_event_trigger_ms": {},
                        "per_cam_trigger_miss_count": {}}
        self.timebase = {"system_wall_time_us": 0, "system_mono_ns": 0,
                         "sync_start_wall_us": 0, "sync_start_mono_ns": 0,
                         "event_timebase_valid": False, "event_ts_to_wall_offset_us": 0,
                         "event_last_wall_age_ms": -1.0, "mvs_timebase_valid": False,
                         "mvs_dev_ts_to_wall_offset_us": 0, "mvs_last_wall_age_ms": -1.0,
                         "clock_drift_event_vs_mvs_ms": -1.0, "clock_drift_event_vs_wall_ms": -1.0}
        self.event = {"event_active_count": 0, "per_cam_ready": {c: False for c in self.cams},
                      "per_cam_event_trigger_fps": {c: 0.0 for c in self.cams},
                      "per_cam_event_rate_mevs": {c: 0.0 for c in self.cams},
                      "per_cam_event_slice_age_ms": {c: -1.0 for c in self.cams},
                      "per_cam_event_last_ts_us": {c: 0 for c in self.cams},
                      "per_cam_event_last_wall_age_ms": {c: -1.0 for c in self.cams},
                      "per_cam_event_trigger_recv_count": {c: 0 for c in self.cams},
                      "per_cam_event_trigger_last_age_ms": {c: -1.0 for c in self.cams},
                      "per_cam_event_trigger_gap_count": {c: 0 for c in self.cams},
                      "per_cam_event_drop_count": {c: 0 for c in self.cams},
                      "per_cam_event_nonmonotonic_count": {c: 0 for c in self.cams},
                      "per_cam_event_guard_state": {c: "OK" for c in self.cams},
                      "per_cam_event_shake_guard_active": {c: False for c in self.cams},
                      "per_cam_event_pre_spatter_guard_active": {c: False for c in self.cams}}
        self.bullet = {"udp_recv_per_s": 0.0, "track_count_active": 0, "latest_age_ms": -1.0,
                       "latest_cam": "", "latest_id": 0, "latest_point_count": 0,
                       "latest_start_wall_us": 0, "latest_end_wall_us": 0,
                       "trajectory_shm_seq": 0, "trajectory_shm_segment_count": 0,
                       "trajectory_visible_segment_count": 0, "trajectory_latest_age_ms": -1.0,
                       "trajectory_drawn_vtx": 0, "trajectory_stale_guard_count": 0,
                       "trajectory_future_drop_count": 0, "trajectory_fresh_not_drawn": 0,
                       "trajectory_draw_state": "init",
                       "bullet_status_ttl_ms": 0.0, "fresh_point_count": 0,
                       "stale_point_count": 0, "future_point_count": 0,
                       "stale_drop_count": 0, "recent": []}
        self.sync_health = {"latest_cam": "", "bullet_id": 0, "event_to_mvs_delta_ms": -1.0,
                            "event_to_human_capture_delta_ms": -1.0, "event_to_human_result_age_ms": -1.0,
                            "event_to_human_delta_frames": 999999, "mask_valid_at_event": False,
                            "person_count_at_event": 0, "matched_person_id": "", "matched_stable_id": "",
                            "hit_inside_mask": False, "hit_distance_to_mask_px": -1.0,
                            "hit_distance_to_bbox_px": -1.0, "sync_quality": "UNKNOWN",
                            "recent": []}
        self.mvs_active_count = 0
        self.prev_event_csv: Dict[str, Tuple[int, float]] = {}
        self.event_trigger_gap_count: Dict[str, int] = {c: 0 for c in self.cams}
        self.last_poll_ms = 0.0
        logs = os.path.join(self.sync_root, 'launcher_logs')
        self.trigger_log = LightweightLogTailer(os.path.join(logs, 'trigger.log'))
        self.event_log = LightweightLogTailer(os.path.join(logs, 'event.log'))
        self.mvs_agg_log = LightweightLogTailer(os.path.join(logs, 'mvs_agg.log'))
        self.hit_log = LightweightLogTailer(os.path.join(logs, 'hit.log'))
        # Diagnostics must tolerate the project's mixed-root transition.
        # The launcher passes --sync-root for current runtime files, while the
        # unified MVS config may still publish human/mvs/hit sidecars under an
        # older sync.root.  Keep an ordered root list and choose the freshest
        # existing file per data source instead of assuming one root.
        self.sync_roots = self._resolve_sync_roots()
        self.last_snapshot: Dict[str, Any] = {}
        self.last_poll_errors: Dict[str, str] = {}

    def _open_raw(self, cam: str) -> Optional[RawPreviewFrameReader]:
        if cam in self.raw_readers:
            return self.raw_readers[cam]
        try:
            name = _shm_name_from_template(self.raw_template, cam)
            rd = RawPreviewFrameReader(name)
            self.raw_readers[cam] = rd
            return rd
        except Exception:
            return None

    def _open_mvs(self, cam: str) -> Optional[SharedFrameHeaderReader]:
        if cam in self.mvs_readers:
            return self.mvs_readers[cam]
        try:
            name = self.mvs_template.format(camera=cam, cam=cam, id=cam)
            rd = SharedFrameHeaderReader(name)
            self.mvs_readers[cam] = rd
            return rd
        except Exception:
            return None

    def _resolve_sync_roots(self) -> List[str]:
        roots: List[str] = []

        def add_root(x: Any) -> None:
            if not x:
                return
            try:
                pp = Path(str(x)).expanduser()
                if not pp.is_absolute():
                    pp = Path.cwd() / pp
                sp = str(pp)
                if sp not in roots:
                    roots.append(sp)
            except Exception:
                pass

        add_root(self.sync_root)
        # Common default from current working tree.
        add_root("sync_ipc")
        try:
            cfg_path = str(getattr(self.args, "config", "") or "")
            if cfg_path and os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                mvs = cfg.get("mvs_config", cfg) if isinstance(cfg, dict) else {}
                sync = mvs.get("sync", {}) if isinstance(mvs, dict) else {}
                if isinstance(sync, dict):
                    add_root(sync.get("root", ""))
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] telemetry failed to parse sync roots from config: {exc}", flush=True)
        return roots

    def _candidate_paths(self, filename: str) -> List[str]:
        out: List[str] = []
        for root in getattr(self, "sync_roots", [self.sync_root]):
            try:
                p = os.path.join(str(root), filename)
                if p not in out:
                    out.append(p)
            except Exception:
                pass
        return out

    def _best_existing_path(self, filename: str, prefer_json_wall: bool = False) -> str:
        candidates = [p for p in self._candidate_paths(filename) if os.path.exists(p)]
        if not candidates:
            # Return current-root path as a useful fallback for callers that only
            # want a path string.
            return self._candidate_paths(filename)[0] if self._candidate_paths(filename) else os.path.join(self.sync_root, filename)
        if not prefer_json_wall:
            try:
                return max(candidates, key=lambda x: os.path.getmtime(x))
            except Exception:
                return candidates[0]

        def score(path: str) -> float:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    wall = self._pick_wall_us(obj)
                    if wall > 0:
                        return wall
            except Exception:
                pass
            try:
                return os.path.getmtime(path) * 1_000_000.0
            except Exception:
                return 0.0

        try:
            return max(candidates, key=score)
        except Exception:
            return candidates[0]

    def _meta_path(self, cam: str) -> str:
        return self._best_existing_path(f"mvs_latest_{cam}.json", prefer_json_wall=True)

    def _pick_wall_us(self, meta: Dict[str, Any]) -> float:
        """Return a wall-clock timestamp in microseconds from sidecar metadata.

        The mmap timestamp can be device/sync time on some paths.  For status
        ages, prefer host wall-time fields from the JSON sidecar and also check
        nested mvs_meta because human_result records store the same fields there.
        """
        vals: List[Any] = []
        for key in ("publish_wall_us", "receiver_wall_us", "host_read_wall_us", "timestamp_us", "capture_wall_us"):
            vals.append(meta.get(key))
        mm = meta.get("mvs_meta")
        if isinstance(mm, dict):
            for key in ("publish_wall_us", "receiver_wall_us", "host_read_wall_us", "timestamp_us", "capture_wall_us"):
                vals.append(mm.get(key))
        for v in vals:
            fv = _safe_float(v, 0.0)
            # Unix wall us in current deployments is around 1.7e15.
            if fv > 1_000_000_000_000_000:
                return fv
        return 0.0

    def _read_meta(self, cam: str) -> Dict[str, Any]:
        try:
            with open(self._meta_path(cam), 'r', encoding='utf-8') as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _read_status_json(self, filename: str) -> Dict[str, Any]:
        try:
            path = self._best_existing_path(str(filename), prefer_json_wall=True)
            with open(path, "r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return {}
            return obj
        except Exception:
            return {}

    def _status_wall_age_ms(self, obj: Dict[str, Any]) -> float:
        if not isinstance(obj, dict):
            return -1.0
        now_us = int(time.time() * 1_000_000)
        for key in (
            "ts_wall_us",
            "timestamp_us",
            "wall_time_us",
            "wall_us",
            "time_wall_us",
            "update_wall_us",
            "updated_wall_us",
            "updated_us",
        ):
            val = _safe_float(obj.get(key), 0.0)
            if val > 1_000_000_000_000_000:
                return max(0.0, (now_us - val) / 1000.0)
        return -1.0

    def _event_ready_path(self, cam: str) -> str:
        return self._best_existing_path(f"event_ready_{cam}.flag")

    def _event_worker_status_path(self, cam: str) -> str:
        return self._best_existing_path(f"event_worker_{cam}_status.json")

    def _poll_event_worker_status_files(self) -> None:
        now_ms = time.time() * 1000.0
        for cam in self.cams:
            path = self._event_worker_status_path(cam)
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                if not isinstance(st, dict):
                    continue
                self.event['per_cam_event_rate_mevs'][cam] = _safe_float(st.get('event_rate_mevs'), self.event['per_cam_event_rate_mevs'].get(cam, 0.0))
                self.event['per_cam_event_slice_age_ms'][cam] = max(0.0, now_ms - _safe_float(st.get('wall_time_us'), 0.0) / 1000.0) if _safe_float(st.get('wall_time_us'), 0.0) > 1_000_000_000_000_000 else self.event['per_cam_event_slice_age_ms'].get(cam, -1.0)
                hm = st.get('human_mask') if isinstance(st.get('human_mask'), dict) else {}
                guard = str(st.get('guard_state') or 'OK')
                self.event['per_cam_event_guard_state'][cam] = guard
                self.event['per_cam_event_shake_guard_active'][cam] = (guard == 'SHAKE')
                self.event['per_cam_event_pre_spatter_guard_active'][cam] = (guard == 'PRE_SPATTER')
                # Store extra status for tabs/log snapshot without changing the legacy schema.
                self.event.setdefault('per_cam_worker_state', {})[cam] = st.get('state', '')
                self.event.setdefault('per_cam_worker_loop_fps', {})[cam] = _safe_float(st.get('loop_fps'), 0.0)
                self.event.setdefault('per_cam_worker_empty_ratio', {})[cam] = _safe_float(st.get('empty_slice_ratio'), 0.0)
                self.event.setdefault('per_cam_worker_fastpath_count', {})[cam] = _safe_int(st.get('fastpath_slice_count'), 0)
                bd = st.get('bullet_display_diag') if isinstance(st.get('bullet_display_diag'), dict) else st
                self.event.setdefault('per_cam_draw_paths_count', {})[cam] = _safe_int(bd.get('draw_paths_count'), 0)
                self.event.setdefault('per_cam_draw_path_points_count', {})[cam] = _safe_int(bd.get('draw_path_points_count'), 0)
                self.event.setdefault('per_cam_display_eligible_count', {})[cam] = _safe_int(bd.get('display_eligible_count'), 0)
                self.event.setdefault('per_cam_active_track_count', {})[cam] = _safe_int(bd.get('active_track_count'), 0)
                self.event.setdefault('per_cam_active_but_not_display_count', {})[cam] = _safe_int(bd.get('active_but_not_display_count'), 0)
                self.event.setdefault('per_cam_last_overlay_bullet_point_age_ms', {})[cam] = _safe_float(bd.get('last_overlay_bullet_point_age_ms'), -1.0)
                self.event.setdefault('per_cam_bullet_sender_queue_size', {})[cam] = _safe_int(bd.get('bullet_sender_queue_size'), 0)
                self.event.setdefault('per_cam_bullet_sender_dropped', {})[cam] = _safe_int(bd.get('bullet_sender_dropped'), 0)
                self.event.setdefault('per_cam_bullet_sender_drop_delta', {})[cam] = _safe_int(bd.get('bullet_sender_drop_delta'), 0)
                self.event.setdefault('per_cam_bullet_source_state', {})[cam] = str(bd.get('bullet_source_state', st.get('bullet_source_state', '')) or '')
                self.event.setdefault('per_cam_human_mask_cache_poll', {})[cam] = _safe_int(hm.get('jsonl_poll_count'), 0) if isinstance(hm, dict) else 0
                self.event.setdefault('per_cam_human_mask_cache_skip', {})[cam] = _safe_int(hm.get('jsonl_poll_skip_count'), 0) if isinstance(hm, dict) else 0
                viz = st.get('viz') if isinstance(st.get('viz'), dict) else {}
                vbuf = viz.get('buffer') if isinstance(viz.get('buffer'), dict) else {}
                vpub = viz.get('publisher') if isinstance(viz.get('publisher'), dict) else {}
                lp = vpub.get('last_packet') if isinstance(vpub.get('last_packet'), dict) else {}
                self.event.setdefault('per_cam_viz_buffer_size', {})[cam] = _safe_int(vbuf.get('size'), 0)
                self.event.setdefault('per_cam_viz_buffer_drop_count', {})[cam] = _safe_int(vbuf.get('drop_count'), 0)
                self.event.setdefault('per_cam_viz_buffer_latest_age_ms', {})[cam] = _safe_float(vbuf.get('latest_age_ms'), -1.0)
                self.event.setdefault('per_cam_viz_publish_count', {})[cam] = _safe_int(vpub.get('publish_count'), 0)
                self.event.setdefault('per_cam_viz_overlay_count', {})[cam] = _safe_int(vpub.get('overlay_publish_count'), 0)
                self.event.setdefault('per_cam_viz_obs_count', {})[cam] = _safe_int(vpub.get('obs_publish_count'), 0)
                self.event.setdefault('per_cam_viz_tsf1_count', {})[cam] = _safe_int(vpub.get('tsf1_publish_count'), 0)
                self.event.setdefault('per_cam_viz_source_stage', {})[cam] = str(lp.get('source_stage', vpub.get('source_stage', '')) or '')
                self.event.setdefault('per_cam_viz_raw_count', {})[cam] = _safe_int(lp.get('raw_event_count'), 0)
                self.event.setdefault('per_cam_viz_filtered_count', {})[cam] = _safe_int(lp.get('filtered_event_count'), 0)
                self.event.setdefault('per_cam_viz_reduction_ratio', {})[cam] = _safe_float(lp.get('filter_reduction_ratio'), 0.0)
                self.event.setdefault('per_cam_viz_mask_valid', {})[cam] = bool(lp.get('mask_valid', False))
                self.event.setdefault('per_cam_viz_event_to_human_ms', {})[cam] = _safe_float(lp.get('event_to_human_capture_delta_ms'), -1.0)
            except Exception:
                continue

    def _event_trigger_csv_path(self, cam: str) -> str:
        return self._best_existing_path(f"event_trigger_{cam}.csv")

    def _read_last_csv_row(self, path: str, tail_bytes: int = 65536) -> Dict[str, Any]:
        try:
            if not os.path.exists(path):
                return {}
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                f.seek(max(0, size - int(tail_bytes)))
                data = f.read().decode("utf-8", errors="ignore")
            lines = [ln for ln in data.splitlines() if ln.strip()]
            if len(lines) < 2:
                return {}
            if not lines[0].startswith("all_index"):
                # If we started in the middle, find the header; otherwise use the
                # project default external trigger columns.
                header = None
                for ln in lines:
                    if ln.startswith("all_index") and "event_ts_us" in ln:
                        header = ln
                        break
                if header is None:
                    header = "all_index,sync_index,event_ts_us,polarity,trigger_id,slice_ts_us,wall_time_us"
                rows = [ln for ln in lines if not ln.startswith("all_index")]
                if not rows:
                    return {}
                csv_text = header + "\n" + rows[-1] + "\n"
            else:
                csv_text = lines[0] + "\n" + lines[-1] + "\n"
            rr = list(csv.DictReader(csv_text.splitlines()))
            return rr[-1] if rr else {}
        except Exception:
            return {}

    def _read_sync_start_times(self) -> Tuple[int, int]:
        wall_us = 0
        mono_ns = 0
        path = self._best_existing_path("sync_start.flag")
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                wall_us = _safe_int(obj.get("wall_time_us", obj.get("sync_start_wall_us", 0)), 0)
                mono_ns = _safe_int(obj.get("monotonic_ns", obj.get("sync_start_mono_ns", 0)), 0)
        except Exception:
            try:
                st = os.stat(path)
                wall_us = int(st.st_mtime * 1_000_000)
            except Exception:
                pass
        status_path = self._best_existing_path("trigger_status.json", prefer_json_wall=True)
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            wall_us = _safe_int(obj.get("sync_start_wall_us", wall_us), wall_us)
            mono_ns = _safe_int(obj.get("sync_start_mono_ns", mono_ns), mono_ns)
        except Exception:
            pass
        return wall_us, mono_ns

    def _poll_timebase(self) -> None:
        now_us = int(time.time() * 1_000_000)
        now_mono = int(time.monotonic_ns())
        sync_wall_us, sync_mono_ns = self._read_sync_start_times()
        self.timebase["system_wall_time_us"] = now_us
        self.timebase["system_mono_ns"] = now_mono
        self.timebase["sync_start_wall_us"] = sync_wall_us
        self.timebase["sync_start_mono_ns"] = sync_mono_ns
        # Derive event<->wall offset from the latest event external trigger row.
        offsets: List[float] = []
        last_event_wall_us = 0.0
        for cam in self.cams:
            row = self._read_last_csv_row(self._event_trigger_csv_path(cam), tail_bytes=16384)
            if not row:
                continue
            ev_ts = _safe_float(row.get("event_ts_us"), 0.0)
            wall_us = _safe_float(row.get("wall_time_us"), 0.0)
            if ev_ts > 0 and wall_us > 1_000_000_000_000_000:
                offsets.append(wall_us - ev_ts)
                last_event_wall_us = max(last_event_wall_us, wall_us)
        if offsets:
            self.timebase["event_timebase_valid"] = True
            self.timebase["event_ts_to_wall_offset_us"] = int(sum(offsets) / len(offsets))
            self.timebase["event_last_wall_age_ms"] = max(0.0, now_us / 1000.0 - last_event_wall_us / 1000.0)
            self.timebase["clock_drift_event_vs_wall_ms"] = 0.0
        else:
            self.timebase["event_timebase_valid"] = False
        # MVS timebase from sidecar wall timestamp freshness.
        mvs_ages = [v for v in self.mvs_age_ms.values() if isinstance(v, (int, float)) and v >= 0]
        if mvs_ages:
            self.timebase["mvs_timebase_valid"] = True
            self.timebase["mvs_last_wall_age_ms"] = min(mvs_ages)
        else:
            self.timebase["mvs_timebase_valid"] = False
        ev_age = _safe_float(self.timebase.get("event_last_wall_age_ms"), -1.0)
        mv_age = _safe_float(self.timebase.get("mvs_last_wall_age_ms"), -1.0)
        self.timebase["clock_drift_event_vs_mvs_ms"] = abs(ev_age - mv_age) if ev_age >= 0 and mv_age >= 0 else -1.0

    def _poll_event_triggers(self) -> None:
        now_s = time.time()
        now_ms = now_s * 1000.0
        active = 0
        for cam in self.cams:
            ready_path = self._event_ready_path(cam)
            ready = os.path.exists(ready_path)
            if ready:
                try:
                    with open(ready_path, "r", encoding="utf-8") as f:
                        robj = json.load(f)
                    if isinstance(robj, dict):
                        ready = bool(robj.get("ready", True))
                except Exception:
                    ready = True
            self.event["per_cam_ready"][cam] = bool(ready)
            if ready:
                active += 1
            row = self._read_last_csv_row(self._event_trigger_csv_path(cam), tail_bytes=65536)
            if not row:
                continue
            sync_i = _safe_int(row.get("sync_index", -1), -1)
            all_i = _safe_int(row.get("all_index", -1), -1)
            ev_ts = _safe_int(row.get("event_ts_us", 0), 0)
            slice_ts = _safe_int(row.get("slice_ts_us", 0), 0)
            wall_us = _safe_float(row.get("wall_time_us", 0.0), 0.0)
            count = max(sync_i, all_i, 0)
            self.event["per_cam_event_trigger_recv_count"][cam] = int(count)
            self.event["per_cam_event_last_ts_us"][cam] = int(ev_ts)
            if wall_us > 1_000_000_000_000_000:
                age_ms = max(0.0, now_ms - wall_us / 1000.0)
                self.event["per_cam_event_trigger_last_age_ms"][cam] = age_ms
                self.event["per_cam_event_last_wall_age_ms"][cam] = age_ms
                # event_active_count is based on ready flags.  The age field
                # remains useful to spot a stale trigger CSV without marking a
                # ready camera inactive.
            if slice_ts > 0 and ev_ts > 0:
                self.event["per_cam_event_slice_age_ms"][cam] = max(0.0, (ev_ts - slice_ts) / 1000.0)
            prev = self.prev_event_csv.get(cam)
            if prev is not None:
                prev_idx, prev_t = prev
                dt = max(1e-6, now_s - prev_t)
                if sync_i >= prev_idx >= 0:
                    self.event["per_cam_event_trigger_fps"][cam] = float(sync_i - prev_idx) / dt
                    if sync_i - prev_idx > 1 and dt < 2.0:
                        self.event_trigger_gap_count[cam] = int(self.event_trigger_gap_count.get(cam, 0)) + int(sync_i - prev_idx - 1)
            if sync_i >= 0:
                self.prev_event_csv[cam] = (sync_i, now_s)
            self.event["per_cam_event_trigger_gap_count"][cam] = int(self.event_trigger_gap_count.get(cam, 0))
            # Cross-check current MVS sidecar trigger index against latest event trigger index.
            meta = self._read_meta(cam)
            mvs_meta = meta.get("mvs_meta") if isinstance(meta.get("mvs_meta"), dict) else meta
            mvs_sync = _safe_int(mvs_meta.get("program_sync_index", mvs_meta.get("sync_index", -1)), -1) if isinstance(mvs_meta, dict) else -1
            cap_wall = _safe_float(mvs_meta.get("host_read_wall_us", mvs_meta.get("timestamp_us", 0)), 0.0) if isinstance(mvs_meta, dict) else 0.0
            if sync_i >= 0 and mvs_sync >= 0:
                self.trigger.setdefault("per_cam_ack", {})[cam] = int(mvs_sync)
                self.trigger["per_cam_sent"][cam] = int(sync_i)
                miss = max(0, int(sync_i) - int(mvs_sync))
                self.trigger["per_cam_trigger_miss_count"][cam] = miss
            if cap_wall > 1_000_000_000_000_000 and wall_us > 1_000_000_000_000_000:
                self.trigger["per_cam_trigger_to_mvs_frame_ms"][cam] = (cap_wall - wall_us) / 1000.0
            self.trigger["per_cam_trigger_to_event_trigger_ms"][cam] = _safe_float(self.event["per_cam_event_trigger_last_age_ms"].get(cam), -1.0)
        self.event["event_active_count"] = int(active)

    def _poll_event_log(self) -> None:
        self._poll_event_worker_status_files()
        for ln in self.event_log.poll(max_lines=1024):
            m = re.search(r'\[EventPerf\]\[([^\]]+)\].*?raw=(\d+).*?raw_rate=([0-9.]+)MEv/s', ln)
            if m:
                cam = str(m.group(1)).strip()
                if cam in self.cams:
                    self.event["per_cam_event_rate_mevs"][cam] = float(m.group(3))
                    self.event["per_cam_event_drop_count"][cam] = _safe_int(self.event["per_cam_event_drop_count"].get(cam, 0), 0)
                    self.event["per_cam_event_guard_state"][cam] = "OK"
                continue
            nm = re.search(r'\[HAL\]\[ERROR\].*NonMonotonic', ln)
            if nm:
                # The HAL error line is not always tagged with camera alias. Mark all
                # event-active cameras as suspect rather than silently hiding it.
                for cam in self.cams:
                    if self.event["per_cam_ready"].get(cam, False):
                        self.event["per_cam_event_nonmonotonic_count"][cam] = int(self.event["per_cam_event_nonmonotonic_count"].get(cam, 0)) + 1
                        self.event["per_cam_event_guard_state"][cam] = "TIME_ERR"
                continue
            sm = re.search(r'\[ShakeGuard\]\[([^\]]+)\].*?ACTIVE', ln)
            if sm and sm.group(1) in self.cams:
                self.event["per_cam_event_shake_guard_active"][sm.group(1)] = True
                self.event["per_cam_event_guard_state"][sm.group(1)] = "SHAKE"
            rm = re.search(r'\[ShakeGuard\]\[([^\]]+)\].*?RELEASE', ln)
            if rm and rm.group(1) in self.cams:
                self.event["per_cam_event_shake_guard_active"][rm.group(1)] = False
                self.event["per_cam_event_guard_state"][rm.group(1)] = "OK"
            pm = re.search(r'\[PreSpatter\]\[([^\]]+)\].*?ENTER', ln)
            if pm and pm.group(1) in self.cams:
                self.event["per_cam_event_pre_spatter_guard_active"][pm.group(1)] = True
                self.event["per_cam_event_guard_state"][pm.group(1)] = "PRE_SPATTER"

    def _poll_frames(self) -> None:
        now_s = time.time()
        now_ms = now_s * 1000.0
        now_mono_ns = time.monotonic_ns()
        active = 0
        raw_missing_map: Dict[str, bool] = {}
        mvs_missing_map: Dict[str, bool] = {}
        for cam in self.cams:
            raw_missing = True
            rd = self._open_raw(cam)
            if rd is not None:
                try:
                    fid = int(rd.header[_R_FRAME_ID])
                    update_ns = int(rd.header[_R_UPDATE_MONO_NS])
                    raw_missing = False
                    if cam in self.prev_raw:
                        pfid, pt = self.prev_raw[cam]
                        dt = max(1e-6, now_s - pt)
                        if fid >= pfid:
                            self.raw_fps[cam] = (fid - pfid) / dt
                    self.prev_raw[cam] = (fid, now_s)
                    self.raw_age_ms[cam] = max(0.0, (now_mono_ns - update_ns) / 1e6) if update_ns > 0 else -1.0
                except Exception:
                    raw_missing = True
            mvs_missing = True
            mr = self._open_mvs(cam)
            if mr is not None:
                try:
                    sm = mr.sample()
                    if sm.get('ok'):
                        fid = int(sm.get('frame_id', -1))
                        mvs_missing = False
                        if cam in self.prev_mvs:
                            pfid, pt = self.prev_mvs[cam]
                            dt = max(1e-6, now_s - pt)
                            if fid >= pfid:
                                self.mvs_fps[cam] = (fid - pfid) / dt
                        self.prev_mvs[cam] = (fid, now_s)
                        if self.mvs_fps.get(cam, 0.0) > 1.0:
                            active += 1
                except Exception:
                    mvs_missing = True

            meta = self._read_meta(cam)
            if meta:
                self.pending_q[cam] = _safe_int(meta.get('split_worker_pending_q', meta.get('pending_q', -1)), -1)
                wall_us = self._pick_wall_us(meta)
                if wall_us > 0:
                    age = max(0.0, now_ms - wall_us / 1000.0)
                    # If the sidecar wall time is stale but frames are updating, do not
                    # show a misleading multi-minute age.  The raw age is the freshest
                    # producer-side signal for preview.
                    if age <= 5000.0 or self.mvs_fps.get(cam, 0.0) < 1.0:
                        self.mvs_age_ms[cam] = age
                    elif self.raw_age_ms.get(cam, -1.0) >= 0.0:
                        self.mvs_age_ms[cam] = self.raw_age_ms[cam]

            raw_missing_map[cam] = raw_missing
            mvs_missing_map[cam] = mvs_missing
            self.shm_missing[cam] = bool(raw_missing or mvs_missing)
        self.raw_missing = raw_missing_map
        self.mvs_missing = mvs_missing_map
        self.mvs_active_count = active

    def _poll_trigger_status(self) -> None:
        status_path = self._best_existing_path('trigger_status.json', prefer_json_wall=True)
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            wall_us = _safe_float(obj.get('wall_time_us', 0.0), 0.0)
            self.trigger.update({
                'state': str(obj.get('state', 'running') or 'running'),
                'fps': _safe_float(obj.get('trigger_fps', obj.get('fps', 0.0)), 0.0),
                'period_ms': _safe_float(obj.get('period_ms', self.trigger.get('period_ms', -1.0)), -1.0),
                'jitter_p50_ms': _safe_float(obj.get('jitter_p50_ms', -1.0), -1.0),
                'jitter_p95_ms': _safe_float(obj.get('jitter_p95_ms', -1.0), -1.0),
                'jitter_max_ms': _safe_float(obj.get('jitter_max_ms', -1.0), -1.0),
                'last_age_ms': max(0.0, time.time() * 1000.0 - wall_us / 1000.0) if wall_us > 0 else -1.0,
                'ok_cams': _safe_int(obj.get('ok_cams', 0), 0),
            })
            if isinstance(obj.get('per_cam_sent'), dict):
                self.trigger['per_cam_sent'] = dict(obj.get('per_cam_sent') or {})
            if isinstance(obj.get('per_cam_fail'), dict):
                self.trigger['per_cam_fail'] = dict(obj.get('per_cam_fail') or {})
            return
        except Exception:
            pass
        for ln in self.trigger_log.poll():
            m = re.search(r'tick=(\d+) ok_cams=(\d+)/(\d+) sent_rate=([0-9.]+)/s', ln)
            if m:
                ok = int(m.group(2)); total = max(1, int(m.group(3))); sent_rate = float(m.group(4))
                self.trigger.update({'state': 'running', 'fps': sent_rate / max(1, ok), 'ok_cams': ok, 'last_age_ms': 0.0})
            elif 'waiting' in ln:
                self.trigger['state'] = 'waiting'
            elif '[WARN]' in ln and 'late' in ln:
                self.trigger['state'] = 'late'

    def _poll_mvs_agg_log(self) -> None:
        for ln in self.mvs_agg_log.poll():
            if '[perf]' not in ln:
                continue
            m = re.search(r'active=(\d+)/(\d+).*?batch/s=([0-9.]+).*?infer_frames/s=([0-9.]+).*?last_infer_ms=([0-9.]+)', ln)
            if m:
                self.mvs_active_count = int(m.group(1))
                self.yolo['infer_fps'] = float(m.group(4))
                if self.yolo.get('predict_ms', 0.0) <= 0:
                    self.yolo['predict_ms'] = float(m.group(5))
            cm = re.search(r'cap_fps\[([^\]]+)\]', ln)
            if cm:
                for part in cm.group(1).split():
                    if ':' in part:
                        c, v = part.split(':', 1)
                        if c in self.mvs_fps:
                            self.mvs_fps[c] = _safe_float(v, self.mvs_fps[c])

    def _poll_yolo_worker_status(self) -> None:
        """Aggregate per-shard YOLO worker status files.

        In sharded mode there may be no MVS_AGG perf line.  Each worker writes
        yolo_worker_<id>_status.json under sync.root; aggregate camera FPS and
        timings here for the status window and offline CSV.
        """
        paths: List[str] = []
        for root in getattr(self, "sync_roots", [self.sync_root]):
            try:
                paths.extend(glob.glob(os.path.join(str(root), "yolo_worker_*_status.json")))
                paths.extend(glob.glob(os.path.join(str(root), "global_yolo_worker_*_status.json")))
            except Exception:
                pass
        latest_by_worker: Dict[str, Dict[str, Any]] = {}
        now_us = int(time.time() * 1_000_000)
        for path in sorted(set(paths)):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if not isinstance(obj, dict):
                    continue
                worker_id = str(obj.get("worker_id", os.path.basename(path))).strip() or os.path.basename(path)
                wall_us = _safe_int(obj.get("wall_us", 0), 0)
                # Ignore stale status files from previous runs.
                if wall_us > 0 and now_us - wall_us > 5_000_000:
                    continue
                old = latest_by_worker.get(worker_id)
                if old is None or _safe_int(obj.get("wall_us", 0), 0) >= _safe_int(old.get("wall_us", 0), 0):
                    latest_by_worker[worker_id] = obj
            except Exception:
                continue
        if not latest_by_worker:
            return
        total_camera_fps = 0.0
        total_inferred_fps = 0.0
        total_published_fps = 0.0
        postprocess_queue_depth = 0
        postprocess_dropped_batches = 0
        polygon_contract_ok = True
        async_record_postprocess_enabled = False
        published_records_with_polygon_ratio = 1.0
        published_persons_with_polygon_ratio = 1.0
        total_batch_size = 0
        newest = None
        for rec in latest_by_worker.values():
            last_batch = rec.get("last_batch", {}) if isinstance(rec.get("last_batch"), dict) else {}
            batch_size = _safe_int(rec.get("last_batch_size", last_batch.get("batch_size", 0)), 0)
            total_ms = _safe_float(last_batch.get("total_ms", rec.get("total_ms", 0.0)), 0.0)
            fps = _safe_float(rec.get("infer_camera_fps_window", rec.get("instant_camera_fps", 0.0)), 0.0)
            if fps <= 0.0 and batch_size > 0 and total_ms > 0.0:
                fps = 1000.0 * float(batch_size) / max(1e-6, total_ms)
            total_camera_fps += float(fps)
            inferred_fps = _safe_float(rec.get("inferred_fps_total", rec.get("inferred_fps", fps)), fps)
            published_fps = _safe_float(rec.get("published_fps_total", rec.get("published_fps", rec.get("fps", fps))), fps)
            total_inferred_fps += float(inferred_fps)
            total_published_fps += float(published_fps)
            postprocess_queue_depth += _safe_int(rec.get("postprocess_queue_depth"), 0)
            postprocess_dropped_batches += _safe_int(rec.get("postprocess_dropped_batches"), 0)
            async_record_postprocess_enabled = async_record_postprocess_enabled or bool(rec.get("async_record_postprocess_enabled", False))
            if rec.get("polygon_contract_ok") is False:
                polygon_contract_ok = False
            published_records_with_polygon_ratio = min(
                published_records_with_polygon_ratio,
                _safe_float(rec.get("published_records_with_polygon_ratio"), 1.0),
            )
            published_persons_with_polygon_ratio = min(
                published_persons_with_polygon_ratio,
                _safe_float(rec.get("published_persons_with_polygon_ratio"), 1.0),
            )
            total_batch_size += int(batch_size)
            if newest is None or _safe_int(rec.get("wall_us", 0), 0) > _safe_int(newest.get("wall_us", 0), 0):
                newest = rec
        self.yolo["workers"] = latest_by_worker
        self.yolo["worker_count"] = len(latest_by_worker)
        self.yolo["infer_fps"] = total_published_fps if total_published_fps > 0.0 else total_camera_fps
        self.yolo["inferred_fps_total"] = total_inferred_fps if total_inferred_fps > 0.0 else total_camera_fps
        self.yolo["published_fps_total"] = total_published_fps if total_published_fps > 0.0 else total_camera_fps
        self.yolo["async_record_postprocess_enabled"] = bool(async_record_postprocess_enabled)
        self.yolo["postprocess_queue_depth"] = int(postprocess_queue_depth)
        self.yolo["postprocess_dropped_batches"] = int(postprocess_dropped_batches)
        self.yolo["polygon_contract_ok"] = bool(polygon_contract_ok and postprocess_dropped_batches == 0)
        self.yolo["published_records_with_polygon_ratio"] = float(published_records_with_polygon_ratio)
        self.yolo["published_persons_with_polygon_ratio"] = float(published_persons_with_polygon_ratio)
        self.yolo["fps_source"] = "worker_status.published_fps_total"
        if newest:
            newest_last_batch = newest.get("last_batch", {}) if isinstance(newest.get("last_batch"), dict) else {}
            self.yolo["batch_size"] = total_batch_size or _safe_int(newest.get("last_batch_size", newest_last_batch.get("batch_size", 0)), 0)
            self.yolo["predict_ms"] = _safe_float(newest.get("predict_ms", newest_last_batch.get("predict_ms", self.yolo.get("predict_ms", 0.0))), 0.0)
            self.yolo["capture_to_result_ms"] = _safe_float(newest.get("capture_to_result_ms", newest_last_batch.get("total_ms", self.yolo.get("capture_to_result_ms", 0.0))), self.yolo.get("capture_to_result_ms", 0.0))

    def _poll_global_yolo_status(self) -> None:
        status = self._read_status_json("global_yolo_status.json")
        if not status:
            return
        age_ms = self._status_wall_age_ms(status)
        if age_ms >= 0.0 and age_ms > 5000.0:
            return
        summary = status.get("summary", {}) if isinstance(status.get("summary"), dict) else {}
        last_batch = status.get("last_batch", {}) if isinstance(status.get("last_batch"), dict) else {}
        published_fps = _safe_float(
            status.get("published_fps_total", status.get("fps", summary.get("published_fps_total", 0.0))),
            0.0,
        )
        inferred_fps = _safe_float(
            status.get("inferred_fps_total", summary.get("inferred_fps_total", published_fps)),
            published_fps,
        )
        capacity_fps = _safe_float(
            summary.get("last_predict_fps_total", status.get("last_predict_fps_total", 0.0)),
            0.0,
        )
        if capacity_fps <= 0.0:
            capacity_fps = _safe_float(summary.get("last_total_fps_total", status.get("last_total_fps_total", published_fps)), published_fps)
        polygon_ok_raw = status.get("polygon_contract_ok", summary.get("polygon_contract_ok", None))
        self.yolo.update({
            "infer_fps": published_fps if published_fps > 0.0 else inferred_fps,
            "published_fps_total": published_fps,
            "inferred_fps_total": inferred_fps,
            "capacity_fps": capacity_fps,
            "fps_source": "global_yolo_status.published_fps_total",
            "worker_count": _safe_int(status.get("worker_count"), _safe_int(summary.get("worker_count"), self.yolo.get("worker_count", 0))),
            "batch_size": _safe_int(last_batch.get("batch_size"), _safe_int(self.yolo.get("batch_size"), 0)),
            "predict_ms": _safe_float(last_batch.get("predict_ms"), _safe_float(self.yolo.get("predict_ms"), 0.0)),
            "capture_to_result_ms": _safe_float(last_batch.get("total_ms"), _safe_float(self.yolo.get("capture_to_result_ms"), 0.0)),
            "async_record_postprocess_enabled": bool(status.get("async_record_postprocess_enabled", summary.get("async_record_postprocess_enabled", False))),
            "postprocess_queue_depth": _safe_int(status.get("postprocess_queue_depth", summary.get("postprocess_queue_depth", 0)), 0),
            "postprocess_lag_ms": _safe_float(status.get("postprocess_lag_ms"), 0.0),
            "postprocess_backpressure_ms": _safe_float(status.get("postprocess_backpressure_ms"), 0.0),
            "postprocess_dropped_batches": _safe_int(status.get("postprocess_dropped_batches", summary.get("postprocess_dropped_batches", 0)), 0),
            "polygon_contract_ok": None if polygon_ok_raw is None else bool(polygon_ok_raw),
            "published_records_with_polygon_ratio": _safe_float(status.get("published_records_with_polygon_ratio", summary.get("published_records_with_polygon_ratio", 1.0)), 1.0),
            "published_persons_with_polygon_ratio": _safe_float(status.get("published_persons_with_polygon_ratio", summary.get("published_persons_with_polygon_ratio", 1.0)), 1.0),
            "global_yolo_status_age_ms": age_ms,
        })
        if isinstance(status.get("workers"), list):
            self.yolo["merge_workers"] = status.get("workers")

    def _poll_hit_log(self) -> None:
        for ln in self.hit_log.poll():
            if '[perf]' in ln:
                m = re.search(r'recv=(\d+).*?cand=(\d+).*?wait=(\d+).*?drop=(\d+).*?no_human=(\d+).*?no_cam=(\d+).*?pending=(\d+).*?pending_impacts=(\d+)', ln)
                if m:
                    recv = int(m.group(1)); cand = int(m.group(2)); wait = int(m.group(3)); drop = int(m.group(4)); nh = int(m.group(5)); no_cam = int(m.group(6)); pending = int(m.group(7)); pending_impacts = int(m.group(8))
                    self.hit['recv_per_s'] = float(recv)
                    self.hit['eval_per_s'] = float(recv + wait + cand)
                    self.hit['candidate_per_s'] = float(cand)
                    self.hit['confirmed_per_s'] = float(cand)
                    self.hit['drop_per_s'] = float(drop)
                    self.hit['no_human_per_s'] = float(nh)
                    self.hit['no_cam_per_s'] = float(no_cam)
                    self.hit['pending_count'] = int(pending)
                    self.hit['pending_impacts'] = int(pending_impacts)
                    self.hit['no_human_count'] = int(self.hit.get('no_human_count', 0)) + nh
                    continue
                m2 = re.search(r'recv=(\d+).*?no_human=(\d+)', ln)
                if m2:
                    recv = int(m2.group(1)); nh = int(m2.group(2))
                    self.hit['recv_per_s'] = float(recv)
                    self.hit['no_human_per_s'] = float(nh)
                    self.hit['no_human_count'] = int(self.hit.get('no_human_count', 0)) + nh
                    continue
            # Reason counters from debug/status lines. These are approximate per-poll
            # rates because the source log does not always expose a fixed interval.
            if 'no_human_result' in ln or 'point_no_human' in ln:
                self.hit['no_human_per_s'] = max(float(self.hit.get('no_human_per_s', 0.0)), 1.0)
            if 'SYNC_DELTA_TOO_LARGE' in ln or 'frame_delta' in ln and 'reject' in ln:
                self.hit['sync_bad_per_s'] = max(float(self.hit.get('sync_bad_per_s', 0.0)), 1.0)
            if 'outside_mask' in ln or 'no_person_geometry_match' in ln:
                self.hit['outside_mask_per_s'] = max(float(self.hit.get('outside_mask_per_s', 0.0)), 1.0)
            if 'stale' in ln and 'human' in ln:
                self.hit['stale_human_per_s'] = max(float(self.hit.get('stale_human_per_s', 0.0)), 1.0)

    def _sync_quality(self, delta_ms: float, human_age_ms: float, has_human: bool) -> str:
        if not has_human:
            return "MISSING"
        if human_age_ms > 500.0:
            return "STALE"
        if delta_ms < -900000.0:
            return "UNKNOWN"
        ad = abs(float(delta_ms))
        if ad <= 25.0:
            return "OK"
        if ad <= 75.0:
            return "WARN"
        if ad <= 125.0:
            return "BAD"
        return "BAD"

    def _poll_bullet_and_sync(self, last_frame_ids: Dict[str, int]) -> None:
        snap = self.overlay_state.snapshot()
        pts = list(snap.get('bullet_points', []))
        events = list(snap.get('bullet_events', []))
        humans = snap.get('humans', {}) if isinstance(snap, dict) else {}
        now_ms = _now_ms()
        ttl_arg = _safe_float(getattr(self.args, 'bullet_status_ttl_ms', 0.0), 0.0)
        traj_life_ms = _safe_float(getattr(self.args, 'trajectory_life_ms', 700.0), 700.0)
        status_ttl_ms = ttl_arg if ttl_arg > 0 else max(5000.0, traj_life_ms * 4.0)
        future_tolerance_ms = min(1000.0, max(250.0, status_ttl_ms * 0.1))
        fresh_pts: List[Dict[str, Any]] = []
        stale_point_count = 0
        future_point_count = 0
        for pt in pts:
            if not isinstance(pt, dict):
                stale_point_count += 1
                continue
            ts_ms = _safe_float(pt.get('ts_ms', 0.0), 0.0)
            age = now_ms - ts_ms if ts_ms > 0 else status_ttl_ms + 1.0
            if ts_ms <= 0 or age > status_ttl_ms:
                stale_point_count += 1
                continue
            if age < -future_tolerance_ms:
                future_point_count += 1
                continue
            fresh_pts.append(pt)
        active_track_keys = set()
        point_count_by_key: Dict[Tuple[str, int], int] = {}
        latest_pt = None
        latest_ts = -1.0
        for pt in fresh_pts:
            if not isinstance(pt, dict):
                continue
            cam = str(pt.get('cam', ''))
            bid = _safe_int(pt.get('bullet_id', 0), 0)
            key = (cam, bid)
            age = max(0.0, now_ms - _safe_float(pt.get('ts_ms', 0.0), 0.0))
            if age <= 1500.0:
                active_track_keys.add(key)
                point_count_by_key[key] = int(point_count_by_key.get(key, 0)) + 1
            if _safe_float(pt.get('ts_ms', 0.0), 0.0) > latest_ts:
                latest_ts = _safe_float(pt.get('ts_ms', 0.0), 0.0)
                latest_pt = pt
        if latest_pt:
            cam = str(latest_pt.get('cam', ''))
            bid = _safe_int(latest_pt.get('bullet_id', 0), 0)
            latest_age = max(0.0, now_ms - _safe_float(latest_pt.get('ts_ms', 0.0), 0.0))
            self.bullet.update({
                'latest_age_ms': latest_age,
                'latest_cam': cam,
                'latest_id': bid,
                'latest_point_count': int(point_count_by_key.get((cam, bid), 1)),
                'latest_end_wall_us': _safe_int(latest_pt.get('wall_time_us', 0), 0),
                'track_count_active': len(active_track_keys),
            })
        else:
            self.bullet.update({
                'latest_age_ms': -1.0,
                'latest_cam': '',
                'latest_id': 0,
                'latest_point_count': 0,
                'latest_end_wall_us': 0,
                'track_count_active': len(active_track_keys),
            })
        self.bullet['recent'] = fresh_pts[-12:]
        self.bullet['udp_recv_per_s'] = float(self.hit.get('recv_per_s', 0.0))
        self.bullet['trajectory_drawn_vtx'] = _safe_int(getattr(self.args, '_trajectory_drawn_vtx', 0), 0)
        self.bullet['trajectory_visible_segment_count'] = max(0, int(self.bullet.get('trajectory_drawn_vtx', 0)) // 2)
        self.bullet['trajectory_latest_age_ms'] = _safe_float(self.bullet.get('latest_age_ms', -1.0), -1.0)
        self.bullet['trajectory_stale_guard_count'] = _safe_int(getattr(self.args, '_trajectory_stale_guard_count', 0), 0)
        self.bullet['trajectory_future_drop_count'] = _safe_int(getattr(self.args, '_trajectory_future_drop_count', 0), 0)
        self.bullet['bullet_status_ttl_ms'] = status_ttl_ms
        self.bullet['fresh_point_count'] = len(fresh_pts)
        self.bullet['stale_point_count'] = int(stale_point_count)
        self.bullet['future_point_count'] = int(future_point_count)
        self.bullet['stale_drop_count'] = int(stale_point_count + future_point_count)
        self.bullet['trajectory_fresh_not_drawn'] = int(
            max(0, int(self.bullet.get('fresh_point_count', 0) or 0))
            if int(self.bullet.get('trajectory_drawn_vtx', 0) or 0) <= 0 else 0
        )
        if int(self.bullet.get('trajectory_drawn_vtx', 0) or 0) > 0:
            draw_state = 'drawing'
        elif int(self.bullet.get('fresh_point_count', 0) or 0) > 0:
            draw_state = 'fresh_not_drawn'
        elif int(self.bullet.get('future_point_count', 0) or 0) > 0:
            draw_state = 'future_points'
        elif int(self.bullet.get('stale_point_count', 0) or 0) > 0:
            draw_state = 'stale_points'
        else:
            draw_state = 'idle'
        self.bullet['trajectory_draw_state'] = draw_state
        # Bullet <-> human sync, based on the latest visual point and latest human mask for that camera.
        recent_sync: List[Dict[str, Any]] = []
        latest_sync = None
        candidates = fresh_pts[-32:]
        for pt in candidates:
            if not isinstance(pt, dict):
                continue
            cam = str(pt.get('cam', ''))
            if not cam:
                continue
            human = humans.get(cam) if isinstance(humans, dict) else None
            bullet_wall_us = _safe_float(pt.get('wall_time_us', 0.0), 0.0)
            if bullet_wall_us <= 0:
                bullet_wall_us = _safe_float(pt.get('ts_ms', 0.0), 0.0) * 1000.0
            mvs_wall_us = 0.0
            meta = self._read_meta(cam)
            if meta:
                mvs_wall_us = self._pick_wall_us(meta)
            human_capture_us = _safe_float(human.get('capture_wall_us', 0.0), 0.0) if isinstance(human, dict) else 0.0
            human_record_us = _safe_float(human.get('record_wall_us', 0.0), 0.0) if isinstance(human, dict) else 0.0
            event_to_mvs_ms = (bullet_wall_us - mvs_wall_us) / 1000.0 if bullet_wall_us > 0 and mvs_wall_us > 0 else -999999.0
            event_to_human_ms = (bullet_wall_us - human_capture_us) / 1000.0 if bullet_wall_us > 0 and human_capture_us > 0 else -999999.0
            human_age_ms = max(0.0, now_ms - human_record_us / 1000.0) if human_record_us > 0 else -1.0
            human_frame = _safe_int(human.get('frame_id', -1), -1) if isinstance(human, dict) else -1
            cur_frame = _safe_int(last_frame_ids.get(cam, -1), -1)
            delta_frames = cur_frame - human_frame if cur_frame >= 0 and human_frame >= 0 else 999999
            person_count = _safe_int(human.get('person_count', 0), 0) if isinstance(human, dict) else 0
            quality = self._sync_quality(event_to_human_ms, human_age_ms, person_count > 0)
            rec = {
                'cam': cam,
                'bullet_id': _safe_int(pt.get('bullet_id', 0), 0),
                'point_index': _safe_int(pt.get('point_index', -1), -1),
                'event_ts_us': _safe_int(pt.get('point_ts_us', 0), 0),
                'bullet_wall_us': int(bullet_wall_us),
                'event_to_mvs_delta_ms': event_to_mvs_ms,
                'event_to_human_capture_delta_ms': event_to_human_ms,
                'event_to_human_result_age_ms': human_age_ms,
                'event_to_human_delta_frames': delta_frames,
                'mask_valid_at_event': bool(person_count > 0 and quality in ('OK', 'WARN')),
                'person_count_at_event': person_count,
                'matched_person_id': '',
                'matched_stable_id': '',
                'hit_inside_mask': False,
                'hit_distance_to_mask_px': -1.0,
                'hit_distance_to_bbox_px': -1.0,
                'sync_quality': quality,
            }
            recent_sync.append(rec)
            latest_sync = rec
        # Override latest sync with final hit candidate if available because it carries
        # hit-judge-selected person/frame metadata.
        raw_decisions = list(snap.get('hit_decisions', []))[-32:]
        decisions: List[Dict[str, Any]] = []
        for dec in raw_decisions:
            if not isinstance(dec, dict):
                continue
            dec_ts = _safe_float(dec.get('ts_ms', 0.0), 0.0)
            dec_age = now_ms - dec_ts if dec_ts > 0 else status_ttl_ms + 1.0
            if -future_tolerance_ms <= dec_age <= status_ttl_ms:
                decisions.append(dec)
        if decisions:
            last_dec = decisions[-1]
            cam = str(last_dec.get('cam', ''))
            if cam:
                human = humans.get(cam) if isinstance(humans, dict) else None
                human_age_ms = max(0.0, now_ms - _safe_float(human.get('record_wall_us', 0.0), 0.0) / 1000.0) if isinstance(human, dict) and _safe_float(human.get('record_wall_us', 0.0), 0.0) > 0 else -1.0
                df = _safe_int(last_dec.get('person_frame_delta_sync', 999999), 999999)
                quality = 'OK' if abs(df) <= 1 else ('WARN' if abs(df) <= 3 else 'BAD')
                latest_sync = {
                    'cam': cam,
                    'bullet_id': _safe_int(last_dec.get('bullet_id', 0), 0),
                    'event_ts_us': _safe_int(last_dec.get('event_ts_us', 0), 0),
                    'event_to_mvs_delta_ms': _safe_float(last_dec.get('impact_offset_ms', -999999.0), -999999.0),
                    'event_to_human_capture_delta_ms': _safe_float(last_dec.get('impact_offset_ms', -999999.0), -999999.0),
                    'event_to_human_result_age_ms': human_age_ms,
                    'event_to_human_delta_frames': df,
                    'mask_valid_at_event': True,
                    'person_count_at_event': _safe_int(human.get('person_count', 0), 0) if isinstance(human, dict) else 0,
                    'matched_person_id': '',
                    'matched_stable_id': str(last_dec.get('stable_id', '')),
                    'hit_inside_mask': True,
                    'hit_distance_to_mask_px': _safe_float(last_dec.get('mask_distance_px', -1.0), -1.0),
                    'hit_distance_to_bbox_px': -1.0,
                    'sync_quality': quality,
                    'latest_hit_reason': str(last_dec.get('reason', 'HIT')),
                }
        if latest_sync:
            self.sync_health.update(latest_sync)
        else:
            self.sync_health.update({
                'cam': '',
                'latest_cam': '',
                'bullet_id': 0,
                'event_to_mvs_delta_ms': -1.0,
                'event_to_human_capture_delta_ms': -1.0,
                'event_to_human_result_age_ms': -1.0,
                'event_to_human_delta_frames': 999999,
                'mask_valid_at_event': False,
                'person_count_at_event': 0,
                'matched_person_id': '',
                'matched_stable_id': '',
                'hit_inside_mask': False,
                'hit_distance_to_mask_px': -1.0,
                'hit_distance_to_bbox_px': -1.0,
                'sync_quality': 'UNKNOWN',
                'latest_hit_reason': '',
            })
        self.sync_health['recent'] = recent_sync[-12:]
        self.sync_health['hit_decisions'] = decisions[-20:]

    def _poll_yolo_from_humans(self, last_frame_ids: Dict[str, int]) -> None:
        snap = self.overlay_state.snapshot()
        humans = snap.get('humans', {}) if isinstance(snap, dict) else {}
        ages: Dict[str, float] = {}
        deltas: Dict[str, int] = {}
        latest_rec = None
        latest_ts = -1.0
        now_ms = _now_ms()
        for cam, rec in humans.items():
            if not isinstance(rec, dict):
                continue
            ts_ms = _safe_float(rec.get('ts_ms', 0.0), 0.0)
            ages[str(cam)] = max(0.0, now_ms - ts_ms) if ts_ms > 0 else -1.0
            fid = _safe_int(rec.get('frame_id', -1), -1)
            cur_fid = int(last_frame_ids.get(str(cam), -1))
            if cur_fid >= 0 and fid >= 0:
                deltas[str(cam)] = cur_fid - fid
            if ts_ms > latest_ts:
                latest_ts = ts_ms
                latest_rec = rec
        if latest_rec:
            self.yolo['batch_size'] = _safe_int(latest_rec.get('yolo_batch_size', self.yolo.get('batch_size', 0)), 0)
            self.yolo['predict_ms'] = _safe_float(latest_rec.get('yolo_predict_ms', self.yolo.get('predict_ms', 0.0)), 0.0)
            self.yolo['capture_to_result_ms'] = _safe_float(latest_rec.get('yolo_capture_to_result_ms', self.yolo.get('capture_to_result_ms', 0.0)), 0.0)
        self.yolo['result_video_delta_frames'] = deltas
        self.human_ages = ages

    def _safe_poll_step(self, name: str, fn: Any) -> None:
        try:
            fn()
            self.last_poll_errors.pop(name, None)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            self.last_poll_errors[name] = msg
            # Throttle repeated identical errors but keep the first/fresh ones visible.
            print(f"[pygl_cuda_preview][WARN] telemetry step {name} failed: {msg}", flush=True)

    def poll(self, last_frame_ids: Dict[str, int]) -> Dict[str, Any]:
        now = _now_ms()
        if now - self.last_poll_ms < float(getattr(self.args, 'telemetry_poll_ms', 500.0)):
            return self.last_snapshot
        self.last_poll_ms = now
        self._safe_poll_step('frames', self._poll_frames)
        self._safe_poll_step('timebase', self._poll_timebase)
        self._safe_poll_step('trigger_status', self._poll_trigger_status)
        self._safe_poll_step('event_triggers', self._poll_event_triggers)
        self._safe_poll_step('event_log', self._poll_event_log)
        self._safe_poll_step('mvs_agg_log', self._poll_mvs_agg_log)
        self._safe_poll_step('yolo_worker_status', self._poll_yolo_worker_status)
        self._safe_poll_step('global_yolo_status', self._poll_global_yolo_status)
        self._safe_poll_step('hit_log', self._poll_hit_log)
        self._safe_poll_step('yolo_from_humans', lambda: self._poll_yolo_from_humans(last_frame_ids))
        self._safe_poll_step('bullet_and_sync', lambda: self._poll_bullet_and_sync(last_frame_ids))
        # Prefer live shm-based active count if available; it is less fragile than
        # log parsing and continues to work when mvs_agg perf lines are sparse.
        try:
            self.mvs_active_count = sum(1 for c in self.cams if self.mvs_fps.get(c, 0.0) > 1.0 or self.raw_fps.get(c, 0.0) > 1.0)
        except Exception:
            pass
        self.last_snapshot = {
            'timebase': dict(self.timebase),
            'trigger': dict(self.trigger),
            'event': dict(self.event),
            'bullet': dict(self.bullet),
            'sync_health': dict(self.sync_health),
            'mvs_active_count': int(self.mvs_active_count),
            'mvs_fps': dict(self.mvs_fps),
            'raw_fps': dict(self.raw_fps),
            'frame_age_ms': dict(self.mvs_age_ms),
            'raw_age_ms': dict(self.raw_age_ms),
            'raw_missing': dict(getattr(self, 'raw_missing', {})),
            'mvs_missing': dict(getattr(self, 'mvs_missing', {})),
            'shm_missing': dict(self.shm_missing),
            'pending_q': dict(self.pending_q),
            'yolo': dict(self.yolo),
            'human_age_by_cam': dict(getattr(self, 'human_ages', {})),
            'hit': dict(self.hit),
            'poll_errors': dict(self.last_poll_errors),
        }
        return self.last_snapshot


class OverlayWsClient:
    def __init__(self, url: str, state: OverlayState, enabled: bool = True):
        self.url = str(url or "").strip()
        self.state = state
        self.enabled = bool(enabled and self.url)
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._thread_main, name="overlay-ws-client", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_evt.set()

    def _thread_main(self) -> None:
        try:
            import websockets  # type: ignore
        except Exception as exc:
            self.state.set_ws_status(False, f"import websockets failed: {exc}")
            print(f"[pygl_cuda_preview][WARN] overlay_ws disabled: import websockets failed: {exc}", flush=True)
            return

        async def run_loop() -> None:
            while not self.stop_evt.is_set():
                try:
                    async with websockets.connect(self.url, ping_interval=10, ping_timeout=5, max_size=2**20) as ws:
                        self.state.set_ws_status(True, "")
                        print(f"[pygl_cuda_preview] overlay_ws connected: {self.url}", flush=True)
                        async for raw in ws:
                            if self.stop_evt.is_set():
                                break
                            try:
                                obj = json.loads(raw)
                                if isinstance(obj, list):
                                    for x in obj:
                                        if isinstance(x, dict):
                                            self.state.update_from_overlay_msg(x, source="ws")
                                elif isinstance(obj, dict):
                                    self.state.update_from_overlay_msg(obj, source="ws")
                            except Exception as exc:
                                self.state.set_ws_status(True, f"decode failed: {exc}")
                except Exception as exc:
                    self.state.set_ws_status(False, str(exc))
                    time.sleep(1.0)

        asyncio.run(run_loop())




def _resolve_hit_jsonl_paths(args: argparse.Namespace, cams: List[str]) -> Tuple[List[str], str]:
    """Resolve hit_candidate jsonl paths from runtime/unified config when possible.

    The project often writes hit JSONL under the root configured in
    mvs_config.sync.root, which may differ from the current test directory.  The
    renderer still accepts command-line fallback templates, but config wins.
    """
    cfg_root = ""
    hit_template = str(getattr(args, "hit_jsonl_template", "hit_candidate_{camera}.jsonl") or "hit_candidate_{camera}.jsonl")
    hit_all = str(getattr(args, "hit_all_jsonl", "hit_candidate_all.jsonl") or "hit_candidate_all.jsonl")
    try:
        cfg_path = str(getattr(args, "config", "") or "")
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mvs = cfg.get("mvs_config", cfg) if isinstance(cfg, dict) else {}
            sync = mvs.get("sync", {}) if isinstance(mvs, dict) else {}
            hit = mvs.get("hit_judge", {}) if isinstance(mvs, dict) else {}
            if isinstance(sync, dict):
                cfg_root = str(sync.get("root", "") or "")
            if isinstance(hit, dict):
                hit_template = str(hit.get("hit_jsonl_template", hit_template) or hit_template)
                hit_all = str(hit.get("hit_all_jsonl", hit_all) or hit_all)
    except Exception as exc:
        print(f"[pygl_cuda_preview][WARN] failed to parse hit jsonl paths from config: {exc}", flush=True)
    root = Path(cfg_root or str(getattr(args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    paths: List[str] = []
    for cam in cams:
        hp = hit_template.format(camera=cam, cam=cam, root=str(root))
        hp_path = Path(hp).expanduser()
        if not hp_path.is_absolute():
            hp_path = root / hp_path
        paths.append(str(hp_path))
    if hit_all:
        ap = hit_all.format(root=str(root))
        ap_path = Path(ap).expanduser()
        if not ap_path.is_absolute():
            ap_path = root / ap_path
        paths.append(str(ap_path))
    return paths, str(root)


def _resolve_pseudo_hit_jsonl_paths(args: argparse.Namespace, cams: List[str]) -> Tuple[List[str], str]:
    cfg_root = ""
    pseudo_template = str(getattr(args, "pseudo_hit_jsonl_template", "pseudo_hit_{camera}.jsonl") or "pseudo_hit_{camera}.jsonl")
    pseudo_all = str(getattr(args, "pseudo_hit_all_jsonl", "pseudo_hit_all.jsonl") or "pseudo_hit_all.jsonl")
    try:
        cfg_path = str(getattr(args, "config", "") or "")
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mvs = cfg.get("mvs_config", cfg) if isinstance(cfg, dict) else {}
            sync = mvs.get("sync", {}) if isinstance(mvs, dict) else {}
            hit = mvs.get("hit_judge", {}) if isinstance(mvs, dict) else {}
            if isinstance(sync, dict):
                cfg_root = str(sync.get("root", "") or "")
            if isinstance(hit, dict):
                pseudo_template = str(hit.get("pseudo_hit_jsonl_template", pseudo_template) or pseudo_template)
                pseudo_all = str(hit.get("pseudo_hit_all_jsonl", pseudo_all) or pseudo_all)
    except Exception as exc:
        print(f"[pygl_cuda_preview][WARN] failed to parse pseudo hit jsonl paths from config: {exc}", flush=True)
    root = Path(cfg_root or str(getattr(args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    paths: List[str] = []
    for cam in cams:
        hp = pseudo_template.format(camera=cam, cam=cam, root=str(root))
        hp_path = Path(hp).expanduser()
        if not hp_path.is_absolute():
            hp_path = root / hp_path
        paths.append(str(hp_path))
    if pseudo_all:
        ap = pseudo_all.format(root=str(root))
        ap_path = Path(ap).expanduser()
        if not ap_path.is_absolute():
            ap_path = root / ap_path
        paths.append(str(ap_path))
    return paths, str(root)


class JsonlTailer:
    def __init__(self, paths: List[str], state: OverlayState, start_tail_bytes: int = 1_000_000):
        self.paths = [str(p) for p in paths if str(p)]
        self.state = state
        self.offsets: Dict[str, int] = {}
        self.start_tail_bytes = int(max(0, start_tail_bytes))

    def poll(self, max_lines_per_file: int = 256) -> None:
        for path in self.paths:
            try:
                if not os.path.exists(path):
                    continue
                size = os.path.getsize(path)
                off = self.offsets.get(path)
                if off is None:
                    off = max(0, size - self.start_tail_bytes)
                if size < off:
                    off = 0
                if size == off:
                    self.offsets[path] = off
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(off)
                    lines = f.readlines()
                    self.offsets[path] = f.tell()
                for line in lines[-int(max_lines_per_file):]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        self.state.update_from_overlay_msg(obj, source="jsonl")
            except Exception:
                continue


class PseudoHitSuppressionTailer:
    def __init__(self, paths: List[str], preview: Any, start_tail_bytes: int = 1_000_000):
        self.paths = [str(p) for p in paths if str(p)]
        self.preview = preview
        self.offsets: Dict[str, int] = {}
        self.start_tail_bytes = int(max(0, start_tail_bytes))
        self.rx_count = 0
        self.suppressed_rx_count = 0
        self.error = ""

    def poll(self, max_lines_per_file: int = 512) -> None:
        for path in self.paths:
            try:
                if not os.path.exists(path):
                    continue
                size = os.path.getsize(path)
                off = self.offsets.get(path)
                if off is None:
                    off = max(0, size - self.start_tail_bytes)
                if size < off:
                    off = 0
                if size == off:
                    self.offsets[path] = off
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(off)
                    lines = f.readlines()
                    self.offsets[path] = f.tell()
                for line in lines[-int(max_lines_per_file):]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    self.rx_count += 1
                    if not bool(obj.get("final_hit_suppressed_by_human_noise", False)):
                        continue
                    self.suppressed_rx_count += 1
                    try:
                        self.preview.mark_trajectory_suppressed_from_pseudo_hit(obj)
                    except Exception:
                        continue
                self.error = ""
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"



def _resolve_human_jsonl_paths(args: argparse.Namespace, cams: List[str]) -> Tuple[List[str], str]:
    """Resolve human_result JSONL paths.

    Prefer --sync-root from the active launcher, but also include the root from
    config when it differs.  This protects against older unified configs whose
    sync.root still points at a previous test directory.
    """
    roots: List[Path] = []
    templates: List[str] = [str(getattr(args, "human_jsonl_template", "human_result_{camera}.jsonl") or "human_result_{camera}.jsonl")]

    def add_root(x: Any) -> None:
        if not x:
            return
        r = Path(str(x)).expanduser()
        if not r.is_absolute():
            r = Path.cwd() / r
        if str(r) not in [str(v) for v in roots]:
            roots.append(r)

    add_root(getattr(args, "sync_root", "sync_ipc"))
    try:
        cfg_path = str(getattr(args, "config", "") or "")
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            mvs = cfg.get("mvs_config", cfg) if isinstance(cfg, dict) else {}
            sync = mvs.get("sync", {}) if isinstance(mvs, dict) else {}
            direct = mvs.get("direct", {}) if isinstance(mvs, dict) else {}
            common = direct.get("common", {}) if isinstance(direct, dict) else {}
            if isinstance(sync, dict):
                add_root(sync.get("root", ""))
                ht = sync.get("human_jsonl_template", "")
                if ht and str(ht) not in templates:
                    templates.append(str(ht))
            if isinstance(common, dict):
                eht = common.get("event_human_mask_jsonl_template", "")
                if eht and str(eht) not in templates:
                    templates.append(str(eht))
    except Exception as exc:
        print(f"[pygl_cuda_preview][WARN] failed to parse human jsonl paths from config: {exc}", flush=True)

    paths: List[str] = []
    for root in roots:
        for tmpl in templates:
            for cam in cams:
                try:
                    hp = tmpl.format(camera=cam, cam=cam, root=str(root))
                except Exception:
                    hp = tmpl.replace("{camera}", cam).replace("{cam}", cam).replace("{root}", str(root))
                hp_path = Path(hp).expanduser()
                if not hp_path.is_absolute():
                    hp_path = root / hp_path
                sp = str(hp_path)
                if sp not in paths:
                    paths.append(sp)
    roots_s = ",".join(str(x) for x in roots) if roots else ""
    return paths, roots_s


class HumanJsonlTailer(JsonlTailer):
    def poll(self, max_lines_per_file: int = 96) -> None:
        for path in self.paths:
            try:
                if not os.path.exists(path):
                    continue
                size = os.path.getsize(path)
                off = self.offsets.get(path)
                if off is None:
                    off = max(0, size - self.start_tail_bytes)
                if size < off:
                    off = 0
                if size == off:
                    self.offsets[path] = off
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(off)
                    lines = f.readlines()
                    self.offsets[path] = f.tell()
                for line in lines[-int(max_lines_per_file):]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        self.state.update_from_human_msg(obj, source="human_jsonl")
            except Exception:
                continue


def _read_latest_human_records_for_status(args: argparse.Namespace, cams: List[str], tail_bytes: int = 2_000_000) -> Dict[str, Dict[str, Any]]:
    paths, _roots = _resolve_human_jsonl_paths(args, cams)
    out: Dict[str, Dict[str, Any]] = {}
    cam_set = {str(c) for c in cams}
    for path in paths:
        try:
            if not os.path.exists(path):
                continue
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                f.seek(max(0, size - int(max(8192, tail_bytes))))
                raw = f.read().decode("utf-8", "ignore")
            for line in raw.splitlines()[-512:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("msg_type", obj.get("type", "")) or "") != "person_regions":
                    continue
                cam = str(obj.get("camera", obj.get("camera_name", "")) or "").strip()
                if cam not in cam_set:
                    continue
                old = out.get(cam)
                wall = _safe_float(obj.get("record_wall_us", obj.get("capture_wall_us", 0)), 0.0)
                old_wall = _safe_float(old.get("record_wall_us", old.get("capture_wall_us", 0)) if isinstance(old, dict) else 0, 0.0)
                if old is None or wall >= old_wall:
                    out[cam] = obj
        except Exception:
            continue
    return out


def _human_record_to_status_hrec(msg: Dict[str, Any]) -> Dict[str, Any]:
    persons = msg.get("persons", [])
    if not isinstance(persons, list):
        persons = []
    clean_persons: List[Dict[str, Any]] = []
    for person in persons[:64]:
        if not isinstance(person, dict):
            continue
        clean_persons.append({
            "stable_id": person.get("stable_id", person.get("person_id", "")),
            "person_id": person.get("person_id", ""),
            "confidence": _safe_float(person.get("confidence", 0.0), 0.0),
            "bbox_xyxy": person.get("bbox_xyxy", None),
            "foot_pixel": person.get("foot_pixel", person.get("centroid", None)),
            "local_xy": person.get("local_xy", None),
            "physical_coord": person.get("physical_coord", None),
        })
    capture_wall_us = _safe_float(msg.get("capture_wall_us", 0), 0.0)
    record_wall_us = _safe_float(msg.get("record_wall_us", capture_wall_us), capture_wall_us)
    return {
        "cam": str(msg.get("camera", msg.get("camera_name", "")) or ""),
        "record_wall_us": record_wall_us,
        "capture_wall_us": capture_wall_us,
        "frame_id": _safe_int(msg.get("frame_id_local", msg.get("mvs_frame_num", -1)), -1),
        "persons": clean_persons,
        "person_count": len(clean_persons),
        "projector": msg.get("projector", None),
        "yolo_batch_size": _safe_int(msg.get("yolo_batch_size", 0), 0),
        "yolo_predict_ms": _safe_float(msg.get("yolo_predict_ms", 0.0), 0.0),
        "yolo_total_batch_ms": _safe_float(msg.get("yolo_total_batch_ms", 0.0), 0.0),
        "yolo_capture_to_result_ms": max(0.0, (record_wall_us - capture_wall_us) / 1000.0) if record_wall_us > 0 and capture_wall_us > 0 else 0.0,
    }


class CudaPboVideoBackend:
    def __init__(self, cuda_device: int, tile_w: int, tile_h: int, cam_count: int):
        self.cuda_device = int(cuda_device)
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.cam_count = int(cam_count)
        self.cuda = None
        self.cudagl = None
        self.ctx = None
        self.streams: List[Any] = []
        self.module = None
        self.kernel = None
        self.textures: List[int] = []
        self.pbos: List[int] = []
        self.resources: List[Any] = []
        self.d_bayer: List[Any] = []
        self.d_bayer_sizes: List[int] = []
        self._kernel_cache_digest = ""
        self.init_stage = "not_started"
        self.device_name = ""

    def _mark_stage(self, stage: str) -> None:
        self.init_stage = str(stage)
        print(f"[pygl_cuda_preview][cuda_init] stage={self.init_stage}", flush=True)

    def _find_nvcc(self) -> str:
        candidates = [
            str(os.environ.get("CUDA_NVCC", "") or "").strip(),
            shutil.which("nvcc") or "",
            "/usr/local/cuda/bin/nvcc",
            "/usr/bin/nvcc",
        ]
        for path in candidates:
            if path and os.path.exists(path) and os.access(path, os.X_OK):
                return path
        raise RuntimeError("nvcc not found; set CUDA_NVCC or install CUDA toolkit nvcc for CUDA preview kernel compilation")

    def _kernel_cache_paths(self) -> Tuple[Path, Path, Path]:
        cache_root = Path(os.environ.get("PYGL_CUDA_KERNEL_CACHE", "") or (Path.cwd() / "sync_ipc" / "cuda_kernel_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        key_src = "\n".join([
            "pygl_cuda_preview_kernel_v3_cxx17",
            CUDA_KERNEL,
            str(self.tile_w),
            str(self.tile_h),
        ]).encode("utf-8", "ignore")
        digest = hashlib.sha256(key_src).hexdigest()[:16]
        self._kernel_cache_digest = digest
        cu_path = cache_root / f"bayer_resize_to_rgba_{digest}.cu"
        ptx_path = cache_root / f"bayer_resize_to_rgba_{digest}.ptx"
        meta_path = cache_root / f"bayer_resize_to_rgba_{digest}.json"
        return cu_path, ptx_path, meta_path

    def _compile_kernel_with_nvcc(self, cuda: Any) -> Any:
        cu_path, ptx_path, meta_path = self._kernel_cache_paths()
        if ptx_path.exists():
            self._mark_stage("compile_kernel_load_cached_ptx")
            ptx = ptx_path.read_bytes()
            module = cuda.module_from_buffer(ptx)
            self._mark_stage("compile_kernel_get_function")
            return module, module.get_function("bayer_resize_to_rgba")

        self._mark_stage("compile_kernel_nvcc_prepare")
        nvcc = self._find_nvcc()
        if not ptx_path.exists():
            self._mark_stage("compile_kernel_nvcc_write_source")
            cu_path.write_text(CUDA_KERNEL, encoding="utf-8")
            cmd = [
                nvcc,
                "--ptx",
                "-O3",
                "-std=c++17",
                "-o",
                str(ptx_path),
                str(cu_path),
            ]
            self._mark_stage("compile_kernel_nvcc_run")
            proc = subprocess.run(cmd, cwd=str(cache_root), capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                raise RuntimeError(
                    "nvcc PTX compile failed rc="
                    f"{proc.returncode} cmd={' '.join(cmd)} stdout={proc.stdout[-2000:]} stderr={proc.stderr[-4000:]}"
                )
            try:
                meta_path.write_text(json.dumps({
                    "cmd": cmd,
                    "created_wall_us": int(time.time() * 1_000_000),
                    "stdout_tail": proc.stdout[-2000:],
                    "stderr_tail": proc.stderr[-4000:],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        self._mark_stage("compile_kernel_load_ptx")
        ptx = ptx_path.read_bytes()
        module = cuda.module_from_buffer(ptx)
        self._mark_stage("compile_kernel_get_function")
        return module, module.get_function("bayer_resize_to_rgba")

    def initialize(self) -> None:
        self._mark_stage("import_pycuda")
        try:
            import pycuda.driver as cuda
            import pycuda.gl as cudagl
        except Exception as exc:
            raise RuntimeError(f"PyCUDA with GL interop is required for CUDA preview: {type(exc).__name__}: {exc}") from exc
        try:
            self._mark_stage("cuda_init")
            cuda.init()
            self._mark_stage("select_device")
            dev = cuda.Device(self.cuda_device)
            self.device_name = str(dev.name())
            self._mark_stage("make_cuda_gl_context")
            self.ctx = cudagl.make_context(dev)
            self.cuda = cuda
            self.cudagl = cudagl
            self._mark_stage("compile_kernel")
            self.module, self.kernel = self._compile_kernel_with_nvcc(cuda)
            self._mark_stage("create_streams")
            self.streams = [cuda.Stream() for _ in range(self.cam_count)]

            self.textures = []
            self.pbos = []
            self.resources = []
            pbo_bytes = int(self.tile_w * self.tile_h * 4)
            for idx in range(self.cam_count):
                self._mark_stage(f"create_texture_pbo_{idx}")
                tex = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.tile_w, self.tile_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
                pbo = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
                GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, pbo_bytes, None, GL.GL_STREAM_DRAW)
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
                self._mark_stage(f"register_pbo_{idx}")
                try:
                    res = cudagl.RegisteredBuffer(int(pbo), cudagl.graphics_map_flags.WRITE_DISCARD)
                except AttributeError:
                    # Older pycuda.gl exposes flags through cuda.graphics_map_flags.
                    res = cudagl.RegisteredBuffer(int(pbo), cuda.graphics_map_flags.WRITE_DISCARD)
                self.textures.append(int(tex))
                self.pbos.append(int(pbo))
                self.resources.append(res)
                self.d_bayer.append(None)
                self.d_bayer_sizes.append(0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            self._mark_stage("ready")
            print(f"[pygl_cuda_preview] CUDA-GL backend ready: device={self.device_name} cams={self.cam_count} tile={self.tile_w}x{self.tile_h}", flush=True)
        except Exception:
            print(
                f"[pygl_cuda_preview][ERROR] CUDA-GL backend init failed at stage={self.init_stage}\n{traceback.format_exc()}",
                flush=True,
            )
            try:
                self.release()
            except Exception:
                pass
            raise

    @staticmethod
    def _pattern_code(name: str) -> int:
        return {"BG": 1, "RG": 2, "GB": 3, "GR": 4}.get(str(name or "BG").upper(), 1)

    def update_texture(self, cam_idx: int, raw: np.ndarray, meta: Dict[str, Any]) -> None:
        cuda = self.cuda
        if cuda is None or self.kernel is None:
            raise RuntimeError("CUDA backend not initialized")
        h = int(meta.get("height", raw.shape[0]))
        w = int(meta.get("width", raw.shape[1]))
        stride = int(meta.get("stride", raw.shape[1]))
        nbytes = int(h * stride)
        if self.d_bayer[cam_idx] is None or self.d_bayer_sizes[cam_idx] < nbytes:
            self.d_bayer[cam_idx] = cuda.mem_alloc(nbytes)
            self.d_bayer_sizes[cam_idx] = nbytes
        arr = np.asarray(raw)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        stream = self.streams[cam_idx]
        cuda.memcpy_htod_async(self.d_bayer[cam_idx], arr, stream)

        mapping = self.resources[cam_idx].map(stream)
        try:
            dptr, size = mapping.device_ptr_and_size()
            block = (16, 16, 1)
            grid = ((self.tile_w + block[0] - 1) // block[0], (self.tile_h + block[1] - 1) // block[1], 1)
            self.kernel(
                self.d_bayer[cam_idx], np.int32(w), np.int32(h), np.int32(stride),
                np.intp(dptr), np.int32(self.tile_w), np.int32(self.tile_h),
                np.int32(self._pattern_code(str(meta.get("bayer_pattern", "BG")))),
                block=block, grid=grid, stream=stream,
            )
        finally:
            mapping.unmap(stream)
        stream.synchronize()

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[cam_idx])
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbos[cam_idx])
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.tile_w, self.tile_h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def release(self) -> None:
        try:
            for res in self.resources:
                try:
                    res.unregister()
                except Exception:
                    pass
        finally:
            self.resources = []
        try:
            if self.pbos:
                GL.glDeleteBuffers(len(self.pbos), self.pbos)
        except Exception:
            pass
        try:
            if self.textures:
                GL.glDeleteTextures(len(self.textures), self.textures)
        except Exception:
            pass
        self.pbos = []
        self.textures = []
        if self.ctx is not None:
            try:
                self.ctx.pop()
                self.ctx.detach()
            except Exception:
                pass
            self.ctx = None


class CpuUploadFallbackBackend:
    def __init__(self, tile_w: int, tile_h: int, cam_count: int):
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.cam_count = int(cam_count)
        self.textures: List[int] = []

    def initialize(self) -> None:
        try:
            import cv2  # noqa
        except Exception as exc:
            raise RuntimeError(f"CPU fallback requires cv2: {exc}")
        self.textures = []
        for _ in range(self.cam_count):
            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.tile_w, self.tile_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            self.textures.append(int(tex))
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        print("[pygl_cuda_preview][WARN] using CPU upload fallback; CUDA-GL objective is NOT met", flush=True)

    def update_texture(self, cam_idx: int, raw: np.ndarray, meta: Dict[str, Any]) -> None:
        import cv2
        bayer = np.asarray(raw[:, :int(meta.get("width", raw.shape[1]))])
        pat = str(meta.get("bayer_pattern", "BG")).upper()
        code = {
            "BG": cv2.COLOR_BayerBG2RGB,
            "RG": cv2.COLOR_BayerRG2RGB,
            "GB": cv2.COLOR_BayerGB2RGB,
            "GR": cv2.COLOR_BayerGR2RGB,
        }.get(pat, cv2.COLOR_BayerBG2RGB)
        rgb = cv2.cvtColor(bayer, code)
        small = cv2.resize(rgb, (self.tile_w, self.tile_h), interpolation=cv2.INTER_AREA)
        rgba = np.empty((self.tile_h, self.tile_w, 4), dtype=np.uint8)
        rgba[:, :, :3] = small
        rgba[:, :, 3] = 255
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[cam_idx])
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.tile_w, self.tile_h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def release(self) -> None:
        pass


class PreviewWidget(QOpenGLWidget):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.cams = _parse_cams(args.cams)
        if not self.cams:
            raise RuntimeError("no cameras specified")
        self.rows, self.cols = _grid_layout(self.cams, args.layout)
        self.tile_w = int(args.tile_width)
        self.tile_h = int(args.tile_height)
        self.setWindowTitle("PyOpenGL CUDA Preview 40Hz video / 250Hz trajectory")
        self.resize(int(args.window_width), int(args.window_height))
        self.readers: Dict[str, RawPreviewFrameReader] = {}
        self.reader_errors: Dict[str, str] = {}
        self.last_video_update_ns = 0
        self.last_frame_ids = {c: -1 for c in self.cams}
        self.video_backend: Any = None
        self.video_backend_state = "not_started"
        self.video_backend_init_stage = "not_started"
        self.video_backend_error = ""
        self.video_backend_error_trace = ""
        self.video_backend_error_count = 0
        self.video_backend_attempts = 0
        self.video_backend_retry_count = 0
        self.video_timer_started = False
        self.render_timer_started = False
        self.overlay_timer_started = False
        self._runtime_timers_logged = False
        self._last_backend_retry_ms = 0.0
        self._backend_retry_interval_ms_default = 2500.0
        self._backend_retry_interval_ms_unrecoverable = 60000.0
        self._backend_retry_interval_ms = self._backend_retry_interval_ms_default
        self.last_paint_error = ""
        self.last_render_tick_error = ""
        self.last_video_tick_error = ""
        self.trajectory_reader: Optional[PreviewTrajectoryReader] = None
        self.trajectory_reader_error = ""
        self.trajectory_reader_reopen_count = 0
        self.trajectory_vbo = 0
        self.trajectory_vertex_count = 0
        self.trajectory_stale_guard_count = 0
        self.trajectory_future_drop_count = 0
        self.last_traj_update_ns = 0
        self.frame_counter = 0
        self.t0 = time.time()
        self.overlay_state = OverlayState(self.cams)
        self.overlay_ws_client: Optional[OverlayWsClient] = None
        self.hit_tailer: Optional[JsonlTailer] = None
        self.pseudo_hit_tailer: Optional[PseudoHitSuppressionTailer] = None
        self.trajectory_suppression_lock = threading.RLock()
        self.trajectory_suppressed_keys: Dict[Tuple[int, int], float] = {}
        self.trajectory_suppressed_total = 0
        self.trajectory_filter_segment_count = 0
        self.trajectory_filter_enabled = bool(getattr(args, "preview_trajectory_filter_human_noise", True))
        self.trajectory_filter_reason = ""
        self.human_tailer: Optional[HumanJsonlTailer] = None
        self.human_jsonl_backfill_count = 0
        self.human_jsonl_backfill_last_ms = 0.0
        self.human_jsonl_backfill_error = ""
        self.human_mask_render_stats: Dict[str, Any] = {
            "enabled": bool(getattr(args, "human_mask_draw_enable", True)),
            "mode": str(getattr(args, "human_mask_render_mode", "auto") or "auto"),
            "drawn": False,
            "mask_count": 0,
            "person_count": 0,
            "polygon_fill_count": 0,
            "bbox_fill_count": 0,
            "bbox_line_count": 0,
            "vertex_count": 0,
            "label_count": 0,
            "source": "none",
            "latest_age_ms": -1.0,
            "error": "",
        }
        self.manual_hit_targets: List[Dict[str, Any]] = []
        self.manual_hit_command_seq = 0
        self.manual_hit_last_result: Dict[str, Any] = {}
        self.preview_control_mtime_ns = 0
        self.preview_control_error = ""
        self.preview_hit_display_enable = bool(getattr(args, "preview_hit_display_enable", True))
        self.preview_trajectory_only_mode = bool(getattr(args, "preview_trajectory_only_mode", False))
        self.manual_hit_click_enable = bool(getattr(args, "manual_hit_click_enable", False))
        try:
            self.setMouseTracking(True)
        except Exception:
            pass
        self.last_status_ms = 0.0
        self.render_fps_current = 0.0
        self.renderer_draw_ms = 0.0
        self.renderer_paint_ms = 0.0
        self.renderer_paint_interval_ms = 0.0
        self.renderer_tick_ms = 0.0
        self.renderer_tick_interval_ms = 0.0
        self._last_paint_perf = 0.0
        self._last_tick_perf = 0.0
        self._render_fps_t0 = time.time()
        self._render_fps_frames = 0
        self.telemetry = PreviewTelemetry(args, self.cams, self.overlay_state)
        # Event overlay preview state is read by both paintGL() and the status/logging
        # path.  Keep these attributes present even when event overlay preview is
        # disabled, otherwise a status refresh can fail before paintGL lazily opens
        # the overlay shm readers.
        self.event_overlay_readers: Dict[str, EventOverlayPreviewReader] = {}
        self.event_overlay_textures: Dict[str, int] = {}
        self.event_overlay_stats: Dict[str, Dict[str, Any]] = {
            c: {
                "drawn": False,
                "stale": True,
                "age_ms": -1.0,
                "dt_mvs_ms": -1.0,
                "dt_human_ms": -1.0,
                "raw": 0,
                "filtered": 0,
                "mode": str(getattr(args, "event_overlay_sync_mode", "nearest_mvs")),
                "reason": "not_initialized",
            }
            for c in self.cams
        }
        self._event_overlay_overview_status_cache: Dict[str, Dict[str, Any]] = {}
        self._event_overlay_overview_status_last_ms = 0.0

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.PreciseTimer)
        self.render_timer.timeout.connect(self._render_tick)
        self.video_timer = QTimer(self)
        self.video_timer.setTimerType(Qt.PreciseTimer)
        self.video_timer.timeout.connect(self._video_tick)
        self.overlay_timer = QTimer(self)
        self.overlay_timer.setTimerType(Qt.PreciseTimer)
        self.overlay_timer.timeout.connect(self._overlay_tick)
        self.event_overlay_overview_window = None

    def set_event_overlay_overview_window(self, window: Optional[QWidget]) -> None:
        self.event_overlay_overview_window = window

    def set_event_overlay_overview_visible(self, visible: bool) -> None:
        win = getattr(self, "event_overlay_overview_window", None)
        if win is None:
            return
        try:
            if visible:
                win.show()
                win.showNormal()
                win.raise_()
                win.activateWindow()
            else:
                win.hide()
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] event overlay overview visibility failed: {type(exc).__name__}: {exc}", flush=True)

    def _preview_sync_path(self, value: str, default_name: str) -> Path:
        raw = str(value or default_name or "").strip()
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p
        normalized = raw.replace("\\", "/")
        if normalized.startswith("sync_ipc/"):
            return Path.cwd() / p
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / (p if raw else Path(default_name))

    def _preview_control_path(self) -> Path:
        return self._preview_sync_path(str(getattr(self.args, "preview_control_json", "") or ""), "preview_control.json")

    def _manual_hit_command_path(self) -> Path:
        return self._preview_sync_path(str(getattr(self.args, "manual_hit_command_json", "") or ""), "manual_hit_commands.jsonl")

    def _poll_preview_control_json(self, force: bool = False) -> None:
        path = self._preview_control_path()
        try:
            st = path.stat()
        except FileNotFoundError:
            return
        except Exception as exc:
            self.preview_control_error = f"{type(exc).__name__}: {exc}"
            return
        if not force and int(st.st_mtime_ns) <= int(self.preview_control_mtime_ns):
            return
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(obj, dict):
                return
            self.preview_control_mtime_ns = int(st.st_mtime_ns)
            if "human_label_mode" in obj:
                mode = str(obj.get("human_label_mode") or "detail").strip().lower()
                if mode in {"id_only", "detail", "off"}:
                    setattr(self.args, "human_label_mode", mode)
            if "preview_hit_display_enable" in obj:
                self.preview_hit_display_enable = _as_bool(obj.get("preview_hit_display_enable"), True)
                setattr(self.args, "hit_prompt_enable", bool(self.preview_hit_display_enable))
            if "preview_trajectory_only_mode" in obj:
                self.preview_trajectory_only_mode = _as_bool(obj.get("preview_trajectory_only_mode"), False)
            if "manual_hit_click_enable" in obj:
                self.manual_hit_click_enable = _as_bool(obj.get("manual_hit_click_enable"), False)
            if bool(getattr(self, "manual_hit_click_enable", False)) and bool(getattr(self, "preview_trajectory_only_mode", False)):
                self.preview_trajectory_only_mode = False
            self.preview_control_error = ""
        except Exception as exc:
            self.preview_control_error = f"{type(exc).__name__}: {exc}"

    def _append_preview_feedback(self, text: str, level: str = "info", ttl_ms: float = 1800.0) -> None:
        try:
            with self.overlay_state.lock:
                self.overlay_state._append_text_locked(str(text), cam="", ttl_ms=float(ttl_ms), level=str(level))
        except Exception:
            pass

    def _event_pos_xy(self, event: Any) -> Tuple[float, float]:
        try:
            pos = event.position()
            return float(pos.x()), float(pos.y())
        except Exception:
            pos = event.pos()
            return float(pos.x()), float(pos.y())

    def _find_manual_hit_target(self, px: float, py: float) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_score = float("inf")
        now_ms = _now_ms()
        for target in list(getattr(self, "manual_hit_targets", []) or []):
            if not isinstance(target, dict):
                continue
            age_ms = max(0.0, now_ms - _safe_float(target.get("cache_ts_ms"), now_ms))
            if age_ms > max(3000.0, float(getattr(self.args, "human_mask_life_ms", 5000.0))):
                continue
            label_rect = target.get("label_rect")
            if isinstance(label_rect, (list, tuple)) and len(label_rect) >= 4:
                x0, y0, x1, y1 = [_safe_float(v) for v in label_rect[:4]]
                if x0 <= px <= x1 and y0 <= py <= y1:
                    score = 0.0
                    if score < best_score:
                        best_score = score
                        best = target
                        continue
            bbox_rect = target.get("bbox_screen_rect")
            if not (isinstance(bbox_rect, (list, tuple)) and len(bbox_rect) >= 4):
                continue
            x0, y0, x1, y1 = [_safe_float(v) for v in bbox_rect[:4]]
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            dist = float(np.hypot(px - cx, py - cy))
            if x0 <= px <= x1 and y0 <= py <= y1:
                score = 100.0 + dist
            elif dist <= 28.0:
                score = 1000.0 + dist
            else:
                continue
            if score < best_score:
                best_score = score
                best = target
        return dict(best) if best is not None else None

    def _write_manual_hit_command(self, target: Dict[str, Any], click_xy: Tuple[float, float]) -> None:
        path = self._manual_hit_command_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.manual_hit_command_seq += 1
        wall_us = int(time.time() * 1_000_000)
        command_id = f"PREVIEW-CLICK-{wall_us}-{int(self.manual_hit_command_seq)}"
        cmd: Dict[str, Any] = {
            "schema_version": 1,
            "type": "preview_manual_hit_command",
            "source": "pyopengl_cuda_preview",
            "command_id": command_id,
            "wall_time_us": wall_us,
            "click_xy": [round(float(click_xy[0]), 2), round(float(click_xy[1]), 2)],
            "camera": str(target.get("camera") or ""),
            "stable_id": target.get("stable_id"),
            "person_id": target.get("person_id"),
            "confidence": _safe_float(target.get("confidence"), 0.0),
            "bbox_xyxy": target.get("bbox_xyxy"),
            "bbox_screen_rect": target.get("bbox_screen_rect"),
            "label": str(target.get("label") or ""),
            "foot_pixel": target.get("foot_pixel"),
            "local_xy": target.get("local_xy"),
            "field_xy": target.get("field_xy") or target.get("local_xy"),
            "physical_coord": target.get("physical_coord"),
            "target_display_player_id": str(target.get("display_player_id") or target.get("target_display_player_id") or ""),
            "target_display_slot_id": _safe_int(target.get("display_slot_id", target.get("target_display_slot_id")), -1),
            "target_source_global_id_int": _safe_gid_int(target.get("source_global_id", target.get("global_id")), -1),
            "target_source_global_id": str(target.get("source_global_id_label") or target.get("global_player_id") or ""),
            "target_global_id_int": _safe_gid_int(target.get("target_global_id_int", target.get("global_id")), -1),
            "target_id_semantics": str(target.get("target_id_semantics") or ""),
            "frame_id": target.get("frame_id"),
            "record_wall_us": target.get("record_wall_us"),
            "capture_wall_us": target.get("capture_wall_us"),
        }
        line = json.dumps(cmd, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8", buffering=1) as fp:
            fp.write(line + "\n")
        self.manual_hit_last_result = {
            "command_id": command_id,
            "camera": cmd.get("camera"),
            "stable_id": cmd.get("stable_id"),
            "label": cmd.get("label"),
            "path": str(path),
            "wall_time_us": wall_us,
        }
        print(f"[pygl_cuda_preview][manual_hit] command={command_id} target={cmd.get('label')} path={path}", flush=True)
        self._append_preview_feedback(f"manual hit command {cmd.get('label') or cmd.get('stable_id')} -> ID99", level="hit", ttl_ms=1800.0)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        left_button = getattr(Qt, "LeftButton", None)
        try:
            left_button = getattr(Qt.MouseButton, "LeftButton", left_button)
        except Exception:
            pass
        try:
            if left_button is not None and event.button() != left_button:
                return super().mousePressEvent(event)
        except Exception:
            pass
        self._poll_preview_control_json(force=True)
        if not bool(getattr(self, "manual_hit_click_enable", False)):
            self._append_preview_feedback("manual hit click disabled", level="info", ttl_ms=900.0)
            return super().mousePressEvent(event)
        px, py = self._event_pos_xy(event)
        target = self._find_manual_hit_target(px, py)
        if target is None:
            self._append_preview_feedback("manual hit target not found", level="warn", ttl_ms=1200.0)
            return super().mousePressEvent(event)
        try:
            self._write_manual_hit_command(target, (px, py))
            event.accept()
            return
        except Exception as exc:
            self.manual_hit_last_result = {"error": f"{type(exc).__name__}: {exc}", "wall_time_us": int(time.time() * 1_000_000)}
            print(f"[pygl_cuda_preview][manual_hit][WARN] click command failed: {type(exc).__name__}: {exc}", flush=True)
            self._append_preview_feedback(f"manual hit command failed: {type(exc).__name__}", level="warn", ttl_ms=1800.0)
            return super().mousePressEvent(event)

    def get_effective_event_overlay_stats(self) -> Dict[str, Dict[str, Any]]:
        fallback = self.event_overlay_stats if isinstance(getattr(self, "event_overlay_stats", None), dict) else {}
        now_ms = _now_ms()
        if now_ms - float(getattr(self, "_event_overlay_overview_status_last_ms", 0.0) or 0.0) < 250.0:
            cached = getattr(self, "_event_overlay_overview_status_cache", {})
            return cached if isinstance(cached, dict) and cached else fallback
        self._event_overlay_overview_status_last_ms = now_ms
        status_path = os.path.join(
            str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
            "event_overlay_overview_status.json",
        )
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            overview = payload.get("event_overlay", {}) if isinstance(payload, dict) else {}
            if not isinstance(overview, dict):
                return fallback
            merged: Dict[str, Dict[str, Any]] = {}
            for cam in self.cams:
                rec = overview.get(cam)
                if isinstance(rec, dict):
                    out = dict(rec)
                    out["status_source"] = "overview_process"
                    merged[cam] = out
                else:
                    merged[cam] = fallback.get(cam, {})
            self._event_overlay_overview_status_cache = merged
            return merged
        except Exception:
            return fallback

    def _record_video_backend_error(self, stage: str, exc: BaseException) -> None:
        stage_s = str(stage or "unknown")
        msg = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
        self.video_backend_state = "error"
        self.video_backend_init_stage = stage_s
        self.video_backend_error = f"{stage_s}: {msg}"
        self.video_backend_error_trace = trace[-6000:]
        self.video_backend_error_count += 1
        err_l = f"{stage_s} {msg}".lower()
        if "nvcc not found" in err_l:
            self._backend_retry_interval_ms = self._backend_retry_interval_ms_unrecoverable
        else:
            self._backend_retry_interval_ms = self._backend_retry_interval_ms_default
        print(f"[pygl_cuda_preview][ERROR] video backend failed stage={stage_s}: {msg}\n{trace}", flush=True)

    def _clear_video_backend_error(self, state: str) -> None:
        self.video_backend_state = str(state)
        self.video_backend_init_stage = "ready"
        self.video_backend_error = ""
        self.video_backend_error_trace = ""
        self._backend_retry_interval_ms = self._backend_retry_interval_ms_default

    def _try_initialize_video_backend(self, reason: str = "initial") -> bool:
        if self.video_backend is not None:
            return True
        if self.video_backend_state == "initializing":
            return False
        self.video_backend_attempts += 1
        self.video_backend_state = "initializing"
        self.video_backend_init_stage = f"{reason}:select_backend"
        backend: Any = None
        try:
            if self.args.allow_cpu_upload_fallback:
                try:
                    backend = CudaPboVideoBackend(self.args.cuda_device, self.tile_w, self.tile_h, len(self.cams))
                    self.video_backend_init_stage = "cuda_pbo.initialize"
                    backend.initialize()
                    self.video_backend = backend
                    self._clear_video_backend_error("cuda_pbo_ready")
                    return True
                except Exception as cuda_exc:
                    self._record_video_backend_error(getattr(backend, "init_stage", "cuda_pbo.initialize"), cuda_exc)
                    try:
                        if backend is not None:
                            backend.release()
                    except Exception:
                        pass
                    backend = CpuUploadFallbackBackend(self.tile_w, self.tile_h, len(self.cams))
                    self.video_backend_init_stage = "cpu_fallback.initialize"
                    backend.initialize()
                    self.video_backend = backend
                    self._clear_video_backend_error("cpu_fallback_ready")
                    return True

            backend = CudaPboVideoBackend(self.args.cuda_device, self.tile_w, self.tile_h, len(self.cams))
            self.video_backend_init_stage = "cuda_pbo.initialize"
            backend.initialize()
            self.video_backend = backend
            self._clear_video_backend_error("cuda_pbo_ready")
            return True
        except Exception as exc:
            try:
                if backend is not None:
                    backend.release()
            except Exception:
                pass
            self.video_backend = None
            self._record_video_backend_error(getattr(backend, "init_stage", self.video_backend_init_stage), exc)
            return False

    def _start_runtime_timers(self) -> None:
        if not self.video_timer_started:
            self.video_timer.start(max(1, int(round(1000.0 / max(1e-3, float(self.args.video_fps))))))
            self.video_timer_started = True
        if not self.render_timer_started:
            self.render_timer.start(max(1, int(round(1000.0 / max(1e-3, float(self.args.render_fps))))))
            self.render_timer_started = True
        if not self.overlay_timer_started:
            self.overlay_timer.start(max(10, int(round(float(self.args.overlay_poll_ms)))))
            self.overlay_timer_started = True
        if not self._runtime_timers_logged:
            print(
                f"[pygl_cuda_preview] started video_fps={self.args.video_fps} render_fps={self.args.render_fps} "
                f"trajectory_fps={self.args.trajectory_fps} tile={self.tile_w}x{self.tile_h} layout={self.rows}x{self.cols} "
                f"video_backend_state={self.video_backend_state}",
                flush=True,
            )
            self._runtime_timers_logged = True

    def _retry_video_backend_if_needed(self) -> None:
        if self.video_backend is not None:
            return
        now_ms = _now_ms()
        if now_ms - self._last_backend_retry_ms < self._backend_retry_interval_ms:
            return
        self._last_backend_retry_ms = now_ms
        self.video_backend_retry_count += 1
        try:
            self.makeCurrent()
            self._try_initialize_video_backend(reason=f"retry{self.video_backend_retry_count}")
        except Exception as exc:
            self._record_video_backend_error("retry_make_current", exc)
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass

    def _draw_video_backend_error_overlay(self, title: str = "Preview video backend is not ready") -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setFont(QFont("DejaVu Sans Mono", max(13, int(getattr(self.args, "status_font_px", 14)))))
            fm = painter.fontMetrics()
            pad = 14
            line_h = max(19, fm.height() + 3)
            trace_tail = str(self.video_backend_error_trace or "").strip().splitlines()[-6:]
            lines = [
                title,
                f"state={self.video_backend_state} stage={self.video_backend_init_stage}",
                f"attempts={self.video_backend_attempts} retries={self.video_backend_retry_count} errors={self.video_backend_error_count}",
                f"timers video={self.video_timer_started} render={self.render_timer_started} overlay={self.overlay_timer_started}",
                f"error={self.video_backend_error or '-'}",
                "MVS/raw frames are sampled by diagnostics; this panel means CUDA/PBO video draw path is not ready.",
            ]
            if self.last_paint_error:
                lines.append(f"paint_error={self.last_paint_error}")
            if trace_tail:
                lines.append("trace tail:")
                lines.extend(trace_tail)
            box_w = min(max(900, max(fm.horizontalAdvance(s) for s in lines) + pad * 2), max(320, self.width() - 32))
            box_h = min(line_h * len(lines) + pad * 2, max(120, self.height() - 32))
            x = 16
            y = 16
            painter.fillRect(x, y, int(box_w), int(box_h), QColor(0, 0, 0, 210))
            painter.setPen(QPen(QColor(255, 110, 90, 255)))
            cy = y + pad + line_h
            for idx, line in enumerate(lines):
                if cy > y + box_h - 6:
                    break
                if idx == 0:
                    painter.setPen(QPen(QColor(255, 170, 120, 255)))
                elif str(line).startswith("error=") or str(line).startswith("trace"):
                    painter.setPen(QPen(QColor(255, 210, 120, 255)))
                else:
                    painter.setPen(QPen(QColor(220, 245, 255, 245)))
                painter.drawText(x + pad, cy, fm.elidedText(str(line), Qt.ElideRight, max(40, int(box_w) - pad * 2)))
                cy += line_h
        finally:
            painter.end()

    def initializeGL(self) -> None:
        vendor = GL.glGetString(GL.GL_VENDOR)
        renderer = GL.glGetString(GL.GL_RENDERER)
        version = GL.glGetString(GL.GL_VERSION)
        print(f"[pygl_cuda_preview] GL_VENDOR={vendor!r} GL_RENDERER={renderer!r} GL_VERSION={version!r}", flush=True)
        if not bool(self.args.allow_non_nvidia_gl):
            text = (renderer or b"").decode("utf-8", errors="ignore").lower() + " " + (vendor or b"").decode("utf-8", errors="ignore").lower()
            if "nvidia" not in text:
                raise RuntimeError("CUDA-GL interop requires the OpenGL context on NVIDIA GPU; pass --allow-non-nvidia-gl only for debugging")

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.02, 0.02, 0.025, 1.0)

        for cam in self.cams:
            name = _shm_name_from_template(self.args.raw_template, cam)
            try:
                self.readers[cam] = RawPreviewFrameReader(name)
                print(f"[pygl_cuda_preview][{cam}] raw shm opened: {name}", flush=True)
            except Exception as exc:
                self.reader_errors[cam] = f"{type(exc).__name__}: {exc}"
                print(f"[pygl_cuda_preview][{cam}][WARN] raw shm not ready: {name}: {exc}", flush=True)

        self._try_initialize_video_backend(reason="initializeGL")

        try:
            self.trajectory_reader = PreviewTrajectoryReader(self.args.trajectory_shm)
            print(f"[pygl_cuda_preview] trajectory shm opened: {self.args.trajectory_shm}", flush=True)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] trajectory shm not ready: {self.args.trajectory_shm}: {exc}", flush=True)
            self.trajectory_reader = None

        if not bool(self.args.overlay_ws_disable):
            url = str(self.args.overlay_ws_url or "").strip()
            if url:
                self.overlay_ws_client = OverlayWsClient(url, self.overlay_state, enabled=True)
                self.overlay_ws_client.start()

        if bool(self.args.hit_jsonl_enable):
            hit_paths, sync_root_s = _resolve_hit_jsonl_paths(self.args, self.cams)
            self.hit_tailer = JsonlTailer(hit_paths, self.overlay_state)
            print(f"[pygl_cuda_preview] hit_candidate jsonl tailer paths={len(hit_paths)} root={sync_root_s}", flush=True)

        if bool(getattr(self.args, "preview_trajectory_filter_human_noise", True)):
            pseudo_paths, pseudo_root_s = _resolve_pseudo_hit_jsonl_paths(self.args, self.cams)
            self.pseudo_hit_tailer = PseudoHitSuppressionTailer(pseudo_paths, self)
            print(f"[pygl_cuda_preview] pseudo_hit suppression tailer paths={len(pseudo_paths)} root={pseudo_root_s}", flush=True)

        if bool(self.args.human_jsonl_enable):
            human_paths, human_roots_s = _resolve_human_jsonl_paths(self.args, self.cams)
            self.human_tailer = HumanJsonlTailer(human_paths, self.overlay_state, start_tail_bytes=int(self.args.human_jsonl_tail_bytes))
            print(f"[pygl_cuda_preview] human_result jsonl tailer paths={len(human_paths)} roots={human_roots_s}", flush=True)

        self.trajectory_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.trajectory_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 6 * 131072, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self._start_runtime_timers()

    def _try_open_missing_readers(self) -> None:
        for cam in self.cams:
            if cam in self.readers:
                continue
            name = _shm_name_from_template(self.args.raw_template, cam)
            try:
                self.readers[cam] = RawPreviewFrameReader(name)
                print(f"[pygl_cuda_preview][{cam}] raw shm opened late: {name}", flush=True)
            except Exception:
                pass
        if self.trajectory_reader is None:
            try:
                self.trajectory_reader = PreviewTrajectoryReader(self.args.trajectory_shm)
                self.trajectory_reader_error = ""
                self.trajectory_reader_reopen_count += 1
                print(f"[pygl_cuda_preview] trajectory shm opened late: {self.args.trajectory_shm}", flush=True)
            except Exception as exc:
                self.trajectory_reader_error = f"{type(exc).__name__}: {exc}"

    def _video_tick(self) -> None:
        self._try_open_missing_readers()
        if self.video_backend is None:
            self._retry_video_backend_if_needed()
            return
        try:
            self.makeCurrent()
        except Exception as exc:
            self._record_video_backend_error("video_tick_make_current", exc)
            return
        try:
            for idx, cam in enumerate(self.cams):
                rd = self.readers.get(cam)
                if rd is None:
                    continue
                try:
                    item = rd.read_latest_view(require_new=True)
                except Exception as exc:
                    print(f"[pygl_cuda_preview][{cam}][WARN] read raw failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if item is None:
                    continue
                if int(item.get("pixel_format", RAW_PIXEL_BAYER8)) != RAW_PIXEL_BAYER8:
                    print(f"[pygl_cuda_preview][{cam}][WARN] non-Bayer raw pixel_format={item.get('pixel_format')}; skip", flush=True)
                    continue
                try:
                    self.video_backend.update_texture(idx, item["frame"], item)
                    self.last_frame_ids[cam] = int(item.get("frame_id", -1))
                    self.last_video_tick_error = ""
                except Exception as exc:
                    self.last_video_tick_error = f"{cam}: {type(exc).__name__}: {exc}"
                    print(f"[pygl_cuda_preview][{cam}][ERROR] update texture failed: {self.last_video_tick_error}\n{traceback.format_exc()}", flush=True)
                    try:
                        if self.video_backend is not None:
                            self.video_backend.release()
                    except Exception:
                        pass
                    self.video_backend = None
                    self._record_video_backend_error(f"update_texture_{cam}", exc)
                    return
        finally:
            try:
                self.doneCurrent()
            except Exception:
                pass

    def mark_trajectory_suppressed_from_pseudo_hit(self, obj: Dict[str, Any]) -> None:
        cam = str(obj.get("camera_alias") or obj.get("camera") or obj.get("pair_id") or "").strip()
        if not cam:
            return
        try:
            cam_index = int(self.cams.index(cam))
            bullet_id = int(obj.get("bullet_id"))
        except Exception:
            return
        ttl_ms = max(1000.0, float(getattr(self.args, "trajectory_life_ms", 2500.0) or 2500.0) * 2.0)
        key = (int(cam_index), int(bullet_id))
        with self.trajectory_suppression_lock:
            if key not in self.trajectory_suppressed_keys:
                self.trajectory_suppressed_total += 1
            self.trajectory_suppressed_keys[key] = _now_ms() + ttl_ms
            if len(self.trajectory_suppressed_keys) > 4096:
                self._cleanup_trajectory_suppressed_keys_locked()

    def _cleanup_trajectory_suppressed_keys_locked(self) -> None:
        now_ms = _now_ms()
        for key, expire_ms in list(self.trajectory_suppressed_keys.items()):
            if float(expire_ms) <= now_ms:
                self.trajectory_suppressed_keys.pop(key, None)

    def _trajectory_suppressed_key_snapshot(self) -> set:
        with self.trajectory_suppression_lock:
            self._cleanup_trajectory_suppressed_keys_locked()
            return set(self.trajectory_suppressed_keys.keys())

    def _update_trajectory_vbo(self) -> None:
        if self.trajectory_reader is None or self.trajectory_vbo == 0:
            return
        try:
            segs = self.trajectory_reader.read_segments(max_age_ms=float(self.args.trajectory_life_ms), require_new=False)
            self.trajectory_reader_error = ""
        except Exception as exc:
            self.trajectory_reader_error = f"{type(exc).__name__}: {exc}"
            print(f"[pygl_cuda_preview][WARN] trajectory shm read failed; will reopen: {self.trajectory_reader_error}", flush=True)
            try:
                close = getattr(self.trajectory_reader, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            self.trajectory_reader = None
            self.trajectory_vertex_count = 0
            return
        if segs is None or len(segs) == 0:
            self.trajectory_vertex_count = 0
            setattr(self.args, '_trajectory_stale_guard_count', int(self.trajectory_stale_guard_count))
            setattr(self.args, '_trajectory_future_drop_count', int(self.trajectory_future_drop_count))
            return
        rows, cols = self.rows, self.cols
        verts: List[Tuple[float, float, float, float, float, float]] = []
        now_ms = time.time() * 1000.0
        max_age_ms = max(1.0, float(self.args.trajectory_life_ms))
        future_tolerance_ms = min(1000.0, max(250.0, max_age_ms))
        suppressed_keys = self._trajectory_suppressed_key_snapshot() if bool(getattr(self.args, "preview_trajectory_filter_human_noise", True)) else set()
        filtered_segments = 0
        for rec in segs[-int(self.args.max_trajectory_segments):]:
            ci = int(rec["cam_index"])
            if ci < 0 or ci >= len(self.cams):
                continue
            if suppressed_keys:
                try:
                    if (ci, int(rec["bullet_id"])) in suppressed_keys:
                        filtered_segments += 1
                        continue
                except Exception:
                    pass
            col = ci % cols
            row = ci // cols
            tile_x0 = -1.0 + 2.0 * (col / cols)
            tile_y0 =  1.0 - 2.0 * ((row + 1) / rows)
            tile_w_ndc = 2.0 / cols
            tile_h_ndc = 2.0 / rows
            dw = max(float(rec["display_w"]), 1.0)
            dh = max(float(rec["display_h"]), 1.0)
            x0 = tile_x0 + (float(rec["x0"]) / dw) * tile_w_ndc
            x1 = tile_x0 + (float(rec["x1"]) / dw) * tile_w_ndc
            y0 = tile_y0 + (1.0 - float(rec["y0"]) / dh) * tile_h_ndc
            y1 = tile_y0 + (1.0 - float(rec["y1"]) / dh) * tile_h_ndc
            raw_age = now_ms - float(rec["ts_ms"])
            if raw_age < -future_tolerance_ms:
                self.trajectory_future_drop_count += 1
                continue
            life_ms = max(1.0, float(rec["life_ms"]))
            if raw_age > max(life_ms, max_age_ms):
                self.trajectory_stale_guard_count += 1
                continue
            age = max(0.0, raw_age)
            alpha = float(rec["a"]) * max(0.0, min(1.0, 1.0 - age / life_ms))
            if alpha <= 0.01:
                continue
            color = (float(rec["r"]), float(rec["g"]), float(rec["b"]), alpha)
            verts.append((x0, y0, color[0], color[1], color[2], color[3]))
            verts.append((x1, y1, color[0], color[1], color[2], color[3]))
        if not verts:
            self.trajectory_vertex_count = 0
            self.trajectory_filter_segment_count = int(filtered_segments)
            setattr(self.args, '_trajectory_stale_guard_count', int(self.trajectory_stale_guard_count))
            setattr(self.args, '_trajectory_future_drop_count', int(self.trajectory_future_drop_count))
            return
        arr = np.asarray(verts, dtype=np.float32)
        self.trajectory_vertex_count = int(arr.shape[0])
        self.trajectory_filter_segment_count = int(filtered_segments)
        setattr(self.args, '_trajectory_stale_guard_count', int(self.trajectory_stale_guard_count))
        setattr(self.args, '_trajectory_future_drop_count', int(self.trajectory_future_drop_count))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.trajectory_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, int(arr.nbytes), arr, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _overlay_tick(self) -> None:
        self._poll_preview_control_json()
        if self.hit_tailer is not None:
            self.hit_tailer.poll(max_lines_per_file=256)
        if self.pseudo_hit_tailer is not None:
            self.pseudo_hit_tailer.poll(max_lines_per_file=512)
        if self.human_tailer is not None:
            self.human_tailer.poll(max_lines_per_file=96)
        if bool(getattr(self.args, "human_jsonl_enable", True)):
            now_ms = _now_ms()
            interval_ms = max(100.0, float(getattr(self.args, "human_jsonl_backfill_interval_ms", 500.0) or 500.0))
            if now_ms - float(self.human_jsonl_backfill_last_ms) >= interval_ms:
                self.human_jsonl_backfill_last_ms = now_ms
                try:
                    records = _read_latest_human_records_for_status(self.args, self.cams, tail_bytes=int(getattr(self.args, "human_jsonl_tail_bytes", 2_000_000)))
                    updated = 0
                    for rec in records.values():
                        if isinstance(rec, dict):
                            self.overlay_state.update_from_human_msg(rec, source="human_jsonl_backfill")
                            updated += 1
                    if updated > 0:
                        self.human_jsonl_backfill_count += int(updated)
                    self.human_jsonl_backfill_error = ""
                except Exception as exc:
                    self.human_jsonl_backfill_error = f"{type(exc).__name__}: {exc}"

    def _render_tick(self) -> None:
        # Trajectory redraw target is 250 FPS.  Data update is small and independent
        # of video texture updates.  This timer timing is tracked separately from
        # paintGL because Qt/compositor/swap can throttle paint frequency.
        now_perf = time.perf_counter()
        if self._last_tick_perf > 0.0:
            interval_ms = (now_perf - self._last_tick_perf) * 1000.0
            self.renderer_tick_interval_ms = 0.85 * float(self.renderer_tick_interval_ms) + 0.15 * interval_ms if self.renderer_tick_interval_ms > 0 else interval_ms
        self._last_tick_perf = now_perf
        t0 = time.perf_counter()
        try:
            self._update_trajectory_vbo()
            self.last_render_tick_error = ""
        except Exception as exc:
            self.last_render_tick_error = f"{type(exc).__name__}: {exc}"
            print(f"[pygl_cuda_preview][WARN] render tick failed: {self.last_render_tick_error}\n{traceback.format_exc()}", flush=True)
        self.update()
        tick_ms = (time.perf_counter() - t0) * 1000.0
        self.renderer_tick_ms = 0.85 * float(self.renderer_tick_ms) + 0.15 * tick_ms if self.renderer_tick_ms > 0 else tick_ms

    def paintGL(self) -> None:
        now_perf = time.perf_counter()
        if self._last_paint_perf > 0.0:
            interval_ms = (now_perf - self._last_paint_perf) * 1000.0
            self.renderer_paint_interval_ms = 0.85 * float(self.renderer_paint_interval_ms) + 0.15 * interval_ms if self.renderer_paint_interval_ms > 0 else interval_ms
        self._last_paint_perf = now_perf
        t_draw0 = time.perf_counter()
        try:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            if self.video_backend is None:
                self._draw_video_backend_error_overlay()
            else:
                self._draw_video_grid()
            trajectory_only = bool(getattr(self, "preview_trajectory_only_mode", False))
            if not trajectory_only:
                self._draw_human_masks()
            else:
                self.manual_hit_targets = []
            # Event camera overview is shown in an independent window.  Keep the
            # historical inline path available only for explicit fallback/debug use.
            if bool(getattr(self.args, "event_overlay_main_inline_enable", False)) and not trajectory_only:
                self._draw_event_overlay_preview()
            self._draw_trajectory()
            if not trajectory_only:
                self._draw_overlay_markers()
                self._draw_text_overlay()
            self.last_paint_error = ""
        except Exception as exc:
            self.last_paint_error = f"{type(exc).__name__}: {exc}"
            print(f"[pygl_cuda_preview][ERROR] paintGL failed: {self.last_paint_error}\n{traceback.format_exc()}", flush=True)
            try:
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                self._draw_video_backend_error_overlay(title="Preview paintGL failed")
            except Exception:
                pass
        finally:
            self.frame_counter += 1
            self._render_fps_frames += 1
            now_s = time.time()
            fps_dt = max(1e-6, now_s - self._render_fps_t0)
            if fps_dt >= 1.0:
                self.render_fps_current = float(self._render_fps_frames) / fps_dt
                self._render_fps_frames = 0
                self._render_fps_t0 = now_s
            paint_ms = (time.perf_counter() - t_draw0) * 1000.0
            self.renderer_paint_ms = 0.85 * float(self.renderer_paint_ms) + 0.15 * paint_ms if self.renderer_paint_ms > 0 else paint_ms
            # renderer_draw_ms is kept for backward compatibility with the status UI;
            # it measures paintGL body time and intentionally excludes Qt buffer swap.
            self.renderer_draw_ms = self.renderer_paint_ms
            if self.frame_counter % max(1, int(float(self.args.render_fps) * 5.0)) == 0:
                dt = max(1e-6, time.time() - self.t0)
                print(
                    f"[pygl_cuda_preview] render_fps={self.frame_counter/dt:.1f} "
                    f"paint_ms={self.renderer_paint_ms:.2f} paint_interval_ms={self.renderer_paint_interval_ms:.2f} "
                    f"tick_ms={self.renderer_tick_ms:.2f} tick_interval_ms={self.renderer_tick_interval_ms:.2f} "
                    f"video_backend_state={self.video_backend_state} last_frame_ids={self.last_frame_ids}",
                    flush=True,
                )

    def _try_open_event_overlay_readers(self) -> None:
        if not bool(getattr(self.args, "event_overlay_preview_enable", False)):
            return
        for cam in self.cams:
            if cam in self.event_overlay_readers:
                continue
            try:
                rd = EventOverlayPreviewReader(
                    cam,
                    str(getattr(self.args, "event_overlay_preview_template", "event_overlay_latest_{camera}")),
                    str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(self.args, "event_overlay_sidecar_template", "event_overlay_{camera}_latest.json")),
                )
                self.event_overlay_readers[cam] = rd
                tex = GL.glGenTextures(1)
                self.event_overlay_textures[cam] = int(tex)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, rd.width, rd.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                print(f"[pygl_cuda_preview][{cam}] event overlay opened: {rd.path} sidecar={rd.sidecar_path}", flush=True)
            except Exception as exc:
                self.event_overlay_stats.setdefault(cam, {})["open_error"] = f"{type(exc).__name__}: {exc}"

    def _bgr_to_rgba_overlay(self, frame: np.ndarray, alpha: float) -> np.ndarray:
        if frame.ndim == 2:
            b = g = r = frame
        else:
            if frame.shape[2] >= 3:
                b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            else:
                b = g = r = frame[:, :, 0]
        rgba = np.empty((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0] = r
        rgba[:, :, 1] = g
        rgba[:, :, 2] = b
        # Use luminance as alpha so black background does not wash out the RGB video.
        lum = np.maximum(np.maximum(r, g), b).astype(np.float32)
        rgba[:, :, 3] = np.clip(lum * max(0.0, min(1.0, float(alpha))), 0, 255).astype(np.uint8)
        return rgba

    def _event_overlay_decision(self, cam: str, meta: Dict[str, Any], item: Dict[str, Any]) -> Tuple[bool, bool, float, float, float, str]:
        mode = str(getattr(self.args, "event_overlay_sync_mode", "nearest_mvs") or "nearest_mvs").lower()
        now_ms = _now_ms()
        published_us = _safe_float(meta.get("published_wall_us", meta.get("event_wall_us", 0)), 0.0)
        age_ms = max(0.0, now_ms - published_us / 1000.0) if published_us > 1_000_000_000_000_000 else max(0.0, now_ms - os.path.getmtime(self.event_overlay_readers[cam].sidecar_path) * 1000.0) if cam in self.event_overlay_readers and os.path.exists(self.event_overlay_readers[cam].sidecar_path) else -1.0
        max_age = float(getattr(self.args, "event_overlay_preview_max_age_ms", 100.0))
        max_delta = float(getattr(self.args, "event_overlay_max_sync_delta_ms", 50.0))
        dt_mvs = _safe_float(meta.get("event_to_mvs_delta_ms"), float('nan'))
        dt_human = _safe_float(meta.get("event_to_human_capture_delta_ms"), float('nan'))
        reason = ""
        draw = True
        stale = False
        if age_ms >= 0 and age_ms > max_age:
            stale = True
            reason = f"age>{max_age:.0f}ms"
        if mode == "latest":
            pass
        elif mode == "nearest_mvs":
            if np.isfinite(dt_mvs):
                if abs(float(dt_mvs)) > max_delta:
                    stale = True; reason = f"dt_mvs>{max_delta:.0f}ms"
            else:
                # Fallback: compare packet mvs_frame_id with current tile frame_id if present.
                mvfid = _safe_int(meta.get("mvs_frame_id", -1), -1)
                curfid = _safe_int(self.last_frame_ids.get(cam, -1), -1)
                if mvfid >= 0 and curfid >= 0 and abs(mvfid - curfid) > 2:
                    stale = True; reason = "mvs_frame_delta>2"
        elif mode == "nearest_human":
            if np.isfinite(dt_human):
                if abs(float(dt_human)) > max_delta:
                    stale = True; reason = f"dt_human>{max_delta:.0f}ms"
            else:
                hf = _safe_int(meta.get("human_frame_id", meta.get("human_mvs_frame_id", -1)), -1)
                try:
                    hrec = self.overlay_state.snapshot().get("humans", {}).get(cam, {})
                    curhf = _safe_int(hrec.get("frame_id", -1), -1)
                    if hf >= 0 and curhf >= 0 and abs(hf - curhf) > 2:
                        stale = True; reason = "human_frame_delta>2"
                except Exception:
                    pass
        else:
            mode = "latest"
        if bool(getattr(self.args, "event_overlay_hide_stale", False)) and stale:
            draw = False
        return bool(draw), bool(stale), float(age_ms), float(dt_mvs) if np.isfinite(dt_mvs) else -1.0, float(dt_human) if np.isfinite(dt_human) else -1.0, reason

    def _draw_event_overlay_preview(self) -> None:
        if not bool(getattr(self.args, "event_overlay_preview_enable", False)):
            return
        self._try_open_event_overlay_readers()
        if not self.event_overlay_readers:
            return
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        rows, cols = self.rows, self.cols
        base_alpha = float(getattr(self.args, "event_overlay_preview_alpha", 0.55))
        stale_alpha_scale = float(getattr(self.args, "event_overlay_stale_alpha_scale", 0.25))
        for i, cam in enumerate(self.cams):
            rd = self.event_overlay_readers.get(cam)
            tex = self.event_overlay_textures.get(cam)
            if rd is None or tex is None:
                continue
            try:
                item = rd.read_latest()
            except Exception as exc:
                self.event_overlay_stats.setdefault(cam, {})["read_error"] = f"{type(exc).__name__}: {exc}"
                continue
            if item is None:
                continue
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            draw, stale, age_ms, dt_mvs, dt_human, reason = self._event_overlay_decision(cam, meta, item)
            raw_n = _safe_int(meta.get("raw_event_count", 0), 0)
            filt_n = _safe_int(meta.get("filtered_event_count", 0), 0)
            mask_valid = bool(meta.get("mask_valid", False))
            self.event_overlay_stats[cam] = {
                "drawn": bool(draw), "stale": bool(stale), "age_ms": float(age_ms), "dt_mvs_ms": float(dt_mvs), "dt_human_ms": float(dt_human),
                "raw": int(raw_n), "filtered": int(filt_n), "mask_valid": bool(mask_valid), "source_stage": str(meta.get("source_stage", "")),
                "seq": _safe_int(meta.get("seq", item.get("frame_id", -1)), -1), "reason": str(reason or ""), "mode": str(getattr(self.args, "event_overlay_sync_mode", "nearest_mvs")),
            }
            if not draw:
                continue
            alpha = base_alpha * (stale_alpha_scale if stale else 1.0)
            if item.get("new_frame", False) or not getattr(rd, "_uploaded_once", False):
                rgba = self._bgr_to_rgba_overlay(item["frame"], alpha=1.0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, rgba.shape[1], rgba.shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                setattr(rd, "_uploaded_once", True)
            col = i % cols
            row = i // cols
            x0 = -1.0 + 2.0 * (col / cols)
            x1 = -1.0 + 2.0 * ((col + 1) / cols)
            y0 = 1.0 - 2.0 * ((row + 1) / rows)
            y1 = 1.0 - 2.0 * (row / rows)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
            GL.glColor4f(1.0, 1.0, 1.0, float(alpha))
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(x0, y0)
            GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(x1, y0)
            GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(x1, y1)
            GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(x0, y1)
            GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

    def _draw_video_grid(self) -> None:
        if self.video_backend is None:
            return
        GL.glEnable(GL.GL_TEXTURE_2D)
        rows, cols = self.rows, self.cols
        for i, cam in enumerate(self.cams):
            if i >= len(self.video_backend.textures):
                continue
            col = i % cols
            row = i // cols
            x0 = -1.0 + 2.0 * (col / cols)
            x1 = -1.0 + 2.0 * ((col + 1) / cols)
            y0 = 1.0 - 2.0 * ((row + 1) / rows)
            y1 = 1.0 - 2.0 * (row / rows)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.video_backend.textures[i])
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(x0, y0)
            GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(x1, y0)
            GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(x1, y1)
            GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(x0, y1)
            GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

    def _draw_trajectory(self) -> None:
        if self.trajectory_vbo == 0 or self.trajectory_vertex_count <= 0:
            return
        GL.glLineWidth(float(self.args.trajectory_line_width))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.trajectory_vbo)
        stride = 6 * 4
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glVertexPointer(2, GL.GL_FLOAT, stride, ctypes.c_void_p(0))
        GL.glColorPointer(4, GL.GL_FLOAT, stride, ctypes.c_void_p(2 * 4))
        GL.glDrawArrays(GL.GL_LINES, 0, int(self.trajectory_vertex_count))
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _tile_ndc_for_cam(self, cam: str, x: float, y: float, dw: float, dh: float) -> Optional[Tuple[float, float]]:
        try:
            ci = self.cams.index(str(cam))
        except ValueError:
            return None
        col = ci % self.cols
        row = ci // self.cols
        tile_x0 = -1.0 + 2.0 * (col / self.cols)
        tile_y0 =  1.0 - 2.0 * ((row + 1) / self.rows)
        tile_w_ndc = 2.0 / self.cols
        tile_h_ndc = 2.0 / self.rows
        xx = tile_x0 + (float(x) / max(float(dw), 1.0)) * tile_w_ndc
        yy = tile_y0 + (1.0 - float(y) / max(float(dh), 1.0)) * tile_h_ndc
        return float(xx), float(yy)

    def _tile_px_for_cam(self, cam: str, x: float, y: float, dw: float, dh: float) -> Optional[Tuple[float, float]]:
        try:
            ci = self.cams.index(str(cam))
        except ValueError:
            return None
        col = ci % self.cols
        row = ci // self.cols
        tile_w_px = max(1.0, float(self.width()) / max(1, self.cols))
        tile_h_px = max(1.0, float(self.height()) / max(1, self.rows))
        px = col * tile_w_px + (float(x) / max(float(dw), 1.0)) * tile_w_px
        py = row * tile_h_px + (float(y) / max(float(dh), 1.0)) * tile_h_px
        return float(px), float(py)

    def _draw_circle_ndc(self, x: float, y: float, radius_ndc: float, color: Tuple[float, float, float, float], segments: int = 32) -> None:
        GL.glColor4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        GL.glBegin(GL.GL_LINE_LOOP)
        for k in range(int(segments)):
            a = 2.0 * np.pi * float(k) / float(segments)
            GL.glVertex2f(float(x + np.cos(a) * radius_ndc), float(y + np.sin(a) * radius_ndc))
        GL.glEnd()
        # Crosshair improves visibility on bright backgrounds.
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f(float(x - radius_ndc * 1.4), float(y)); GL.glVertex2f(float(x + radius_ndc * 1.4), float(y))
        GL.glVertex2f(float(x), float(y - radius_ndc * 1.4)); GL.glVertex2f(float(x), float(y + radius_ndc * 1.4))
        GL.glEnd()

    def _draw_human_masks(self) -> None:
        self.manual_hit_targets = []
        mode = str(getattr(self.args, "human_mask_render_mode", "auto") or "auto").strip().lower()
        if mode not in ("off", "auto", "polygon", "bbox"):
            mode = "auto"
        stats: Dict[str, Any] = {
            "enabled": bool(self.args.human_mask_draw_enable) and mode != "off",
            "mode": mode,
            "label_mode": str(getattr(self.args, "human_label_mode", "detail") or "detail"),
            "drawn": False,
            "mask_count": 0,
            "person_count": 0,
            "polygon_fill_count": 0,
            "bbox_fill_count": 0,
            "bbox_line_count": 0,
            "vertex_count": 0,
            "label_count": 0,
            "source": "none",
            "latest_age_ms": -1.0,
            "error": "",
        }
        if bool(getattr(self, "preview_trajectory_only_mode", False)):
            stats["enabled"] = False
            stats["source"] = "trajectory_only_mode"
            self.human_mask_render_stats = stats
            return
        if not stats["enabled"]:
            self.human_mask_render_stats = stats
            return
        snap = self.overlay_state.snapshot(marker_ttl_ms=float(self.args.overlay_marker_life_ms), text_ttl_ms=float(self.args.overlay_text_life_ms))
        humans = snap.get("humans", {})
        if not humans:
            self.human_mask_render_stats = stats
            return
        now = _now_ms()
        labels: List[Tuple[int, int, int, str]] = []
        hit_targets: List[Dict[str, Any]] = []
        mask_alpha = max(0.0, min(1.0, float(self.args.human_mask_alpha) / 255.0))
        line_h = int(self.args.status_font_px) + 4

        GL.glEnable(GL.GL_BLEND)
        for cam, rec in humans.items():
            rec_age_ms = now - _safe_float(rec.get("ts_ms", 0.0), 0.0)
            if rec_age_ms > float(self.args.human_mask_life_ms):
                continue
            stats["latest_age_ms"] = max(float(stats.get("latest_age_ms", -1.0)), float(rec_age_ms))
            fw = _safe_float(rec.get("frame_w", 1), 1.0)
            fh = _safe_float(rec.get("frame_h", 1), 1.0)
            persons = rec.get("persons", [])
            if not isinstance(persons, list):
                continue
            for person in persons[:int(self.args.max_human_draw)]:
                if not isinstance(person, dict):
                    continue
                stats["person_count"] = int(stats.get("person_count", 0)) + 1
                conf = _safe_float(person.get("confidence", 0.0), 0.0)
                if conf < float(self.args.human_min_conf):
                    continue
                drew_poly = False
                polygons = person.get("mask_polygons", [])
                if mode in ("auto", "polygon") and isinstance(polygons, list) and bool(self.args.human_mask_fill_enable):
                    GL.glLineWidth(float(self.args.human_mask_line_width))
                    for poly in polygons[:int(self.args.max_human_polygons_per_person)]:
                        if not isinstance(poly, list) or len(poly) < 3:
                            continue
                        pts: List[Tuple[float, float]] = []
                        for pt in poly[:int(self.args.max_human_polygon_points)]:
                            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                                continue
                            xy = self._tile_ndc_for_cam(str(cam), _safe_float(pt[0]), _safe_float(pt[1]), fw, fh)
                            if xy is not None:
                                pts.append((float(xy[0]), float(xy[1])))
                        if len(pts) < 3:
                            continue
                        GL.glColor4f(0.0, 210.0 / 255.0, 70.0 / 255.0, mask_alpha)
                        GL.glBegin(GL.GL_TRIANGLE_FAN)
                        for x_ndc, y_ndc in pts:
                            GL.glVertex2f(x_ndc, y_ndc)
                        GL.glEnd()
                        GL.glColor4f(0.0, 1.0, 120.0 / 255.0, 0.90)
                        GL.glBegin(GL.GL_LINE_LOOP)
                        for x_ndc, y_ndc in pts:
                            GL.glVertex2f(x_ndc, y_ndc)
                        GL.glEnd()
                        drew_poly = True
                        stats["drawn"] = True
                        stats["mask_count"] = int(stats.get("mask_count", 0)) + 1
                        stats["polygon_fill_count"] = int(stats.get("polygon_fill_count", 0)) + 1
                        stats["vertex_count"] = int(stats.get("vertex_count", 0)) + len(pts)
                bbox = person.get("bbox_xyxy", None)
                if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                    continue
                p0_ndc = self._tile_ndc_for_cam(str(cam), _safe_float(bbox[0]), _safe_float(bbox[1]), fw, fh)
                p1_ndc = self._tile_ndc_for_cam(str(cam), _safe_float(bbox[2]), _safe_float(bbox[3]), fw, fh)
                p0_px = self._tile_px_for_cam(str(cam), _safe_float(bbox[0]), _safe_float(bbox[1]), fw, fh)
                p1_px = self._tile_px_for_cam(str(cam), _safe_float(bbox[2]), _safe_float(bbox[3]), fw, fh)
                if p0_ndc is None or p1_ndc is None or p0_px is None or p1_px is None:
                    continue
                bx0, by0 = float(min(p0_px[0], p1_px[0])), float(min(p0_px[1], p1_px[1]))
                bx1, by1 = float(max(p0_px[0], p1_px[0])), float(max(p0_px[1], p1_px[1]))
                x0, y0 = p0_ndc
                x1, y1 = p1_ndc
                x_min, x_max = float(min(x0, x1)), float(max(x0, x1))
                y_min, y_max = float(min(y0, y1)), float(max(y0, y1))
                draw_bbox_fill = bool(self.args.human_mask_fill_enable) and (
                    mode == "bbox" or (mode == "auto" and not drew_poly)
                )
                if draw_bbox_fill:
                    GL.glColor4f(0.0, 210.0 / 255.0, 70.0 / 255.0, max(0.06, mask_alpha * 0.72))
                    GL.glBegin(GL.GL_QUADS)
                    GL.glVertex2f(x_min, y_min)
                    GL.glVertex2f(x_max, y_min)
                    GL.glVertex2f(x_max, y_max)
                    GL.glVertex2f(x_min, y_max)
                    GL.glEnd()
                    stats["drawn"] = True
                    stats["mask_count"] = int(stats.get("mask_count", 0)) + 1
                    stats["bbox_fill_count"] = int(stats.get("bbox_fill_count", 0)) + 1
                    stats["vertex_count"] = int(stats.get("vertex_count", 0)) + 4
                GL.glLineWidth(float(self.args.human_bbox_line_width if drew_poly else self.args.human_mask_line_width))
                if drew_poly:
                    GL.glColor4f(1.0, 230.0 / 255.0, 60.0 / 255.0, 0.90)
                else:
                    GL.glColor4f(0.0, 1.0, 120.0 / 255.0, 0.90)
                GL.glBegin(GL.GL_LINE_LOOP)
                GL.glVertex2f(x_min, y_min)
                GL.glVertex2f(x_max, y_min)
                GL.glVertex2f(x_max, y_max)
                GL.glVertex2f(x_min, y_max)
                GL.glEnd()
                stats["bbox_line_count"] = int(stats.get("bbox_line_count", 0)) + 1

                sid = person.get("stable_id", person.get("person_id", ""))
                display_player_id = str(person.get("display_player_id") or person.get("global_player_id") or person.get("source_global_player_id") or "").strip()
                display_slot_id = _safe_int(person.get("display_slot_id", -1), -1)
                source_global_id = _safe_gid_int(person.get("source_global_id", person.get("global_id", -1)), -1)
                target_global_id = _safe_gid_int(person.get("global_id", source_global_id), source_global_id)
                label_global_id = source_global_id if source_global_id >= 0 else target_global_id
                label_global_text = f"G{label_global_id}" if label_global_id >= 0 else ""
                foot = person.get("foot_pixel", None)
                local_xy = person.get("local_xy", None)
                foot_txt = ""
                if isinstance(foot, (list, tuple)) and len(foot) >= 2:
                    foot_txt = f" px=({_safe_float(foot[0]):.0f},{_safe_float(foot[1]):.0f})"
                    if bool(getattr(self.args, "human_coord_point_enable", True)):
                        pfoot = self._tile_ndc_for_cam(str(cam), _safe_float(foot[0]), _safe_float(foot[1]), fw, fh)
                        if pfoot is not None:
                            GL.glPointSize(7.0)
                            GL.glColor4f(1.0, 80.0 / 255.0, 20.0 / 255.0, 0.95)
                            GL.glBegin(GL.GL_POINTS)
                            GL.glVertex2f(float(pfoot[0]), float(pfoot[1]))
                            GL.glEnd()
                            self._draw_circle_ndc(float(pfoot[0]), float(pfoot[1]), 0.006, (1.0, 1.0, 1.0, 0.9), segments=18)
                local_txt = ""
                if isinstance(local_xy, (list, tuple)) and len(local_xy) >= 2:
                    local_txt = f" xy=({_safe_float(local_xy[0]):.2f},{_safe_float(local_xy[1]):.2f})"
                elif isinstance(local_xy, dict):
                    lx = local_xy.get("x", local_xy.get("X", None))
                    ly = local_xy.get("y", local_xy.get("Y", None))
                    if lx is not None and ly is not None:
                        local_txt = f" xy=({_safe_float(lx):.2f},{_safe_float(ly):.2f})"
                label_mode = str(getattr(self.args, "human_label_mode", "detail") or "detail").strip().lower()
                if label_mode not in {"id_only", "detail", "off"}:
                    label_mode = "detail"
                txt = ""
                label_parts: List[str] = []
                if display_player_id:
                    label_parts.append(display_player_id)
                if label_global_text and label_global_text.upper() not in {str(p).upper() for p in label_parts}:
                    label_parts.append(label_global_text)
                label_id = " ".join(label_parts) if label_parts else str(sid)
                if label_mode == "id_only":
                    txt = f"id={label_id}"
                elif label_mode == "detail":
                    txt = f"id={label_id} conf={conf:.2f}{foot_txt}{local_txt}"
                px0, py0 = p0_px
                px1, py1 = p1_px
                tx = int(min(px0, px1))
                ty = max(14, int(min(py0, py1)) - 3)
                box_w = min(max(220, 8 + len(txt) * max(7, int(self.args.status_font_px) - 3)), max(220, self.width() - tx - 8))
                label_rect: List[float] = []
                if txt:
                    label_rect = [float(tx), float(ty - line_h + 2), float(tx + int(box_w)), float(ty + 5)]
                    labels.append((tx, ty, int(box_w), txt))
                target_semantics = "global_runtime_id" if target_global_id >= 0 else ""
                if display_slot_id >= 0:
                    target_semantics = "global_bev_display_id"
                hit_targets.append({
                    "camera": str(cam),
                    "stable_id": sid,
                    "person_id": person.get("person_id", ""),
                    "confidence": float(conf),
                    "bbox_xyxy": list(bbox[:4]) if isinstance(bbox, (list, tuple)) else [],
                    "bbox_screen_rect": [round(bx0, 2), round(by0, 2), round(bx1, 2), round(by1, 2)],
                    "label_rect": label_rect,
                    "label": txt or f"id={label_id}",
                    "display_player_id": display_player_id,
                    "display_slot_id": display_slot_id,
                    "global_id": target_global_id,
                    "global_player_id": str(person.get("global_player_id") or ""),
                    "source_global_id": source_global_id,
                    "source_global_id_label": label_global_text or str(person.get("source_global_player_id") or person.get("global_player_id") or ""),
                    "target_global_id_int": display_slot_id if display_slot_id >= 0 else target_global_id,
                    "target_id_semantics": target_semantics,
                    "foot_pixel": foot,
                    "local_xy": local_xy,
                    "field_xy": person.get("field_xy", local_xy),
                    "physical_coord": person.get("physical_coord", None),
                    "frame_id": rec.get("frame_id"),
                    "record_wall_us": rec.get("record_wall_us"),
                    "capture_wall_us": rec.get("capture_wall_us"),
                    "cache_ts_ms": now,
                })

        polygon_count = int(stats.get("polygon_fill_count", 0))
        bbox_count = int(stats.get("bbox_fill_count", 0))
        if polygon_count > 0 and bbox_count > 0:
            stats["source"] = "mixed"
        elif polygon_count > 0:
            stats["source"] = "polygon"
        elif bbox_count > 0:
            stats["source"] = "bbox_fallback"
        elif int(stats.get("bbox_line_count", 0)) > 0:
            stats["source"] = "bbox_line_only"
        stats["label_count"] = len(labels)
        self.human_mask_render_stats = stats
        self.manual_hit_targets = hit_targets[-256:]
        if labels:
            painter = QPainter(self)
            try:
                painter.setRenderHint(QPainter.TextAntialiasing, True)
                painter.setFont(QFont("DejaVu Sans Mono", max(10, int(self.args.status_font_px) - 1)))
                text_pen = QPen(QColor(235, 255, 235, 245))
                bg = QColor(0, 0, 0, 135)
                for tx, ty, box_w, txt in labels:
                    painter.fillRect(tx, ty - line_h + 2, int(box_w), line_h + 3, bg)
                    painter.setPen(text_pen)
                    painter.drawText(tx + 4, ty, painter.fontMetrics().elidedText(txt, Qt.ElideRight, max(20, int(box_w) - 8)))
            finally:
                painter.end()

    def _draw_overlay_markers(self) -> None:
        snap = self.overlay_state.snapshot(marker_ttl_ms=float(self.args.overlay_marker_life_ms), text_ttl_ms=float(self.args.overlay_text_life_ms))
        markers = snap.get("markers", [])
        if not markers:
            return
        w = max(float(self.width()), 1.0)
        h = max(float(self.height()), 1.0)
        # Approximate pixel radius in NDC, adjusted by the larger dimension.
        GL.glLineWidth(float(self.args.overlay_marker_line_width))
        for m in markers[-int(self.args.max_overlay_markers):]:
            if not bool(getattr(self, "preview_hit_display_enable", True)) and str(m.get("kind", "")).lower() == "hit":
                continue
            xy = self._tile_ndc_for_cam(m.get("cam", ""), _safe_float(m.get("x")), _safe_float(m.get("y")), _safe_float(m.get("display_w", 1), 1), _safe_float(m.get("display_h", 1), 1))
            if xy is None:
                continue
            r_px = _safe_float(m.get("radius_px", 10.0), 10.0)
            r_ndc = max(0.002, r_px * 2.0 / max(w, h))
            color = tuple(m.get("color", _marker_color(str(m.get("kind", "event")))))
            self._draw_circle_ndc(xy[0], xy[1], r_ndc, color)  # type: ignore[arg-type]


    def _fmt_value(self, v: Any, digits: int = 1, missing: str = "-") -> str:
        if v is None:
            return missing
        if isinstance(v, bool):
            return "Y" if v else "N"
        try:
            fv = float(v)
            if fv <= -999998 or abs(fv - 999999.0) < 1e-6:
                return missing
            if abs(fv - int(fv)) < 1e-9 and digits <= 0:
                return str(int(fv))
            return f"{fv:.{digits}f}"
        except Exception:
            s = str(v)
            return s if s else missing

    def build_status_tabs(self) -> Dict[str, List[Tuple[str, List[Tuple[str, str]]]]]:
        """Return diagnostics grouped for the separate status window.

        This is deliberately read-only: it samples existing shm/jsonl/log state
        and does not touch the video, trajectory, human-mask, or hit rendering
        paths.
        """
        try:
            setattr(self.args, '_trajectory_drawn_vtx', int(self.trajectory_vertex_count))
        except Exception:
            pass
        snap = self.overlay_state.snapshot(marker_ttl_ms=float(self.args.overlay_marker_life_ms), text_ttl_ms=float(self.args.overlay_text_life_ms))
        try:
            telem = self.telemetry.poll(self.last_frame_ids)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] build_status_tabs telemetry poll failed: {type(exc).__name__}: {exc}", flush=True)
            telem = getattr(self.telemetry, 'last_snapshot', {}) or {}
        tb = telem.get('timebase', {})
        trig = telem.get('trigger', {})
        ev = telem.get('event', {})
        yolo = telem.get('yolo', {})
        hit = telem.get('hit', {})
        bullet = dict(telem.get('bullet', {}) or {})
        # Direct trajectory shm header sample for status only.  Header layout is
        # defined in preview_gpu_shm.py: count=5, seq=6, update_mono_ns=7.
        try:
            if self.trajectory_reader is not None and getattr(self.trajectory_reader, 'header', None) is not None:
                bullet['trajectory_shm_segment_count'] = int(self.trajectory_reader.header[5])
                bullet['trajectory_shm_seq'] = int(self.trajectory_reader.header[6])
                upd = int(self.trajectory_reader.header[7])
                bullet['trajectory_latest_age_ms'] = max(0.0, (time.monotonic_ns() - upd) / 1e6) if upd > 0 else -1.0
        except Exception:
            pass
        sync = telem.get('sync_health', {})
        humans = snap.get('humans', {}) if isinstance(snap, dict) else {}
        human_status_source = "tailer"
        if not isinstance(humans, dict) or len(humans) == 0 or _safe_int(snap.get("human_jsonl_rx_count", 0), 0) <= 0:
            fallback_records = _read_latest_human_records_for_status(self.args, self.cams)
            if fallback_records:
                humans = {cam: _human_record_to_status_hrec(rec) for cam, rec in fallback_records.items()}
                human_status_source = "latest_jsonl_fallback"
        ws_state = "ON" if snap.get("ws_connected") else "OFF"
        tabs: Dict[str, List[Tuple[str, List[Tuple[str, str]]]]] = {}

        fps = max(0.0, float(self.render_fps_current))
        frame_period_ms = (1000.0 / fps) if fps > 0.001 else -1.0
        target_period_ms = 1000.0 / max(1e-3, float(self.args.render_fps))
        unmeasured_ms = max(0.0, frame_period_ms - float(self.renderer_paint_ms)) if frame_period_ms > 0 else -1.0

        tabs["总览"] = [
            ("全局时基", [
                ("system_wall_time_us", str(_safe_int(tb.get('system_wall_time_us', 0), 0))),
                ("system_mono_ns", str(_safe_int(tb.get('system_mono_ns', 0), 0))),
                ("sync_start_wall_us", str(_safe_int(tb.get('sync_start_wall_us', 0), 0))),
                ("sync_start_mono_ns", str(_safe_int(tb.get('sync_start_mono_ns', 0), 0))),
                ("event_timebase_valid", self._fmt_value(tb.get('event_timebase_valid'))),
                ("event_ts_to_wall_offset_us", str(_safe_int(tb.get('event_ts_to_wall_offset_us', 0), 0))),
                ("event_last_wall_age_ms", self._fmt_value(tb.get('event_last_wall_age_ms'), 1)),
                ("mvs_timebase_valid", self._fmt_value(tb.get('mvs_timebase_valid'))),
                ("mvs_last_wall_age_ms", self._fmt_value(tb.get('mvs_last_wall_age_ms'), 1)),
                ("clock_drift_event_vs_mvs_ms", self._fmt_value(tb.get('clock_drift_event_vs_mvs_ms'), 1)),
                ("clock_drift_event_vs_wall_ms", self._fmt_value(tb.get('clock_drift_event_vs_wall_ms'), 1)),
            ]),
            ("触发汇总", [
                ("trigger_state", str(trig.get('state', 'unknown'))),
                ("trigger_fps", self._fmt_value(trig.get('fps'), 2)),
                ("trigger_period_ms", self._fmt_value(trig.get('period_ms', 1000.0 / max(1e-3, _safe_float(trig.get('fps'), 0.0))) if _safe_float(trig.get('fps'), 0.0) > 0 else -1, 2)),
                ("trigger_jitter_p50_ms", self._fmt_value(trig.get('jitter_p50_ms'), 2)),
                ("trigger_jitter_p95_ms", self._fmt_value(trig.get('jitter_p95_ms'), 2)),
                ("trigger_jitter_max_ms", self._fmt_value(trig.get('jitter_max_ms'), 2)),
                ("trigger_last_age_ms", self._fmt_value(trig.get('last_age_ms'), 0)),
                ("ok_cams", f"{_safe_int(trig.get('ok_cams', 0), 0)}/{len(self.cams)}"),
            ]),
            ("链路汇总", [
                ("mvs_active_count", f"{_safe_int(telem.get('mvs_active_count',0),0)}/{len(self.cams)}"),
                ("event_active_count", f"{_safe_int(ev.get('event_active_count',0),0)}/{len(self.cams)}"),
                ("overlay_ws_connected", ws_state),
                ("overlay_ws_rx_count", str(_safe_int(snap.get('ws_rx_count', 0), 0))),
                ("overlay_ws_age_ms", self._fmt_value(snap.get('ws_age_ms', -1), 0)),
                ("human_jsonl_rx_count", str(_safe_int(snap.get('human_jsonl_rx_count', 0), 0))),
                ("human_status_source", human_status_source),
                ("hit_jsonl_rx_count", str(_safe_int(snap.get('hit_jsonl_rx_count', 0), 0))),
            ]),
            ("人体物理坐标", [
                ("projector_ok_cams", f"{getattr(self, '_physical_projector_ok_count', 0)}/{len(self.cams)}"),
                ("note", "per-camera xy in 每路相机链路 -> 每路物理坐标"),
            ]),
            ("YOLO汇总", [
                ("yolo_worker_count", str(_safe_int(yolo.get('worker_count', 0), 0))),
                ("yolo_published_fps_total", self._fmt_value(yolo.get('published_fps_total', yolo.get('infer_fps')), 1)),
                ("yolo_inferred_fps_total", self._fmt_value(yolo.get('inferred_fps_total'), 1)),
                ("yolo_async_record_postprocess_enabled", self._fmt_value(yolo.get('async_record_postprocess_enabled'))),
                ("yolo_postprocess_queue_depth", self._fmt_value(yolo.get('postprocess_queue_depth'), 0)),
                ("yolo_postprocess_dropped_batches", self._fmt_value(yolo.get('postprocess_dropped_batches'), 0)),
                ("yolo_polygon_contract_ok", self._fmt_value(yolo.get('polygon_contract_ok'))),
                ("yolo_batch_size", str(_safe_int(yolo.get('batch_size', 0), 0))),
                ("yolo_predict_ms", self._fmt_value(yolo.get('predict_ms'), 1)),
                ("yolo_capture_to_result_ms", self._fmt_value(yolo.get('capture_to_result_ms'), 1)),
            ]),
            ("命中汇总", [
                ("hit_udp_recv_per_s", self._fmt_value(hit.get('recv_per_s'), 1)),
                ("hit_eval_per_s", self._fmt_value(hit.get('eval_per_s'), 1)),
                ("hit_candidate_per_s", self._fmt_value(hit.get('candidate_per_s'), 1)),
                ("hit_confirmed_per_s", self._fmt_value(hit.get('confirmed_per_s'), 1)),
                ("hit_no_human_per_s", self._fmt_value(hit.get('no_human_per_s'), 1)),
                ("hit_no_human_count", str(_safe_int(hit.get('no_human_count', 0), 0))),
                ("hit_pending_count", str(_safe_int(hit.get('pending_count', 0), 0))),
                ("hit_pending_impacts", str(_safe_int(hit.get('pending_impacts', 0), 0))),
            ]),
            ("渲染时序", [
                ("video_backend_state", str(getattr(self, "video_backend_state", "unknown"))),
                ("video_backend_stage", str(getattr(self, "video_backend_init_stage", ""))),
                ("video_backend_error", str(getattr(self, "video_backend_error", ""))[:260] or "-"),
                ("video_backend_attempts", str(_safe_int(getattr(self, "video_backend_attempts", 0), 0))),
                ("video_backend_retries", str(_safe_int(getattr(self, "video_backend_retry_count", 0), 0))),
                ("timers_started", f"video={bool(getattr(self, 'video_timer_started', False))} render={bool(getattr(self, 'render_timer_started', False))} overlay={bool(getattr(self, 'overlay_timer_started', False))}"),
                ("last_video_tick_error", str(getattr(self, "last_video_tick_error", ""))[:180] or "-"),
                ("last_render_tick_error", str(getattr(self, "last_render_tick_error", ""))[:180] or "-"),
                ("last_paint_error", str(getattr(self, "last_paint_error", ""))[:180] or "-"),
                ("renderer_paint_ms", self._fmt_value(self.renderer_paint_ms, 2)),
                ("render_fps", self._fmt_value(fps, 1)),
                ("frame_period_ms", self._fmt_value(frame_period_ms, 2)),
                ("target_period_ms", self._fmt_value(target_period_ms, 2)),
                ("renderer_tick_ms", self._fmt_value(self.renderer_tick_ms, 2)),
                ("tick_interval_ms", self._fmt_value(self.renderer_tick_interval_ms, 2)),
                ("paint_interval_ms", self._fmt_value(self.renderer_paint_interval_ms, 2)),
                ("unmeasured_qt_swap_event_ms", self._fmt_value(unmeasured_ms, 2)),
                ("trajectory_drawn_vtx", str(int(self.trajectory_vertex_count))),
            ]),
        ]

        per_rows: List[Tuple[str, str]] = []
        mvs_fps = telem.get('mvs_fps', {})
        raw_fps = telem.get('raw_fps', {})
        raw_age = telem.get('raw_age_ms', {})
        frame_age = telem.get('frame_age_ms', {})
        raw_missing = telem.get('raw_missing', {})
        mvs_missing = telem.get('mvs_missing', {})
        pending_q = telem.get('pending_q', {})
        human_age = telem.get('human_age_by_cam', {})
        deltas = yolo.get('result_video_delta_frames', {}) if isinstance(yolo, dict) else {}
        for cam in self.cams:
            hrec = humans.get(cam, {}) if isinstance(humans, dict) else {}
            per_rows.append((cam, (
                f"mvs={self._fmt_value(mvs_fps.get(cam),1)} raw={self._fmt_value(raw_fps.get(cam),1)} "
                f"raw_age={self._fmt_value(raw_age.get(cam),0)}ms frame_age={self._fmt_value(frame_age.get(cam),0)}ms "
                f"raw_missing={'Y' if raw_missing.get(cam, True) else 'N'} mvs_missing={'Y' if mvs_missing.get(cam, True) else 'N'} "
                f"pending_q={self._fmt_value(pending_q.get(cam),0)} human_age={self._fmt_value(human_age.get(cam),0)}ms "
                f"delta_frames={self._fmt_value(deltas.get(cam),0)} persons={_safe_int(hrec.get('person_count', 0), 0)}"
            )))
        trigger_rows: List[Tuple[str, str]] = []
        for cam in self.cams:
            trigger_rows.append((cam, (
                f"sent={trig.get('per_cam_sent',{}).get(cam,'-')} ack={trig.get('per_cam_ack',{}).get(cam,'-')} "
                f"fail={trig.get('per_cam_fail',{}).get(cam,'-')} miss={trig.get('per_cam_trigger_miss_count',{}).get(cam,'-')} "
                f"trig_to_mvs_ms={self._fmt_value(trig.get('per_cam_trigger_to_mvs_frame_ms',{}).get(cam),1)} "
                f"trig_to_event_age_ms={self._fmt_value(trig.get('per_cam_trigger_to_event_trigger_ms',{}).get(cam),1)}"
            )))
        yolo_rows: List[Tuple[str, str]] = []
        for cam in self.cams:
            hrec = humans.get(cam, {}) if isinstance(humans, dict) else {}
            yolo_rows.append((cam, (
                f"human_frame={hrec.get('frame_id','-')} persons={_safe_int(hrec.get('person_count',0),0)} "
                f"capture_wall_us={_safe_int(hrec.get('capture_wall_us',0),0)} record_wall_us={_safe_int(hrec.get('record_wall_us',0),0)} "
                f"capture_to_result_ms={self._fmt_value(hrec.get('yolo_capture_to_result_ms', -1),1)} "
                f"result_video_delta_frames={self._fmt_value(deltas.get(cam),0)}"
            )))
        physical_rows: List[Tuple[str, str]] = []
        projector_ok = 0
        for cam in self.cams:
            hrec = humans.get(cam, {}) if isinstance(humans, dict) else {}
            proj = hrec.get('projector', {}) if isinstance(hrec.get('projector', {}), dict) else {}
            persons = hrec.get('persons', []) if isinstance(hrec.get('persons', []), list) else []
            valid = bool(proj.get('valid', False)) if proj else any(isinstance(p, dict) and (p.get('physical_coord') or p.get('local_xy')) for p in persons)
            if valid:
                projector_ok += 1
            parts = [f"proj={'OK' if valid else 'BAD'}", f"frame={hrec.get('frame_id','-')}"]
            if proj:
                parts.append(f"coord={proj.get('coord_frame','local_ground')}/{proj.get('unit','m')}")
                if not valid and proj.get('last_error'):
                    parts.append(f"reason={proj.get('last_error')}")
            shown = 0
            for person in persons:
                if not isinstance(person, dict):
                    continue
                pc = person.get('physical_coord') if isinstance(person.get('physical_coord'), dict) else {}
                lx = ly = None
                if pc and pc.get('ok', False):
                    lx, ly = pc.get('x_m'), pc.get('y_m')
                elif isinstance(person.get('local_xy'), (list, tuple)) and len(person.get('local_xy')) >= 2:
                    lx, ly = person.get('local_xy')[0], person.get('local_xy')[1]
                if lx is None or ly is None:
                    continue
                sid = person.get('stable_id', person.get('person_id',''))
                foot = person.get('foot_pixel')
                foot_s = ''
                if isinstance(foot, (list, tuple)) and len(foot) >= 2:
                    foot_s = f" foot=({_safe_float(foot[0]):.0f},{_safe_float(foot[1]):.0f})"
                parts.append(f"id={sid} xy=({_safe_float(lx):.2f},{_safe_float(ly):.2f})m{foot_s}")
                shown += 1
                if shown >= 3:
                    break
            if shown == 0:
                parts.append('persons_xy=-')
            physical_rows.append((cam, ' '.join(parts)))
        try:
            self._physical_projector_ok_count = int(projector_ok)
        except Exception:
            pass
        tabs["每路相机链路"] = [
            ("每路实时链路", per_rows),
            ("每路触发关联", trigger_rows),
            ("每路YOLO/人体结果", yolo_rows),
            ("每路物理坐标", physical_rows),
        ]

        event_rows: List[Tuple[str, str]] = []
        effective_event_overlay_stats = self.get_effective_event_overlay_stats()
        for cam in self.cams:
            event_rows.append((cam, (
                f"ready={'Y' if ev.get('per_cam_ready',{}).get(cam,False) else 'N'} "
                f"event_trigger_fps={self._fmt_value(ev.get('per_cam_event_trigger_fps',{}).get(cam),1)} "
                f"event_rate_mevs={self._fmt_value(ev.get('per_cam_event_rate_mevs',{}).get(cam),2)} "
                f"loop_fps={self._fmt_value(ev.get('per_cam_worker_loop_fps',{}).get(cam),1)} "
                f"empty={self._fmt_value(ev.get('per_cam_worker_empty_ratio',{}).get(cam),2)} "
                f"fast={ev.get('per_cam_worker_fastpath_count',{}).get(cam,'-')} "
                f"slice_age_ms={self._fmt_value(ev.get('per_cam_event_slice_age_ms',{}).get(cam),1)} "
                f"last_ts_us={_safe_int(ev.get('per_cam_event_last_ts_us',{}).get(cam,0),0)} "
                f"trigger_age_ms={self._fmt_value(ev.get('per_cam_event_trigger_last_age_ms',{}).get(cam),0)} "
                f"triggers={_safe_int(ev.get('per_cam_event_trigger_recv_count',{}).get(cam,0),0)} "
                f"gaps={_safe_int(ev.get('per_cam_event_trigger_gap_count',{}).get(cam,0),0)} "
                f"nonmono={_safe_int(ev.get('per_cam_event_nonmonotonic_count',{}).get(cam,0),0)} "
                f"guard={ev.get('per_cam_event_guard_state',{}).get(cam,'OK')} "
                f"viz_stage={ev.get('per_cam_viz_source_stage',{}).get(cam,'')} "
                f"viz_raw={ev.get('per_cam_viz_raw_count',{}).get(cam,'-')} "
                f"viz_filt={ev.get('per_cam_viz_filtered_count',{}).get(cam,'-')} "
                f"viz_red={self._fmt_value(100.0 * _safe_float(ev.get('per_cam_viz_reduction_ratio',{}).get(cam,0.0),0.0),1)}% "
                f"viz_mask={'Y' if ev.get('per_cam_viz_mask_valid',{}).get(cam,False) else 'N'} "
                f"viz_buf={ev.get('per_cam_viz_buffer_size',{}).get(cam,'-')} "
                f"viz_drop={ev.get('per_cam_viz_buffer_drop_count',{}).get(cam,'-')} "
                f"evt_ovl_draw={'Y' if effective_event_overlay_stats.get(cam,{}).get('drawn', False) else 'N'} "
                f"evt_ovl_stale={'Y' if effective_event_overlay_stats.get(cam,{}).get('stale', True) else 'N'} "
                f"evt_ovl_mode={effective_event_overlay_stats.get(cam,{}).get('mode', getattr(self.args,'event_overlay_sync_mode','nearest_mvs'))} "
                f"evt_ovl_age={self._fmt_value(effective_event_overlay_stats.get(cam,{}).get('age_ms'),0)}ms "
                f"evt_ovl_dt_mvs={self._fmt_value(effective_event_overlay_stats.get(cam,{}).get('dt_mvs_ms'),1)} "
                f"evt_ovl_dt_human={self._fmt_value(effective_event_overlay_stats.get(cam,{}).get('dt_human_ms'),1)} "
                f"evt_ovl_raw={effective_event_overlay_stats.get(cam,{}).get('raw','-')} "
                f"evt_ovl_filt={effective_event_overlay_stats.get(cam,{}).get('filtered','-')}"
            )))
        bullet_rows = [
            ("bullet_udp_recv_per_s", self._fmt_value(bullet.get('udp_recv_per_s'),1)),
            ("bullet_track_count_active", str(_safe_int(bullet.get('track_count_active',0),0))),
            ("bullet_latest_age_ms", self._fmt_value(bullet.get('latest_age_ms'),1)),
            ("bullet_latest_cam", str(bullet.get('latest_cam',''))),
            ("bullet_latest_id", str(_safe_int(bullet.get('latest_id',0),0))),
            ("bullet_latest_point_count", str(_safe_int(bullet.get('latest_point_count',0),0))),
            ("trajectory_shm_seq", str(_safe_int(bullet.get('trajectory_shm_seq',0),0))),
            ("trajectory_shm_segment_count", str(_safe_int(bullet.get('trajectory_shm_segment_count',0),0))),
            ("trajectory_visible_segment_count", str(_safe_int(bullet.get('trajectory_visible_segment_count',0),0))),
            ("trajectory_latest_age_ms", self._fmt_value(bullet.get('trajectory_latest_age_ms'),1)),
            ("trajectory_drawn_vtx", str(int(self.trajectory_vertex_count))),
        ]
        recent_bullet_rows: List[Tuple[str, str]] = []
        for pt in list(bullet.get('recent', []))[-12:]:
            if not isinstance(pt, dict):
                continue
            recent_bullet_rows.append((f"{pt.get('cam','')} b{pt.get('bullet_id',0)} p{pt.get('point_index',-1)}", f"age={self._fmt_value(_now_ms()-_safe_float(pt.get('ts_ms',0),0),0)}ms event_ts={pt.get('point_ts_us',0)} xy=({_safe_float(pt.get('x'),0):.0f},{_safe_float(pt.get('y'),0):.0f}) dt={self._fmt_value(pt.get('point_delta_ms'),1)}ms state={pt.get('status','')}"))
        tabs["事件/弹道"] = [
            ("事件相机健康", event_rows),
            ("弹道/轨迹汇总", bullet_rows),
            ("最近弹道点", recent_bullet_rows or [("recent", "-")]),
        ]

        latest_sync_rows = [
            ("latest_cam", str(sync.get('cam', sync.get('latest_cam', '')))),
            ("bullet_id", str(_safe_int(sync.get('bullet_id', 0), 0))),
            ("event_ts_us", str(_safe_int(sync.get('event_ts_us', 0), 0))),
            ("event_to_mvs_delta_ms", self._fmt_value(sync.get('event_to_mvs_delta_ms'), 2)),
            ("event_to_human_capture_delta_ms", self._fmt_value(sync.get('event_to_human_capture_delta_ms'), 2)),
            ("event_to_human_result_age_ms", self._fmt_value(sync.get('event_to_human_result_age_ms'), 1)),
            ("event_to_human_delta_frames", self._fmt_value(sync.get('event_to_human_delta_frames'), 0)),
            ("mask_valid_at_event", self._fmt_value(sync.get('mask_valid_at_event', False))),
            ("person_count_at_event", str(_safe_int(sync.get('person_count_at_event', 0), 0))),
            ("matched_person_id", str(sync.get('matched_person_id', ''))),
            ("matched_stable_id", str(sync.get('matched_stable_id', ''))),
            ("hit_inside_mask", self._fmt_value(sync.get('hit_inside_mask', False))),
            ("hit_distance_to_mask_px", self._fmt_value(sync.get('hit_distance_to_mask_px'), 1)),
            ("hit_distance_to_bbox_px", self._fmt_value(sync.get('hit_distance_to_bbox_px'), 1)),
            ("sync_quality", str(sync.get('sync_quality', 'UNKNOWN'))),
            ("latest_hit_reason", str(sync.get('latest_hit_reason', ''))),
        ]
        thresholds = [
            ("OK", "abs(event_to_human_capture_delta_ms) <= 25ms"),
            ("WARN", "<= 75ms"),
            ("BAD", "> 75ms or > 3 frames"),
            ("STALE", "human_result_age_ms > 500ms"),
            ("MISSING", "no usable human_result for bullet camera"),
        ]
        recent_sync_rows: List[Tuple[str, str]] = []
        for rec in list(sync.get('recent', []))[-16:]:
            if not isinstance(rec, dict):
                continue
            recent_sync_rows.append((f"{rec.get('cam','')} b{rec.get('bullet_id',0)} p{rec.get('point_index','')}", f"quality={rec.get('sync_quality','')} e2mvs={self._fmt_value(rec.get('event_to_mvs_delta_ms'),1)}ms e2human={self._fmt_value(rec.get('event_to_human_capture_delta_ms'),1)}ms frames={rec.get('event_to_human_delta_frames','-')} human_age={self._fmt_value(rec.get('event_to_human_result_age_ms'),0)}ms persons={rec.get('person_count_at_event',0)} mask_valid={rec.get('mask_valid_at_event',False)}"))
        tabs["弹道-人体同步"] = [
            ("最新弹道-人体同步", latest_sync_rows),
            ("质量阈值", thresholds),
            ("最近同步样本", recent_sync_rows or [("recent", "-")]),
        ]

        hit_reason_rows = [
            ("hit_udp_recv_per_s", self._fmt_value(hit.get('recv_per_s'),1)),
            ("hit_eval_per_s", self._fmt_value(hit.get('eval_per_s'),1)),
            ("hit_candidate_per_s", self._fmt_value(hit.get('candidate_per_s'),1)),
            ("hit_confirmed_per_s", self._fmt_value(hit.get('confirmed_per_s'),1)),
            ("hit_no_human_per_s", self._fmt_value(hit.get('no_human_per_s'),1)),
            ("hit_no_human_count", str(_safe_int(hit.get('no_human_count',0),0))),
            ("hit_no_mask_per_s", self._fmt_value(hit.get('no_mask_per_s'),1)),
            ("hit_stale_human_per_s", self._fmt_value(hit.get('stale_human_per_s'),1)),
            ("hit_sync_bad_per_s", self._fmt_value(hit.get('sync_bad_per_s'),1)),
            ("hit_outside_mask_per_s", self._fmt_value(hit.get('outside_mask_per_s'),1)),
            ("hit_no_cam_per_s", self._fmt_value(hit.get('no_cam_per_s'),1)),
            ("hit_drop_per_s", self._fmt_value(hit.get('drop_per_s'),1)),
            ("hit_pending_count", str(_safe_int(hit.get('pending_count',0),0))),
            ("hit_pending_impacts", str(_safe_int(hit.get('pending_impacts',0),0))),
        ]
        decision_rows: List[Tuple[str, str]] = []
        for dec in list(sync.get('hit_decisions', []))[-20:]:
            if not isinstance(dec, dict):
                continue
            age = max(0.0, _now_ms() - _safe_float(dec.get('ts_ms',0),0))
            decision_rows.append((f"{dec.get('cam','')} b{dec.get('bullet_id','')} id={dec.get('stable_id','')}", f"age={age:.0f}ms result={dec.get('result','')} reason={dec.get('reason','')} score={self._fmt_value(dec.get('score'),2)} sync_frames={dec.get('person_frame_delta_sync','-')} mask_dist={self._fmt_value(dec.get('mask_distance_px'),1)} method={dec.get('method','')}"))
        event_lines = [str(tmsg.get('text', '')) for tmsg in list(snap.get('texts', []))[-int(self.args.max_overlay_text_lines):]]
        recent_event_rows = [(f"event_{i}", evs[:300]) for i, evs in enumerate(event_lines[-12:], 1)]
        tabs["命中判定"] = [
            ("命中判定计数/原因码", hit_reason_rows),
            ("最近命中判定", decision_rows or [("recent", "-")]),
            ("最近叠加事件", recent_event_rows or [("recent", "-")]),
        ]
        return tabs

    def _draw_panel_lines(self, painter: QPainter, x: int, y: int, w: int, title: str, lines: List[str],
                          pen: QPen, bg: QColor, title_pen: Optional[QPen] = None, max_lines: int = 64) -> int:
        if w <= 40 or not lines:
            return y
        fm = painter.fontMetrics()
        line_h = max(int(self.args.status_font_px) + 5, fm.height() + 2)
        use_lines = [str(x) for x in lines[:max_lines]]
        h = line_h * (len(use_lines) + (1 if title else 0)) + 8
        max_h = max(40, self.height() - y - 8)
        h = min(h, max_h)
        painter.fillRect(x, y, w, h, bg)
        old_clip = painter.hasClipping()
        old_region = painter.clipRegion()
        painter.setClipRect(x, y, w, h)
        cy = y + line_h
        if title:
            painter.setPen(title_pen or pen)
            painter.drawText(x + 6, cy, fm.elidedText(str(title), Qt.ElideRight, max(10, w - 12)))
            cy += line_h
        painter.setPen(pen)
        for line in use_lines:
            if cy > y + h - 4:
                break
            painter.drawText(x + 6, cy, fm.elidedText(str(line), Qt.ElideRight, max(10, w - 12)))
            cy += line_h
        if old_clip:
            painter.setClipRegion(old_region)
        else:
            painter.setClipping(False)
        return y + h + 6

    def _fmt_cam_map(self, mp: Dict[str, Any], fmt: str = "{:.1f}", missing: str = "-") -> str:
        parts: List[str] = []
        for c in self.cams:
            v = mp.get(c, None)
            if v is None:
                parts.append(f"{c}:{missing}")
            else:
                try:
                    parts.append(f"{c}:" + fmt.format(float(v)))
                except Exception:
                    parts.append(f"{c}:{v}")
        return " ".join(parts)

    def _fmt_cam_map_lines(self, label: str, mp: Dict[str, Any], fmt: str = "{:.1f}", missing: str = "-", group: int = 4) -> List[str]:
        parts: List[str] = []
        for c in self.cams:
            v = mp.get(c, None)
            if v is None:
                parts.append(f"{c}:{missing}")
            else:
                try:
                    parts.append(f"{c}:" + fmt.format(float(v)))
                except Exception:
                    parts.append(f"{c}:{v}")
        out: List[str] = []
        g = max(1, int(group))
        for i in range(0, len(parts), g):
            prefix = f"{label:<24}" if i == 0 else " " * 24
            out.append(prefix + " ".join(parts[i:i + g]))
        return out

    def _fmt_bool_map(self, mp: Dict[str, Any], true_s: str = "Y", false_s: str = "N") -> str:
        return " ".join([f"{c}:{true_s if bool(mp.get(c, True)) else false_s}" for c in self.cams])

    def _fmt_bool_map_lines(self, label: str, mp: Dict[str, Any], true_s: str = "Y", false_s: str = "N", group: int = 4) -> List[str]:
        parts = [f"{c}:{true_s if bool(mp.get(c, True)) else false_s}" for c in self.cams]
        out: List[str] = []
        g = max(1, int(group))
        for i in range(0, len(parts), g):
            prefix = f"{label:<24}" if i == 0 else " " * 24
            out.append(prefix + " ".join(parts[i:i + g]))
        return out

    def _fmt_shm_missing_lines(self, raw_missing: Dict[str, Any], mvs_missing: Dict[str, Any], group: int = 4) -> List[str]:
        parts = [f"{c}:R{'Y' if raw_missing.get(c, True) else 'N'}/M{'Y' if mvs_missing.get(c, True) else 'N'}" for c in self.cams]
        out: List[str] = []
        g = max(1, int(group))
        for i in range(0, len(parts), g):
            prefix = f"{'per_cam_shm_missing':<24}" if i == 0 else " " * 24
            out.append(prefix + " ".join(parts[i:i + g]))
        return out

    def _renderer_timing_lines(self) -> List[str]:
        fps = max(0.0, float(self.render_fps_current))
        frame_period_ms = (1000.0 / fps) if fps > 0.001 else -1.0
        target_period_ms = 1000.0 / max(1e-3, float(self.args.render_fps))
        # paint_ms is what paintGL executed.  frame_period_ms/paint_interval_ms
        # include external throttling by Qt, swap, compositor, event loop, or timers.
        unmeasured_ms = max(0.0, frame_period_ms - float(self.renderer_paint_ms)) if frame_period_ms > 0 else -1.0
        return [
            f"renderer_paint_ms={self.renderer_paint_ms:.2f} render_fps={fps:.1f} frame_period_ms={frame_period_ms:.2f} target_period_ms={target_period_ms:.2f}",
            f"renderer_tick_ms={self.renderer_tick_ms:.2f} tick_interval_ms={self.renderer_tick_interval_ms:.2f} paint_interval_ms={self.renderer_paint_interval_ms:.2f} unmeasured_qt_swap_event_ms={unmeasured_ms:.2f}",
            f"target={float(self.args.render_fps):.0f}Hz traj_vtx={self.trajectory_vertex_count} traj_stale_guard={self.trajectory_stale_guard_count}",
        ]

    def build_status_text_lines(self, include_recent: bool = True) -> List[str]:
        snap = self.overlay_state.snapshot(marker_ttl_ms=float(self.args.overlay_marker_life_ms), text_ttl_ms=float(self.args.overlay_text_life_ms))
        try:
            telem = self.telemetry.poll(self.last_frame_ids)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] build_status_tabs telemetry poll failed: {type(exc).__name__}: {exc}", flush=True)
            telem = getattr(self.telemetry, 'last_snapshot', {}) or {}
        t = telem.get('trigger', {})
        yolo = telem.get('yolo', {})
        hit = telem.get('hit', {})
        ws_state = "ON" if snap.get("ws_connected") else "OFF"
        lines: List[str] = []
        lines.extend([
            "SYSTEM STATUS",
            f"trigger_state={t.get('state','unknown')}  trigger_fps={_safe_float(t.get('fps'),0.0):.2f}  jitter_p95_ms={_safe_float(t.get('jitter_p95_ms'),-1.0):.2f}  trigger_last_age_ms={_safe_float(t.get('last_age_ms'),-1.0):.0f}",
            f"mvs_active_count={telem.get('mvs_active_count',0)}/{len(self.cams)}  overlay_ws_connected={ws_state}  rx_count={snap.get('ws_rx_count',0)}",
            f"yolo_published_fps={_safe_float(yolo.get('published_fps_total', yolo.get('infer_fps')),0.0):.1f}  yolo_inferred_fps={_safe_float(yolo.get('inferred_fps_total'),0.0):.1f}  yolo_queue={_safe_int(yolo.get('postprocess_queue_depth'),0)}  yolo_drop={_safe_int(yolo.get('postprocess_dropped_batches'),0)}  polygon_ok={yolo.get('polygon_contract_ok')}",
            f"yolo_batch_size={_safe_int(yolo.get('batch_size'),0)}  yolo_predict_ms={_safe_float(yolo.get('predict_ms'),0.0):.1f}  yolo_capture_to_result_ms={_safe_float(yolo.get('capture_to_result_ms'),0.0):.1f}",
            f"hit_udp_recv_per_s={_safe_float(hit.get('recv_per_s'),0.0):.1f}  hit_no_human_per_s={_safe_float(hit.get('no_human_per_s'),0.0):.1f}  hit_no_human_count={_safe_int(hit.get('no_human_count'),0)}",
        ])
        lines.extend(self._renderer_timing_lines())
        lines.append("")
        lines.append("PER CAMERA")
        group = max(1, int(getattr(self.args, 'status_cam_group', 4)))
        lines.extend(self._fmt_cam_map_lines('per_cam_mvs_fps', telem.get('mvs_fps', {}), "{:.1f}", group=group))
        lines.extend(self._fmt_cam_map_lines('per_cam_raw_fps', telem.get('raw_fps', {}), "{:.1f}", group=group))
        lines.extend(self._fmt_cam_map_lines('per_cam_raw_age_ms', telem.get('raw_age_ms', {}), "{:.0f}", group=group))
        lines.extend(self._fmt_cam_map_lines('per_cam_frame_age_ms', telem.get('frame_age_ms', {}), "{:.0f}", group=group))
        lines.extend(self._fmt_bool_map_lines('per_cam_raw_missing', telem.get('raw_missing', {}), group=group))
        lines.extend(self._fmt_bool_map_lines('per_cam_mvs_missing', telem.get('mvs_missing', {}), group=group))
        lines.extend(self._fmt_shm_missing_lines(telem.get('raw_missing', {}), telem.get('mvs_missing', {}), group=group))
        lines.extend(self._fmt_cam_map_lines('per_cam_pending_q', telem.get('pending_q', {}), "{:.0f}", group=group))
        lines.extend(self._fmt_cam_map_lines('human_result_age_ms', telem.get('human_age_by_cam', {}), "{:.0f}", group=group))
        lines.extend(self._fmt_cam_map_lines('video_delta_frames', yolo.get('result_video_delta_frames', {}), "{:.0f}", group=group))
        if include_recent:
            event_lines = [str(tmsg.get('text', '')) for tmsg in list(snap.get('texts', []))[-int(self.args.max_overlay_text_lines):]]
            hit_prompts = list(snap.get('hit_prompts', []))[-int(getattr(self.args, 'max_hit_prompts', 4)):]
            if event_lines or hit_prompts:
                lines.append("")
                lines.append("RECENT EVENTS")
                for hp in hit_prompts:
                    lines.append(str(hp.get('label', 'HIT'))[:220])
                for ev in event_lines:
                    lines.append(ev[:220])
        return lines


    def _fmt_value(self, v: Any, digits: int = 1, missing: str = "-") -> str:
        if v is None:
            return missing
        if isinstance(v, bool):
            return "Y" if v else "N"
        try:
            fv = float(v)
            if fv <= -999998 or abs(fv - 999999.0) < 1e-6:
                return missing
            if abs(fv - int(fv)) < 1e-9 and digits <= 0:
                return str(int(fv))
            return f"{fv:.{digits}f}"
        except Exception:
            s = str(v)
            return s if s else missing

    def build_status_tabs(self) -> Dict[str, List[Tuple[str, List[Tuple[str, str]]]]]:
        """Return diagnostics grouped for the separate status window.

        This is deliberately read-only: it samples existing shm/jsonl/log state
        and does not touch the video, trajectory, human-mask, or hit rendering
        paths.
        """
        try:
            setattr(self.args, '_trajectory_drawn_vtx', int(self.trajectory_vertex_count))
        except Exception:
            pass
        snap = self.overlay_state.snapshot(marker_ttl_ms=float(self.args.overlay_marker_life_ms), text_ttl_ms=float(self.args.overlay_text_life_ms))
        try:
            telem = self.telemetry.poll(self.last_frame_ids)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] build_status_tabs telemetry poll failed: {type(exc).__name__}: {exc}", flush=True)
            telem = getattr(self.telemetry, 'last_snapshot', {}) or {}
        tb = telem.get('timebase', {})
        trig = telem.get('trigger', {})
        ev = telem.get('event', {})
        yolo = telem.get('yolo', {})
        hit = telem.get('hit', {})
        bullet = dict(telem.get('bullet', {}) or {})
        # Direct trajectory shm header sample for status only.  Header layout is
        # defined in preview_gpu_shm.py: count=5, seq=6, update_mono_ns=7.
        try:
            if self.trajectory_reader is not None and getattr(self.trajectory_reader, 'header', None) is not None:
                bullet['trajectory_shm_segment_count'] = int(self.trajectory_reader.header[5])
                bullet['trajectory_shm_seq'] = int(self.trajectory_reader.header[6])
                upd = int(self.trajectory_reader.header[7])
                bullet['trajectory_latest_age_ms'] = max(0.0, (time.monotonic_ns() - upd) / 1e6) if upd > 0 else -1.0
        except Exception:
            pass
        sync = telem.get('sync_health', {})
        humans = snap.get('humans', {}) if isinstance(snap, dict) else {}
        ws_state = "ON" if snap.get("ws_connected") else "OFF"
        tabs: Dict[str, List[Tuple[str, List[Tuple[str, str]]]]] = {}

        fps = max(0.0, float(self.render_fps_current))
        frame_period_ms = (1000.0 / fps) if fps > 0.001 else -1.0
        target_period_ms = 1000.0 / max(1e-3, float(self.args.render_fps))
        unmeasured_ms = max(0.0, frame_period_ms - float(self.renderer_paint_ms)) if frame_period_ms > 0 else -1.0

        tabs["总览"] = [
            ("全局时基", [
                ("system_wall_time_us", str(_safe_int(tb.get('system_wall_time_us', 0), 0))),
                ("system_mono_ns", str(_safe_int(tb.get('system_mono_ns', 0), 0))),
                ("sync_start_wall_us", str(_safe_int(tb.get('sync_start_wall_us', 0), 0))),
                ("sync_start_mono_ns", str(_safe_int(tb.get('sync_start_mono_ns', 0), 0))),
                ("event_timebase_valid", self._fmt_value(tb.get('event_timebase_valid'))),
                ("event_ts_to_wall_offset_us", str(_safe_int(tb.get('event_ts_to_wall_offset_us', 0), 0))),
                ("event_last_wall_age_ms", self._fmt_value(tb.get('event_last_wall_age_ms'), 1)),
                ("mvs_timebase_valid", self._fmt_value(tb.get('mvs_timebase_valid'))),
                ("mvs_last_wall_age_ms", self._fmt_value(tb.get('mvs_last_wall_age_ms'), 1)),
                ("clock_drift_event_vs_mvs_ms", self._fmt_value(tb.get('clock_drift_event_vs_mvs_ms'), 1)),
                ("clock_drift_event_vs_wall_ms", self._fmt_value(tb.get('clock_drift_event_vs_wall_ms'), 1)),
            ]),
            ("触发汇总", [
                ("trigger_state", str(trig.get('state', 'unknown'))),
                ("trigger_fps", self._fmt_value(trig.get('fps'), 2)),
                ("trigger_period_ms", self._fmt_value(trig.get('period_ms', 1000.0 / max(1e-3, _safe_float(trig.get('fps'), 0.0))) if _safe_float(trig.get('fps'), 0.0) > 0 else -1, 2)),
                ("trigger_jitter_p50_ms", self._fmt_value(trig.get('jitter_p50_ms'), 2)),
                ("trigger_jitter_p95_ms", self._fmt_value(trig.get('jitter_p95_ms'), 2)),
                ("trigger_jitter_max_ms", self._fmt_value(trig.get('jitter_max_ms'), 2)),
                ("trigger_last_age_ms", self._fmt_value(trig.get('last_age_ms'), 0)),
                ("ok_cams", f"{_safe_int(trig.get('ok_cams', 0), 0)}/{len(self.cams)}"),
            ]),
            ("链路汇总", [
                ("mvs_active_count", f"{_safe_int(telem.get('mvs_active_count',0),0)}/{len(self.cams)}"),
                ("event_active_count", f"{_safe_int(ev.get('event_active_count',0),0)}/{len(self.cams)}"),
                ("overlay_ws_connected", ws_state),
                ("overlay_ws_rx_count", str(_safe_int(snap.get('ws_rx_count', 0), 0))),
                ("overlay_ws_age_ms", self._fmt_value(snap.get('ws_age_ms', -1), 0)),
                ("human_jsonl_rx_count", str(_safe_int(snap.get('human_jsonl_rx_count', 0), 0))),
                ("hit_jsonl_rx_count", str(_safe_int(snap.get('hit_jsonl_rx_count', 0), 0))),
            ]),
            ("人体物理坐标", [
                ("projector_ok_cams", f"{getattr(self, '_physical_projector_ok_count', 0)}/{len(self.cams)}"),
                ("note", "per-camera xy in 每路相机链路 -> 每路物理坐标"),
            ]),
            ("YOLO汇总", [
                ("yolo_worker_count", str(_safe_int(yolo.get('worker_count', 0), 0))),
                ("yolo_published_fps_total", self._fmt_value(yolo.get('published_fps_total', yolo.get('infer_fps')), 1)),
                ("yolo_inferred_fps_total", self._fmt_value(yolo.get('inferred_fps_total'), 1)),
                ("yolo_async_record_postprocess_enabled", self._fmt_value(yolo.get('async_record_postprocess_enabled'))),
                ("yolo_postprocess_queue_depth", self._fmt_value(yolo.get('postprocess_queue_depth'), 0)),
                ("yolo_postprocess_dropped_batches", self._fmt_value(yolo.get('postprocess_dropped_batches'), 0)),
                ("yolo_polygon_contract_ok", self._fmt_value(yolo.get('polygon_contract_ok'))),
                ("yolo_batch_size", str(_safe_int(yolo.get('batch_size', 0), 0))),
                ("yolo_predict_ms", self._fmt_value(yolo.get('predict_ms'), 1)),
                ("yolo_capture_to_result_ms", self._fmt_value(yolo.get('capture_to_result_ms'), 1)),
            ]),
            ("命中汇总", [
                ("hit_udp_recv_per_s", self._fmt_value(hit.get('recv_per_s'), 1)),
                ("hit_eval_per_s", self._fmt_value(hit.get('eval_per_s'), 1)),
                ("hit_candidate_per_s", self._fmt_value(hit.get('candidate_per_s'), 1)),
                ("hit_confirmed_per_s", self._fmt_value(hit.get('confirmed_per_s'), 1)),
                ("hit_no_human_per_s", self._fmt_value(hit.get('no_human_per_s'), 1)),
                ("hit_no_human_count", str(_safe_int(hit.get('no_human_count', 0), 0))),
                ("hit_pending_count", str(_safe_int(hit.get('pending_count', 0), 0))),
                ("hit_pending_impacts", str(_safe_int(hit.get('pending_impacts', 0), 0))),
            ]),
            ("渲染时序", [
                ("video_backend_state", str(getattr(self, "video_backend_state", "unknown"))),
                ("video_backend_stage", str(getattr(self, "video_backend_init_stage", ""))),
                ("video_backend_error", str(getattr(self, "video_backend_error", ""))[:260] or "-"),
                ("video_backend_attempts", str(_safe_int(getattr(self, "video_backend_attempts", 0), 0))),
                ("video_backend_retries", str(_safe_int(getattr(self, "video_backend_retry_count", 0), 0))),
                ("timers_started", f"video={bool(getattr(self, 'video_timer_started', False))} render={bool(getattr(self, 'render_timer_started', False))} overlay={bool(getattr(self, 'overlay_timer_started', False))}"),
                ("last_video_tick_error", str(getattr(self, "last_video_tick_error", ""))[:180] or "-"),
                ("last_render_tick_error", str(getattr(self, "last_render_tick_error", ""))[:180] or "-"),
                ("last_paint_error", str(getattr(self, "last_paint_error", ""))[:180] or "-"),
                ("renderer_paint_ms", self._fmt_value(self.renderer_paint_ms, 2)),
                ("render_fps", self._fmt_value(fps, 1)),
                ("frame_period_ms", self._fmt_value(frame_period_ms, 2)),
                ("target_period_ms", self._fmt_value(target_period_ms, 2)),
                ("renderer_tick_ms", self._fmt_value(self.renderer_tick_ms, 2)),
                ("tick_interval_ms", self._fmt_value(self.renderer_tick_interval_ms, 2)),
                ("paint_interval_ms", self._fmt_value(self.renderer_paint_interval_ms, 2)),
                ("unmeasured_qt_swap_event_ms", self._fmt_value(unmeasured_ms, 2)),
                ("trajectory_drawn_vtx", str(int(self.trajectory_vertex_count))),
            ]),
        ]

        per_rows: List[Tuple[str, str]] = []
        mvs_fps = telem.get('mvs_fps', {})
        raw_fps = telem.get('raw_fps', {})
        raw_age = telem.get('raw_age_ms', {})
        frame_age = telem.get('frame_age_ms', {})
        raw_missing = telem.get('raw_missing', {})
        mvs_missing = telem.get('mvs_missing', {})
        pending_q = telem.get('pending_q', {})
        human_age = telem.get('human_age_by_cam', {})
        deltas = yolo.get('result_video_delta_frames', {}) if isinstance(yolo, dict) else {}
        for cam in self.cams:
            hrec = humans.get(cam, {}) if isinstance(humans, dict) else {}
            per_rows.append((cam, (
                f"mvs={self._fmt_value(mvs_fps.get(cam),1)} raw={self._fmt_value(raw_fps.get(cam),1)} "
                f"raw_age={self._fmt_value(raw_age.get(cam),0)}ms frame_age={self._fmt_value(frame_age.get(cam),0)}ms "
                f"raw_missing={'Y' if raw_missing.get(cam, True) else 'N'} mvs_missing={'Y' if mvs_missing.get(cam, True) else 'N'} "
                f"pending_q={self._fmt_value(pending_q.get(cam),0)} human_age={self._fmt_value(human_age.get(cam),0)}ms "
                f"delta_frames={self._fmt_value(deltas.get(cam),0)} persons={_safe_int(hrec.get('person_count', 0), 0)}"
            )))
        trigger_rows: List[Tuple[str, str]] = []
        for cam in self.cams:
            trigger_rows.append((cam, (
                f"sent={trig.get('per_cam_sent',{}).get(cam,'-')} ack={trig.get('per_cam_ack',{}).get(cam,'-')} "
                f"fail={trig.get('per_cam_fail',{}).get(cam,'-')} miss={trig.get('per_cam_trigger_miss_count',{}).get(cam,'-')} "
                f"trig_to_mvs_ms={self._fmt_value(trig.get('per_cam_trigger_to_mvs_frame_ms',{}).get(cam),1)} "
                f"trig_to_event_age_ms={self._fmt_value(trig.get('per_cam_trigger_to_event_trigger_ms',{}).get(cam),1)}"
            )))
        yolo_rows: List[Tuple[str, str]] = []
        for cam in self.cams:
            hrec = humans.get(cam, {}) if isinstance(humans, dict) else {}
            yolo_rows.append((cam, (
                f"human_frame={hrec.get('frame_id','-')} persons={_safe_int(hrec.get('person_count',0),0)} "
                f"capture_wall_us={_safe_int(hrec.get('capture_wall_us',0),0)} record_wall_us={_safe_int(hrec.get('record_wall_us',0),0)} "
                f"capture_to_result_ms={self._fmt_value(hrec.get('yolo_capture_to_result_ms', -1),1)} "
                f"result_video_delta_frames={self._fmt_value(deltas.get(cam),0)}"
            )))
        physical_rows: List[Tuple[str, str]] = []
        projector_ok = 0
        for cam in self.cams:
            hrec = humans.get(cam, {}) if isinstance(humans, dict) else {}
            proj = hrec.get('projector', {}) if isinstance(hrec.get('projector', {}), dict) else {}
            persons = hrec.get('persons', []) if isinstance(hrec.get('persons', []), list) else []
            valid = bool(proj.get('valid', False)) if proj else any(isinstance(p, dict) and (p.get('physical_coord') or p.get('local_xy')) for p in persons)
            if valid:
                projector_ok += 1
            parts = [f"proj={'OK' if valid else 'BAD'}", f"frame={hrec.get('frame_id','-')}"]
            if proj:
                parts.append(f"coord={proj.get('coord_frame','local_ground')}/{proj.get('unit','m')}")
                if not valid and proj.get('last_error'):
                    parts.append(f"reason={proj.get('last_error')}")
            shown = 0
            for person in persons:
                if not isinstance(person, dict):
                    continue
                pc = person.get('physical_coord') if isinstance(person.get('physical_coord'), dict) else {}
                lx = ly = None
                if pc and pc.get('ok', False):
                    lx, ly = pc.get('x_m'), pc.get('y_m')
                elif isinstance(person.get('local_xy'), (list, tuple)) and len(person.get('local_xy')) >= 2:
                    lx, ly = person.get('local_xy')[0], person.get('local_xy')[1]
                if lx is None or ly is None:
                    continue
                sid = person.get('stable_id', person.get('person_id',''))
                foot = person.get('foot_pixel')
                foot_s = ''
                if isinstance(foot, (list, tuple)) and len(foot) >= 2:
                    foot_s = f" foot=({_safe_float(foot[0]):.0f},{_safe_float(foot[1]):.0f})"
                parts.append(f"id={sid} xy=({_safe_float(lx):.2f},{_safe_float(ly):.2f})m{foot_s}")
                shown += 1
                if shown >= 3:
                    break
            if shown == 0:
                parts.append('persons_xy=-')
            physical_rows.append((cam, ' '.join(parts)))
        try:
            self._physical_projector_ok_count = int(projector_ok)
        except Exception:
            pass
        tabs["每路相机链路"] = [
            ("每路实时链路", per_rows),
            ("每路触发关联", trigger_rows),
            ("每路YOLO/人体结果", yolo_rows),
            ("每路物理坐标", physical_rows),
        ]

        event_rows: List[Tuple[str, str]] = []
        effective_event_overlay_stats = self.get_effective_event_overlay_stats()
        for cam in self.cams:
            event_rows.append((cam, (
                f"ready={'Y' if ev.get('per_cam_ready',{}).get(cam,False) else 'N'} "
                f"event_trigger_fps={self._fmt_value(ev.get('per_cam_event_trigger_fps',{}).get(cam),1)} "
                f"event_rate_mevs={self._fmt_value(ev.get('per_cam_event_rate_mevs',{}).get(cam),2)} "
                f"loop_fps={self._fmt_value(ev.get('per_cam_worker_loop_fps',{}).get(cam),1)} "
                f"empty={self._fmt_value(ev.get('per_cam_worker_empty_ratio',{}).get(cam),2)} "
                f"fast={ev.get('per_cam_worker_fastpath_count',{}).get(cam,'-')} "
                f"slice_age_ms={self._fmt_value(ev.get('per_cam_event_slice_age_ms',{}).get(cam),1)} "
                f"last_ts_us={_safe_int(ev.get('per_cam_event_last_ts_us',{}).get(cam,0),0)} "
                f"trigger_age_ms={self._fmt_value(ev.get('per_cam_event_trigger_last_age_ms',{}).get(cam),0)} "
                f"triggers={_safe_int(ev.get('per_cam_event_trigger_recv_count',{}).get(cam,0),0)} "
                f"gaps={_safe_int(ev.get('per_cam_event_trigger_gap_count',{}).get(cam,0),0)} "
                f"nonmono={_safe_int(ev.get('per_cam_event_nonmonotonic_count',{}).get(cam,0),0)} "
                f"guard={ev.get('per_cam_event_guard_state',{}).get(cam,'OK')} "
                f"viz_stage={ev.get('per_cam_viz_source_stage',{}).get(cam,'')} "
                f"viz_raw={ev.get('per_cam_viz_raw_count',{}).get(cam,'-')} "
                f"viz_filt={ev.get('per_cam_viz_filtered_count',{}).get(cam,'-')} "
                f"viz_red={self._fmt_value(100.0 * _safe_float(ev.get('per_cam_viz_reduction_ratio',{}).get(cam,0.0),0.0),1)}% "
                f"viz_mask={'Y' if ev.get('per_cam_viz_mask_valid',{}).get(cam,False) else 'N'} "
                f"viz_buf={ev.get('per_cam_viz_buffer_size',{}).get(cam,'-')} "
                f"viz_drop={ev.get('per_cam_viz_buffer_drop_count',{}).get(cam,'-')} "
                f"evt_ovl_draw={'Y' if effective_event_overlay_stats.get(cam,{}).get('drawn', False) else 'N'} "
                f"evt_ovl_stale={'Y' if effective_event_overlay_stats.get(cam,{}).get('stale', True) else 'N'} "
                f"evt_ovl_mode={effective_event_overlay_stats.get(cam,{}).get('mode', getattr(self.args,'event_overlay_sync_mode','nearest_mvs'))} "
                f"evt_ovl_age={self._fmt_value(effective_event_overlay_stats.get(cam,{}).get('age_ms'),0)}ms "
                f"evt_ovl_dt_mvs={self._fmt_value(effective_event_overlay_stats.get(cam,{}).get('dt_mvs_ms'),1)} "
                f"evt_ovl_dt_human={self._fmt_value(effective_event_overlay_stats.get(cam,{}).get('dt_human_ms'),1)} "
                f"evt_ovl_raw={effective_event_overlay_stats.get(cam,{}).get('raw','-')} "
                f"evt_ovl_filt={effective_event_overlay_stats.get(cam,{}).get('filtered','-')}"
            )))
        bullet_rows = [
            ("bullet_udp_recv_per_s", self._fmt_value(bullet.get('udp_recv_per_s'),1)),
            ("bullet_track_count_active", str(_safe_int(bullet.get('track_count_active',0),0))),
            ("bullet_latest_age_ms", self._fmt_value(bullet.get('latest_age_ms'),1)),
            ("bullet_latest_cam", str(bullet.get('latest_cam',''))),
            ("bullet_latest_id", str(_safe_int(bullet.get('latest_id',0),0))),
            ("bullet_latest_point_count", str(_safe_int(bullet.get('latest_point_count',0),0))),
            ("bullet_status_ttl_ms", self._fmt_value(bullet.get('bullet_status_ttl_ms'),0)),
            ("bullet_fresh_point_count", str(_safe_int(bullet.get('fresh_point_count',0),0))),
            ("bullet_stale_point_count", str(_safe_int(bullet.get('stale_point_count',0),0))),
            ("bullet_future_point_count", str(_safe_int(bullet.get('future_point_count',0),0))),
            ("bullet_stale_drop_count", str(_safe_int(bullet.get('stale_drop_count',0),0))),
            ("trajectory_shm_seq", str(_safe_int(bullet.get('trajectory_shm_seq',0),0))),
            ("trajectory_shm_segment_count", str(_safe_int(bullet.get('trajectory_shm_segment_count',0),0))),
            ("trajectory_visible_segment_count", str(_safe_int(bullet.get('trajectory_visible_segment_count',0),0))),
            ("trajectory_latest_age_ms", self._fmt_value(bullet.get('trajectory_latest_age_ms'),1)),
            ("trajectory_stale_guard_count", str(_safe_int(bullet.get('trajectory_stale_guard_count',0),0))),
            ("trajectory_drawn_vtx", str(int(self.trajectory_vertex_count))),
        ]
        recent_bullet_rows: List[Tuple[str, str]] = []
        for pt in list(bullet.get('recent', []))[-12:]:
            if not isinstance(pt, dict):
                continue
            recent_bullet_rows.append((f"{pt.get('cam','')} b{pt.get('bullet_id',0)} p{pt.get('point_index',-1)}", f"age={self._fmt_value(_now_ms()-_safe_float(pt.get('ts_ms',0),0),0)}ms event_ts={pt.get('point_ts_us',0)} xy=({_safe_float(pt.get('x'),0):.0f},{_safe_float(pt.get('y'),0):.0f}) dt={self._fmt_value(pt.get('point_delta_ms'),1)}ms state={pt.get('status','')}"))
        tabs["事件/弹道"] = [
            ("事件相机健康", event_rows),
            ("弹道/轨迹汇总", bullet_rows),
            ("最近弹道点", recent_bullet_rows or [("recent", "-")]),
        ]

        latest_sync_rows = [
            ("latest_cam", str(sync.get('cam', sync.get('latest_cam', '')))),
            ("bullet_id", str(_safe_int(sync.get('bullet_id', 0), 0))),
            ("event_ts_us", str(_safe_int(sync.get('event_ts_us', 0), 0))),
            ("event_to_mvs_delta_ms", self._fmt_value(sync.get('event_to_mvs_delta_ms'), 2)),
            ("event_to_human_capture_delta_ms", self._fmt_value(sync.get('event_to_human_capture_delta_ms'), 2)),
            ("event_to_human_result_age_ms", self._fmt_value(sync.get('event_to_human_result_age_ms'), 1)),
            ("event_to_human_delta_frames", self._fmt_value(sync.get('event_to_human_delta_frames'), 0)),
            ("mask_valid_at_event", self._fmt_value(sync.get('mask_valid_at_event', False))),
            ("person_count_at_event", str(_safe_int(sync.get('person_count_at_event', 0), 0))),
            ("matched_person_id", str(sync.get('matched_person_id', ''))),
            ("matched_stable_id", str(sync.get('matched_stable_id', ''))),
            ("hit_inside_mask", self._fmt_value(sync.get('hit_inside_mask', False))),
            ("hit_distance_to_mask_px", self._fmt_value(sync.get('hit_distance_to_mask_px'), 1)),
            ("hit_distance_to_bbox_px", self._fmt_value(sync.get('hit_distance_to_bbox_px'), 1)),
            ("sync_quality", str(sync.get('sync_quality', 'UNKNOWN'))),
            ("latest_hit_reason", str(sync.get('latest_hit_reason', ''))),
        ]
        thresholds = [
            ("OK", "abs(event_to_human_capture_delta_ms) <= 25ms"),
            ("WARN", "<= 75ms"),
            ("BAD", "> 75ms or > 3 frames"),
            ("STALE", "human_result_age_ms > 500ms"),
            ("MISSING", "no usable human_result for bullet camera"),
        ]
        recent_sync_rows: List[Tuple[str, str]] = []
        for rec in list(sync.get('recent', []))[-16:]:
            if not isinstance(rec, dict):
                continue
            recent_sync_rows.append((f"{rec.get('cam','')} b{rec.get('bullet_id',0)} p{rec.get('point_index','')}", f"quality={rec.get('sync_quality','')} e2mvs={self._fmt_value(rec.get('event_to_mvs_delta_ms'),1)}ms e2human={self._fmt_value(rec.get('event_to_human_capture_delta_ms'),1)}ms frames={rec.get('event_to_human_delta_frames','-')} human_age={self._fmt_value(rec.get('event_to_human_result_age_ms'),0)}ms persons={rec.get('person_count_at_event',0)} mask_valid={rec.get('mask_valid_at_event',False)}"))
        tabs["Bullet ↔ Human Sync"] = [
            ("Latest bullet-human sync", latest_sync_rows),
            ("Quality thresholds", thresholds),
            ("Recent sync samples", recent_sync_rows or [("recent", "-")]),
        ]

        hit_reason_rows = [
            ("hit_udp_recv_per_s", self._fmt_value(hit.get('recv_per_s'),1)),
            ("hit_eval_per_s", self._fmt_value(hit.get('eval_per_s'),1)),
            ("hit_candidate_per_s", self._fmt_value(hit.get('candidate_per_s'),1)),
            ("hit_confirmed_per_s", self._fmt_value(hit.get('confirmed_per_s'),1)),
            ("hit_no_human_per_s", self._fmt_value(hit.get('no_human_per_s'),1)),
            ("hit_no_human_count", str(_safe_int(hit.get('no_human_count',0),0))),
            ("hit_no_mask_per_s", self._fmt_value(hit.get('no_mask_per_s'),1)),
            ("hit_stale_human_per_s", self._fmt_value(hit.get('stale_human_per_s'),1)),
            ("hit_sync_bad_per_s", self._fmt_value(hit.get('sync_bad_per_s'),1)),
            ("hit_outside_mask_per_s", self._fmt_value(hit.get('outside_mask_per_s'),1)),
            ("hit_no_cam_per_s", self._fmt_value(hit.get('no_cam_per_s'),1)),
            ("hit_drop_per_s", self._fmt_value(hit.get('drop_per_s'),1)),
            ("hit_pending_count", str(_safe_int(hit.get('pending_count',0),0))),
            ("hit_pending_impacts", str(_safe_int(hit.get('pending_impacts',0),0))),
        ]
        decision_rows: List[Tuple[str, str]] = []
        for dec in list(sync.get('hit_decisions', []))[-20:]:
            if not isinstance(dec, dict):
                continue
            age = max(0.0, _now_ms() - _safe_float(dec.get('ts_ms',0),0))
            decision_rows.append((f"{dec.get('cam','')} b{dec.get('bullet_id','')} id={dec.get('stable_id','')}", f"age={age:.0f}ms result={dec.get('result','')} reason={dec.get('reason','')} score={self._fmt_value(dec.get('score'),2)} sync_frames={dec.get('person_frame_delta_sync','-')} mask_dist={self._fmt_value(dec.get('mask_distance_px'),1)} method={dec.get('method','')}"))
        event_lines = [str(tmsg.get('text', '')) for tmsg in list(snap.get('texts', []))[-int(self.args.max_overlay_text_lines):]]
        recent_event_rows = [(f"event_{i}", evs[:300]) for i, evs in enumerate(event_lines[-12:], 1)]
        tabs["命中判定"] = [
            ("命中判定计数/原因码", hit_reason_rows),
            ("最近命中判定", decision_rows or [("recent", "-")]),
            ("最近叠加事件", recent_event_rows or [("recent", "-")]),
        ]
        return tabs

    def _draw_panel_lines(self, painter: QPainter, x: int, y: int, w: int, title: str, lines: List[str],
                          pen: QPen, bg: QColor, title_pen: Optional[QPen] = None, max_lines: int = 64) -> int:
        if w <= 40 or not lines:
            return y
        fm = painter.fontMetrics()
        line_h = max(int(self.args.status_font_px) + 5, fm.height() + 2)
        use_lines = [str(x) for x in lines[:max_lines]]
        h = line_h * (len(use_lines) + (1 if title else 0)) + 8
        max_h = max(40, self.height() - y - 8)
        h = min(h, max_h)
        painter.fillRect(x, y, w, h, bg)
        old_clip = painter.hasClipping()
        old_region = painter.clipRegion()
        painter.setClipRect(x, y, w, h)
        cy = y + line_h
        if title:
            painter.setPen(title_pen or pen)
            painter.drawText(x + 6, cy, fm.elidedText(str(title), Qt.ElideRight, max(10, w - 12)))
            cy += line_h
        painter.setPen(pen)
        for line in use_lines:
            if cy > y + h - 4:
                break
            painter.drawText(x + 6, cy, fm.elidedText(str(line), Qt.ElideRight, max(10, w - 12)))
            cy += line_h
        if old_clip:
            painter.setClipRegion(old_region)
        else:
            painter.setClipping(False)
        return y + h + 6

    def _draw_text_overlay(self) -> None:
        # Keep only video-attached labels/prompts in the preview window.  The
        # dense diagnostic status layer is shown in a separate StatusWindow by
        # default to avoid covering video/human/hit overlays.
        if not bool(self.args.status_text_enable):
            return
        snap = self.overlay_state.snapshot(marker_ttl_ms=float(self.args.overlay_marker_life_ms), text_ttl_ms=float(self.args.overlay_text_life_ms))
        try:
            telem = self.telemetry.poll(self.last_frame_ids)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] build_status_tabs telemetry poll failed: {type(exc).__name__}: {exc}", flush=True)
            telem = getattr(self.telemetry, 'last_snapshot', {}) or {}
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            font = QFont("DejaVu Sans Mono", int(self.args.status_font_px))
            painter.setFont(font)
            pen_info = QPen(QColor(230, 245, 255, 235))
            pen_warn = QPen(QColor(255, 210, 60, 240))
            pen_hit = QPen(QColor(255, 80, 40, 245))
            title_pen = QPen(QColor(160, 220, 255, 245))
            bg = QColor(0, 0, 0, int(getattr(self.args, "status_panel_alpha", 150)))

            if bool(getattr(self.args, 'status_inline_enable', False)):
                w_panel = min(int(getattr(self.args, "status_panel_width", 760)), max(420, self.width() // 2 - 12))
                x = 8
                y = 8
                status_lines = self.build_status_text_lines(include_recent=False)
                # Split into compact inline panels, but leave the default off.
                try:
                    idx = status_lines.index("")
                    main_lines = [ln for ln in status_lines[1:idx] if ln]
                    per_title_idx = status_lines.index("PER CAMERA")
                    per_lines = [ln for ln in status_lines[per_title_idx + 1:] if ln]
                except Exception:
                    main_lines = status_lines[:8]
                    per_lines = status_lines[8:]
                y = self._draw_panel_lines(painter, x, y, w_panel, "SYSTEM STATUS", main_lines, pen_info, bg, title_pen, max_lines=12)
                self._draw_panel_lines(painter, x, y, w_panel, "PER CAMERA", per_lines, pen_info, bg, title_pen, max_lines=12)

            # Compact per-tile labels only.  Full metrics live in the separate status window.
            if bool(getattr(self.args, "tile_label_enable", True)):
                fm = painter.fontMetrics()
                line_h = max(int(self.args.status_font_px) + 5, fm.height() + 2)
                tile_w_px = max(1, int(self.width() / max(1, self.cols)))
                tile_h_px = max(1, int(self.height() / max(1, self.rows)))
                for i, cam in enumerate(self.cams):
                    col = i % self.cols
                    row = i // self.cols
                    x0 = int(col * tile_w_px + 10)
                    y0 = int((row + 1) * tile_h_px - 12)
                    txt = f"{cam} fid={self.last_frame_ids.get(cam, -1)} raw={telem.get('raw_fps',{}).get(cam,0.0):.1f} mvs={telem.get('mvs_fps',{}).get(cam,0.0):.1f}"
                    box_w = min(tile_w_px - 20, max(180, painter.fontMetrics().horizontalAdvance(txt) + 12))
                    painter.fillRect(x0 - 4, y0 - line_h + 2, int(box_w), line_h + 4, bg)
                    painter.setPen(pen_info)
                    painter.drawText(x0, y0, painter.fontMetrics().elidedText(txt, Qt.ElideRight, max(10, int(box_w) - 8)))

            if bool(getattr(self.args, "hit_prompt_enable", True)) and bool(getattr(self, "preview_hit_display_enable", True)):
                hit_prompts = list(snap.get("hit_prompts", []))[-int(getattr(self.args, "max_hit_prompts", 4)):]
                if hit_prompts:
                    big_font = QFont("DejaVu Sans Mono", int(getattr(self.args, "hit_prompt_font_px", 32)))
                    old_font = painter.font()
                    painter.setFont(big_font)
                    fm_big = painter.fontMetrics()
                    hp = hit_prompts[-1]
                    label = str(hp.get("label", "HIT 命中"))
                    cam_s = str(hp.get("cam", ""))
                    method_s = str(hp.get("method", ""))
                    banner = f"{cam_s} {label}" + (f"  {method_s}" if method_s else "")
                    bh = int(getattr(self.args, "hit_prompt_font_px", 32)) + 16
                    bw = min(self.width() - 40, max(460, fm_big.horizontalAdvance(banner) + 28))
                    bx = int((self.width() - bw) / 2)
                    by = 48
                    painter.fillRect(bx, by, int(bw), bh, QColor(90, 0, 0, 210))
                    painter.setPen(QPen(QColor(255, 245, 80, 255)))
                    painter.drawText(bx + 14, by + bh - 12, fm_big.elidedText(banner, Qt.ElideRight, int(bw) - 28))
                    painter.setFont(old_font)
                    fm = painter.fontMetrics()
                    line_h = max(int(self.args.status_font_px) + 5, fm.height() + 2)
                    for hp in hit_prompts:
                        xy = self._tile_px_for_cam(hp.get("cam", ""), _safe_float(hp.get("x")), _safe_float(hp.get("y")), _safe_float(hp.get("display_w", 1), 1), _safe_float(hp.get("display_h", 1), 1))
                        if xy is None:
                            continue
                        px, py = xy
                        label2 = str(hp.get("label", "HIT 命中"))
                        box_w = min(420, max(120, fm.horizontalAdvance(label2) + 12))
                        painter.setPen(pen_hit)
                        painter.fillRect(int(px) + 12, int(py) + 10, int(box_w), line_h + 5, QColor(80, 0, 0, 180))
                        painter.drawText(int(px) + 17, int(py) + 10 + line_h, fm.elidedText(label2, Qt.ElideRight, int(box_w) - 8))

            fm = painter.fontMetrics()
            line_h = max(int(self.args.status_font_px) + 5, fm.height() + 2)
            for m in list(snap.get("markers", []))[-int(self.args.max_overlay_marker_labels):]:
                if not bool(getattr(self, "preview_hit_display_enable", True)) and str(m.get("kind", "")).lower() == "hit":
                    continue
                label = str(m.get("label", ""))
                if not label:
                    continue
                xy = self._tile_px_for_cam(m.get("cam", ""), _safe_float(m.get("x")), _safe_float(m.get("y")), _safe_float(m.get("display_w", 1), 1), _safe_float(m.get("display_h", 1), 1))
                if xy is None:
                    continue
                px, py = xy
                box_w = min(260, max(72, fm.horizontalAdvance(label) + 12))
                painter.setPen(pen_hit if str(m.get("kind")) == "hit" else pen_warn)
                painter.fillRect(int(px) + 8, int(py) - line_h, int(box_w), line_h + 4, bg)
                painter.drawText(int(px) + 12, int(py), fm.elidedText(label, Qt.ElideRight, int(box_w) - 8))
        finally:
            painter.end()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self.overlay_ws_client is not None:
                self.overlay_ws_client.stop()
        except Exception:
            pass
        try:
            if self.video_backend is not None:
                self.video_backend.release()
        except Exception:
            pass
        return super().closeEvent(event)






class EventOverlayOverviewWindow(QOpenGLWidget):
    """Independent window for event-camera overlay overview.

    This window intentionally renders event-camera visualization separately from
    the RGB preview.  The main preview remains dedicated to visible-light video,
    human masks, bullet trajectories, and HIT prompts.  This separation prevents
    event overlay texture uploads from reducing the main preview FPS and makes
    event/MVS/human sync diagnostics visible without hiding the RGB image.
    """

    def __init__(self, preview: PreviewWidget, args: argparse.Namespace):
        super().__init__()
        self.preview = preview
        self.args = args
        self.cams = list(preview.cams)
        self.rows, self.cols = _grid_layout(self.cams, getattr(args, "event_overlay_window_layout", getattr(args, "layout", "grid2x4")))
        self.setWindowTitle("PyOpenGL CUDA Preview - Event Overlay Overview")
        self.resize(int(getattr(args, "event_overlay_window_width", 1280)), int(getattr(args, "event_overlay_window_height", 720)))
        self.readers: Dict[str, EventOverlayPreviewReader] = {}
        self.reader_errors: Dict[str, str] = {}
        self.texture_ids: Dict[str, int] = {}
        self.texture_shapes: Dict[str, Tuple[int, int, int, int]] = {}
        self.uploaded_once: Dict[str, bool] = {}
        self.last_seq: Dict[str, int] = {c: -1 for c in self.cams}
        self.overlay_shader_program: int = 0
        self.overlay_shader_error: str = ""
        self.overlay_shader_uniforms: Dict[str, int] = {}
        self.font_label = QFont("DejaVu Sans Mono", max(9, int(getattr(args, "event_overlay_window_font_px", 12))))
        label_fps = max(0.0, float(getattr(args, "event_overlay_window_label_fps", 10.0) or 0.0))
        self.label_interval_ms = (1000.0 / label_fps) if label_fps > 0.001 else 0.0
        self._last_label_update_ms = 0.0
        self._cached_label_stats: List[Tuple[str, int, int, int, int, Dict[str, Any]]] = []
        self.paint_fps_current = 0.0
        self.paint_ms_current = 0.0
        self.paint_interval_ms = 0.0
        self._last_paint_perf = 0.0
        self._fps_window_start = time.perf_counter()
        self._fps_window_count = 0
        self.setAutoFillBackground(False)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update)
        fps = max(1.0, float(getattr(args, "event_overlay_window_fps", 50.0)))
        self.timer.start(max(5, int(round(1000.0 / fps))))

    def initializeGL(self) -> None:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.03, 0.035, 0.04, 1.0)
        renderer = GL.glGetString(GL.GL_RENDERER)
        print(f"[pygl_cuda_preview] event overlay overview GPU renderer={renderer!r}", flush=True)
        self._ensure_overlay_shader()

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, max(1, int(w)), max(1, int(h)))

    def _try_open_readers(self) -> None:
        if not bool(getattr(self.args, "event_overlay_preview_enable", False)):
            return
        sync_root = str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")
        for cam in self.cams:
            if cam in self.readers:
                continue
            try:
                rd = EventOverlayPreviewReader(
                    cam,
                    str(getattr(self.args, "event_overlay_preview_template", "event_overlay_latest_{camera}")),
                    sync_root,
                    str(getattr(self.args, "event_overlay_sidecar_template", "event_overlay_{camera}_latest.json")),
                )
                self.readers[cam] = rd
                self.reader_errors.pop(cam, None)
                print(f"[pygl_cuda_preview] event overlay overview opened {cam}: {rd.path}", flush=True)
            except Exception as exc:
                self.reader_errors[cam] = f"{type(exc).__name__}: {exc}"
                self.preview.event_overlay_stats.setdefault(cam, {})["open_error"] = self.reader_errors[cam]

    def _ensure_texture(self, cam: str, width: int, height: int, internal_format: int, pixel_format: int) -> int:
        tex = int(self.texture_ids.get(cam, 0) or 0)
        if tex <= 0:
            tex = int(GL.glGenTextures(1))
            self.texture_ids[cam] = tex
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        shape = (int(width), int(height), int(internal_format), int(pixel_format))
        if self.texture_shapes.get(cam) != shape:
            self.uploaded_once[cam] = False
            self.texture_shapes[cam] = shape
        return tex

    def _requested_dilate_backend(self) -> str:
        raw = str(getattr(self.args, "event_overlay_preview_dilate_backend", "gl_shader") or "gl_shader").strip().lower()
        if raw not in ("gl_shader", "cpu_sparse", "off", "auto"):
            return "gl_shader"
        return raw

    def _active_dilate_backend(self) -> str:
        requested = self._requested_dilate_backend()
        if requested == "off":
            return "off"
        if int(getattr(self.args, "event_overlay_preview_dilate_px", 4) or 0) <= 0:
            return "off"
        if requested in ("gl_shader", "auto"):
            if int(self.overlay_shader_program or 0) > 0:
                return "gl_shader"
            if requested == "auto":
                return "cpu_sparse"
            return "off"
        return "cpu_sparse"

    def _shader_info_log(self, obj: int, is_program: bool = False) -> str:
        try:
            if is_program:
                return GL.glGetProgramInfoLog(obj).decode("utf-8", "replace")
            return GL.glGetShaderInfoLog(obj).decode("utf-8", "replace")
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _compile_shader(self, shader_type: int, source: str) -> int:
        shader = int(GL.glCreateShader(shader_type))
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        ok = int(GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS))
        if not ok:
            log = self._shader_info_log(shader)
            try:
                GL.glDeleteShader(shader)
            except Exception:
                pass
            raise RuntimeError(log)
        return shader

    def _ensure_overlay_shader(self) -> None:
        if self._requested_dilate_backend() in ("off", "cpu_sparse"):
            return
        if int(self.overlay_shader_program or 0) > 0:
            return
        vertex_src = """
#version 120
void main() {
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = ftransform();
}
"""
        fragment_src = """
#version 120
uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform int u_radius;
uniform float u_alpha;
uniform float u_gain;
uniform float u_gamma;
uniform float u_min_alpha;
void main() {
    vec2 uv = gl_TexCoord[0].st;
    vec3 best = vec3(0.0);
    float best_luma = 0.0;
    for (float fy = -8.0; fy <= 8.0; fy += 1.0) {
        for (float fx = -8.0; fx <= 8.0; fx += 1.0) {
            if (abs(fx) <= float(u_radius) && abs(fy) <= float(u_radius)) {
                vec3 c = texture2D(u_tex, uv + vec2(fx, fy) * u_texel).rgb;
                float l = max(max(c.r, c.g), c.b);
                if (l > best_luma) {
                    best_luma = l;
                    best = c;
                }
            }
        }
    }
    if (best_luma <= 0.0) {
        gl_FragColor = vec4(0.0);
        return;
    }
    vec3 rgb = pow(clamp(best * u_gain, 0.0, 1.0), vec3(u_gamma));
    float a = pow(clamp(best_luma * u_gain, 0.0, 1.0), u_gamma);
    a = max(a, u_min_alpha) * u_alpha;
    gl_FragColor = vec4(rgb, clamp(a, 0.0, 1.0));
}
"""
        vs = fs = prog = 0
        try:
            vs = self._compile_shader(GL.GL_VERTEX_SHADER, vertex_src)
            fs = self._compile_shader(GL.GL_FRAGMENT_SHADER, fragment_src)
            prog = int(GL.glCreateProgram())
            GL.glAttachShader(prog, vs)
            GL.glAttachShader(prog, fs)
            GL.glLinkProgram(prog)
            ok = int(GL.glGetProgramiv(prog, GL.GL_LINK_STATUS))
            if not ok:
                raise RuntimeError(self._shader_info_log(prog, is_program=True))
            self.overlay_shader_program = prog
            self.overlay_shader_uniforms = {
                name: int(GL.glGetUniformLocation(prog, name))
                for name in ("u_tex", "u_texel", "u_radius", "u_alpha", "u_gain", "u_gamma", "u_min_alpha")
            }
            self.overlay_shader_error = ""
            print("[pygl_cuda_preview] event overlay GPU dilation shader ready", flush=True)
        except Exception as exc:
            self.overlay_shader_program = 0
            self.overlay_shader_error = f"{type(exc).__name__}: {exc}"
            print(f"[pygl_cuda_preview][WARN] event overlay GPU dilation shader unavailable; CPU fallback disabled for gl_shader: {self.overlay_shader_error}", flush=True)
            try:
                if prog:
                    GL.glDeleteProgram(prog)
            except Exception:
                pass
        finally:
            for shader in (vs, fs):
                if shader:
                    try:
                        GL.glDeleteShader(shader)
                    except Exception:
                        pass

    def _prepare_overlay_bgr(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        arr = np.asarray(frame)
        if arr.size <= 0:
            return np.zeros((1, 1), dtype=np.uint8), {
                "src_nonzero_pixels": 0,
                "display_nonzero_pixels": 0,
                "src_max_luma": 0,
                "preview_upload_channels": 1,
                "preview_texture_format": "luminance",
            }
        if arr.ndim == 2:
            img = arr if (arr.dtype == np.uint8 and arr.flags.c_contiguous) else np.ascontiguousarray(arr.astype(np.uint8, copy=False))
            src_luma = img
            upload_channels = 1
            texture_format = "luminance"
        else:
            if arr.shape[2] < 3:
                img = arr[:, :, 0]
                img = img if (img.dtype == np.uint8 and img.flags.c_contiguous) else np.ascontiguousarray(img.astype(np.uint8, copy=False))
                src_luma = img
                upload_channels = 1
                texture_format = "luminance"
            else:
                img = arr[:, :, :3]
                img = img if (img.dtype == np.uint8 and img.flags.c_contiguous) else np.ascontiguousarray(img.astype(np.uint8, copy=False))
                src_luma = np.max(img, axis=2)
                upload_channels = 3
                texture_format = "bgr"
        src_nonzero = int(np.count_nonzero(src_luma))
        src_max = int(np.max(src_luma)) if src_luma.size else 0
        requested_backend = self._requested_dilate_backend()
        active_backend = self._active_dilate_backend()
        requested_dilate_px = max(0, int(getattr(self.args, "event_overlay_preview_dilate_px", 4) or 0))
        effective_dilate_px = requested_dilate_px if active_backend == "gl_shader" else 0
        gain = max(0.01, float(getattr(self.args, "event_overlay_preview_gain", 3.0) or 3.0))
        gamma = max(0.05, float(getattr(self.args, "event_overlay_preview_gamma", 0.65) or 0.65))
        min_alpha = max(0.0, min(1.0, float(getattr(self.args, "event_overlay_preview_min_alpha", 0.35) or 0.0)))
        return np.ascontiguousarray(img), {
            "src_nonzero_pixels": int(src_nonzero),
            "display_nonzero_pixels": int(src_nonzero),
            "src_max_luma": int(src_max),
            "preview_upload_channels": int(upload_channels),
            "preview_texture_format": str(texture_format),
            "preview_gain": float(gain),
            "preview_gamma": float(gamma),
            "preview_min_alpha": float(min_alpha),
            "preview_dilate_requested_px": int(requested_dilate_px),
            "preview_dilate_px": int(effective_dilate_px),
            "preview_dilate_requested_backend": str(requested_backend),
            "preview_dilate_backend": str(active_backend),
            "preview_shader_ready": bool(int(self.overlay_shader_program or 0) > 0),
            "preview_shader_error": str(self.overlay_shader_error or ""),
        }

    def _prepare_overlay_rgba(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        arr = np.asarray(frame)
        if arr.size <= 0:
            return np.zeros((1, 1, 4), dtype=np.uint8), {"src_nonzero_pixels": 0, "display_nonzero_pixels": 0, "src_max_luma": 0}
        if arr.ndim == 2:
            bgr = np.repeat(arr[:, :, None], 3, axis=2)
        else:
            if arr.shape[2] < 3:
                bgr = np.repeat(arr[:, :, :1], 3, axis=2)
            else:
                bgr = arr[:, :, :3]
        bgr = bgr if (bgr.dtype == np.uint8 and bgr.flags.c_contiguous) else np.ascontiguousarray(bgr.astype(np.uint8, copy=False))
        src_luma = np.max(bgr, axis=2)
        src_nonzero = int(np.count_nonzero(src_luma))
        src_max = int(np.max(src_luma)) if src_luma.size else 0

        requested_backend = self._requested_dilate_backend()
        active_backend = self._active_dilate_backend()
        requested_dilate_px = max(0, int(getattr(self.args, "event_overlay_preview_dilate_px", 4) or 0))
        dilate_px = requested_dilate_px if active_backend == "cpu_sparse" else 0
        dilate_max_pixels = max(0, int(getattr(self.args, "event_overlay_preview_dilate_max_pixels", 5000) or 0))
        dilate_skipped_dense = False
        if dilate_px > 0 and src_nonzero > 0:
            # Sparse events need expansion to stay visible after tile downscale.  Full-frame
            # 9x9 dilation on eight 720p streams blocks the Qt event loop, so expand only
            # the nonzero event pixels and skip already-dense frames.
            if dilate_max_pixels > 0 and src_nonzero > dilate_max_pixels:
                dilate_skipped_dense = True
            else:
                ys, xs = np.nonzero(src_luma)
                if ys.size > 0:
                    h, w = bgr.shape[:2]
                    expanded = bgr.copy()
                    flat = expanded.reshape(-1, 3)
                    colors = np.ascontiguousarray(bgr[ys, xs, :])
                    for dy in range(-dilate_px, dilate_px + 1):
                        yy = np.clip(ys + dy, 0, h - 1)
                        row = yy * w
                        for dx in range(-dilate_px, dilate_px + 1):
                            xx = np.clip(xs + dx, 0, w - 1)
                            idx = row + xx
                            np.maximum.at(flat[:, 0], idx, colors[:, 0])
                            np.maximum.at(flat[:, 1], idx, colors[:, 1])
                            np.maximum.at(flat[:, 2], idx, colors[:, 2])
                    bgr = expanded

        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        luma = np.max(rgb, axis=2)
        gain = max(0.01, float(getattr(self.args, "event_overlay_preview_gain", 3.0) or 3.0))
        gamma = max(0.05, float(getattr(self.args, "event_overlay_preview_gamma", 0.65) or 0.65))
        min_alpha = max(0.0, min(1.0, float(getattr(self.args, "event_overlay_preview_min_alpha", 0.35) or 0.0)))
        enhanced = np.clip(rgb * gain, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            enhanced = np.power(enhanced, gamma)
        alpha = np.clip(luma * gain, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            alpha = np.power(alpha, gamma)
        alpha = np.where(luma > 0.0, np.maximum(alpha, min_alpha), 0.0)

        rgba = np.empty((bgr.shape[0], bgr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        display_nonzero = int(np.count_nonzero(rgba[:, :, 3]))
        return np.ascontiguousarray(rgba), {
            "src_nonzero_pixels": int(src_nonzero),
            "display_nonzero_pixels": int(display_nonzero),
            "src_max_luma": int(src_max),
            "preview_gain": float(gain),
            "preview_gamma": float(gamma),
            "preview_min_alpha": float(min_alpha),
            "preview_dilate_requested_px": int(requested_dilate_px),
            "preview_dilate_px": int(dilate_px),
            "preview_dilate_max_pixels": int(dilate_max_pixels),
            "preview_dilate_skipped_dense": bool(dilate_skipped_dense),
            "preview_dilate_requested_backend": str(requested_backend),
            "preview_dilate_backend": str(active_backend),
            "preview_shader_ready": bool(int(self.overlay_shader_program or 0) > 0),
            "preview_shader_error": str(self.overlay_shader_error or ""),
        }

    def _upload_texture(self, cam: str, frame: np.ndarray) -> Optional[Tuple[int, Dict[str, Any]]]:
        backend = self._active_dilate_backend()
        if backend == "gl_shader":
            img, diag = self._prepare_overlay_bgr(frame)
            if img.ndim == 2:
                pixel_format = int(getattr(GL, "GL_LUMINANCE", 0x1909))
                internal_format = pixel_format
            else:
                pixel_format = int(getattr(GL, "GL_BGR", GL.GL_RGB))
                internal_format = GL.GL_RGB8
        else:
            img, diag = self._prepare_overlay_rgba(frame)
            pixel_format = GL.GL_RGBA
            internal_format = GL.GL_RGBA8
        h, w = img.shape[:2]
        tex = self._ensure_texture(cam, w, h, int(internal_format), int(pixel_format))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if not bool(self.uploaded_once.get(cam, False)):
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal_format, w, h, 0, pixel_format, GL.GL_UNSIGNED_BYTE, img)
            self.uploaded_once[cam] = True
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, pixel_format, GL.GL_UNSIGNED_BYTE, img)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex, diag

    def _decision(self, cam: str, meta: Dict[str, Any], item: Dict[str, Any], rd: EventOverlayPreviewReader) -> Tuple[bool, bool, float, float, float, str]:
        mode = str(getattr(self.args, "event_overlay_sync_mode", "nearest_mvs") or "nearest_mvs").lower()
        now_ms = _now_ms()
        published_us = _safe_float(meta.get("published_wall_us", meta.get("event_wall_us", 0)), 0.0)
        if published_us > 1_000_000_000_000_000:
            age_ms = max(0.0, now_ms - published_us / 1000.0)
        else:
            try:
                age_ms = max(0.0, now_ms - os.path.getmtime(rd.sidecar_path) * 1000.0)
            except Exception:
                age_ms = -1.0
        max_age = float(getattr(self.args, "event_overlay_preview_max_age_ms", 100.0))
        max_delta = float(getattr(self.args, "event_overlay_max_sync_delta_ms", 50.0))
        dt_mvs = _safe_float(meta.get("event_to_mvs_delta_ms"), float('nan'))
        dt_human = _safe_float(meta.get("event_to_human_capture_delta_ms"), float('nan'))
        reason = ""
        draw = True
        stale = False
        if age_ms >= 0 and age_ms > max_age:
            stale = True
            reason = f"age>{max_age:.0f}ms"
        if mode == "latest":
            pass
        elif mode == "nearest_mvs":
            if np.isfinite(dt_mvs):
                if abs(float(dt_mvs)) > max_delta:
                    stale = True
                    reason = f"dt_mvs>{max_delta:.0f}ms"
            else:
                mvfid = _safe_int(meta.get("mvs_frame_id", -1), -1)
                curfid = _safe_int(self.preview.last_frame_ids.get(cam, -1), -1)
                if mvfid >= 0 and curfid >= 0 and abs(mvfid - curfid) > 2:
                    stale = True
                    reason = "mvs_frame_delta>2"
        elif mode == "nearest_human":
            if np.isfinite(dt_human):
                if abs(float(dt_human)) > max_delta:
                    stale = True
                    reason = f"dt_human>{max_delta:.0f}ms"
            else:
                hf = _safe_int(meta.get("human_frame_id", meta.get("human_mvs_frame_id", -1)), -1)
                try:
                    hrec = self.preview.overlay_state.snapshot().get("humans", {}).get(cam, {})
                    curhf = _safe_int(hrec.get("frame_id", -1), -1)
                    if hf >= 0 and curhf >= 0 and abs(hf - curhf) > 2:
                        stale = True
                        reason = "human_frame_delta>2"
                except Exception:
                    pass
        else:
            mode = "latest"
        if bool(getattr(self.args, "event_overlay_hide_stale", False)) and stale:
            draw = False
        return bool(draw), bool(stale), float(age_ms), float(dt_mvs) if np.isfinite(dt_mvs) else -1.0, float(dt_human) if np.isfinite(dt_human) else -1.0, reason

    def _poll_one(self, cam: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        rd = self.readers.get(cam)
        if rd is None:
            return None
        try:
            item = rd.read_latest()
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self.reader_errors[cam] = err
            self.preview.event_overlay_stats.setdefault(cam, {})["read_error"] = err
            return None
        if item is None:
            return None
        meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
        draw, stale, age_ms, dt_mvs, dt_human, reason = self._decision(cam, meta, item, rd)
        raw_n = _safe_int(meta.get("raw_event_count", 0), 0)
        filt_n = _safe_int(meta.get("filtered_event_count", 0), 0)
        mask_valid = bool(meta.get("mask_valid", False))
        seq = _safe_int(meta.get("seq", item.get("frame_id", -1)), -1)
        upload_diag: Dict[str, Any] = {}
        stat = {
            "drawn": bool(draw),
            "stale": bool(stale),
            "age_ms": float(age_ms),
            "dt_mvs_ms": float(dt_mvs),
            "dt_human_ms": float(dt_human),
            "raw": int(raw_n),
            "filtered": int(filt_n),
            "mask_valid": bool(mask_valid),
            "source_stage": str(meta.get("source_stage", "")),
            "seq": int(seq),
            "shm_frame_id": int(_safe_int(item.get("frame_id", meta.get("shm_frame_id", -1)), -1)),
            "meta_shm_frame_id": int(_safe_int(meta.get("shm_frame_id", -1), -1)),
            "meta_kind": str(meta.get("kind", "")),
            "reader_name": str(meta.get("_reader_name", getattr(rd, "name", ""))),
            "reader_path": str(meta.get("_reader_path", getattr(rd, "path", ""))),
            "reader_sidecar_path": str(meta.get("_reader_sidecar_path", getattr(rd, "sidecar_path", ""))),
            "overlay_nonzero_pixels": int(_safe_int(meta.get("overlay_nonzero_pixels", -1), -1)),
            "overlay_max_luma": int(_safe_int(meta.get("overlay_max_luma", -1), -1)),
            "reason": str(reason or ""),
            "mode": str(getattr(self.args, "event_overlay_sync_mode", "nearest_mvs")),
            "overview_backend": "opengl_texture",
        }
        tex = int(self.texture_ids.get(cam, 0) or 0)
        if item.get("new_frame", False) or tex <= 0 or self.last_seq.get(cam, -1) != seq:
            tex_new = self._upload_texture(cam, item["frame"])
            if tex_new is not None:
                tex = int(tex_new[0])
                upload_diag = dict(tex_new[1])
            self.last_seq[cam] = seq
        if upload_diag:
            stat.update(upload_diag)
        if stat.get("overlay_nonzero_pixels", -1) == 0 and stat.get("filtered", 0) > 0:
            stat["reason"] = "sidecar_nonzero=0"
        if stat.get("display_nonzero_pixels", -1) == 0 and stat.get("filtered", 0) > 0:
            stat["reason"] = "upload_alpha=0"
        self.preview.event_overlay_stats[cam] = stat
        if tex <= 0:
            return None
        return tex, stat

    def _draw_texture_quad(self, cam: str, tex: int, x0: float, y0: float, x1: float, y1: float, alpha: float) -> None:
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
        use_shader = self._active_dilate_backend() == "gl_shader" and int(self.overlay_shader_program or 0) > 0
        if use_shader:
            try:
                GL.glUseProgram(int(self.overlay_shader_program))
                uniforms = self.overlay_shader_uniforms
                if uniforms.get("u_tex", -1) >= 0:
                    GL.glUniform1i(uniforms["u_tex"], 0)
                shape = self.texture_shapes.get(cam, (1, 1, 0, 0))
                tw, th = int(shape[0]), int(shape[1])
                if uniforms.get("u_texel", -1) >= 0:
                    GL.glUniform2f(uniforms["u_texel"], 1.0 / max(1.0, float(tw)), 1.0 / max(1.0, float(th)))
                radius = max(0, min(8, int(getattr(self.args, "event_overlay_preview_dilate_px", 4) or 0)))
                if uniforms.get("u_radius", -1) >= 0:
                    GL.glUniform1i(uniforms["u_radius"], int(radius))
                if uniforms.get("u_alpha", -1) >= 0:
                    GL.glUniform1f(uniforms["u_alpha"], max(0.0, min(1.0, float(alpha))))
                if uniforms.get("u_gain", -1) >= 0:
                    GL.glUniform1f(uniforms["u_gain"], max(0.01, float(getattr(self.args, "event_overlay_preview_gain", 3.0) or 3.0)))
                if uniforms.get("u_gamma", -1) >= 0:
                    GL.glUniform1f(uniforms["u_gamma"], max(0.05, float(getattr(self.args, "event_overlay_preview_gamma", 0.65) or 0.65)))
                if uniforms.get("u_min_alpha", -1) >= 0:
                    GL.glUniform1f(uniforms["u_min_alpha"], max(0.0, min(1.0, float(getattr(self.args, "event_overlay_preview_min_alpha", 0.35) or 0.0))))
                GL.glColor4f(1.0, 1.0, 1.0, 1.0)
            except Exception as exc:
                self.overlay_shader_error = f"{type(exc).__name__}: {exc}"
                use_shader = False
                try:
                    GL.glUseProgram(0)
                except Exception:
                    pass
        GL.glColor4f(1.0, 1.0, 1.0, max(0.0, min(1.0, float(alpha))))
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(float(x0), float(y0))
        GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(float(x1), float(y0))
        GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(float(x1), float(y1))
        GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(float(x0), float(y1))
        GL.glEnd()
        if use_shader:
            GL.glUseProgram(0)

    def _draw_grid_border(self, x0: float, y0: float, x1: float, y1: float) -> None:
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glColor4f(0.28, 0.35, 0.43, 0.9)
        GL.glLineWidth(1.0)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex2f(float(x0), float(y0))
        GL.glVertex2f(float(x1), float(y0))
        GL.glVertex2f(float(x1), float(y1))
        GL.glVertex2f(float(x0), float(y1))
        GL.glEnd()
        GL.glEnable(GL.GL_TEXTURE_2D)

    def _draw_text_labels(self, label_stats: List[Tuple[str, int, int, int, int, Dict[str, Any]]]) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setFont(self.font_label)
            bg = QColor(0, 0, 0, 150)
            for cam, x, y, tw, _th, stat in label_stats:
                stale = bool(stat.get("stale", True))
                drawn = bool(stat.get("drawn", False))
                color = QColor(255, 220, 120) if stale else QColor(160, 255, 210)
                if not drawn:
                    color = QColor(210, 210, 210)
                label = (
                    f"{cam} {'DRAW' if drawn else 'NO'} {'STALE' if stale else 'OK'} "
                    f"mode={stat.get('mode', getattr(self.args, 'event_overlay_sync_mode', 'nearest_mvs'))} "
                    f"age={_safe_float(stat.get('age_ms'), -1.0):.0f}ms "
                    f"raw={stat.get('raw', 0)} filt={stat.get('filtered', 0)} "
                    f"nz={stat.get('display_nonzero_pixels', stat.get('overlay_nonzero_pixels', '-'))} "
                    f"dil={stat.get('preview_dilate_backend', '-')}/{stat.get('preview_dilate_px', '-')} "
                    f"max={stat.get('src_max_luma', stat.get('overlay_max_luma', '-'))} "
                    f"fid={stat.get('shm_frame_id', '-')}/{stat.get('meta_shm_frame_id', '-')} "
                    f"kind={stat.get('meta_kind', '')}"
                )
                text_w = min(max(80, tw - 16), max(80, painter.fontMetrics().horizontalAdvance(label) + 12))
                painter.fillRect(x + 4, y + 6, int(text_w), 20, bg)
                painter.setPen(color)
                painter.drawText(x + 8, y + 22, painter.fontMetrics().elidedText(label, Qt.ElideRight, max(20, int(text_w) - 8)))
                reason = str(stat.get("reason", "") or self.reader_errors.get(cam, ""))
                if reason:
                    text_w2 = min(max(80, tw - 16), max(80, painter.fontMetrics().horizontalAdvance(reason) + 12))
                    painter.fillRect(x + 4, y + 28, int(text_w2), 20, bg)
                    painter.setPen(QColor(255, 160, 120))
                    painter.drawText(x + 8, y + 44, painter.fontMetrics().elidedText(reason, Qt.ElideRight, max(20, int(text_w2) - 8)))
        finally:
            painter.end()

    def paintGL(self) -> None:
        paint_start = time.perf_counter()
        if self._last_paint_perf > 0.0:
            self.paint_interval_ms = (paint_start - self._last_paint_perf) * 1000.0
        self._last_paint_perf = paint_start
        self._fps_window_count += 1
        if paint_start - self._fps_window_start >= 1.0:
            self.paint_fps_current = self._fps_window_count / max(1e-6, paint_start - self._fps_window_start)
            self._fps_window_count = 0
            self._fps_window_start = paint_start
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if not bool(getattr(self.args, "event_overlay_preview_enable", False)):
            painter = QPainter(self)
            try:
                painter.setFont(self.font_label)
                painter.setPen(QColor(220, 220, 220))
                painter.drawText(16, 32, "Event overlay overview disabled. Enable it in the status window or launch with --event-overlay-preview-enable.")
            finally:
                painter.end()
            return

        self._try_open_readers()
        w = max(1, self.width())
        h = max(1, self.height())
        tile_w = w / max(1, self.cols)
        tile_h = h / max(1, self.rows)
        base_alpha = float(getattr(self.args, "event_overlay_preview_alpha", 0.55))
        stale_alpha_scale = float(getattr(self.args, "event_overlay_stale_alpha_scale", 0.25))
        label_stats: List[Tuple[str, int, int, int, int, Dict[str, Any]]] = []

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        for i, cam in enumerate(self.cams):
            col = i % self.cols
            row = i // self.cols
            x = int(round(col * tile_w))
            y = int(round(row * tile_h))
            tw = int(round(tile_w))
            th = int(round(tile_h))
            x0 = -1.0 + 2.0 * (col / max(1, self.cols))
            x1 = -1.0 + 2.0 * ((col + 1) / max(1, self.cols))
            y0 = 1.0 - 2.0 * ((row + 1) / max(1, self.rows))
            y1 = 1.0 - 2.0 * (row / max(1, self.rows))

            item = self._poll_one(cam)
            if item is not None:
                tex, stat = item
                if stat.get("drawn", False):
                    alpha = base_alpha * (stale_alpha_scale if stat.get("stale", True) else 1.0)
                    self._draw_texture_quad(cam, tex, x0, y0, x1, y1, alpha)
            else:
                stat = self.preview.event_overlay_stats.get(cam, {})
            self._draw_grid_border(x0, y0, x1, y1)
            label_stats.append((cam, x, y, tw, th, stat))
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

        if bool(getattr(self.args, "event_overlay_show_stats", True)):
            now_ms = _now_ms()
            if self.label_interval_ms <= 0.0 or not self._cached_label_stats or now_ms - self._last_label_update_ms >= self.label_interval_ms:
                self._cached_label_stats = label_stats
                self._last_label_update_ms = now_ms
            self._draw_text_labels(self._cached_label_stats)
        self.paint_ms_current = (time.perf_counter() - paint_start) * 1000.0

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.makeCurrent()
            texs = [int(t) for t in self.texture_ids.values() if int(t) > 0]
            if texs:
                GL.glDeleteTextures(texs)
            if int(self.overlay_shader_program or 0) > 0:
                GL.glDeleteProgram(int(self.overlay_shader_program))
                self.overlay_shader_program = 0
            self.doneCurrent()
        except Exception:
            pass
        return super().closeEvent(event)


def _screen_summary(app: QApplication) -> str:
    try:
        parts = []
        for i, sc in enumerate(app.screens()):
            g = sc.availableGeometry()
            parts.append(f"{i}:{sc.name()} avail={g.x()},{g.y()},{g.width()}x{g.height()}")
        return "; ".join(parts) or "<no screens>"
    except Exception as exc:
        return f"<screen query failed: {type(exc).__name__}: {exc}>"


def _place_window(widget: QWidget, app: QApplication, *, preferred_x: int, preferred_y: int,
                  preferred_w: int, preferred_h: int, label: str) -> None:
    """Force a sane on-screen geometry for a top-level window."""
    try:
        screen = app.primaryScreen() or (app.screens()[0] if app.screens() else None)
        if screen is None:
            widget.resize(max(240, int(preferred_w)), max(160, int(preferred_h)))
            widget.move(max(0, int(preferred_x)), max(0, int(preferred_y)))
            print(f"[pygl_cuda_preview] {label} window geometry without screen: {widget.geometry().getRect()}", flush=True)
            return
        g = screen.availableGeometry()
        w = max(240, min(int(preferred_w), max(240, int(g.width()))))
        h = max(160, min(int(preferred_h), max(160, int(g.height()))))
        x = int(preferred_x)
        y = int(preferred_y)
        if x < -99999:
            x = int(g.x())
        if y < -99999:
            y = int(g.y())
        x = max(int(g.x()), min(x, int(g.x() + max(0, g.width() - w))))
        y = max(int(g.y()), min(y, int(g.y() + max(0, g.height() - h))))
        widget.setGeometry(x, y, w, h)
        print(f"[pygl_cuda_preview] {label} window geometry={x},{y},{w}x{h} screen={screen.name()} avail={g.x()},{g.y()},{g.width()}x{g.height()}", flush=True)
    except Exception as exc:
        try:
            widget.resize(max(240, int(preferred_w)), max(160, int(preferred_h)))
            widget.move(max(0, int(preferred_x)), max(0, int(preferred_y)))
        except Exception:
            pass
        print(f"[pygl_cuda_preview][WARN] failed to place {label} window: {type(exc).__name__}: {exc}", flush=True)


def _show_and_raise(widget: QWidget, *, label: str, force_raise: bool = True) -> None:
    try:
        widget.show()
        widget.showNormal()
        if force_raise:
            widget.raise_()
            widget.activateWindow()
        print(f"[pygl_cuda_preview] {label} window shown visible={widget.isVisible()} geom={widget.geometry().getRect()}", flush=True)
    except Exception as exc:
        print(f"[pygl_cuda_preview][ERROR] failed to show {label} window: {type(exc).__name__}: {exc}", flush=True)



def _json_safe(v: Any) -> Any:
    """Convert common numpy/Qt/path values into JSON-serializable values."""
    try:
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {str(_json_safe(k)): _json_safe(val) for k, val in v.items()}
        if isinstance(v, (list, tuple, set, deque)):
            return [_json_safe(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


class StatusLogWriter:
    """Low-overhead status diagnostics logger for offline analysis.

    The logger is intentionally attached to the separate status window and only
    samples already-collected diagnostics.  It does not touch the video upload,
    trajectory, human-mask, or hit rendering path.
    """

    SCHEMA_VERSION = 1

    def __init__(self, preview: Any, args: argparse.Namespace):
        self.preview = preview
        self.args = args
        self.enabled = bool(getattr(args, "status_log_enable", False))
        self.interval_ms = max(100.0, float(getattr(args, "status_log_interval_ms", 1000.0)))
        self.tabs_jsonl = bool(getattr(args, "status_log_tabs_jsonl", False))
        self.latest_enable = not bool(getattr(args, "status_log_latest_disable", False))
        self.records = 0
        self.last_write_ms = 0.0
        self.last_error = ""
        self.run_id = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        self.run_dir = ""
        self.snapshot_path = ""
        self.tabs_path = ""
        self.csv_path = ""
        self.latest_path = ""
        self.system_latest_path = ""
        self._snapshot_fp = None
        self._tabs_fp = None
        self._csv_fp = None
        self._csv_writer = None
        self._csv_fields: List[str] = []
        self._global_yolo_prev_counters: Dict[str, Tuple[int, int, float]] = {}
        try:
            sync_root = str(getattr(args, "sync_root", "sync_ipc") or "sync_ipc")
            root = Path(sync_root).expanduser()
            if not root.is_absolute():
                root = Path.cwd() / root
            self.system_latest_path = str(root / "system_status_latest.json")
        except Exception:
            self.system_latest_path = "system_status_latest.json"
        if not self.enabled:
            return
        try:
            base = str(getattr(args, "status_log_dir", "") or "").strip()
            if not base:
                sync_root = str(getattr(args, "sync_root", "sync_ipc") or "sync_ipc")
                base = os.path.join(sync_root, "status_logs")
            pp = Path(base).expanduser()
            if not pp.is_absolute():
                pp = Path.cwd() / pp
            self.run_dir = str(pp / self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)
            self.snapshot_path = os.path.join(self.run_dir, "status_snapshot.jsonl")
            self.tabs_path = os.path.join(self.run_dir, "status_tabs.jsonl")
            self.csv_path = os.path.join(self.run_dir, "status_summary.csv")
            self.latest_path = os.path.join(self.run_dir, "status_latest.json")
            self._snapshot_fp = open(self.snapshot_path, "a", encoding="utf-8", buffering=1)
            if self.tabs_jsonl:
                self._tabs_fp = open(self.tabs_path, "a", encoding="utf-8", buffering=1)
            self._csv_fields = self._build_csv_fields()
            self._csv_fp = open(self.csv_path, "a", encoding="utf-8", newline="", buffering=1)
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=self._csv_fields, extrasaction="ignore")
            if self._csv_fp.tell() == 0:
                self._csv_writer.writeheader()
            self._write_readme()
            print(f"[pygl_cuda_preview] status diagnostics logging enabled dir={self.run_dir}", flush=True)
        except Exception as exc:
            self.enabled = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[pygl_cuda_preview][WARN] status diagnostics logging disabled: {self.last_error}", flush=True)

    def _build_csv_fields(self) -> List[str]:
        fields = [
            "ts_wall_us", "mono_ns", "sample_id", "trigger_state", "trigger_fps",
            "trigger_period_ms", "trigger_jitter_p95_ms", "trigger_last_age_ms", "ok_cams",
            "mvs_active_count", "event_active_count", "overlay_ws_connected", "overlay_ws_rx_count",
            "overlay_ws_age_ms", "human_jsonl_rx_count", "hit_jsonl_rx_count",
            "yolo_infer_fps", "yolo_capacity_fps", "yolo_fps_source", "yolo_worker_count",
            "yolo_published_fps_total", "yolo_inferred_fps_total",
            "yolo_async_record_postprocess_enabled", "yolo_postprocess_queue_depth",
            "yolo_postprocess_dropped_batches", "yolo_polygon_contract_ok",
            "yolo_published_persons_with_polygon_ratio",
            "yolo_batch_size", "yolo_predict_ms", "yolo_capture_to_result_ms",
            "hit_udp_recv_per_s", "hit_eval_per_s", "hit_candidate_per_s", "hit_confirmed_per_s",
            "hit_no_human_per_s", "hit_no_human_count", "hit_pending_count", "hit_pending_impacts",
            "render_fps", "renderer_paint_ms", "frame_period_ms", "tick_interval_ms", "paint_interval_ms",
            "trajectory_drawn_vtx", "bullet_udp_recv_per_s", "bullet_track_count_active",
            "human_mask_drawn", "human_mask_source", "human_mask_count", "human_mask_person_count",
            "human_mask_polygon_fill_count", "human_mask_bbox_fill_count", "human_mask_latest_age_ms",
            "bullet_latest_age_ms", "bullet_latest_cam", "bullet_latest_id", "bullet_status_ttl_ms",
            "bullet_fresh_point_count", "bullet_stale_point_count", "bullet_future_point_count",
            "bullet_stale_drop_count",
            "trajectory_stale_guard_count", "trajectory_future_drop_count",
            "trajectory_fresh_not_drawn", "trajectory_draw_state", "sync_latest_cam",
            "sync_bullet_id", "event_to_mvs_delta_ms", "event_to_human_capture_delta_ms",
            "event_to_human_result_age_ms", "event_to_human_delta_frames", "sync_quality",
            "latest_hit_reason", "poll_error_count",
            "video_backend_state", "video_backend_stage", "video_backend_error_count",
            "video_backend_attempts", "video_backend_retries",
            "video_timer_started", "render_timer_started", "overlay_timer_started",
            "last_video_tick_error", "last_render_tick_error", "last_paint_error",
        ]
        fields.extend([
            "raw_record_state",
            "raw_recording",
            "raw_record_session_id",
            "raw_record_last_command",
            "raw_record_last_command_targets",
            "raw_record_command_age_ms",
            "raw_record_mvs_status_mode",
            "raw_record_mvs_status_count",
            "raw_record_mvs_expected_status_count",
            "raw_record_mvs_recording",
            "raw_record_mvs_written_frames_total",
            "raw_record_mvs_dropped_frames_total",
            "raw_record_mvs_queue_size_total",
            "raw_record_event_recording_count",
            "raw_record_event_status_count",
            "raw_record_event_slices_written_total",
            "raw_record_event_events_written_total",
            "raw_record_event_dropped_slices_total",
            "raw_record_event_queue_size_total",
            "raw_record_degraded_links",
            "evidence_replay_state",
            "evidence_replay_pack_dir",
            "evidence_replay_run_dir",
            "evidence_replay_diff_state",
            "evidence_replay_original_hit_count",
            "evidence_replay_replay_hit_count",
            "evidence_replay_global_person_track_count",
            "experiment_state",
            "experiment_id",
            "experiment_stage",
            "experiment_config_hash",
            "experiment_source",
            "experiment_status_age_ms",
            "experiment_final_hit_authority",
            "experiment_human_noise_filter_mode",
            "experiment_bbox_start_expand_px",
            "experiment_global_person_target",
            "experiment_global_person_filter_backend",
            "experiment_run_state",
            "experiment_runtime_overall_state",
        ])
        for svc in ("global_bullet_fusion", "damage_attribution", "manual_hit", "foam_dart_speed", "game_event_bridge"):
            fields.extend([
                f"{svc}_state",
                f"{svc}_mode",
                f"{svc}_enabled",
                f"{svc}_input_age_ms",
                f"{svc}_output_age_ms",
                f"{svc}_queue_size",
                f"{svc}_dropped_count",
                f"{svc}_last_error",
            ])
        cams = list(getattr(self.preview, "cams", []) or [])
        for cam in cams:
            fields.extend([
                f"{cam}_raw_fps", f"{cam}_mvs_fps", f"{cam}_raw_age_ms", f"{cam}_frame_age_ms",
                f"{cam}_raw_missing", f"{cam}_mvs_missing", f"{cam}_pending_q", f"{cam}_human_age_ms",
                f"{cam}_yolo_delta_frames", f"{cam}_persons", f"{cam}_event_ready",
                f"{cam}_event_trigger_fps", f"{cam}_event_rate_mevs", f"{cam}_event_trigger_age_ms",
                f"{cam}_event_triggers", f"{cam}_event_nonmono", f"{cam}_event_guard",
                f"{cam}_physical_projector_valid", f"{cam}_physical_person_count",
                f"{cam}_physical_first_id", f"{cam}_physical_first_x_m", f"{cam}_physical_first_y_m",
                f"{cam}_evt_ovl_drawn", f"{cam}_evt_ovl_stale", f"{cam}_evt_ovl_age_ms",
                f"{cam}_evt_ovl_dt_mvs_ms", f"{cam}_evt_ovl_dt_human_ms",
                f"{cam}_evt_ovl_raw", f"{cam}_evt_ovl_filtered", f"{cam}_evt_ovl_mode",
            ])
        return fields

    def _write_readme(self) -> None:
        try:
            p = os.path.join(self.run_dir, "README_STATUS_LOGS.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("PyOpenGL CUDA status diagnostics log\n")
                f.write(f"schema_version={self.SCHEMA_VERSION}\n")
                f.write(f"run_id={self.run_id}\n")
                f.write("\nFiles:\n")
                f.write("  status_snapshot.jsonl : machine-readable full telemetry sample per row\n")
                f.write("  status_summary.csv    : flat key metrics for spreadsheet/offline plotting\n")
                f.write("  status_latest.json    : last sample overwritten atomically\n")
                f.write("  ../system_status_latest.json : unified full-pipeline status snapshot for UI/scripts\n")
                if self.tabs_jsonl:
                    f.write("  status_tabs.jsonl     : UI tab text rows per sample\n")
                f.write("\nThis logger is read-only and samples the diagnostics layer only.\n")
        except Exception:
            pass

    def _overlay_summary(self) -> Dict[str, Any]:
        try:
            snap = self.preview.overlay_state.snapshot(
                marker_ttl_ms=float(getattr(self.args, "overlay_marker_life_ms", 3000.0)),
                text_ttl_ms=float(getattr(self.args, "overlay_text_life_ms", 3200.0)),
            )
        except Exception:
            snap = {}
        humans = snap.get("humans", {}) if isinstance(snap, dict) else {}
        human_meta: Dict[str, Dict[str, Any]] = {}
        if isinstance(humans, dict):
            for cam, rec in humans.items():
                if not isinstance(rec, dict):
                    continue
                proj = rec.get("projector", {}) if isinstance(rec.get("projector", {}), dict) else {}
                persons = rec.get("persons", []) if isinstance(rec.get("persons", []), list) else []
                first_id = ""
                first_x = None
                first_y = None
                for person in persons:
                    if not isinstance(person, dict):
                        continue
                    pc = person.get("physical_coord") if isinstance(person.get("physical_coord"), dict) else {}
                    if pc and pc.get("ok", False):
                        first_x = pc.get("x_m")
                        first_y = pc.get("y_m")
                    elif isinstance(person.get("local_xy"), (list, tuple)) and len(person.get("local_xy")) >= 2:
                        first_x = person.get("local_xy")[0]
                        first_y = person.get("local_xy")[1]
                    if first_x is not None and first_y is not None:
                        first_id = str(person.get("stable_id", person.get("person_id", "")))
                        break
                human_meta[str(cam)] = {
                    "frame_id": rec.get("frame_id"),
                    "person_count": rec.get("person_count"),
                    "ts_ms": rec.get("ts_ms"),
                    "capture_wall_us": rec.get("capture_wall_us"),
                    "record_wall_us": rec.get("record_wall_us"),
                    "yolo_batch_size": rec.get("yolo_batch_size"),
                    "yolo_predict_ms": rec.get("yolo_predict_ms"),
                    "yolo_capture_to_result_ms": rec.get("yolo_capture_to_result_ms"),
                    "projector_valid": bool(proj.get("valid", False)) if proj else (first_x is not None and first_y is not None),
                    "physical_first_id": first_id,
                    "physical_first_x_m": first_x,
                    "physical_first_y_m": first_y,
                }
        return {
            "ws_connected": bool(snap.get("ws_connected", False)) if isinstance(snap, dict) else False,
            "ws_rx_count": _safe_int(snap.get("ws_rx_count", 0) if isinstance(snap, dict) else 0, 0),
            "ws_age_ms": _safe_float(snap.get("ws_age_ms", -1.0) if isinstance(snap, dict) else -1.0, -1.0),
            "human_jsonl_rx_count": _safe_int(snap.get("human_jsonl_rx_count", 0) if isinstance(snap, dict) else 0, 0),
            "hit_jsonl_rx_count": _safe_int(snap.get("hit_jsonl_rx_count", 0) if isinstance(snap, dict) else 0, 0),
            "human_meta": human_meta,
            "counts": {
                "humans": len(human_meta),
                "bullet_points": len(snap.get("bullet_points", [])) if isinstance(snap, dict) else 0,
                "hit_decisions": len(snap.get("hit_decisions", [])) if isinstance(snap, dict) else 0,
                "markers": len(snap.get("markers", [])) if isinstance(snap, dict) else 0,
                "texts": len(snap.get("texts", [])) if isinstance(snap, dict) else 0,
            },
        }

    def _renderer_summary(self) -> Dict[str, Any]:
        fps = max(0.0, _safe_float(getattr(self.preview, "render_fps_current", 0.0), 0.0))
        frame_period_ms = (1000.0 / fps) if fps > 0.001 else -1.0
        if hasattr(self.preview, "get_effective_event_overlay_stats"):
            event_overlay = self.preview.get_effective_event_overlay_stats()
        else:
            event_overlay = getattr(self.preview, "event_overlay_stats", {})
        if not isinstance(event_overlay, dict):
            event_overlay = {}
        backend = getattr(self.preview, "video_backend", None)
        backend_info = {
            "state": str(getattr(self.preview, "video_backend_state", "unknown")),
            "stage": str(getattr(self.preview, "video_backend_init_stage", "")),
            "error": str(getattr(self.preview, "video_backend_error", "")),
            "trace_tail": str(getattr(self.preview, "video_backend_error_trace", ""))[-3000:],
            "error_count": _safe_int(getattr(self.preview, "video_backend_error_count", 0), 0),
            "attempts": _safe_int(getattr(self.preview, "video_backend_attempts", 0), 0),
            "retries": _safe_int(getattr(self.preview, "video_backend_retry_count", 0), 0),
            "backend_class": type(backend).__name__ if backend is not None else "",
            "video_timer_started": bool(getattr(self.preview, "video_timer_started", False)),
            "render_timer_started": bool(getattr(self.preview, "render_timer_started", False)),
            "overlay_timer_started": bool(getattr(self.preview, "overlay_timer_started", False)),
            "last_video_tick_error": str(getattr(self.preview, "last_video_tick_error", "")),
            "last_render_tick_error": str(getattr(self.preview, "last_render_tick_error", "")),
            "last_paint_error": str(getattr(self.preview, "last_paint_error", "")),
        }
        telemetry_bullet = getattr(getattr(self.preview, "telemetry", None), "bullet", {}) or {}
        drawn_vtx = _safe_int(getattr(self.preview, "trajectory_vertex_count", 0), 0)
        fresh_points = _safe_int(telemetry_bullet.get("fresh_point_count", 0), 0) if isinstance(telemetry_bullet, dict) else 0
        stale_points = _safe_int(telemetry_bullet.get("stale_point_count", 0), 0) if isinstance(telemetry_bullet, dict) else 0
        future_points = _safe_int(telemetry_bullet.get("future_point_count", 0), 0) if isinstance(telemetry_bullet, dict) else 0
        stale_guard = _safe_int(getattr(self.preview, "trajectory_stale_guard_count", 0), 0)
        future_drop = _safe_int(getattr(self.preview, "trajectory_future_drop_count", 0), 0)
        reader_open = getattr(self.preview, "trajectory_reader", None) is not None
        reader_error = str(getattr(self.preview, "trajectory_reader_error", ""))
        try:
            with getattr(self.preview, "trajectory_suppression_lock"):
                suppressed_track_count = len(getattr(self.preview, "trajectory_suppressed_keys", {}) or {})
        except Exception:
            suppressed_track_count = 0
        pseudo_tailer = getattr(self.preview, "pseudo_hit_tailer", None)
        if drawn_vtx > 0:
            trajectory_draw_state = "drawing"
        elif not reader_open:
            trajectory_draw_state = "reader_closed"
        elif reader_error:
            trajectory_draw_state = "reader_error"
        elif fresh_points > 0:
            trajectory_draw_state = "fresh_not_drawn"
        elif future_points > 0 or future_drop > 0:
            trajectory_draw_state = "future_points"
        elif stale_points > 0 or stale_guard > 0:
            trajectory_draw_state = "stale_points"
        else:
            trajectory_draw_state = "idle"
        return {
            "render_fps": fps,
            "renderer_paint_ms": _safe_float(getattr(self.preview, "renderer_paint_ms", -1.0), -1.0),
            "renderer_tick_ms": _safe_float(getattr(self.preview, "renderer_tick_ms", -1.0), -1.0),
            "tick_interval_ms": _safe_float(getattr(self.preview, "renderer_tick_interval_ms", -1.0), -1.0),
            "paint_interval_ms": _safe_float(getattr(self.preview, "renderer_paint_interval_ms", -1.0), -1.0),
            "frame_period_ms": frame_period_ms,
            "target_period_ms": 1000.0 / max(1e-3, _safe_float(getattr(self.args, "render_fps", 250.0), 250.0)),
            "trajectory_drawn_vtx": drawn_vtx,
            "trajectory_stale_guard_count": stale_guard,
            "trajectory_future_drop_count": future_drop,
            "trajectory_fresh_not_drawn": int(fresh_points if drawn_vtx <= 0 else 0),
            "trajectory_draw_state": trajectory_draw_state,
            "trajectory_filter_human_noise": {
                "enable": bool(getattr(self.args, "preview_trajectory_filter_human_noise", True)),
                "suppressed_track_count": int(suppressed_track_count),
                "suppressed_total": _safe_int(getattr(self.preview, "trajectory_suppressed_total", 0), 0),
                "filtered_segment_count": _safe_int(getattr(self.preview, "trajectory_filter_segment_count", 0), 0),
                "pseudo_hit_tailer_open": pseudo_tailer is not None,
                "pseudo_hit_tailer_rx": _safe_int(getattr(pseudo_tailer, "rx_count", 0), 0) if pseudo_tailer is not None else 0,
                "pseudo_hit_tailer_suppressed_rx": _safe_int(getattr(pseudo_tailer, "suppressed_rx_count", 0), 0) if pseudo_tailer is not None else 0,
                "pseudo_hit_tailer_error": str(getattr(pseudo_tailer, "error", "")) if pseudo_tailer is not None else "",
            },
            "bullet_fresh_point_count": fresh_points,
            "bullet_stale_point_count": stale_points,
            "bullet_future_point_count": future_points,
            "trajectory_reader": {
                "open": reader_open,
                "error": reader_error,
                "reopen_count": _safe_int(getattr(self.preview, "trajectory_reader_reopen_count", 0), 0),
                "life_ms": _safe_float(getattr(self.args, "trajectory_life_ms", 0.0), 0.0),
            },
            "human_jsonl_backfill": {
                "count": _safe_int(getattr(self.preview, "human_jsonl_backfill_count", 0), 0),
                "last_age_ms": max(0.0, _now_ms() - _safe_float(getattr(self.preview, "human_jsonl_backfill_last_ms", 0.0), 0.0)) if _safe_float(getattr(self.preview, "human_jsonl_backfill_last_ms", 0.0), 0.0) > 0 else -1.0,
                "error": str(getattr(self.preview, "human_jsonl_backfill_error", "")),
            },
            "human_mask_render": getattr(self.preview, "human_mask_render_stats", {}),
            "manual_hit": {
                "click_enable": bool(getattr(self.preview, "manual_hit_click_enable", False)),
                "preview_hit_display_enable": bool(getattr(self.preview, "preview_hit_display_enable", True)),
                "trajectory_only_mode": bool(getattr(self.preview, "preview_trajectory_only_mode", False)),
                "target_cache_count": len(getattr(self.preview, "manual_hit_targets", []) or []),
                "command_seq": _safe_int(getattr(self.preview, "manual_hit_command_seq", 0), 0),
                "last_command": dict(getattr(self.preview, "manual_hit_last_result", {}) or {}),
                "preview_control_error": str(getattr(self.preview, "preview_control_error", "")),
                "command_jsonl": str(getattr(self.preview, "_manual_hit_command_path")()) if hasattr(self.preview, "_manual_hit_command_path") else "",
            },
            "event_overlay": event_overlay,
            "video_backend": backend_info,
        }

    def _sync_root_path(self, filename: str) -> Path:
        p = Path(str(filename or "")).expanduser()
        if p.is_absolute():
            return p
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / p

    def _read_json_status(self, filename: str) -> Dict[str, Any]:
        path = self._sync_root_path(filename)
        try:
            if hasattr(getattr(self.preview, "telemetry", None), "_best_existing_path") and not Path(str(filename)).is_absolute():
                path = Path(self.preview.telemetry._best_existing_path(str(filename), prefer_json_wall=True))
        except Exception:
            pass
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
        try:
            mtime_ms = os.path.getmtime(path) * 1000.0 if path.exists() else 0.0
        except Exception:
            mtime_ms = 0.0
        obj["_status_path"] = str(path)
        obj["_file_age_ms"] = max(0.0, _now_ms() - mtime_ms) if mtime_ms > 0 else -1.0
        obj["_file_exists"] = bool(path.exists())
        return obj

    def _resolve_sync_or_project_path(self, value: str, default_name: str) -> Path:
        raw = str(value or default_name or "").strip()
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p
        normalized = raw.replace("\\", "/")
        if normalized.startswith("sync_ipc/"):
            return Path.cwd() / p
        return self._sync_root_path(raw or default_name)

    def _read_jsonl_tail_records(self, path: Path, max_records: int = 80, tail_bytes: int = 512_000) -> Tuple[List[Dict[str, Any]], str, int, float]:
        records: List[Dict[str, Any]] = []
        error = ""
        size = 0
        file_age_ms = -1.0
        try:
            if not path.exists():
                return [], "missing", 0, -1.0
            stat = path.stat()
            size = int(stat.st_size)
            file_age_ms = max(0.0, _now_ms() - float(stat.st_mtime) * 1000.0)
            with path.open("r", encoding="utf-8", errors="replace") as fp:
                start = max(0, size - int(max(8192, tail_bytes)))
                fp.seek(start)
                if start > 0:
                    fp.readline()
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except Exception:
                        continue
            if len(records) > max_records:
                records = records[-max_records:]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return records, error, size, file_age_ms

    def _status_wall_age_ms(self, obj: Dict[str, Any]) -> float:
        if not isinstance(obj, dict):
            return -1.0
        now_us = int(time.time() * 1_000_000)
        for key in (
            "ts_wall_us",
            "timestamp_us",
            "wall_time_us",
            "wall_us",
            "time_wall_us",
            "update_wall_us",
            "updated_wall_us",
            "updated_us",
        ):
            val = _safe_float(obj.get(key), 0.0)
            if val > 1_000_000_000_000_000:
                return max(0.0, (now_us - val) / 1000.0)
        return _safe_float(obj.get("_file_age_ms"), -1.0)

    def _build_mvs_status(self, telem: Dict[str, Any]) -> Dict[str, Any]:
        raw_age = telem.get("raw_age_ms", {}) if isinstance(telem, dict) else {}
        frame_age = telem.get("frame_age_ms", {}) if isinstance(telem, dict) else {}
        raw_missing = telem.get("raw_missing", {}) if isinstance(telem, dict) else {}
        mvs_missing = telem.get("mvs_missing", {}) if isinstance(telem, dict) else {}
        raw_fps = telem.get("raw_fps", {}) if isinstance(telem, dict) else {}
        mvs_fps = telem.get("mvs_fps", {}) if isinstance(telem, dict) else {}
        per_cam: Dict[str, Any] = {}
        stuck: List[str] = []
        for cam in list(getattr(self.preview, "cams", []) or []):
            fid = _safe_int(getattr(self.preview, "last_frame_ids", {}).get(cam, -1), -1)
            r_age = _safe_float(raw_age.get(cam), -1.0) if isinstance(raw_age, dict) else -1.0
            f_age = _safe_float(frame_age.get(cam), -1.0) if isinstance(frame_age, dict) else -1.0
            missing = bool(raw_missing.get(cam, True) if isinstance(raw_missing, dict) else True) or bool(mvs_missing.get(cam, True) if isinstance(mvs_missing, dict) else True)
            is_stuck = bool(missing or r_age > 250.0 or f_age > 250.0)
            if is_stuck:
                stuck.append(cam)
            per_cam[cam] = {
                "frame_id": fid,
                "frame_age_ms": f_age,
                "shm_age_ms": r_age,
                "raw_fps": _safe_float(raw_fps.get(cam), 0.0) if isinstance(raw_fps, dict) else 0.0,
                "mvs_fps": _safe_float(mvs_fps.get(cam), 0.0) if isinstance(mvs_fps, dict) else 0.0,
                "raw_missing": bool(raw_missing.get(cam, True)) if isinstance(raw_missing, dict) else True,
                "mvs_missing": bool(mvs_missing.get(cam, True)) if isinstance(mvs_missing, dict) else True,
                "stuck": is_stuck,
            }
        return {
            "title_cn": "MVS取流/软触发",
            "state": "OK" if not stuck else "DEGRADED",
            "active_count": _safe_int(telem.get("mvs_active_count", 0) if isinstance(telem, dict) else 0, 0),
            "stuck_cameras": stuck,
            "per_camera": per_cam,
        }

    def _build_event_status(self, telem: Dict[str, Any], renderer: Dict[str, Any]) -> Dict[str, Any]:
        ev = telem.get("event", {}) if isinstance(telem, dict) else {}
        overlay = renderer.get("event_overlay", {}) if isinstance(renderer, dict) else {}
        overview = self._read_json_status("event_overlay_overview_status.json")
        overview_rows = overview.get("event_overlay", {}) if isinstance(overview.get("event_overlay"), dict) else {}
        overview_age = self._status_wall_age_ms(overview)
        overview_ok = bool(overview.get("_file_exists")) and (overview_age < 0 or overview_age <= 3000.0)
        mask_evidence_status = self._read_json_status("event_mask_evidence_status.json")
        mask_evidence_age = self._status_wall_age_ms(mask_evidence_status)
        mask_evidence_ok = bool(mask_evidence_status.get("_file_exists")) and mask_evidence_age >= 0 and mask_evidence_age <= 3000.0
        mask_evidence_rows = mask_evidence_status.get("per_camera", {}) if isinstance(mask_evidence_status.get("per_camera"), dict) else {}
        broker_rows, broker_tail_error, _broker_size, broker_age_ms = self._read_jsonl_tail_records(
            self._sync_root_path("event_native_hal_broker_stats.jsonl"),
            max_records=1,
            tail_bytes=256_000,
        )
        broker_latest = broker_rows[-1] if broker_rows else {}
        broker_visual_rows: Dict[str, Any] = {}
        if isinstance(broker_latest.get("cameras"), list):
            for item in broker_latest.get("cameras") or []:
                if not isinstance(item, dict):
                    continue
                cam_name = str(item.get("alias", "") or item.get("requested_serial", "") or "")
                if not cam_name:
                    continue
                visual = item.get("visual_mask_overlay") if isinstance(item.get("visual_mask_overlay"), dict) else {}
                if isinstance(visual, dict):
                    broker_visual_rows[cam_name] = dict(visual)
        broker_visual_enabled = bool(broker_latest.get("visual_mask_overlay_enable", False))
        per_cam: Dict[str, Any] = {}
        broken: List[str] = []
        if not overview_ok:
            broken.append("overview_renderer")
        if not mask_evidence_ok:
            broken.append("mask_evidence_service")
        for cam in list(getattr(self.preview, "cams", []) or []):
            worker = self._read_json_status(f"event_worker_{cam}_status.json")
            worker_overlay_sidecar = self._read_json_status(f"event_overlay_{cam}_latest.json")
            sync_health = self._read_json_status(f"event_sync_health_{cam}.json")
            mask_ev = mask_evidence_rows.get(cam, {}) if isinstance(mask_evidence_rows, dict) else {}
            if not isinstance(mask_ev, dict):
                mask_ev = {}
            mask_raster = mask_ev.get("mask_raster", {}) if isinstance(mask_ev.get("mask_raster"), dict) else {}
            broker_visual = broker_visual_rows.get(cam, {}) if isinstance(broker_visual_rows.get(cam), dict) else {}
            ov_source = "preview_renderer"
            ov = {}
            overview_ov = overview_rows.get(cam) if isinstance(overview_rows, dict) else None
            if isinstance(overview_ov, dict):
                ov = dict(overview_ov)
                ov_source = "overview_renderer"
            elif isinstance(overlay, dict) and isinstance(overlay.get(cam), dict):
                ov = dict(overlay.get(cam) or {})
            worker_age = self._status_wall_age_ms(worker)
            worker_overlay_sidecar_age = self._status_wall_age_ms(worker_overlay_sidecar)
            worker_ok_from_status = bool(worker.get("_file_exists")) and worker_age >= 0 and worker_age <= 2000.0 and str(worker.get("state", "")).lower() not in ("error", "stopping", "stopped")
            worker_ok_from_overlay_sidecar = bool(
                worker_overlay_sidecar.get("_file_exists")
                and worker_overlay_sidecar_age >= 0
                and worker_overlay_sidecar_age <= 2000.0
                and str(worker_overlay_sidecar.get("camera", "") or cam) == str(cam)
            )
            worker_ok = bool(worker_ok_from_status or worker_ok_from_overlay_sidecar)
            if not worker_ok:
                broken.append(f"{cam}:worker")
            worker_state_s = str(worker.get("state", "") or "").lower()
            worker_active_reason_s = str(worker.get("active_reason", "") or "").lower()
            worker_events_per_slice = _safe_int(worker.get("events_per_slice", 0), 0)
            worker_event_rate = _safe_float(worker.get("event_rate_mevs"), -1.0)
            worker_loop_fps = _safe_float(worker.get("loop_fps"), -1.0)
            worker_guard_state = str(worker.get("guard_state", "") or "")
            overlay_enabled = bool(worker.get("event_overlay_enabled", True))
            overlay_drawn = bool(ov.get("drawn", False))
            overlay_raw_stale = bool(ov.get("stale", True))
            overlay_age = _safe_float(ov.get("age_ms"), -1.0)
            overlay_status_stale = bool(overlay_age >= 0 and overlay_age > 2000.0)
            overlay_idle_empty = bool(
                worker_ok
                and worker_events_per_slice <= 0
                and (
                    "empty" in worker_active_reason_s
                    or worker_state_s in ("idle", "active", "running")
                )
            )
            if overlay_enabled and worker_ok and not overlay_idle_empty:
                if not ov:
                    broken.append(f"{cam}:overlay_missing")
                elif overlay_status_stale:
                    broken.append(f"{cam}:overlay_stale")
                elif not overlay_drawn:
                    broken.append(f"{cam}:overlay_not_drawn")
            human_mask_buffer = worker.get("human_mask_buffer", {}) if isinstance(worker.get("human_mask_buffer"), dict) else {}
            viz_state = worker.get("viz", {}) if isinstance(worker.get("viz"), dict) else {}
            bullet_diag = worker.get("bullet_display_diag", {}) if isinstance(worker.get("bullet_display_diag"), dict) else {}
            if not bullet_diag:
                bullet_diag = {
                    "draw_paths_count": worker.get("draw_paths_count"),
                    "draw_path_points_count": worker.get("draw_path_points_count"),
                    "display_eligible_count": worker.get("display_eligible_count"),
                    "active_track_count": worker.get("active_track_count"),
                    "active_but_not_display_count": worker.get("active_but_not_display_count"),
                    "last_overlay_bullet_point_age_ms": worker.get("last_overlay_bullet_point_age_ms"),
                    "bullet_sender_queue_size": worker.get("bullet_sender_queue_size"),
                    "bullet_sender_dropped": worker.get("bullet_sender_dropped"),
                    "bullet_sender_drop_delta": worker.get("bullet_sender_drop_delta"),
                    "bullet_sender_sent": worker.get("bullet_sender_sent"),
                    "bullet_sender_camera_alias": worker.get("bullet_sender_camera_alias"),
                    "bullet_sender_name": worker.get("bullet_sender_name"),
                    "visual_point_submit_count": worker.get("visual_point_submit_count"),
                    "visual_point_submit_failed_count": worker.get("visual_point_submit_failed_count"),
                    "visual_point_skip_count": worker.get("visual_point_skip_count"),
                    "bullet_source_state": worker.get("bullet_source_state"),
                }
            per_cam[cam] = {
                "worker_state": str(worker.get("state", "") or ("active_inferred" if worker_ok_from_overlay_sidecar else "")),
                "worker_age_ms": worker_age if worker_ok_from_status else worker_overlay_sidecar_age,
                "worker_status_source": "event_worker_status" if worker_ok_from_status else ("event_overlay_sidecar" if worker_ok_from_overlay_sidecar else "missing"),
                "worker_status_inferred_active": bool((not worker_ok_from_status) and worker_ok_from_overlay_sidecar),
                "worker_overlay_sidecar_age_ms": worker_overlay_sidecar_age,
                "worker_overlay_sidecar_seq": _safe_int(worker_overlay_sidecar.get("seq"), -1),
                "worker_overlay_sidecar_path": str(worker_overlay_sidecar.get("_status_path", "")),
                "event_rate_mevs": worker_event_rate if worker_event_rate >= 0.0 else (_safe_float(ev.get("per_cam_event_rate_mevs", {}).get(cam), 0.0) if isinstance(ev.get("per_cam_event_rate_mevs", {}), dict) else 0.0),
                "loop_fps": worker_loop_fps if worker_loop_fps >= 0.0 else (_safe_float(ev.get("per_cam_worker_loop_fps", {}).get(cam), 0.0) if isinstance(ev.get("per_cam_worker_loop_fps", {}), dict) else 0.0),
                "guard_state": worker_guard_state or (str(ev.get("per_cam_event_guard_state", {}).get(cam, "OK")) if isinstance(ev.get("per_cam_event_guard_state", {}), dict) else "OK"),
                "overlay_source": ov_source,
                "overlay_shm_drawn": overlay_drawn,
                "overlay_shm_stale": overlay_raw_stale,
                "overlay_status_stale": overlay_status_stale,
                "overlay_shm_age_ms": overlay_age,
                "overlay_shm_seq": _safe_int(ov.get("seq"), 0),
                "overlay_draw_reason": str(ov.get("reason", "")),
                "overlay_idle_empty": overlay_idle_empty,
                "dilate_px": int(getattr(self.args, "event_overlay_preview_dilate_px", 4) or 4),
                "dilate_backend": str(getattr(self.args, "event_overlay_preview_dilate_backend", "gl_shader") or "gl_shader"),
                "actual_draw_state": "stale" if (overlay_status_stale and not overlay_idle_empty) else ("drawn" if overlay_drawn else ("idle_empty_slice" if overlay_idle_empty else "not_drawn")),
                "viz_raw_count": _safe_int(ev.get("per_cam_viz_raw_count", {}).get(cam), 0) if isinstance(ev.get("per_cam_viz_raw_count", {}), dict) else 0,
                "viz_filtered_count": _safe_int(ev.get("per_cam_viz_filtered_count", {}).get(cam), 0) if isinstance(ev.get("per_cam_viz_filtered_count", {}), dict) else 0,
                "viz_mask_valid": bool(ev.get("per_cam_viz_mask_valid", {}).get(cam, False)) if isinstance(ev.get("per_cam_viz_mask_valid", {}), dict) else False,
                "sync_lock_state": str(sync_health.get("lock_state", "")),
                "sync_master": str(sync_health.get("sync_master", "")),
                "ext_trigger": worker.get("ext_trigger", {}) if isinstance(worker.get("ext_trigger"), dict) else {},
                "event_sync": worker.get("event_sync", {}) if isinstance(worker.get("event_sync"), dict) else {},
                "obs": worker.get("obs", {}) if isinstance(worker.get("obs"), dict) else {"enabled": bool(worker.get("obs_enabled", False))},
                "viz": viz_state,
                "human_mask_buffer": human_mask_buffer,
                "track_path_position": str(worker.get("track_path_position", "")),
                "track_input_policy": str(worker.get("track_input_policy", "")),
                "track_input_policy_effective": str(worker.get("track_input_policy_effective", "")),
                "track_decoupled_from_mask": bool(worker.get("track_decoupled_from_mask", False)),
                "mask_buffer_blocks_tracking": bool(worker.get("mask_buffer_blocks_tracking", False)),
                "track_event_count": _safe_int(worker.get("track_event_count"), 0),
                "track_event_count_before_hot_pixel": _safe_int(worker.get("track_event_count_before_hot_pixel"), 0),
                "track_raw_event_count": _safe_int(worker.get("track_raw_event_count"), 0),
                "mask_released_event_count": _safe_int(worker.get("mask_released_event_count"), 0),
                "mask_filtered_event_count": _safe_int(worker.get("mask_filtered_event_count"), 0),
                "mask_release_consumed_for_viz_only": bool(worker.get("mask_release_consumed_for_viz_only", False)),
                "mask_evidence_service_enabled": bool(mask_evidence_status.get("enabled", False)),
                "mask_evidence_period_ms": _safe_float(mask_evidence_status.get("period_ms"), -1.0),
                "mask_evidence_effective_fps": _safe_float(mask_evidence_status.get("effective_fps"), 0.0),
                "mask_evidence_fresh": bool(mask_ev.get("fresh", False)),
                "mask_evidence_age_ms": _safe_float(mask_ev.get("age_ms"), -1.0),
                "mask_evidence_sync_index": _safe_int(mask_ev.get("sync_index"), -1),
                "mask_evidence_person_count": _safe_int(mask_ev.get("person_count"), 0),
                "mask_evidence_polygon_count": _safe_int(mask_ev.get("polygon_count"), 0),
                "mask_evidence_bbox_count": _safe_int(mask_ev.get("bbox_count"), 0),
                "mask_raster_enabled": bool(mask_raster.get("enabled", False)),
                "mask_raster_path": str(mask_raster.get("path", "")),
                "mask_raster_seq": _safe_int(mask_raster.get("seq"), 0),
                "mask_raster_pixels": _safe_int(mask_raster.get("mask_pixels"), 0),
                "mask_raster_used_persons": _safe_int(mask_raster.get("used_persons"), 0),
                "mask_raster_used_polygons": _safe_int(mask_raster.get("used_polygons"), 0),
                "mask_raster_projection_state": str(mask_raster.get("projection_state", "")),
                "mask_raster_projected_total_points": _safe_int(mask_raster.get("projected_total_points"), 0),
                "mask_raster_projected_in_bounds_points": _safe_int(mask_raster.get("projected_in_bounds_points"), 0),
                "mask_raster_projected_bbox_xyxy": mask_raster.get("projected_bbox_xyxy"),
                "mask_raster_raw_bbox_xyxy": mask_raster.get("raw_bbox_xyxy"),
                "mask_raster_homography_source": str(mask_raster.get("homography_source", "")),
                "mask_raster_error": str(mask_raster.get("error", "")),
                "hal_visual_mask_overlay": {
                    "enabled": bool(broker_visual.get("enabled", False)),
                    "mask_valid": bool(broker_visual.get("mask_valid", False)),
                    "mask_data_available": bool(broker_visual.get("mask_data_available", False)),
                    "mask_hold_active": bool(broker_visual.get("mask_hold_active", False)),
                    "mask_age_ms": _safe_float(broker_visual.get("mask_age_ms"), -1.0),
                    "filter_policy": str(broker_visual.get("filter_policy", "")),
                    "raw_events_seen": _safe_int(broker_visual.get("raw_events_seen", broker_visual.get("raw_event_count")), 0),
                    "filtered_events_seen": _safe_int(broker_visual.get("filtered_events_seen", broker_visual.get("filtered_event_count")), 0),
                    "dropped_by_mask": _safe_int(broker_visual.get("dropped_by_mask", broker_visual.get("dropped_event_count")), 0),
                    "reduction_pct": round(100.0 * _safe_float(broker_visual.get("reduction_ratio", broker_visual.get("filter_reduction_ratio")), 0.0), 3),
                    "publish_count": _safe_int(broker_visual.get("publish_count"), 0),
                    "nonzero_pixels": _safe_int(broker_visual.get("nonzero_pixels", broker_visual.get("last_nonzero_pixels")), 0),
                    "slice_us": _safe_int(broker_visual.get("slice_us", broker_visual.get("visual_mask_slice_us")), 0),
                    "accumulation_us": _safe_int(broker_visual.get("accumulation_us", broker_visual.get("visual_accumulation_us")), 0),
                    "control_seq": _safe_int(broker_visual.get("control_seq", broker_visual.get("visual_mask_control_seq")), 0),
                    "control_apply_count": _safe_int(broker_visual.get("control_apply_count", broker_visual.get("visual_mask_control_apply_count")), 0),
                    "control_error": str(broker_visual.get("control_error", broker_visual.get("visual_mask_control_error", ""))),
                    "sync_timestamp_policy": str(broker_visual.get("sync_timestamp_policy", broker_visual.get("visual_mask_sync_timestamp_policy", ""))),
                    "last_error": str(broker_visual.get("last_error", "")),
                },
                "track_input_starved_reason": str(worker.get("track_input_starved_reason", "")),
                "human_mask_buffer_reason": str(human_mask_buffer.get("reason", "")),
                "human_mask_buffer_drop_counts": human_mask_buffer.get("drop_counts", {}),
                "human_mask_buffer_release_counts": human_mask_buffer.get("release_counts", {}),
                "event_block_stage_reason": str(worker.get("event_block_stage_reason", "")),
                "event_spatial_collapse_suspect": bool(worker.get("event_spatial_collapse_suspect", False)),
                "event_has_data_but_no_display_track": bool(worker.get("event_has_data_but_no_display_track", False)),
                "overlay_has_data_but_no_display_track": bool(worker.get("overlay_has_data_but_no_display_track", False)),
                "raw_event_spatial_diag": worker.get("raw_event_spatial_diag", {}) if isinstance(worker.get("raw_event_spatial_diag"), dict) else {},
                "track_event_spatial_diag": worker.get("track_event_spatial_diag", {}) if isinstance(worker.get("track_event_spatial_diag"), dict) else {},
                "spatter_cluster_diag": worker.get("spatter_cluster_diag", {}) if isinstance(worker.get("spatter_cluster_diag"), dict) else {},
                "line_filter_reject_diag": worker.get("line_filter_reject_diag", {}) if isinstance(worker.get("line_filter_reject_diag"), dict) else {},
                "event_hot_pixel_filter": worker.get("event_hot_pixel_filter", {}) if isinstance(worker.get("event_hot_pixel_filter"), dict) else {},
                "event_cluster_sanitize": worker.get("event_cluster_sanitize", {}) if isinstance(worker.get("event_cluster_sanitize"), dict) else {},
                "line_filter_shadow_diag": worker.get("line_filter_shadow_diag", {}) if isinstance(worker.get("line_filter_shadow_diag"), dict) else {},
                "bullet_display_diag": {
                    "draw_paths_count": _safe_int(bullet_diag.get("draw_paths_count"), 0),
                    "draw_path_points_count": _safe_int(bullet_diag.get("draw_path_points_count"), 0),
                    "display_eligible_count": _safe_int(bullet_diag.get("display_eligible_count"), 0),
                    "active_track_count": _safe_int(bullet_diag.get("active_track_count"), 0),
                    "active_but_not_display_count": _safe_int(bullet_diag.get("active_but_not_display_count"), 0),
                    "last_overlay_bullet_point_age_ms": _safe_float(bullet_diag.get("last_overlay_bullet_point_age_ms"), -1.0),
                    "bullet_sender_queue_size": _safe_int(bullet_diag.get("bullet_sender_queue_size"), 0),
                    "bullet_sender_dropped": _safe_int(bullet_diag.get("bullet_sender_dropped"), 0),
                    "bullet_sender_drop_delta": _safe_int(bullet_diag.get("bullet_sender_drop_delta"), 0),
                    "bullet_sender_sent": _safe_int(bullet_diag.get("bullet_sender_sent"), 0),
                    "bullet_sender_camera_alias": str(bullet_diag.get("bullet_sender_camera_alias", "")),
                    "bullet_sender_name": str(bullet_diag.get("bullet_sender_name", "")),
                    "visual_point_submit_count": _safe_int(bullet_diag.get("visual_point_submit_count"), 0),
                    "visual_point_submit_failed_count": _safe_int(bullet_diag.get("visual_point_submit_failed_count"), 0),
                    "visual_point_skip_count": _safe_int(bullet_diag.get("visual_point_skip_count"), 0),
                    "bullet_source_state": str(bullet_diag.get("bullet_source_state", "")),
                },
                "bullet_source_state": str(bullet_diag.get("bullet_source_state", "")),
            }
        return {
            "title_cn": "事件相机叠加",
            "state": "OK" if not broken else "DEGRADED",
            "active_count": max(
                _safe_int(ev.get("event_active_count", 0), 0) if isinstance(ev, dict) else 0,
                sum(
                    1
                    for rec in per_cam.values()
                    if isinstance(rec, dict)
                    and _safe_float(rec.get("worker_age_ms"), -1.0) >= 0.0
                    and _safe_float(rec.get("worker_age_ms"), 999999.0) <= 2000.0
                    and str(rec.get("worker_state", "")).lower() not in ("error", "stopping", "stopped")
                ),
            ),
            "broken_links": broken,
            "overview_renderer": {
                "exists": bool(overview.get("_file_exists")),
                "age_ms": self._status_wall_age_ms(overview),
                "visible": bool((overview.get("window", {}) if isinstance(overview.get("window"), dict) else {}).get("visible", False)),
                "visible_requested": bool((overview.get("control", {}) if isinstance(overview.get("control"), dict) else {}).get("visible_requested", True)),
                "status": overview,
            },
            "mask_evidence_service": {
                "exists": bool(mask_evidence_status.get("_file_exists")),
                "age_ms": mask_evidence_age,
                "enabled": bool(mask_evidence_status.get("enabled", False)),
                "period_ms": _safe_float(mask_evidence_status.get("period_ms"), -1.0),
                "effective_fps": _safe_float(mask_evidence_status.get("effective_fps"), 0.0),
                "fresh_count": _safe_int(mask_evidence_status.get("fresh_count"), 0),
                "camera_count": _safe_int(mask_evidence_status.get("camera_count"), 0),
                "mask_raster": mask_evidence_status.get("mask_raster", {}) if isinstance(mask_evidence_status.get("mask_raster"), dict) else {},
                "control_seq": _safe_int(mask_evidence_status.get("control_seq"), 0),
                "control_error": str(mask_evidence_status.get("control_error", "")),
                "status": mask_evidence_status,
            },
            "hal_visual_mask_overlay": {
                "enabled": broker_visual_enabled,
                "broker_status_exists": bool(broker_latest),
                "broker_status_age_ms": broker_age_ms,
                "broker_tail_error": str(broker_tail_error or ""),
                "per_camera": broker_visual_rows,
            },
            "per_camera": per_cam,
        }

    def _global_yolo_batch_metrics(self, status: Dict[str, Any]) -> Dict[str, Any]:
        last_batch = status.get("last_batch", {}) if isinstance(status.get("last_batch"), dict) else {}
        stats = status.get("stats", {}) if isinstance(status.get("stats"), dict) else {}
        batch_size = _safe_int(status.get("last_batch_size", last_batch.get("batch_size", 0)), 0)
        predict_ms = _safe_float(status.get("predict_ms", last_batch.get("predict_ms", 0.0)), 0.0)
        postprocess_ms = _safe_float(last_batch.get("postprocess_ms", status.get("postprocess_ms", 0.0)), 0.0)
        total_ms = _safe_float(status.get("total_ms", last_batch.get("total_ms", 0.0)), 0.0)
        predict_fps_total = _safe_float(last_batch.get("predict_fps_total", status.get("predict_fps_total", 0.0)), 0.0)
        total_fps_total = _safe_float(last_batch.get("total_fps_total", status.get("total_fps_total", 0.0)), 0.0)
        if predict_fps_total <= 0.0 and batch_size > 0 and predict_ms > 0.0:
            predict_fps_total = 1000.0 * float(batch_size) / max(1e-6, predict_ms)
        if total_fps_total <= 0.0 and batch_size > 0 and total_ms > 0.0:
            total_fps_total = 1000.0 * float(batch_size) / max(1e-6, total_ms)
        fps = _safe_float(
            status.get(
                "infer_camera_fps_window",
                status.get("instant_camera_fps", status.get("camera_fps", 0.0)),
            ),
            0.0,
        )
        if fps <= 0.0:
            fps = total_fps_total if total_fps_total > 0.0 else predict_fps_total
        published_fps = _safe_float(
            status.get("published_fps_total", status.get("published_fps", status.get("fps", fps))),
            fps,
        )
        inferred_fps = _safe_float(
            status.get("inferred_fps_total", status.get("inferred_fps", published_fps)),
            published_fps,
        )
        return {
            "fps": published_fps if published_fps > 0.0 else fps,
            "published_fps_total": published_fps,
            "inferred_fps_total": inferred_fps,
            "capacity_fps": fps,
            "batch_size": batch_size,
            "predict_ms": predict_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
            "predict_fps_total": predict_fps_total,
            "total_fps_total": total_fps_total,
            "partial_batches": _safe_int(stats.get("partial_batches", 0), 0),
            "full_batches": _safe_int(stats.get("full_batches", 0), 0),
            "async_record_postprocess_enabled": bool(status.get("async_record_postprocess_enabled", stats.get("async_record_postprocess_enabled", False))),
            "postprocess_queue_depth": _safe_int(status.get("postprocess_queue_depth", stats.get("postprocess_queue_depth", 0)), 0),
            "postprocess_dropped_batches": _safe_int(status.get("postprocess_dropped_batches", stats.get("postprocess_dropped_batches", 0)), 0),
            "polygon_contract_ok": status.get("polygon_contract_ok", None),
            "polygon_contract_state": str(status.get("polygon_contract_state", "")),
            "polygon_output_enabled": bool(status.get("polygon_output_enabled", False)),
            "published_persons_without_polygon": _safe_int(status.get("published_persons_without_polygon", stats.get("published_persons_without_polygon", 0)), 0),
            "published_records_with_polygon_ratio": _safe_float(status.get("published_records_with_polygon_ratio", stats.get("published_records_with_polygon_ratio", 1.0)), 1.0),
            "published_persons_with_polygon_ratio": _safe_float(status.get("published_persons_with_polygon_ratio", stats.get("published_persons_with_polygon_ratio", 1.0)), 1.0),
        }

    def _global_yolo_actual_fps(self, worker_id: str, status: Dict[str, Any], fallback_fps: float) -> Tuple[float, str]:
        stats = status.get("stats", {}) if isinstance(status.get("stats"), dict) else {}
        frames = _safe_int(stats.get("frames_inferred"), -1)
        wall_us = _safe_int(status.get("wall_us", status.get("timestamp_us", status.get("wall_time_us", 0))), 0)
        if frames < 0 or wall_us <= 0:
            return float(fallback_fps), "batch_capacity_fallback"
        prev = self._global_yolo_prev_counters.get(str(worker_id))
        if prev is None:
            self._global_yolo_prev_counters[str(worker_id)] = (int(frames), int(wall_us), float(fallback_fps))
            return float(fallback_fps), "batch_capacity_first_sample"
        prev_frames, prev_wall_us, prev_fps = prev
        delta_frames = int(frames) - int(prev_frames)
        delta_us = int(wall_us) - int(prev_wall_us)
        if delta_us == 0:
            return float(prev_fps), "frames_inferred_hold"
        if delta_frames >= 0 and delta_us > 0:
            fps = 1_000_000.0 * float(delta_frames) / float(delta_us)
            self._global_yolo_prev_counters[str(worker_id)] = (int(frames), int(wall_us), float(fps))
            return fps, "frames_inferred_delta"
        self._global_yolo_prev_counters[str(worker_id)] = (int(frames), int(wall_us), float(fallback_fps))
        return float(fallback_fps), "batch_capacity_counter_reset"

    def _global_yolo_status_age_ms(self, status: Dict[str, Any]) -> float:
        age = self._status_wall_age_ms(status)
        if age < 0.0:
            age = _safe_float(status.get("age_ms"), -1.0)
        return age

    def _build_global_yolo_status(self, telem: Dict[str, Any]) -> Dict[str, Any]:
        yolo = telem.get("yolo", {}) if isinstance(telem, dict) else {}
        merge = self._read_json_status("global_yolo_status.json")
        chain = self._read_json_status("yolo_chain_mode_status.json")
        worker_statuses: Dict[str, Any] = {}
        sync_root = self._sync_root_path("")
        for path in sorted(glob.glob(str(sync_root / "global_yolo_worker_*_status.json"))):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    wid = str(obj.get("worker_id", Path(path).stem))
                    obj["_status_path"] = path
                    obj["_file_age_ms"] = max(0.0, _now_ms() - os.path.getmtime(path) * 1000.0)
                    worker_statuses[wid] = obj
            except Exception:
                continue
        if not worker_statuses:
            merge_workers = merge.get("workers", []) if isinstance(merge.get("workers"), list) else []
            for idx, item in enumerate(merge_workers):
                if isinstance(item, dict):
                    wid = str(item.get("worker", idx))
                    worker_statuses[wid] = item
        if not worker_statuses and bool(merge.get("_file_exists")) and str(merge.get("msg_type", "")) == "global_yolo_status":
            worker_statuses["global_yolo"] = merge
        if not worker_statuses and isinstance(yolo.get("workers"), dict):
            worker_statuses = dict(yolo.get("workers") or {})
        worker_rows: Dict[str, Any] = {}
        partial_batches = 0
        total_worker_fps = 0.0
        total_inferred_fps = 0.0
        total_published_fps = 0.0
        total_capacity_fps = 0.0
        total_batch_size = 0
        queue_depth = 0
        dropped_batches = 0
        async_enabled = False
        polygon_contract_ok = True
        polygon_contract_state = "valid_empty"
        polygon_output_enabled = True
        persons_without_polygon = 0
        records_polygon_ratio = 1.0
        persons_polygon_ratio = 1.0
        fps_sources: Dict[str, int] = {}
        newest_row: Dict[str, Any] = {}
        for wid, st in worker_statuses.items():
            last_batch = st.get("last_batch", {}) if isinstance(st, dict) and isinstance(st.get("last_batch"), dict) else {}
            metrics = self._global_yolo_batch_metrics(st if isinstance(st, dict) else {})
            capacity_fps = _safe_float(metrics.get("capacity_fps"), 0.0)
            actual_fps, fps_source = self._global_yolo_actual_fps(str(wid), st if isinstance(st, dict) else {}, capacity_fps)
            metrics["fps"] = actual_fps
            metrics["capacity_fps"] = capacity_fps
            metrics["fps_source"] = fps_source
            partial_batches += _safe_int(metrics.get("partial_batches"), 0)
            total_worker_fps += _safe_float(metrics.get("fps"), 0.0)
            total_inferred_fps += _safe_float(metrics.get("inferred_fps_total"), 0.0)
            total_published_fps += _safe_float(metrics.get("published_fps_total"), 0.0)
            total_capacity_fps += _safe_float(metrics.get("capacity_fps"), 0.0)
            total_batch_size += _safe_int(metrics.get("batch_size"), 0)
            queue_depth += _safe_int(metrics.get("postprocess_queue_depth"), 0)
            dropped_batches += _safe_int(metrics.get("postprocess_dropped_batches"), 0)
            async_enabled = async_enabled or bool(metrics.get("async_record_postprocess_enabled", False))
            if metrics.get("polygon_contract_ok") is False:
                polygon_contract_ok = False
            metric_state = str(metrics.get("polygon_contract_state", "") or "")
            if metric_state.startswith("invalid") or metric_state == "disabled":
                polygon_contract_state = metric_state
            elif metric_state == "valid_complete" and not polygon_contract_state.startswith("invalid"):
                polygon_contract_state = "valid_complete"
            polygon_output_enabled = polygon_output_enabled and bool(metrics.get("polygon_output_enabled", False))
            persons_without_polygon += _safe_int(metrics.get("published_persons_without_polygon"), 0)
            records_polygon_ratio = min(records_polygon_ratio, _safe_float(metrics.get("published_records_with_polygon_ratio"), 1.0))
            persons_polygon_ratio = min(persons_polygon_ratio, _safe_float(metrics.get("published_persons_with_polygon_ratio"), 1.0))
            source_name = str(metrics.get("fps_source") or "unknown")
            fps_sources[source_name] = int(fps_sources.get(source_name, 0)) + 1
            age_ms = self._global_yolo_status_age_ms(st if isinstance(st, dict) else {})
            if not newest_row or (age_ms >= 0.0 and age_ms < _safe_float(newest_row.get("age_ms"), 999999999.0)):
                newest_row = {
                    "age_ms": age_ms,
                    "predict_ms": metrics.get("predict_ms"),
                    "total_ms": metrics.get("total_ms"),
                    "last_batch": last_batch,
                }
            worker_rows[str(wid)] = {
                "state": st.get("state", ""),
                "fps": metrics.get("fps"),
                "capacity_fps": metrics.get("capacity_fps"),
                "fps_source": metrics.get("fps_source"),
                "batch_size": metrics.get("batch_size"),
                "predict_ms": metrics.get("predict_ms"),
                "postprocess_ms": metrics.get("postprocess_ms"),
                "total_ms": metrics.get("total_ms"),
                "predict_fps_total": metrics.get("predict_fps_total"),
                "total_fps_total": metrics.get("total_fps_total"),
                "partial_batches": metrics.get("partial_batches"),
                "full_batches": metrics.get("full_batches"),
                "inferred_fps_total": metrics.get("inferred_fps_total"),
                "published_fps_total": metrics.get("published_fps_total"),
                "async_record_postprocess_enabled": metrics.get("async_record_postprocess_enabled"),
                "postprocess_queue_depth": metrics.get("postprocess_queue_depth"),
                "postprocess_dropped_batches": metrics.get("postprocess_dropped_batches"),
                "polygon_contract_ok": metrics.get("polygon_contract_ok"),
                "polygon_contract_state": metrics.get("polygon_contract_state"),
                "polygon_output_enabled": metrics.get("polygon_output_enabled"),
                "published_persons_without_polygon": metrics.get("published_persons_without_polygon"),
                "published_persons_with_polygon_ratio": metrics.get("published_persons_with_polygon_ratio"),
                "last_batch": last_batch,
                "age_ms": age_ms,
            }
        merge_age = self._status_wall_age_ms(merge)
        merge_summary = merge.get("summary", {}) if isinstance(merge.get("summary"), dict) else {}
        merge_fresh = bool(merge.get("_file_exists")) and (merge_age < 0.0 or merge_age <= 3000.0)
        merge_published_fps = _safe_float(merge.get("published_fps_total", merge_summary.get("published_fps_total", merge.get("fps", 0.0))), 0.0)
        merge_inferred_fps = _safe_float(merge.get("inferred_fps_total", merge_summary.get("inferred_fps_total", merge_published_fps)), merge_published_fps)
        legacy_capacity_fps = _safe_float(yolo.get("infer_fps"), 0.0) if isinstance(yolo, dict) else 0.0
        if legacy_capacity_fps <= 0.0:
            legacy_capacity_fps = _safe_float(merge.get("infer_fps", merge_published_fps), 0.0)
        if total_capacity_fps <= 0.0:
            total_capacity_fps = legacy_capacity_fps
        if merge_fresh and merge_published_fps > 0.0:
            infer_fps = merge_published_fps
            total_published_fps = merge_published_fps
            total_inferred_fps = merge_inferred_fps
            total_capacity_fps = _safe_float(
                merge_summary.get("last_predict_fps_total", merge.get("last_predict_fps_total", total_capacity_fps)),
                total_capacity_fps,
            )
            queue_depth = _safe_int(merge.get("postprocess_queue_depth", merge_summary.get("postprocess_queue_depth", queue_depth)), queue_depth)
            dropped_batches = _safe_int(merge.get("postprocess_dropped_batches", merge_summary.get("postprocess_dropped_batches", dropped_batches)), dropped_batches)
            async_enabled = bool(merge.get("async_record_postprocess_enabled", merge_summary.get("async_record_postprocess_enabled", async_enabled)))
            polygon_contract_ok = bool(merge.get("polygon_contract_ok", merge_summary.get("polygon_contract_ok", polygon_contract_ok)))
            polygon_contract_state = str(merge.get("polygon_contract_state", merge_summary.get("polygon_contract_state", polygon_contract_state)) or polygon_contract_state)
            polygon_output_enabled = bool(merge.get("polygon_output_enabled", merge_summary.get("polygon_output_enabled", polygon_output_enabled)))
            persons_without_polygon = _safe_int(merge.get("published_persons_without_polygon", merge_summary.get("published_persons_without_polygon", persons_without_polygon)), persons_without_polygon)
            records_polygon_ratio = _safe_float(merge.get("published_records_with_polygon_ratio", merge_summary.get("published_records_with_polygon_ratio", records_polygon_ratio)), records_polygon_ratio)
            persons_polygon_ratio = _safe_float(merge.get("published_persons_with_polygon_ratio", merge_summary.get("published_persons_with_polygon_ratio", persons_polygon_ratio)), persons_polygon_ratio)
        else:
            infer_fps = total_published_fps if total_published_fps > 0.0 else (total_worker_fps if total_worker_fps > 0.0 else legacy_capacity_fps)
        if not fps_sources:
            fps_source = "legacy_capacity" if legacy_capacity_fps > 0.0 else "none"
        elif len(fps_sources) == 1:
            fps_source = next(iter(fps_sources.keys()))
        else:
            fps_source = "mixed:" + ",".join(f"{k}={v}" for k, v in sorted(fps_sources.items()))
        if merge_fresh and merge_published_fps > 0.0:
            fps_source = "global_yolo_status.published_fps_total"
        batch_size = _safe_int(yolo.get("batch_size"), 0) if isinstance(yolo, dict) else 0
        if batch_size <= 0:
            merge_last_batch = merge.get("last_batch", {}) if isinstance(merge.get("last_batch"), dict) else {}
            batch_size = _safe_int(merge_last_batch.get("batch_size"), 0)
        if batch_size <= 0:
            batch_size = total_batch_size
        worker_count = len(worker_rows) or _safe_int(merge.get("worker_count"), 0)
        return {
            "title_cn": "全局YOLO",
            "state": "OK" if (worker_rows or merge_fresh) and (merge_age < 0 or merge_age <= 3000.0) else "DEGRADED",
            "fps": infer_fps,
            "infer_fps": infer_fps,
            "published_fps_total": total_published_fps if total_published_fps > 0.0 else infer_fps,
            "inferred_fps_total": total_inferred_fps if total_inferred_fps > 0.0 else infer_fps,
            "capacity_fps": total_capacity_fps,
            "legacy_capacity_fps": legacy_capacity_fps,
            "fps_source": fps_source,
            "batch_size": batch_size,
            "predict_ms": _safe_float(yolo.get("predict_ms"), _safe_float(newest_row.get("predict_ms"), 0.0)) if isinstance(yolo, dict) else _safe_float(newest_row.get("predict_ms"), 0.0),
            "total_ms": _safe_float(newest_row.get("total_ms"), 0.0),
            "partial_batches": partial_batches,
            "worker_count": worker_count,
            "async_record_postprocess_enabled": bool(async_enabled),
            "postprocess_queue_depth": int(queue_depth),
            "postprocess_dropped_batches": int(dropped_batches),
            "polygon_contract_ok": bool(polygon_contract_ok and dropped_batches == 0),
            "polygon_contract_state": str(polygon_contract_state),
            "polygon_output_enabled": bool(polygon_output_enabled),
            "published_persons_without_polygon": int(persons_without_polygon),
            "published_records_with_polygon_ratio": float(records_polygon_ratio),
            "published_persons_with_polygon_ratio": float(persons_polygon_ratio),
            "merge_status": merge,
            "merge_age_ms": merge_age,
            "workers": worker_rows,
            "legacy_yolo": {
                "enabled": bool(chain.get("legacy_yolo_shards_enabled", False)),
                "reason": str(chain.get("legacy_yolo_reason", "")),
                "chain_mode_status": chain,
            },
        }

    def _build_global_person_status(self) -> Dict[str, Any]:
        st = self._read_json_status("global_person_status.json")
        latest = self._read_json_status("global_person_latest.json")
        shadow_st = self._read_json_status("global_person_shadow_status.json")
        shadow_latest = self._read_json_status("global_person_shadow_latest.json")
        tracks = latest.get("tracks", latest.get("persons", latest.get("objects", [])))
        track_count = len(tracks) if isinstance(tracks, list) else _safe_int(latest.get("track_count", st.get("track_count", 0)), 0)
        shadow_tracks = shadow_latest.get("tracks", shadow_latest.get("persons", shadow_latest.get("objects", [])))
        shadow_track_count = len(shadow_tracks) if isinstance(shadow_tracks, list) else _safe_int(shadow_latest.get("track_count", shadow_st.get("track_count", 0)), 0)
        lost_ids = latest.get("lost_ids", st.get("lost_ids", []))
        if not isinstance(lost_ids, list):
            lost_ids = []
        runtime = st.get("runtime_config", {}) if isinstance(st.get("runtime_config"), dict) else {}
        shadow_runtime = shadow_st.get("runtime_config", {}) if isinstance(shadow_st.get("runtime_config"), dict) else {}
        control = st.get("control", {}) if isinstance(st.get("control"), dict) else {}
        shadow_control = shadow_st.get("control", {}) if isinstance(shadow_st.get("control"), dict) else {}
        stats = st.get("stats", {}) if isinstance(st.get("stats"), dict) else {}
        shadow_stats = shadow_st.get("stats", {}) if isinstance(shadow_st.get("stats"), dict) else {}
        count_gate = st.get("count_gate", {}) if isinstance(st.get("count_gate"), dict) else {}
        shadow_count_gate = shadow_st.get("count_gate", {}) if isinstance(shadow_st.get("count_gate"), dict) else {}
        return {
            "title_cn": "全局人员ID",
            "state": "OK" if bool(st.get("_file_exists")) and self._status_wall_age_ms(st) <= 3000.0 else "DEGRADED",
            "track_count": track_count,
            "filter_backend": str(st.get("filter_backend", runtime.get("filter_backend", ""))),
            "stream_role": str(st.get("stream_role", runtime.get("stream_role", "official"))),
            "control_seq": _safe_int(control.get("seq"), 0),
            "created_tracks": _safe_int(stats.get("created_tracks"), 0),
            "retired_tracks": _safe_int(stats.get("retired_tracks"), 0),
            "id_switch_suspects": _safe_int(stats.get("id_switch_suspects"), 0),
            "shadow_track_count": shadow_track_count,
            "shadow_filter_backend": str(shadow_st.get("filter_backend", shadow_runtime.get("filter_backend", ""))),
            "shadow_stream_role": str(shadow_st.get("stream_role", shadow_runtime.get("stream_role", "shadow"))),
            "shadow_control_seq": _safe_int(shadow_control.get("seq"), 0),
            "count_gate_enable": bool(count_gate.get("enable", runtime.get("count_gate_enable", True))),
            "count_gate_short_edge_zone": str(count_gate.get("short_edge_zone", "")),
            "count_gate_birth_blocked_non_short_edge": _safe_int(count_gate.get("birth_blocked_non_short_edge"), 0),
            "count_gate_birth_blocked_handoff": _safe_int(count_gate.get("birth_blocked_handoff"), 0),
            "count_gate_lost_held_non_short_edge": _safe_int(count_gate.get("lost_held_non_short_edge"), 0),
            "shadow_count_gate_enable": bool(shadow_count_gate.get("enable", shadow_runtime.get("count_gate_enable", True))),
            "shadow_created_tracks": _safe_int(shadow_stats.get("created_tracks"), 0),
            "shadow_retired_tracks": _safe_int(shadow_stats.get("retired_tracks"), 0),
            "shadow_id_switch_suspects": _safe_int(shadow_stats.get("id_switch_suspects"), 0),
            "shadow_status_age_ms": self._status_wall_age_ms(shadow_st),
            "duplicate_suppress_enable": bool(runtime.get("duplicate_suppress_enable", st.get("duplicate_suppress_enable", st.get("duplicate_suppress", True)))),
            "coord_transform_mode": str(
                st.get(
                    "coord_transform_mode",
                    (st.get("coord_transform", {}) if isinstance(st.get("coord_transform"), dict) else {}).get(
                        "mode",
                        latest.get("coord_transform_mode", ""),
                    ),
                )
            ),
            "lost_id_count": len(lost_ids),
            "lost_ids": lost_ids[-32:],
            "count_gate": count_gate,
            "shadow_count_gate": shadow_count_gate,
            "status": st,
            "latest": latest,
            "shadow_status": shadow_st,
            "shadow_latest": shadow_latest,
            "status_age_ms": self._status_wall_age_ms(st),
            "latest_age_ms": self._status_wall_age_ms(latest),
        }

    def _build_person_id_dataset_status(self) -> Dict[str, Any]:
        status_name = str(getattr(self.args, "person_id_dataset_status_json", "") or "person_id_dataset_status.json")
        latest_name = str(getattr(self.args, "person_id_dataset_latest_json", "") or "person_id_dataset_latest.json")
        control_name = str(getattr(self.args, "person_id_dataset_control_json", "") or "person_id_dataset_control.json")
        st = self._read_json_status(status_name)
        latest = self._read_json_status(latest_name)
        control = self._read_json_status(control_name)
        counts = st.get("counts", {}) if isinstance(st.get("counts"), dict) else {}
        per_camera = counts.get("per_camera", {}) if isinstance(counts.get("per_camera"), dict) else {}
        age_ms = self._status_wall_age_ms(st)
        file_exists = bool(st.get("_file_exists"))
        state = str(st.get("state", "MISSING" if not file_exists else "UNKNOWN") or "UNKNOWN").upper()
        if not file_exists:
            state = "MISSING"
        elif age_ms >= 0 and age_ms > 3000.0:
            state = "DEGRADED"
        return {
            "title_cn": "人员ID数据集",
            "state": state,
            "recording": bool(st.get("recording", False)),
            "dataset_kind": str(st.get("dataset_kind", "")),
            "dataset_type": str(st.get("dataset_type", "")),
            "session_id": str(st.get("session_id", "")),
            "session_dir": str(st.get("session_dir", "")),
            "run_id": str(st.get("run_id", "")),
            "experiment_id": str(st.get("experiment_id", "")),
            "experiment_stage": str(st.get("experiment_stage", "")),
            "experiment_config_hash": str(st.get("experiment_config_hash", "")),
            "output_root": str(st.get("output_root", "")),
            "contains_mask_data": bool(st.get("contains_mask_data", False)),
            "contains_polygon_data": bool(st.get("contains_polygon_data", False)),
            "include_mask_polygons": bool(st.get("include_mask_polygons", False)),
            "include_bullet_trajectory": bool(st.get("include_bullet_trajectory", False)),
            "include_hit_debug": bool(st.get("include_hit_debug", False)),
            "source_stage": str(st.get("source_stage", "")),
            "control_seq": _safe_int(st.get("control_seq", control.get("seq", 0)), 0),
            "last_command": str(st.get("last_command", control.get("cmd", "")) or ""),
            "human_rows_total": _safe_int(counts.get("human_rows_total"), 0),
            "human_person_rows_total": _safe_int(counts.get("human_person_rows_total"), 0),
            "human_zero_person_rows_total": _safe_int(counts.get("human_zero_person_rows_total"), 0),
            "human_zero_person_skipped_total": _safe_int(counts.get("human_zero_person_skipped_total"), 0),
            "global_person_track_rows": _safe_int(counts.get("global_person_track_rows"), 0),
            "bullet_point_rows": _safe_int(counts.get("bullet_point_rows"), 0),
            "bullet_event_rows": _safe_int(counts.get("bullet_event_rows"), 0),
            "pseudo_hit_rows": _safe_int(counts.get("pseudo_hit_rows"), 0),
            "final_hit_rows": _safe_int(counts.get("final_hit_rows"), 0),
            "shadow_hit_rows": _safe_int(counts.get("shadow_hit_rows"), 0),
            "hit_reject_rows": _safe_int(counts.get("hit_reject_rows"), 0),
            "status_samples": _safe_int(counts.get("status_samples"), 0),
            "annotation_rows": _safe_int(counts.get("annotation_rows"), 0),
            "sanitized_mask_key_count": _safe_int(counts.get("sanitized_mask_key_count"), 0),
            "bad_lines": _safe_int(counts.get("bad_lines"), 0),
            "last_write_age_ms": _safe_float(st.get("last_write_age_ms"), -1.0),
            "zero_person_keep_ms": _safe_float(st.get("zero_person_keep_ms"), 0.0),
            "status_age_ms": age_ms,
            "last_error": str(st.get("last_error", "")),
            "per_camera": per_camera,
            "status": st,
            "latest": latest,
            "control": control,
        }

    def _build_hit_judge_status(self) -> Dict[str, Any]:
        st = self._read_json_status("hit_judge_status.json")
        age_ms = self._status_wall_age_ms(st)
        stage_total = st.get("visual_point_stage_total", {}) if isinstance(st.get("visual_point_stage_total"), dict) else {}
        result_total = st.get("visual_point_result_total_by_reason", {}) if isinstance(st.get("visual_point_result_total_by_reason"), dict) else {}
        reject_total = st.get("hit_reject_total_by_reason", {}) if isinstance(st.get("hit_reject_total_by_reason"), dict) else {}
        return {
            "title_cn": "命中判定",
            "state": "OK" if bool(st.get("_file_exists")) and age_ms <= 3000.0 else "DEGRADED",
            "hit_detection_enabled": bool(st.get("hit_detection_enabled", True)),
            "hit_candidate_output_enabled": bool(st.get("hit_candidate_output_enabled", True)),
            "strong_gate_enabled": bool(st.get("strong_gate_enabled", True)),
            "strong_reasoning_enabled": bool(st.get("strong_reasoning_enabled", True)),
            "strong_reasoning_shadow_only": bool(st.get("strong_reasoning_shadow_only", True)),
            "final_hit_authority": str(st.get("final_hit_authority", "strong_gate") or "strong_gate"),
            "pseudo_hit_enable": bool(st.get("pseudo_hit_enable", True)),
            "pseudo_hit_total": _safe_int(st.get("pseudo_hit_total"), 0),
            "pseudo_hit_by_camera": st.get("pseudo_hit_by_camera", {}) if isinstance(st.get("pseudo_hit_by_camera"), dict) else {},
            "pseudo_hit_reason_total": st.get("pseudo_hit_reason_total", {}) if isinstance(st.get("pseudo_hit_reason_total"), dict) else {},
            "pseudo_hit_duplicate_suppressed": _safe_int(st.get("pseudo_hit_duplicate_suppressed"), 0),
            "pseudo_hit_promoted_to_final_count": _safe_int(st.get("pseudo_hit_promoted_to_final_count"), 0),
            "pseudo_hit_human_noise_suspect": _safe_int(st.get("pseudo_hit_human_noise_suspect"), 0),
            "pseudo_hit_human_noise_suppressed": _safe_int(st.get("pseudo_hit_human_noise_suppressed"), 0),
            "pseudo_hit_human_noise_reason_total": st.get("pseudo_hit_human_noise_reason_total", {}) if isinstance(st.get("pseudo_hit_human_noise_reason_total"), dict) else {},
            "human_noise_filter": st.get("human_noise_filter", {}) if isinstance(st.get("human_noise_filter"), dict) else {},
            "strong_gate_output_suppressed": _safe_int(st.get("strong_gate_output_suppressed"), 0),
            "strong_reasoning_output_suppressed": _safe_int(st.get("strong_reasoning_output_suppressed"), 0),
            "latest_pseudo_hit": st.get("latest_pseudo_hit", {}) if isinstance(st.get("latest_pseudo_hit"), dict) else {},
            "control": st.get("control", {}) if isinstance(st.get("control"), dict) else {},
            "recv_total": _safe_int(st.get("hit_recv_total"), 0),
            "hit_candidate_total": _safe_int(st.get("hit_candidate_total"), 0),
            "visual_point_active_track_keys": _safe_int(st.get("visual_point_active_track_keys"), 0),
            "visual_point_pending_human": _safe_int(st.get("visual_point_pending_human"), 0),
            "stage_total": stage_total,
            "result_total": result_total,
            "reject_total": reject_total,
            "latest_visual_point_stage": st.get("latest_visual_point_stage", {}) if isinstance(st.get("latest_visual_point_stage"), dict) else {},
            "latest_visual_point_reject": st.get("latest_visual_point_reject", {}) if isinstance(st.get("latest_visual_point_reject"), dict) else {},
            "latest_visual_point_candidate": st.get("latest_visual_point_candidate", {}) if isinstance(st.get("latest_visual_point_candidate"), dict) else {},
            "visual_point_stage_by_camera": st.get("visual_point_stage_by_camera", {}) if isinstance(st.get("visual_point_stage_by_camera"), dict) else {},
            "status_age_ms": age_ms,
            "status": st,
        }

    def _build_global_bev_status(self, telem: Dict[str, Any]) -> Dict[str, Any]:
        st = self._read_json_status("global_bev_status.json")
        sidecar = self._read_json_status("/dev/shm/global_bev_latest.json")
        control = self._read_json_status("global_bev_control.json")
        input_age = st.get("input_frame_age_ms", st.get("per_camera_input_age_ms", st.get("frame_age_ms", {})))
        event_layer = st.get("event_layer", {}) if isinstance(st.get("event_layer"), dict) else {}
        if not event_layer:
            event_layer = st.get("event_layer_overlay", {}) if isinstance(st.get("event_layer_overlay"), dict) else {}
        event_bullet = st.get("event_bullet", {}) if isinstance(st.get("event_bullet"), dict) else {}
        if not event_bullet:
            event_bullet = st.get("event_bullet_overlay", {}) if isinstance(st.get("event_bullet_overlay"), dict) else {}
        global_person = st.get("global_person", {}) if isinstance(st.get("global_person"), dict) else {}
        if not global_person:
            global_person = st.get("global_person_overlay", {}) if isinstance(st.get("global_person_overlay"), dict) else {}
        readback = st.get("readback", {}) if isinstance(st.get("readback"), dict) else {}
        pbo_status = st.get("pbo_status", {}) if isinstance(st.get("pbo_status"), dict) else {}
        mvs_sync = st.get("mvs_sync", {}) if isinstance(st.get("mvs_sync"), dict) else {}
        mvs_texture_upload = st.get("mvs_texture_upload", {}) if isinstance(st.get("mvs_texture_upload"), dict) else {}
        presentation = st.get("presentation", {}) if isinstance(st.get("presentation"), dict) else {}
        event_per_cam = event_layer.get("per_camera", {}) if isinstance(event_layer.get("per_camera"), dict) else {}
        event_sync_delta_by_cam = {
            str(cam): _safe_int(rec.get("event_to_bev_sync_delta_ticks"), -999999)
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict)
        }
        event_sync_state_by_cam = {
            str(cam): str(rec.get("event_layer_sync_state", rec.get("sync_state", "")))
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict)
        }
        event_sync_delta_ms_by_cam = {
            str(cam): _safe_float(rec.get("event_to_bev_sync_delta_ms"), -999999.0)
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict)
        }
        event_sync_display_state_by_cam = dict(event_layer.get("sync_display_state_by_cam", {})) if isinstance(event_layer.get("sync_display_state_by_cam"), dict) else {
            str(cam): str(rec.get("event_layer_sync_display_state", rec.get("event_layer_sync_state", "")))
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict)
        }
        event_sync_display_level_by_cam = dict(event_layer.get("sync_display_level_by_cam", {})) if isinstance(event_layer.get("sync_display_level_by_cam"), dict) else {
            str(cam): str(rec.get("event_layer_sync_display_level", ""))
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict)
        }
        event_history_source_by_cam = {
            str(cam): str(rec.get("history_source", ""))
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict)
        }
        event_sync_hidden_cams = [
            str(cam)
            for cam, rec in event_per_cam.items()
            if isinstance(rec, dict) and not bool(rec.get("draw", False))
            and str(rec.get("sync_reason", rec.get("reason", ""))).strip()
        ]
        event_layer_summary = {
            "enabled": bool(event_layer.get("enabled", False)),
            "render_target": str(event_layer.get("render_target", "")),
            "transparent_background": bool(event_layer.get("transparent_background", True)),
            "draw_cam_count": _safe_int(event_layer.get("shader_draw_cam_count", event_layer.get("draw_cam_count", 0)), 0),
            "draw_cams": event_layer.get("shader_draw_cams", event_layer.get("draw_cams", [])),
            "sync_mode": str(event_layer.get("sync_mode", "")),
            "sync_timestamp_policy": str(event_layer.get("sync_timestamp_policy", "")),
            "max_sync_delta_ms": _safe_float(event_layer.get("max_sync_delta_ms"), -1.0),
            "max_sync_delta_ticks": _safe_int(event_layer.get("max_sync_delta_ticks"), -1),
            "sync_normal_ms": _safe_float(event_layer.get("sync_normal_ms"), 50.0),
            "sync_warn_ms": _safe_float(event_layer.get("sync_warn_ms"), 80.0),
            "sync_hard_ms": _safe_float(event_layer.get("sync_hard_ms"), 80.0),
            "hold_last_good_on_hard_outlier": bool(event_layer.get("hold_last_good_on_hard_outlier", True)),
            "hold_last_good_ms": _safe_float(event_layer.get("hold_last_good_ms"), -1.0),
            "event_sync_delta_by_cam": event_sync_delta_by_cam,
            "event_sync_delta_ms_by_cam": event_sync_delta_ms_by_cam,
            "event_sync_state_by_cam": event_sync_state_by_cam,
            "event_sync_display_state_by_cam": event_sync_display_state_by_cam,
            "event_sync_display_level_by_cam": event_sync_display_level_by_cam,
            "event_sync_hidden_cams": event_sync_hidden_cams,
            "event_sync_warn_cams": event_layer.get("sync_warn_cams", []),
            "event_sync_hard_outlier_cams": event_layer.get("sync_hard_outlier_cams", []),
            "event_sync_hold_cams": event_layer.get("sync_hold_cams", []),
            "event_sync_hidden_cams_strict": event_layer.get("sync_hidden_cams", event_sync_hidden_cams),
            "display_budget_enable": bool(event_layer.get("display_budget_enable", False)),
            "display_budget_ms": _safe_float(event_layer.get("display_budget_ms"), -1.0),
            "display_budget_elapsed_ms": _safe_float(event_layer.get("display_budget_elapsed_ms"), -1.0),
            "display_budget_elapsed_p95_ms": _safe_float(event_layer.get("display_budget_elapsed_p95_ms"), -1.0),
            "display_budget_overrun": bool(event_layer.get("display_budget_overrun", False)),
            "display_budget_overrun_rate": _safe_float(event_layer.get("display_budget_overrun_rate"), 0.0),
            "display_budget_held_cams": event_layer.get("display_budget_held_cams", []),
            "display_budget_held_cam_count": _safe_int(event_layer.get("display_budget_held_cam_count"), 0),
            "display_budget_held_cam_count_p95": _safe_float(event_layer.get("display_budget_held_cam_count_p95"), 0.0),
            "display_budget_order": event_layer.get("display_budget_order", []),
            "sparse_read_elapsed_ms": _safe_float(event_layer.get("sparse_read_elapsed_ms"), 0.0),
            "sparse_sync_gate_elapsed_ms": _safe_float(event_layer.get("sparse_sync_gate_elapsed_ms"), 0.0),
            "sparse_project_elapsed_ms": _safe_float(event_layer.get("sparse_project_elapsed_ms"), 0.0),
            "history_enable": bool(event_layer.get("history_enable", False)),
            "history_selected_cam_count": _safe_int(event_layer.get("history_selected_cam_count"), 0),
            "history_latest_fallback_cam_count": _safe_int(event_layer.get("history_latest_fallback_cam_count"), 0),
            "history_source_by_cam": event_history_source_by_cam,
            "alpha": _safe_float(event_layer.get("alpha"), -1.0),
            "gain": _safe_float(event_layer.get("gain"), -1.0),
            "min_value": _safe_float(event_layer.get("min_value"), -1.0),
            "max_age_ms": _safe_float(event_layer.get("max_age_ms"), -1.0),
            "color_mode": str(event_layer.get("color_mode", "camera") or "camera"),
            "available_color_modes": event_layer.get("available_color_modes", []),
            "dilate_px": _safe_float(event_layer.get("dilate_px"), -1.0),
            "dilate_backend": str(event_layer.get("dilate_backend", "")),
            "dilate_max_pixels": _safe_int(event_layer.get("dilate_max_pixels"), 0),
            "dilate_skipped_dense_cam_count": _safe_int(event_layer.get("dilate_skipped_dense_cam_count"), 0),
            "skip_reason": str(event_layer.get("skip_reason", "")),
            "stale_cam_count": _safe_int(event_layer.get("stale_cam_count"), 0),
            "empty_cam_count": _safe_int(event_layer.get("empty_cam_count"), 0),
            "error_cam_count": _safe_int(event_layer.get("error_cam_count"), 0),
            "per_camera": event_layer.get("per_camera", {}) if isinstance(event_layer.get("per_camera"), dict) else {},
        }
        event_bullet_summary = {
            "enabled": bool(event_bullet.get("enabled", False)),
            "render_target": str(event_bullet.get("render_target", "")),
            "recent_records": _safe_int(event_bullet.get("recent_records"), 0),
            "projected_points": _safe_int(event_bullet.get("projected_points"), 0),
            "visible_points": _safe_int(event_bullet.get("visible_points"), 0),
            "draw_points": _safe_int(event_bullet.get("draw_points"), 0),
            "draw_segments": _safe_int(event_bullet.get("draw_segments"), 0),
            "latest_age_ms": _safe_float(event_bullet.get("latest_age_ms"), -1.0),
            "file_age_ms": _safe_float(event_bullet.get("file_age_ms"), -1.0),
            "records_buffered": _safe_int(event_bullet.get("records_buffered"), 0),
            "skip_reason": str(event_bullet.get("skip_reason", "")),
            "skip_counts": event_bullet.get("skip_counts", {}) if isinstance(event_bullet.get("skip_counts"), dict) else {},
            "projection_fail_count_by_reason": event_bullet.get("projection_fail_count_by_reason", {}) if isinstance(event_bullet.get("projection_fail_count_by_reason"), dict) else {},
            "last_record_camera": str(event_bullet.get("last_record_camera", "")),
            "last_record_bullet_id": _safe_int(event_bullet.get("last_record_bullet_id"), 0),
            "last_record_age_ms": _safe_float(event_bullet.get("last_record_age_ms"), -1.0),
            "error": str(event_bullet.get("error", "")),
        }
        return {
            "title_cn": "全局BEV",
            "state": "OK" if bool(st.get("_file_exists")) and self._status_wall_age_ms(st) <= 3000.0 else "DEGRADED",
            "render_fps": _safe_float(st.get("render_fps"), 0.0),
            "output_fps": _safe_float(st.get("output_fps"), 0.0),
            "fbo_status": st.get("fbo_stats", st.get("fbo_status", {})),
            "pbo_ready": bool(st.get("pbo_ready", pbo_status.get("ready", readback.get("pbo_ready", False)))),
            "pbo_mode": str(readback.get("mode", pbo_status.get("mode", ""))),
            "pbo_ready_count": _safe_int(readback.get("pbo_ready_count", pbo_status.get("ready_count", 0)), 0),
            "input_frame_age_ms": input_age,
            "mvs_sync": mvs_sync,
            "mvs_texture_upload": mvs_texture_upload,
            "presentation": presentation,
            "control_state": control,
            "sidecar_age_ms": self._status_wall_age_ms(sidecar),
            "global_person": global_person,
            "global_person_overlay": global_person,
            "event_layer": event_layer_summary,
            "event_layer_overlay": event_layer_summary,
            "event_bullet": event_bullet_summary,
            "event_bullet_overlay": event_bullet_summary,
            "status": st,
        }

    def _build_bullet_reprocess_status(self) -> Dict[str, Any]:
        st = self._read_json_status("global_bullet_reprocess_status.json")
        latest = self._read_json_status("global_bullet_reprocess_latest.json")
        trajectories = latest.get("trajectories", latest.get("bullets", latest.get("latest_trajectories", [])))
        trajectory_count = len(trajectories) if isinstance(trajectories, list) else _safe_int(latest.get("trajectory_count", st.get("trajectory_count", 0)), 0)
        return {
            "title_cn": "弹道重处理",
            "state": "OK" if bool(st.get("_file_exists")) and self._status_wall_age_ms(st) <= 5000.0 else "DEGRADED",
            "scan_state": str(st.get("state", st.get("scan_state", ""))),
            "latest_trajectory_count": trajectory_count,
            "output_file_age_ms": self._status_wall_age_ms(latest),
            "status": st,
            "latest": latest,
        }

    def _build_raw_recording_status(self) -> Dict[str, Any]:
        legacy_mvs = self._read_json_status("record_status.json")
        cmd = self._read_json_status("record_control.json")
        cams = [str(c) for c in list(getattr(self.preview, "cams", []) or [])]
        command_age = self._status_wall_age_ms(cmd)
        legacy_exists = bool(legacy_mvs.get("_file_exists"))
        legacy_age = self._status_wall_age_ms(legacy_mvs)

        split_per_camera: Dict[str, Any] = {}
        split_status_paths: List[str] = []
        split_status_count = 0
        split_recording_count = 0
        split_stale_count = 0
        split_written_total = 0
        split_dropped_total = 0
        split_queue_total = 0
        split_queue_limit = 0
        split_ages: List[float] = []
        split_session_id = ""
        split_record_root = ""
        split_session_dir = ""
        split_video_fps = 0.0
        split_fourcc = ""
        for cam in cams:
            st = self._read_json_status(f"record_status_{cam}.json")
            exists = bool(st.get("_file_exists"))
            age_ms = self._status_wall_age_ms(st)
            if not exists:
                split_per_camera[cam] = {
                    "status_exists": False,
                    "status_age_ms": age_ms,
                    "stale": False,
                    "enabled": False,
                    "recording": False,
                    "camera": cam,
                    "writer_alive": False,
                    "writer_open": False,
                    "csv_open": False,
                    "queue_size": 0,
                    "queue_limit": 0,
                    "written_frames": 0,
                    "dropped_frames": 0,
                    "last_record_frame_idx": -1,
                    "last_error": "status_missing",
                    "video_path": "",
                    "csv_path": "",
                    "session_id": "",
                    "session_dir": "",
                }
                continue
            split_status_count += 1
            split_status_paths.append(str(st.get("_status_path", self._sync_root_path(f"record_status_{cam}.json"))))
            if age_ms >= 0:
                split_ages.append(float(age_ms))
            stale = bool(age_ms >= 0 and age_ms > 3000.0)
            if stale:
                split_stale_count += 1
            rec = bool(st.get("recording", False))
            if rec:
                split_recording_count += 1
                if not split_session_id:
                    split_session_id = str(st.get("session_id", "") or "")
            written = _safe_int(st.get("written_frames", st.get("written_frames_total")), 0)
            dropped = _safe_int(st.get("dropped_frames", st.get("dropped_frames_total")), 0)
            qsize = _safe_int(st.get("queue_size", 0), 0)
            qlimit = _safe_int(st.get("queue_limit", 0), 0)
            qpeak = _safe_int(st.get("queue_peak", qsize), qsize)
            split_written_total += written
            split_dropped_total += dropped
            split_queue_total += qsize
            split_queue_limit = max(split_queue_limit, qlimit)
            if not split_record_root:
                split_record_root = str(st.get("record_root", "") or "")
            if not split_session_dir:
                split_session_dir = str(st.get("session_dir", "") or "")
            if split_video_fps <= 0:
                split_video_fps = _safe_float(st.get("video_fps"), 0.0)
            if not split_fourcc:
                split_fourcc = str(st.get("fourcc", "") or "")
            split_per_camera[cam] = {
                "status_exists": True,
                "status_age_ms": age_ms,
                "stale": stale,
                "enabled": bool(st.get("enabled", True)),
                "recording": rec,
                "camera": cam,
                "serial": str(st.get("serial", "") or ""),
                "writer_alive": bool(st.get("writer_alive", False)),
                "writer_open": bool(st.get("writer_open", False)),
                "csv_open": bool(st.get("csv_open", False)),
                "queue_size": qsize,
                "queue_limit": qlimit,
                "queue_peak": qpeak,
                "written_frames": written,
                "dropped_frames": dropped,
                "last_record_frame_idx": _safe_int(st.get("last_record_frame_idx"), -1),
                "last_write_wall_us": _safe_int(st.get("last_write_wall_us"), 0),
                "last_error": str(st.get("last_error", "") or ""),
                "video_path": str(st.get("video_path", "") or ""),
                "csv_path": str(st.get("csv_path", "") or ""),
                "session_id": str(st.get("session_id", "") or ""),
                "session_dir": str(st.get("session_dir", "") or ""),
            }

        split_mode = split_status_count > 0
        if split_mode:
            mvs_exists = True
            mvs_age = max(split_ages) if split_ages else -1.0
            mvs_recording = split_recording_count > 0
            mvs_per_camera = split_per_camera
            mvs_written_total = split_written_total
            mvs_dropped_total = split_dropped_total
            mvs_queue_total = split_queue_total
            mvs_queue_limit = split_queue_limit
            mvs_record_root = split_record_root
            mvs_session_dir = split_session_dir
            mvs_session_id = split_session_id
            mvs_routes_enabled = [cam for cam, rec in split_per_camera.items() if bool(rec.get("enabled"))]
            mvs_video_fps = split_video_fps
            mvs_fourcc = split_fourcc
            mvs_status_file = "record_status_*.json"
            mvs_status_files = split_status_paths
            mvs_status_mode = "split_workers"
        else:
            mvs_exists = legacy_exists
            mvs_age = legacy_age
            mvs_recording = bool(legacy_mvs.get("recording", False)) if legacy_exists else False
            mvs_per_camera = legacy_mvs.get("per_camera", {}) if isinstance(legacy_mvs.get("per_camera"), dict) else {}
            mvs_written_total = _safe_int(legacy_mvs.get("written_frames_total"), 0)
            mvs_dropped_total = _safe_int(legacy_mvs.get("dropped_frames_total"), 0)
            mvs_queue_total = _safe_int(legacy_mvs.get("queue_size_total"), 0)
            mvs_queue_limit = _safe_int(legacy_mvs.get("queue_limit"), 0)
            mvs_record_root = str(legacy_mvs.get("record_root", "") or "")
            mvs_session_dir = str(legacy_mvs.get("session_dir", "") or "")
            mvs_session_id = str(legacy_mvs.get("session_id", "") or "")
            mvs_routes_enabled = legacy_mvs.get("routes_enabled", [])
            mvs_video_fps = _safe_float(legacy_mvs.get("video_fps"), 0.0)
            mvs_fourcc = str(legacy_mvs.get("fourcc", "") or "")
            mvs_status_file = str(legacy_mvs.get("_status_path", self._sync_root_path("record_status.json")))
            mvs_status_files = [mvs_status_file] if legacy_exists else []
            mvs_status_mode = "legacy_single_process"

        session_id = str(mvs_session_id or cmd.get("session_id", "") or "")
        per_event: Dict[str, Any] = {}
        event_recording_count = 0
        event_status_count = 0
        event_stale_count = 0
        event_slices_total = 0
        event_events_total = 0
        event_bytes_total = 0
        event_dropped_total = 0
        event_queue_total = 0
        for cam in list(getattr(self.preview, "cams", []) or []):
            worker = self._read_json_status(f"event_worker_{cam}_status.json")
            age_ms = self._status_wall_age_ms(worker)
            exists = bool(worker.get("_file_exists"))
            if exists:
                event_status_count += 1
            stale = bool(exists and age_ms >= 0 and age_ms > 3000.0)
            if stale:
                event_stale_count += 1
            rec = bool(worker.get("raw_recording", False))
            if rec:
                event_recording_count += 1
            raw_status = worker.get("raw_recording_status", {}) if isinstance(worker.get("raw_recording_status"), dict) else {}
            event_pack = raw_status.get("event_slice_pack", {}) if isinstance(raw_status.get("event_slice_pack"), dict) else {}
            event_slices_total += _safe_int(event_pack.get("slices_written"), 0)
            event_events_total += _safe_int(event_pack.get("events_written"), 0)
            event_bytes_total += _safe_int(event_pack.get("bytes_written"), 0)
            event_dropped_total += _safe_int(event_pack.get("dropped_slices"), 0)
            event_queue_total += _safe_int(event_pack.get("queue_size"), 0)
            per_event[cam] = {
                "status_exists": exists,
                "status_age_ms": age_ms,
                "stale": stale,
                "raw_recording": rec,
                "worker_state": str(worker.get("state", "")),
                "backend": str(event_pack.get("backend", raw_status.get("backend", "")) or ""),
                "session_id": str(event_pack.get("session_id", "") or ""),
                "session_dir": str(event_pack.get("session_dir", "") or ""),
                "manifest_path": str(event_pack.get("manifest_path", "") or ""),
                "data_path": str(event_pack.get("data_path", "") or ""),
                "queue_size": _safe_int(event_pack.get("queue_size"), 0),
                "queue_limit": _safe_int(event_pack.get("queue_limit"), 0),
                "writer_alive": bool(event_pack.get("writer_alive", False)),
                "slices_written": _safe_int(event_pack.get("slices_written"), 0),
                "events_written": _safe_int(event_pack.get("events_written"), 0),
                "bytes_written": _safe_int(event_pack.get("bytes_written"), 0),
                "dropped_slices": _safe_int(event_pack.get("dropped_slices"), 0),
                "last_error": str(event_pack.get("last_error", "") or ""),
            }
        broker_path = self._sync_root_path("event_native_hal_broker_stats.jsonl")
        broker_rows, broker_tail_error, broker_file_size, broker_file_age_ms = self._read_jsonl_tail_records(
            broker_path,
            max_records=1,
            tail_bytes=256_000,
        )
        broker_latest = broker_rows[-1] if broker_rows else {}
        broker_cameras = broker_latest.get("cameras", []) if isinstance(broker_latest.get("cameras"), list) else []
        broker_raw_per_camera: Dict[str, Any] = {}
        broker_raw_recording_count = 0
        broker_raw_error_count = 0
        for item in broker_cameras:
            if not isinstance(item, dict):
                continue
            cam = str(item.get("alias", "") or item.get("requested_serial", "") or "")
            if not cam:
                continue
            rec = bool(item.get("raw_recording", False))
            err = str(item.get("raw_record_last_error", "") or "")
            if rec:
                broker_raw_recording_count += 1
            broker_raw_error_count += _safe_int(item.get("raw_record_error_count"), 0)
            broker_raw_per_camera[cam] = {
                "enabled": bool(item.get("raw_record_enabled", False)),
                "recording": rec,
                "session_id": str(item.get("raw_record_session_id", "") or ""),
                "path": str(item.get("raw_record_path", "") or ""),
                "last_command": str(item.get("raw_record_last_command", "") or ""),
                "last_seq": _safe_int(item.get("raw_record_last_seq"), 0),
                "commands_seen": _safe_int(item.get("raw_record_commands_seen"), 0),
                "start_count": _safe_int(item.get("raw_record_start_count"), 0),
                "stop_count": _safe_int(item.get("raw_record_stop_count"), 0),
                "error_count": _safe_int(item.get("raw_record_error_count"), 0),
                "last_error": err,
            }
        recording_any = bool(mvs_recording or event_recording_count > 0)
        broken: List[str] = []
        degraded: List[str] = []
        if not mvs_exists:
            degraded.append("mvs_record_status_missing")
        elif split_mode and cams and split_status_count < len(cams):
            degraded.append("mvs_record_status_partial")
        elif (not split_mode) and mvs_age >= 0 and mvs_age > 3000.0:
            degraded.append("mvs_record_status_stale")
        if split_mode and split_stale_count > 0:
            degraded.append("mvs_record_status_stale")
        if event_status_count <= 0:
            degraded.append("event_worker_status_missing")
        if event_stale_count > 0:
            degraded.append("event_worker_status_stale")
        if broker_tail_error and broker_tail_error != "missing":
            degraded.append("event_hal_broker_stats_read_error")
        if broker_raw_error_count > 0:
            degraded.append("event_hal_broker_raw_record_error")
        if event_dropped_total > 0:
            degraded.append("event_record_dropped_slices")
        if mvs_dropped_total > 0:
            degraded.append("mvs_record_dropped_frames")
        recording_any = bool(recording_any or broker_raw_recording_count > 0)
        state = "RECORDING" if recording_any else "READY"
        if broken:
            state = "BROKEN"
        elif degraded and recording_any:
            state = "RECORDING_DEGRADED"
        elif degraded and str(cmd.get("cmd", cmd.get("command", "")) or "").strip().lower() == "start":
            state = "DEGRADED" if not recording_any else "RECORDING_DEGRADED"
        return {
            "title_cn": "原始数据录制",
            "state": state,
            "recording": recording_any,
            "session_id": session_id,
            "command_file": str(cmd.get("_status_path", self._sync_root_path("record_control.json"))),
            "status_file": mvs_status_file,
            "status_files": mvs_status_files,
            "command_exists": bool(cmd.get("_file_exists")),
            "command_age_ms": command_age,
            "last_command": str(cmd.get("cmd", cmd.get("command", "")) or ""),
            "last_command_targets": cmd.get("targets", cmd.get("target", cmd.get("record_targets", ""))),
            "last_command_source": str(cmd.get("source", "") or ""),
            "last_command_session_id": str(cmd.get("session_id", "") or ""),
            "mvs": {
                "status_exists": mvs_exists,
                "status_age_ms": mvs_age,
                "status_mode": mvs_status_mode,
                "status_count": split_status_count if split_mode else (1 if legacy_exists else 0),
                "expected_status_count": len(cams) if split_mode else (1 if legacy_exists else 0),
                "enabled": bool(mvs_exists),
                "recording": mvs_recording,
                "recording_count": split_recording_count if split_mode else (1 if mvs_recording else 0),
                "stale_count": split_stale_count if split_mode else (1 if mvs_age >= 0 and mvs_age > 3000.0 else 0),
                "session_id": mvs_session_id,
                "reason": str(legacy_mvs.get("reason", "") or ""),
                "record_root": mvs_record_root,
                "session_dir": mvs_session_dir,
                "routes_enabled": mvs_routes_enabled,
                "written_frames_total": mvs_written_total,
                "dropped_frames_total": mvs_dropped_total,
                "queue_size_total": mvs_queue_total,
                "queue_limit": mvs_queue_limit,
                "video_fps": mvs_video_fps,
                "fourcc": mvs_fourcc,
                "per_camera": mvs_per_camera,
            },
            "event_raw": {
                "status_count": event_status_count,
                "recording_count": event_recording_count,
                "stale_count": event_stale_count,
                "slices_written_total": event_slices_total,
                "events_written_total": event_events_total,
                "bytes_written_total": event_bytes_total,
                "dropped_slices_total": event_dropped_total,
                "queue_size_total": event_queue_total,
                "per_camera": per_event,
            },
            "event_hal_broker": {
                "status_exists": bool(broker_latest),
                "status_path": str(broker_path),
                "status_file_size": broker_file_size,
                "status_age_ms": broker_file_age_ms,
                "tail_error": broker_tail_error,
                "raw_backend": "hal_i_events_stream_log_raw_data",
                "raw_recording_count": broker_raw_recording_count,
                "raw_record_command_file": str(broker_latest.get("raw_record_command_file", "") or ""),
                "raw_record_dir": str(broker_latest.get("raw_record_dir", "") or ""),
                "raw_record_errors": _safe_int(broker_latest.get("raw_record_errors"), broker_raw_error_count),
                "per_camera": broker_raw_per_camera,
            },
            "broken_links": broken,
            "degraded_links": degraded,
        }

    def _build_replay_evidence_status(self) -> Dict[str, Any]:
        pack = self._read_json_status("replay_pack_status.json")
        evidence = self._read_json_status("evidence_replay_status.json")
        control = self._read_json_status("evidence_replay_control.json")
        annotation_path = Path(self._sync_root_path("operator_annotations.jsonl"))
        annotation_count = 0
        last_annotation: Dict[str, Any] = {}
        if annotation_path.exists():
            try:
                with annotation_path.open("rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - 512 * 1024), os.SEEK_SET)
                    lines = f.read().decode("utf-8", errors="ignore").splitlines()
                for line in lines:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        annotation_count += 1
                        last_annotation = obj
            except Exception as exc:
                last_annotation = {"error": f"{type(exc).__name__}: {exc}"}
        age_ms = self._status_wall_age_ms(pack)
        evidence_age_ms = self._status_wall_age_ms(evidence)
        pack_exists = bool(pack.get("_file_exists"))
        evidence_exists = bool(evidence.get("_file_exists"))
        state = "READY"
        if evidence_exists:
            state = str(evidence.get("state", evidence.get("evidence_state", "READY")) or "READY").upper()
            if evidence_age_ms > 24 * 3600 * 1000:
                state = "STALE"
        elif pack_exists:
            state = str(pack.get("state", "READY") or "READY").upper()
            if age_ms > 24 * 3600 * 1000:
                state = "STALE"
        selected_pack_dir = str(
            control.get("selected_pack_dir", "")
            or evidence.get("selected_pack_dir", "")
            or evidence.get("pack_dir", "")
            or ""
        )
        pack_data_types = evidence.get("pack_data_types", [])
        if not isinstance(pack_data_types, list):
            pack_data_types = []
        missing_data_types = evidence.get("missing_data_types", [])
        if not isinstance(missing_data_types, list):
            missing_data_types = []
        return {
            "affects_overall": False,
            "title_cn": "复盘证据包",
            "state": state,
            "status_exists": pack_exists,
            "status_age_ms": age_ms,
            "evidence_status_exists": evidence_exists,
            "evidence_status_age_ms": evidence_age_ms,
            "evidence_replay_state": str(evidence.get("evidence_state", evidence.get("state", "")) or ""),
            "selected_pack_dir": selected_pack_dir,
            "pack_dir": str(evidence.get("pack_dir", "") or selected_pack_dir),
            "pack_data_types": pack_data_types,
            "missing_data_types": missing_data_types,
            "replay_run_dir": str(evidence.get("replay_run_dir", "") or ""),
            "replay_sync_root": str(evidence.get("replay_sync_root", "") or ""),
            "replay_speed": str(evidence.get("replay_speed", "") or ""),
            "logic_clock": str(evidence.get("logic_clock", "") or ""),
            "global_person_replay_state": str(evidence.get("global_person_replay_state", "") or ""),
            "hit_judge_replay_state": str(evidence.get("hit_judge_replay_state", "") or ""),
            "original_hit_count": _safe_int(evidence.get("original_hit_count"), 0),
            "replay_hit_count": _safe_int(evidence.get("replay_hit_count"), 0),
            "original_global_person_rows": _safe_int(evidence.get("original_global_person_rows"), 0),
            "replay_global_person_rows": _safe_int(evidence.get("replay_global_person_rows"), 0),
            "global_person_track_count": _safe_int(evidence.get("global_person_track_count"), 0),
            "diff_state": str(evidence.get("diff_state", "") or ""),
            "summary_path": str(evidence.get("summary_path", "") or ""),
            "session_id": str(evidence.get("session_id", "") or pack.get("session_id", "") or ""),
            "session_dir": str(pack.get("session_dir", "") or ""),
            "manifest_path": str(evidence.get("manifest_path", "") or pack.get("manifest_path", "") or ""),
            "replay_check_path": str(pack.get("replay_check_path", "") or ""),
            "record_replay_match": bool(pack.get("record_replay_match", False)) if pack_exists else False,
            "mvs_record_replay_match": bool(pack.get("mvs_record_replay_match", False)) if pack_exists else False,
            "event_record_replay_match": bool(pack.get("event_record_replay_match", False)) if pack_exists else False,
            "annotation_count": int(annotation_count),
            "last_annotation_kind": str(last_annotation.get("kind", "") or ""),
            "last_annotation_label": str(last_annotation.get("label", "") or ""),
            "last_annotation_age_ms": max(0.0, (time.time() * 1_000_000 - _safe_float(last_annotation.get("wall_us"), 0.0)) / 1000.0) if _safe_float(last_annotation.get("wall_us"), 0.0) > 0 else -1.0,
            "last_error": str(evidence.get("last_error", "") or pack.get("last_error", "") or last_annotation.get("error", "") or ""),
            "status": pack,
            "evidence_status": evidence,
            "control": control,
        }

    def _build_source_mode_replay_status(self) -> Dict[str, Any]:
        effective = self._read_json_status("source_mode_effective.json")
        effective_runtime = self._read_json_status("_runtime_config/source_mode_effective.json")
        manifest = self._read_json_status("v25_resolved_manifest_latest.json")
        manifest_runtime = self._read_json_status("_runtime_config/v25_resolved_manifest.json")
        replay_control = self._read_json_status("v25_replay_control.json")
        replay_status = self._read_json_status("v25_replay_status.json")

        effective_src = effective if bool(effective.get("_file_exists")) else effective_runtime
        manifest_src = manifest if bool(manifest.get("_file_exists")) else manifest_runtime
        effective_exists = bool(effective_src.get("_file_exists"))
        manifest_exists = bool(manifest_src.get("_file_exists"))
        replay_control_exists = bool(replay_control.get("_file_exists"))
        replay_status_exists = bool(replay_status.get("_file_exists"))

        event_mode = str(effective_src.get("event_source_mode_effective", "") or "").strip().lower()
        mvs_mode = str(effective_src.get("mvs_source_mode_effective", "") or "").strip().lower()
        dataset_dir = str(effective_src.get("dataset_dir", "") or manifest_src.get("dataset_dir", "") or "")
        dataset_id = str(effective_src.get("dataset_id", "") or manifest_src.get("dataset_id", "") or "")
        session_id = str(effective_src.get("session_id", "") or manifest_src.get("session_id", "") or "")
        capabilities = manifest_src.get("capabilities", effective_src.get("capabilities", {}))
        if not isinstance(capabilities, dict):
            capabilities = {}
        degraded_reasons = manifest_src.get("degraded_reasons", [])
        if not isinstance(degraded_reasons, list):
            degraded_reasons = []

        broken: List[str] = []
        degraded: List[str] = []
        if not effective_exists:
            degraded.append("source_mode_effective_missing")
        if (event_mode == "file_replay" or mvs_mode == "file_replay") and not manifest_exists:
            degraded.append("v25_resolved_manifest_missing_for_file_replay")
        if bool(effective_src.get("missing")):
            degraded.append("source_mode_effective_missing_inputs")
        if bool(effective_src.get("warnings")):
            degraded.append("source_mode_effective_warnings")
        if bool(degraded_reasons):
            degraded.append("v25_resolved_manifest_degraded")

        if not effective_exists:
            state = "MISSING"
        elif broken:
            state = "BROKEN"
        elif degraded:
            state = "DEGRADED"
        else:
            state = "OK"

        return {
            "title_cn": "启动源模式/离线回放",
            "state": state,
            "source_mode_effective_exists": effective_exists,
            "source_mode_effective_age_ms": self._status_wall_age_ms(effective_src),
            "source_mode_effective_path": str(effective_src.get("_status_path", "")),
            "source_mode_effective_runtime_path": str(effective_runtime.get("_status_path", "")),
            "event_source_mode_effective": event_mode or "-",
            "mvs_source_mode_effective": mvs_mode or "-",
            "dataset_dir": dataset_dir,
            "dataset_id": dataset_id,
            "session_id": session_id,
            "profile": str(effective_src.get("profile", "") or ""),
            "event_file_fifo_broker_enabled": bool(effective_src.get("event_file_fifo_broker_enabled", False)),
            "event_live_hal_enabled": bool(effective_src.get("event_live_hal_enabled", False)),
            "mvs_file_frame_broker_enabled": bool(effective_src.get("mvs_file_frame_broker_enabled", False)),
            "mvs_live_camera_enabled": bool(effective_src.get("mvs_live_camera_enabled", False)),
            "event_file_fifo_broker_raw_camera_count": _safe_int(effective_src.get("event_file_fifo_broker_raw_camera_count"), 0),
            "mvs_file_frame_broker_dataset_dir": str(effective_src.get("mvs_file_frame_broker_dataset_dir", "") or ""),
            "event_file_fifo_broker_replay_timing": str(effective_src.get("event_file_fifo_broker_replay_timing", "") or ""),
            "mvs_file_frame_broker_replay_timing": str(effective_src.get("mvs_file_frame_broker_replay_timing", "") or ""),
            "event_file_fifo_broker_loop": bool(effective_src.get("event_file_fifo_broker_loop", False)),
            "mvs_file_frame_broker_loop": bool(effective_src.get("mvs_file_frame_broker_loop", False)),
            "mutual_exclusion": effective_src.get("mutual_exclusion", {}),
            "warnings": effective_src.get("warnings", []),
            "missing": effective_src.get("missing", []),
            "v25_resolved_manifest_exists": manifest_exists,
            "v25_resolved_manifest_age_ms": self._status_wall_age_ms(manifest_src),
            "v25_resolved_manifest_path": str(manifest_src.get("_status_path", "")),
            "v25_resolved_manifest_runtime_path": str(manifest_runtime.get("_status_path", "")),
            "scenario": manifest_src.get("scenario", {}),
            "capabilities": capabilities,
            "usable_for_event_file_replay": bool(capabilities.get("usable_for_event_file_replay", False)),
            "usable_for_mvs_file_replay": bool(capabilities.get("usable_for_mvs_file_replay", False)),
            "usable_for_full_replay_startup": bool(capabilities.get("usable_for_full_replay_startup", False)),
            "usable_for_full_replay_sync": bool(capabilities.get("usable_for_full_replay_sync", False)),
            "degraded_reasons": degraded_reasons,
            "recommended_launch": manifest_src.get("recommended_launch", {}),
            "v25_replay_control_exists": replay_control_exists,
            "v25_replay_control_age_ms": self._status_wall_age_ms(replay_control),
            "v25_replay_status_exists": replay_status_exists,
            "v25_replay_status_age_ms": self._status_wall_age_ms(replay_status),
            "replay_control_mode": str(replay_control.get("mode", replay_status.get("mode", "")) or ""),
            "replay_timing": str(replay_control.get("replay_timing", replay_status.get("replay_timing", "")) or ""),
            "replay_launch_command": str(replay_status.get("launch_command", replay_control.get("launch_command", "")) or ""),
            "broken_links": broken,
            "degraded_links": degraded,
            "source_mode_effective": effective_src,
            "v25_resolved_manifest": manifest_src,
            "v25_replay_control": replay_control,
            "v25_replay_status": replay_status,
        }

    def _build_experiment_control_status(self) -> Dict[str, Any]:
        control = self._read_json_status("experiment_control.json")
        status = self._read_json_status("experiment_status_latest.json")
        run_status = self._read_json_status("test_run_status_latest.json")
        control_exists = bool(control.get("_file_exists"))
        status_exists = bool(status.get("_file_exists"))
        age_ms = self._status_wall_age_ms(status if status_exists else control)
        state = str(status.get("state", "") or ("READY" if control_exists else "DISABLED")).upper()
        if status_exists and age_ms >= 0 and age_ms > 5000.0:
            state = "STALE"
        hit_payload = control.get("hit_payload", {}) if isinstance(control.get("hit_payload"), dict) else {}
        global_payload = control.get("global_person_payload", {}) if isinstance(control.get("global_person_payload"), dict) else {}
        run = run_status.get("run", {}) if isinstance(run_status.get("run"), dict) else {}
        compact = run_status.get("compact", {}) if isinstance(run_status.get("compact"), dict) else {}
        return {
            "affects_overall": False,
            "title_cn": "实验热更新",
            "state": state,
            "enabled": control_exists,
            "experiment_id": str(control.get("experiment_id", status.get("experiment_id", "")) or ""),
            "stage": str(control.get("stage", status.get("stage", "")) or ""),
            "config_hash": str(control.get("config_hash", status.get("config_hash", "")) or ""),
            "source": str(status.get("source", control.get("source", "")) or ""),
            "status_age_ms": age_ms,
            "hit_profile_path": str(control.get("hit_profile_path", "") or ""),
            "global_id_profile_path": str(control.get("global_id_profile_path", "") or ""),
            "final_hit_authority": str(hit_payload.get("final_hit_authority", "")),
            "human_noise_filter_mode": str(hit_payload.get("human_noise_filter_mode", "")),
            "human_noise_track_start_bbox_expand_px": hit_payload.get("human_noise_track_start_bbox_expand_px"),
            "human_noise_track_start_bbox_policy": str(hit_payload.get("human_noise_track_start_bbox_policy", "")),
            "global_person_target": str(control.get("global_person_target", "")),
            "global_person_filter_backend": str(global_payload.get("filter_backend", "")),
            "run_state": str(run.get("state", "")),
            "run_dir": str(run.get("run_dir", "")),
            "runtime_overall_state": str(compact.get("overall_state", "")),
            "runtime_broken_links": compact.get("broken_links", []),
            "runtime_degraded_links": compact.get("degraded_links", []),
            "control": control,
            "status": status,
            "run_status": run_status,
        }

    def _build_hot_update_service_status(self, service_key: str, title_cn: str, status_json: str, latest_json: str) -> Dict[str, Any]:
        st = self._read_json_status(status_json)
        latest = self._read_json_status(latest_json)
        if not bool(st.get("_file_exists")):
            return {
                "schema_version": 1,
                "service": service_key,
                "title_cn": title_cn,
                "state": "DISABLED",
                "mode": "disabled",
                "pid": None,
                "enabled": False,
                "input_age_ms": -1.0,
                "output_age_ms": -1.0,
                "queue_size": 0,
                "dropped_count": 0,
                "last_error": "",
                "control_seq": 0,
                "status_age_ms": -1.0,
                "latest_age_ms": -1.0,
                "broken_links": [],
                "status": st,
                "latest": latest,
            }
        mode = str(st.get("mode", "disabled") or "disabled")
        enabled = bool(st.get("enabled", mode != "disabled"))
        age_ms = self._status_wall_age_ms(st)
        latest_age_ms = self._status_wall_age_ms(latest)
        raw_state = str(st.get("state", st.get("status", "")) or "").upper()
        state = raw_state if raw_state and raw_state != "UNKNOWN" else "OK"
        broken: List[str] = []
        if not enabled or mode == "disabled":
            state = "DISABLED"
        elif age_ms >= 0 and age_ms > 3000.0:
            state = "DEGRADED"
            broken.append("status_stale")
        if enabled and str(st.get("last_error", "") or ""):
            broken.append("last_error")
            if state == "OK":
                state = "DEGRADED"
        pid_value = st.get("pid") if enabled and isinstance(st.get("pid"), int) else None
        input_age_ms = _safe_float(st.get("input_age_ms"), -1.0) if enabled else -1.0
        output_age_ms = _safe_float(st.get("output_age_ms"), -1.0) if enabled else -1.0
        return {
            "schema_version": _safe_int(st.get("schema_version", 1), 1),
            "service": str(st.get("service", service_key)),
            "title_cn": str(st.get("title_cn", title_cn)),
            "state": state,
            "mode": mode,
            "pid": pid_value,
            "enabled": enabled,
            "input_age_ms": input_age_ms,
            "output_age_ms": output_age_ms,
            "queue_size": _safe_int(st.get("queue_size"), 0) if enabled else 0,
            "dropped_count": _safe_int(st.get("dropped_count"), 0) if enabled else 0,
            "last_error": str(st.get("last_error", "") or "") if enabled else "",
            "control_seq": _safe_int(st.get("control_seq"), 0),
            "status_age_ms": age_ms,
            "latest_age_ms": latest_age_ms,
            "broken_links": broken,
            "status": st,
            "latest": latest,
        }

    def _build_manual_hit_status(self, renderer: Dict[str, Any]) -> Dict[str, Any]:
        st = self._read_json_status("manual_hit_status.json")
        latest = self._read_json_status("manual_hit_latest.json")
        base = self._build_hot_update_service_status("manual_hit", "人工命中确认", "manual_hit_status.json", "manual_hit_latest.json")
        preview_manual = renderer.get("manual_hit", {}) if isinstance(renderer.get("manual_hit"), dict) else {}
        base["affects_overall"] = False
        base["preview_click_enable"] = bool(preview_manual.get("click_enable", False))
        base["preview_hit_display_enable"] = bool(preview_manual.get("preview_hit_display_enable", True))
        base["preview_trajectory_only_mode"] = bool(preview_manual.get("trajectory_only_mode", False))
        base["preview_target_cache_count"] = _safe_int(preview_manual.get("target_cache_count"), 0)
        base["preview_command_seq"] = _safe_int(preview_manual.get("command_seq"), 0)
        base["preview_last_command"] = preview_manual.get("last_command", {}) if isinstance(preview_manual.get("last_command"), dict) else {}
        base["preview_control_error"] = str(preview_manual.get("preview_control_error", "") or "")
        base["command_jsonl"] = str(st.get("command_jsonl") or preview_manual.get("command_jsonl") or self._sync_root_path("manual_hit_commands.jsonl"))
        base["confirmed_jsonl"] = str(st.get("confirmed_jsonl") or self._sync_root_path("manual_hit_confirmed.jsonl"))
        base["events_jsonl"] = str(st.get("events_jsonl") or self._sync_root_path("damage_attribution_events.jsonl"))
        base["control_json"] = str(st.get("control_json") or self._manual_hit_control_path())
        base["source_global_id"] = _safe_int(st.get("source_global_id"), 99)
        base["dedup_ms"] = _safe_float(st.get("dedup_ms"), 800.0)
        base["target_max_age_ms"] = _safe_float(st.get("target_max_age_ms"), 1200.0)
        base["stats"] = st.get("stats", {}) if isinstance(st.get("stats"), dict) else {}
        base["latest_result"] = st.get("latest_result", latest.get("latest_result", {})) if isinstance(st.get("latest_result", latest.get("latest_result", {})), dict) else {}
        if str(base.get("state", "")).upper() == "DISABLED" and bool(base.get("preview_click_enable")):
            base["state"] = "DEGRADED"
            broken = list(base.get("broken_links", []) or [])
            broken.append("manual_hit_service_status_missing")
            base["broken_links"] = broken
        return base

    def _build_outbound_hit_events_status(self) -> Dict[str, Any]:
        bridge = self._read_json_status("game_event_bridge_status.json")
        latest = self._read_json_status("game_event_bridge_latest.json")
        sent_path = self._resolve_sync_or_project_path(str(bridge.get("sent_jsonl", "") or ""), "game_event_bridge_sent.jsonl")
        rows, tail_error, file_size, file_age_ms = self._read_jsonl_tail_records(sent_path, max_records=160, tail_bytes=768_000)
        hit_rows: List[Dict[str, Any]] = []
        now_s = time.time()
        for rec in rows:
            payload = rec.get("payload", {}) if isinstance(rec.get("payload"), dict) else {}
            if str(payload.get("type", "")).lower() != "hit":
                continue
            wall_s = _safe_float(rec.get("wall_time"), 0.0)
            age_ms = max(0.0, (now_s - wall_s) * 1000.0) if wall_s > 0 else -1.0
            hit_rows.append({
                "wall_time": wall_s,
                "age_ms": age_ms,
                "mode": str(rec.get("mode", "")),
                "sent": bool(rec.get("sent", False)),
                "reason": str(rec.get("reason", "")),
                "source_yolo_id": _safe_int(payload.get("source_yolo_id"), -1),
                "target_yolo_id": _safe_int(payload.get("target_yolo_id"), -1),
                "source_global_id": str(payload.get("source_global_id", "")),
                "target_global_id": str(payload.get("target_global_id", "")),
                "target_id_source": str(payload.get("target_id_source", "")),
                "target_display_player_id": str(payload.get("target_display_player_id", "")),
                "target_display_slot_id": _safe_int(payload.get("target_display_slot_id"), -1),
                "event_id": str(payload.get("event_id", "")),
                "camera": str(payload.get("camera", "")),
                "confidence": _safe_float(payload.get("confidence"), 0.0),
                "source_override": bool(payload.get("source_override", False)),
                "event_source": str(payload.get("event_source", "")),
                "manual": bool(payload.get("manual", False)),
                "manual_authority": str(payload.get("manual_authority", "")),
                "id_semantics": str(payload.get("id_semantics", "")),
            })
        recent_hits = hit_rows[-30:]
        sent_recent = sum(1 for rec in recent_hits if bool(rec.get("sent")))
        failed_recent = sum(1 for rec in recent_hits if (not bool(rec.get("sent"))) and str(rec.get("reason", "")) == "send_failed")
        shadow_recent = sum(1 for rec in recent_hits if (not bool(rec.get("sent"))) and str(rec.get("reason", "")) == "shadow")
        stats = bridge.get("stats", {}) if isinstance(bridge.get("stats"), dict) else {}
        mode = str(bridge.get("mode", latest.get("mode", "")) or "")
        send_hits_enabled = bool(bridge.get("send_hits_enabled", False))
        client_error = str(bridge.get("client_last_error", "") or "")
        state = "OK"
        broken: List[str] = []
        if not bool(bridge.get("_file_exists")):
            state = "DISABLED"
            broken.append("bridge_status_missing")
        elif mode in {"disabled", "off"}:
            state = "DISABLED"
        elif tail_error and tail_error != "missing":
            state = "DEGRADED"
            broken.append("sent_jsonl_read_error")
        elif send_hits_enabled and not sent_path.exists():
            state = "DEGRADED"
            broken.append("sent_jsonl_missing")
        elif send_hits_enabled and client_error:
            state = "DEGRADED"
            broken.append("client_last_error")
        latest_hit = dict(recent_hits[-1]) if recent_hits else {}
        return {
            "title_cn": "已发送命中事件",
            "affects_overall": False,
            "state": state,
            "mode": mode,
            "hit_source_mode": str(bridge.get("hit_source_mode", "all") or "all"),
            "control_json": str(bridge.get("control_json", "") or ""),
            "control_error": str(bridge.get("control_error", "") or ""),
            "send_hits_enabled": send_hits_enabled,
            "send_presence_enabled": bool(bridge.get("send_presence_enabled", False)),
            "client_connected": bool(bridge.get("client_connected", False)),
            "client_last_error": client_error,
            "windows_host": str(bridge.get("windows_host", "")),
            "windows_port": bridge.get("windows_port"),
            "force_hit_source_global_id": _safe_int(bridge.get("force_hit_source_global_id"), -1),
            "virtual_presence_global_id": _safe_int(bridge.get("virtual_presence_global_id"), -1),
            "sent_jsonl": str(sent_path),
            "sent_jsonl_exists": bool(sent_path.exists()),
            "sent_jsonl_size": file_size,
            "sent_jsonl_age_ms": file_age_ms,
            "tail_error": tail_error if tail_error != "missing" else "",
            "recent_hit_count": len(recent_hits),
            "recent_sent_count": sent_recent,
            "recent_shadow_count": shadow_recent,
            "recent_send_failed_count": failed_recent,
            "total_sent_hit": _safe_int(stats.get("sent_hit"), 0),
            "total_shadow_hit": _safe_int(stats.get("shadow_hit"), 0),
            "total_send_failed_hit": _safe_int(stats.get("send_failed_hit"), 0),
            "suppress_non_manual_hit_by_source_mode": _safe_int(stats.get("suppress_non_manual_hit_by_source_mode"), 0),
            "suppress_manual_hit_by_source_mode": _safe_int(stats.get("suppress_manual_hit_by_source_mode"), 0),
            "latest_hit_age_ms": latest_hit.get("age_ms", -1.0),
            "latest_hit": latest_hit,
            "recent_hits": recent_hits,
            "stats": stats,
            "broken_links": broken,
            "bridge_status": bridge,
            "bridge_latest": latest,
        }

    def _build_preview_status(self, overlay: Dict[str, Any], renderer: Dict[str, Any]) -> Dict[str, Any]:
        per_cam: Dict[str, Any] = {}
        raw_age = getattr(getattr(self.preview, "telemetry", None), "raw_age_ms", {}) or {}
        for cam in list(getattr(self.preview, "cams", []) or []):
            per_cam[cam] = {
                "visible_frame_id": _safe_int(getattr(self.preview, "last_frame_ids", {}).get(cam, -1), -1),
                "raw_update_age_ms": _safe_float(raw_age.get(cam), -1.0) if isinstance(raw_age, dict) else -1.0,
                "reader_open": cam in getattr(self.preview, "readers", {}),
                "reader_error": str(getattr(self.preview, "reader_errors", {}).get(cam, "")),
            }
        video_backend = renderer.get("video_backend", {}) if isinstance(renderer, dict) else {}
        return {
            "title_cn": "主预览",
            "state": "OK" if str(video_backend.get("state", "")).lower() in ("ready", "running", "ok") or video_backend.get("backend_class") else "DEGRADED",
            "cuda_pbo_backend": video_backend,
            "render_fps": renderer.get("render_fps"),
            "trajectory_drawn_vtx": renderer.get("trajectory_drawn_vtx"),
            "trajectory_draw_state": renderer.get("trajectory_draw_state"),
            "trajectory_stale_guard_count": renderer.get("trajectory_stale_guard_count"),
            "trajectory_future_drop_count": renderer.get("trajectory_future_drop_count"),
            "trajectory_fresh_not_drawn": renderer.get("trajectory_fresh_not_drawn"),
            "bullet_fresh_point_count": renderer.get("bullet_fresh_point_count"),
            "bullet_stale_point_count": renderer.get("bullet_stale_point_count"),
            "bullet_future_point_count": renderer.get("bullet_future_point_count"),
            "trajectory_reader": renderer.get("trajectory_reader", {}),
            "human_jsonl_backfill": renderer.get("human_jsonl_backfill", {}),
            "human_mask_render": renderer.get("human_mask_render", {}),
            "human_label_mode": str(getattr(self.args, "human_label_mode", "detail") or "detail"),
            "overlay_counts": overlay.get("counts", {}) if isinstance(overlay, dict) else {},
            "per_camera": per_cam,
        }

    def _build_v25_shadow_services_status(self) -> Dict[str, Any]:
        enabled = bool(getattr(self.args, "v25_live_shadow_enable", False))
        root_raw = str(getattr(self.args, "v25_live_shadow_root", "v25_shadow/live") or "v25_shadow/live")
        root = Path(root_raw).expanduser()
        if not root.is_absolute():
            root = self._sync_root_path(root_raw)
        expected = {
            "visualization": ("mrg-visualization", "shadow_only", root / "visualization" / "status.json"),
            "combat_engine": ("mrg-combat-engine", "shadow_only", root / "combat" / "status.json"),
            "game_bridge": ("mrg-game-bridge", "dry_run_only", root / "game_bridge" / "status.json"),
        }
        if not enabled:
            return {
                "title_cn": "V25影子链路",
                "state": "DISABLED",
                "enabled": False,
                "affects_overall": False,
                "mode": "disabled",
                "stream_role": "shadow_bundle",
                "authority": "shadow_only_and_dry_run",
                "official_result": False,
                "official_writer_count": 0,
                "root": str(root),
                "services": {},
                "broken_links": [],
            }

        services: Dict[str, Any] = {}
        broken: List[str] = []
        degraded: List[str] = []
        sessions: set[str] = set()
        max_age_ms = max(2500.0, float(getattr(self.args, "status_window_poll_ms", 500.0) or 500.0) * 6.0)
        official_writer_count = 0
        for name, (component, authority, path) in expected.items():
            status = self._read_json_status(str(path))
            file_age_ms = _safe_float(status.get("_file_age_ms"), -1.0)
            service_state = str(status.get("state", "missing") or "missing").lower()
            session_id = str(status.get("session_id", "") or "")
            if session_id:
                sessions.add(session_id)
            writer_count = _safe_int(status.get("official_writer_count"), 0)
            official_writer_count += writer_count
            contract_ok = (
                str(status.get("component", "")) == component
                and str(status.get("authority", "")) == authority
                and status.get("official_result") is False
                and writer_count == 0
                and status.get("writes_legacy_ipc") is False
                and status.get("production_network_enabled") is False
            )
            fresh = bool(status.get("_file_exists")) and 0.0 <= file_age_ms <= max_age_ms
            if not fresh:
                broken.append(f"{name}:status_missing_or_stale")
            if not contract_ok:
                broken.append(f"{name}:authority_contract")
            if service_state == "degraded" and fresh and contract_ok:
                degraded.append(f"{name}:degraded")
            elif service_state not in {"ok", "idle", "running", "ready", "degraded"} and fresh:
                broken.append(f"{name}:state_{service_state}")
            services[name] = {
                **status,
                "expected_component": component,
                "expected_authority": authority,
                "status_fresh": fresh,
                "authority_contract_ok": contract_ok,
                "status_file_age_ms": file_age_ms,
            }
        if len(sessions) > 1:
            broken.append("session_id_mismatch")
        state = "BROKEN" if broken else ("DEGRADED" if degraded else "OK")
        return {
            "title_cn": "V25影子链路",
            "state": state,
            "enabled": True,
            "affects_overall": True,
            "mode": "live_shadow",
            "stream_role": "shadow_bundle",
            "authority": "shadow_only_and_dry_run",
            "official_result": False,
            "official_writer_count": int(official_writer_count),
            "production_network_enabled": False,
            "session_id": next(iter(sessions)) if len(sessions) == 1 else "",
            "root": str(root),
            "services": services,
            "degraded_links": degraded,
            "broken_links": broken,
        }

    def build_system_status(self) -> Dict[str, Any]:
        try:
            telem = self.preview.telemetry.poll(self.preview.last_frame_ids)
        except Exception:
            telem = getattr(getattr(self.preview, "telemetry", None), "last_snapshot", {}) or {}
        overlay = self._overlay_summary()
        renderer = self._renderer_summary()
        categories = {
            "mvs": self._build_mvs_status(telem),
            "event": self._build_event_status(telem, renderer),
            "global_yolo": self._build_global_yolo_status(telem),
            "global_person_id": self._build_global_person_status(),
            "global_bev": self._build_global_bev_status(telem),
            "hit_judge": self._build_hit_judge_status(),
            "bullet_reprocess": self._build_bullet_reprocess_status(),
            "raw_recording": self._build_raw_recording_status(),
            "person_id_dataset": self._build_person_id_dataset_status(),
            "experiment_control": self._build_experiment_control_status(),
            "global_bullet_fusion": self._build_hot_update_service_status("global_bullet_fusion", "全局弹道融合", "global_bullet_fusion_status.json", "global_bullet_fusion_latest.json"),
            "damage_attribution": self._build_hot_update_service_status("damage_attribution", "伤害来源判定", "damage_attribution_status.json", "damage_attribution_latest.json"),
            "foam_dart_speed": self._build_hot_update_service_status("foam_dart_speed", "海绵弹速度监测", "foam_dart_speed_status.json", "foam_dart_speed_latest.json"),
            "game_event_bridge": self._build_hot_update_service_status("game_event_bridge", "游戏事件桥接", "game_event_bridge_status.json", "game_event_bridge_latest.json"),
            "manual_hit": self._build_manual_hit_status(renderer),
            "outbound_hit_events": self._build_outbound_hit_events_status(),
            "replay_evidence": self._build_replay_evidence_status(),
            "source_mode_replay": self._build_source_mode_replay_status(),
            "v25_shadow_services": self._build_v25_shadow_services_status(),
            "preview": self._build_preview_status(overlay, renderer),
        }
        broken: List[str] = []
        degraded: List[str] = []
        for key, cat in categories.items():
            state = str(cat.get("state", "UNKNOWN")).upper() if isinstance(cat, dict) else "UNKNOWN"
            mode = str(cat.get("mode", "")).lower() if isinstance(cat, dict) else ""
            enabled = bool(cat.get("enabled", True)) if isinstance(cat, dict) else True
            affects_overall = bool(cat.get("affects_overall", True)) if isinstance(cat, dict) else True
            if not affects_overall:
                continue
            if state == "DISABLED" and (not enabled or mode == "disabled"):
                pass
            elif state in ("BROKEN", "ERROR", "MISSING"):
                broken.append(key)
            elif state not in ("OK", "RUNNING", "READY", "RECORDING"):
                degraded.append(key)
            if isinstance(cat, dict) and isinstance(cat.get("broken_links"), list):
                broken.extend([f"{key}:{x}" for x in cat.get("broken_links", [])])
        overall = "OK" if not broken and not degraded else ("BROKEN" if broken else "DEGRADED")
        chain = categories["global_yolo"].get("legacy_yolo", {}) if isinstance(categories.get("global_yolo"), dict) else {}
        required_status_categories = (
            "mvs",
            "event",
            "global_yolo",
            "global_person_id",
            "global_bev",
            "hit_judge",
            "bullet_reprocess",
            "raw_recording",
            "person_id_dataset",
            "source_mode_replay",
            "v25_shadow_services",
            "preview",
        )
        status_key_ok = all(isinstance(categories.get(key), dict) for key in required_status_categories)
        return {
            "schema_version": 1,
            "kind": "system_status_latest",
            "ts_wall_us": int(time.time() * 1_000_000),
            "mono_ns": int(time.monotonic_ns()),
            "status_cn": "正常" if overall == "OK" else ("断链" if overall == "BROKEN" else "降级"),
            "summary": {
                "overall_state": overall,
                "broken_links": broken,
                "degraded_links": degraded,
                "status_key_ok": bool(status_key_ok),
                "legacy_yolo_shards_enabled": bool(chain.get("enabled", False)),
                "legacy_yolo_reason": str(chain.get("reason", "")),
                "v25_shadow_enabled": bool(categories.get("v25_shadow_services", {}).get("enabled", False)),
                "v25_shadow_state": str(categories.get("v25_shadow_services", {}).get("state", "DISABLED")),
            },
            "categories": categories,
            "legacy": {
                "status_latest_compatible": True,
                "status_summary_csv_compatible": True,
                "legacy_yolo_shards": chain,
            },
        }

    def system_status_to_tabs(self, system_status: Dict[str, Any]) -> Dict[str, List[Tuple[str, List[Tuple[str, str]]]]]:
        cats = system_status.get("categories", {}) if isinstance(system_status, dict) else {}
        summary = system_status.get("summary", {}) if isinstance(system_status, dict) else {}
        mvs_overview = cats.get("mvs", {}) if isinstance(cats, dict) else {}
        event_overview = cats.get("event", {}) if isinstance(cats, dict) else {}
        yolo_overview = cats.get("global_yolo", {}) if isinstance(cats, dict) else {}
        person_overview = cats.get("global_person_id", {}) if isinstance(cats, dict) else {}
        bev_overview = cats.get("global_bev", {}) if isinstance(cats, dict) else {}
        hit_overview = cats.get("hit_judge", {}) if isinstance(cats, dict) else {}
        raw_overview = cats.get("raw_recording", {}) if isinstance(cats, dict) else {}
        pds_overview = cats.get("person_id_dataset", {}) if isinstance(cats, dict) else {}
        exp_overview = cats.get("experiment_control", {}) if isinstance(cats, dict) else {}
        src_overview = cats.get("source_mode_replay", {}) if isinstance(cats, dict) else {}
        shadow_overview = cats.get("v25_shadow_services", {}) if isinstance(cats, dict) else {}
        shadow_services = shadow_overview.get("services", {}) if isinstance(shadow_overview.get("services"), dict) else {}
        bev_layer_overview = bev_overview.get("event_layer", {}) if isinstance(bev_overview.get("event_layer"), dict) else {}
        bev_bullet_overview = bev_overview.get("event_bullet", {}) if isinstance(bev_overview.get("event_bullet"), dict) else {}
        broken_links = summary.get("broken_links", []) or []
        degraded_links = summary.get("degraded_links", []) or []
        tabs: Dict[str, List[Tuple[str, List[Tuple[str, str]]]]] = {
            "总览": [
                ("系统健康", [
                    ("overall_state", str(summary.get("overall_state", ""))),
                    ("status_cn", str(system_status.get("status_cn", ""))),
                    ("broken_count", str(len(broken_links))),
                    ("degraded_count", str(len(degraded_links))),
                    ("broken_top", ",".join(str(x) for x in broken_links[:6]) or "-"),
                    ("degraded_top", ",".join(str(x) for x in degraded_links[:6]) or "-"),
                    ("legacy_yolo_shards_enabled", str(bool(summary.get("legacy_yolo_shards_enabled", False)))),
                ]),
                ("性能速览", [
                    ("mvs_active", str(mvs_overview.get("active_count"))),
                    ("event_active", str(event_overview.get("active_count"))),
                    ("global_yolo_fps", str(yolo_overview.get("published_fps_total", yolo_overview.get("infer_fps")))),
                    ("global_yolo_inferred_fps", str(yolo_overview.get("inferred_fps_total"))),
                    ("global_yolo_queue_depth", str(yolo_overview.get("postprocess_queue_depth"))),
                    ("global_yolo_polygon_ok", str(yolo_overview.get("polygon_contract_ok"))),
                    ("global_yolo_workers", str(yolo_overview.get("worker_count"))),
                    ("global_bev_render_fps", str(bev_overview.get("render_fps"))),
                    ("global_bev_output_fps", str(bev_overview.get("output_fps"))),
                    ("pbo_ready", str(bev_overview.get("pbo_ready"))),
                ]),
                ("启动源模式", [
                    ("source_mode_state", str(src_overview.get("state") or "-")),
                    ("event_source_mode_effective", str(src_overview.get("event_source_mode_effective") or "-")),
                    ("mvs_source_mode_effective", str(src_overview.get("mvs_source_mode_effective") or "-")),
                    ("dataset_id", str(src_overview.get("dataset_id") or "-")),
                    ("session_id", str(src_overview.get("session_id") or "-")),
                    ("full_replay_startup", str(src_overview.get("usable_for_full_replay_startup"))),
                    ("full_replay_sync", str(src_overview.get("usable_for_full_replay_sync"))),
                ]),
                ("V25影子服务", [
                    ("enabled", str(shadow_overview.get("enabled"))),
                    ("state", str(shadow_overview.get("state") or "DISABLED")),
                    ("session_id", str(shadow_overview.get("session_id") or "-")),
                    ("official_result", str(shadow_overview.get("official_result"))),
                    ("official_writer_count", str(shadow_overview.get("official_writer_count"))),
                    ("visualization", str((shadow_services.get("visualization") or {}).get("state", "-"))),
                    ("combat_engine", str((shadow_services.get("combat_engine") or {}).get("state", "-"))),
                    ("game_bridge", str((shadow_services.get("game_bridge") or {}).get("state", "-"))),
                ]),
                ("业务速览", [
                    ("person_track_count", str(person_overview.get("track_count"))),
                    ("hit_detection_enabled", str(hit_overview.get("hit_detection_enabled"))),
                    ("hit_candidate_total", str(hit_overview.get("hit_candidate_total"))),
                    ("raw_recording", str(raw_overview.get("recording"))),
                    ("person_dataset_recording", str(pds_overview.get("recording"))),
                    ("person_dataset_rows", str(pds_overview.get("human_rows_total"))),
                    ("experiment_id", str(exp_overview.get("experiment_id") or "-")),
                    ("experiment_stage", str(exp_overview.get("stage") or "-")),
                ]),
                ("显示控制状态", [
                    ("event_layer_enabled", str(bev_layer_overview.get("enabled"))),
                    ("event_layer_draw_cam_count", str(bev_layer_overview.get("draw_cam_count"))),
                    ("event_layer_alpha", str(bev_layer_overview.get("alpha"))),
                    ("event_layer_color_mode", str(bev_layer_overview.get("color_mode"))),
                    ("event_layer_dilate", f"{bev_layer_overview.get('dilate_px')}px/{bev_layer_overview.get('dilate_backend') or '-'}"),
                    ("event_bullet_enabled", str(bev_bullet_overview.get("enabled"))),
                    ("event_bullet_draw_points", str(bev_bullet_overview.get("draw_points"))),
                ]),
            ],
        }

        def add_cat(tab: str, key: str, rows: List[Tuple[str, Any]]) -> None:
            cat = cats.get(key, {}) if isinstance(cats, dict) else {}
            base = [("state", str(cat.get("state", ""))), ("title_cn", str(cat.get("title_cn", "")))]
            base.extend((str(k), str(v)) for k, v in rows)
            tabs.setdefault(tab, []).append(("核心指标", base))

        mvs = cats.get("mvs", {}) if isinstance(cats, dict) else {}
        add_cat("MVS取流", "mvs", [("active_count", mvs.get("active_count")), ("stuck_cameras", ",".join(mvs.get("stuck_cameras", []) or []) or "-")])
        mvs_rows = []
        for cam, rec in (mvs.get("per_camera", {}) if isinstance(mvs.get("per_camera"), dict) else {}).items():
            mvs_rows.append((cam, f"frame_id={rec.get('frame_id')} frame_age={rec.get('frame_age_ms')}ms shm_age={rec.get('shm_age_ms')}ms stuck={rec.get('stuck')} raw_fps={rec.get('raw_fps')} mvs_fps={rec.get('mvs_fps')}"))
        tabs.setdefault("MVS取流", []).append(("每路相机", mvs_rows))

        event = cats.get("event", {}) if isinstance(cats, dict) else {}
        add_cat("事件相机链路", "event", [("active_count", event.get("active_count")), ("broken_links", ",".join(event.get("broken_links", []) or []) or "-")])
        event_rows = []
        for cam, rec in (event.get("per_camera", {}) if isinstance(event.get("per_camera"), dict) else {}).items():
            ext = rec.get("ext_trigger", {}) if isinstance(rec.get("ext_trigger"), dict) else {}
            bd = rec.get("bullet_display_diag", {}) if isinstance(rec.get("bullet_display_diag"), dict) else {}
            drop_counts = rec.get("human_mask_buffer_drop_counts", {})
            if not isinstance(drop_counts, dict):
                drop_counts = {}
            hot = rec.get("event_hot_pixel_filter", {}) if isinstance(rec.get("event_hot_pixel_filter"), dict) else {}
            sanitize = rec.get("event_cluster_sanitize", {}) if isinstance(rec.get("event_cluster_sanitize"), dict) else {}
            shadow = rec.get("line_filter_shadow_diag", {}) if isinstance(rec.get("line_filter_shadow_diag"), dict) else {}
            event_rows.append((cam, f"worker={rec.get('worker_state')} age={rec.get('worker_age_ms')}ms loop={rec.get('loop_fps')} track_path={rec.get('track_path_position')} decoupled={rec.get('track_decoupled_from_mask')} blocks_tracking={rec.get('mask_buffer_blocks_tracking')} track_policy={rec.get('track_input_policy_effective')} raw={rec.get('track_raw_event_count')} track={rec.get('track_event_count')}/{rec.get('track_event_count_before_hot_pixel')} released={rec.get('mask_released_event_count')} filtered={rec.get('mask_filtered_event_count')} block={rec.get('event_block_stage_reason')} spatial_collapse={rec.get('event_spatial_collapse_suspect')} hot_rm={hot.get('removed_count','-')}/{hot.get('input_count','-')} hot_px={len(hot.get('active_hot_pixels', []) or [])} sanitize_rm={sanitize.get('removed_count','-')}/{sanitize.get('input_count','-')} shadow_acc={shadow.get('accepted_count','-')} shadow_draw={shadow.get('draw_paths_count','-')} viz_only={rec.get('mask_release_consumed_for_viz_only')} overlay={rec.get('actual_draw_state')} ov_age={rec.get('overlay_shm_age_ms')}ms ov_seq={rec.get('overlay_shm_seq')} source={rec.get('overlay_source')} dilate={rec.get('dilate_backend')}:{rec.get('dilate_px')}px bullet_state={bd.get('bullet_source_state', rec.get('bullet_source_state',''))} active={bd.get('active_track_count','-')} display={bd.get('display_eligible_count','-')} draw={bd.get('draw_paths_count','-')} pts={bd.get('draw_path_points_count','-')} point_age={bd.get('last_overlay_bullet_point_age_ms','-')}ms sender_sent={bd.get('bullet_sender_sent','-')} sender_q={bd.get('bullet_sender_queue_size','-')} sender_drop={bd.get('bullet_sender_dropped','-')} sender_drop_delta={bd.get('bullet_sender_drop_delta','-')} vpoint_submit={bd.get('visual_point_submit_count','-')} vpoint_fail={bd.get('visual_point_submit_failed_count','-')} vpoint_skip={bd.get('visual_point_skip_count','-')} mask={rec.get('viz_mask_valid')} buffer={rec.get('human_mask_buffer_reason')} mask_timeout_drop={drop_counts.get('mask_timeout_drop','-')} selected_edges={ext.get('selected_edge_count','-')} skipped_nonselected={ext.get('nonselected_log_skipped_edges','-')} lock={rec.get('sync_lock_state')}"))
        tabs.setdefault("事件相机链路", []).append(("每路事件 worker/overlay", event_rows))

        mask_svc = event.get("mask_evidence_service", {}) if isinstance(event.get("mask_evidence_service"), dict) else {}
        mask_raster_cfg = mask_svc.get("mask_raster", {}) if isinstance(mask_svc.get("mask_raster"), dict) else {}
        mask_rows = []
        raster_rows = []
        visual_rows = []
        for cam, rec in (event.get("per_camera", {}) if isinstance(event.get("per_camera"), dict) else {}).items():
            mask_rows.append((cam, f"fresh={rec.get('mask_evidence_fresh')} age={rec.get('mask_evidence_age_ms')}ms sync={rec.get('mask_evidence_sync_index')} persons={rec.get('mask_evidence_person_count')} polygons={rec.get('mask_evidence_polygon_count')} bboxes={rec.get('mask_evidence_bbox_count')}"))
            raster_rows.append((cam, f"enabled={rec.get('mask_raster_enabled')} seq={rec.get('mask_raster_seq')} pixels={rec.get('mask_raster_pixels')} persons={rec.get('mask_raster_used_persons')} polygons={rec.get('mask_raster_used_polygons')} proj={rec.get('mask_raster_projection_state') or '-'} h={rec.get('mask_raster_homography_source') or '-'} in_bounds={rec.get('mask_raster_projected_in_bounds_points')}/{rec.get('mask_raster_projected_total_points')} raw_bbox={rec.get('mask_raster_raw_bbox_xyxy')} event_bbox={rec.get('mask_raster_projected_bbox_xyxy')} err={rec.get('mask_raster_error') or '-'}"))
            hv = rec.get("hal_visual_mask_overlay", {}) if isinstance(rec.get("hal_visual_mask_overlay"), dict) else {}
            visual_rows.append((cam, f"enabled={hv.get('enabled')} fresh={hv.get('mask_valid')} hold={hv.get('mask_hold_active')} data={hv.get('mask_data_available')} policy={hv.get('filter_policy') or '-'} slice={hv.get('slice_us')}us acc={hv.get('accumulation_us')}us ctrl_seq={hv.get('control_seq')} apply={hv.get('control_apply_count')} sync_ts={hv.get('sync_timestamp_policy') or '-'} mask_age={hv.get('mask_age_ms')}ms raw={hv.get('raw_events_seen')} kept={hv.get('filtered_events_seen')} dropped={hv.get('dropped_by_mask')} reduction={hv.get('reduction_pct')}% nonzero={hv.get('nonzero_pixels')} pub={hv.get('publish_count')} err={hv.get('control_error') or hv.get('last_error') or '-'}"))
        tabs.setdefault("事件相机链路", []).append(("mask evidence service", [
            ("enabled", str(mask_svc.get("enabled"))),
            ("period_ms", str(mask_svc.get("period_ms"))),
            ("effective_fps", str(mask_svc.get("effective_fps"))),
            ("fresh_count", str(mask_svc.get("fresh_count"))),
            ("camera_count", str(mask_svc.get("camera_count"))),
            ("mask_raster_enabled", str(mask_raster_cfg.get("enabled"))),
            ("mask_raster_size", f"{mask_raster_cfg.get('width')}x{mask_raster_cfg.get('height')}"),
            ("mask_raster_dilate_px", str(mask_raster_cfg.get("dilate_px"))),
            ("status_age_ms", str(mask_svc.get("age_ms"))),
            ("control_seq", str(mask_svc.get("control_seq"))),
            ("control_error", str(mask_svc.get("control_error") or "-")),
        ]))
        tabs.setdefault("事件相机链路", []).append(("per-camera mask evidence", mask_rows))
        tabs.setdefault("事件相机链路", []).append(("per-camera mask raster", raster_rows))
        hv_svc = event.get("hal_visual_mask_overlay", {}) if isinstance(event.get("hal_visual_mask_overlay"), dict) else {}
        tabs.setdefault("事件相机链路", []).append(("HAL visual mask overlay", [
            ("enabled", str(hv_svc.get("enabled"))),
            ("broker_status_exists", str(hv_svc.get("broker_status_exists"))),
            ("broker_status_age_ms", str(hv_svc.get("broker_status_age_ms"))),
            ("broker_tail_error", str(hv_svc.get("broker_tail_error") or "-")),
        ]))
        tabs.setdefault("事件相机链路", []).append(("per-camera HAL visual mask overlay", visual_rows))

        gy = cats.get("global_yolo", {}) if isinstance(cats, dict) else {}
        add_cat("全局YOLO", "global_yolo", [
            ("published_fps_total", gy.get("published_fps_total", gy.get("infer_fps"))),
            ("inferred_fps_total", gy.get("inferred_fps_total")),
            ("capacity_fps", gy.get("capacity_fps")),
            ("fps_source", gy.get("fps_source")),
            ("batch_size", gy.get("batch_size")),
            ("worker_count", gy.get("worker_count")),
            ("partial_batches", gy.get("partial_batches")),
            ("predict_ms", gy.get("predict_ms")),
            ("total_ms", gy.get("total_ms")),
            ("async_record_postprocess_enabled", gy.get("async_record_postprocess_enabled")),
            ("postprocess_queue_depth", gy.get("postprocess_queue_depth")),
            ("postprocess_dropped_batches", gy.get("postprocess_dropped_batches")),
            ("polygon_contract_ok", gy.get("polygon_contract_ok")),
            ("published_persons_with_polygon_ratio", gy.get("published_persons_with_polygon_ratio")),
            ("merge_age_ms", gy.get("merge_age_ms")),
        ])
        yolo_rows = []
        for wid, rec in (gy.get("workers", {}) if isinstance(gy.get("workers"), dict) else {}).items():
            yolo_rows.append((wid, f"state={rec.get('state')} fps={rec.get('fps')} published={rec.get('published_fps_total')} inferred={rec.get('inferred_fps_total')} fps_source={rec.get('fps_source')} capacity_fps={rec.get('capacity_fps')} batch={rec.get('batch_size')} predict_ms={rec.get('predict_ms')} total_ms={rec.get('total_ms')} queue={rec.get('postprocess_queue_depth')} drop={rec.get('postprocess_dropped_batches')} polygon_ok={rec.get('polygon_contract_ok')} partial={rec.get('partial_batches')} full={rec.get('full_batches')} age={rec.get('age_ms')}ms"))
        tabs.setdefault("全局YOLO", []).append(("双 worker", yolo_rows))

        gp = cats.get("global_person_id", {}) if isinstance(cats, dict) else {}
        tabs.setdefault("Global Person ID", []).append(("hot-update / shadow", [
            ("track_count", str(gp.get("track_count"))),
            ("filter_backend", str(gp.get("filter_backend") or "-")),
            ("control_seq", str(gp.get("control_seq"))),
            ("created_tracks", str(gp.get("created_tracks"))),
            ("retired_tracks", str(gp.get("retired_tracks"))),
            ("id_switch_suspects", str(gp.get("id_switch_suspects"))),
            ("shadow_track_count", str(gp.get("shadow_track_count"))),
            ("shadow_filter_backend", str(gp.get("shadow_filter_backend") or "-")),
            ("shadow_control_seq", str(gp.get("shadow_control_seq"))),
            ("shadow_created_tracks", str(gp.get("shadow_created_tracks"))),
            ("shadow_retired_tracks", str(gp.get("shadow_retired_tracks"))),
            ("shadow_id_switch_suspects", str(gp.get("shadow_id_switch_suspects"))),
            ("shadow_status_age_ms", str(gp.get("shadow_status_age_ms"))),
        ]))
        add_cat("全局人员ID", "global_person_id", [("track_count", gp.get("track_count")), ("duplicate_suppress_enable", gp.get("duplicate_suppress_enable")), ("coord_transform_mode", gp.get("coord_transform_mode")), ("lost_id_count", gp.get("lost_id_count")), ("status_age_ms", gp.get("status_age_ms"))])

        gb = cats.get("global_bev", {}) if isinstance(cats, dict) else {}
        gb_layer = gb.get("event_layer", {}) if isinstance(gb.get("event_layer"), dict) else {}
        if not gb_layer:
            gb_layer = gb.get("event_layer_overlay", {}) if isinstance(gb.get("event_layer_overlay"), dict) else {}
        gb_layer_presentation = gb_layer.get("presentation_worker", {}) if isinstance(gb_layer.get("presentation_worker"), dict) else {}
        gb_bullet = gb.get("event_bullet", {}) if isinstance(gb.get("event_bullet"), dict) else {}
        if not gb_bullet:
            gb_bullet = gb.get("event_bullet_overlay", {}) if isinstance(gb.get("event_bullet_overlay"), dict) else {}
        gb_mvs_upload = gb.get("mvs_texture_upload", {}) if isinstance(gb.get("mvs_texture_upload"), dict) else {}
        gb_mvs_sync = gb.get("mvs_sync", {}) if isinstance(gb.get("mvs_sync"), dict) else {}
        gb_presentation = gb.get("presentation", {}) if isinstance(gb.get("presentation"), dict) else {}
        gb_alignment = gb.get("alignment", gb.get("alignment_shadow", {})) if isinstance(gb, dict) else {}
        if not isinstance(gb_alignment, dict):
            gb_alignment = {}
        gb_align_mvs = gb_alignment.get("mvs", {}) if isinstance(gb_alignment.get("mvs"), dict) else {}
        gb_align_event = gb_alignment.get("event_layer", {}) if isinstance(gb_alignment.get("event_layer"), dict) else {}
        def _align_cam_summary(section):
            per = section.get("per_camera", {}) if isinstance(section, dict) else {}
            if not isinstance(per, dict):
                return {}
            out = {}
            for cam, rec in per.items():
                if not isinstance(rec, dict):
                    continue
                out[str(cam)] = {
                    "state": rec.get("state"),
                    "delta_ms": rec.get("sync_delta_ms_est", rec.get("event_to_bev_sync_delta_ms")),
                    "selected_id": rec.get("selected_id", rec.get("frame_id")),
                    "reason": rec.get("reason"),
                }
            return out
        tabs.setdefault("全局BEV", []).append(("同步诊断", [
            ("bev_presentation_delay_enable", gb_presentation.get("enabled")),
            ("bev_presentation_delay_ms", gb_presentation.get("delay_ms")),
            ("bev_effective_lag_ms_est", gb_presentation.get("effective_lag_ms_est")),
            ("bev_latest_common_sync_id", gb_presentation.get("latest_common_sync_id")),
            ("bev_target_sync_id", gb_presentation.get("target_sync_id")),
            ("bev_sync_tolerance_ms", gb_presentation.get("sync_tolerance_ms")),
            ("mvs_target_sync_id", gb_mvs_sync.get("target_sync_id")),
            ("mvs_mixed_frame_detected", gb_mvs_sync.get("mixed_frame_detected")),
            ("mvs_raw_sync_span_ticks", gb_mvs_sync.get("raw_sync_span_ticks")),
            ("mvs_raw_sync_span_ms_est", gb_mvs_sync.get("raw_sync_span_ms_est")),
            ("mvs_selected_sync_span_ticks", gb_mvs_sync.get("selected_sync_span_ticks")),
            ("mvs_selected_sync_span_ms_est", gb_mvs_sync.get("selected_sync_span_ms_est")),
            ("mvs_hidden_cams", gb_mvs_sync.get("hidden_by_mvs_sync_delta_cams")),
            ("event_layer_sync_mode", gb_layer.get("sync_mode")),
            ("event_layer_max_sync_delta_ticks", gb_layer.get("max_sync_delta_ticks")),
            ("event_layer_max_sync_delta_ms", gb_layer.get("max_sync_delta_ms")),
            ("event_layer_sync_normal_ms", gb_layer.get("sync_normal_ms")),
            ("event_layer_sync_warn_ms", gb_layer.get("sync_warn_ms")),
            ("event_layer_sync_hard_ms", gb_layer.get("sync_hard_ms")),
            ("event_layer_hold_last_good_on_hard_outlier", gb_layer.get("hold_last_good_on_hard_outlier")),
            ("event_layer_hold_last_good_ms", gb_layer.get("hold_last_good_ms")),
            ("event_layer_event_sync_delta_by_cam", gb_layer.get("event_sync_delta_by_cam")),
            ("event_layer_event_sync_delta_ms_by_cam", gb_layer.get("event_sync_delta_ms_by_cam")),
            ("event_layer_sync_display_state_by_cam", gb_layer.get("event_sync_display_state_by_cam")),
            ("event_layer_sync_display_level_by_cam", gb_layer.get("event_sync_display_level_by_cam")),
            ("event_layer_sync_warn_cams", gb_layer.get("event_sync_warn_cams")),
            ("event_layer_sync_hard_outlier_cams", gb_layer.get("event_sync_hard_outlier_cams")),
            ("event_layer_sync_hold_cams", gb_layer.get("event_sync_hold_cams")),
            ("event_layer_event_sync_hidden_cams", gb_layer.get("event_sync_hidden_cams")),
            ("event_layer_sync_hidden_cams_strict", gb_layer.get("event_sync_hidden_cams_strict")),
            ("event_layer_history_enable", gb_layer.get("history_enable")),
            ("event_layer_history_selected_cam_count", gb_layer.get("history_selected_cam_count")),
            ("event_layer_history_latest_fallback_cam_count", gb_layer.get("history_latest_fallback_cam_count")),
            ("event_layer_history_source_by_cam", gb_layer.get("history_source_by_cam")),
            ("event_layer_sync_timestamp_policy", gb_layer.get("sync_timestamp_policy")),
            ("alignment_mode", gb_alignment.get("mode")),
            ("alignment_state", gb_alignment.get("state")),
            ("alignment_overall_state", gb_alignment.get("overall_state")),
            ("alignment_target_sync_id", gb_alignment.get("target_sync_id")),
            ("alignment_delay_ms", gb_alignment.get("delay_ms")),
            ("alignment_age_ms", gb_alignment.get("updated_age_ms")),
            ("alignment_elapsed_ms", gb_alignment.get("elapsed_ms")),
            ("alignment_worker_elapsed_p95_ms", gb_alignment.get("worker_elapsed_p95_ms")),
            ("alignment_worker_interval_ms", gb_alignment.get("worker_interval_ms")),
            ("alignment_gui_refresh_p95_ms", gb_alignment.get("gui_refresh_p95_ms")),
            ("alignment_gui_upload_selected_p95_ms", gb_alignment.get("gui_upload_selected_p95_ms")),
            ("alignment_gui_upload_event_layer_p95_ms", gb_alignment.get("gui_upload_event_layer_p95_ms")),
            ("alignment_gui_event_sparse_read_p95_ms", gb_alignment.get("gui_event_sparse_read_p95_ms")),
            ("alignment_gui_event_sparse_sync_gate_p95_ms", gb_alignment.get("gui_event_sparse_sync_gate_p95_ms")),
            ("alignment_gui_event_sparse_project_p95_ms", gb_alignment.get("gui_event_sparse_project_p95_ms")),
            ("alignment_gui_paint_total_p95_ms", gb_alignment.get("gui_paint_total_p95_ms")),
            ("mvs_upload_budget_ms", gb_mvs_upload.get("budget_ms")),
            ("mvs_upload_elapsed_ms", gb_mvs_upload.get("elapsed_ms")),
            ("mvs_upload_budget_overrun_rate", gb_mvs_upload.get("budget_overrun_rate")),
            ("mvs_upload_held_cam_count_p95", gb_mvs_upload.get("held_cam_count_p95")),
            ("mvs_upload_order", gb_mvs_upload.get("upload_order")),
            ("mvs_upload_uploaded_cams", gb_mvs_upload.get("uploaded_cams")),
            ("mvs_upload_duplicate_skip_cams", gb_mvs_upload.get("duplicate_skip_cams")),
            ("mvs_upload_held_by_budget_cams", gb_mvs_upload.get("held_by_budget_cams")),
            ("event_layer_display_budget_ms", gb_layer.get("display_budget_ms")),
            ("event_layer_display_budget_elapsed_ms", gb_layer.get("display_budget_elapsed_ms")),
            ("event_layer_display_budget_elapsed_p95_ms", gb_layer.get("display_budget_elapsed_p95_ms")),
            ("event_layer_display_budget_overrun_rate", gb_layer.get("display_budget_overrun_rate")),
            ("event_layer_display_budget_held_cam_count_p95", gb_layer.get("display_budget_held_cam_count_p95")),
            ("event_layer_display_budget_order", gb_layer.get("display_budget_order")),
            ("event_layer_display_budget_held_cams", gb_layer.get("display_budget_held_cams")),
            ("event_layer_sparse_read_elapsed_ms", gb_layer.get("sparse_read_elapsed_ms")),
            ("event_layer_sparse_sync_gate_elapsed_ms", gb_layer.get("sparse_sync_gate_elapsed_ms")),
            ("event_layer_sparse_project_elapsed_ms", gb_layer.get("sparse_project_elapsed_ms")),
            ("event_layer_presentation_worker_enabled", gb_layer_presentation.get("enabled")),
            ("event_layer_presentation_worker_mode", gb_layer_presentation.get("mode")),
            ("event_layer_presentation_worker_state", gb_layer_presentation.get("state")),
            ("event_layer_presentation_worker_worker_state", gb_layer_presentation.get("worker_state")),
            ("event_layer_presentation_worker_payload_policy", gb_layer_presentation.get("payload_policy")),
            ("event_layer_presentation_worker_compare", gb_layer_presentation.get("compare_state")),
            ("event_layer_presentation_worker_mismatch_cam_count", gb_layer_presentation.get("mismatch_cam_count")),
            ("event_layer_presentation_worker_display_mismatch_cam_count", gb_layer_presentation.get("display_mismatch_cam_count")),
            ("event_layer_presentation_worker_delta_mismatch_cam_count", gb_layer_presentation.get("delta_mismatch_cam_count")),
            ("event_layer_presentation_worker_compare_lookup_state", gb_layer_presentation.get("compare_lookup_state")),
            ("event_layer_presentation_worker_compare_target_delta_ticks", gb_layer_presentation.get("compare_target_delta_ticks")),
            ("event_layer_presentation_worker_compare_tolerance_ticks", gb_layer_presentation.get("compare_tolerance_ticks")),
            ("event_layer_presentation_worker_compare_hit_ratio", gb_layer_presentation.get("compare_hit_ratio")),
            ("event_layer_presentation_worker_compare_exact_count", gb_layer_presentation.get("compare_exact_count")),
            ("event_layer_presentation_worker_compare_miss_count", gb_layer_presentation.get("compare_miss_count")),
            ("event_layer_presentation_worker_target_delta_ticks", gb_layer_presentation.get("target_delta_ticks")),
            ("event_layer_presentation_worker_updated_age_ms", gb_layer_presentation.get("updated_age_ms")),
            ("event_layer_presentation_worker_build_p95_ms", gb_layer_presentation.get("build_p95_ms")),
            ("event_layer_presentation_worker_build_elapsed_ms", gb_layer_presentation.get("build_elapsed_ms")),
            ("event_layer_presentation_worker_active_consume_elapsed_ms", gb_layer_presentation.get("active_consume_elapsed_ms")),
            ("event_layer_presentation_worker_active_consume_count", gb_layer_presentation.get("active_consume_count")),
            ("event_layer_presentation_worker_fallback_count", gb_layer_presentation.get("fallback_count")),
            ("event_layer_presentation_worker_fallback_reason", gb_layer_presentation.get("fallback_reason")),
            ("event_layer_presentation_worker_bundle_hit_ratio", gb_layer_presentation.get("bundle_hit_ratio")),
            ("event_layer_presentation_worker_bundle_hit_count", gb_layer_presentation.get("bundle_hit_count")),
            ("event_layer_presentation_worker_bundle_miss_count", gb_layer_presentation.get("bundle_miss_count")),
            ("event_layer_presentation_worker_consume_exact_count", gb_layer_presentation.get("consume_exact_count")),
            ("event_layer_presentation_worker_consume_near_count", gb_layer_presentation.get("consume_near_count")),
            ("event_layer_presentation_worker_bundle_ring_count", gb_layer_presentation.get("bundle_ring_count")),
            ("event_layer_presentation_worker_cached_target_min", gb_layer_presentation.get("cached_target_min")),
            ("event_layer_presentation_worker_cached_target_max", gb_layer_presentation.get("cached_target_max")),
            ("alignment_cpu_prep_only", gb_alignment.get("cpu_prep_only")),
            ("alignment_opengl_thread", gb_alignment.get("opengl_thread")),
            ("alignment_compare", gb_alignment.get("compare_to_active")),
            ("alignment_mvs_offset_ms_by_cam", gb_alignment.get("mvs_offset_ms_by_cam")),
            ("alignment_event_delta_ms_by_cam", gb_alignment.get("event_delta_ms_by_cam")),
            ("alignment_quality", gb_alignment.get("quality")),
            ("alignment_mvs_cam_state", _align_cam_summary(gb_align_mvs)),
            ("alignment_event_cam_state", _align_cam_summary(gb_align_event)),
        ]))
        add_cat("全局BEV", "global_bev", [("render_fps", gb.get("render_fps")), ("output_fps", gb.get("output_fps")), ("pbo_ready", gb.get("pbo_ready")), ("pbo_mode", gb.get("pbo_mode")), ("pbo_ready_count", gb.get("pbo_ready_count")), ("sidecar_age_ms", gb.get("sidecar_age_ms")), ("alignment_state", gb_alignment.get("overall_state", gb_alignment.get("state"))), ("alignment_target_sync_id", gb_alignment.get("target_sync_id")), ("alignment_mvs_synced", (gb_alignment.get("quality") or {}).get("mvs_synced_cam_count") if isinstance(gb_alignment.get("quality"), dict) else None), ("alignment_mvs_hold_warn", (gb_alignment.get("quality") or {}).get("mvs_hold_warn_cam_count") if isinstance(gb_alignment.get("quality"), dict) else None), ("event_layer_enabled", gb_layer.get("enabled")), ("event_layer_draw_cam_count", gb_layer.get("draw_cam_count")), ("event_layer_sync_timestamp_policy", gb_layer.get("sync_timestamp_policy")), ("event_layer_transparent_background", gb_layer.get("transparent_background")), ("event_layer_alpha", gb_layer.get("alpha")), ("event_layer_color_mode", gb_layer.get("color_mode")), ("event_layer_dilate_px", gb_layer.get("dilate_px")), ("event_layer_dilate_backend", gb_layer.get("dilate_backend")), ("event_layer_dilate_skipped_dense_cam_count", gb_layer.get("dilate_skipped_dense_cam_count")), ("event_layer_skip_reason", gb_layer.get("skip_reason")), ("event_bullet_enabled", gb_bullet.get("enabled")), ("event_bullet_draw_points", gb_bullet.get("draw_points")), ("event_bullet_recent_records", gb_bullet.get("recent_records")), ("event_bullet_records_buffered", gb_bullet.get("records_buffered")), ("event_bullet_skip_reason", gb_bullet.get("skip_reason")), ("event_bullet_latest_age_ms", gb_bullet.get("latest_age_ms")), ("event_bullet_last_record_camera", gb_bullet.get("last_record_camera")), ("event_bullet_last_record_age_ms", gb_bullet.get("last_record_age_ms"))])

        tabs.setdefault("全局BEV", []).append(("event_layer_presentation_worker", [
            ("mode", gb_layer_presentation.get("mode")),
            ("state", gb_layer_presentation.get("state")),
            ("worker_state", gb_layer_presentation.get("worker_state")),
            ("payload_policy", gb_layer_presentation.get("payload_policy")),
            ("compare_state", gb_layer_presentation.get("compare_state")),
            ("display_compare_state", gb_layer_presentation.get("display_compare_state")),
            ("delta_compare_state", gb_layer_presentation.get("delta_compare_state")),
            ("mismatch_cam_count", gb_layer_presentation.get("mismatch_cam_count")),
            ("display_mismatch_cam_count", gb_layer_presentation.get("display_mismatch_cam_count")),
            ("delta_mismatch_cam_count", gb_layer_presentation.get("delta_mismatch_cam_count")),
            ("compare_lookup_state", gb_layer_presentation.get("compare_lookup_state")),
            ("compare_target_delta_ticks", gb_layer_presentation.get("compare_target_delta_ticks")),
            ("compare_tolerance_ticks", gb_layer_presentation.get("compare_tolerance_ticks")),
            ("compare_hit_ratio", gb_layer_presentation.get("compare_hit_ratio")),
            ("compare_exact_count", gb_layer_presentation.get("compare_exact_count")),
            ("compare_miss_count", gb_layer_presentation.get("compare_miss_count")),
            ("fallback_count", gb_layer_presentation.get("fallback_count")),
            ("fallback_reason", gb_layer_presentation.get("fallback_reason")),
            ("active_consume_count", gb_layer_presentation.get("active_consume_count")),
            ("bundle_hit_ratio", gb_layer_presentation.get("bundle_hit_ratio")),
            ("bundle_hit_count", gb_layer_presentation.get("bundle_hit_count")),
            ("bundle_miss_count", gb_layer_presentation.get("bundle_miss_count")),
            ("consume_exact_count", gb_layer_presentation.get("consume_exact_count")),
            ("consume_near_count", gb_layer_presentation.get("consume_near_count")),
            ("bundle_ring_count", gb_layer_presentation.get("bundle_ring_count")),
            ("cached_target_min", gb_layer_presentation.get("cached_target_min")),
            ("cached_target_max", gb_layer_presentation.get("cached_target_max")),
            ("build_p95_ms", gb_layer_presentation.get("build_p95_ms")),
            ("updated_age_ms", gb_layer_presentation.get("updated_age_ms")),
        ]))

        hj = cats.get("hit_judge", {}) if isinstance(cats, dict) else {}
        hit_control = hj.get("control", {}) if isinstance(hj.get("control"), dict) else {}
        stage_total = hj.get("stage_total", {}) if isinstance(hj.get("stage_total"), dict) else {}
        result_total = hj.get("result_total", {}) if isinstance(hj.get("result_total"), dict) else {}
        tabs.setdefault("命中判定", []).append(("强门控 / 强推理", [
            ("hit_detection_enabled", hj.get("hit_detection_enabled")),
            ("hit_candidate_output_enabled", hj.get("hit_candidate_output_enabled")),
            ("strong_gate_enabled", hj.get("strong_gate_enabled")),
            ("strong_reasoning_enabled", hj.get("strong_reasoning_enabled")),
            ("strong_reasoning_shadow_only", hj.get("strong_reasoning_shadow_only")),
            ("final_hit_authority", hj.get("final_hit_authority")),
            ("pseudo_hit_enable", hj.get("pseudo_hit_enable")),
            ("pseudo_hit_total", hj.get("pseudo_hit_total")),
            ("pseudo_hit_promoted_to_final_count", hj.get("pseudo_hit_promoted_to_final_count")),
            ("pseudo_hit_duplicate_suppressed", hj.get("pseudo_hit_duplicate_suppressed")),
            ("strong_gate_output_suppressed", hj.get("strong_gate_output_suppressed")),
            ("strong_reasoning_output_suppressed", hj.get("strong_reasoning_output_suppressed")),
            ("pseudo_hit_reason_total", hj.get("pseudo_hit_reason_total")),
            ("pseudo_hit_by_camera", hj.get("pseudo_hit_by_camera")),
        ]))
        add_cat("命中判定", "hit_judge", [("hit_detection_enabled", hj.get("hit_detection_enabled")), ("control_seq", hit_control.get("seq")), ("control_error", hit_control.get("error") or "-"), ("recv_total", hj.get("recv_total")), ("hit_candidate_total", hj.get("hit_candidate_total")), ("visual_point_active_track_keys", hj.get("visual_point_active_track_keys")), ("visual_point_pending_human", hj.get("visual_point_pending_human")), ("udp_recv", stage_total.get("udp_recv")), ("segment_attempt", stage_total.get("segment_attempt")), ("point_emitted", result_total.get("point_emitted")), ("hit_detection_disabled", result_total.get("hit_detection_disabled")), ("point_no_human", result_total.get("point_no_human")), ("point_no_mask_contact", result_total.get("point_no_mask_contact"))])
        hj_cam_rows = []
        for cam, rec in (hj.get("visual_point_stage_by_camera", {}) if isinstance(hj.get("visual_point_stage_by_camera"), dict) else {}).items():
            if isinstance(rec, dict):
                hj_cam_rows.append((cam, f"udp={rec.get('udp_recv', 0)} projected={rec.get('point_projected', 0)} segment={rec.get('segment_attempt', 0)} wait_prev={rec.get('waiting_prev_point', 0)} emitted={rec.get('result:point_emitted', 0)} no_human={rec.get('result:point_no_human', 0)} no_mask={rec.get('result:point_no_mask_contact', 0)}"))
        tabs.setdefault("命中判定", []).append(("每路点流判定阶段", hj_cam_rows))

        manual_hit = cats.get("manual_hit", {}) if isinstance(cats, dict) else {}
        mh_stats = manual_hit.get("stats", {}) if isinstance(manual_hit.get("stats"), dict) else {}
        mh_latest = manual_hit.get("latest_result", {}) if isinstance(manual_hit.get("latest_result"), dict) else {}
        add_cat("人工命中确认", "manual_hit", [
            ("state", manual_hit.get("state") or "-"),
            ("mode", manual_hit.get("mode") or "-"),
            ("preview_click_enable", manual_hit.get("preview_click_enable")),
            ("preview_hit_display_enable", manual_hit.get("preview_hit_display_enable")),
            ("preview_trajectory_only_mode", manual_hit.get("preview_trajectory_only_mode")),
            ("preview_target_cache_count", manual_hit.get("preview_target_cache_count")),
            ("source_global_id", manual_hit.get("source_global_id")),
            ("dedup_ms", manual_hit.get("dedup_ms")),
            ("input_commands", mh_stats.get("input_commands")),
            ("accepted", mh_stats.get("accepted")),
            ("publishable_events", mh_stats.get("publishable_events")),
            ("latest_accepted", mh_latest.get("accepted")),
            ("latest_target", mh_latest.get("target_display_player_id") or mh_latest.get("target_global_id") or "-"),
            ("latest_reject_reason", mh_latest.get("reject_reason") or "-"),
            ("command_jsonl", manual_hit.get("command_jsonl") or "-"),
            ("confirmed_jsonl", manual_hit.get("confirmed_jsonl") or "-"),
            ("control_json", manual_hit.get("control_json") or "-"),
        ])

        out_hit = cats.get("outbound_hit_events", {}) if isinstance(cats, dict) else {}
        add_cat("已发送命中事件", "outbound_hit_events", [
            ("mode", out_hit.get("mode") or "-"),
            ("hit_source_mode", out_hit.get("hit_source_mode") or "-"),
            ("send_hits_enabled", out_hit.get("send_hits_enabled")),
            ("client_connected", out_hit.get("client_connected")),
            ("client_last_error", out_hit.get("client_last_error") or "-"),
            ("control_error", out_hit.get("control_error") or "-"),
            ("windows_target", f"{out_hit.get('windows_host') or '-'}:{out_hit.get('windows_port') or '-'}"),
            ("force_hit_source_global_id", out_hit.get("force_hit_source_global_id")),
            ("virtual_presence_global_id", out_hit.get("virtual_presence_global_id")),
            ("recent_hit_count", out_hit.get("recent_hit_count")),
            ("recent_sent_count", out_hit.get("recent_sent_count")),
            ("recent_shadow_count", out_hit.get("recent_shadow_count")),
            ("recent_send_failed_count", out_hit.get("recent_send_failed_count")),
            ("total_sent_hit", out_hit.get("total_sent_hit")),
            ("total_shadow_hit", out_hit.get("total_shadow_hit")),
            ("total_send_failed_hit", out_hit.get("total_send_failed_hit")),
            ("suppress_non_manual_hit_by_source_mode", out_hit.get("suppress_non_manual_hit_by_source_mode")),
            ("suppress_manual_hit_by_source_mode", out_hit.get("suppress_manual_hit_by_source_mode")),
            ("latest_hit_age_ms", out_hit.get("latest_hit_age_ms")),
            ("sent_jsonl_age_ms", out_hit.get("sent_jsonl_age_ms")),
            ("control_json", out_hit.get("control_json") or "-"),
            ("sent_jsonl", out_hit.get("sent_jsonl") or "-"),
            ("tail_error", out_hit.get("tail_error") or "-"),
        ])
        sent_rows = []
        for idx, rec in enumerate(list(out_hit.get("recent_hits", []) or [])[-24:], 1):
            if not isinstance(rec, dict):
                continue
            event_id = str(rec.get("event_id", "") or f"event_{idx}")
            key = f"{idx:02d} {rec.get('camera') or '-'} {event_id[-12:]}"
            value = (
                f"age={_safe_float(rec.get('age_ms'), -1.0):.0f}ms "
                f"sent={rec.get('sent')} reason={rec.get('reason') or '-'} "
                f"src={rec.get('source_yolo_id')} tgt={rec.get('target_yolo_id')} "
                f"target_global={rec.get('target_global_id') or '-'} display={rec.get('target_display_player_id') or '-'} "
                f"id_source={rec.get('target_id_source') or '-'} conf={_safe_float(rec.get('confidence'), 0.0):.2f} "
                f"manual={rec.get('manual')} event_source={rec.get('event_source') or '-'} override={rec.get('source_override')}"
            )
            sent_rows.append((key, value))
        tabs.setdefault("已发送命中事件", []).append(("最近 outbound hit", sent_rows or [("recent", "-")]))

        br = cats.get("bullet_reprocess", {}) if isinstance(cats, dict) else {}
        add_cat("弹道重处理", "bullet_reprocess", [("scan_state", br.get("scan_state")), ("latest_trajectory_count", br.get("latest_trajectory_count")), ("output_file_age_ms", br.get("output_file_age_ms"))])

        rr = cats.get("raw_recording", {}) if isinstance(cats, dict) else {}
        rr_mvs = rr.get("mvs", {}) if isinstance(rr.get("mvs"), dict) else {}
        rr_event = rr.get("event_raw", {}) if isinstance(rr.get("event_raw"), dict) else {}
        add_cat("原始数据录制", "raw_recording", [
            ("recording", rr.get("recording")),
            ("session_id", rr.get("session_id") or "-"),
            ("last_command", rr.get("last_command") or "-"),
            ("last_command_targets", rr.get("last_command_targets") or "-"),
            ("command_age_ms", rr.get("command_age_ms")),
            ("mvs_status_mode", rr_mvs.get("status_mode")),
            ("mvs_status_count", rr_mvs.get("status_count")),
            ("mvs_expected_status_count", rr_mvs.get("expected_status_count")),
            ("mvs_recording", rr_mvs.get("recording")),
            ("mvs_written_frames_total", rr_mvs.get("written_frames_total")),
            ("mvs_dropped_frames_total", rr_mvs.get("dropped_frames_total")),
            ("mvs_queue_size_total", rr_mvs.get("queue_size_total")),
            ("event_recording_count", rr_event.get("recording_count")),
            ("event_status_count", rr_event.get("status_count")),
            ("event_slices_written_total", rr_event.get("slices_written_total")),
            ("event_events_written_total", rr_event.get("events_written_total")),
            ("event_dropped_slices_total", rr_event.get("dropped_slices_total")),
            ("event_queue_size_total", rr_event.get("queue_size_total")),
            ("degraded_links", ",".join(rr.get("degraded_links", []) or []) or "-"),
        ])
        tabs.setdefault("原始数据录制", []).append(("控制文件", [
            ("command_file", rr.get("command_file")),
            ("status_file", rr.get("status_file")),
            ("record_root", rr_mvs.get("record_root") or "-"),
            ("session_dir", rr_mvs.get("session_dir") or "-"),
            ("routes_enabled", ",".join(str(x) for x in (rr_mvs.get("routes_enabled", []) or [])) or "-"),
            ("video_fps", rr_mvs.get("video_fps")),
            ("fourcc", rr_mvs.get("fourcc")),
        ]))
        mvs_rec_rows = []
        for cam, rec in (rr_mvs.get("per_camera", {}) if isinstance(rr_mvs.get("per_camera"), dict) else {}).items():
            if isinstance(rec, dict):
                mvs_rec_rows.append((cam, f"enabled={rec.get('enabled')} writer_alive={rec.get('writer_alive')} writer_open={rec.get('writer_open')} queue={rec.get('queue_size')}/{rec.get('queue_limit')} peak={rec.get('queue_peak')} written={rec.get('written_frames')} dropped={rec.get('dropped_frames')} last_idx={rec.get('last_record_frame_idx')} err={rec.get('last_error') or '-'} path={rec.get('video_path') or rec.get('session_dir') or '-'}"))
        tabs.setdefault("原始数据录制", []).append(("MVS raw 每路", mvs_rec_rows))
        event_rec_rows = []
        for cam, rec in (rr_event.get("per_camera", {}) if isinstance(rr_event.get("per_camera"), dict) else {}).items():
            if isinstance(rec, dict):
                event_rec_rows.append((cam, f"raw_recording={rec.get('raw_recording')} worker={rec.get('worker_state')} age={rec.get('status_age_ms')}ms stale={rec.get('stale')} backend={rec.get('backend') or '-'} writer={rec.get('writer_alive')} queue={rec.get('queue_size')}/{rec.get('queue_limit')} slices={rec.get('slices_written')} events={rec.get('events_written')} dropped={rec.get('dropped_slices')} err={rec.get('last_error') or '-'} path={rec.get('data_path') or rec.get('session_dir') or '-'}"))
        tabs.setdefault("原始数据录制", []).append(("Event raw 每路", event_rec_rows))

        rr_broker = rr.get("event_hal_broker", {}) if isinstance(rr.get("event_hal_broker"), dict) else {}
        broker_rec_rows = []
        for cam, rec in (rr_broker.get("per_camera", {}) if isinstance(rr_broker.get("per_camera"), dict) else {}).items():
            if isinstance(rec, dict):
                broker_rec_rows.append((cam, f"recording={rec.get('recording')} enabled={rec.get('enabled')} session={rec.get('session_id') or '-'} cmd={rec.get('last_command') or '-'} seq={rec.get('last_seq')} starts={rec.get('start_count')} stops={rec.get('stop_count')} err={rec.get('last_error') or '-'} path={rec.get('path') or '-'}"))
        tabs.setdefault("Event HAL broker raw", []).append(("per camera", broker_rec_rows))

        pds = cats.get("person_id_dataset", {}) if isinstance(cats, dict) else {}
        add_cat("人员ID数据集", "person_id_dataset", [
            ("recording", pds.get("recording")),
            ("dataset_kind", pds.get("dataset_kind") or "-"),
            ("dataset_type", pds.get("dataset_type") or "-"),
            ("session_id", pds.get("session_id") or "-"),
            ("contains_mask_data", pds.get("contains_mask_data")),
            ("contains_polygon_data", pds.get("contains_polygon_data")),
            ("include_mask_polygons", pds.get("include_mask_polygons")),
            ("include_bullet_trajectory", pds.get("include_bullet_trajectory")),
            ("include_hit_debug", pds.get("include_hit_debug")),
            ("source_stage", pds.get("source_stage") or "-"),
            ("human_rows_total", pds.get("human_rows_total")),
            ("human_person_rows_total", pds.get("human_person_rows_total")),
            ("human_zero_person_rows_total", pds.get("human_zero_person_rows_total")),
            ("human_zero_person_skipped_total", pds.get("human_zero_person_skipped_total")),
            ("zero_person_keep_ms", pds.get("zero_person_keep_ms")),
            ("global_person_track_rows", pds.get("global_person_track_rows")),
            ("bullet_point_rows", pds.get("bullet_point_rows")),
            ("bullet_event_rows", pds.get("bullet_event_rows")),
            ("pseudo_hit_rows", pds.get("pseudo_hit_rows")),
            ("final_hit_rows", pds.get("final_hit_rows")),
            ("shadow_hit_rows", pds.get("shadow_hit_rows")),
            ("hit_reject_rows", pds.get("hit_reject_rows")),
            ("status_samples", pds.get("status_samples")),
            ("annotation_rows", pds.get("annotation_rows")),
            ("sanitized_mask_key_count", pds.get("sanitized_mask_key_count")),
            ("bad_lines", pds.get("bad_lines")),
            ("last_write_age_ms", pds.get("last_write_age_ms")),
            ("status_age_ms", pds.get("status_age_ms")),
            ("last_command", pds.get("last_command") or "-"),
            ("last_error", pds.get("last_error") or "-"),
        ])
        pds_rows = []
        for cam, rec in (pds.get("per_camera", {}) if isinstance(pds.get("per_camera"), dict) else {}).items():
            if isinstance(rec, dict):
                pds_rows.append((cam, f"rows={rec.get('human_rows')} persons={rec.get('person_rows')} zero={rec.get('zero_person_rows')} zero_skip={rec.get('zero_person_skipped')} sync={rec.get('last_sync_index')} frame={rec.get('last_frame_id')} age={rec.get('last_record_age_ms')}ms mask_removed={rec.get('sanitized_mask_key_count')}"))
        tabs.setdefault("人员ID数据集", []).append(("YOLO后识别每路", pds_rows))
        tabs.setdefault("人员ID数据集", []).append(("文件", [
            ("session_dir", pds.get("session_dir") or "-"),
            ("output_root", pds.get("output_root") or "-"),
        ]))

        rep = cats.get("replay_evidence", {}) if isinstance(cats, dict) else {}
        add_cat("复盘证据包", "replay_evidence", [
            ("session_id", rep.get("session_id") or "-"),
            ("record_replay_match", rep.get("record_replay_match")),
            ("mvs_record_replay_match", rep.get("mvs_record_replay_match")),
            ("event_record_replay_match", rep.get("event_record_replay_match")),
            ("annotation_count", rep.get("annotation_count")),
            ("last_annotation_kind", rep.get("last_annotation_kind") or "-"),
            ("last_annotation_label", rep.get("last_annotation_label") or "-"),
            ("last_annotation_age_ms", rep.get("last_annotation_age_ms")),
            ("last_error", rep.get("last_error") or "-"),
            ("evidence_replay_state", rep.get("evidence_replay_state") or rep.get("state") or "-"),
            ("selected_pack_dir", rep.get("selected_pack_dir") or "-"),
            ("pack_data_types", ",".join(str(x) for x in (rep.get("pack_data_types", []) or [])) or "-"),
            ("missing_data_types", ",".join(str(x) for x in (rep.get("missing_data_types", []) or [])) or "-"),
            ("replay_run_dir", rep.get("replay_run_dir") or "-"),
            ("replay_speed", rep.get("replay_speed") or "-"),
            ("logic_clock", rep.get("logic_clock") or "-"),
            ("global_person_replay_state", rep.get("global_person_replay_state") or "-"),
            ("hit_judge_replay_state", rep.get("hit_judge_replay_state") or "-"),
            ("original_hit_count", rep.get("original_hit_count")),
            ("replay_hit_count", rep.get("replay_hit_count")),
            ("original_global_person_rows", rep.get("original_global_person_rows")),
            ("replay_global_person_rows", rep.get("replay_global_person_rows")),
            ("global_person_track_count", rep.get("global_person_track_count")),
            ("diff_state", rep.get("diff_state") or "-"),
        ])
        tabs.setdefault("复盘证据包", []).append(("文件", [
            ("session_dir", rep.get("session_dir") or "-"),
            ("manifest_path", rep.get("manifest_path") or "-"),
            ("replay_check_path", rep.get("replay_check_path") or "-"),
            ("summary_path", rep.get("summary_path") or "-"),
            ("replay_sync_root", rep.get("replay_sync_root") or "-"),
        ]))

        src = cats.get("source_mode_replay", {}) if isinstance(cats, dict) else {}
        src_caps = src.get("capabilities", {}) if isinstance(src.get("capabilities"), dict) else {}
        src_scenario = src.get("scenario", {}) if isinstance(src.get("scenario"), dict) else {}
        add_cat("启动/回放源", "source_mode_replay", [
            ("event_source_mode_effective", src.get("event_source_mode_effective") or "-"),
            ("mvs_source_mode_effective", src.get("mvs_source_mode_effective") or "-"),
            ("dataset_id", src.get("dataset_id") or "-"),
            ("session_id", src.get("session_id") or "-"),
            ("dataset_label", src_scenario.get("dataset_label") or "-"),
            ("event_file_fifo_broker_enabled", src.get("event_file_fifo_broker_enabled")),
            ("event_live_hal_enabled", src.get("event_live_hal_enabled")),
            ("mvs_file_frame_broker_enabled", src.get("mvs_file_frame_broker_enabled")),
            ("mvs_live_camera_enabled", src.get("mvs_live_camera_enabled")),
            ("event_file_fifo_broker_raw_camera_count", src.get("event_file_fifo_broker_raw_camera_count")),
            ("mvs_file_frame_broker_dataset_dir", src.get("mvs_file_frame_broker_dataset_dir") or "-"),
            ("event_replay_timing", src.get("event_file_fifo_broker_replay_timing") or "-"),
            ("mvs_replay_timing", src.get("mvs_file_frame_broker_replay_timing") or "-"),
            ("usable_for_event_file_replay", src_caps.get("usable_for_event_file_replay")),
            ("usable_for_mvs_file_replay", src_caps.get("usable_for_mvs_file_replay")),
            ("usable_for_full_replay_startup", src_caps.get("usable_for_full_replay_startup")),
            ("usable_for_full_replay_sync", src_caps.get("usable_for_full_replay_sync")),
            ("degraded_reasons", ",".join(str(x) for x in (src.get("degraded_reasons", []) or [])) or "-"),
            ("replay_control_mode", src.get("replay_control_mode") or "-"),
            ("replay_timing", src.get("replay_timing") or "-"),
        ])
        tabs.setdefault("启动/回放源", []).append(("状态文件", [
            ("source_mode_effective_exists", src.get("source_mode_effective_exists")),
            ("source_mode_effective_age_ms", src.get("source_mode_effective_age_ms")),
            ("source_mode_effective_path", src.get("source_mode_effective_path") or "-"),
            ("v25_resolved_manifest_exists", src.get("v25_resolved_manifest_exists")),
            ("v25_resolved_manifest_age_ms", src.get("v25_resolved_manifest_age_ms")),
            ("v25_resolved_manifest_path", src.get("v25_resolved_manifest_path") or "-"),
            ("v25_replay_control_exists", src.get("v25_replay_control_exists")),
            ("v25_replay_status_exists", src.get("v25_replay_status_exists")),
            ("replay_launch_command", src.get("replay_launch_command") or "-"),
        ]))

        exp = cats.get("experiment_control", {}) if isinstance(cats, dict) else {}
        add_cat("实验热更新", "experiment_control", [
            ("experiment_id", exp.get("experiment_id") or "-"),
            ("stage", exp.get("stage") or "-"),
            ("config_hash", exp.get("config_hash") or "-"),
            ("source", exp.get("source") or "-"),
            ("status_age_ms", exp.get("status_age_ms")),
            ("final_hit_authority", exp.get("final_hit_authority") or "-"),
            ("human_noise_filter_mode", exp.get("human_noise_filter_mode") or "-"),
            ("human_noise_track_start_bbox_expand_px", exp.get("human_noise_track_start_bbox_expand_px")),
            ("human_noise_track_start_bbox_policy", exp.get("human_noise_track_start_bbox_policy") or "-"),
            ("global_person_target", exp.get("global_person_target") or "-"),
            ("global_person_filter_backend", exp.get("global_person_filter_backend") or "-"),
            ("run_state", exp.get("run_state") or "-"),
            ("runtime_overall_state", exp.get("runtime_overall_state") or "-"),
            ("runtime_broken_links", ",".join(str(x) for x in (exp.get("runtime_broken_links", []) or [])) or "-"),
            ("runtime_degraded_links", ",".join(str(x) for x in (exp.get("runtime_degraded_links", []) or [])) or "-"),
            ("run_dir", exp.get("run_dir") or "-"),
        ])

        hot_rows = []
        for key in ("global_bullet_fusion", "damage_attribution", "manual_hit", "foam_dart_speed", "game_event_bridge"):
            rec = cats.get(key, {}) if isinstance(cats, dict) else {}
            hot_rows.append((
                key,
                f"state={rec.get('state')} mode={rec.get('mode')} enabled={rec.get('enabled')} "
                f"pid={rec.get('pid')} input_age={rec.get('input_age_ms')}ms output_age={rec.get('output_age_ms')}ms "
                f"queue={rec.get('queue_size')} dropped={rec.get('dropped_count')} control_seq={rec.get('control_seq')} "
                f"error={rec.get('last_error') or '-'}",
            ))
        tabs.setdefault("A测热更新", []).append(("独立进程状态", hot_rows))

        pv = cats.get("preview", {}) if isinstance(cats, dict) else {}
        tr = pv.get("trajectory_reader", {}) if isinstance(pv.get("trajectory_reader"), dict) else {}
        tf = pv.get("trajectory_filter_human_noise", {}) if isinstance(pv.get("trajectory_filter_human_noise"), dict) else {}
        hb = pv.get("human_jsonl_backfill", {}) if isinstance(pv.get("human_jsonl_backfill"), dict) else {}
        hm = pv.get("human_mask_render", {}) if isinstance(pv.get("human_mask_render"), dict) else {}
        add_cat("主预览", "preview", [("render_fps", pv.get("render_fps")), ("trajectory_drawn_vtx", pv.get("trajectory_drawn_vtx")), ("trajectory_draw_state", pv.get("trajectory_draw_state")), ("trajectory_fresh_not_drawn", pv.get("trajectory_fresh_not_drawn")), ("trajectory_filter_enable", tf.get("enable")), ("trajectory_filter_segments", tf.get("filtered_segment_count")), ("trajectory_filter_tracks", tf.get("suppressed_track_count")), ("trajectory_filter_total", tf.get("suppressed_total")), ("trajectory_filter_rx", tf.get("pseudo_hit_tailer_suppressed_rx")), ("trajectory_stale_guard_count", pv.get("trajectory_stale_guard_count")), ("trajectory_future_drop_count", pv.get("trajectory_future_drop_count")), ("bullet_fresh_point_count", pv.get("bullet_fresh_point_count")), ("trajectory_reader_open", tr.get("open")), ("trajectory_reader_error", tr.get("error")), ("human_backfill_count", hb.get("count")), ("human_backfill_error", hb.get("error")), ("human_label_mode", pv.get("human_label_mode")), ("human_mask_drawn", hm.get("drawn")), ("human_mask_source", hm.get("source")), ("human_mask_count", hm.get("mask_count")), ("human_mask_bbox_fill", hm.get("bbox_fill_count")), ("human_mask_polygon_fill", hm.get("polygon_fill_count")), ("human_mask_label_mode", hm.get("label_mode"))])
        preview_rows = []
        for cam, rec in (pv.get("per_camera", {}) if isinstance(pv.get("per_camera"), dict) else {}).items():
            preview_rows.append((cam, f"visible_frame_id={rec.get('visible_frame_id')} raw_age={rec.get('raw_update_age_ms')}ms reader_open={rec.get('reader_open')} err={rec.get('reader_error') or '-'}"))
        tabs.setdefault("主预览", []).append(("每路可见光更新", preview_rows))
        return tabs

    def _csv_row(self, telem: Dict[str, Any], overlay: Dict[str, Any], renderer: Dict[str, Any]) -> Dict[str, Any]:
        trig = telem.get("trigger", {}) if isinstance(telem, dict) else {}
        ev = telem.get("event", {}) if isinstance(telem, dict) else {}
        yolo = telem.get("yolo", {}) if isinstance(telem, dict) else {}
        hit = telem.get("hit", {}) if isinstance(telem, dict) else {}
        bullet = telem.get("bullet", {}) if isinstance(telem, dict) else {}
        sync = telem.get("sync_health", {}) if isinstance(telem, dict) else {}
        video_backend = renderer.get("video_backend", {}) if isinstance(renderer.get("video_backend", {}), dict) else {}
        human_mask_render = renderer.get("human_mask_render", {}) if isinstance(renderer.get("human_mask_render", {}), dict) else {}
        row: Dict[str, Any] = {
            "ts_wall_us": int(time.time() * 1_000_000),
            "mono_ns": int(time.monotonic_ns()),
            "sample_id": self.records,
            "trigger_state": trig.get("state", ""),
            "trigger_fps": trig.get("fps"),
            "trigger_period_ms": trig.get("period_ms"),
            "trigger_jitter_p95_ms": trig.get("jitter_p95_ms"),
            "trigger_last_age_ms": trig.get("last_age_ms"),
            "ok_cams": trig.get("ok_cams"),
            "mvs_active_count": telem.get("mvs_active_count"),
            "event_active_count": ev.get("event_active_count"),
            "overlay_ws_connected": int(bool(overlay.get("ws_connected"))),
            "overlay_ws_rx_count": overlay.get("ws_rx_count"),
            "overlay_ws_age_ms": overlay.get("ws_age_ms"),
            "human_jsonl_rx_count": overlay.get("human_jsonl_rx_count"),
            "hit_jsonl_rx_count": overlay.get("hit_jsonl_rx_count"),
            "yolo_infer_fps": yolo.get("infer_fps"),
            "yolo_published_fps_total": yolo.get("published_fps_total", yolo.get("infer_fps")),
            "yolo_inferred_fps_total": yolo.get("inferred_fps_total"),
            "yolo_async_record_postprocess_enabled": int(bool(yolo.get("async_record_postprocess_enabled", False))),
            "yolo_postprocess_queue_depth": yolo.get("postprocess_queue_depth"),
            "yolo_postprocess_dropped_batches": yolo.get("postprocess_dropped_batches"),
            "yolo_polygon_contract_ok": yolo.get("polygon_contract_ok"),
            "yolo_published_persons_with_polygon_ratio": yolo.get("published_persons_with_polygon_ratio"),
            "yolo_capacity_fps": yolo.get("capacity_fps"),
            "yolo_fps_source": yolo.get("fps_source"),
            "yolo_worker_count": yolo.get("worker_count"),
            "yolo_batch_size": yolo.get("batch_size"),
            "yolo_predict_ms": yolo.get("predict_ms"),
            "yolo_capture_to_result_ms": yolo.get("capture_to_result_ms"),
            "hit_udp_recv_per_s": hit.get("recv_per_s"),
            "hit_eval_per_s": hit.get("eval_per_s"),
            "hit_candidate_per_s": hit.get("candidate_per_s"),
            "hit_confirmed_per_s": hit.get("confirmed_per_s"),
            "hit_no_human_per_s": hit.get("no_human_per_s"),
            "hit_no_human_count": hit.get("no_human_count"),
            "hit_pending_count": hit.get("pending_count"),
            "hit_pending_impacts": hit.get("pending_impacts"),
            "render_fps": renderer.get("render_fps"),
            "renderer_paint_ms": renderer.get("renderer_paint_ms"),
            "frame_period_ms": renderer.get("frame_period_ms"),
            "tick_interval_ms": renderer.get("tick_interval_ms"),
            "paint_interval_ms": renderer.get("paint_interval_ms"),
            "trajectory_drawn_vtx": renderer.get("trajectory_drawn_vtx"),
            "human_mask_drawn": int(bool(human_mask_render.get("drawn", False))),
            "human_mask_source": human_mask_render.get("source"),
            "human_mask_count": human_mask_render.get("mask_count"),
            "human_mask_person_count": human_mask_render.get("person_count"),
            "human_mask_polygon_fill_count": human_mask_render.get("polygon_fill_count"),
            "human_mask_bbox_fill_count": human_mask_render.get("bbox_fill_count"),
            "human_mask_latest_age_ms": human_mask_render.get("latest_age_ms"),
            "bullet_udp_recv_per_s": bullet.get("udp_recv_per_s"),
            "bullet_track_count_active": bullet.get("track_count_active"),
            "bullet_latest_age_ms": bullet.get("latest_age_ms"),
            "bullet_latest_cam": bullet.get("latest_cam"),
            "bullet_latest_id": bullet.get("latest_id"),
            "bullet_status_ttl_ms": bullet.get("bullet_status_ttl_ms"),
            "bullet_fresh_point_count": bullet.get("fresh_point_count"),
            "bullet_stale_point_count": bullet.get("stale_point_count"),
            "bullet_future_point_count": bullet.get("future_point_count"),
            "bullet_stale_drop_count": bullet.get("stale_drop_count"),
            "trajectory_stale_guard_count": bullet.get("trajectory_stale_guard_count"),
            "trajectory_future_drop_count": bullet.get("trajectory_future_drop_count"),
            "trajectory_fresh_not_drawn": bullet.get("trajectory_fresh_not_drawn"),
            "trajectory_draw_state": bullet.get("trajectory_draw_state"),
            "sync_latest_cam": sync.get("cam", sync.get("latest_cam", "")),
            "sync_bullet_id": sync.get("bullet_id"),
            "event_to_mvs_delta_ms": sync.get("event_to_mvs_delta_ms"),
            "event_to_human_capture_delta_ms": sync.get("event_to_human_capture_delta_ms"),
            "event_to_human_result_age_ms": sync.get("event_to_human_result_age_ms"),
            "event_to_human_delta_frames": sync.get("event_to_human_delta_frames"),
            "sync_quality": sync.get("sync_quality"),
            "latest_hit_reason": sync.get("latest_hit_reason", ""),
            "poll_error_count": len(telem.get("poll_errors", {}) if isinstance(telem.get("poll_errors", {}), dict) else {}),
            "video_backend_state": video_backend.get("state"),
            "video_backend_stage": video_backend.get("stage"),
            "video_backend_error_count": video_backend.get("error_count"),
            "video_backend_attempts": video_backend.get("attempts"),
            "video_backend_retries": video_backend.get("retries"),
            "video_timer_started": int(bool(video_backend.get("video_timer_started", False))),
            "render_timer_started": int(bool(video_backend.get("render_timer_started", False))),
            "overlay_timer_started": int(bool(video_backend.get("overlay_timer_started", False))),
            "last_video_tick_error": video_backend.get("last_video_tick_error"),
            "last_render_tick_error": video_backend.get("last_render_tick_error"),
            "last_paint_error": video_backend.get("last_paint_error"),
        }
        raw_rec = telem.get("raw_recording_status", {}) if isinstance(telem, dict) else {}
        if not isinstance(raw_rec, dict):
            raw_rec = {}
        raw_mvs = raw_rec.get("mvs", {}) if isinstance(raw_rec.get("mvs"), dict) else {}
        raw_event = raw_rec.get("event_raw", {}) if isinstance(raw_rec.get("event_raw"), dict) else {}
        row.update({
            "raw_record_state": raw_rec.get("state"),
            "raw_recording": int(bool(raw_rec.get("recording", False))),
            "raw_record_session_id": raw_rec.get("session_id"),
            "raw_record_last_command": raw_rec.get("last_command"),
            "raw_record_last_command_targets": raw_rec.get("last_command_targets"),
            "raw_record_command_age_ms": raw_rec.get("command_age_ms"),
            "raw_record_mvs_status_mode": raw_mvs.get("status_mode"),
            "raw_record_mvs_status_count": raw_mvs.get("status_count"),
            "raw_record_mvs_expected_status_count": raw_mvs.get("expected_status_count"),
            "raw_record_mvs_recording": int(bool(raw_mvs.get("recording", False))),
            "raw_record_mvs_written_frames_total": raw_mvs.get("written_frames_total"),
            "raw_record_mvs_dropped_frames_total": raw_mvs.get("dropped_frames_total"),
            "raw_record_mvs_queue_size_total": raw_mvs.get("queue_size_total"),
            "raw_record_event_recording_count": raw_event.get("recording_count"),
            "raw_record_event_status_count": raw_event.get("status_count"),
            "raw_record_event_slices_written_total": raw_event.get("slices_written_total"),
            "raw_record_event_events_written_total": raw_event.get("events_written_total"),
            "raw_record_event_dropped_slices_total": raw_event.get("dropped_slices_total"),
            "raw_record_event_queue_size_total": raw_event.get("queue_size_total"),
            "raw_record_degraded_links": ",".join(raw_rec.get("degraded_links", []) or []) if isinstance(raw_rec.get("degraded_links", []), list) else raw_rec.get("degraded_links"),
        })
        replay_rec = telem.get("replay_evidence_status", {}) if isinstance(telem, dict) else {}
        if not isinstance(replay_rec, dict):
            replay_rec = {}
        row.update({
            "evidence_replay_state": replay_rec.get("state") or replay_rec.get("evidence_replay_state"),
            "evidence_replay_pack_dir": replay_rec.get("pack_dir") or replay_rec.get("selected_pack_dir"),
            "evidence_replay_run_dir": replay_rec.get("replay_run_dir"),
            "evidence_replay_diff_state": replay_rec.get("diff_state"),
            "evidence_replay_original_hit_count": replay_rec.get("original_hit_count"),
            "evidence_replay_replay_hit_count": replay_rec.get("replay_hit_count"),
            "evidence_replay_global_person_track_count": replay_rec.get("global_person_track_count"),
        })
        exp_rec = telem.get("experiment_control_status", {}) if isinstance(telem, dict) else {}
        if not isinstance(exp_rec, dict):
            exp_rec = {}
        row.update({
            "experiment_state": exp_rec.get("state"),
            "experiment_id": exp_rec.get("experiment_id"),
            "experiment_stage": exp_rec.get("stage"),
            "experiment_config_hash": exp_rec.get("config_hash"),
            "experiment_source": exp_rec.get("source"),
            "experiment_status_age_ms": exp_rec.get("status_age_ms"),
            "experiment_final_hit_authority": exp_rec.get("final_hit_authority"),
            "experiment_human_noise_filter_mode": exp_rec.get("human_noise_filter_mode"),
            "experiment_bbox_start_expand_px": exp_rec.get("human_noise_track_start_bbox_expand_px"),
            "experiment_bbox_start_policy": exp_rec.get("human_noise_track_start_bbox_policy"),
            "experiment_global_person_target": exp_rec.get("global_person_target"),
            "experiment_global_person_filter_backend": exp_rec.get("global_person_filter_backend"),
            "experiment_run_state": exp_rec.get("run_state"),
            "experiment_runtime_overall_state": exp_rec.get("runtime_overall_state"),
        })
        hot = telem.get("hot_update_services", {}) if isinstance(telem, dict) else {}
        for svc in ("global_bullet_fusion", "damage_attribution", "manual_hit", "foam_dart_speed", "game_event_bridge"):
            rec = hot.get(svc, {}) if isinstance(hot, dict) else {}
            if not isinstance(rec, dict):
                rec = {}
            row.update({
                f"{svc}_state": rec.get("state"),
                f"{svc}_mode": rec.get("mode"),
                f"{svc}_enabled": int(bool(rec.get("enabled", False))),
                f"{svc}_input_age_ms": rec.get("input_age_ms"),
                f"{svc}_output_age_ms": rec.get("output_age_ms"),
                f"{svc}_queue_size": rec.get("queue_size"),
                f"{svc}_dropped_count": rec.get("dropped_count"),
                f"{svc}_last_error": rec.get("last_error"),
            })
        raw_fps = telem.get("raw_fps", {}) if isinstance(telem, dict) else {}
        mvs_fps = telem.get("mvs_fps", {}) if isinstance(telem, dict) else {}
        raw_age = telem.get("raw_age_ms", {}) if isinstance(telem, dict) else {}
        frame_age = telem.get("frame_age_ms", {}) if isinstance(telem, dict) else {}
        raw_missing = telem.get("raw_missing", {}) if isinstance(telem, dict) else {}
        mvs_missing = telem.get("mvs_missing", {}) if isinstance(telem, dict) else {}
        pending_q = telem.get("pending_q", {}) if isinstance(telem, dict) else {}
        human_age = telem.get("human_age_by_cam", {}) if isinstance(telem, dict) else {}
        deltas = yolo.get("result_video_delta_frames", {}) if isinstance(yolo, dict) else {}
        human_meta = overlay.get("human_meta", {}) if isinstance(overlay.get("human_meta"), dict) else {}
        for cam in list(getattr(self.preview, "cams", []) or []):
            hrec = human_meta.get(cam, {}) if isinstance(human_meta, dict) else {}
            row.update({
                f"{cam}_raw_fps": raw_fps.get(cam),
                f"{cam}_mvs_fps": mvs_fps.get(cam),
                f"{cam}_raw_age_ms": raw_age.get(cam),
                f"{cam}_frame_age_ms": frame_age.get(cam),
                f"{cam}_raw_missing": int(bool(raw_missing.get(cam, True))),
                f"{cam}_mvs_missing": int(bool(mvs_missing.get(cam, True))),
                f"{cam}_pending_q": pending_q.get(cam),
                f"{cam}_human_age_ms": human_age.get(cam),
                f"{cam}_yolo_delta_frames": deltas.get(cam) if isinstance(deltas, dict) else None,
                f"{cam}_persons": hrec.get("person_count") if isinstance(hrec, dict) else None,
                f"{cam}_event_ready": int(bool(ev.get("per_cam_ready", {}).get(cam, False))) if isinstance(ev.get("per_cam_ready", {}), dict) else 0,
                f"{cam}_event_trigger_fps": ev.get("per_cam_event_trigger_fps", {}).get(cam) if isinstance(ev.get("per_cam_event_trigger_fps", {}), dict) else None,
                f"{cam}_event_rate_mevs": ev.get("per_cam_event_rate_mevs", {}).get(cam) if isinstance(ev.get("per_cam_event_rate_mevs", {}), dict) else None,
                f"{cam}_event_trigger_age_ms": ev.get("per_cam_event_trigger_last_age_ms", {}).get(cam) if isinstance(ev.get("per_cam_event_trigger_last_age_ms", {}), dict) else None,
                f"{cam}_event_triggers": ev.get("per_cam_event_trigger_recv_count", {}).get(cam) if isinstance(ev.get("per_cam_event_trigger_recv_count", {}), dict) else None,
                f"{cam}_event_nonmono": ev.get("per_cam_event_nonmonotonic_count", {}).get(cam) if isinstance(ev.get("per_cam_event_nonmonotonic_count", {}), dict) else None,
                f"{cam}_event_guard": ev.get("per_cam_event_guard_state", {}).get(cam) if isinstance(ev.get("per_cam_event_guard_state", {}), dict) else None,
                f"{cam}_physical_projector_valid": int(bool(hrec.get("projector_valid", False))) if isinstance(hrec, dict) else 0,
                f"{cam}_physical_person_count": hrec.get("person_count") if isinstance(hrec, dict) else None,
                f"{cam}_physical_first_id": hrec.get("physical_first_id") if isinstance(hrec, dict) else "",
                f"{cam}_physical_first_x_m": hrec.get("physical_first_x_m") if isinstance(hrec, dict) else None,
                f"{cam}_physical_first_y_m": hrec.get("physical_first_y_m") if isinstance(hrec, dict) else None,
            })
            ev_ovl = renderer.get("event_overlay", {}).get(cam, {}) if isinstance(renderer.get("event_overlay", {}), dict) else {}
            if not isinstance(ev_ovl, dict):
                ev_ovl = {}
            row.update({
                f"{cam}_evt_ovl_drawn": int(bool(ev_ovl.get("drawn", False))),
                f"{cam}_evt_ovl_stale": int(bool(ev_ovl.get("stale", True))),
                f"{cam}_evt_ovl_age_ms": ev_ovl.get("age_ms"),
                f"{cam}_evt_ovl_dt_mvs_ms": ev_ovl.get("dt_mvs_ms"),
                f"{cam}_evt_ovl_dt_human_ms": ev_ovl.get("dt_human_ms"),
                f"{cam}_evt_ovl_raw": ev_ovl.get("raw"),
                f"{cam}_evt_ovl_filtered": ev_ovl.get("filtered"),
                f"{cam}_evt_ovl_mode": ev_ovl.get("mode"),
            })
        return row

    def _write_system_latest(self, system_status: Dict[str, Any]) -> None:
        if not self.system_latest_path:
            return
        tmp = self.system_latest_path + ".tmp"
        os.makedirs(os.path.dirname(self.system_latest_path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_json_safe(system_status), f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, self.system_latest_path)

    def maybe_write(self, tabs: Dict[str, Any], system_status: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled and not self.system_latest_path:
            return
        now_ms = _now_ms()
        if now_ms - self.last_write_ms < self.interval_ms:
            return
        self.last_write_ms = now_ms
        try:
            if system_status is None:
                system_status = self.build_system_status()
            self._write_system_latest(system_status)
            if not self.enabled:
                self.records += 1
                self.last_error = ""
                return
            telem = getattr(getattr(self.preview, "telemetry", None), "last_snapshot", {}) or {}
            overlay = self._overlay_summary()
            renderer = self._renderer_summary()
            rec = {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "sample_id": self.records,
                "ts_wall_us": int(time.time() * 1_000_000),
                "mono_ns": int(time.monotonic_ns()),
                "telemetry": telem,
                "overlay": overlay,
                "renderer": renderer,
                "system_status_latest": system_status,
            }
            if self._snapshot_fp is not None:
                self._snapshot_fp.write(json.dumps(_json_safe(rec), ensure_ascii=False, separators=(",", ":")) + "\n")
            if self.tabs_jsonl and self._tabs_fp is not None:
                tab_rec = {
                    "schema_version": self.SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "sample_id": self.records,
                    "ts_wall_us": rec["ts_wall_us"],
                    "tabs": tabs,
                }
                self._tabs_fp.write(json.dumps(_json_safe(tab_rec), ensure_ascii=False, separators=(",", ":")) + "\n")
            if self._csv_writer is not None:
                csv_telem = dict(telem) if isinstance(telem, dict) else {}
                try:
                    cats = system_status.get("categories", {}) if isinstance(system_status, dict) else {}
                    if isinstance(cats, dict):
                        csv_telem["raw_recording_status"] = cats.get("raw_recording", {})
                        csv_telem["replay_evidence_status"] = cats.get("replay_evidence", {})
                        csv_telem["experiment_control_status"] = cats.get("experiment_control", {})
                        csv_telem["hot_update_services"] = {
                            key: cats.get(key, {})
                            for key in ("global_bullet_fusion", "damage_attribution", "manual_hit", "foam_dart_speed", "game_event_bridge")
                        }
                    gy = cats.get("global_yolo", {}) if isinstance(cats, dict) else {}
                    if isinstance(gy, dict):
                        csv_yolo = dict(csv_telem.get("yolo", {}) if isinstance(csv_telem.get("yolo"), dict) else {})
                        csv_yolo.update({
                            "infer_fps": gy.get("infer_fps", csv_yolo.get("infer_fps")),
                            "published_fps_total": gy.get("published_fps_total", csv_yolo.get("published_fps_total")),
                            "inferred_fps_total": gy.get("inferred_fps_total", csv_yolo.get("inferred_fps_total")),
                            "capacity_fps": gy.get("capacity_fps", csv_yolo.get("capacity_fps")),
                            "fps_source": gy.get("fps_source", csv_yolo.get("fps_source")),
                            "batch_size": gy.get("batch_size", csv_yolo.get("batch_size")),
                            "predict_ms": gy.get("predict_ms", csv_yolo.get("predict_ms")),
                            "capture_to_result_ms": gy.get("total_ms", csv_yolo.get("capture_to_result_ms")),
                            "worker_count": gy.get("worker_count", csv_yolo.get("worker_count")),
                            "async_record_postprocess_enabled": gy.get("async_record_postprocess_enabled", csv_yolo.get("async_record_postprocess_enabled")),
                            "postprocess_queue_depth": gy.get("postprocess_queue_depth", csv_yolo.get("postprocess_queue_depth")),
                            "postprocess_dropped_batches": gy.get("postprocess_dropped_batches", csv_yolo.get("postprocess_dropped_batches")),
                            "polygon_contract_ok": gy.get("polygon_contract_ok", csv_yolo.get("polygon_contract_ok")),
                            "published_persons_with_polygon_ratio": gy.get("published_persons_with_polygon_ratio", csv_yolo.get("published_persons_with_polygon_ratio")),
                        })
                        csv_telem["yolo"] = csv_yolo
                except Exception:
                    csv_telem = telem
                self._csv_writer.writerow(_json_safe(self._csv_row(csv_telem, overlay, renderer)))
            if self.latest_enable and self.latest_path:
                tmp = self.latest_path + ".tmp"
                latest = dict(rec)
                latest["tabs"] = tabs
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(_json_safe(latest), f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.latest_path)
            self.records += 1
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[pygl_cuda_preview][WARN] status diagnostics log write failed: {self.last_error}", flush=True)

    def status_rows(self) -> List[Tuple[str, str]]:
        if not self.enabled:
            return [("enabled", "N"), ("system_latest_json", self.system_latest_path or "disabled")]
        age = _now_ms() - self.last_write_ms if self.last_write_ms > 0 else -1.0
        return [
            ("enabled", "Y"),
            ("run_dir", self.run_dir),
            ("snapshot_jsonl", self.snapshot_path),
            ("summary_csv", self.csv_path),
            ("latest_json", self.latest_path if self.latest_enable else "disabled"),
            ("system_latest_json", self.system_latest_path or "disabled"),
            ("tabs_jsonl", self.tabs_path if self.tabs_jsonl else "disabled"),
            ("interval_ms", f"{self.interval_ms:.0f}"),
            ("records", str(self.records)),
            ("last_write_age_ms", f"{age:.0f}" if age >= 0 else "-"),
            ("last_error", self.last_error or "-"),
        ]

    def close(self) -> None:
        for fp in (self._snapshot_fp, self._tabs_fp, self._csv_fp):
            try:
                if fp is not None:
                    fp.flush()
                    fp.close()
            except Exception:
                pass


class StatusWindow(QWidget):
    """Separate diagnostics window with tabs and collapsible sections.

    The status UI samples telemetry only. It does not modify the video,
    trajectory, human-mask, or hit rendering pipeline.  It is intentionally
    best-effort: diagnostics bugs must never prevent the main preview window
    from opening.
    """

    def __init__(self, preview: PreviewWidget, args: argparse.Namespace):
        super().__init__()
        self.preview = preview
        self.args = args
        self.setWindowTitle("全链路真实状态窗口 - PyOpenGL CUDA Preview")
        self.resize(int(getattr(args, 'status_window_width', 1180)), int(getattr(args, 'status_window_height', 860)))
        self.setStyleSheet("""
            QWidget { background: #101419; color: #d9f2ff; }
            QTabWidget::pane { border: 1px solid #2c4656; background: #101419; }
            QTabBar::tab { background: #17212a; color: #d9f2ff; padding: 5px 10px; }
            QTabBar::tab:selected { background: #24465a; color: #ffffff; }
            QTreeWidget { background: #0d1117; color: #d9f2ff; alternate-background-color: #151b22; border: 1px solid #283946; }
            QHeaderView::section { background: #1b2a34; color: #e8f8ff; padding: 4px; border: 0px; }
            QLabel, QCheckBox, QComboBox, QDoubleSpinBox { color: #d9f2ff; background: #101419; }
        """)
        self.tabs = QTabWidget(self)
        self.tree_by_tab: Dict[str, QTreeWidget] = {}
        self.tab_page_by_tab: Dict[str, QWidget] = {}
        self.expanded_state: Dict[Tuple[str, str], bool] = {}
        self._last_good_status_data: Dict[str, List[Tuple[str, Any]]] = {}
        self._refreshing = False
        self.evidence_replay_proc = None
        self.evidence_replay_log_fp = None
        self.evidence_selected_pack_dir = ""
        self.recording_tab_name = "\u5f55\u5236\u63a7\u5236"
        self.control_tab_name = "控制参数"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.fallback_label = QLabel("Status: waiting for telemetry refresh")
        self.fallback_label.setFont(QFont("DejaVu Sans Mono", max(10, int(getattr(args, 'status_window_font_px', 13)) - 1)))
        self.fallback_label.setStyleSheet("QLabel { color: #b8f0ff; background: #0d1117; padding: 4px; border: 1px solid #283946; }")
        layout.addWidget(self.fallback_label)
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.font_mono = QFont("DejaVu Sans Mono", int(getattr(args, 'status_window_font_px', 13)))
        self.log_writer = StatusLogWriter(preview, args)
        self._install_recording_page()
        self._install_control_page()
        self._install_placeholder()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(max(100, int(getattr(args, 'status_window_poll_ms', 500))))
        # Defer first heavy telemetry scan until Qt event loop is alive.
        QTimer.singleShot(250, self.refresh)

    def _on_event_overlay_enable_changed(self, checked: bool) -> None:
        try:
            setattr(self.args, "event_overlay_preview_enable", bool(checked))
            # The event overlay is shown in an independent overview window.
            # Keep the status-control hot switch mapped to that window, not the main RGB preview.
            try:
                self.preview.set_event_overlay_overview_visible(bool(checked))
            except Exception:
                pass
            try:
                self._write_control_json(self._event_overlay_overview_control_path(), {
                    "event_overlay_overview_visible": bool(checked),
                    "visible": bool(checked),
                    "event_overlay_preview_enable": bool(checked),
                })
                if hasattr(self, "event_overlay_overview_visible_check"):
                    self.event_overlay_overview_visible_check.blockSignals(True)
                    self.event_overlay_overview_visible_check.setChecked(bool(checked))
                    self.event_overlay_overview_visible_check.blockSignals(False)
            except Exception:
                pass
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] event overlay enable switch failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_event_overlay_overview_visible_changed(self, checked: bool) -> None:
        try:
            try:
                self.preview.set_event_overlay_overview_visible(bool(checked))
            except Exception:
                pass
            self._write_control_json(self._event_overlay_overview_control_path(), {
                "event_overlay_overview_visible": bool(checked),
                "visible": bool(checked),
                "event_overlay_preview_enable": bool(checked),
            })
            self.fallback_label.setText(f"Event Overlay 独立窗口 {'显示' if checked else '隐藏'}")
        except Exception as exc:
            self.fallback_label.setText(f"Event Overlay 独立窗口控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] event overlay overview window control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_event_overlay_mode_changed(self, mode: str) -> None:
        try:
            setattr(self.args, "event_overlay_sync_mode", str(mode))
        except Exception:
            pass

    def _on_event_overlay_alpha_changed(self, value: float) -> None:
        try:
            setattr(self.args, "event_overlay_preview_alpha", float(value))
        except Exception:
            pass

    def _on_event_overlay_delta_changed(self, value: float) -> None:
        try:
            setattr(self.args, "event_overlay_max_sync_delta_ms", float(value))
        except Exception:
            pass

    def _event_visual_mask_timing_payload(self) -> Dict[str, Any]:
        slice_ms = 33.333
        accumulation_ms = 33.333
        if getattr(self, "event_visual_mask_slice_spin", None) is not None:
            slice_ms = float(self.event_visual_mask_slice_spin.value())
        else:
            slice_ms = float(getattr(self.args, "event_visual_mask_slice_us", 33333) or 33333) / 1000.0
        if getattr(self, "event_visual_mask_accumulation_spin", None) is not None:
            accumulation_ms = float(self.event_visual_mask_accumulation_spin.value())
        else:
            accumulation_ms = float(getattr(self.args, "event_visual_mask_accumulation_us", 33333) or 33333) / 1000.0
        slice_us = max(1000, int(round(float(slice_ms) * 1000.0)))
        accumulation_us = max(slice_us, int(round(float(accumulation_ms) * 1000.0)))
        return {
            "event_visual_mask_slice_us": int(slice_us),
            "event_visual_mask_accumulation_us": int(accumulation_us),
            "event_visual_mask_slice_ms": round(float(slice_us) / 1000.0, 3),
            "event_visual_mask_accumulation_ms": round(float(accumulation_us) / 1000.0, 3),
        }

    def _event_mask_control_path(self) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / "event_mask_control.json"

    def _sync_control_path(self, attr: str, default_name: str) -> Path:
        raw = str(getattr(self.args, attr, "") or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            return path
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / default_name

    def _sync_root_path(self, filename: str) -> Path:
        p = Path(str(filename or "")).expanduser()
        if p.is_absolute():
            return p
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / p

    def _read_json_status(self, filename: str) -> Dict[str, Any]:
        try:
            if hasattr(self, "log_writer") and self.log_writer is not None:
                return self.log_writer._read_json_status(str(filename))
        except Exception:
            pass
        path = self._sync_root_path(str(filename))
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
        try:
            mtime_ms = path.stat().st_mtime * 1000.0 if path.exists() else 0.0
        except Exception:
            mtime_ms = 0.0
        obj["_status_path"] = str(path)
        obj["_file_exists"] = bool(path.exists())
        obj["_file_age_ms"] = max(0.0, _now_ms() - mtime_ms) if mtime_ms > 0 else -1.0
        return obj

    def _hit_control_path(self) -> Path:
        return self._sync_control_path("hit_control_json", "hit_judge_control.json")

    def _global_bev_control_path(self) -> Path:
        return self._sync_control_path("global_bev_control_json", "global_bev_control.json")

    def _event_overlay_overview_control_path(self) -> Path:
        return self._sync_control_path("event_overlay_overview_control_json", "event_overlay_overview_control.json")

    def _preview_control_path(self) -> Path:
        return self._sync_control_path("preview_control_json", "preview_control.json")

    def _manual_hit_control_path(self) -> Path:
        return self._sync_control_path("manual_hit_control_json", "manual_hit_control.json")

    def _game_event_bridge_control_path(self) -> Path:
        return self._sync_control_path("game_event_bridge_control_json", "game_event_bridge_control.json")

    def _global_person_id_control_path(self) -> Path:
        return self._sync_control_path("global_person_id_control_json", "global_person_id_control.json")

    def _global_person_id_shadow_control_path(self) -> Path:
        return self._sync_control_path("global_person_id_shadow_control_json", "global_person_shadow_control.json")

    def _person_id_dataset_control_path(self) -> Path:
        return self._sync_control_path("person_id_dataset_control_json", "person_id_dataset_control.json")

    def _person_id_dataset_status_path(self) -> Path:
        return self._sync_control_path("person_id_dataset_status_json", "person_id_dataset_status.json")

    def _read_bool_control(self, path: Path, key: Any, default: bool) -> bool:
        try:
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                keys = list(key) if isinstance(key, (list, tuple)) else [key]
                if isinstance(obj, dict):
                    for item in keys:
                        if item in obj:
                            return bool(obj.get(item))
        except Exception:
            pass
        return bool(default)

    def _read_float_control(self, path: Path, key: str, default: float) -> float:
        try:
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and key in obj:
                    return float(obj.get(key))
        except Exception:
            pass
        return float(default)

    def _read_str_control(self, path: Path, key: Any, default: str) -> str:
        try:
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                keys = list(key) if isinstance(key, (list, tuple)) else [key]
                if isinstance(obj, dict):
                    for item in keys:
                        if item in obj:
                            return str(obj.get(item) or default)
        except Exception:
            pass
        return str(default)

    def _write_control_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        current: Dict[str, Any] = {}
        try:
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    current = obj
        except Exception:
            current = {}
        out = dict(current)
        out.update(payload)
        out["schema_version"] = 1
        out["seq"] = _safe_int(current.get("seq"), 0) + 1
        out["source"] = "status_window"
        out["updated_wall_us"] = int(time.time() * 1_000_000)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    def _on_hit_detection_enable_changed(self, checked: bool) -> None:
        try:
            self._write_control_json(self._hit_control_path(), {
                "hit_detection_enabled": bool(checked),
                "hit_candidate_output_enabled": bool(checked),
            })
            self.fallback_label.setText(f"命中判定 {'启用' if checked else '关闭'}")
        except Exception as exc:
            self.fallback_label.setText(f"命中判定控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] hit detection control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_hit_chain_control_changed(self, *_args: Any) -> None:
        try:
            authority = "strong_gate"
            combo = getattr(self, "final_hit_authority_combo", None)
            if combo is not None:
                data = combo.currentData()
                authority = str(data or combo.currentText() or "strong_gate")
            strong_reasoning_enabled = bool(getattr(self, "strong_reasoning_check", None).isChecked()) if getattr(self, "strong_reasoning_check", None) is not None else True
            shadow_only = bool(getattr(self, "strong_reasoning_shadow_check", None).isChecked()) if getattr(self, "strong_reasoning_shadow_check", None) is not None else True
            if authority == "strong_reasoning":
                shadow_only = False
            payload = {
                "strong_gate_enabled": bool(getattr(self, "strong_gate_check", None).isChecked()) if getattr(self, "strong_gate_check", None) is not None else True,
                "strong_reasoning_enabled": strong_reasoning_enabled,
                "strong_reasoning_shadow_only": bool(shadow_only),
                "final_hit_authority": authority,
                "pseudo_hit_enable": strong_reasoning_enabled,
                "human_noise_filter_enable": bool(getattr(self, "human_noise_filter_check", None).isChecked()) if getattr(self, "human_noise_filter_check", None) is not None else True,
                "human_noise_filter_mode": str(getattr(self, "human_noise_mode_combo", None).currentData() or "soft") if getattr(self, "human_noise_mode_combo", None) is not None else "soft",
                "human_noise_track_start_bbox_filter_enable": True,
                "human_noise_track_start_bbox_expand_px": float(getattr(self, "human_noise_bbox_expand_spin", None).value()) if getattr(self, "human_noise_bbox_expand_spin", None) is not None else 48.0,
                "human_noise_track_start_bbox_policy": str(getattr(self, "human_noise_bbox_policy_combo", None).currentData() or "soft") if getattr(self, "human_noise_bbox_policy_combo", None) is not None else "soft",
            }
            self._write_control_json(self._hit_control_path(), payload)
            self.fallback_label.setText(f"Hit chain authority: {authority}")
        except Exception as exc:
            self.fallback_label.setText(f"Hit chain control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] hit chain control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_person_count_gate_changed(self, checked: bool) -> None:
        try:
            payload = {"count_gate_enable": bool(checked)}
            self._write_control_json(self._global_person_id_control_path(), payload)
            self._write_control_json(self._global_person_id_shadow_control_path(), payload)
            self.fallback_label.setText(f"全局ID短边门控 {'启用' if checked else '关闭'}")
        except Exception as exc:
            self.fallback_label.setText(f"全局ID短边门控控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global person count gate control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_bullet_overlay_changed(self, checked: bool) -> None:
        try:
            self._write_control_json(self._global_bev_control_path(), {
                "event_bullet_overlay_enable": bool(checked),
            })
            self.fallback_label.setText(f"Global BEV 弹道显示 {'启用' if checked else '关闭'}")
        except Exception as exc:
            self.fallback_label.setText(f"BEV弹道控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV bullet overlay control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_event_layer_changed(self, checked: bool) -> None:
        try:
            self._write_control_json(self._global_bev_control_path(), {
                "event_layer_enable": bool(checked),
                "show_event_layer": bool(checked),
            })
            self.fallback_label.setText(f"Global BEV 事件层 {'启用' if checked else '关闭'}")
        except Exception as exc:
            self.fallback_label.setText(f"BEV事件层控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV event layer control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_event_layer_dilate_changed(self, value: float) -> None:
        try:
            px = max(0.0, min(64.0, float(value)))
            self._write_control_json(self._global_bev_control_path(), {
                "event_layer_dilate_px": float(px),
            })
            self.fallback_label.setText(f"Global BEV 事件层膨胀: {px:.0f}px")
        except Exception as exc:
            self.fallback_label.setText(f"BEV事件层膨胀控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV event layer dilation control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_event_layer_alpha_changed(self, value: float) -> None:
        try:
            alpha = max(0.0, min(1.0, float(value)))
            self._write_control_json(self._global_bev_control_path(), {
                "event_layer_alpha": float(alpha),
            })
            self.fallback_label.setText(f"Global BEV 事件层透明度: {alpha:.2f}")
        except Exception as exc:
            self.fallback_label.setText(f"BEV事件层透明度控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV event layer alpha control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_event_layer_color_changed(self, mode: Any = None) -> None:
        try:
            mode_data = None
            try:
                mode_data = self.global_bev_event_layer_color_combo.currentData()
            except Exception:
                mode_data = None
            mode_s = str(mode_data if mode_data is not None else mode or "camera").strip().lower().replace("-", "_")
            if mode_s not in {"camera", "red", "yellow"}:
                mode_s = "camera"
            self._write_control_json(self._global_bev_control_path(), {
                "event_layer_color_mode": mode_s,
            })
            self.fallback_label.setText(f"Global BEV 事件层颜色: {mode_s}")
        except Exception as exc:
            self.fallback_label.setText(f"BEV事件层颜色控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV event layer color control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_event_layer_backend_changed(self, mode: Any = None) -> None:
        try:
            mode_data = None
            try:
                mode_data = self.global_bev_event_layer_backend_combo.currentData()
            except Exception:
                mode_data = None
            backend = str(mode_data if mode_data is not None else mode or "texture").strip().lower().replace("-", "_")
            if backend in {"sparse", "sparse_points", "opengl_sparse", "opengl_sparse_points"}:
                backend = "sparse_gl"
            if backend not in {"texture", "sparse_gl"}:
                backend = "texture"
            self._write_control_json(self._global_bev_control_path(), {
                "event_layer_backend": backend,
            })
            self.fallback_label.setText(f"Global BEV 事件层后端: {backend}")
        except Exception as exc:
            self.fallback_label.setText(f"BEV事件层后端控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV event layer backend control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_global_bev_event_layer_presentation_worker_changed(self, *_args: Any) -> None:
        try:
            mode_data = None
            try:
                mode_data = self.global_bev_event_layer_presentation_mode_combo.currentData()
            except Exception:
                mode_data = None
            mode_s = str(mode_data or "shadow").strip().lower().replace("-", "_")
            if mode_s not in {"off", "shadow", "active"}:
                mode_s = "shadow"
            interval_ms = 33.333
            try:
                interval_ms = float(self.global_bev_event_layer_presentation_interval_spin.value())
            except Exception:
                interval_ms = 33.333
            interval_ms = max(5.0, min(1000.0, float(interval_ms)))
            self._write_control_json(self._global_bev_control_path(), {
                "event_layer_presentation_worker_enable": bool(mode_s != "off"),
                "event_layer_presentation_worker_mode": mode_s,
                "event_layer_presentation_worker_interval_ms": float(interval_ms),
            })
            self.fallback_label.setText(f"Global BEV event presentation worker: {mode_s} @ {interval_ms:.1f}ms")
        except Exception as exc:
            self.fallback_label.setText(f"BEV event presentation worker control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] global BEV event presentation worker control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_human_label_mode_changed(self, mode: Any = None) -> None:
        try:
            mode_data = None
            try:
                mode_data = self.human_label_mode_combo.currentData()
            except Exception:
                mode_data = None
            mode_s = str(mode_data if mode_data is not None else mode or "detail").strip().lower()
            if mode_s not in {"id_only", "detail", "off"}:
                mode_s = "detail"
            setattr(self.args, "human_label_mode", mode_s)
            try:
                setattr(self.preview.args, "human_label_mode", mode_s)
            except Exception:
                pass
            self._write_control_json(self._preview_control_path(), {"human_label_mode": mode_s})
            self.fallback_label.setText(f"主预览人员ID显示: {mode_s}")
        except Exception as exc:
            self.fallback_label.setText(f"人员ID显示控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] human label mode switch failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_manual_hit_click_changed(self, checked: bool) -> None:
        try:
            enabled = bool(checked)
            preview_payload: Dict[str, Any] = {"manual_hit_click_enable": enabled}
            if enabled:
                preview_payload.update({
                    "preview_hit_display_enable": True,
                    "preview_trajectory_only_mode": False,
                })
            self._write_control_json(self._preview_control_path(), preview_payload)
            self._write_control_json(self._global_bev_control_path(), {"manual_hit_click_enable": enabled})
            self._write_control_json(self._manual_hit_control_path(), {"enabled": enabled, "source_global_id": 99})
            self._write_control_json(self._game_event_bridge_control_path(), {
                "hit_source_mode": "manual_only" if enabled else "all",
                "manual_hit_only": bool(enabled),
            })
            try:
                setattr(self.preview, "manual_hit_click_enable", enabled)
                if enabled:
                    setattr(self.preview, "preview_hit_display_enable", True)
                    setattr(self.preview, "preview_trajectory_only_mode", False)
                    setattr(self.preview.args, "hit_prompt_enable", True)
            except Exception:
                pass
            if enabled:
                hit_widget = getattr(self, "preview_hit_display_check", None)
                if hit_widget is not None:
                    hit_widget.blockSignals(True)
                    hit_widget.setChecked(True)
                    hit_widget.blockSignals(False)
                traj_widget = getattr(self, "preview_trajectory_only_check", None)
                if traj_widget is not None:
                    traj_widget.blockSignals(True)
                    traj_widget.setChecked(False)
                    traj_widget.blockSignals(False)
            self.fallback_label.setText(f"Manual hit click {'enabled' if enabled else 'disabled'}")
        except Exception as exc:
            self.fallback_label.setText(f"Manual hit click control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] manual hit click control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_preview_hit_display_changed(self, checked: bool) -> None:
        try:
            enabled = bool(checked)
            self._write_control_json(self._preview_control_path(), {"preview_hit_display_enable": enabled})
            try:
                setattr(self.preview, "preview_hit_display_enable", enabled)
                setattr(self.preview.args, "hit_prompt_enable", enabled)
            except Exception:
                pass
            self.fallback_label.setText(f"Preview hit display {'enabled' if enabled else 'disabled'}")
        except Exception as exc:
            self.fallback_label.setText(f"Preview hit display control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] preview hit display control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_preview_trajectory_only_changed(self, checked: bool) -> None:
        try:
            enabled = bool(checked)
            payload: Dict[str, Any] = {"preview_trajectory_only_mode": enabled}
            if enabled:
                payload["preview_hit_display_enable"] = False
            self._write_control_json(self._preview_control_path(), payload)
            try:
                setattr(self.preview, "preview_trajectory_only_mode", enabled)
                if enabled:
                    setattr(self.preview, "preview_hit_display_enable", False)
                    setattr(self.preview.args, "hit_prompt_enable", False)
                    widget = getattr(self, "preview_hit_display_check", None)
                    if widget is not None:
                        widget.blockSignals(True)
                        widget.setChecked(False)
                        widget.blockSignals(False)
            except Exception:
                pass
            self.fallback_label.setText(f"Preview trajectory-only {'enabled' if enabled else 'disabled'}")
        except Exception as exc:
            self.fallback_label.setText(f"Preview trajectory-only control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] preview trajectory-only control failed: {type(exc).__name__}: {exc}", flush=True)

    def _write_person_id_dataset_command(
        self,
        cmd: str,
        dataset_type: str = "",
        label: str = "",
        dataset_kind: str = "person_id",
        session_id: str = "",
        include_mask_polygons: bool = False,
        include_bullet_trajectory: bool = False,
        include_hit_debug: bool = False,
    ) -> None:
        try:
            cmd_s = str(cmd or "").strip().lower()
            dtype = str(dataset_type or "").strip().lower()
            if dtype not in {"negative", "positive", "mixed", "battle_1v1"}:
                dtype = "mixed"
            kind = str(dataset_kind or "person_id").strip().lower()
            if kind not in {"person_id", "evidence_replay"}:
                kind = "person_id"
            payload: Dict[str, Any] = {
                "cmd": cmd_s,
                "dataset_kind": kind,
                "dataset_type": dtype,
                "label": str(label or dtype or "status_window"),
                "description": "status_window_control",
                "include_mask_polygons": bool(include_mask_polygons),
                "include_bullet_trajectory": bool(include_bullet_trajectory),
                "include_hit_debug": bool(include_hit_debug),
            }
            if str(session_id or "").strip():
                payload["session_id"] = str(session_id).strip()
            self._write_control_json(self._person_id_dataset_control_path(), payload)
            suffix = f" session={session_id}" if str(session_id or "").strip() else ""
            self.fallback_label.setText(f"Dataset command: {cmd_s} kind={kind} type={dtype}{suffix}")
        except Exception as exc:
            self.fallback_label.setText(f"Person ID dataset control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] person ID dataset control failed: {type(exc).__name__}: {exc}", flush=True)

    def _dataset_type_from_combo(self, combo_attr: str = "person_id_dataset_type_combo") -> str:
        dtype = "mixed"
        try:
            combo = getattr(self, combo_attr, None)
            if combo is not None:
                dtype_data = combo.currentData()
                dtype = str(dtype_data or dtype)
        except Exception:
            pass
        if dtype not in {"negative", "positive", "mixed", "battle_1v1"}:
            dtype = "mixed"
        return dtype

    def _on_person_id_dataset_start(self) -> None:
        dtype = self._dataset_type_from_combo("person_id_dataset_type_combo")
        self._write_person_id_dataset_command("start", dtype, f"{dtype}_status_window", dataset_kind="person_id")

    def _on_person_id_dataset_stop(self) -> None:
        self._write_person_id_dataset_command("stop", "", "status_window_stop")

    def _on_evidence_replay_dataset_start(self) -> None:
        dtype = self._dataset_type_from_combo("person_id_dataset_type_combo")
        self._write_person_id_dataset_command(
            "start",
            dtype,
            f"{dtype}_evidence_replay",
            dataset_kind="evidence_replay",
            include_mask_polygons=True,
            include_bullet_trajectory=True,
            include_hit_debug=True,
        )

    def _on_evidence_replay_dataset_stop(self) -> None:
        self._write_person_id_dataset_command("stop", "", "evidence_replay_stop", dataset_kind="evidence_replay")

    def _on_recording_person_id_dataset_start(self) -> None:
        dtype = self._dataset_type_from_combo("recording_dataset_type_combo")
        self._write_person_id_dataset_command("start", dtype, f"{dtype}_recording_page", dataset_kind="person_id")

    def _on_recording_evidence_dataset_start(self) -> None:
        dtype = self._dataset_type_from_combo("recording_dataset_type_combo")
        self._write_person_id_dataset_command(
            "start",
            dtype,
            f"{dtype}_evidence_replay",
            dataset_kind="evidence_replay",
            include_mask_polygons=True,
            include_bullet_trajectory=True,
            include_hit_debug=True,
        )

    def _new_full_recording_session_id(self) -> str:
        return f"{time.strftime('%Y%m%d_%H%M%S')}_full"

    def _on_full_recording_start(self, combo_attr: str = "recording_dataset_type_combo") -> None:
        dtype = self._dataset_type_from_combo(combo_attr)
        sid = self._new_full_recording_session_id()
        self._write_record_command("start", ["event_raw", "mvs_raw"], "full_raw", session_id=sid)
        self._write_person_id_dataset_command(
            "start",
            dtype,
            f"{dtype}_full_recording",
            dataset_kind="evidence_replay",
            session_id=sid,
            include_mask_polygons=True,
            include_bullet_trajectory=True,
            include_hit_debug=True,
        )
        self.fallback_label.setText(f"Full recording started session={sid}")

    def _on_full_recording_stop(self) -> None:
        self._write_record_command("stop", ["event_raw", "mvs_raw"], "full_stop")
        self._write_person_id_dataset_command("stop", "", "full_recording_stop", dataset_kind="evidence_replay")
        self.fallback_label.setText("Full recording stop requested")

    def _record_control_path(self) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / "record_control.json"

    def _evidence_control_path(self) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / "evidence_replay_control.json"

    def _evidence_status_path(self) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / "evidence_replay_status.json"

    def _evidence_launcher_log_path(self, label: str) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        log_dir = root / "launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "evidence_replay"))
        return log_dir / f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def _write_evidence_control(self, payload: Dict[str, Any]) -> None:
        path = self._evidence_control_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        current: Dict[str, Any] = {}
        try:
            if path.exists():
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    current = obj
        except Exception:
            current = {}
        out = dict(current)
        out.update(payload)
        out["schema_version"] = 1
        out["seq"] = _safe_int(current.get("seq"), 0) + 1
        out["source"] = "status_window"
        out["updated_wall_us"] = int(time.time() * 1_000_000)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    def _read_evidence_selected_pack_dir(self) -> str:
        for path in (self._evidence_control_path(), self._evidence_status_path()):
            try:
                if not path.exists():
                    continue
                obj = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    continue
                val = str(obj.get("selected_pack_dir", "") or obj.get("pack_dir", "") or "")
                if val:
                    return val
            except Exception:
                continue
        try:
            packs = sorted((Path.cwd() / "recordings").glob("evidence_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in packs:
                if p.is_dir():
                    return str(p)
        except Exception:
            pass
        return ""

    def _launch_evidence_tool(self, label: str, cmd: List[str]) -> None:
        try:
            proc = self.evidence_replay_proc
            if proc is not None and proc.poll() is None:
                self.fallback_label.setText("Evidence Replay tool is already running")
                return
            try:
                if self.evidence_replay_log_fp is not None:
                    self.evidence_replay_log_fp.close()
            except Exception:
                pass
            log_path = self._evidence_launcher_log_path(label)
            fp = log_path.open("a", encoding="utf-8", buffering=1)
            fp.write(f"[status_window] start {time.strftime('%Y-%m-%d %H:%M:%S')} cmd={cmd}\n")
            self.evidence_replay_log_fp = fp
            self.evidence_replay_proc = subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                stdout=fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._write_evidence_control({"cmd": str(label), "tool_pid": int(self.evidence_replay_proc.pid), "tool_log": str(log_path)})
            self.fallback_label.setText(f"Evidence Replay {label} started pid={self.evidence_replay_proc.pid} log={log_path}")
        except Exception as exc:
            self.fallback_label.setText(f"Evidence Replay {label} failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] evidence replay {label} failed: {type(exc).__name__}: {exc}", flush=True)

    def _run_evidence_pack_collector(self) -> None:
        tool = Path.cwd() / "tools" / "evidence_pack_collector.py"
        sync_root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not sync_root.is_absolute():
            sync_root = Path.cwd() / sync_root
        cmd = [
            sys.executable,
            str(tool),
            "--project-root",
            str(Path.cwd()),
            "--sync-root",
            str(sync_root),
            "--name",
            "status_window",
            "--test-note",
            "Sync Diagnostics 状态诊断窗口手动打包",
        ]
        self._launch_evidence_tool("pack", cmd)

    def _select_evidence_pack_dir(self) -> None:
        try:
            start_dir = str(Path.cwd() / "recordings")
            selected = QFileDialog.getExistingDirectory(self, "选择 Evidence Replay 文件夹", start_dir)
            if not selected:
                return
            self.evidence_selected_pack_dir = str(selected)
            self._write_evidence_control({"cmd": "select_pack", "selected_pack_dir": str(selected)})
            status_path = self._evidence_status_path()
            status = {}
            try:
                if status_path.exists():
                    obj = json.loads(status_path.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        status = obj
            except Exception:
                status = {}
            status.update({
                "schema_version": 1,
                "kind": "evidence_replay_status",
                "state": str(status.get("state", "READY") or "READY"),
                "evidence_state": str(status.get("evidence_state", status.get("state", "READY")) or "READY"),
                "selected_pack_dir": str(selected),
                "updated_wall_us": int(time.time() * 1_000_000),
            })
            tmp = status_path.with_suffix(status_path.suffix + ".tmp")
            tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, status_path)
            self.fallback_label.setText(f"Evidence Replay selected: {selected}")
        except Exception as exc:
            self.fallback_label.setText(f"Evidence Replay select failed: {type(exc).__name__}: {exc}")

    def _run_evidence_replay(self) -> None:
        pack_dir = self.evidence_selected_pack_dir or self._read_evidence_selected_pack_dir()
        if not pack_dir:
            self.fallback_label.setText("Evidence Replay: no pack selected")
            return
        tool = Path.cwd() / "tools" / "evidence_replay_runner.py"
        cmd = [
            sys.executable,
            str(tool),
            "--project-root",
            str(Path.cwd()),
            "--pack-dir",
            str(pack_dir),
        ]
        self._launch_evidence_tool("replay", cmd)

    def _stop_evidence_replay(self) -> None:
        try:
            proc = self.evidence_replay_proc
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    proc.kill()
                self._write_evidence_control({"cmd": "stop_replay", "tool_pid": int(getattr(proc, "pid", 0) or 0)})
                self.fallback_label.setText("Evidence Replay stop requested")
            else:
                self._write_evidence_control({"cmd": "stop_replay"})
                self.fallback_label.setText("Evidence Replay is not running")
        except Exception as exc:
            self.fallback_label.setText(f"Evidence Replay stop failed: {type(exc).__name__}: {exc}")

    def _operator_annotation_path(self) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / "operator_annotations.jsonl"

    def _current_record_session_id(self) -> str:
        for path in (self._record_control_path(), Path(str(getattr(self.log_writer, "system_latest_path", "") or ""))):
            try:
                if not path or not path.exists():
                    continue
                obj = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    continue
                if "categories" in obj:
                    raw = obj.get("categories", {}).get("raw_recording", {}) if isinstance(obj.get("categories"), dict) else {}
                    sid = str(raw.get("session_id", "") or "")
                else:
                    sid = str(obj.get("session_id", "") or "")
                if sid:
                    return sid
            except Exception:
                continue
        return ""

    def _write_operator_annotation(self, kind: str, label: str) -> None:
        try:
            session_id = self._current_record_session_id()
            payload: Dict[str, Any] = {
                "schema_version": 1,
                "kind": str(kind or "mark"),
                "label": str(label or kind or "mark"),
                "session_id": session_id,
                "wall_us": int(time.time() * 1_000_000),
                "mono_ns": int(time.monotonic_ns()),
                "source": "status_window",
            }
            paths = [self._operator_annotation_path()]
            if session_id:
                paths.append(Path.cwd() / "recordings" / session_id / "operator_annotations.jsonl")
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line)
            suffix = f" session={session_id}" if session_id else ""
            self.fallback_label.setText(f"Operator annotation: {label}{suffix}")
            print(f"[pygl_cuda_preview][annotation] kind={kind} label={label} session={session_id}", flush=True)
        except Exception as exc:
            self.fallback_label.setText(f"Operator annotation failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] operator annotation failed: {type(exc).__name__}: {exc}", flush=True)

    def _write_record_command(self, cmd: str, targets: List[str], label: str, session_id: str = "") -> None:
        try:
            cmd_s = str(cmd or "").strip().lower()
            target_list = [str(x).strip() for x in (targets or []) if str(x).strip()]
            payload: Dict[str, Any] = {
                "seq": int(time.time() * 1000),
                "cmd": cmd_s,
                "targets": target_list or ["event_raw", "mvs_raw"],
                "wall_us": int(time.time() * 1_000_000),
                "source": "status_window",
            }
            if cmd_s == "start":
                payload["session_id"] = str(session_id or "").strip() or time.strftime("%Y%m%d_%H%M%S")
            elif str(session_id or "").strip():
                payload["session_id"] = str(session_id).strip()
            path = self._record_control_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, path)
            sid = str(payload.get("session_id", ""))
            suffix = f" session={sid}" if sid else ""
            self.fallback_label.setText(f"Raw record command: {cmd_s} {label}{suffix}")
            print(f"[pygl_cuda_preview][raw_record] command={cmd_s} targets={target_list} session={sid} file={path}", flush=True)
        except Exception as exc:
            self.fallback_label.setText(f"Raw record command failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] raw record command failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_event_mask_period_changed(self, text: str) -> None:
        try:
            period_ms = max(1.0, float(str(text).strip()))
            setattr(self.args, "event_mask_evidence_period_ms", float(period_ms))
            payload = {
                "event_human_mask_evidence_enable": True,
                "event_human_mask_evidence_period_ms": float(period_ms),
            }
            if getattr(self, "event_mask_raster_check", None) is not None:
                payload["mask_raster_enable"] = bool(self.event_mask_raster_check.isChecked())
            if getattr(self, "event_mask_raster_dilate_spin", None) is not None:
                payload["mask_raster_dilate_px"] = int(self.event_mask_raster_dilate_spin.value())
            payload.update(self._event_visual_mask_timing_payload())
            self._write_control_json(self._event_mask_control_path(), payload)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] event mask period switch failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_event_mask_raster_control_changed(self, *_args: Any) -> None:
        try:
            payload: Dict[str, Any] = {
                "event_human_mask_evidence_enable": True,
                "mask_raster_enable": bool(getattr(self, "event_mask_raster_check", None).isChecked()) if getattr(self, "event_mask_raster_check", None) is not None else True,
                "mask_raster_dilate_px": int(getattr(self, "event_mask_raster_dilate_spin", None).value()) if getattr(self, "event_mask_raster_dilate_spin", None) is not None else 8,
            }
            if getattr(self, "event_mask_period_combo", None) is not None:
                payload["event_human_mask_evidence_period_ms"] = max(1.0, float(str(self.event_mask_period_combo.currentText()).strip()))
            payload.update(self._event_visual_mask_timing_payload())
            self._write_control_json(self._event_mask_control_path(), payload)
            self.fallback_label.setText(f"Event mask raster {'启用' if payload['mask_raster_enable'] else '关闭'} dilate={payload['mask_raster_dilate_px']}px")
        except Exception as exc:
            self.fallback_label.setText(f"Event mask raster 控制失败: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] event mask raster control failed: {type(exc).__name__}: {exc}", flush=True)

    def _on_event_visual_mask_timing_changed(self, *_args: Any) -> None:
        try:
            payload: Dict[str, Any] = {
                "event_human_mask_evidence_enable": True,
            }
            if getattr(self, "event_mask_period_combo", None) is not None:
                payload["event_human_mask_evidence_period_ms"] = max(1.0, float(str(self.event_mask_period_combo.currentText()).strip()))
            if getattr(self, "event_mask_raster_check", None) is not None:
                payload["mask_raster_enable"] = bool(self.event_mask_raster_check.isChecked())
            if getattr(self, "event_mask_raster_dilate_spin", None) is not None:
                payload["mask_raster_dilate_px"] = int(self.event_mask_raster_dilate_spin.value())
            payload.update(self._event_visual_mask_timing_payload())
            self._write_control_json(self._event_mask_control_path(), payload)
            self.fallback_label.setText(
                "Event visual mask timing: "
                f"slice={payload['event_visual_mask_slice_ms']:.3f}ms "
                f"acc={payload['event_visual_mask_accumulation_ms']:.3f}ms"
            )
        except Exception as exc:
            self.fallback_label.setText(f"Event visual mask timing control failed: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] event visual mask timing control failed: {type(exc).__name__}: {exc}", flush=True)

    def _new_status_tree(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setFont(self.font_mono)
        tree.setColumnCount(2)
        tree.setHeaderLabels(["metric", "value"])
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setRootIsDecorated(True)
        tree.setItemsExpandable(True)
        tree.setSortingEnabled(False)
        try:
            tree.setTextElideMode(Qt.ElideMiddle)
            tree.setWordWrap(False)
        except Exception:
            pass
        try:
            tree.setColumnWidth(0, 300)
            tree.setColumnWidth(1, 900)
        except Exception:
            pass
        return tree

    def _make_tree(self, tab_name: str) -> QTreeWidget:
        tree = self._new_status_tree()
        self.tabs.addTab(tree, tab_name)
        self.tree_by_tab[tab_name] = tree
        self.tab_page_by_tab[tab_name] = tree
        return tree

    def _install_recording_page(self) -> None:
        try:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(6, 6, 6, 6)
            page_layout.setSpacing(6)

            title = QLabel("\u5f55\u5236\u63a7\u5236 / Evidence Replay Dataset")
            title.setStyleSheet("QLabel { color: #ffffff; font-weight: 600; }")
            page_layout.addWidget(title)

            row_dataset = QHBoxLayout()
            row_dataset.addWidget(QLabel("\u6570\u636e\u96c6\u7c7b\u578b"))
            self.recording_dataset_type_combo = QComboBox()
            self.recording_dataset_type_combo.addItem("\u8d1f\u6837\u672c", "negative")
            self.recording_dataset_type_combo.addItem("\u6b63\u6837\u672c", "positive")
            self.recording_dataset_type_combo.addItem("\u6df7\u5408", "mixed")
            self.recording_dataset_type_combo.addItem("1v1", "battle_1v1")
            row_dataset.addWidget(self.recording_dataset_type_combo)
            self.recording_evidence_start_button = QPushButton("\u5f00\u59cbA1\u8bc1\u636e\u5f55\u5236")
            self.recording_evidence_stop_button = QPushButton("\u505c\u6b62A1\u8bc1\u636e\u5f55\u5236")
            self.recording_person_start_button = QPushButton("\u5f00\u59cb\u4eba\u5458ID\u8f7b\u91cf\u5f55\u5236")
            self.recording_person_stop_button = QPushButton("\u505c\u6b62\u4eba\u5458ID\u5f55\u5236")
            self.recording_full_start_button = QPushButton("\u5f00\u59cb\u5168\u91cf\u5f55\u5236")
            self.recording_full_stop_button = QPushButton("\u505c\u6b62\u5168\u91cf\u5f55\u5236")
            self.recording_evidence_start_button.clicked.connect(self._on_recording_evidence_dataset_start)
            self.recording_evidence_stop_button.clicked.connect(self._on_evidence_replay_dataset_stop)
            self.recording_person_start_button.clicked.connect(self._on_recording_person_id_dataset_start)
            self.recording_person_stop_button.clicked.connect(self._on_person_id_dataset_stop)
            self.recording_full_start_button.clicked.connect(lambda: self._on_full_recording_start("recording_dataset_type_combo"))
            self.recording_full_stop_button.clicked.connect(self._on_full_recording_stop)
            for btn in (
                self.recording_full_start_button,
                self.recording_full_stop_button,
                self.recording_evidence_start_button,
                self.recording_evidence_stop_button,
                self.recording_person_start_button,
                self.recording_person_stop_button,
            ):
                row_dataset.addWidget(btn)
            row_dataset.addStretch(1)
            page_layout.addLayout(row_dataset)

            row_raw = QHBoxLayout()
            row_raw.addWidget(QLabel("\u539f\u59cbRAW\u5f55\u5236"))
            self.recording_raw_all_button = QPushButton("\u5f55\u5236\u5168\u90e8")
            self.recording_raw_event_button = QPushButton("\u4ec5Event RAW")
            self.recording_raw_mvs_button = QPushButton("\u4ec5MVS RAW")
            self.recording_raw_stop_button = QPushButton("\u505c\u6b62RAW\u5f55\u5236")
            self.recording_raw_all_button.clicked.connect(lambda: self._write_record_command("start", ["event_raw", "mvs_raw"], "all"))
            self.recording_raw_event_button.clicked.connect(lambda: self._write_record_command("start", ["event_raw"], "event_raw"))
            self.recording_raw_mvs_button.clicked.connect(lambda: self._write_record_command("start", ["mvs_raw"], "mvs_raw"))
            self.recording_raw_stop_button.clicked.connect(lambda: self._write_record_command("stop", ["event_raw", "mvs_raw"], "stop"))
            for btn in (
                self.recording_raw_all_button,
                self.recording_raw_event_button,
                self.recording_raw_mvs_button,
                self.recording_raw_stop_button,
            ):
                row_raw.addWidget(btn)
            row_raw.addStretch(1)
            page_layout.addLayout(row_raw)

            row_mark = QHBoxLayout()
            row_mark.addWidget(QLabel("\u73b0\u573a\u6807\u8bb0"))
            self.recording_mark_shot_button = QPushButton("\u6807\u8bb0\u5c04\u51fb")
            self.recording_mark_hit_button = QPushButton("\u6807\u8bb0\u547d\u4e2d")
            self.recording_mark_miss_button = QPushButton("\u6807\u8bb0\u672a\u4e2d")
            self.recording_mark_false_button = QPushButton("\u6807\u8bb0\u8bef\u5224")
            self.recording_mark_issue_button = QPushButton("\u76f4\u64ad\u5f02\u5e38")
            self.recording_mark_shot_button.clicked.connect(lambda: self._write_operator_annotation("shot", "\u6807\u8bb0\u5c04\u51fb"))
            self.recording_mark_hit_button.clicked.connect(lambda: self._write_operator_annotation("hit", "\u6807\u8bb0\u547d\u4e2d"))
            self.recording_mark_miss_button.clicked.connect(lambda: self._write_operator_annotation("miss", "\u6807\u8bb0\u672a\u4e2d"))
            self.recording_mark_false_button.clicked.connect(lambda: self._write_operator_annotation("false_hit", "\u6807\u8bb0\u8bef\u5224"))
            self.recording_mark_issue_button.clicked.connect(lambda: self._write_operator_annotation("live_issue", "\u76f4\u64ad\u5f02\u5e38"))
            for btn in (
                self.recording_mark_shot_button,
                self.recording_mark_hit_button,
                self.recording_mark_miss_button,
                self.recording_mark_false_button,
                self.recording_mark_issue_button,
            ):
                row_mark.addWidget(btn)
            row_mark.addStretch(1)
            page_layout.addLayout(row_mark)

            row_replay = QHBoxLayout()
            row_replay.addWidget(QLabel("\u672c\u5730\u56de\u653e\u5de5\u5177"))
            self.recording_pack_button = QPushButton("\u6253\u5305\u8bc1\u636e")
            self.recording_select_replay_button = QPushButton("\u9009\u62e9\u56de\u653e\u6587\u4ef6\u5939")
            self.recording_run_replay_button = QPushButton("\u5f00\u59cb\u56de\u653e")
            self.recording_stop_replay_button = QPushButton("\u505c\u6b62\u56de\u653e")
            self.recording_pack_button.clicked.connect(self._run_evidence_pack_collector)
            self.recording_select_replay_button.clicked.connect(self._select_evidence_pack_dir)
            self.recording_run_replay_button.clicked.connect(self._run_evidence_replay)
            self.recording_stop_replay_button.clicked.connect(self._stop_evidence_replay)
            for btn in (
                self.recording_pack_button,
                self.recording_select_replay_button,
                self.recording_run_replay_button,
                self.recording_stop_replay_button,
            ):
                row_replay.addWidget(btn)
            row_replay.addStretch(1)
            page_layout.addLayout(row_replay)

            tree = self._new_status_tree()
            page_layout.addWidget(tree, 1)
            self.tabs.addTab(page, self.recording_tab_name)
            self.tree_by_tab[self.recording_tab_name] = tree
            self.tab_page_by_tab[self.recording_tab_name] = page
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] recording page install failed: {type(exc).__name__}: {exc}", flush=True)

    def _install_control_page(self) -> None:
        try:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(6, 6, 6, 6)
            page_layout.setSpacing(6)

            title = QLabel("控制按钮 / 运行参数")
            title.setStyleSheet("QLabel { color: #ffffff; font-weight: 600; }")
            page_layout.addWidget(title)

            row_display = QHBoxLayout()
            row_display.addWidget(QLabel("显示渲染"))
            self.event_overlay_check = QCheckBox("Event Overlay")
            self.event_overlay_check.setChecked(bool(getattr(self.args, "event_overlay_preview_enable", False)))
            self.event_overlay_check.toggled.connect(self._on_event_overlay_enable_changed)
            row_display.addWidget(self.event_overlay_check)
            self.event_overlay_overview_visible_check = QCheckBox("Event独立窗")
            self.event_overlay_overview_visible_check.setChecked(
                self._read_bool_control(
                    self._event_overlay_overview_control_path(),
                    "event_overlay_overview_visible",
                    bool(getattr(self.args, "event_overlay_preview_enable", False)),
                )
            )
            self.event_overlay_overview_visible_check.toggled.connect(self._on_event_overlay_overview_visible_changed)
            row_display.addWidget(self.event_overlay_overview_visible_check)
            self.global_bev_event_layer_check = QCheckBox("BEV事件层")
            self.global_bev_event_layer_check.setChecked(self._read_bool_control(self._global_bev_control_path(), ("event_layer_enable", "show_event_layer"), True))
            self.global_bev_event_layer_check.toggled.connect(self._on_global_bev_event_layer_changed)
            row_display.addWidget(self.global_bev_event_layer_check)
            self.global_bev_bullet_check = QCheckBox("BEV弹道")
            self.global_bev_bullet_check.setChecked(self._read_bool_control(self._global_bev_control_path(), "event_bullet_overlay_enable", True))
            self.global_bev_bullet_check.toggled.connect(self._on_global_bev_bullet_overlay_changed)
            row_display.addWidget(self.global_bev_bullet_check)
            row_display.addWidget(QLabel("人员ID显示"))
            self.human_label_mode_combo = QComboBox()
            self.human_label_mode_combo.addItem("仅ID", "id_only")
            self.human_label_mode_combo.addItem("详版", "detail")
            self.human_label_mode_combo.addItem("关闭文字", "off")
            try:
                idx = self.human_label_mode_combo.findData(str(getattr(self.args, "human_label_mode", "detail") or "detail"))
                if idx >= 0:
                    self.human_label_mode_combo.setCurrentIndex(idx)
            except Exception:
                pass
            self.human_label_mode_combo.currentIndexChanged.connect(self._on_human_label_mode_changed)
            row_display.addWidget(self.human_label_mode_combo)
            self.manual_hit_click_check = QCheckBox("人工点击命中")
            self.manual_hit_click_check.setChecked(self._read_bool_control(self._preview_control_path(), "manual_hit_click_enable", True))
            self.manual_hit_click_check.toggled.connect(self._on_manual_hit_click_changed)
            row_display.addWidget(self.manual_hit_click_check)
            self.preview_hit_display_check = QCheckBox("主预览命中")
            self.preview_hit_display_check.setChecked(self._read_bool_control(self._preview_control_path(), "preview_hit_display_enable", True))
            self.preview_hit_display_check.toggled.connect(self._on_preview_hit_display_changed)
            row_display.addWidget(self.preview_hit_display_check)
            self.preview_trajectory_only_check = QCheckBox("仅弹道")
            self.preview_trajectory_only_check.setChecked(self._read_bool_control(self._preview_control_path(), "preview_trajectory_only_mode", False))
            self.preview_trajectory_only_check.toggled.connect(self._on_preview_trajectory_only_changed)
            row_display.addWidget(self.preview_trajectory_only_check)
            row_display.addStretch(1)
            page_layout.addLayout(row_display)

            row_bev_event = QHBoxLayout()
            row_bev_event.addWidget(QLabel("BEV事件层参数"))
            row_bev_event.addWidget(QLabel("后端"))
            self.global_bev_event_layer_backend_combo = QComboBox()
            self.global_bev_event_layer_backend_combo.addItem("纹理", "texture")
            self.global_bev_event_layer_backend_combo.addItem("稀疏GL", "sparse_gl")
            try:
                backend = self._read_str_control(
                    self._global_bev_control_path(),
                    ("event_layer_backend", "global_bev_event_layer_backend"),
                    str(getattr(self.args, "global_bev_event_layer_backend", "sparse_gl") or "sparse_gl"),
                ).strip().lower().replace("-", "_")
                if backend in {"sparse", "sparse_points", "opengl_sparse", "opengl_sparse_points"}:
                    backend = "sparse_gl"
                idx = self.global_bev_event_layer_backend_combo.findData(backend)
                if idx >= 0:
                    self.global_bev_event_layer_backend_combo.setCurrentIndex(idx)
            except Exception:
                pass
            self.global_bev_event_layer_backend_combo.currentIndexChanged.connect(self._on_global_bev_event_layer_backend_changed)
            row_bev_event.addWidget(self.global_bev_event_layer_backend_combo)
            row_bev_event.addWidget(QLabel("透明"))
            self.global_bev_event_layer_alpha_spin = QDoubleSpinBox()
            self.global_bev_event_layer_alpha_spin.setRange(0.0, 1.0)
            self.global_bev_event_layer_alpha_spin.setDecimals(2)
            self.global_bev_event_layer_alpha_spin.setSingleStep(0.05)
            self.global_bev_event_layer_alpha_spin.setValue(
                self._read_float_control(
                    self._global_bev_control_path(),
                    "event_layer_alpha",
                    float(getattr(self.args, "global_bev_event_layer_alpha", 0.45) or 0.45),
                )
            )
            self.global_bev_event_layer_alpha_spin.valueChanged.connect(self._on_global_bev_event_layer_alpha_changed)
            row_bev_event.addWidget(self.global_bev_event_layer_alpha_spin)
            row_bev_event.addWidget(QLabel("颜色"))
            self.global_bev_event_layer_color_combo = QComboBox()
            self.global_bev_event_layer_color_combo.addItem("相机色", "camera")
            self.global_bev_event_layer_color_combo.addItem("红色", "red")
            self.global_bev_event_layer_color_combo.addItem("黄色", "yellow")
            try:
                color_mode = self._read_str_control(
                    self._global_bev_control_path(),
                    ("event_layer_color_mode", "event_overlay_layer_color_mode"),
                    str(getattr(self.args, "global_bev_event_layer_color_mode", "camera") or "camera"),
                ).strip().lower().replace("-", "_")
                idx = self.global_bev_event_layer_color_combo.findData(color_mode)
                if idx >= 0:
                    self.global_bev_event_layer_color_combo.setCurrentIndex(idx)
            except Exception:
                pass
            self.global_bev_event_layer_color_combo.currentIndexChanged.connect(self._on_global_bev_event_layer_color_changed)
            row_bev_event.addWidget(self.global_bev_event_layer_color_combo)
            row_bev_event.addWidget(QLabel("膨胀px"))
            self.global_bev_event_layer_dilate_spin = QDoubleSpinBox()
            self.global_bev_event_layer_dilate_spin.setRange(0.0, 64.0)
            self.global_bev_event_layer_dilate_spin.setDecimals(0)
            self.global_bev_event_layer_dilate_spin.setSingleStep(4.0)
            self.global_bev_event_layer_dilate_spin.setValue(
                self._read_float_control(
                    self._global_bev_control_path(),
                    "event_layer_dilate_px",
                    float(getattr(self.args, "global_bev_event_layer_dilate_px", 16.0) or 16.0),
                )
            )
            self.global_bev_event_layer_dilate_spin.valueChanged.connect(self._on_global_bev_event_layer_dilate_changed)
            row_bev_event.addWidget(self.global_bev_event_layer_dilate_spin)
            row_bev_event.addWidget(QLabel("presentation"))
            self.global_bev_event_layer_presentation_mode_combo = QComboBox()
            self.global_bev_event_layer_presentation_mode_combo.addItem("off", "off")
            self.global_bev_event_layer_presentation_mode_combo.addItem("shadow", "shadow")
            self.global_bev_event_layer_presentation_mode_combo.addItem("active", "active")
            try:
                mode = self._read_str_control(
                    self._global_bev_control_path(),
                    ("event_layer_presentation_worker_mode", "global_bev_event_layer_presentation_worker_mode"),
                    str(getattr(self.args, "global_bev_event_layer_presentation_worker_mode", "shadow") or "shadow"),
                ).strip().lower().replace("-", "_")
                idx = self.global_bev_event_layer_presentation_mode_combo.findData(mode)
                if idx >= 0:
                    self.global_bev_event_layer_presentation_mode_combo.setCurrentIndex(idx)
            except Exception:
                pass
            self.global_bev_event_layer_presentation_mode_combo.currentIndexChanged.connect(self._on_global_bev_event_layer_presentation_worker_changed)
            row_bev_event.addWidget(self.global_bev_event_layer_presentation_mode_combo)
            row_bev_event.addWidget(QLabel("worker ms"))
            self.global_bev_event_layer_presentation_interval_spin = QDoubleSpinBox()
            self.global_bev_event_layer_presentation_interval_spin.setRange(5.0, 1000.0)
            self.global_bev_event_layer_presentation_interval_spin.setDecimals(1)
            self.global_bev_event_layer_presentation_interval_spin.setSingleStep(5.0)
            self.global_bev_event_layer_presentation_interval_spin.setValue(
                self._read_float_control(
                    self._global_bev_control_path(),
                    "event_layer_presentation_worker_interval_ms",
                    float(getattr(self.args, "global_bev_event_layer_presentation_worker_interval_ms", 33.333) or 33.333),
                )
            )
            self.global_bev_event_layer_presentation_interval_spin.valueChanged.connect(self._on_global_bev_event_layer_presentation_worker_changed)
            row_bev_event.addWidget(self.global_bev_event_layer_presentation_interval_spin)
            row_bev_event.addStretch(1)
            page_layout.addLayout(row_bev_event)

            row_sync = QHBoxLayout()
            row_sync.addWidget(QLabel("同步与判定"))
            row_sync.addWidget(QLabel("Event sync"))
            self.event_overlay_mode_combo = QComboBox()
            self.event_overlay_mode_combo.addItems(["latest", "nearest_mvs", "nearest_human"])
            try:
                self.event_overlay_mode_combo.setCurrentText(str(getattr(self.args, "event_overlay_sync_mode", "nearest_mvs")))
            except Exception:
                pass
            self.event_overlay_mode_combo.currentTextChanged.connect(self._on_event_overlay_mode_changed)
            row_sync.addWidget(self.event_overlay_mode_combo)
            row_sync.addWidget(QLabel("alpha"))
            self.event_overlay_alpha_spin = QDoubleSpinBox()
            self.event_overlay_alpha_spin.setRange(0.0, 1.0)
            self.event_overlay_alpha_spin.setSingleStep(0.05)
            self.event_overlay_alpha_spin.setValue(float(getattr(self.args, "event_overlay_preview_alpha", 0.55)))
            self.event_overlay_alpha_spin.valueChanged.connect(self._on_event_overlay_alpha_changed)
            row_sync.addWidget(self.event_overlay_alpha_spin)
            row_sync.addWidget(QLabel("max dt ms"))
            self.event_overlay_delta_spin = QDoubleSpinBox()
            self.event_overlay_delta_spin.setRange(0.0, 1000.0)
            self.event_overlay_delta_spin.setSingleStep(5.0)
            self.event_overlay_delta_spin.setValue(float(getattr(self.args, "event_overlay_max_sync_delta_ms", 50.0)))
            self.event_overlay_delta_spin.valueChanged.connect(self._on_event_overlay_delta_changed)
            row_sync.addWidget(self.event_overlay_delta_spin)
            row_sync.addWidget(QLabel("mask evidence ms"))
            self.event_mask_period_combo = QComboBox()
            self.event_mask_period_combo.addItems(["4", "8", "12", "16", "25", "30", "33"])
            try:
                self.event_mask_period_combo.setCurrentText(str(int(round(float(getattr(self.args, "event_mask_evidence_period_ms", 30.0))))))
            except Exception:
                pass
            self.event_mask_period_combo.currentTextChanged.connect(self._on_event_mask_period_changed)
            row_sync.addWidget(self.event_mask_period_combo)
            self.event_mask_raster_check = QCheckBox("mask raster")
            self.event_mask_raster_check.setChecked(self._read_bool_control(self._event_mask_control_path(), "mask_raster_enable", bool(getattr(self.args, "event_mask_raster_enable", False))))
            self.event_mask_raster_check.toggled.connect(self._on_event_mask_raster_control_changed)
            row_sync.addWidget(self.event_mask_raster_check)
            row_sync.addWidget(QLabel("raster dilate"))
            self.event_mask_raster_dilate_spin = QSpinBox()
            self.event_mask_raster_dilate_spin.setRange(0, 128)
            self.event_mask_raster_dilate_spin.setSingleStep(1)
            self.event_mask_raster_dilate_spin.setValue(int(self._read_float_control(self._event_mask_control_path(), "mask_raster_dilate_px", float(getattr(self.args, "event_mask_raster_dilate_px", 8)))))
            self.event_mask_raster_dilate_spin.valueChanged.connect(self._on_event_mask_raster_control_changed)
            row_sync.addWidget(self.event_mask_raster_dilate_spin)
            row_sync.addWidget(QLabel("visual slice ms"))
            self.event_visual_mask_slice_spin = QDoubleSpinBox()
            self.event_visual_mask_slice_spin.setRange(1.0, 250.0)
            self.event_visual_mask_slice_spin.setDecimals(2)
            self.event_visual_mask_slice_spin.setSingleStep(1.0)
            default_visual_slice_ms = float(getattr(self.args, "event_visual_mask_slice_us", 33333) or 33333) / 1000.0
            visual_slice_us = self._read_float_control(
                self._event_mask_control_path(),
                "event_visual_mask_slice_us",
                default_visual_slice_ms * 1000.0,
            )
            self.event_visual_mask_slice_spin.setValue(max(1.0, float(visual_slice_us) / 1000.0))
            self.event_visual_mask_slice_spin.valueChanged.connect(self._on_event_visual_mask_timing_changed)
            row_sync.addWidget(self.event_visual_mask_slice_spin)
            row_sync.addWidget(QLabel("visual acc ms"))
            self.event_visual_mask_accumulation_spin = QDoubleSpinBox()
            self.event_visual_mask_accumulation_spin.setRange(1.0, 250.0)
            self.event_visual_mask_accumulation_spin.setDecimals(2)
            self.event_visual_mask_accumulation_spin.setSingleStep(1.0)
            default_visual_acc_ms = float(getattr(self.args, "event_visual_mask_accumulation_us", 33333) or 33333) / 1000.0
            visual_acc_us = self._read_float_control(
                self._event_mask_control_path(),
                "event_visual_mask_accumulation_us",
                default_visual_acc_ms * 1000.0,
            )
            self.event_visual_mask_accumulation_spin.setValue(max(1.0, float(visual_acc_us) / 1000.0))
            self.event_visual_mask_accumulation_spin.valueChanged.connect(self._on_event_visual_mask_timing_changed)
            row_sync.addWidget(self.event_visual_mask_accumulation_spin)
            self.hit_detection_check = QCheckBox("命中判定")
            self.hit_detection_check.setChecked(self._read_bool_control(self._hit_control_path(), "hit_detection_enabled", True))
            self.hit_detection_check.toggled.connect(self._on_hit_detection_enable_changed)
            row_sync.addWidget(self.hit_detection_check)
            self.strong_gate_check = QCheckBox("强门控")
            self.strong_gate_check.setChecked(self._read_bool_control(self._hit_control_path(), "strong_gate_enabled", True))
            self.strong_gate_check.toggled.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.strong_gate_check)
            self.strong_reasoning_check = QCheckBox("强推理")
            self.strong_reasoning_check.setChecked(self._read_bool_control(self._hit_control_path(), "strong_reasoning_enabled", True))
            self.strong_reasoning_check.toggled.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.strong_reasoning_check)
            self.strong_reasoning_shadow_check = QCheckBox("推理影子")
            self.strong_reasoning_shadow_check.setChecked(self._read_bool_control(self._hit_control_path(), "strong_reasoning_shadow_only", True))
            self.strong_reasoning_shadow_check.toggled.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.strong_reasoning_shadow_check)
            row_sync.addWidget(QLabel("final authority"))
            self.final_hit_authority_combo = QComboBox()
            self.final_hit_authority_combo.addItem("强门控", "strong_gate")
            self.final_hit_authority_combo.addItem("强推理", "strong_reasoning")
            self.final_hit_authority_combo.addItem("影子对比", "both_shadow_compare")
            try:
                authority = self._read_str_control(self._hit_control_path(), "final_hit_authority", "strong_gate")
                idx = self.final_hit_authority_combo.findData(authority)
                if idx >= 0:
                    self.final_hit_authority_combo.setCurrentIndex(idx)
            except Exception:
                pass
            self.final_hit_authority_combo.currentIndexChanged.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.final_hit_authority_combo)
            self.human_noise_filter_check = QCheckBox("人体噪声过滤")
            self.human_noise_filter_check.setChecked(self._read_bool_control(self._hit_control_path(), "human_noise_filter_enable", True))
            self.human_noise_filter_check.toggled.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.human_noise_filter_check)
            row_sync.addWidget(QLabel("噪声模式"))
            self.human_noise_mode_combo = QComboBox()
            self.human_noise_mode_combo.addItem("观察", "shadow")
            self.human_noise_mode_combo.addItem("柔性", "soft")
            self.human_noise_mode_combo.addItem("严格", "hard")
            try:
                noise_mode = self._read_str_control(self._hit_control_path(), "human_noise_filter_mode", "soft")
                idx = self.human_noise_mode_combo.findData(noise_mode)
                if idx >= 0:
                    self.human_noise_mode_combo.setCurrentIndex(idx)
                else:
                    self.human_noise_mode_combo.setCurrentIndex(1)
            except Exception:
                self.human_noise_mode_combo.setCurrentIndex(1)
            self.human_noise_mode_combo.currentIndexChanged.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.human_noise_mode_combo)
            row_sync.addWidget(QLabel("bbox start px"))
            self.human_noise_bbox_expand_spin = QDoubleSpinBox()
            self.human_noise_bbox_expand_spin.setRange(0.0, 160.0)
            self.human_noise_bbox_expand_spin.setSingleStep(8.0)
            self.human_noise_bbox_expand_spin.setValue(self._read_float_control(self._hit_control_path(), "human_noise_track_start_bbox_expand_px", 48.0))
            self.human_noise_bbox_expand_spin.valueChanged.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.human_noise_bbox_expand_spin)
            row_sync.addWidget(QLabel("bbox policy"))
            self.human_noise_bbox_policy_combo = QComboBox()
            self.human_noise_bbox_policy_combo.addItem("soft", "soft")
            self.human_noise_bbox_policy_combo.addItem("hard", "hard")
            self.human_noise_bbox_policy_combo.addItem("off", "off")
            try:
                bbox_policy = self._read_str_control(self._hit_control_path(), "human_noise_track_start_bbox_policy", "soft")
                idx = self.human_noise_bbox_policy_combo.findData(bbox_policy)
                self.human_noise_bbox_policy_combo.setCurrentIndex(idx if idx >= 0 else 0)
            except Exception:
                self.human_noise_bbox_policy_combo.setCurrentIndex(0)
            self.human_noise_bbox_policy_combo.currentIndexChanged.connect(self._on_hit_chain_control_changed)
            row_sync.addWidget(self.human_noise_bbox_policy_combo)
            self.global_person_count_gate_check = QCheckBox("全局ID短边门控")
            self.global_person_count_gate_check.setChecked(self._read_bool_control(self._global_person_id_control_path(), "count_gate_enable", True))
            self.global_person_count_gate_check.toggled.connect(self._on_global_person_count_gate_changed)
            row_sync.addWidget(self.global_person_count_gate_check)
            row_sync.addStretch(1)
            page_layout.addLayout(row_sync)

            row_dataset = QHBoxLayout()
            row_dataset.addWidget(QLabel("人员ID数据集"))
            self.person_id_dataset_type_combo = QComboBox()
            self.person_id_dataset_type_combo.addItem("负样本", "negative")
            self.person_id_dataset_type_combo.addItem("正样本", "positive")
            self.person_id_dataset_type_combo.addItem("混合", "mixed")
            self.person_id_dataset_type_combo.addItem("1v1", "battle_1v1")
            row_dataset.addWidget(self.person_id_dataset_type_combo)
            self.person_id_dataset_start_button = QPushButton("开始")
            self.person_id_dataset_stop_button = QPushButton("停止")
            self.person_id_dataset_start_button.clicked.connect(self._on_person_id_dataset_start)
            self.person_id_dataset_stop_button.clicked.connect(self._on_person_id_dataset_stop)
            row_dataset.addWidget(self.person_id_dataset_start_button)
            row_dataset.addWidget(self.person_id_dataset_stop_button)
            self.evidence_replay_start_button = QPushButton("Evidence Replay Start")
            self.evidence_replay_stop_button = QPushButton("Evidence Replay Stop")
            self.full_recording_start_button = QPushButton("Full Record Start")
            self.full_recording_stop_button = QPushButton("Full Record Stop")
            self.evidence_replay_start_button.clicked.connect(self._on_evidence_replay_dataset_start)
            self.evidence_replay_stop_button.clicked.connect(self._on_evidence_replay_dataset_stop)
            self.full_recording_start_button.clicked.connect(lambda: self._on_full_recording_start("person_id_dataset_type_combo"))
            self.full_recording_stop_button.clicked.connect(self._on_full_recording_stop)
            row_dataset.addWidget(self.evidence_replay_start_button)
            row_dataset.addWidget(self.evidence_replay_stop_button)
            row_dataset.addWidget(self.full_recording_start_button)
            row_dataset.addWidget(self.full_recording_stop_button)
            row_dataset.addStretch(1)
            page_layout.addLayout(row_dataset)

            row_raw = QHBoxLayout()
            row_raw.addWidget(QLabel("原始数据录制"))
            self.record_all_button = QPushButton("录制全部")
            self.record_event_button = QPushButton("仅Event RAW")
            self.record_mvs_button = QPushButton("仅MVS RAW")
            self.record_stop_button = QPushButton("停止录制")
            self.record_all_button.clicked.connect(lambda: self._write_record_command("start", ["event_raw", "mvs_raw"], "全部"))
            self.record_event_button.clicked.connect(lambda: self._write_record_command("start", ["event_raw"], "Event RAW"))
            self.record_mvs_button.clicked.connect(lambda: self._write_record_command("start", ["mvs_raw"], "MVS RAW"))
            self.record_stop_button.clicked.connect(lambda: self._write_record_command("stop", ["event_raw", "mvs_raw"], "停止"))
            for btn in (self.record_all_button, self.record_event_button, self.record_mvs_button, self.record_stop_button):
                row_raw.addWidget(btn)
            row_raw.addStretch(1)
            page_layout.addLayout(row_raw)

            row_replay = QHBoxLayout()
            row_replay.addWidget(QLabel("Evidence Replay"))
            self.evidence_pack_button = QPushButton("打包证据")
            self.evidence_select_button = QPushButton("选择回放文件夹")
            self.evidence_replay_button = QPushButton("开始回放")
            self.evidence_stop_button = QPushButton("停止回放")
            self.evidence_pack_button.clicked.connect(self._run_evidence_pack_collector)
            self.evidence_select_button.clicked.connect(self._select_evidence_pack_dir)
            self.evidence_replay_button.clicked.connect(self._run_evidence_replay)
            self.evidence_stop_button.clicked.connect(self._stop_evidence_replay)
            for btn in (self.evidence_pack_button, self.evidence_select_button, self.evidence_replay_button, self.evidence_stop_button):
                row_replay.addWidget(btn)
            row_replay.addStretch(1)
            page_layout.addLayout(row_replay)

            row_mark = QHBoxLayout()
            row_mark.addWidget(QLabel("现场标记"))
            self.annot_shot_button = QPushButton("标记射击")
            self.annot_hit_button = QPushButton("标记命中")
            self.annot_miss_button = QPushButton("标记未中")
            self.annot_false_hit_button = QPushButton("标记误判")
            self.annot_issue_button = QPushButton("直播异常")
            self.annot_shot_button.clicked.connect(lambda: self._write_operator_annotation("shot", "标记射击"))
            self.annot_hit_button.clicked.connect(lambda: self._write_operator_annotation("hit", "标记命中"))
            self.annot_miss_button.clicked.connect(lambda: self._write_operator_annotation("miss", "标记未中"))
            self.annot_false_hit_button.clicked.connect(lambda: self._write_operator_annotation("false_hit", "标记误判"))
            self.annot_issue_button.clicked.connect(lambda: self._write_operator_annotation("live_issue", "直播异常"))
            for btn in (self.annot_shot_button, self.annot_hit_button, self.annot_miss_button, self.annot_false_hit_button, self.annot_issue_button):
                row_mark.addWidget(btn)
            row_mark.addStretch(1)
            page_layout.addLayout(row_mark)

            tree = self._new_status_tree()
            page_layout.addWidget(tree, 1)
            self.tabs.addTab(page, self.control_tab_name)
            self.tree_by_tab[self.control_tab_name] = tree
            self.tab_page_by_tab[self.control_tab_name] = page
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] control page install failed: {type(exc).__name__}: {exc}", flush=True)

    def _install_placeholder(self) -> None:
        try:
            tree = self._make_tree("总览")
            tree.clear()
            self._add_section(tree, "总览", "启动状态", [
                ("status", "waiting for first telemetry refresh"),
                ("note", "preview window should already be visible"),
            ])
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] status placeholder failed: {type(exc).__name__}: {exc}", flush=True)

    def _remember_expanded(self) -> None:
        try:
            for tab_name, tree in self.tree_by_tab.items():
                for i in range(tree.topLevelItemCount()):
                    item = tree.topLevelItem(i)
                    if item is not None:
                        self.expanded_state[(tab_name, item.text(0))] = bool(item.isExpanded())
        except Exception:
            pass

    def _normalize_rows(self, rows: Any) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if rows is None:
            return out
        if isinstance(rows, dict):
            return [(str(k), str(v)) for k, v in rows.items()]
        if isinstance(rows, (str, bytes)):
            return [("value", rows.decode("utf-8", "replace") if isinstance(rows, bytes) else str(rows))]
        try:
            iterator = iter(rows)
        except Exception:
            return [("value", str(rows))]
        for item in iterator:
            try:
                if isinstance(item, dict):
                    out.extend((str(k), str(v)) for k, v in item.items())
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    out.append((str(item[0]), str(item[1])))
                else:
                    out.append(("value", str(item)))
            except Exception as exc:
                out.append(("row_error", f"{type(exc).__name__}: {exc}"))
        return out

    def _normalize_tabs(self, data: Any) -> Dict[str, List[Tuple[str, Any]]]:
        if not isinstance(data, dict):
            return {"总览": [("状态数据错误", [("type", type(data).__name__), ("value", str(data))])]}
        out: Dict[str, List[Tuple[str, Any]]] = {}
        for tab_name, sections in data.items():
            tab_key = str(tab_name) if str(tab_name) else "总览"
            out.setdefault(tab_key, [])
            if isinstance(sections, dict):
                sections_iter = sections.items()
            else:
                try:
                    sections_iter = iter(sections)
                except Exception:
                    sections_iter = [("value", [("value", str(sections))])]
            for sec in sections_iter:
                if isinstance(sec, (tuple, list)) and len(sec) >= 2:
                    title, rows = sec[0], sec[1]
                else:
                    title, rows = "section", [("value", str(sec))]
                out[tab_key].append((str(title), rows))
        return out

    def _recording_tab_sections(self, system_status: Optional[Dict[str, Any]]) -> List[Tuple[str, Any]]:
        cats = {}
        summary = {}
        if isinstance(system_status, dict):
            cats = system_status.get("categories", {}) if isinstance(system_status.get("categories"), dict) else {}
            summary = system_status.get("summary", {}) if isinstance(system_status.get("summary"), dict) else {}
        pds = cats.get("person_id_dataset", {}) if isinstance(cats.get("person_id_dataset"), dict) else {}
        rr = cats.get("raw_recording", {}) if isinstance(cats.get("raw_recording"), dict) else {}
        rr_mvs = rr.get("mvs", {}) if isinstance(rr.get("mvs"), dict) else {}
        rr_event = rr.get("event_raw", {}) if isinstance(rr.get("event_raw"), dict) else {}
        rep = cats.get("replay_evidence", {}) if isinstance(cats.get("replay_evidence"), dict) else {}
        raw_session = str(rr.get("session_id", "") or "")
        dataset_session = str(pds.get("session_id", "") or "")
        full_active = bool(
            rr.get("recording")
            and pds.get("recording")
            and str(pds.get("dataset_kind", "") or "") == "evidence_replay"
            and bool(pds.get("include_mask_polygons"))
            and bool(pds.get("include_bullet_trajectory"))
            and bool(pds.get("include_hit_debug"))
        )
        full_session_match = bool(raw_session and dataset_session and raw_session == dataset_session)
        return [
            ("\u5f55\u5236\u603b\u89c8", [
                ("overall_state", summary.get("overall_state") or "-"),
                ("full_recording_active", full_active),
                ("full_recording_session_match", full_session_match),
                ("raw_recording", rr.get("recording")),
                ("person_dataset_recording", pds.get("recording")),
                ("dataset_kind", pds.get("dataset_kind") or "-"),
                ("dataset_type", pds.get("dataset_type") or "-"),
                ("session_id", dataset_session or raw_session or "-"),
                ("raw_session_id", raw_session or "-"),
                ("dataset_session_id", dataset_session or "-"),
                ("status_key_ok", summary.get("status_key_ok")),
                ("broken_links", ",".join(summary.get("broken_links", []) or []) or "-"),
                ("degraded_links", ",".join(summary.get("degraded_links", []) or []) or "-"),
            ]),
            ("A1 Evidence Replay Dataset v1", [
                ("recording", pds.get("recording")),
                ("dataset_kind", pds.get("dataset_kind") or "-"),
                ("contains_polygon_data", pds.get("contains_polygon_data")),
                ("include_mask_polygons", pds.get("include_mask_polygons")),
                ("include_bullet_trajectory", pds.get("include_bullet_trajectory")),
                ("include_hit_debug", pds.get("include_hit_debug")),
                ("human_rows_total", pds.get("human_rows_total")),
                ("human_person_rows_total", pds.get("human_person_rows_total")),
                ("global_person_track_rows", pds.get("global_person_track_rows")),
                ("bullet_point_rows", pds.get("bullet_point_rows")),
                ("bullet_event_rows", pds.get("bullet_event_rows")),
                ("pseudo_hit_rows", pds.get("pseudo_hit_rows")),
                ("final_hit_rows", pds.get("final_hit_rows")),
                ("shadow_hit_rows", pds.get("shadow_hit_rows")),
                ("hit_reject_rows", pds.get("hit_reject_rows")),
                ("damage_event_rows", pds.get("damage_event_rows")),
                ("bridge_sent_rows", pds.get("bridge_sent_rows")),
                ("bad_lines", pds.get("bad_lines")),
                ("session_dir", pds.get("session_dir") or "-"),
                ("control_file", str(self._person_id_dataset_control_path())),
            ]),
            ("\u539f\u59cbRAW\u5f55\u5236", [
                ("recording", rr.get("recording")),
                ("session_id", rr.get("session_id") or "-"),
                ("last_command", rr.get("last_command") or "-"),
                ("last_command_targets", rr.get("last_command_targets") or "-"),
                ("command_age_ms", rr.get("command_age_ms")),
                ("mvs_recording", rr_mvs.get("recording")),
                ("mvs_recording_count", rr_mvs.get("recording_count")),
                ("mvs_written_frames_total", rr_mvs.get("written_frames_total")),
                ("mvs_dropped_frames_total", rr_mvs.get("dropped_frames_total")),
                ("mvs_queue_size_total", rr_mvs.get("queue_size_total")),
                ("event_recording_count", rr_event.get("recording_count")),
                ("event_slices_written_total", rr_event.get("slices_written_total")),
                ("event_events_written_total", rr_event.get("events_written_total")),
                ("event_dropped_slices_total", rr_event.get("dropped_slices_total")),
                ("event_queue_size_total", rr_event.get("queue_size_total")),
                ("control_file", str(self._record_control_path())),
            ]),
            ("\u73b0\u573a\u6807\u8bb0\u4e0e\u56de\u653e", [
                ("annotation_count", rep.get("annotation_count")),
                ("last_annotation_kind", rep.get("last_annotation_kind") or "-"),
                ("last_annotation_label", rep.get("last_annotation_label") or "-"),
                ("last_annotation_age_ms", rep.get("last_annotation_age_ms")),
                ("selected_replay_folder", self.evidence_selected_pack_dir or rep.get("selected_pack_dir") or "-"),
                ("evidence_replay_proc_running", bool(self.evidence_replay_proc and self.evidence_replay_proc.poll() is None)),
                ("evidence_replay_control", str(self._evidence_control_path())),
                ("operator_annotations", str(self._operator_annotation_path())),
            ]),
        ]

    def _control_tab_sections(self, system_status: Optional[Dict[str, Any]]) -> List[Tuple[str, Any]]:
        cats = {}
        summary = {}
        if isinstance(system_status, dict):
            cats = system_status.get("categories", {}) if isinstance(system_status.get("categories"), dict) else {}
            summary = system_status.get("summary", {}) if isinstance(system_status.get("summary"), dict) else {}
        pds = cats.get("person_id_dataset", {}) if isinstance(cats.get("person_id_dataset"), dict) else {}
        rr = cats.get("raw_recording", {}) if isinstance(cats.get("raw_recording"), dict) else {}
        rr_mvs = rr.get("mvs", {}) if isinstance(rr.get("mvs"), dict) else {}
        rr_event = rr.get("event_raw", {}) if isinstance(rr.get("event_raw"), dict) else {}
        rep = cats.get("replay_evidence", {}) if isinstance(cats.get("replay_evidence"), dict) else {}
        src = cats.get("source_mode_replay", {}) if isinstance(cats.get("source_mode_replay"), dict) else {}
        exp = cats.get("experiment_control", {}) if isinstance(cats.get("experiment_control"), dict) else {}
        gy = cats.get("global_yolo", {}) if isinstance(cats.get("global_yolo"), dict) else {}
        gp = cats.get("global_person_id", {}) if isinstance(cats.get("global_person_id"), dict) else {}
        gp_status = gp.get("status", {}) if isinstance(gp.get("status"), dict) else {}
        gp_runtime = gp_status.get("runtime_config", {}) if isinstance(gp_status.get("runtime_config"), dict) else {}
        gp_count_gate = gp.get("count_gate", {}) if isinstance(gp.get("count_gate"), dict) else {}
        gb = cats.get("global_bev", {}) if isinstance(cats.get("global_bev"), dict) else {}
        hj = cats.get("hit_judge", {}) if isinstance(cats.get("hit_judge"), dict) else {}
        hj_noise = hj.get("human_noise_filter", {}) if isinstance(hj.get("human_noise_filter"), dict) else {}
        mask_control = self._read_json_status(str(self._event_mask_control_path()))
        gb_gp = gb.get("global_person", {}) if isinstance(gb.get("global_person"), dict) else {}
        gb_layer = gb.get("event_layer", {}) if isinstance(gb.get("event_layer"), dict) else {}
        gb_bullet = gb.get("event_bullet", {}) if isinstance(gb.get("event_bullet"), dict) else {}
        mh = cats.get("manual_hit", {}) if isinstance(cats.get("manual_hit"), dict) else {}
        return [
            ("显示与渲染控制", [
                ("event_overlay_preview_enable", bool(getattr(self.args, "event_overlay_preview_enable", False))),
                ("event_overlay_sync_mode", str(getattr(self.args, "event_overlay_sync_mode", ""))),
                ("event_overlay_alpha", float(getattr(self.args, "event_overlay_preview_alpha", 0.0) or 0.0)),
                ("event_overlay_max_sync_delta_ms", float(getattr(self.args, "event_overlay_max_sync_delta_ms", 0.0) or 0.0)),
                ("event_layer_enabled", gb_layer.get("enabled")),
                ("event_layer_alpha", gb_layer.get("alpha")),
                ("event_layer_color_mode", gb_layer.get("color_mode")),
                ("event_layer_dilate_px", gb_layer.get("dilate_px")),
                ("event_layer_backend", gb_layer.get("dilate_backend") or "-"),
                ("event_bullet_enabled", gb_bullet.get("enabled")),
                ("event_visual_mask_slice_us", mask_control.get("event_visual_mask_slice_us")),
                ("event_visual_mask_accumulation_us", mask_control.get("event_visual_mask_accumulation_us")),
                ("event_visual_mask_slice_ms", mask_control.get("event_visual_mask_slice_ms")),
                ("event_visual_mask_accumulation_ms", mask_control.get("event_visual_mask_accumulation_ms")),
                ("human_label_mode", str(getattr(self.args, "human_label_mode", ""))),
            ]),
            ("判定与链路参数", [
                ("hit_detection_enabled", hj.get("hit_detection_enabled")),
                ("hit_candidate_output_enabled", hj.get("hit_candidate_output_enabled")),
                ("strong_gate_enabled", hj.get("strong_gate_enabled")),
                ("strong_reasoning_enabled", hj.get("strong_reasoning_enabled")),
                ("strong_reasoning_shadow_only", hj.get("strong_reasoning_shadow_only")),
                ("final_hit_authority", hj.get("final_hit_authority")),
                ("pseudo_hit_enable", hj.get("pseudo_hit_enable")),
                ("pseudo_hit_total", hj.get("pseudo_hit_total")),
                ("pseudo_hit_promoted_to_final_count", hj.get("pseudo_hit_promoted_to_final_count")),
                ("pseudo_hit_duplicate_suppressed", hj.get("pseudo_hit_duplicate_suppressed")),
                ("human_noise_filter_enable", hj_noise.get("enable")),
                ("human_noise_filter_mode", hj_noise.get("mode")),
                ("pseudo_hit_human_noise_suspect", hj.get("pseudo_hit_human_noise_suspect")),
                ("pseudo_hit_human_noise_suppressed", hj.get("pseudo_hit_human_noise_suppressed")),
                ("pseudo_hit_human_noise_reason_total", hj.get("pseudo_hit_human_noise_reason_total")),
                ("final_hit_duplicate_suppressed", hj.get("final_hit_duplicate_suppressed")),
                ("final_hit_cross_camera_dedup_ms", hj.get("final_hit_cross_camera_dedup_ms")),
                ("global_person_count_gate_enable", gp.get("count_gate_enable", gp_runtime.get("count_gate_enable"))),
                ("global_person_count_gate_zone", gp.get("count_gate_short_edge_zone") or gp_count_gate.get("short_edge_zone") or "-"),
                ("global_person_birth_blocked_non_short_edge", gp.get("count_gate_birth_blocked_non_short_edge", gp_count_gate.get("birth_blocked_non_short_edge"))),
                ("global_person_birth_blocked_handoff", gp.get("count_gate_birth_blocked_handoff", gp_count_gate.get("birth_blocked_handoff"))),
                ("global_person_lost_held_non_short_edge", gp.get("count_gate_lost_held_non_short_edge", gp_count_gate.get("lost_held_non_short_edge"))),
                ("mask_evidence_control", str(self._event_mask_control_path())),
                ("hit_control", str(self._hit_control_path())),
                ("global_person_id_control", str(self._global_person_id_control_path())),
                ("global_person_id_shadow_control", str(self._global_person_id_shadow_control_path())),
                ("global_bev_control", str(self._global_bev_control_path())),
                ("event_overlay_overview_control", str(self._event_overlay_overview_control_path())),
                ("preview_control", str(self._preview_control_path())),
                ("manual_hit_control", str(self._manual_hit_control_path())),
                ("game_event_bridge_control", str(self._game_event_bridge_control_path())),
                ("manual_hit_preview_click_enable", mh.get("preview_click_enable")),
                ("manual_hit_preview_target_cache_count", mh.get("preview_target_cache_count")),
                ("manual_hit_source_global_id", mh.get("source_global_id")),
                ("manual_hit_state", mh.get("state") or "-"),
            ]),
            ("数据与复盘入口", [
                ("person_id_dataset_control", str(self._person_id_dataset_control_path())),
                ("raw_record_control", str(self._record_control_path())),
                ("evidence_replay_control", str(self._evidence_control_path())),
                ("selected_replay_folder", self.evidence_selected_pack_dir or "-"),
                ("evidence_replay_proc_running", bool(self.evidence_replay_proc and self.evidence_replay_proc.poll() is None)),
            ]),
            ("V25启动源模式/离线回放", [
                ("source_mode_state", src.get("state") or "-"),
                ("event_source_mode_effective", src.get("event_source_mode_effective") or "-"),
                ("mvs_source_mode_effective", src.get("mvs_source_mode_effective") or "-"),
                ("dataset_id", src.get("dataset_id") or "-"),
                ("session_id", src.get("session_id") or "-"),
                ("source_mode_effective_exists", src.get("source_mode_effective_exists")),
                ("v25_resolved_manifest_exists", src.get("v25_resolved_manifest_exists")),
                ("usable_for_full_replay_startup", src.get("usable_for_full_replay_startup")),
                ("usable_for_full_replay_sync", src.get("usable_for_full_replay_sync")),
                ("v25_replay_control_exists", src.get("v25_replay_control_exists")),
                ("v25_replay_status_exists", src.get("v25_replay_status_exists")),
                ("replay_control_mode", src.get("replay_control_mode") or "-"),
                ("replay_timing", src.get("replay_timing") or "-"),
                ("source_mode_effective_path", src.get("source_mode_effective_path") or "-"),
                ("v25_resolved_manifest_path", src.get("v25_resolved_manifest_path") or "-"),
                ("v25_replay_control", str(self._sync_root_path("v25_replay_control.json"))),
                ("v25_replay_status", str(self._sync_root_path("v25_replay_status.json"))),
            ]),
            ("实验热更新", [
                ("experiment_id", exp.get("experiment_id") or "-"),
                ("stage", exp.get("stage") or "-"),
                ("config_hash", exp.get("config_hash") or "-"),
                ("state", exp.get("state") or "-"),
                ("source", exp.get("source") or "-"),
                ("final_hit_authority", exp.get("final_hit_authority") or "-"),
                ("human_noise_filter_mode", exp.get("human_noise_filter_mode") or "-"),
                ("human_noise_track_start_bbox_expand_px", exp.get("human_noise_track_start_bbox_expand_px")),
                ("global_person_target", exp.get("global_person_target") or "-"),
                ("global_person_filter_backend", exp.get("global_person_filter_backend") or "-"),
                ("run_state", exp.get("run_state") or "-"),
                ("runtime_overall_state", exp.get("runtime_overall_state") or "-"),
                ("run_dir", exp.get("run_dir") or "-"),
                ("experiment_control", str(self._sync_control_path("experiment_control_json", "experiment_control.json"))),
                ("experiment_status", str(self._sync_control_path("experiment_status_json", "experiment_status_latest.json"))),
                ("test_run_status", str(self._sync_control_path("test_run_status_json", "test_run_status_latest.json"))),
            ]),
            ("人员ID数据集", [
                ("recording", pds.get("recording")),
                ("dataset_type", pds.get("dataset_type") or "-"),
                ("session_id", pds.get("session_id") or "-"),
                ("run_id", pds.get("run_id") or "-"),
                ("experiment_id", pds.get("experiment_id") or "-"),
                ("experiment_stage", pds.get("experiment_stage") or "-"),
                ("experiment_config_hash", pds.get("experiment_config_hash") or "-"),
                ("contains_mask_data", pds.get("contains_mask_data")),
                ("source_stage", pds.get("source_stage") or "-"),
                ("human_rows_total", pds.get("human_rows_total")),
                ("human_person_rows_total", pds.get("human_person_rows_total")),
                ("human_zero_person_rows_total", pds.get("human_zero_person_rows_total")),
                ("human_zero_person_skipped_total", pds.get("human_zero_person_skipped_total")),
                ("global_person_track_rows", pds.get("global_person_track_rows")),
                ("status_samples", pds.get("status_samples")),
                ("bad_lines", pds.get("bad_lines")),
                ("last_error", pds.get("last_error") or "-"),
                ("session_dir", pds.get("session_dir") or "-"),
            ]),
            ("原始数据录制", [
                ("recording", rr.get("recording")),
                ("session_id", rr.get("session_id") or "-"),
                ("last_command", rr.get("last_command") or "-"),
                ("mvs_recording", rr_mvs.get("recording")),
                ("mvs_written_frames_total", rr_mvs.get("written_frames_total")),
                ("mvs_dropped_frames_total", rr_mvs.get("dropped_frames_total")),
                ("event_recording_count", rr_event.get("recording_count")),
                ("event_slices_written_total", rr_event.get("slices_written_total")),
                ("event_events_written_total", rr_event.get("events_written_total")),
                ("event_dropped_slices_total", rr_event.get("dropped_slices_total")),
                ("degraded_links", ",".join(rr.get("degraded_links", []) or []) or "-"),
            ]),
            ("Evidence Replay / 标记", [
                ("session_id", rep.get("session_id") or "-"),
                ("record_replay_match", rep.get("record_replay_match")),
                ("annotation_count", rep.get("annotation_count")),
                ("last_annotation_kind", rep.get("last_annotation_kind") or "-"),
                ("last_annotation_label", rep.get("last_annotation_label") or "-"),
                ("last_annotation_age_ms", rep.get("last_annotation_age_ms")),
            ]),
            ("实时链路对齐", [
                ("overall_state", summary.get("overall_state") or "-"),
                ("broken_links", ",".join(summary.get("broken_links", []) or []) or "-"),
                ("degraded_links", ",".join(summary.get("degraded_links", []) or []) or "-"),
                ("global_yolo_worker_count", gy.get("worker_count")),
                ("global_yolo_published_fps_total", gy.get("published_fps_total", gy.get("infer_fps"))),
                ("global_yolo_inferred_fps_total", gy.get("inferred_fps_total")),
                ("global_yolo_async_record_postprocess_enabled", gy.get("async_record_postprocess_enabled")),
                ("global_yolo_postprocess_queue_depth", gy.get("postprocess_queue_depth")),
                ("global_yolo_postprocess_dropped_batches", gy.get("postprocess_dropped_batches")),
                ("global_yolo_polygon_contract_ok", gy.get("polygon_contract_ok")),
                ("global_person_track_count", gp.get("track_count")),
                ("global_person_count_gate_enable", gp.get("count_gate_enable")),
                ("global_bev_render_fps", gb.get("render_fps")),
                ("global_bev_output_fps", gb.get("output_fps")),
                ("global_bev_person_track_count", gb_gp.get("track_count")),
            ]),
        ]

    def _add_section(self, tree: QTreeWidget, tab_name: str, title: str, rows: Any) -> None:
        top = QTreeWidgetItem([str(title), ""])
        try:
            top.setFirstColumnSpanned(True)
        except Exception:
            pass
        tree.addTopLevelItem(top)
        for k, v in self._normalize_rows(rows):
            child = QTreeWidgetItem([str(k), str(v)])
            top.addChild(child)
        top.setExpanded(self.expanded_state.get((tab_name, str(title)), True))

    def refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            system_status: Optional[Dict[str, Any]] = None
            try:
                if hasattr(self, "log_writer") and self.log_writer is not None:
                    system_status = self.log_writer.build_system_status()
                    data = self._normalize_tabs(self.log_writer.system_status_to_tabs(system_status))
                else:
                    data = self._normalize_tabs(self.preview.build_status_tabs())
                # Keep a copy so a transient diagnostics bug does not remove the
                # established monitoring tabs from the status window.
                self._last_good_status_data = {k: list(v) for k, v in data.items()}
            except Exception as exc:
                if self._last_good_status_data:
                    data = {k: list(v) for k, v in self._last_good_status_data.items()}
                    data.setdefault("总览", [])
                    data["总览"] = [("状态窗口刷新失败", [(type(exc).__name__, str(exc))])] + list(data["总览"])
                else:
                    data = {"总览": [("状态窗口刷新失败", [(type(exc).__name__, str(exc))])]}
                print(f"[pygl_cuda_preview][WARN] status build failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                if hasattr(self, "log_writer") and self.log_writer is not None:
                    log_tab = "总览" if "总览" in data else "Overview"
                    data.setdefault(log_tab, []).append(("状态日志", self.log_writer.status_rows()))
                    data[self.recording_tab_name] = self._recording_tab_sections(system_status)
                    data[self.control_tab_name] = self._control_tab_sections(system_status)
                    self.log_writer.maybe_write(data, system_status=system_status)
            except Exception as exc:
                print(f"[pygl_cuda_preview][WARN] status log hook failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                data.setdefault(self.recording_tab_name, self._recording_tab_sections(system_status))
                data.setdefault(self.control_tab_name, self._control_tab_sections(system_status))
            except Exception as exc:
                print(f"[pygl_cuda_preview][WARN] control tab build failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                rows = []
                top_tab = "总览" if "总览" in data else "Overview"
                for title, sec_rows in data.get(top_tab, [])[:4]:
                    rows.append(str(title))
                self.fallback_label.setText(f"Status: refreshed {_now_ms():.0f}ms tabs={len(data)} sections={','.join(rows[:4])}")
            except Exception:
                pass
            self._remember_expanded()
            current = self.tabs.currentIndex()
            existing_names = set(self.tree_by_tab.keys())
            for tab_name in data.keys():
                if tab_name not in existing_names:
                    self._make_tree(tab_name)
            for tab_name in list(self.tree_by_tab.keys()):
                if tab_name not in data:
                    tree = self.tree_by_tab.pop(tab_name)
                    page = self.tab_page_by_tab.pop(tab_name, tree)
                    idx = self.tabs.indexOf(page)
                    if idx >= 0:
                        self.tabs.removeTab(idx)
                    try:
                        page.deleteLater()
                    except Exception:
                        tree.deleteLater()
            for tab_name, sections in data.items():
                tree = self.tree_by_tab.get(tab_name) or self._make_tree(tab_name)
                tree.setUpdatesEnabled(False)
                tree.clear()
                for title, rows in sections:
                    self._add_section(tree, tab_name, title, rows)
                try:
                    tree.resizeColumnToContents(0)
                except Exception:
                    pass
                tree.setUpdatesEnabled(True)
            if 0 <= current < self.tabs.count():
                self.tabs.setCurrentIndex(current)
        except Exception as exc:
            print(f"[pygl_cuda_preview][WARN] status window refresh failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            self._refreshing = False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if hasattr(self, "log_writer") and self.log_writer is not None:
                self.log_writer.close()
        except Exception:
            pass
        return super().closeEvent(event)


class OutboundHitEventsWindow(QWidget):
    """Small top-level window for the actual game-event-bridge hit output."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.setWindowTitle("\u5df2\u53d1\u9001\u547d\u4e2d\u4e8b\u4ef6 - Game Event Bridge")
        self.resize(int(getattr(args, "outbound_hit_window_width", 760)), int(getattr(args, "outbound_hit_window_height", 520)))
        self.setStyleSheet("""
            QWidget { background: #101419; color: #d9f2ff; }
            QTreeWidget { background: #0d1117; color: #d9f2ff; alternate-background-color: #151b22; border: 1px solid #283946; }
            QHeaderView::section { background: #1b2a34; color: #e8f8ff; padding: 4px; border: 0px; }
            QLabel { color: #b8f0ff; background: #0d1117; padding: 5px; border: 1px solid #283946; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.summary_label = QLabel("\u7b49\u5f85\u53d1\u9001\u94fe\u8def\u72b6\u6001...")
        self.summary_label.setFont(QFont("DejaVu Sans Mono", max(10, int(getattr(args, "status_window_font_px", 13)) - 1)))
        layout.addWidget(self.summary_label)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["key", "value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setFont(QFont("DejaVu Sans Mono", int(getattr(args, "status_window_font_px", 13))))
        layout.addWidget(self.tree, 1)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(max(100, int(getattr(args, "outbound_hit_window_poll_ms", 500))))
        QTimer.singleShot(150, self.refresh)

    def _sync_root_path(self, filename: str) -> Path:
        p = Path(str(filename or "")).expanduser()
        if p.is_absolute():
            return p
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root / p

    def _resolve_sync_or_project_path(self, value: str, default_name: str) -> Path:
        raw = str(value or default_name or "").strip()
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p
        if raw.replace("\\", "/").startswith("sync_ipc/"):
            return Path.cwd() / p
        return self._sync_root_path(raw or default_name)

    def _read_json(self, filename: str) -> Dict[str, Any]:
        path = self._sync_root_path(filename)
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
        try:
            mtime_ms = path.stat().st_mtime * 1000.0 if path.exists() else 0.0
        except Exception:
            mtime_ms = 0.0
        obj["_status_path"] = str(path)
        obj["_file_exists"] = bool(path.exists())
        obj["_file_age_ms"] = max(0.0, _now_ms() - mtime_ms) if mtime_ms > 0 else -1.0
        return obj

    def _tail_jsonl(self, path: Path) -> Tuple[List[Dict[str, Any]], str, int, float]:
        records: List[Dict[str, Any]] = []
        error = ""
        size = 0
        age_ms = -1.0
        tail_bytes = int(getattr(self.args, "outbound_hit_window_tail_bytes", 768_000) or 768_000)
        try:
            if not path.exists():
                return [], "missing", 0, -1.0
            stat = path.stat()
            size = int(stat.st_size)
            age_ms = max(0.0, _now_ms() - float(stat.st_mtime) * 1000.0)
            with path.open("r", encoding="utf-8", errors="replace") as fp:
                start = max(0, size - int(max(8192, tail_bytes)))
                fp.seek(start)
                if start > 0:
                    fp.readline()
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except Exception:
                        continue
            if len(records) > 200:
                records = records[-200:]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return records, error, size, age_ms

    def _add_section(self, title: str, rows: List[Tuple[str, Any]]) -> None:
        top = QTreeWidgetItem([str(title), ""])
        try:
            top.setFirstColumnSpanned(True)
        except Exception:
            pass
        self.tree.addTopLevelItem(top)
        for key, value in rows:
            top.addChild(QTreeWidgetItem([str(key), str(value)]))
        top.setExpanded(True)

    def refresh(self) -> None:
        try:
            bridge = self._read_json("game_event_bridge_status.json")
            latest = self._read_json("game_event_bridge_latest.json")
            sent_path = self._resolve_sync_or_project_path(str(bridge.get("sent_jsonl", "") or ""), "game_event_bridge_sent.jsonl")
            rows, tail_error, file_size, file_age_ms = self._tail_jsonl(sent_path)
            hit_rows: List[Dict[str, Any]] = []
            now_s = time.time()
            for rec in rows:
                payload = rec.get("payload", {}) if isinstance(rec.get("payload"), dict) else {}
                if str(payload.get("type", "")).lower() != "hit":
                    continue
                wall_s = _safe_float(rec.get("wall_time"), 0.0)
                hit_rows.append({
                    "age_ms": max(0.0, (now_s - wall_s) * 1000.0) if wall_s > 0 else -1.0,
                    "sent": bool(rec.get("sent", False)),
                    "reason": str(rec.get("reason", "")),
                    "event_id": str(payload.get("event_id", "")),
                    "camera": str(payload.get("camera", "")),
                    "source_yolo_id": _safe_int(payload.get("source_yolo_id"), -1),
                    "target_yolo_id": _safe_int(payload.get("target_yolo_id"), -1),
                    "target_global_id": str(payload.get("target_global_id", "")),
                    "target_display_player_id": str(payload.get("target_display_player_id", "")),
                    "target_id_source": str(payload.get("target_id_source", "")),
                    "confidence": _safe_float(payload.get("confidence"), 0.0),
                    "source_override": bool(payload.get("source_override", False)),
                    "event_source": str(payload.get("event_source", "")),
                    "manual": bool(payload.get("manual", False)),
                    "manual_authority": str(payload.get("manual_authority", "")),
                })
            recent_hits = hit_rows[-32:]
            sent_count = sum(1 for rec in recent_hits if bool(rec.get("sent")))
            fail_count = sum(1 for rec in recent_hits if (not bool(rec.get("sent"))) and str(rec.get("reason", "")) == "send_failed")
            shadow_count = sum(1 for rec in recent_hits if (not bool(rec.get("sent"))) and str(rec.get("reason", "")) == "shadow")
            stats = bridge.get("stats", {}) if isinstance(bridge.get("stats"), dict) else {}
            client_error = str(bridge.get("client_last_error", "") or "")
            mode = str(bridge.get("mode", latest.get("mode", "")) or "")
            state = "OK"
            if not bool(bridge.get("_file_exists")):
                state = "DISABLED"
            elif client_error:
                state = "DEGRADED"
            elif tail_error and tail_error != "missing":
                state = "DEGRADED"
            self.summary_label.setText(
                f"\u5df2\u53d1\u9001\u547d\u4e2d\u4e8b\u4ef6: {state} mode={mode or '-'} "
                f"hit={len(recent_hits)} sent={sent_count} failed={fail_count} "
                f"target={bridge.get('windows_host', '-') or '-'}:{bridge.get('windows_port', '-') or '-'}"
            )
            self.tree.setUpdatesEnabled(False)
            self.tree.clear()
            self._add_section("\u53d1\u9001\u94fe\u8def", [
                ("state", state),
                ("mode", mode or "-"),
                ("hit_source_mode", bridge.get("hit_source_mode") or "-"),
                ("send_hits_enabled", bridge.get("send_hits_enabled")),
                ("send_presence_enabled", bridge.get("send_presence_enabled")),
                ("client_connected", bridge.get("client_connected")),
                ("client_last_error", client_error or "-"),
                ("control_error", bridge.get("control_error") or "-"),
                ("windows_target", f"{bridge.get('windows_host', '-') or '-'}:{bridge.get('windows_port', '-') or '-'}"),
                ("force_hit_source_global_id", bridge.get("force_hit_source_global_id")),
                ("virtual_presence_global_id", bridge.get("virtual_presence_global_id")),
                ("control_json", bridge.get("control_json") or "-"),
                ("sent_jsonl", str(sent_path)),
                ("sent_jsonl_size", file_size),
                ("sent_jsonl_age_ms", f"{file_age_ms:.0f}"),
                ("tail_error", tail_error if tail_error != "missing" else "-"),
            ])
            self._add_section("\u547d\u4e2d\u7edf\u8ba1", [
                ("recent_hit_count", len(recent_hits)),
                ("recent_sent_count", sent_count),
                ("recent_shadow_count", shadow_count),
                ("recent_send_failed_count", fail_count),
                ("total_sent_hit", _safe_int(stats.get("sent_hit"), 0)),
                ("total_shadow_hit", _safe_int(stats.get("shadow_hit"), 0)),
                ("total_send_failed_hit", _safe_int(stats.get("send_failed_hit"), 0)),
                ("suppress_non_manual_hit_by_source_mode", _safe_int(stats.get("suppress_non_manual_hit_by_source_mode"), 0)),
                ("suppress_manual_hit_by_source_mode", _safe_int(stats.get("suppress_manual_hit_by_source_mode"), 0)),
                ("latest_payload_type", (latest.get("latest_payload", {}).get("payload", {}) if isinstance(latest.get("latest_payload"), dict) else {}).get("type", "-")),
            ])
            event_rows: List[Tuple[str, Any]] = []
            for idx, rec in enumerate(recent_hits[-24:], 1):
                event_id = str(rec.get("event_id") or f"event_{idx}")
                key = f"{idx:02d} {rec.get('camera') or '-'} {event_id[-12:]}"
                event_rows.append((
                    key,
                    f"age={_safe_float(rec.get('age_ms'), -1.0):.0f}ms sent={rec.get('sent')} reason={rec.get('reason') or '-'} "
                    f"src={rec.get('source_yolo_id')} tgt={rec.get('target_yolo_id')} "
                    f"target_global={rec.get('target_global_id') or '-'} display={rec.get('target_display_player_id') or '-'} "
                    f"id_source={rec.get('target_id_source') or '-'} conf={_safe_float(rec.get('confidence'), 0.0):.2f} "
                    f"manual={rec.get('manual')} event_source={rec.get('event_source') or '-'} override={rec.get('source_override')}",
                ))
            self._add_section("\u6700\u8fd1 outbound hit", event_rows or [("recent", "-")])
            try:
                self.tree.resizeColumnToContents(0)
            except Exception:
                pass
            self.tree.setUpdatesEnabled(True)
        except Exception as exc:
            self.summary_label.setText(f"\u5df2\u53d1\u9001\u547d\u4e2d\u4e8b\u4ef6\u7a97\u53e3\u5237\u65b0\u5931\u8d25: {type(exc).__name__}: {exc}")
            print(f"[pygl_cuda_preview][WARN] outbound hit events window refresh failed: {type(exc).__name__}: {exc}", flush=True)

def main() -> int:
    ap = argparse.ArgumentParser(description="Independent PyOpenGL/CUDA preview renderer: 40 FPS video + 250 FPS bullet trajectory")
    ap.add_argument("--config", default="", help="Kept for launcher compatibility; not parsed by this renderer yet")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--raw-template", default="mvs_raw_latest_{camera}")
    ap.add_argument("--trajectory-shm", default="preview_trajectory")
    ap.add_argument("--video-fps", type=float, default=40.0)
    ap.add_argument("--render-fps", type=float, default=250.0)
    ap.add_argument("--trajectory-fps", type=float, default=250.0)
    ap.add_argument("--trajectory-life-ms", type=float, default=2500.0)
    ap.add_argument("--bullet-status-ttl-ms", type=float, default=0.0,
                    help="TTL for bullet diagnostics; 0 uses max(5000, trajectory_life_ms*4)")
    ap.add_argument("--max-trajectory-segments", type=int, default=65536)
    ap.add_argument("--trajectory-line-width", type=float, default=2.0)
    ap.add_argument("--cuda-device", type=int, default=0)
    ap.add_argument("--tile-width", type=int, default=640)
    ap.add_argument("--tile-height", type=int, default=360)
    ap.add_argument("--layout", default="grid2x4")
    ap.add_argument("--window-width", type=int, default=2560)
    ap.add_argument("--window-height", type=int, default=1440)
    ap.add_argument("--vsync", type=int, default=0)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--v25-live-shadow-enable", action="store_true", default=False)
    ap.add_argument("--v25-live-shadow-root", default="v25_shadow/live")
    ap.add_argument("--overlay-ws-url", default="ws://127.0.0.1:8765")
    ap.add_argument("--overlay-ws-disable", action="store_true")
    ap.add_argument("--hit-jsonl-enable", action="store_true", default=True)
    ap.add_argument("--hit-jsonl-template", default="hit_candidate_{camera}.jsonl")
    ap.add_argument("--hit-all-jsonl", default="hit_candidate_all.jsonl")
    ap.add_argument("--pseudo-hit-jsonl-template", default="pseudo_hit_{camera}.jsonl")
    ap.add_argument("--pseudo-hit-all-jsonl", default="pseudo_hit_all.jsonl")
    ap.add_argument("--preview-trajectory-filter-human-noise", action="store_true", default=True,
                    help="Filter main preview trajectory segments whose pseudo hit was suppressed by human-noise rules.")
    ap.add_argument("--preview-trajectory-filter-human-noise-disable", dest="preview_trajectory_filter_human_noise", action="store_false",
                    help="Show raw preview trajectory shm without human-noise suppression filtering.")
    ap.add_argument("--overlay-poll-ms", type=float, default=50.0)
    ap.add_argument("--overlay-marker-life-ms", type=float, default=3000.0)
    ap.add_argument("--overlay-text-life-ms", type=float, default=3200.0)
    ap.add_argument("--overlay-marker-line-width", type=float, default=2.5)
    ap.add_argument("--max-overlay-markers", type=int, default=4096)
    ap.add_argument("--max-overlay-marker-labels", type=int, default=96)
    ap.add_argument("--max-overlay-text-lines", type=int, default=6)
    ap.add_argument("--status-text-enable", action="store_true", default=True)
    ap.add_argument("--status-font-px", type=int, default=14)
    ap.add_argument("--status-panel-width", type=int, default=980)
    ap.add_argument("--status-panel-alpha", type=int, default=190)
    ap.add_argument("--telemetry-poll-ms", type=float, default=500.0)
    ap.add_argument("--mvs-shm-template", default="mvs_latest_{camera}")
    ap.add_argument("--tile-label-enable", action="store_true", default=True)
    ap.add_argument("--human-jsonl-enable", action="store_true", default=True)
    ap.add_argument("--human-jsonl-template", default="human_result_{camera}.jsonl")
    ap.add_argument("--human-jsonl-tail-bytes", type=int, default=4_000_000)
    ap.add_argument("--human-jsonl-backfill-interval-ms", type=float, default=500.0)
    ap.add_argument("--human-mask-draw-enable", action="store_true", default=True)
    ap.add_argument("--human-mask-render-mode", choices=["off", "auto", "polygon", "bbox"], default="auto",
                    help="Preview-only human mask fill mode. auto uses polygons when present, otherwise bbox fallback.")
    ap.add_argument("--human-mask-fill-enable", action="store_true", default=True)
    ap.add_argument("--human-mask-life-ms", type=float, default=5000.0)
    ap.add_argument("--human-min-conf", type=float, default=0.15)
    ap.add_argument("--human-mask-alpha", type=int, default=70)
    ap.add_argument("--human-mask-line-width", type=float, default=1.5)
    ap.add_argument("--human-bbox-line-width", type=float, default=1.5)
    ap.add_argument("--human-label-mode", choices=["id_only", "detail", "off"], default="detail",
                    help="Preview human label mode: id_only, detail, or off.")
    ap.add_argument("--human-coord-point-enable", action="store_true", default=True)
    ap.add_argument("--hit-prompt-enable", action="store_true", default=True)
    ap.add_argument("--hit-prompt-life-ms", type=float, default=4200.0)
    ap.add_argument("--hit-prompt-font-px", type=int, default=32)
    ap.add_argument("--max-hit-prompts", type=int, default=4)
    ap.add_argument("--preview-hit-display-enable", action="store_true", default=True)
    ap.add_argument("--preview-hit-display-disable", dest="preview_hit_display_enable", action="store_false")
    ap.add_argument("--preview-trajectory-only-mode", action="store_true", default=False)
    ap.add_argument("--manual-hit-click-enable", action="store_true", default=False)
    ap.add_argument("--manual-hit-command-json", default="manual_hit_commands.jsonl")
    ap.add_argument("--manual-hit-control-json", default="manual_hit_control.json")
    ap.add_argument("--game-event-bridge-control-json", default="game_event_bridge_control.json")
    ap.add_argument("--max-human-draw", type=int, default=32)
    ap.add_argument("--max-human-polygons-per-person", type=int, default=8)
    ap.add_argument("--max-human-polygon-points", type=int, default=512)
    ap.add_argument("--status-window-disable", action="store_true", help="Disable the separate diagnostics status window")
    ap.add_argument("--status-window-width", type=int, default=1180)
    ap.add_argument("--status-window-height", type=int, default=860)
    ap.add_argument("--status-window-font-px", type=int, default=13)
    ap.add_argument("--status-window-poll-ms", type=float, default=500.0)
    ap.add_argument("--hit-control-json", default="")
    ap.add_argument("--global-bev-control-json", default="")
    ap.add_argument("--event-overlay-overview-control-json", default="")
    ap.add_argument("--global-bev-event-layer-alpha", type=float, default=0.45)
    ap.add_argument("--global-bev-event-layer-color-mode", choices=["camera", "red", "yellow"], default="camera")
    ap.add_argument("--global-bev-event-layer-dilate-px", type=float, default=16.0)
    ap.add_argument("--preview-control-json", default="")
    ap.add_argument("--person-id-dataset-control-json", default="")
    ap.add_argument("--person-id-dataset-status-json", default="")
    ap.add_argument("--person-id-dataset-latest-json", default="")
    ap.add_argument("--status-inline-enable", action="store_true", help="Also draw the dense status panel inside the preview window; default off")
    ap.add_argument("--status-cam-group", type=int, default=4, help="How many cameras to show per line in the status window")
    ap.add_argument("--status-log-enable", action="store_true", help="Save status diagnostics to JSONL/CSV for offline analysis")
    ap.add_argument("--status-log-dir", default="", help="Directory for status logs; default <sync_root>/status_logs/run_<time>_pid<PID>")
    ap.add_argument("--status-log-interval-ms", type=float, default=1000.0, help="Minimum status log write interval")
    ap.add_argument("--status-log-tabs-jsonl", action="store_true", help="Also save the status-window tab rows to status_tabs.jsonl")
    ap.add_argument("--status-log-latest-disable", action="store_true", help="Do not maintain status_latest.json")
    ap.add_argument("--window-x", type=int, default=-100000, help="Preview window X position; default primary screen left")
    ap.add_argument("--window-y", type=int, default=-100000, help="Preview window Y position; default primary screen top")
    ap.add_argument("--status-window-x", type=int, default=-100000, help="Status window X position; default primary screen left + 32")
    ap.add_argument("--status-window-y", type=int, default=-100000, help="Status window Y position; default primary screen top + 32")
    ap.add_argument("--outbound-hit-window-disable", action="store_true", help="Disable the independent outbound hit event window")
    ap.add_argument("--outbound-hit-window-width", type=int, default=760)
    ap.add_argument("--outbound-hit-window-height", type=int, default=520)
    ap.add_argument("--outbound-hit-window-x", type=int, default=-100000)
    ap.add_argument("--outbound-hit-window-y", type=int, default=-100000)
    ap.add_argument("--outbound-hit-window-poll-ms", type=float, default=500.0)
    ap.add_argument("--outbound-hit-window-tail-bytes", type=int, default=768_000)
    ap.add_argument("--no-force-raise-windows", action="store_true", help="Do not call raise_/activateWindow after creating preview/status windows")
    ap.add_argument("--event-overlay-preview-enable", action="store_true", help="Show event camera visualization in an independent overview window")
    ap.add_argument("--event-overlay-main-inline-enable", action="store_true", help="Debug fallback: also draw event overlay over the RGB preview tiles")
    ap.add_argument("--event-overlay-window-disable", action="store_true", help="Do not create the independent event overlay overview window")
    ap.add_argument("--event-overlay-window-width", type=int, default=1280)
    ap.add_argument("--event-overlay-window-height", type=int, default=720)
    ap.add_argument("--event-overlay-window-x", type=int, default=-100000)
    ap.add_argument("--event-overlay-window-y", type=int, default=-100000)
    ap.add_argument("--event-overlay-window-fps", type=float, default=50.0)
    ap.add_argument("--event-overlay-window-label-fps", type=float, default=10.0)
    ap.add_argument("--event-overlay-window-font-px", type=int, default=12)
    ap.add_argument("--event-overlay-window-layout", default="grid2x4")
    ap.add_argument("--event-overlay-preview-template", default="event_overlay_latest_{camera},event_overlay_masked_latest_{camera}")
    ap.add_argument("--event-overlay-sidecar-template", default="event_overlay_{camera}_latest.json,event_overlay_masked_{camera}_latest.json")
    ap.add_argument("--event-overlay-preview-alpha", type=float, default=0.55)
    ap.add_argument("--event-overlay-preview-gain", type=float, default=3.0)
    ap.add_argument("--event-overlay-preview-gamma", type=float, default=0.65)
    ap.add_argument("--event-overlay-preview-min-alpha", type=float, default=0.35)
    ap.add_argument("--event-overlay-preview-dilate-px", type=int, default=4)
    ap.add_argument("--event-overlay-preview-dilate-max-pixels", type=int, default=5000)
    ap.add_argument("--event-overlay-preview-dilate-backend", choices=["gl_shader", "cpu_sparse", "off", "auto"], default="gl_shader")
    ap.add_argument("--event-overlay-stale-alpha-scale", type=float, default=0.25)
    ap.add_argument("--event-overlay-preview-max-age-ms", type=float, default=100.0)
    ap.add_argument("--event-overlay-sync-mode", choices=["latest", "nearest_mvs", "nearest_human"], default="nearest_mvs")
    ap.add_argument("--event-overlay-max-sync-delta-ms", type=float, default=50.0)
    ap.add_argument("--event-overlay-hide-stale", action="store_true")
    ap.add_argument("--event-overlay-show-stats", action="store_true", default=True)
    ap.add_argument("--allow-cpu-upload-fallback", action="store_true")
    ap.add_argument("--allow-non-nvidia-gl", action="store_true")
    args = ap.parse_args()

    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(int(args.vsync))
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    print(f"[pygl_cuda_preview] QApplication created; Qt screens: {_screen_summary(app)}", flush=True)
    force_raise = not bool(getattr(args, "no_force_raise_windows", False))

    try:
        w = PreviewWidget(args)
        _place_window(
            w, app,
            preferred_x=int(getattr(args, "window_x", -100000)),
            preferred_y=int(getattr(args, "window_y", -100000)),
            preferred_w=int(args.window_width),
            preferred_h=int(args.window_height),
            label="preview",
        )
        _show_and_raise(w, label="preview", force_raise=force_raise)
        print("[pygl_cuda_preview] preview window shown", flush=True)
    except Exception as exc:
        print(f"[pygl_cuda_preview][ERROR] preview window init failed: {type(exc).__name__}: {exc}", flush=True)
        raise

    app._pygl_cuda_preview_window = w  # type: ignore[attr-defined]
    app._pygl_cuda_preview_status_window = None  # type: ignore[attr-defined]
    app._pygl_cuda_preview_event_overlay_window = None  # type: ignore[attr-defined]
    app._pygl_cuda_preview_outbound_hit_window = None  # type: ignore[attr-defined]

    def _create_event_overlay_window() -> None:
        if bool(getattr(args, "event_overlay_window_disable", False)):
            print("[pygl_cuda_preview] event overlay overview window disabled by args", flush=True)
            return
        try:
            ew = EventOverlayOverviewWindow(w, args)
            ex = int(getattr(args, "event_overlay_window_x", -100000))
            ey = int(getattr(args, "event_overlay_window_y", -100000))
            if ex < -99999 or ey < -99999:
                try:
                    sc = app.primaryScreen() or (app.screens()[0] if app.screens() else None)
                    if sc is not None:
                        g = sc.availableGeometry()
                        if ex < -99999:
                            ex = int(g.x() + 64)
                        if ey < -99999:
                            ey = int(g.y() + 96)
                except Exception:
                    pass
            _place_window(
                ew, app,
                preferred_x=ex,
                preferred_y=ey,
                preferred_w=int(getattr(args, "event_overlay_window_width", 1280)),
                preferred_h=int(getattr(args, "event_overlay_window_height", 720)),
                label="event-overlay",
            )
            w.set_event_overlay_overview_window(ew)
            app._pygl_cuda_preview_event_overlay_window = ew  # type: ignore[attr-defined]
            if bool(getattr(args, "event_overlay_preview_enable", False)):
                _show_and_raise(ew, label="event-overlay", force_raise=force_raise)
                print("[pygl_cuda_preview] event overlay overview window shown", flush=True)
            else:
                ew.hide()
                print("[pygl_cuda_preview] event overlay overview window created hidden", flush=True)
        except Exception as exc:
            app._pygl_cuda_preview_event_overlay_window = None  # type: ignore[attr-defined]
            print(f"[pygl_cuda_preview][WARN] event overlay overview window disabled after init failure: {type(exc).__name__}: {exc}", flush=True)

    def _create_status_window() -> None:
        if bool(getattr(args, "status_window_disable", False)):
            print("[pygl_cuda_preview] status diagnostics window disabled by args", flush=True)
            return
        try:
            sw = StatusWindow(w, args)
            sx = int(getattr(args, "status_window_x", -100000))
            sy = int(getattr(args, "status_window_y", -100000))
            if sx < -99999 or sy < -99999:
                try:
                    sc = app.primaryScreen() or (app.screens()[0] if app.screens() else None)
                    if sc is not None:
                        g = sc.availableGeometry()
                        if sx < -99999:
                            sx = int(g.x() + 32)
                        if sy < -99999:
                            sy = int(g.y() + 32)
                except Exception:
                    pass
            _place_window(
                sw, app,
                preferred_x=sx,
                preferred_y=sy,
                preferred_w=int(getattr(args, "status_window_width", 1180)),
                preferred_h=int(getattr(args, "status_window_height", 860)),
                label="status",
            )
            _show_and_raise(sw, label="status", force_raise=force_raise)
            app._pygl_cuda_preview_status_window = sw  # type: ignore[attr-defined]
            print("[pygl_cuda_preview] status diagnostics window shown", flush=True)
        except Exception as exc:
            # Diagnostics are optional; never let a status-window bug prevent video preview.
            app._pygl_cuda_preview_status_window = None  # type: ignore[attr-defined]
            print(f"[pygl_cuda_preview][WARN] status diagnostics window disabled after init failure: {type(exc).__name__}: {exc}", flush=True)

    def _create_outbound_hit_window() -> None:
        if bool(getattr(args, "outbound_hit_window_disable", False)):
            print("[pygl_cuda_preview] outbound hit events window disabled by args", flush=True)
            return
        try:
            ohw = OutboundHitEventsWindow(args)
            ow = int(getattr(args, "outbound_hit_window_width", 760))
            oh = int(getattr(args, "outbound_hit_window_height", 520))
            ox = int(getattr(args, "outbound_hit_window_x", -100000))
            oy = int(getattr(args, "outbound_hit_window_y", -100000))
            if ox < -99999 or oy < -99999:
                try:
                    sc = app.primaryScreen() or (app.screens()[0] if app.screens() else None)
                    if sc is not None:
                        g = sc.availableGeometry()
                        if ox < -99999:
                            ox = int(max(g.x(), g.x() + g.width() - ow - 32))
                        if oy < -99999:
                            oy = int(g.y() + 64)
                except Exception:
                    pass
            _place_window(
                ohw, app,
                preferred_x=ox,
                preferred_y=oy,
                preferred_w=ow,
                preferred_h=oh,
                label="outbound-hit",
            )
            _show_and_raise(ohw, label="outbound-hit", force_raise=force_raise)
            app._pygl_cuda_preview_outbound_hit_window = ohw  # type: ignore[attr-defined]
            print("[pygl_cuda_preview] outbound hit events window shown", flush=True)
        except Exception as exc:
            app._pygl_cuda_preview_outbound_hit_window = None  # type: ignore[attr-defined]
            print(f"[pygl_cuda_preview][WARN] outbound hit events window disabled after init failure: {type(exc).__name__}: {exc}", flush=True)

    QTimer.singleShot(0, _create_event_overlay_window)
    QTimer.singleShot(50, _create_status_window)
    QTimer.singleShot(100, _create_outbound_hit_window)

    if force_raise:
        def _late_raise() -> None:
            _show_and_raise(w, label="preview", force_raise=True)
            ew = getattr(app, "_pygl_cuda_preview_event_overlay_window", None)
            if ew is not None and bool(getattr(args, "event_overlay_preview_enable", False)):
                _show_and_raise(ew, label="event-overlay", force_raise=True)
            sw = getattr(app, "_pygl_cuda_preview_status_window", None)
            if sw is not None:
                _show_and_raise(sw, label="status", force_raise=True)
            ohw = getattr(app, "_pygl_cuda_preview_outbound_hit_window", None)
            if ohw is not None:
                _show_and_raise(ohw, label="outbound-hit", force_raise=True)
        QTimer.singleShot(750, _late_raise)

    print("[pygl_cuda_preview] entering Qt event loop", flush=True)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
