#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global physical-space player ID server.

This process is intentionally downstream of YOLO.  It tails the per-camera
human_result_*.jsonl files, merges detections in field coordinates, assigns a
stable global_player_id, and publishes compact latest/jsonl diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return bool(default)


def _parse_cams(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").replace(";", ",").split(",") if x.strip()]


def _sync_index(rec: Dict[str, Any]) -> int:
    for key in ("sync_index", "sync_index_hint", "mvs_sync_index", "program_sync_index"):
        if rec.get(key) is not None:
            return _as_int(rec.get(key), -1)
    meta = rec.get("mvs_meta")
    if isinstance(meta, dict):
        for key in ("program_sync_index", "sync_index", "mvs_sync_index"):
            if meta.get(key) is not None:
                return _as_int(meta.get(key), -1)
    return -1


def _person_xy(p: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for key in ("fused_xy", "local_xy", "local_xy_raw"):
        xy = p.get(key)
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return _as_float(xy[0]), _as_float(xy[1])
    pc = p.get("physical_coord")
    if isinstance(pc, dict) and pc.get("ok", True):
        if pc.get("x_m") is not None and pc.get("y_m") is not None:
            return _as_float(pc.get("x_m")), _as_float(pc.get("y_m"))
    return None


def _coord_mode(s: Any) -> str:
    mode = str(s or "raw").strip().lower().replace("-", "_")
    aliases = {
        "none": "raw",
        "normal": "raw",
        "height_plus_y": "field_height_plus_y",
        "field_h_plus_y": "field_height_plus_y",
        "minus_y": "neg_y",
        "flip_y": "invert_y",
    }
    return aliases.get(mode, mode)


def _person_bbox_center(p: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    bb = p.get("bbox_xyxy") or p.get("bbox")
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return (_as_float(bb[0]) + _as_float(bb[2])) * 0.5, (_as_float(bb[1]) + _as_float(bb[3])) * 0.5
    return None


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


class JsonlTailer:
    def __init__(self, path: Path, start_at_end: bool = True):
        self.path = Path(path)
        self.fp = None
        self.pos = 0
        self.start_at_end = bool(start_at_end)
        self.last_error = ""
        self.bad_lines = 0
        self.last_good_us = 0
        self.last_bad_us = 0

    def _open(self) -> bool:
        if self.fp is not None:
            return True
        try:
            if not self.path.exists():
                return False
            self.fp = self.path.open("r", encoding="utf-8", errors="replace")
            if self.start_at_end:
                self.fp.seek(0, os.SEEK_END)
            self.pos = self.fp.tell()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.fp = None
            return False

    def poll(self, max_lines: int = 256) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._open():
            return out
        try:
            try:
                size = self.path.stat().st_size
                if size < self.pos:
                    self.fp.close()
                    self.fp = None
                    self.start_at_end = False
                    return out
            except Exception:
                pass
            for _ in range(max(1, int(max_lines))):
                start_pos = self.fp.tell()
                line = self.fp.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    self.fp.seek(start_pos, os.SEEK_SET)
                    self.pos = start_pos
                    break
                self.pos = self.fp.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                        self.last_good_us = _now_us()
                        self.last_error = ""
                except Exception as exc:
                    self.bad_lines += 1
                    self.last_bad_us = _now_us()
                    self.last_error = f"json:{type(exc).__name__}: {exc}"
            return out
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None
            return out


@dataclass
class Detection:
    cam: str
    sync_index: int
    timestamp_us: int
    frame_id: int
    local_id: int
    x: float
    y: float
    raw_x: float
    raw_y: float
    confidence: float
    camera_center_weight: float
    anchor_quality: float
    calibration_weight: float
    view_geometry_weight: float
    fusion_weight: float
    fusion_cluster_id: int
    fusion_state: str
    person: Dict[str, Any]
    record: Dict[str, Any]


@dataclass
class Observation:
    sync_index: int
    timestamp_us: int
    x: float
    y: float
    raw_x: float
    raw_y: float
    confidence: float
    cameras: List[str]
    local_ids: Dict[str, int]
    detections: List[Detection] = field(default_factory=list)
    fusion_weight_sum: float = 0.0
    fusion_members: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Track:
    gid: int
    x: float
    y: float
    raw_x: float = 0.0
    raw_y: float = 0.0
    assoc_x: float = 0.0
    assoc_y: float = 0.0
    assoc_vx: float = 0.0
    assoc_vy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    state: str = "tentative"
    hits: int = 1
    misses: int = 0
    created_us: int = 0
    last_ts_us: int = 0
    last_sync_index: int = -1
    last_update_wall: float = field(default_factory=time.time)
    cameras: List[str] = field(default_factory=list)
    local_ids: Dict[str, int] = field(default_factory=dict)
    history: List[Tuple[int, float, float]] = field(default_factory=list)
    measurement_x: float = 0.0
    measurement_y: float = 0.0
    measurement_raw_x: float = 0.0
    measurement_raw_y: float = 0.0
    kalman_state: Optional[np.ndarray] = field(default=None, repr=False)
    kalman_cov: Optional[np.ndarray] = field(default=None, repr=False)
    kalman_last_predict_ts_us: int = 0
    last_innovation_m: float = 0.0
    last_gate_reason: str = ""
    filter_backend: str = "alpha_beta"
    fusion_weight_sum: float = 0.0
    fusion_members: List[Dict[str, Any]] = field(default_factory=list)
    last_handoff_wall: float = 0.0
    handoff_lock_until_wall: float = 0.0
    handoff_lock_reason: str = ""
    handoff_lock_cameras: List[str] = field(default_factory=list)
    count_gate_zone: str = ""
    count_gate_hold_until_wall: float = 0.0
    count_gate_hard_retire_after_wall: float = 0.0
    count_gate_hold_reason: str = ""

    def predict(self, ts_us: int) -> Tuple[float, float]:
        if self.last_ts_us <= 0:
            return self.x, self.y
        dt = _clip((int(ts_us) - int(self.last_ts_us)) / 1_000_000.0, 0.0, 1.5)
        return self.x + self.vx * dt, self.y + self.vy * dt


@dataclass
class BirthCandidate:
    cid: int
    x: float
    y: float
    raw_x: float
    raw_y: float
    confidence: float
    first_ts_us: int
    last_ts_us: int
    last_update_wall: float = field(default_factory=time.time)
    hits: int = 1
    cameras: List[str] = field(default_factory=list)
    local_ids: Dict[str, int] = field(default_factory=dict)
    sync_indices: set = field(default_factory=set, repr=False)
    history: List[Tuple[int, float, float]] = field(default_factory=list)
    count_gate_zone: str = ""
    count_gate_reason: str = ""


class GlobalPersonIdServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cams = _parse_cams(args.cams)
        self.sync_root = Path(args.sync_root)
        self.tailers: Dict[str, JsonlTailer] = {
            cam: JsonlTailer(self._path(args.human_jsonl_template, cam), start_at_end=bool(args.start_tail))
            for cam in self.cams
        }
        self.pending: Dict[int, List[Detection]] = {}
        self.tracks: Dict[int, Track] = {}
        self.birth_candidates: Dict[int, BirthCandidate] = {}
        self.next_gid = 1
        self.next_birth_cid = 1
        self.latest: Dict[str, Any] = {}
        self.status: Dict[str, Any] = {}
        self.out_jsonl_path = self._abs_path(args.output_jsonl)
        self.latest_json_path = self._abs_path(args.latest_json)
        self.status_json_path = self._abs_path(args.status_json)
        self.control_json_path = self._abs_path(getattr(args, "control_json", "global_person_id_control.json"))
        self.out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_fp = self.out_jsonl_path.open("a" if bool(args.append_jsonl) else "w", encoding="utf-8")
        self.stats = {
            "records": 0,
            "detections": 0,
            "detections_out_of_field": 0,
            "observations_rejected_speed": 0,
            "observations": 0,
            "created_tracks": 0,
            "matched_tracks": 0,
            "id_switch_suspects": 0,
            "retired_tracks": 0,
            "birth_candidates_created": 0,
            "birth_candidates_matched": 0,
            "birth_candidates_promoted": 0,
            "birth_candidates_expired": 0,
            "birth_suppressed_near_track": 0,
            "birth_suppressed_duplicate_candidate": 0,
            "handoff_cluster_merges": 0,
            "handoff_match_gate_used": 0,
            "handoff_birth_suppressed": 0,
            "handoff_duplicate_suppressed": 0,
            "handoff_strict_lock_updates": 0,
            "handoff_strict_birth_blocked": 0,
            "handoff_strict_publish_holds": 0,
            "handoff_strict_retire_holds": 0,
            "count_gate_birth_allowed_short_edge": 0,
            "count_gate_birth_allowed_bootstrap": 0,
            "count_gate_birth_blocked_non_short_edge": 0,
            "count_gate_birth_blocked_interior": 0,
            "count_gate_birth_blocked_long_edge": 0,
            "count_gate_birth_blocked_handoff": 0,
            "count_gate_birth_promotion_blocked": 0,
            "count_gate_lost_held_non_short_edge": 0,
            "count_gate_publish_holds": 0,
            "count_gate_retire_holds": 0,
            "count_gate_exit_allowed_short_edge": 0,
            "count_gate_hard_retired_after_cap": 0,
            "control_updates": 0,
            "control_errors": 0,
            "track_resets": 0,
        }
        self.start_wall = time.time()
        self.first_detection_wall = 0.0
        self._last_status_t = 0.0
        self._last_publish_t = 0.0
        self._last_control_t = 0.0
        self._last_control_seq = 0
        self._last_control_mtime_ns = 0
        self.control_status = {
            "path": str(self.control_json_path),
            "seq": 0,
            "source": "",
            "last_error": "",
            "updated_wall_us": 0,
            "applied_wall_us": 0,
        }
        self.latest_input_sync_index = -1
        self.coord_mode = _coord_mode(getattr(args, "coord_transform_mode", "raw"))
        self.filter_backend = self._normalize_filter_backend(getattr(args, "filter_backend", "alpha_beta"))
        setattr(self.args, "filter_backend", self.filter_backend)

    def _transform_xy(self, x: float, y: float) -> Tuple[float, float]:
        mode = self.coord_mode
        ox = float(getattr(self.args, "field_origin_x_m", 0.0))
        oy = float(getattr(self.args, "field_origin_y_m", 0.0))
        fw = float(getattr(self.args, "field_width_m", 33.0))
        fh = float(getattr(self.args, "field_height_m", 22.0))
        if mode == "raw":
            tx, ty = x, y
        elif mode == "neg_y":
            tx, ty = x, -y
        elif mode == "field_height_plus_y":
            tx, ty = x, oy + fh + y
        elif mode == "invert_y":
            tx, ty = x, oy + fh - (y - oy)
        elif mode == "flip_x":
            tx, ty = ox + fw - (x - ox), y
        elif mode == "flip_xy":
            tx, ty = ox + fw - (x - ox), oy + fh - (y - oy)
        elif mode == "swap_xy":
            tx, ty = y, x
        elif mode == "swap_xy_neg_y":
            tx, ty = -y, x
        else:
            tx, ty = x, y
        return float(tx), float(ty)

    def _in_field(self, x: float, y: float, margin_m: float = 2.0) -> bool:
        ox = float(getattr(self.args, "field_origin_x_m", 0.0))
        oy = float(getattr(self.args, "field_origin_y_m", 0.0))
        fw = float(getattr(self.args, "field_width_m", 33.0))
        fh = float(getattr(self.args, "field_height_m", 22.0))
        return (ox - margin_m) <= x <= (ox + fw + margin_m) and (oy - margin_m) <= y <= (oy + fh + margin_m)

    def _path(self, template: str, cam: str) -> Path:
        s = str(template).format(root=str(self.sync_root), camera=cam, cam=cam)
        p = Path(s)
        if not p.is_absolute():
            p = self.sync_root / p
        return p

    def _abs_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.sync_root / p
        return p

    def _normalize_filter_backend(self, value: Any) -> str:
        backend = str(value or "alpha_beta").strip().lower().replace("-", "_")
        if backend in {"alpha", "alpha_beta", "ab", "legacy"}:
            return "alpha_beta"
        if backend in {"kalman", "kalman_cv", "cv"}:
            return "kalman_cv"
        return "alpha_beta"

    def _runtime_config(self) -> Dict[str, Any]:
        return {
            "stream_role": str(getattr(self.args, "stream_role", "official")),
            "filter_backend": str(self.filter_backend),
            "cluster_gate_m": float(self.args.cluster_gate_m),
            "same_cam_cluster_gate_m": float(self.args.same_cam_cluster_gate_m),
            "match_gate_m": float(self.args.match_gate_m),
            "lost_match_gate_m": float(self.args.lost_match_gate_m),
            "match_score_min": float(self.args.match_score_min),
            "duplicate_suppress_enable": bool(self.args.duplicate_suppress_enable),
            "duplicate_suppress_dist_m": float(self.args.duplicate_suppress_dist_m),
            "publish_min_hits": int(self.args.publish_min_hits),
            "publish_lost_max_age_ms": float(self.args.publish_lost_max_age_ms),
            "reid_memory_ms": float(self.args.reid_memory_ms),
            "retire_after_ms": float(self.args.retire_after_ms),
            "position_alpha": float(self.args.position_alpha),
            "velocity_alpha": float(self.args.velocity_alpha),
            "birth_quarantine_enable": bool(getattr(self.args, "birth_quarantine_enable", True)),
            "birth_confirm_hits": int(getattr(self.args, "birth_confirm_hits", 2)),
            "birth_match_gate_m": float(getattr(self.args, "birth_match_gate_m", 0.85)),
            "birth_near_track_suppress_m": float(getattr(self.args, "birth_near_track_suppress_m", 0.65)),
            "birth_candidate_ttl_ms": float(getattr(self.args, "birth_candidate_ttl_ms", 900.0)),
            "handoff_aware_enable": bool(getattr(self.args, "handoff_aware_enable", True)),
            "cross_camera_cluster_gate_m": float(getattr(self.args, "cross_camera_cluster_gate_m", self.args.cluster_gate_m)),
            "handoff_match_gate_m": float(getattr(self.args, "handoff_match_gate_m", self.args.match_gate_m)),
            "handoff_birth_near_track_suppress_m": float(getattr(self.args, "handoff_birth_near_track_suppress_m", self.args.birth_near_track_suppress_m)),
            "handoff_duplicate_suppress_dist_m": float(getattr(self.args, "handoff_duplicate_suppress_dist_m", self.args.duplicate_suppress_dist_m)),
            "handoff_strict_count_lock_enable": bool(getattr(self.args, "handoff_strict_count_lock_enable", True)),
            "handoff_strict_birth_block_m": float(getattr(self.args, "handoff_strict_birth_block_m", 3.0)),
            "handoff_strict_publish_hold_ms": float(getattr(self.args, "handoff_strict_publish_hold_ms", 1800.0)),
            "count_gate_enable": bool(getattr(self.args, "count_gate_enable", True)),
            "count_gate_x_min_m": float(getattr(self.args, "count_gate_x_min_m", 0.0)),
            "count_gate_x_max_m": float(getattr(self.args, "count_gate_x_max_m", 30.0)),
            "count_gate_y_min_m": float(getattr(self.args, "count_gate_y_min_m", 0.0)),
            "count_gate_y_max_m": float(getattr(self.args, "count_gate_y_max_m", 20.0)),
            "count_gate_short_edge_band_m": float(getattr(self.args, "count_gate_short_edge_band_m", 3.0)),
            "count_gate_short_edge_y_pad_m": float(getattr(self.args, "count_gate_short_edge_y_pad_m", 2.0)),
            "count_gate_startup_bootstrap_ms": float(getattr(self.args, "count_gate_startup_bootstrap_ms", 5000.0)),
            "count_gate_bootstrap_allow_anywhere_until_track_count": int(getattr(self.args, "count_gate_bootstrap_allow_anywhere_until_track_count", 8)),
            "count_gate_non_short_edge_lost_hold_ms": float(getattr(self.args, "count_gate_non_short_edge_lost_hold_ms", 5000.0)),
            "count_gate_non_short_edge_retire_hard_cap_ms": float(getattr(self.args, "count_gate_non_short_edge_retire_hard_cap_ms", 15000.0)),
            "kalman_use_for_association": bool(getattr(self.args, "kalman_use_for_association", False)),
            "kalman_process_noise_m": float(getattr(self.args, "kalman_process_noise_m", 0.35)),
            "kalman_measurement_noise_m": float(getattr(self.args, "kalman_measurement_noise_m", 0.45)),
            "kalman_reject_innovation_enable": bool(getattr(self.args, "kalman_reject_innovation_enable", False)),
            "kalman_max_innovation_m": float(getattr(self.args, "kalman_max_innovation_m", 2.5)),
        }

    def _reset_tracks(self, reason: str = "control") -> None:
        self.pending.clear()
        self.tracks.clear()
        self.birth_candidates.clear()
        self.next_gid = 1
        self.next_birth_cid = 1
        self.stats["track_resets"] = int(self.stats.get("track_resets", 0)) + 1
        self.control_status["last_reset_reason"] = str(reason)
        self.control_status["last_reset_wall_us"] = _now_us()

    def _apply_control(self, obj: Dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            raise ValueError("control root must be object")

        old_backend = str(self.filter_backend)
        float_fields = {
            "cluster_gate_m", "same_cam_cluster_gate_m", "duplicate_suppress_dist_m",
            "match_gate_m", "lost_match_gate_m", "match_score_min",
            "position_alpha", "velocity_alpha", "max_match_speed_mps",
            "max_update_speed_mps", "publish_lost_max_age_ms", "reid_memory_ms",
            "retire_after_ms", "miss_grace_ms", "lost_after_ms", "local_id_bonus",
            "local_id_bonus_max", "id_switch_suspect_gate_m", "kalman_process_noise_m",
            "kalman_measurement_noise_m", "kalman_initial_pos_var",
            "kalman_initial_vel_var", "kalman_max_innovation_m",
            "birth_match_gate_m", "birth_near_track_suppress_m",
            "birth_candidate_ttl_ms", "cross_camera_cluster_gate_m",
            "handoff_match_gate_m", "handoff_birth_near_track_suppress_m",
            "handoff_duplicate_suppress_dist_m",
            "handoff_strict_birth_block_m", "handoff_strict_publish_hold_ms",
            "count_gate_x_min_m", "count_gate_x_max_m",
            "count_gate_y_min_m", "count_gate_y_max_m",
            "count_gate_short_edge_band_m", "count_gate_short_edge_y_pad_m",
            "count_gate_startup_bootstrap_ms",
            "count_gate_non_short_edge_lost_hold_ms",
            "count_gate_non_short_edge_retire_hard_cap_ms",
        }
        int_fields = {
            "publish_min_hits", "confirm_hits", "history_len", "output_history_len",
            "sync_delay", "max_lines_per_poll", "max_sync_flush_per_loop",
            "birth_confirm_hits", "count_gate_bootstrap_allow_anywhere_until_track_count",
        }
        bool_fields = {
            "duplicate_suppress_enable", "kalman_reject_innovation_enable",
            "birth_quarantine_enable", "kalman_use_for_association",
            "handoff_aware_enable", "handoff_strict_count_lock_enable",
            "count_gate_enable",
        }

        for key in float_fields:
            if key in obj:
                setattr(self.args, key, _as_float(obj.get(key), float(getattr(self.args, key, 0.0))))
        for key in int_fields:
            if key in obj:
                setattr(self.args, key, _as_int(obj.get(key), int(getattr(self.args, key, 0))))
        for key in bool_fields:
            if key in obj:
                setattr(self.args, key, _as_bool(obj.get(key), bool(getattr(self.args, key, False))))
        if "filter_backend" in obj:
            self.filter_backend = self._normalize_filter_backend(obj.get("filter_backend"))
            setattr(self.args, "filter_backend", self.filter_backend)

        if self.filter_backend == "kalman_cv" and old_backend != "kalman_cv":
            for tr in self.tracks.values():
                self._ensure_kalman_state(tr)
        if _as_bool(obj.get("reset_tracks"), False):
            self._reset_tracks(str(obj.get("reset_reason") or "control_reset_tracks"))

    def _poll_control(self) -> None:
        now = time.time()
        if now - self._last_control_t < max(0.05, float(getattr(self.args, "control_poll_ms", 200.0)) / 1000.0):
            return
        self._last_control_t = now
        try:
            if not self.control_json_path.exists():
                self.control_status["last_error"] = ""
                return
            st = self.control_json_path.stat()
            seq_hint = 0
            if st.st_mtime_ns == self._last_control_mtime_ns:
                return
            obj = json.loads(self.control_json_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                seq_hint = _as_int(obj.get("seq"), 0)
                if seq_hint and seq_hint == self._last_control_seq:
                    self._last_control_mtime_ns = st.st_mtime_ns
                    return
            self._apply_control(obj)
            self._last_control_mtime_ns = st.st_mtime_ns
            if seq_hint:
                self._last_control_seq = seq_hint
            self.control_status.update({
                "seq": int(seq_hint),
                "source": str(obj.get("source", "")) if isinstance(obj, dict) else "",
                "updated_wall_us": _as_int(obj.get("updated_wall_us"), 0) if isinstance(obj, dict) else 0,
                "applied_wall_us": _now_us(),
                "last_error": "",
            })
            self.stats["control_updates"] = int(self.stats.get("control_updates", 0)) + 1
        except Exception as exc:
            self.stats["control_errors"] = int(self.stats.get("control_errors", 0)) + 1
            self.control_status["last_error"] = f"{type(exc).__name__}: {exc}"

    def _ensure_kalman_state(self, tr: Track) -> None:
        if tr.kalman_state is not None and tr.kalman_cov is not None:
            return
        tr.kalman_state = np.asarray([float(tr.x), float(tr.y), float(tr.vx), float(tr.vy)], dtype=np.float64)
        pos_var = max(1e-6, float(getattr(self.args, "kalman_initial_pos_var", 0.6)))
        vel_var = max(1e-6, float(getattr(self.args, "kalman_initial_vel_var", 4.0)))
        tr.kalman_cov = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
        tr.kalman_last_predict_ts_us = int(tr.last_ts_us)

    def _kalman_predict_arrays(self, tr: Track, ts_us: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_kalman_state(tr)
        assert tr.kalman_state is not None
        assert tr.kalman_cov is not None
        dt = _clip((int(ts_us) - int(tr.kalman_last_predict_ts_us or tr.last_ts_us)) / 1_000_000.0, 0.0, 1.5)
        f = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = max(1e-8, float(getattr(self.args, "kalman_process_noise_m", 0.35))) ** 2
        q_pos = q * max(1e-4, dt * dt)
        q_vel = q * max(1e-3, dt)
        q_mat = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        x_pred = f @ tr.kalman_state
        p_pred = f @ tr.kalman_cov @ f.T + q_mat
        return x_pred, p_pred

    def _predict_track(self, tr: Track, ts_us: int, for_association: bool = False) -> Tuple[float, float]:
        use_kalman_for_assoc = bool(getattr(self.args, "kalman_use_for_association", False))
        if for_association and not (str(self.filter_backend) == "kalman_cv" and use_kalman_for_assoc):
            ax = float(tr.assoc_x if tr.assoc_x or tr.assoc_y else tr.x)
            ay = float(tr.assoc_y if tr.assoc_x or tr.assoc_y else tr.y)
            avx = float(tr.assoc_vx)
            avy = float(tr.assoc_vy)
            if tr.last_ts_us <= 0:
                return ax, ay
            dt = _clip((int(ts_us) - int(tr.last_ts_us)) / 1_000_000.0, 0.0, 1.5)
            return ax + avx * dt, ay + avy * dt
        if str(self.filter_backend) == "kalman_cv" and (not for_association or use_kalman_for_assoc):
            try:
                x_pred, _p_pred = self._kalman_predict_arrays(tr, ts_us)
                return float(x_pred[0]), float(x_pred[1])
            except Exception:
                return tr.predict(ts_us)
        return tr.predict(ts_us)

    def _kalman_update_track(self, tr: Track, obs: Observation) -> Dict[str, float]:
        self._ensure_kalman_state(tr)
        x_pred, p_pred = self._kalman_predict_arrays(tr, obs.timestamp_us)
        z = np.asarray([float(obs.x), float(obs.y)], dtype=np.float64)
        h = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        r = max(1e-8, float(getattr(self.args, "kalman_measurement_noise_m", 0.45))) ** 2
        r_mat = np.diag([r, r]).astype(np.float64)
        innovation = z - (h @ x_pred)
        s_mat = h @ p_pred @ h.T + r_mat
        k_gain = p_pred @ h.T @ np.linalg.inv(s_mat)
        x_new = x_pred + k_gain @ innovation
        p_new = (np.eye(4, dtype=np.float64) - k_gain @ h) @ p_pred
        tr.kalman_state = x_new
        tr.kalman_cov = p_new
        tr.kalman_last_predict_ts_us = int(obs.timestamp_us)
        tr.x = float(x_new[0])
        tr.y = float(x_new[1])
        tr.vx = float(x_new[2])
        tr.vy = float(x_new[3])
        tr.last_innovation_m = float(np.linalg.norm(innovation))
        tr.last_gate_reason = "kalman_update"
        return {
            "kalman_innovation_m": float(tr.last_innovation_m),
            "kalman_cov_trace": float(np.trace(p_new)),
            "pred_x": float(x_pred[0]),
            "pred_y": float(x_pred[1]),
        }

    def _record_to_detections(self, cam: str, rec: Dict[str, Any]) -> List[Detection]:
        if str(rec.get("msg_type", "person_regions")) not in {"person_regions", ""}:
            return []
        persons = rec.get("persons")
        if not isinstance(persons, list):
            return []
        if persons and float(getattr(self, "first_detection_wall", 0.0)) <= 0.0:
            self.first_detection_wall = time.time()
        sync_index = _sync_index(rec)
        ts = _as_int(rec.get("capture_wall_us", rec.get("timestamp_us", rec.get("record_wall_us", 0))), _now_us())
        frame_id = _as_int(rec.get("frame_id_local", rec.get("frame_id", 0)), 0)
        out: List[Detection] = []
        for p in persons:
            if not isinstance(p, dict):
                continue
            xy = _person_xy(p)
            if xy is None or not np.all(np.isfinite(xy)):
                continue
            raw_x, raw_y = float(xy[0]), float(xy[1])
            tx, ty = self._transform_xy(raw_x, raw_y)
            if not np.all(np.isfinite((tx, ty))):
                continue
            if not self._in_field(tx, ty, float(self.args.field_margin_m)):
                self.stats["detections_out_of_field"] += 1
                if bool(self.args.drop_out_of_field):
                    continue
            conf = _as_float(p.get("confidence", 0.0), 0.0)
            if conf < float(self.args.min_confidence):
                continue
            local_id = _as_int(p.get("stable_id", p.get("local_stable_id", p.get("person_id", -1))), -1)
            out.append(Detection(
                cam=str(cam),
                sync_index=int(sync_index),
                timestamp_us=int(ts),
                frame_id=int(frame_id),
                local_id=int(local_id),
                x=float(tx),
                y=float(ty),
                raw_x=float(raw_x),
                raw_y=float(raw_y),
                confidence=float(conf),
                camera_center_weight=_clip(_as_float(p.get("camera_center_weight", 1.0), 1.0), 0.01, 1.0),
                anchor_quality=_clip(_as_float(p.get("anchor_quality", 1.0), 1.0), 0.01, 1.0),
                calibration_weight=_clip(_as_float(p.get("calibration_weight", 1.0), 1.0), 0.01, 1.0),
                view_geometry_weight=_clip(_as_float(p.get("view_geometry_weight", 1.0), 1.0), 0.01, 1.0),
                fusion_weight=_clip(
                    _as_float(
                        p.get("fusion_weight"),
                        conf * _clip(_as_float(p.get("camera_center_weight", 1.0), 1.0), 0.01, 1.0),
                    ),
                    0.01,
                    1.0,
                ),
                fusion_cluster_id=_as_int(p.get("fusion_cluster_id", -1), -1),
                fusion_state=str(p.get("fusion_state", "")),
                person=p,
                record=rec,
            ))
        return out

    def _handoff_enabled(self) -> bool:
        return bool(getattr(self.args, "handoff_aware_enable", True))

    def _handoff_strict_enabled(self) -> bool:
        return self._handoff_enabled() and bool(getattr(self.args, "handoff_strict_count_lock_enable", True))

    def _camera_sets_differ(self, a: Iterable[str], b: Iterable[str]) -> bool:
        aa = {str(c) for c in a if str(c)}
        bb = {str(c) for c in b if str(c)}
        if not aa or not bb:
            return False
        return aa != bb

    def _obs_is_camera_handoff_for_track(self, tr: Track, obs: Observation) -> bool:
        if not self._handoff_enabled():
            return False
        obs_cams = {str(c) for c in getattr(obs, "cameras", []) if str(c)}
        track_cams = {str(c) for c in getattr(tr, "cameras", []) if str(c)}
        if not obs_cams or not track_cams:
            return False
        return not obs_cams.issubset(track_cams)

    def _obs_is_strict_handoff_zone_for_track(self, tr: Track, obs: Observation) -> bool:
        if not self._handoff_strict_enabled():
            return False
        obs_cams = {str(c) for c in getattr(obs, "cameras", []) if str(c)}
        if len(obs_cams) > 1:
            return True
        return self._camera_sets_differ(obs_cams, getattr(tr, "cameras", []))

    def _track_handoff_locked(self, tr: Track, now: Optional[float] = None) -> bool:
        if not self._handoff_strict_enabled():
            return False
        t = time.time() if now is None else float(now)
        return float(getattr(tr, "handoff_lock_until_wall", 0.0)) > t

    def _mark_handoff_lock(self, tr: Track, obs: Observation, reason: str) -> None:
        if not self._handoff_strict_enabled():
            return
        hold_ms = max(0.0, float(getattr(self.args, "handoff_strict_publish_hold_ms", 1800.0)))
        if hold_ms <= 0.0:
            return
        now = time.time()
        cams = sorted(
            {str(c) for c in getattr(tr, "cameras", []) if str(c)}
            | {str(c) for c in getattr(obs, "cameras", []) if str(c)}
            | {str(c) for c in getattr(tr, "handoff_lock_cameras", []) if str(c)}
        )
        tr.last_handoff_wall = now
        tr.handoff_lock_until_wall = max(float(getattr(tr, "handoff_lock_until_wall", 0.0)), now + hold_ms / 1000.0)
        tr.handoff_lock_reason = str(reason)
        tr.handoff_lock_cameras = cams
        self.stats["handoff_strict_lock_updates"] = int(self.stats.get("handoff_strict_lock_updates", 0)) + 1

    def _confirmed_track_count(self) -> int:
        return sum(
            1
            for tr in self.tracks.values()
            if tr.state != "retired" and int(getattr(tr, "hits", 0)) >= int(getattr(self.args, "publish_min_hits", 2))
        )

    def _count_gate_enabled(self) -> bool:
        return bool(getattr(self.args, "count_gate_enable", True))

    def _count_gate_zone_xy(self, x: float, y: float) -> str:
        if not self._count_gate_enabled():
            return "disabled"
        x_min = float(getattr(self.args, "count_gate_x_min_m", 0.0))
        x_max = float(getattr(self.args, "count_gate_x_max_m", 30.0))
        y_min = float(getattr(self.args, "count_gate_y_min_m", 0.0))
        y_max = float(getattr(self.args, "count_gate_y_max_m", 20.0))
        band = max(0.0, float(getattr(self.args, "count_gate_short_edge_band_m", 3.0)))
        y_pad = max(0.0, float(getattr(self.args, "count_gate_short_edge_y_pad_m", 2.0)))
        xx = float(x)
        yy = float(y)
        y_near_short_edge = (y_min - y_pad) <= yy <= (y_max + y_pad)
        if (x_min - band) <= xx <= (x_min + band) and y_near_short_edge:
            return "short_edge_left"
        if (x_max - band) <= xx <= (x_max + band) and y_near_short_edge:
            return "short_edge_right"
        if x_min <= xx <= x_max and y_min <= yy <= y_max:
            return "interior"
        if x_min <= xx <= x_max and (y_min - y_pad) <= yy <= (y_max + y_pad):
            return "long_edge"
        return "non_short_edge"

    def _count_gate_zone_for_obs(self, obs: Observation) -> str:
        return self._count_gate_zone_xy(float(obs.x), float(obs.y))

    def _count_gate_zone_for_track(self, tr: Track) -> str:
        zone = str(getattr(tr, "count_gate_zone", "") or "")
        if zone:
            return zone
        return self._count_gate_zone_xy(float(tr.x), float(tr.y))

    def _count_gate_is_short_edge_zone(self, zone: str) -> bool:
        return str(zone or "").startswith("short_edge")

    def _count_gate_bootstrap_allows_birth(self) -> bool:
        if not self._count_gate_enabled():
            return False
        bootstrap_ms = max(0.0, float(getattr(self.args, "count_gate_startup_bootstrap_ms", 5000.0)))
        if bootstrap_ms <= 0.0:
            return False
        base_wall = float(getattr(self, "first_detection_wall", 0.0) or getattr(self, "start_wall", time.time()))
        if (time.time() - base_wall) * 1000.0 > bootstrap_ms:
            return False
        max_tracks = max(0, int(getattr(self.args, "count_gate_bootstrap_allow_anywhere_until_track_count", 8)))
        return self._confirmed_track_count() < max_tracks

    def _count_gate_birth_detail(self, obs: Observation) -> Dict[str, Any]:
        zone = self._count_gate_zone_for_obs(obs)
        detail = {
            "allow": True,
            "zone": zone,
            "reason": "disabled",
        }
        if not self._count_gate_enabled():
            return detail
        if self._count_gate_is_short_edge_zone(zone):
            detail.update({"allow": True, "reason": "short_edge_entry_exit"})
            return detail
        if self._count_gate_bootstrap_allows_birth():
            detail.update({"allow": True, "reason": "startup_bootstrap"})
            return detail
        detail.update({"allow": False, "reason": "non_short_edge_birth_block"})
        return detail

    def _count_gate_birth_promotion_allowed(self, bc: BirthCandidate, obs: Observation) -> bool:
        if not self._count_gate_enabled():
            return True
        bc_zone = str(getattr(bc, "count_gate_zone", "") or "")
        bc_reason = str(getattr(bc, "count_gate_reason", "") or "")
        if self._count_gate_is_short_edge_zone(bc_zone) or bc_reason == "startup_bootstrap":
            return True
        return bool(self._count_gate_birth_detail(obs).get("allow", True))

    def _count_gate_count_birth_block(self, zone: str) -> None:
        self.stats["count_gate_birth_blocked_non_short_edge"] = int(self.stats.get("count_gate_birth_blocked_non_short_edge", 0)) + 1
        if zone == "interior":
            self.stats["count_gate_birth_blocked_interior"] = int(self.stats.get("count_gate_birth_blocked_interior", 0)) + 1
        elif zone == "long_edge":
            self.stats["count_gate_birth_blocked_long_edge"] = int(self.stats.get("count_gate_birth_blocked_long_edge", 0)) + 1

    def _mark_count_gate_lost_hold(self, tr: Track, zone: str, reason: str) -> None:
        if not self._count_gate_enabled() or self._count_gate_is_short_edge_zone(zone):
            return
        now = time.time()
        hold_ms = max(0.0, float(getattr(self.args, "count_gate_non_short_edge_lost_hold_ms", 5000.0)))
        cap_ms = max(hold_ms, float(getattr(self.args, "count_gate_non_short_edge_retire_hard_cap_ms", 15000.0)))
        if float(getattr(tr, "count_gate_hold_until_wall", 0.0)) <= now:
            self.stats["count_gate_lost_held_non_short_edge"] = int(self.stats.get("count_gate_lost_held_non_short_edge", 0)) + 1
        tr.count_gate_zone = str(zone)
        tr.count_gate_hold_reason = str(reason)
        tr.count_gate_hold_until_wall = max(float(getattr(tr, "count_gate_hold_until_wall", 0.0)), now + hold_ms / 1000.0)
        if float(getattr(tr, "count_gate_hard_retire_after_wall", 0.0)) <= 0.0:
            tr.count_gate_hard_retire_after_wall = now + cap_ms / 1000.0

    def _track_count_gate_publish_hold_active(self, tr: Track, now: Optional[float] = None) -> bool:
        if not self._count_gate_enabled():
            return False
        t = time.time() if now is None else float(now)
        return float(getattr(tr, "count_gate_hold_until_wall", 0.0)) > t

    def _track_count_gate_retire_hold_active(self, tr: Track, now: Optional[float] = None) -> bool:
        if not self._count_gate_enabled():
            return False
        t = time.time() if now is None else float(now)
        return float(getattr(tr, "count_gate_hard_retire_after_wall", 0.0)) > t

    def _count_gate_status(self) -> Dict[str, Any]:
        x_min = float(getattr(self.args, "count_gate_x_min_m", 0.0))
        x_max = float(getattr(self.args, "count_gate_x_max_m", 30.0))
        y_min = float(getattr(self.args, "count_gate_y_min_m", 0.0))
        y_max = float(getattr(self.args, "count_gate_y_max_m", 20.0))
        band = float(getattr(self.args, "count_gate_short_edge_band_m", 3.0))
        y_pad = float(getattr(self.args, "count_gate_short_edge_y_pad_m", 2.0))
        return {
            "enable": self._count_gate_enabled(),
            "rule": "only_short_edges_allow_count_change",
            "short_edges": f"x={x_min:g}/y={y_min:g}..{y_max:g}, x={x_max:g}/y={y_min:g}..{y_max:g}",
            "short_edge_zone": f"x={x_min-band:g}..{x_min+band:g} or x={x_max-band:g}..{x_max+band:g}, y={y_min-y_pad:g}..{y_max+y_pad:g}",
            "birth_allowed_short_edge": int(self.stats.get("count_gate_birth_allowed_short_edge", 0)),
            "birth_allowed_bootstrap": int(self.stats.get("count_gate_birth_allowed_bootstrap", 0)),
            "birth_blocked_non_short_edge": int(self.stats.get("count_gate_birth_blocked_non_short_edge", 0)),
            "birth_blocked_interior": int(self.stats.get("count_gate_birth_blocked_interior", 0)),
            "birth_blocked_long_edge": int(self.stats.get("count_gate_birth_blocked_long_edge", 0)),
            "birth_blocked_handoff": int(self.stats.get("count_gate_birth_blocked_handoff", 0)),
            "birth_promotion_blocked": int(self.stats.get("count_gate_birth_promotion_blocked", 0)),
            "lost_held_non_short_edge": int(self.stats.get("count_gate_lost_held_non_short_edge", 0)),
            "publish_holds": int(self.stats.get("count_gate_publish_holds", 0)),
            "retire_holds": int(self.stats.get("count_gate_retire_holds", 0)),
            "exit_allowed_short_edge": int(self.stats.get("count_gate_exit_allowed_short_edge", 0)),
            "hard_retired_after_cap": int(self.stats.get("count_gate_hard_retired_after_cap", 0)),
        }

    def _handoff_strict_birth_block_detail(self, obs: Observation) -> Dict[str, Any]:
        detail = {
            "block": False,
            "reason": "",
            "dist_m": 1e9,
            "gate_m": float(getattr(self.args, "handoff_strict_birth_block_m", 3.0)),
            "track_id": -1,
        }
        if not self._handoff_strict_enabled():
            return detail
        if not self.tracks:
            return detail
        obs_cams = {str(c) for c in getattr(obs, "cameras", []) if str(c)}
        if not obs_cams:
            return detail
        gate = max(0.0, float(getattr(self.args, "handoff_strict_birth_block_m", 3.0)))
        now = time.time()
        multi_camera_overlap = len(obs_cams) > 1
        if multi_camera_overlap and self._confirmed_track_count() > 0:
            detail.update({"block": True, "reason": "multi_camera_overlap_count_lock", "gate_m": gate})
        if gate <= 0.0:
            return detail
        for tr in self.tracks.values():
            if tr.state == "retired":
                continue
            age_ms = (now - tr.last_update_wall) * 1000.0
            if age_ms > max(float(self.args.reid_memory_ms), float(getattr(self.args, "handoff_strict_publish_hold_ms", 1800.0))):
                continue
            handoff_zone = (
                multi_camera_overlap
                or self._obs_is_strict_handoff_zone_for_track(tr, obs)
                or self._track_handoff_locked(tr, now)
            )
            if not handoff_zone:
                continue
            dist_m = self._track_birth_distance(tr, obs)
            if dist_m < float(detail["dist_m"]):
                detail.update({
                    "dist_m": float(dist_m),
                    "track_id": int(tr.gid),
                    "reason": str(detail.get("reason") or "near_handoff_track"),
                })
            if dist_m <= gate:
                detail.update({
                    "block": True,
                    "reason": "near_handoff_track",
                    "dist_m": float(dist_m),
                    "track_id": int(tr.gid),
                })
                return detail
        return detail

    def _tracks_look_like_handoff_duplicates(self, a: Track, b: Track) -> bool:
        if not self._handoff_enabled():
            return False
        cams_a = {str(c) for c in getattr(a, "cameras", []) if str(c)}
        cams_b = {str(c) for c in getattr(b, "cameras", []) if str(c)}
        if not cams_a or not cams_b:
            return False
        return not (cams_a == cams_b and cams_a)

    def _cluster_detections(self, sync_index: int, detections: List[Detection]) -> List[Observation]:
        clusters: List[List[Detection]] = []
        gate = float(self.args.cluster_gate_m)
        cross_cam_gate = max(gate, float(getattr(self.args, "cross_camera_cluster_gate_m", gate)))
        for d in sorted(detections, key=lambda x: -(float(x.confidence) * 0.35 + float(x.camera_center_weight) * 0.65)):
            best_i = -1
            best_dist = 1e9
            best_cross_cam = False
            for i, members in enumerate(clusters):
                member_weights = [max(0.01, float(getattr(m, "fusion_weight", m.confidence * m.camera_center_weight))) for m in members]
                cx = float(np.average([m.x for m in members], weights=member_weights))
                cy = float(np.average([m.y for m in members], weights=member_weights))
                dd = _dist((d.x, d.y), (cx, cy))
                if dd < best_dist:
                    best_dist, best_i = dd, i
                    best_cross_cam = d.cam not in {m.cam for m in members}
            active_gate = cross_cam_gate if best_cross_cam else gate
            if best_i >= 0 and best_dist <= active_gate:
                # One camera can produce duplicate fragments. Keep the stronger
                # one unless it is close enough to be useful evidence.
                cams = {m.cam for m in clusters[best_i]}
                if d.cam in cams and best_dist > float(self.args.same_cam_cluster_gate_m):
                    clusters.append([d])
                else:
                    if best_cross_cam and best_dist > gate:
                        self.stats["handoff_cluster_merges"] = int(self.stats.get("handoff_cluster_merges", 0)) + 1
                    clusters[best_i].append(d)
            else:
                clusters.append([d])
        obs: List[Observation] = []
        for members in clusters:
            weights = np.asarray([max(0.01, float(getattr(m, "fusion_weight", m.confidence * m.camera_center_weight))) for m in members], dtype=np.float64)
            xs = np.asarray([m.x for m in members], dtype=np.float64)
            ys = np.asarray([m.y for m in members], dtype=np.float64)
            rxs = np.asarray([m.raw_x for m in members], dtype=np.float64)
            rys = np.asarray([m.raw_y for m in members], dtype=np.float64)
            ts = int(np.median([m.timestamp_us for m in members]))
            cameras = sorted({m.cam for m in members})
            local_ids = {m.cam: int(m.local_id) for m in members if int(m.local_id) >= 0}
            conf = _clip(float(np.mean([m.confidence for m in members])) + 0.08 * (len(cameras) - 1), 0.0, 1.0)
            weight_sum = float(np.sum(weights))
            fusion_members = [
                {
                    "camera": str(m.cam),
                    "local_xy": [round(float(m.x), 4), round(float(m.y), 4)],
                    "weight": round(float(getattr(m, "fusion_weight", m.confidence * m.camera_center_weight)), 4),
                    "confidence": round(float(m.confidence), 4),
                    "camera_center_weight": round(float(m.camera_center_weight), 4),
                    "anchor_quality": round(float(getattr(m, "anchor_quality", 1.0)), 4),
                    "calibration_weight": round(float(getattr(m, "calibration_weight", 1.0)), 4),
                    "view_geometry_weight": round(float(getattr(m, "view_geometry_weight", 1.0)), 4),
                }
                for m in members
            ]
            obs.append(Observation(
                sync_index=int(sync_index),
                timestamp_us=ts,
                x=float(np.average(xs, weights=weights)),
                y=float(np.average(ys, weights=weights)),
                raw_x=float(np.average(rxs, weights=weights)),
                raw_y=float(np.average(rys, weights=weights)),
                confidence=conf,
                cameras=cameras,
                local_ids=local_ids,
                detections=list(members),
                fusion_weight_sum=weight_sum,
                fusion_members=fusion_members,
            ))
        return obs

    def _track_score(self, tr: Track, obs: Observation) -> Tuple[float, Dict[str, float]]:
        px, py = self._predict_track(tr, obs.timestamp_us, for_association=True)
        pos = _dist((px, py), (obs.x, obs.y))
        dt_s = _clip((int(obs.timestamp_us) - int(tr.last_ts_us)) / 1_000_000.0, 0.001, 1.5)
        implied_speed = pos / max(0.001, dt_s)
        if tr.hits >= 2 and implied_speed > float(self.args.max_match_speed_mps):
            return -1e9, {"pos_m": pos, "pos_score": 0.0, "local_bonus": 0.0, "implied_speed_mps": implied_speed}
        hard_gate = float(self.args.match_gate_m)
        if tr.state == "lost":
            hard_gate = max(hard_gate, float(self.args.lost_match_gate_m))
        handoff_gate = False
        if self._obs_is_camera_handoff_for_track(tr, obs):
            new_gate = max(hard_gate, float(getattr(self.args, "handoff_match_gate_m", hard_gate)))
            if new_gate > hard_gate:
                hard_gate = new_gate
                handoff_gate = True
        if pos > hard_gate:
            return -1e9, {"pos_m": pos, "pos_score": 0.0, "local_bonus": 0.0}
        pos_score = max(0.0, 1.0 - pos / max(0.001, hard_gate))
        local_bonus = 0.0
        for cam, lid in obs.local_ids.items():
            if tr.local_ids.get(cam) == lid and int(lid) >= 0:
                local_bonus += float(self.args.local_id_bonus)
        local_bonus = min(float(self.args.local_id_bonus_max), local_bonus)
        recency_ms = max(0.0, (int(obs.timestamp_us) - int(tr.last_ts_us)) / 1000.0)
        recency = max(0.0, 1.0 - recency_ms / max(1.0, float(self.args.reid_memory_ms)))
        score = 0.72 * pos_score + 0.16 * obs.confidence + 0.08 * recency + local_bonus
        return float(score), {
            "pos_m": pos,
            "pos_score": pos_score,
            "local_bonus": local_bonus,
            "recency": recency,
            "implied_speed_mps": implied_speed,
            "handoff_gate": 1.0 if handoff_gate else 0.0,
            "hard_gate_m": hard_gate,
        }

    def _assign(self, observations: List[Observation]) -> None:
        active_tracks = [
            tr for tr in self.tracks.values()
            if (time.time() - tr.last_update_wall) * 1000.0 <= float(self.args.reid_memory_ms)
        ]
        candidates: List[Tuple[float, int, int, Dict[str, float]]] = []
        for oi, obs in enumerate(observations):
            for tr in active_tracks:
                score, dbg = self._track_score(tr, obs)
                if score >= float(self.args.match_score_min):
                    candidates.append((score, oi, tr.gid, dbg))
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_obs = set()
        used_tracks = set()
        for score, oi, gid, dbg in candidates:
            if oi in used_obs or gid in used_tracks:
                continue
            self._update_track(self.tracks[gid], observations[oi], score, "match", dbg)
            if _as_float(dbg.get("handoff_gate", 0.0), 0.0) > 0.5:
                self.stats["handoff_match_gate_used"] = int(self.stats.get("handoff_match_gate_used", 0)) + 1
            used_obs.add(oi)
            used_tracks.add(gid)
            self.stats["matched_tracks"] += 1
        for oi, obs in enumerate(observations):
            if oi not in used_obs:
                if bool(getattr(self.args, "birth_quarantine_enable", True)):
                    self._handle_unmatched_observation(obs)
                else:
                    self._create_track(obs)
        self._mark_missed(set(used_tracks))
        self._retire_birth_candidates()
        self._retire_old()

    def _track_birth_distance(self, tr: Track, obs: Observation) -> float:
        px, py = self._predict_track(tr, obs.timestamp_us, for_association=True)
        return _dist((px, py), (obs.x, obs.y))

    def _birth_suppression_detail(self, obs: Observation) -> Dict[str, Any]:
        gate = max(0.0, float(getattr(self.args, "birth_near_track_suppress_m", 0.65)))
        handoff_gate = max(gate, float(getattr(self.args, "handoff_birth_near_track_suppress_m", gate)))
        best = {
            "suppress": False,
            "handoff": False,
            "dist_m": 1e9,
            "gate_m": gate,
            "track_id": -1,
        }
        if gate <= 0.0 and handoff_gate <= 0.0:
            return best
        now = time.time()
        for tr in self.tracks.values():
            if tr.state == "retired":
                continue
            age_ms = (now - tr.last_update_wall) * 1000.0
            if age_ms > float(self.args.reid_memory_ms):
                continue
            handoff = self._obs_is_camera_handoff_for_track(tr, obs)
            active_gate = handoff_gate if handoff else gate
            if active_gate <= 0.0:
                continue
            dist_m = self._track_birth_distance(tr, obs)
            if dist_m < float(best["dist_m"]):
                best.update({
                    "suppress": bool(dist_m <= active_gate),
                    "handoff": bool(handoff),
                    "dist_m": float(dist_m),
                    "gate_m": float(active_gate),
                    "track_id": int(tr.gid),
                })
        return best

    def _near_existing_track_for_birth(self, obs: Observation) -> bool:
        return bool(self._birth_suppression_detail(obs).get("suppress", False))

    def _birth_candidate_distance(self, bc: BirthCandidate, obs: Observation) -> float:
        return _dist((bc.x, bc.y), (obs.x, obs.y))

    def _update_birth_candidate(self, bc: BirthCandidate, obs: Observation) -> None:
        alpha = float(self.args.position_alpha)
        bc.x = bc.x * (1.0 - alpha) + float(obs.x) * alpha
        bc.y = bc.y * (1.0 - alpha) + float(obs.y) * alpha
        bc.raw_x = bc.raw_x * (1.0 - alpha) + float(obs.raw_x) * alpha
        bc.raw_y = bc.raw_y * (1.0 - alpha) + float(obs.raw_y) * alpha
        bc.confidence = _clip(0.75 * float(bc.confidence) + 0.25 * float(obs.confidence), 0.0, 1.0)
        before = len(bc.sync_indices)
        bc.sync_indices.add(int(obs.sync_index))
        if len(bc.sync_indices) > before:
            bc.hits += 1
        bc.last_ts_us = int(obs.timestamp_us)
        bc.last_update_wall = time.time()
        bc.cameras = sorted(set(bc.cameras) | set(obs.cameras))
        for cam, lid in obs.local_ids.items():
            bc.local_ids[cam] = int(lid)
        bc.history.append((int(obs.timestamp_us), float(bc.x), float(bc.y)))
        if len(bc.history) > int(self.args.history_len):
            bc.history = bc.history[-int(self.args.history_len):]

    def _birth_candidate_to_observation(self, bc: BirthCandidate, obs: Observation) -> Observation:
        return Observation(
            sync_index=int(obs.sync_index),
            timestamp_us=int(obs.timestamp_us),
            x=float(bc.x),
            y=float(bc.y),
            raw_x=float(bc.raw_x),
            raw_y=float(bc.raw_y),
            confidence=float(bc.confidence),
            cameras=list(bc.cameras),
            local_ids=dict(bc.local_ids),
            detections=list(obs.detections),
        )

    def _promote_birth_candidate(self, cid: int, obs: Observation) -> None:
        bc = self.birth_candidates.pop(cid, None)
        if bc is None:
            return
        promoted = self._birth_candidate_to_observation(bc, obs)
        tr = self._create_track_from_birth(promoted, hits=max(1, int(bc.hits)), first_ts_us=int(bc.first_ts_us))
        tr.history = list(bc.history[-int(self.args.history_len):]) or list(tr.history)
        self.stats["birth_candidates_promoted"] = int(self.stats.get("birth_candidates_promoted", 0)) + 1

    def _handle_unmatched_observation(self, obs: Observation) -> None:
        strict_detail = self._handoff_strict_birth_block_detail(obs)
        if bool(strict_detail.get("block", False)):
            self.stats["handoff_strict_birth_blocked"] = int(self.stats.get("handoff_strict_birth_blocked", 0)) + 1
            self.stats["handoff_birth_suppressed"] = int(self.stats.get("handoff_birth_suppressed", 0)) + 1
            if self._count_gate_enabled():
                self.stats["count_gate_birth_blocked_handoff"] = int(self.stats.get("count_gate_birth_blocked_handoff", 0)) + 1
            return
        count_detail = self._count_gate_birth_detail(obs)
        count_zone = str(count_detail.get("zone", ""))
        count_reason = str(count_detail.get("reason", ""))
        if not bool(count_detail.get("allow", True)):
            self._count_gate_count_birth_block(count_zone)
            return
        detail = self._birth_suppression_detail(obs)
        if bool(detail.get("suppress", False)):
            self.stats["birth_suppressed_near_track"] = int(self.stats.get("birth_suppressed_near_track", 0)) + 1
            if bool(detail.get("handoff", False)):
                self.stats["handoff_birth_suppressed"] = int(self.stats.get("handoff_birth_suppressed", 0)) + 1
            return
        gate = max(0.05, float(getattr(self.args, "birth_match_gate_m", 0.85)))
        best_cid = -1
        best_dist = 1e9
        for cid, bc in self.birth_candidates.items():
            dd = self._birth_candidate_distance(bc, obs)
            if dd < best_dist:
                best_dist = dd
                best_cid = cid
        if best_cid >= 0 and best_dist <= gate:
            bc = self.birth_candidates[best_cid]
            self._update_birth_candidate(bc, obs)
            self.stats["birth_candidates_matched"] = int(self.stats.get("birth_candidates_matched", 0)) + 1
            if int(bc.hits) >= int(getattr(self.args, "birth_confirm_hits", 2)):
                if self._near_existing_track_for_birth(obs):
                    self.birth_candidates.pop(best_cid, None)
                    self.stats["birth_suppressed_duplicate_candidate"] = int(self.stats.get("birth_suppressed_duplicate_candidate", 0)) + 1
                elif not self._count_gate_birth_promotion_allowed(bc, obs):
                    self.birth_candidates.pop(best_cid, None)
                    self.stats["count_gate_birth_promotion_blocked"] = int(self.stats.get("count_gate_birth_promotion_blocked", 0)) + 1
                    self._count_gate_count_birth_block(self._count_gate_zone_for_obs(obs))
                else:
                    self._promote_birth_candidate(best_cid, obs)
            return
        cid = int(self.next_birth_cid)
        self.next_birth_cid += 1
        bc = BirthCandidate(
            cid=cid,
            x=float(obs.x),
            y=float(obs.y),
            raw_x=float(obs.raw_x),
            raw_y=float(obs.raw_y),
            confidence=float(obs.confidence),
            first_ts_us=int(obs.timestamp_us),
            last_ts_us=int(obs.timestamp_us),
            last_update_wall=time.time(),
            hits=1,
            cameras=list(obs.cameras),
            local_ids=dict(obs.local_ids),
            sync_indices={int(obs.sync_index)},
            history=[(int(obs.timestamp_us), float(obs.x), float(obs.y))],
            count_gate_zone=count_zone,
            count_gate_reason=count_reason,
        )
        self.birth_candidates[cid] = bc
        self.stats["birth_candidates_created"] = int(self.stats.get("birth_candidates_created", 0)) + 1
        if self._count_gate_is_short_edge_zone(count_zone):
            self.stats["count_gate_birth_allowed_short_edge"] = int(self.stats.get("count_gate_birth_allowed_short_edge", 0)) + 1
        elif count_reason == "startup_bootstrap":
            self.stats["count_gate_birth_allowed_bootstrap"] = int(self.stats.get("count_gate_birth_allowed_bootstrap", 0)) + 1

    def _retire_birth_candidates(self) -> None:
        ttl_ms = max(1.0, float(getattr(self.args, "birth_candidate_ttl_ms", 900.0)))
        now = time.time()
        for cid in list(self.birth_candidates.keys()):
            bc = self.birth_candidates[cid]
            if (now - bc.last_update_wall) * 1000.0 > ttl_ms:
                self.birth_candidates.pop(cid, None)
                self.stats["birth_candidates_expired"] = int(self.stats.get("birth_candidates_expired", 0)) + 1

    def _create_track(self, obs: Observation) -> Track:
        return self._create_track_from_birth(obs, hits=1, first_ts_us=int(obs.timestamp_us))

    def _create_track_from_birth(self, obs: Observation, hits: int = 1, first_ts_us: Optional[int] = None) -> Track:
        gid = int(self.next_gid)
        self.next_gid += 1
        hit_count = max(1, int(hits))
        tr = Track(
            gid=gid,
            x=float(obs.x),
            y=float(obs.y),
            assoc_x=float(obs.x),
            assoc_y=float(obs.y),
            confidence=float(obs.confidence),
            state="tracking" if hit_count >= int(self.args.confirm_hits) else "tentative",
            hits=hit_count,
            misses=0,
            created_us=int(first_ts_us if first_ts_us is not None else obs.timestamp_us),
            last_ts_us=int(obs.timestamp_us),
            last_sync_index=int(obs.sync_index),
            last_update_wall=time.time(),
            cameras=list(obs.cameras),
            local_ids=dict(obs.local_ids),
            history=[(int(obs.timestamp_us), float(obs.x), float(obs.y))],
            measurement_x=float(obs.x),
            measurement_y=float(obs.y),
            measurement_raw_x=float(obs.raw_x),
            measurement_raw_y=float(obs.raw_y),
            filter_backend=str(self.filter_backend),
            fusion_weight_sum=float(getattr(obs, "fusion_weight_sum", 0.0)),
            fusion_members=list(getattr(obs, "fusion_members", [])),
            count_gate_zone=self._count_gate_zone_for_obs(obs),
        )
        tr.raw_x = float(obs.raw_x)
        tr.raw_y = float(obs.raw_y)
        if str(self.filter_backend) == "kalman_cv":
            self._ensure_kalman_state(tr)
        self.tracks[gid] = tr
        self.stats["created_tracks"] += 1
        return tr

    def _update_track(self, tr: Track, obs: Observation, score: float, source: str, dbg: Dict[str, float]) -> None:
        dt = _clip((int(obs.timestamp_us) - int(tr.last_ts_us)) / 1_000_000.0, 0.001, 1.5)
        pred_x, pred_y = self._predict_track(tr, obs.timestamp_us, for_association=True)
        strict_handoff_update = self._obs_is_strict_handoff_zone_for_track(tr, obs)
        new_vx = (float(obs.x) - float(pred_x)) / dt
        new_vy = (float(obs.y) - float(pred_y)) / dt
        instant_speed = float(math.hypot(new_vx, new_vy))
        if tr.hits >= 2 and instant_speed > float(self.args.max_update_speed_mps):
            self.stats["observations_rejected_speed"] += 1
            tr.misses += 1
            tr.last_gate_reason = "speed_reject"
            tr.last_match = {
                "score": float(score),
                "source": "speed_reject",
                "debug": dict(dbg, instant_speed_mps=instant_speed),
            }
            return
        tr.measurement_x = float(obs.x)
        tr.measurement_y = float(obs.y)
        tr.measurement_raw_x = float(obs.raw_x)
        tr.measurement_raw_y = float(obs.raw_y)
        tr.filter_backend = str(self.filter_backend)
        tr.fusion_weight_sum = float(getattr(obs, "fusion_weight_sum", 0.0))
        tr.fusion_members = list(getattr(obs, "fusion_members", []))
        filter_dbg: Dict[str, float] = {"pred_x": float(pred_x), "pred_y": float(pred_y)}
        alpha = float(self.args.position_alpha)
        beta = float(self.args.velocity_alpha)
        base_ax = float(tr.assoc_x if tr.assoc_x or tr.assoc_y else tr.x)
        base_ay = float(tr.assoc_y if tr.assoc_x or tr.assoc_y else tr.y)
        tr.assoc_x = base_ax * (1.0 - alpha) + float(obs.x) * alpha
        tr.assoc_y = base_ay * (1.0 - alpha) + float(obs.y) * alpha
        tr.assoc_vx = float(tr.assoc_vx) * (1.0 - beta) + new_vx * beta
        tr.assoc_vy = float(tr.assoc_vy) * (1.0 - beta) + new_vy * beta
        tr.raw_x = tr.raw_x * (1.0 - alpha) + float(obs.raw_x) * alpha
        tr.raw_y = tr.raw_y * (1.0 - alpha) + float(obs.raw_y) * alpha
        if str(self.filter_backend) == "kalman_cv":
            innovation = _dist((pred_x, pred_y), (obs.x, obs.y))
            tr.last_innovation_m = float(innovation)
            if (
                tr.hits >= int(getattr(self.args, "confirm_hits", 2))
                and bool(getattr(self.args, "kalman_reject_innovation_enable", False))
                and innovation > float(getattr(self.args, "kalman_max_innovation_m", 2.5))
            ):
                self.stats["observations_rejected_speed"] += 1
                tr.misses += 1
                tr.last_gate_reason = "kalman_innovation_reject"
                tr.last_match = {
                    "score": float(score),
                    "source": "kalman_innovation_reject",
                    "debug": dict(dbg, instant_speed_mps=instant_speed, kalman_innovation_m=innovation),
                }
                return
            filter_dbg.update(self._kalman_update_track(tr, obs))
        else:
            tr.x = float(tr.assoc_x)
            tr.y = float(tr.assoc_y)
            tr.vx = float(tr.assoc_vx)
            tr.vy = float(tr.assoc_vy)
            tr.last_innovation_m = _dist((pred_x, pred_y), (obs.x, obs.y))
            tr.last_gate_reason = "alpha_beta_update"
        tr.confidence = _clip(0.75 * tr.confidence + 0.25 * float(obs.confidence), 0.0, 1.0)
        tr.count_gate_zone = self._count_gate_zone_for_obs(obs)
        tr.hits += 1
        tr.misses = 0
        tr.last_ts_us = int(obs.timestamp_us)
        tr.last_sync_index = int(obs.sync_index)
        tr.last_update_wall = time.time()
        tr.cameras = list(obs.cameras)
        for cam, lid in obs.local_ids.items():
            old = tr.local_ids.get(cam)
            if old is not None and old != lid and _dist((tr.x, tr.y), (obs.x, obs.y)) > float(self.args.id_switch_suspect_gate_m):
                self.stats["id_switch_suspects"] += 1
            tr.local_ids[cam] = int(lid)
        tr.history.append((int(obs.timestamp_us), float(tr.x), float(tr.y)))
        if len(tr.history) > int(self.args.history_len):
            tr.history = tr.history[-int(self.args.history_len):]
        if tr.hits >= int(self.args.confirm_hits):
            tr.state = "tracking"
        tr.last_match = {"score": float(score), "source": source, "debug": dict(dbg, instant_speed_mps=instant_speed, **filter_dbg)}
        if strict_handoff_update or _as_float(dbg.get("handoff_gate", 0.0), 0.0) > 0.5:
            self._mark_handoff_lock(tr, obs, "camera_handoff_update")

    def _mark_missed(self, updated_track_ids: set) -> None:
        now = time.time()
        for gid, tr in self.tracks.items():
            if gid in updated_track_ids:
                continue
            age_ms = (now - tr.last_update_wall) * 1000.0
            if age_ms > float(self.args.miss_grace_ms):
                tr.misses += 1
                if age_ms > float(self.args.lost_after_ms) and tr.state != "lost":
                    zone = self._count_gate_zone_for_track(tr)
                    if self._count_gate_enabled() and self._count_gate_is_short_edge_zone(zone):
                        self.stats["count_gate_exit_allowed_short_edge"] = int(self.stats.get("count_gate_exit_allowed_short_edge", 0)) + 1
                    elif self._count_gate_enabled():
                        self._mark_count_gate_lost_hold(tr, zone, "lost_non_short_edge")
                    tr.state = "lost"

    def _retire_old(self) -> None:
        now = time.time()
        retire_ms = float(self.args.retire_after_ms)
        for gid in list(self.tracks.keys()):
            tr = self.tracks[gid]
            if (now - tr.last_update_wall) * 1000.0 > retire_ms:
                if self._track_handoff_locked(tr, now):
                    self.stats["handoff_strict_retire_holds"] = int(self.stats.get("handoff_strict_retire_holds", 0)) + 1
                    continue
                if self._track_count_gate_retire_hold_active(tr, now):
                    self.stats["count_gate_retire_holds"] = int(self.stats.get("count_gate_retire_holds", 0)) + 1
                    continue
                if float(getattr(tr, "count_gate_hard_retire_after_wall", 0.0)) > 0.0:
                    self.stats["count_gate_hard_retired_after_cap"] = int(self.stats.get("count_gate_hard_retired_after_cap", 0)) + 1
                self.tracks.pop(gid, None)
                self.stats["retired_tracks"] += 1

    def _publish(self, sync_index: int) -> None:
        max_lost_age = float(self.args.publish_lost_max_age_ms)
        now = time.time()
        candidates = []
        for tr in sorted(self.tracks.values(), key=lambda t: t.gid):
            if tr.state == "retired":
                continue
            if tr.state == "lost" and max_lost_age >= 0 and (now - tr.last_update_wall) * 1000.0 > max_lost_age:
                if self._track_handoff_locked(tr, now):
                    self.stats["handoff_strict_publish_holds"] = int(self.stats.get("handoff_strict_publish_holds", 0)) + 1
                elif self._track_count_gate_publish_hold_active(tr, now):
                    self.stats["count_gate_publish_holds"] = int(self.stats.get("count_gate_publish_holds", 0)) + 1
                else:
                    continue
            if tr.hits < int(self.args.publish_min_hits):
                continue
            candidates.append(tr)
        tracks = [self._track_json(tr) for tr in self._suppress_duplicate_publish_tracks(candidates)]
        obj = {
            "schema_version": 1,
            "msg_type": "global_person_tracks",
            "stream_role": str(getattr(self.args, "stream_role", "official")),
            "filter_backend": str(self.filter_backend),
            "timestamp_us": _now_us(),
            "sync_index": int(sync_index),
            "track_count": len(tracks),
            "tracks": tracks,
            "coord_transform": self._coord_status(),
            "runtime_config": self._runtime_config(),
            "count_gate": self._count_gate_status(),
            "control": dict(self.control_status),
            "stats": dict(self.stats),
        }
        self.latest = obj
        self.out_fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.out_fp.flush()
        self._atomic_write_json(self.latest_json_path, obj)
        self._last_publish_t = time.time()

    def _suppress_duplicate_publish_tracks(self, tracks: List[Track]) -> List[Track]:
        if not bool(getattr(self.args, "duplicate_suppress_enable", True)):
            return list(tracks)
        gate = max(0.0, float(getattr(self.args, "duplicate_suppress_dist_m", 0.9)))
        handoff_gate = max(gate, float(getattr(self.args, "handoff_duplicate_suppress_dist_m", gate)))
        if max(gate, handoff_gate) <= 0.0 or len(tracks) <= 1:
            return list(tracks)
        ranked = sorted(
            tracks,
            key=lambda tr: (
                1 if str(tr.state) == "tracking" else 0,
                int(tr.hits),
                -float((time.time() - tr.last_update_wall) * 1000.0),
                float(tr.confidence),
            ),
            reverse=True,
        )
        kept: List[Track] = []
        suppressed = 0
        for tr in ranked:
            duplicate = False
            for old in kept:
                active_gate = handoff_gate if self._tracks_look_like_handoff_duplicates(tr, old) else gate
                if _dist((tr.x, tr.y), (old.x, old.y)) <= active_gate:
                    duplicate = True
                    if active_gate > gate:
                        self.stats["handoff_duplicate_suppressed"] = int(self.stats.get("handoff_duplicate_suppressed", 0)) + 1
                    break
            if duplicate:
                suppressed += 1
                continue
            kept.append(tr)
        self.stats["publish_duplicate_suppressed"] = int(suppressed)
        return sorted(kept, key=lambda t: t.gid)

    def _track_json(self, tr: Track) -> Dict[str, Any]:
        speed = float(math.hypot(tr.vx, tr.vy))
        try:
            pred_x, pred_y = self._predict_track(tr, _now_us())
        except Exception:
            pred_x, pred_y = tr.x, tr.y
        try:
            assoc_pred_x, assoc_pred_y = self._predict_track(tr, _now_us(), for_association=True)
        except Exception:
            assoc_pred_x, assoc_pred_y = tr.assoc_x, tr.assoc_y
        cov_trace = 0.0
        if tr.kalman_cov is not None:
            try:
                cov_trace = float(np.trace(tr.kalman_cov))
            except Exception:
                cov_trace = 0.0
        return {
            "global_player_id": f"P{int(tr.gid):02d}",
            "global_id": int(tr.gid),
            "state": str(tr.state),
            "confidence": round(float(tr.confidence), 4),
            "x_m": round(float(tr.x), 4),
            "y_m": round(float(tr.y), 4),
            "field_xy": [round(float(tr.x), 4), round(float(tr.y), 4)],
            "raw_xy": [round(float(tr.raw_x), 4), round(float(tr.raw_y), 4)],
            "measurement_xy": [round(float(tr.measurement_x), 4), round(float(tr.measurement_y), 4)],
            "measurement_raw_xy": [round(float(tr.measurement_raw_x), 4), round(float(tr.measurement_raw_y), 4)],
            "filtered_xy": [round(float(tr.x), 4), round(float(tr.y), 4)],
            "association_xy": [round(float(tr.assoc_x), 4), round(float(tr.assoc_y), 4)],
            "predicted_xy": [round(float(pred_x), 4), round(float(pred_y), 4)],
            "predicted_association_xy": [round(float(assoc_pred_x), 4), round(float(assoc_pred_y), 4)],
            "coord_transform_mode": str(self.coord_mode),
            "filter_backend": str(getattr(tr, "filter_backend", self.filter_backend)),
            "kalman_cov_trace": round(float(cov_trace), 6),
            "innovation_m": round(float(getattr(tr, "last_innovation_m", 0.0)), 4),
            "gate_reason": str(getattr(tr, "last_gate_reason", "")),
            "vx_mps": round(float(tr.vx), 4),
            "vy_mps": round(float(tr.vy), 4),
            "speed_mps": round(speed, 4),
            "last_ts_us": int(tr.last_ts_us),
            "last_sync_index": int(tr.last_sync_index),
            "age_ms": round((time.time() - tr.last_update_wall) * 1000.0, 2),
            "handoff_lock_active": bool(self._track_handoff_locked(tr)),
            "handoff_lock_remaining_ms": round(max(0.0, float(getattr(tr, "handoff_lock_until_wall", 0.0)) - time.time()) * 1000.0, 2),
            "handoff_lock_reason": str(getattr(tr, "handoff_lock_reason", "")),
            "handoff_lock_cameras": list(getattr(tr, "handoff_lock_cameras", [])),
            "count_gate_zone": str(self._count_gate_zone_for_track(tr)),
            "count_gate_hold_active": bool(self._track_count_gate_publish_hold_active(tr)),
            "count_gate_hold_remaining_ms": round(max(0.0, float(getattr(tr, "count_gate_hold_until_wall", 0.0)) - time.time()) * 1000.0, 2),
            "count_gate_hard_retire_remaining_ms": round(max(0.0, float(getattr(tr, "count_gate_hard_retire_after_wall", 0.0)) - time.time()) * 1000.0, 2),
            "count_gate_hold_reason": str(getattr(tr, "count_gate_hold_reason", "")),
            "hits": int(tr.hits),
            "misses": int(tr.misses),
            "source_cameras": list(tr.cameras),
            "local_ids": dict(tr.local_ids),
            "fusion_weight_sum": round(float(getattr(tr, "fusion_weight_sum", 0.0)), 4),
            "fusion_members": list(getattr(tr, "fusion_members", [])),
            "history": [[int(ts), round(float(x), 3), round(float(y), 3)] for ts, x, y in tr.history[-int(self.args.output_history_len):]],
            "last_match": getattr(tr, "last_match", {}),
        }

    def _coord_status(self) -> Dict[str, Any]:
        return {
            "mode": str(self.coord_mode),
            "field_origin_m": [float(self.args.field_origin_x_m), float(self.args.field_origin_y_m)],
            "field_size_m": [float(self.args.field_width_m), float(self.args.field_height_m)],
            "field_margin_m": float(self.args.field_margin_m),
            "drop_out_of_field": bool(self.args.drop_out_of_field),
            "max_match_speed_mps": float(self.args.max_match_speed_mps),
            "max_update_speed_mps": float(self.args.max_update_speed_mps),
            "local_id_bonus": float(self.args.local_id_bonus),
            "local_id_bonus_max": float(self.args.local_id_bonus_max),
            "position_alpha": float(self.args.position_alpha),
            "velocity_alpha": float(self.args.velocity_alpha),
            "filter_backend": str(self.filter_backend),
        }

    def _atomic_write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _process_records(self) -> int:
        count = 0
        for cam, tailer in self.tailers.items():
            for rec in tailer.poll(max_lines=int(self.args.max_lines_per_poll)):
                self.stats["records"] += 1
                si = _sync_index(rec)
                if si < 0:
                    ts_us = _as_int(rec.get("timestamp_us", rec.get("capture_wall_us", rec.get("record_wall_us", 0))), 0)
                    si = _as_int(ts_us // 40000, -1) if ts_us > 0 else -1
                if si >= 0:
                    self.latest_input_sync_index = max(int(self.latest_input_sync_index), int(si))
                dets = self._record_to_detections(cam, rec)
                if not dets:
                    if si >= 0 and self.tracks:
                        self.pending.setdefault(int(si), [])
                        count += 1
                    continue
                self.stats["detections"] += len(dets)
                si = dets[0].sync_index
                if si < 0:
                    si = _as_int(dets[0].timestamp_us // 40000, 0)
                self.pending.setdefault(int(si), []).extend(dets)
                count += 1
        return count

    def _flush_ready_syncs(self) -> None:
        if not self.pending:
            return
        max_sync = max(self.pending.keys())
        ready_before = max_sync - int(self.args.sync_delay)
        for si in sorted([x for x in self.pending.keys() if x <= ready_before])[:int(self.args.max_sync_flush_per_loop)]:
            dets = self.pending.pop(si, [])
            obs = self._cluster_detections(si, dets)
            self.stats["observations"] += len(obs)
            self._assign(obs)
            self._publish(si)

    def _write_status(self) -> None:
        status = {
            "schema_version": 1,
            "state": "running",
            "stream_role": str(getattr(self.args, "stream_role", "official")),
            "filter_backend": str(self.filter_backend),
            "timestamp_us": _now_us(),
            "cams": self.cams,
            "tailers": {
                cam: {
                    "path": str(t.path),
                    "error": t.last_error,
                    "bad_lines": int(getattr(t, "bad_lines", 0)),
                    "last_good_us": int(getattr(t, "last_good_us", 0)),
                    "last_bad_us": int(getattr(t, "last_bad_us", 0)),
                }
                for cam, t in self.tailers.items()
            },
            "pending_syncs": len(self.pending),
            "track_count": len(self.tracks),
            "birth_candidate_count": len(self.birth_candidates),
            "birth_candidates": [
                {
                    "cid": int(bc.cid),
                    "hits": int(bc.hits),
                    "age_ms": round((time.time() - bc.last_update_wall) * 1000.0, 2),
                    "first_ts_us": int(bc.first_ts_us),
                    "last_ts_us": int(bc.last_ts_us),
                    "xy": [round(float(bc.x), 4), round(float(bc.y), 4)],
                    "cameras": list(bc.cameras),
                    "local_ids": dict(bc.local_ids),
                }
                for bc in sorted(self.birth_candidates.values(), key=lambda x: x.cid)
            ],
            "tracks": [self._track_json(tr) for tr in sorted(self.tracks.values(), key=lambda t: t.gid)],
            "coord_transform": self._coord_status(),
            "runtime_config": self._runtime_config(),
            "count_gate": self._count_gate_status(),
            "control": dict(self.control_status),
            "stats": dict(self.stats),
            "latest_json": str(self.latest_json_path),
            "output_jsonl": str(self.out_jsonl_path),
        }
        self.status = status
        self._atomic_write_json(self.status_json_path, status)

    def run(self) -> int:
        print(
            f"[global_person_id] start cams={','.join(self.cams)} root={self.sync_root} "
            f"out={self.latest_json_path} jsonl={self.out_jsonl_path}",
            flush=True,
        )
        try:
            self._publish(-1)
            while True:
                self._poll_control()
                self._process_records()
                self._flush_ready_syncs()
                now = time.time()
                if now - self._last_status_t >= max(0.2, float(self.args.status_interval_ms) / 1000.0):
                    self._write_status()
                    self._last_status_t = now
                time.sleep(max(0.001, float(self.args.poll_ms) / 1000.0))
        finally:
            try:
                self.out_fp.close()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Global person ID server based on YOLO physical coordinates")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--human-jsonl-template", default="{root}/human_result_{camera}.jsonl")
    ap.add_argument("--output-jsonl", default="global_person_tracks.jsonl")
    ap.add_argument("--latest-json", default="global_person_latest.json")
    ap.add_argument("--status-json", default="global_person_status.json")
    ap.add_argument("--control-json", default="global_person_id_control.json")
    ap.add_argument("--control-poll-ms", type=float, default=200.0)
    ap.add_argument("--stream-role", choices=["official", "shadow"], default="official")
    ap.add_argument("--append-jsonl", action="store_true")
    ap.add_argument("--start-tail", action="store_true", default=True)
    ap.add_argument("--poll-ms", type=float, default=10.0)
    ap.add_argument("--status-interval-ms", type=float, default=1000.0)
    ap.add_argument("--max-lines-per-poll", type=int, default=256)
    ap.add_argument("--max-sync-flush-per-loop", type=int, default=16)
    ap.add_argument("--sync-delay", type=int, default=1)
    ap.add_argument("--min-confidence", type=float, default=0.10)
    ap.add_argument("--coord-transform-mode", choices=["raw", "neg_y", "field_height_plus_y", "invert_y", "flip_x", "flip_xy", "swap_xy", "swap_xy_neg_y"], default="raw")
    ap.add_argument("--field-width-m", type=float, default=33.0)
    ap.add_argument("--field-height-m", type=float, default=22.0)
    ap.add_argument("--field-origin-x-m", type=float, default=0.0)
    ap.add_argument("--field-origin-y-m", type=float, default=0.0)
    ap.add_argument("--field-margin-m", type=float, default=2.0)
    ap.add_argument("--drop-out-of-field", action="store_true")
    ap.add_argument("--max-match-speed-mps", type=float, default=12.0)
    ap.add_argument("--max-update-speed-mps", type=float, default=10.0)
    ap.add_argument("--publish-lost-max-age-ms", type=float, default=500.0)
    ap.add_argument("--publish-min-hits", type=int, default=2)
    ap.add_argument("--cluster-gate-m", type=float, default=0.65)
    ap.add_argument("--same-cam-cluster-gate-m", type=float, default=0.22)
    ap.add_argument("--cross-camera-cluster-gate-m", type=float, default=0.75)
    ap.add_argument("--handoff-aware-enable", action="store_true", default=True)
    ap.add_argument("--no-handoff-aware-enable", dest="handoff_aware_enable", action="store_false")
    ap.add_argument("--handoff-match-gate-m", type=float, default=1.35)
    ap.add_argument("--handoff-birth-near-track-suppress-m", type=float, default=1.25)
    ap.add_argument("--handoff-duplicate-suppress-dist-m", type=float, default=1.35)
    ap.add_argument("--handoff-strict-count-lock-enable", action="store_true", default=True)
    ap.add_argument("--no-handoff-strict-count-lock-enable", dest="handoff_strict_count_lock_enable", action="store_false")
    ap.add_argument("--handoff-strict-birth-block-m", type=float, default=3.0)
    ap.add_argument("--handoff-strict-publish-hold-ms", type=float, default=1800.0)
    ap.add_argument("--count-gate-enable", action="store_true", default=True)
    ap.add_argument("--no-count-gate-enable", dest="count_gate_enable", action="store_false")
    ap.add_argument("--count-gate-x-min-m", type=float, default=0.0)
    ap.add_argument("--count-gate-x-max-m", type=float, default=30.0)
    ap.add_argument("--count-gate-y-min-m", type=float, default=0.0)
    ap.add_argument("--count-gate-y-max-m", type=float, default=20.0)
    ap.add_argument("--count-gate-short-edge-band-m", type=float, default=3.0)
    ap.add_argument("--count-gate-short-edge-y-pad-m", type=float, default=2.0)
    ap.add_argument("--count-gate-startup-bootstrap-ms", type=float, default=5000.0)
    ap.add_argument("--count-gate-bootstrap-allow-anywhere-until-track-count", type=int, default=8)
    ap.add_argument("--count-gate-non-short-edge-lost-hold-ms", type=float, default=5000.0)
    ap.add_argument("--count-gate-non-short-edge-retire-hard-cap-ms", type=float, default=15000.0)
    ap.add_argument("--duplicate-suppress-enable", action="store_true", default=True)
    ap.add_argument("--no-duplicate-suppress-enable", dest="duplicate_suppress_enable", action="store_false")
    ap.add_argument("--duplicate-suppress-dist-m", type=float, default=0.9)
    ap.add_argument("--match-gate-m", type=float, default=1.25)
    ap.add_argument("--lost-match-gate-m", type=float, default=1.8)
    ap.add_argument("--match-score-min", type=float, default=0.32)
    ap.add_argument("--birth-quarantine-enable", action="store_true", default=True)
    ap.add_argument("--no-birth-quarantine-enable", dest="birth_quarantine_enable", action="store_false")
    ap.add_argument("--birth-confirm-hits", type=int, default=2)
    ap.add_argument("--birth-match-gate-m", type=float, default=0.85)
    ap.add_argument("--birth-near-track-suppress-m", type=float, default=0.65)
    ap.add_argument("--birth-candidate-ttl-ms", type=float, default=900.0)
    ap.add_argument("--filter-backend", choices=["alpha_beta", "kalman_cv"], default="alpha_beta")
    ap.add_argument("--position-alpha", type=float, default=0.65)
    ap.add_argument("--velocity-alpha", type=float, default=0.25)
    ap.add_argument("--kalman-use-for-association", action="store_true", default=False)
    ap.add_argument("--no-kalman-use-for-association", dest="kalman_use_for_association", action="store_false")
    ap.add_argument("--kalman-process-noise-m", type=float, default=0.35)
    ap.add_argument("--kalman-measurement-noise-m", type=float, default=0.45)
    ap.add_argument("--kalman-initial-pos-var", type=float, default=0.6)
    ap.add_argument("--kalman-initial-vel-var", type=float, default=4.0)
    ap.add_argument("--kalman-max-innovation-m", type=float, default=2.5)
    ap.add_argument("--kalman-reject-innovation-enable", action="store_true", default=False)
    ap.add_argument("--no-kalman-reject-innovation-enable", dest="kalman_reject_innovation_enable", action="store_false")
    ap.add_argument("--confirm-hits", type=int, default=2)
    ap.add_argument("--miss-grace-ms", type=float, default=180.0)
    ap.add_argument("--lost-after-ms", type=float, default=650.0)
    ap.add_argument("--reid-memory-ms", type=float, default=3000.0)
    ap.add_argument("--retire-after-ms", type=float, default=12000.0)
    ap.add_argument("--local-id-bonus", type=float, default=0.10)
    ap.add_argument("--local-id-bonus-max", type=float, default=0.18)
    ap.add_argument("--id-switch-suspect-gate-m", type=float, default=0.80)
    ap.add_argument("--history-len", type=int, default=64)
    ap.add_argument("--output-history-len", type=int, default=16)
    args = ap.parse_args()
    args.sync_root = str(Path(args.sync_root).expanduser().resolve())
    return GlobalPersonIdServer(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
