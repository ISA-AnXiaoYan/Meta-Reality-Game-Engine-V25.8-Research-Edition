#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global YOLO inference server for the 8-camera split-MVS pipeline.

This process implements the first C+3 path:
  - read all MVS latest-frame shared buffers directly;
  - run one model instance with global micro-batches;
  - keep a fast, tensor-first postprocess path by default;
  - publish legacy-compatible human_result_{camera}.jsonl rows.

The launcher can run it in shadow mode first, writing to a separate directory,
or in primary mode where it replaces the four YOLO shard workers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from global_yolo_polygon_contract import person_has_usable_polygon, summarize_polygon_contract


def now_us() -> int:
    return int(time.time() * 1_000_000)


MICRO_PROFILE_MS_FIELDS = [
    "frame_prepare_ms",
    "predict_ms",
    "boxes_cpu_ms",
    "det_filter_ms",
    "mask_polygon_ms",
    "extract_dets_ms",
    "anchor_ms",
    "project_ms",
    "record_build_ms",
    "postprocess_ms",
    "legacy_jsonl_write_ms",
    "latest_json_write_ms",
    "status_json_write_ms",
    "batch_total_observed_ms",
    "async_enqueue_ms",
    "async_queue_lag_ms",
    "async_backpressure_ms",
    "input_align_visual_mid_spread_ms",
    "input_align_capture_ts_spread_ms",
]

MICRO_PROFILE_COUNT_FIELDS = [
    "raw_boxes_count",
    "mask_polygon_requested",
    "mask_polygon_found",
    "person_count_total",
    "records_count",
]


def _perf_now() -> float:
    return time.perf_counter()


def _elapsed_ms(t0: float) -> float:
    return max(0.0, (time.perf_counter() - t0) * 1000.0)


def _profile_add(profile: Optional[Dict[str, Any]], key: str, value: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0) or 0.0) + float(value)


def _profile_inc(profile: Optional[Dict[str, Any]], key: str, value: int = 1) -> None:
    if profile is None:
        return
    profile[key] = int(profile.get(key, 0) or 0) + int(value)


def _profile_round(profile: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in profile.items():
        if key.endswith("_ms"):
            out[key] = round(float(value or 0.0), 3)
        else:
            try:
                out[key] = int(value)
            except Exception:
                out[key] = value
    return out


def _parse_cams(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").replace(";", ",").split(",") if x.strip()]


def _parse_groups(value: str, cams: Sequence[str]) -> List[List[str]]:
    valid = {str(c) for c in cams}
    raw = str(value or "").strip()
    if not raw:
        return [list(cams)]
    groups: List[List[str]] = []
    for part in raw.split(":"):
        group = [x.strip() for x in part.split(",") if x.strip()]
        group = [x for x in group if x in valid]
        if group:
            groups.append(group)
    used = {c for g in groups for c in g}
    missing = [c for c in cams if c not in used]
    if missing:
        groups.append(missing)
    return groups or [list(cams)]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_project_path(project_root: Path, raw: str) -> Path:
    p = Path(str(raw or "")).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p


def _get_nested(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    obj = cfg.get(section)
    if isinstance(obj, dict) and key in obj:
        return obj.get(key)
    if key in cfg:
        return cfg.get(key)
    return default


def _load_mvs_config(path: Path) -> Dict[str, Any]:
    cfg = _read_json(path)
    if isinstance(cfg.get("mvs_config"), dict):
        return dict(cfg.get("mvs_config") or {})
    return cfg


def _nonempty_str(value: Any) -> str:
    s = str(value or "").strip()
    return s


def _resolve_optional_path(project_root: Path, config_path: Path, raw: Any) -> Tuple[str, bool]:
    text = _nonempty_str(raw)
    if not text:
        return "", False
    p = Path(text).expanduser()
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([project_root / p, config_path.parent / p, p])
    for cand in candidates:
        try:
            if cand.exists():
                return str(cand.resolve()), True
        except Exception:
            continue
    try:
        return str(candidates[0].resolve()), False
    except Exception:
        return text, False


def _find_model_candidate(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Find the first YOLO model/engine path in runtime configs.

    The unified launcher may write several shapes depending on which stage
    produced the runtime JSON: top-level model.model, common.model.model,
    mvs_config.model.model, or route-local copies.  The global worker should
    not depend on one exact shape.
    """
    preferred: List[Tuple[str, Any]] = []
    fallback: List[Tuple[str, Any]] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key in ("engine", "engine_path", "trt_engine"):
                if _nonempty_str(obj.get(key)):
                    preferred.append((f"{path}.{key}" if path else key, obj.get(key)))
            for key in ("model", "weights", "weight", "model_path"):
                value = obj.get(key)
                if isinstance(value, dict):
                    visit(value, f"{path}.{key}" if path else key)
                elif _nonempty_str(value):
                    entry = (f"{path}.{key}" if path else key, value)
                    text = _nonempty_str(value).lower()
                    if text.endswith((".pt", ".engine", ".onnx")):
                        preferred.append(entry)
                    else:
                        fallback.append(entry)
            for key, value in obj.items():
                if key in {"engine", "engine_path", "trt_engine", "model", "weights", "weight", "model_path"}:
                    continue
                if isinstance(value, (dict, list, tuple)):
                    visit(value, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                if isinstance(value, (dict, list, tuple)):
                    visit(value, f"{path}[{idx}]")

    visit(cfg, "")
    candidates = preferred or fallback
    for source, value in candidates:
        text = _nonempty_str(value)
        if text:
            return text, source
    return "", ""


class JsonlWriter:
    def __init__(self, path: Path, truncate: bool):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w" if truncate else "a", encoding="utf-8", buffering=1)

    def write(self, obj: Dict[str, Any]) -> None:
        self.fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass


class PerfCsvWriter:
    fieldnames = [
        "wall_us",
        "batch_index",
        "batch_size",
        "group_size",
        "initial_batch_size",
        "collect_wait_ms",
        "collect_added_frames",
        "predict_ms",
        "postprocess_ms",
        "total_ms",
        "predict_fps_total",
        "total_fps_total",
        "backend",
        "model",
        "engine",
        "device",
        "imgsz",
        "half",
        "retina_masks",
        "cam_ids",
        "sync_indices",
        "frame_ids_local",
        "person_counts",
        "raw_boxes_count",
        "mask_polygon_requested",
        "mask_polygon_found",
        "person_count_total",
        "records_count",
        "frame_prepare_ms",
        "boxes_cpu_ms",
        "det_filter_ms",
        "mask_polygon_ms",
        "extract_dets_ms",
        "anchor_ms",
        "project_ms",
        "record_build_ms",
        "legacy_jsonl_write_ms",
        "latest_json_write_ms",
        "status_json_write_ms",
        "batch_total_observed_ms",
        "async_enqueue_ms",
        "async_queue_lag_ms",
        "async_backpressure_ms",
        "input_align_enabled",
        "input_align_full_batch",
        "input_align_same_sync",
        "input_align_sync_spread",
        "input_align_visual_mid_complete",
        "input_align_visual_mid_aligned",
        "input_align_meta_match",
        "input_align_current_aligned_full",
        "input_align_would_full_seen",
        "input_align_would_full_within_wait",
        "input_align_would_wait_ms",
        "input_align_visual_mid_spread_ms",
        "input_align_capture_ts_spread_ms",
        "input_align_exposure_spread_us",
        "input_align_sync_indices",
        "input_align_cam_ids",
        "input_align_frame_ids",
        "input_align_exposure_us",
        "input_align_exposure_source",
        "input_align_visual_mid_us",
        "input_align_exposure_end_us",
        "input_align_exposure_end_spread_ms",
        "input_align_missing_cams",
        "input_align_late_cam",
        "input_buffer_enabled",
        "input_buffer_decision",
        "input_buffer_fallback_reason",
        "input_buffer_wait_ms",
        "input_buffer_pending_frames",
    ]

    def __init__(self, path: Path, truncate: bool = True, flush_interval_ms: float = 1000.0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w" if truncate else "a", encoding="utf-8", newline="", buffering=1024 * 1024)
        self.writer = csv.DictWriter(self.fp, fieldnames=self.fieldnames)
        self.flush_interval_s = max(0.0, float(flush_interval_ms or 0.0)) / 1000.0
        self.last_flush_s = time.time()
        if truncate or self.path.stat().st_size == 0:
            self.writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        now = time.time()
        if self.flush_interval_s <= 0.0 or now - self.last_flush_s >= self.flush_interval_s:
            self.fp.flush()
            self.last_flush_s = now

    def close(self) -> None:
        try:
            self.fp.flush()
            self.fp.close()
        except Exception:
            pass


class ExposureAuditCsvWriter:
    fieldnames = [
        "wall_us",
        "batch_index",
        "group_cams",
        "group_size",
        "batch_size",
        "full_batch",
        "current_aligned_full",
        "same_sync",
        "visual_mid_aligned",
        "sync_indices",
        "cam_ids",
        "frame_ids_local",
        "missing_cams",
        "late_cam",
        "exposure_us",
        "exposure_source",
        "exposure_start_us",
        "visual_mid_us",
        "exposure_end_us",
        "exposure_spread_us",
        "visual_mid_spread_ms",
        "exposure_end_spread_ms",
        "capture_ts_spread_ms",
        "meta_frame_match",
        "meta_frame_mismatch",
        "would_full_seen",
        "would_full_within_wait",
        "would_wait_ms",
        "would_visual_mid_spread_ms",
        "input_buffer_enabled",
        "input_buffer_decision",
        "input_buffer_fallback_reason",
        "input_buffer_wait_ms",
        "input_buffer_pending_frames",
    ]

    def __init__(self, path: Path, truncate: bool = True, flush_interval_ms: float = 1000.0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w" if truncate else "a", encoding="utf-8", newline="", buffering=1024 * 1024)
        self.writer = csv.DictWriter(self.fp, fieldnames=self.fieldnames)
        self.flush_interval_s = max(0.0, float(flush_interval_ms or 0.0)) / 1000.0
        self.last_flush_s = time.time()
        if truncate or self.path.stat().st_size == 0:
            self.writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self.writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        now = time.time()
        if self.flush_interval_s <= 0.0 or now - self.last_flush_s >= self.flush_interval_s:
            self.fp.flush()
            self.last_flush_s = now

    def close(self) -> None:
        try:
            self.fp.flush()
            self.fp.close()
        except Exception:
            pass


class LocalProjector:
    def __init__(self, camera: str, path: Path):
        self.camera = str(camera)
        self.path = path
        self.enabled = False
        self.last_error = ""
        self.coord_frame = "local_ground"
        self.unit = "m"
        self.projector_type = "ground_plane_homography"
        self.H = None
        self.image_points: List[List[float]] = []
        self.local_points: List[List[float]] = []
        self._load()

    def _load(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            self.last_error = f"missing_cv2_or_numpy:{type(exc).__name__}:{exc}"
            return
        try:
            data = _read_json(self.path)
        except Exception as exc:
            self.last_error = f"load_failed:{type(exc).__name__}:{exc}"
            return
        self.image_points = data.get("image_points", []) or []
        self.local_points = data.get("field_points", None) or data.get("local_points", []) or []
        self.coord_frame = str(data.get("coord_frame", self.coord_frame) or self.coord_frame)
        self.unit = str(data.get("unit", self.unit) or self.unit)
        self.projector_type = str(data.get("projector_type", self.projector_type) or self.projector_type)
        if len(self.image_points) < 4 or len(self.local_points) < 4:
            self.last_error = "need_at_least_4_point_pairs"
            return
        try:
            src = np.asarray(self.image_points, dtype=np.float32)
            dst = np.asarray(self.local_points, dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(src[:4], dst[:4]) if len(src) == 4 else cv2.findHomography(src, dst, cv2.RANSAC, 3.0)[0]
            if self.H is None:
                self.last_error = "find_homography_failed"
                return
            self.enabled = True
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"build_failed:{type(exc).__name__}:{exc}"

    def project_detail(self, pt: Tuple[float, float]) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "ok": False,
            "x_m": None,
            "y_m": None,
            "z_m": 0.0,
            "unit": self.unit,
            "coord_frame": self.coord_frame,
            "source": "GlobalYoloLocalProjector",
            "projector_type": self.projector_type,
            "foot_pixel": [float(pt[0]), float(pt[1])] if pt is not None else None,
            "projector_valid": bool(self.enabled and self.H is not None),
            "reason": self.last_error,
        }
        if not self.enabled or self.H is None or pt is None:
            return base
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            src = np.asarray([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.H)[0, 0]
            base.update({"ok": True, "x_m": float(dst[0]), "y_m": float(dst[1]), "reason": ""})
        except Exception as exc:
            base["reason"] = f"project_failed:{type(exc).__name__}:{exc}"
        return base

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "valid": bool(self.enabled and self.H is not None),
            "camera": self.camera,
            "config_source": str(self.path),
            "coord_frame": self.coord_frame,
            "unit": self.unit,
            "projector_type": self.projector_type,
            "image_points_count": len(self.image_points),
            "local_points_count": len(self.local_points),
            "last_error": self.last_error,
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _point_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _polygon_area_centroid(poly: Optional[List[List[int]]]) -> Tuple[float, Tuple[float, float]]:
    if not poly or len(poly) < 3:
        return 0.0, (0.0, 0.0)
    pts = [(float(p[0]), float(p[1])) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    if len(pts) < 3:
        return 0.0, (0.0, 0.0)
    area2 = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross
    if abs(area2) < 1e-6:
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        mid = len(pts) // 2
        return 0.0, (float(xs[mid]), float(ys[mid]))
    area = area2 * 0.5
    cx = cx_acc / (3.0 * area2)
    cy = cy_acc / (3.0 * area2)
    return abs(float(area)), (float(cx), float(cy))


def _mask_eroded_centroid(poly: Optional[List[List[int]]], bbox: Sequence[float]) -> Optional[Tuple[float, float]]:
    area, centroid = _polygon_area_centroid(poly)
    if area <= 0.0:
        return None
    bx, by = _bbox_footpoint(bbox, "bbox_center")
    # A light erosion approximation: pull the polygon centroid toward the bbox
    # center to reduce ragged mask boundary influence without rasterizing masks.
    return (float(centroid[0]) * 0.82 + bx * 0.18, float(centroid[1]) * 0.82 + by * 0.18)


def _load_nadir_config(camera: str, path: Path) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "camera": str(camera),
        "path": str(path),
        "exists": False,
        "source": "image_center_fallback",
        "nadir_pixel": None,
        "camera_height_m": None,
        "effective_human_height_m": None,
        "k": None,
        "last_error": "",
    }
    try:
        if not path.exists():
            return base
        obj = _read_json(path)
        nadir = obj.get("nadir_pixel")
        if isinstance(nadir, (list, tuple)) and len(nadir) >= 2:
            base.update({
                "exists": True,
                "source": "file",
                "nadir_pixel": [float(nadir[0]), float(nadir[1])],
                "camera_height_m": obj.get("camera_height_m"),
                "effective_human_height_m": obj.get("effective_human_height_m"),
                "k": obj.get("k"),
            })
            return base
        base.update({"exists": True, "last_error": "missing_or_invalid_nadir_pixel"})
    except Exception as exc:
        base["last_error"] = f"{type(exc).__name__}: {exc}"
    return base


def _nadir_detail(nadir_cfg: Dict[str, Any], w: int, h: int, args: argparse.Namespace) -> Dict[str, Any]:
    raw = nadir_cfg.get("nadir_pixel")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        nadir = (float(raw[0]), float(raw[1]))
        source = str(nadir_cfg.get("source") or "file")
    else:
        nadir = (float(max(0, w - 1)) * 0.5, float(max(0, h - 1)) * 0.5)
        source = "image_center_fallback"
    k = nadir_cfg.get("k")
    if k is None:
        k = getattr(args, "anchor_height_correction_k", 0.08)
    return {
        "nadir_pixel": [float(nadir[0]), float(nadir[1])],
        "anchor_nadir_source": source,
        "anchor_height_correction_k": _safe_float(k, float(getattr(args, "anchor_height_correction_k", 0.08))),
    }


class MvsLatestReader:
    def __init__(self, camera: str, shm_template: str, meta_template: str, sync_root: Path):
        self.camera = str(camera)
        self.shm_name = str(shm_template).format(camera=self.camera, cam=self.camera, id=self.camera)
        meta_raw = str(meta_template).format(root=str(sync_root), camera=self.camera, cam=self.camera, id=self.camera)
        self.meta_path = Path(meta_raw)
        if not self.meta_path.is_absolute():
            self.meta_path = sync_root / self.meta_path
        self.reader = None
        self.last_error = ""

    def _ensure_reader(self) -> bool:
        if self.reader is not None:
            return True
        try:
            from latest_frame_shm import SharedFrameReader  # type: ignore

            self.reader = SharedFrameReader(self.shm_name)
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def read(self) -> Optional[Dict[str, Any]]:
        if not self._ensure_reader():
            return None
        try:
            item = self.reader.read_latest()
        except Exception as exc:
            self.last_error = f"read_failed:{type(exc).__name__}: {exc}"
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
            return None
        if not item:
            return None
        meta: Dict[str, Any] = {}
        try:
            if self.meta_path.exists():
                meta = _read_json(self.meta_path)
        except Exception as exc:
            meta = {"meta_read_error": f"{type(exc).__name__}: {exc}"}
        return {
            "camera": self.camera,
            "frame": item.get("frame"),
            "frame_id": int(item.get("frame_id", 0) or 0),
            "timestamp_us": int(item.get("timestamp_us", 0) or 0),
            "mvs_meta": meta,
            "mvs_frame_num": int(meta.get("mvs_frame_num", item.get("frame_id", 0)) or 0),
            "sync_index": int(meta.get("program_sync_index", meta.get("sync_index", -1)) if meta else -1),
        }

    def close(self) -> None:
        try:
            if self.reader is not None:
                self.reader.close()
        except Exception:
            pass


@dataclass
class Det:
    confidence: float
    cls: int
    bbox_xyxy: Tuple[float, float, float, float]
    foot_pixel: Tuple[float, float]
    area: int
    mask_polygon: Optional[List[List[int]]] = None
    anchor_pixel_raw: Tuple[float, float] = (0.0, 0.0)
    anchor_pixel_corrected: Tuple[float, float] = (0.0, 0.0)
    anchor_mode: str = "bbox_center"
    anchor_quality: float = 0.0
    anchor_quality_terms: Dict[str, float] = None  # type: ignore[assignment]
    anchor_nadir_source: str = ""
    anchor_height_correction_k: float = 0.0
    calibration_weight: float = 1.0
    view_geometry_weight: float = 1.0
    fusion_weight: float = 1.0


@dataclass
class PostprocessTask:
    batch_index: int
    group_size: int
    items: List[Dict[str, Any]]
    results: List[Any]
    cam_ids: List[str]
    frame_shapes: List[Sequence[int]]
    collect_meta: Dict[str, Any]
    predict_ms: float
    batch_t0_s: float
    observed_t0: float
    micro_profile: Dict[str, Any]
    enqueue_wall_us: int = 0
    backpressure_ms: float = 0.0


def _bbox_footpoint(bbox: Sequence[float], mode: str) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    mode = str(mode or "center").strip().lower()
    if mode in {"center", "bbox_center", "topdown_center"}:
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    if mode in {"bottom", "bbox_bottom"}:
        return ((x1 + x2) * 0.5, y2)
    if mode in {"top", "bbox_top"}:
        return ((x1 + x2) * 0.5, y1)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _edge_quality(bbox: Sequence[float], w: int, h: int) -> float:
    if w <= 0 or h <= 0:
        return 1.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    margin = min(x1, y1, max(0.0, float(w) - x2), max(0.0, float(h) - y2))
    norm = margin / max(1.0, min(float(w), float(h)) * 0.08)
    return _clamp(0.45 + 0.55 * min(1.0, norm), 0.45, 1.0)


def _mask_quality(mask_polygon: Optional[List[List[int]]], bbox: Sequence[float]) -> float:
    if not mask_polygon:
        return 0.65
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
    area, _centroid = _polygon_area_centroid(mask_polygon)
    if area <= 1.0:
        return 0.5
    ratio = _clamp(area / bbox_area, 0.0, 1.5)
    if ratio < 0.12:
        return 0.55
    if ratio < 0.28:
        return 0.75
    return 1.0


def _select_anchor_raw(
    bbox: Sequence[float],
    mask_polygon: Optional[List[List[int]]],
    mode: str,
) -> Tuple[Tuple[float, float], str, float]:
    requested = str(mode or "auto_topdown_center").strip().lower()
    if requested in {"center", "bbox_center"}:
        return _bbox_footpoint(bbox, "bbox_center"), "bbox_center", 0.65
    if requested in {"bottom", "bbox_bottom"}:
        return _bbox_footpoint(bbox, "bbox_bottom"), "bbox_bottom", 0.55
    if requested in {"mask_eroded_centroid", "eroded_centroid"}:
        eroded = _mask_eroded_centroid(mask_polygon, bbox)
        if eroded is not None:
            return eroded, "mask_eroded_centroid", _mask_quality(mask_polygon, bbox)
        return _bbox_footpoint(bbox, "bbox_center"), "bbox_center_fallback", 0.65
    if requested in {"mask_robust_center", "mask_center"}:
        area, centroid = _polygon_area_centroid(mask_polygon)
        if area > 1.0:
            return centroid, "mask_robust_center", _mask_quality(mask_polygon, bbox)
        return _bbox_footpoint(bbox, "bbox_center"), "bbox_center_fallback", 0.65
    if requested in {"auto", "auto_topdown_center"}:
        area, centroid = _polygon_area_centroid(mask_polygon)
        if area > 1.0:
            return centroid, "mask_robust_center", _mask_quality(mask_polygon, bbox)
        return _bbox_footpoint(bbox, "bbox_center"), "bbox_center_fallback", 0.65
    return _bbox_footpoint(bbox, "bbox_center"), "bbox_center", 0.65


def _apply_anchor_to_det(det: Det, w: int, h: int, args: argparse.Namespace,
                         nadir_cfg: Dict[str, Any]) -> Det:
    raw, actual_mode, mask_q = _select_anchor_raw(det.bbox_xyxy, det.mask_polygon, str(getattr(args, "anchor_mode", "auto_topdown_center")))
    nadir = _nadir_detail(nadir_cfg, w, h, args)
    nx, ny = nadir["nadir_pixel"]
    k = float(nadir["anchor_height_correction_k"])
    if not bool(getattr(args, "anchor_height_correction_enable", True)):
        k = 0.0
    corrected = (float(raw[0]) + k * (float(nx) - float(raw[0])), float(raw[1]) + k * (float(ny) - float(raw[1])))
    center_w = _camera_center_weight(det.bbox_xyxy, w, h)
    edge_q = _edge_quality(det.bbox_xyxy, w, h)
    anchor_q = _clamp(float(det.confidence) * center_w * mask_q * edge_q, 0.0, 1.0)
    max_radius = max(1.0, math.hypot(max(1, w), max(1, h)) * 0.5)
    image_center = (float(max(0, w - 1)) * 0.5, float(max(0, h - 1)) * 0.5)
    view_w = _clamp(1.0 - _point_dist(corrected, (float(nx), float(ny))) / max_radius, 0.45, 1.0)
    calib_w = _clamp(1.0 - _point_dist(corrected, image_center) / max_radius, 0.35, 1.0)
    fusion_w = _clamp(float(det.confidence) * center_w * anchor_q * calib_w * view_w, 0.01, 1.0)
    det.anchor_pixel_raw = (float(raw[0]), float(raw[1]))
    det.anchor_pixel_corrected = (float(corrected[0]), float(corrected[1]))
    det.foot_pixel = det.anchor_pixel_corrected
    det.anchor_mode = actual_mode
    det.anchor_quality = anchor_q
    det.anchor_quality_terms = {
        "confidence": round(float(det.confidence), 4),
        "camera_center_weight": round(float(center_w), 4),
        "mask_quality": round(float(mask_q), 4),
        "edge_quality": round(float(edge_q), 4),
    }
    det.anchor_nadir_source = str(nadir["anchor_nadir_source"])
    det.anchor_height_correction_k = float(k)
    det.view_geometry_weight = float(view_w)
    det.calibration_weight = float(calib_w)
    det.fusion_weight = float(fusion_w)
    return det


def _simplify_polygon(poly: Any, w: int, h: int, max_points: int) -> Optional[List[List[int]]]:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(poly, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    n = int(arr.shape[0])
    max_points = max(3, int(max_points or 96))
    if n > max_points:
        idx = [min(n - 1, int(round(i * (n - 1) / max(1, max_points - 1)))) for i in range(max_points)]
        arr = arr[idx, :]
    points: List[List[int]] = []
    max_x = max(0, int(w) - 1)
    max_y = max(0, int(h) - 1)
    last: Optional[Tuple[int, int]] = None
    for row in arr:
        x = int(round(float(row[0])))
        y = int(round(float(row[1])))
        if w > 0:
            x = max(0, min(max_x, x))
        if h > 0:
            y = max(0, min(max_y, y))
        pt = (x, y)
        if pt == last:
            continue
        points.append([x, y])
        last = pt
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    return points if len(points) >= 3 else None


def _mask_polygon_from_result(result: Any, index: int, w: int, h: int, max_points: int) -> Optional[List[List[int]]]:
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    for attr in ("xy", "xyn"):
        try:
            polys = getattr(masks, attr, None)
        except Exception:
            polys = None
        if polys is None:
            continue
        try:
            if index >= len(polys):
                continue
            poly = polys[index]
        except Exception:
            continue
        if attr == "xyn":
            try:
                import numpy as np  # type: ignore

                arr = np.asarray(poly, dtype=float).copy()
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    arr[:, 0] *= float(w)
                    arr[:, 1] *= float(h)
                    poly = arr
            except Exception:
                continue
        out = _simplify_polygon(poly, w, h, max_points)
        if out:
            return out
    return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _safe_int_or_none(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return int(float(v))
    except Exception:
        return None


def _ratio(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den or 0)
        if d <= 0.0:
            return float(default)
        return round(float(num or 0) / d, 4)
    except Exception:
        return float(default)


def _spread(values: Sequence[int]) -> Optional[int]:
    vals = [int(v) for v in values]
    if not vals:
        return None
    return int(max(vals) - min(vals))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    idx = int(round((len(vals) - 1) * max(0.0, min(1.0, float(q)))))
    return vals[max(0, min(len(vals) - 1, idx))]


def _pipe(values: Sequence[Any]) -> str:
    return "|".join("" if v is None else str(v) for v in values)


def _extract_dets_fast(
    result: Any,
    args: argparse.Namespace,
    frame_shape: Sequence[int],
    profile: Optional[Dict[str, Any]] = None,
) -> List[Det]:
    if result is None or getattr(result, "boxes", None) is None:
        return []
    boxes_t0 = _perf_now()
    try:
        boxes_obj = result.boxes
        xyxy = boxes_obj.xyxy.detach().cpu().numpy()
        conf = boxes_obj.conf.detach().cpu().numpy()
        cls = boxes_obj.cls.detach().cpu().numpy()
    except Exception:
        import numpy as np  # type: ignore

        boxes_obj = result.boxes
        xyxy = np.asarray(boxes_obj.xyxy)
        conf = np.asarray(boxes_obj.conf)
        cls = np.asarray(boxes_obj.cls)
    _profile_add(profile, "boxes_cpu_ms", _elapsed_ms(boxes_t0))
    h = int(frame_shape[0]) if frame_shape else 0
    w = int(frame_shape[1]) if len(frame_shape) >= 2 else 0
    out: List[Det] = []
    n = min(len(xyxy), len(conf), len(cls))
    _profile_inc(profile, "raw_boxes_count", int(n))
    for i in range(n):
        filter_t0 = _perf_now()
        c = int(cls[i])
        cf = _safe_float(conf[i], 0.0)
        if c != int(args.person_class_id) or cf < float(args.conf):
            _profile_add(profile, "det_filter_ms", _elapsed_ms(filter_t0))
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[i][:4]]
        x1 = max(0.0, min(float(max(0, w - 1)), x1)) if w > 0 else x1
        x2 = max(0.0, min(float(max(0, w - 1)), x2)) if w > 0 else x2
        y1 = max(0.0, min(float(max(0, h - 1)), y1)) if h > 0 else y1
        y2 = max(0.0, min(float(max(0, h - 1)), y2)) if h > 0 else y2
        if x2 <= x1 or y2 <= y1:
            _profile_add(profile, "det_filter_ms", _elapsed_ms(filter_t0))
            continue
        foot = _bbox_footpoint((x1, y1, x2, y2), args.footpoint_mode)
        mask_polygon = None
        _profile_add(profile, "det_filter_ms", _elapsed_ms(filter_t0))
        if bool(getattr(args, "include_mask_polygons", False)):
            _profile_inc(profile, "mask_polygon_requested", 1)
            mask_t0 = _perf_now()
            mask_polygon = _mask_polygon_from_result(
                result,
                i,
                w,
                h,
                int(getattr(args, "mask_polygon_max_points", 96) or 96),
            )
            _profile_add(profile, "mask_polygon_ms", _elapsed_ms(mask_t0))
            if mask_polygon:
                _profile_inc(profile, "mask_polygon_found", 1)
        out.append(Det(
            confidence=cf,
            cls=c,
            bbox_xyxy=(x1, y1, x2, y2),
            foot_pixel=foot,
            area=int(max(1.0, (x2 - x1) * (y2 - y1))),
            mask_polygon=mask_polygon,
        ))
    return out[: int(args.max_det)]


def _record_for_cam(
    *,
    cam: str,
    frame_item: Dict[str, Any],
    frame_shape: Sequence[int],
    dets: List[Det],
    projector: LocalProjector,
    nadir_config: Dict[str, Any],
    args: argparse.Namespace,
    infer_meta: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    h = int(frame_shape[0]) if frame_shape else 0
    w = int(frame_shape[1]) if len(frame_shape) >= 2 else 0
    mvs_meta = frame_item.get("mvs_meta") if isinstance(frame_item.get("mvs_meta"), dict) else {}
    fid = int(frame_item.get("frame_id", 0) or 0)
    mvs_frame_num = int(frame_item.get("mvs_frame_num", fid) or fid)
    sync_index = frame_item.get("sync_index", None)
    if sync_index is None or int(sync_index) < 0:
        sync_index = int(mvs_frame_num) - 1 if mvs_frame_num > 0 else int(fid)
    persons: List[Dict[str, Any]] = []
    for idx, d in enumerate(dets, start=1):
        anchor_t0 = _perf_now()
        d = _apply_anchor_to_det(d, w, h, args, nadir_config)
        _profile_add(profile, "anchor_ms", _elapsed_ms(anchor_t0))
        project_t0 = _perf_now()
        phys = projector.project_detail(d.foot_pixel)
        _profile_add(profile, "project_ms", _elapsed_ms(project_t0))
        local_xy = None
        if bool(phys.get("ok", False)):
            local_xy = [round(float(phys["x_m"]), 3), round(float(phys["y_m"]), 3)]
        person = {
            "person_id": int(idx),
            "stable_id": int(idx),
            "local_stable_id": int(idx),
            "global_id_from_A": None,
            "reid_score": 0.0,
            "reid_margin": 0.0,
            "reid_source_anchor": "global_yolo_fast",
            "confidence": round(float(d.confidence), 4),
            "bbox_xyxy": [int(round(v)) for v in d.bbox_xyxy],
            "mask_area": int(d.area),
            "anchor_pixel_raw": [round(float(d.anchor_pixel_raw[0]), 3), round(float(d.anchor_pixel_raw[1]), 3)],
            "anchor_pixel_corrected": [round(float(d.anchor_pixel_corrected[0]), 3), round(float(d.anchor_pixel_corrected[1]), 3)],
            "anchor_mode": str(d.anchor_mode),
            "anchor_quality": round(float(d.anchor_quality), 4),
            "anchor_quality_terms": dict(d.anchor_quality_terms or {}),
            "anchor_height_correction_k": round(float(d.anchor_height_correction_k), 5),
            "anchor_nadir_source": str(d.anchor_nadir_source),
            "foot_pixel": [int(round(d.foot_pixel[0])), int(round(d.foot_pixel[1]))],
            "local_xy": local_xy,
            "physical_coord": phys,
            "visibility": str(args.footpoint_mode),
            "camera_center_weight": round(_camera_center_weight(d.bbox_xyxy, w, h), 4),
            "calibration_weight": round(float(d.calibration_weight), 4),
            "view_geometry_weight": round(float(d.view_geometry_weight), 4),
            "fusion_weight": round(float(d.fusion_weight), 4),
        }
        if bool(args.include_mask_polygons) and d.mask_polygon:
            person["mask_polygons"] = [d.mask_polygon]
        persons.append(person)
    exposure_us = mvs_meta.get("exposure_time_us") if isinstance(mvs_meta, dict) else None
    try:
        exposure_us = int(float(exposure_us)) if exposure_us is not None else None
    except Exception:
        exposure_us = None
    rec = {
        "msg_type": "person_regions",
        "camera": str(cam),
        "camera_name": str(cam),
        "mvs_serial": "",
        "event_camera": "",
        "event_serial": "",
        "frame_id_local": int(fid),
        "mvs_frame_num": int(mvs_frame_num),
        "sync_index_hint": int(sync_index),
        "sync_index_offset_used": 0,
        "capture_wall_us": int(frame_item.get("timestamp_us", 0) or 0),
        "record_wall_us": now_us(),
        "exposure_us": exposure_us,
        "rgb_exposure_time_us": exposure_us,
        "rgb_exposure_source": str(mvs_meta.get("exposure_source", "") or ("unknown" if exposure_us is None else "mvs_meta")) if isinstance(mvs_meta, dict) else "unknown",
        "rgb_exposure_as_sync_anchor": False,
        "rgb_exposure_start_offset_us": mvs_meta.get("rgb_exposure_start_offset_us") if isinstance(mvs_meta, dict) else None,
        "rgb_exposure_mid_offset_us": mvs_meta.get("rgb_exposure_mid_offset_us") if isinstance(mvs_meta, dict) else None,
        "rgb_exposure_end_offset_us": mvs_meta.get("rgb_exposure_end_offset_us") if isinstance(mvs_meta, dict) else None,
        "frame_w": int(w),
        "frame_h": int(h),
        "mvs_meta": mvs_meta,
        "projector": projector.status(),
        "anchor": {
            "mode": str(getattr(args, "anchor_mode", "auto_topdown_center")),
            "height_correction_enable": bool(getattr(args, "anchor_height_correction_enable", True)),
            "height_correction_k": float(getattr(args, "anchor_height_correction_k", 0.08)),
            "nadir_source": str(nadir_config.get("source", "image_center_fallback")),
            "nadir_path": str(nadir_config.get("path", "")),
            "nadir_valid": bool(nadir_config.get("nadir_pixel") is not None),
        },
        "persons": persons,
        "person_count": int(len(persons)),
        "valid": True,
        "infer_source": "global_yolo",
        "postprocess_mode": str(args.postprocess_backend),
        "global_yolo_worker": True,
        "yolo_batch_index": infer_meta.get("batch_index"),
        "yolo_batch_size": infer_meta.get("batch_size"),
        "yolo_predict_ms": infer_meta.get("predict_ms"),
        "yolo_postprocess_batch_ms": infer_meta.get("postprocess_ms"),
        "yolo_total_batch_ms": infer_meta.get("total_ms"),
        "yolo_imgsz": int(args.imgsz),
        "yolo_half": bool(args.half),
        "yolo_retina_masks": bool(args.retina_masks),
        "yolo_max_det": int(args.max_det),
        "yolo_device": str(args.device),
    }
    return rec


def _camera_center_weight(bbox: Sequence[float], w: int, h: int) -> float:
    if w <= 0 or h <= 0:
        return 1.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx = (x1 + x2) * 0.5 / max(1.0, float(w))
    cy = (y1 + y2) * 0.5 / max(1.0, float(h))
    dx = abs(cx - 0.5) * 2.0
    dy = abs(cy - 0.5) * 2.0
    return max(0.05, 1.0 - min(1.0, math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)))


class GlobalYoloServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.sync_root = _resolve_project_path(self.project_root, args.sync_root).resolve()
        self.cams = _parse_cams(args.cams)
        self.groups = _parse_groups(args.microbatch_groups, self.cams)
        self.config_path = _resolve_project_path(self.project_root, args.config)
        self.config = _load_mvs_config(self.config_path)
        self.model_cfg = self.config.get("model", {}) if isinstance(self.config.get("model"), dict) else {}
        self.sync_cfg = self.config.get("sync", {}) if isinstance(self.config.get("sync"), dict) else {}
        discovered_model, discovered_source = _find_model_candidate(self.config)
        self.model_source = "cli.engine" if _nonempty_str(args.engine) else ("cli.model" if _nonempty_str(args.model) else discovered_source)
        raw_engine = _nonempty_str(args.engine)
        raw_model = _nonempty_str(args.model) or discovered_model
        self.engine_path, self.engine_exists = _resolve_optional_path(self.project_root, self.config_path, raw_engine)
        self.model_path, self.model_exists = _resolve_optional_path(self.project_root, self.config_path, raw_model)
        if self.engine_path:
            self.model_load_path = self.engine_path
            self.model_load_exists = self.engine_exists
        else:
            self.model_load_path = self.model_path
            self.model_load_exists = self.model_exists
        self.readers = {
            cam: MvsLatestReader(
                cam,
                args.mvs_shm_template,
                args.mvs_meta_template,
                self.sync_root,
            )
            for cam in self.cams
        }
        self.projectors = {
            cam: LocalProjector(cam, _resolve_project_path(self.project_root, args.calib_template.format(camera=cam, cam=cam, id=cam)))
            for cam in self.cams
        }
        self.nadir_configs = {
            cam: _load_nadir_config(
                cam,
                _resolve_project_path(self.project_root, str(args.anchor_nadir_pixel_template).format(camera=cam, cam=cam, id=cam)),
            )
            for cam in self.cams
        }
        self.last_seen: Dict[str, int] = {}
        self.writers: Dict[str, JsonlWriter] = {}
        self.shadow = bool(args.shadow_mode)
        self.output_root = self._output_root()
        self.status_path = _resolve_project_path(self.project_root, args.status_json)
        self.latest_json_path = _resolve_project_path(self.project_root, args.latest_json)
        self.status_interval_s = max(0.0, float(args.status_interval_ms or 0.0)) / 1000.0
        self.latest_interval_s = max(0.0, float(args.latest_interval_ms or 0.0)) / 1000.0
        self.batch_collect_wait_s = max(0.0, float(args.batch_collect_wait_ms or 0.0)) / 1000.0
        self.batch_collect_retry_s = max(0.00025, float(args.batch_collect_retry_sleep_ms or 0.0) / 1000.0)
        self._last_status_write_s = 0.0
        self._last_latest_write_s = 0.0
        self.perf_writer = PerfCsvWriter(
            _resolve_project_path(self.project_root, args.perf_csv),
            truncate=True,
            flush_interval_ms=float(args.perf_flush_interval_ms),
        )
        self.exposure_audit_writer = ExposureAuditCsvWriter(
            _resolve_project_path(self.project_root, args.exposure_audit_csv),
            truncate=True,
            flush_interval_ms=float(args.perf_flush_interval_ms),
        )
        self.micro_profile_window: Deque[Dict[str, Any]] = deque(
            maxlen=max(1, int(getattr(args, "micro_profile_window", 300) or 300))
        )
        self.input_alignment_window: Deque[Dict[str, Any]] = deque(
            maxlen=max(16, int(getattr(args, "micro_profile_window", 300) or 300))
        )
        self.exposure_windows_by_cam: Dict[str, Deque[int]] = {
            cam: deque(maxlen=max(16, int(getattr(args, "input_alignment_audit_history", 512) or 512)))
            for cam in self.cams
        }
        self.exposure_source_by_cam: Dict[str, str] = {}
        self.last_micro_profile_ms: Dict[str, Any] = {}
        self.model = None
        self.batch_index = 0
        self._state_lock = threading.RLock()
        self.async_record_postprocess = bool(getattr(args, "async_record_postprocess", False))
        self.postprocess_queue: Optional[queue.Queue[Optional[PostprocessTask]]] = None
        self.postprocess_thread: Optional[threading.Thread] = None
        self._postprocess_worker_error = ""
        self.input_alignment_audit_enabled = bool(getattr(args, "input_alignment_audit", False))
        self.input_alignment_audit_max_wait_ms = max(0.0, float(getattr(args, "input_alignment_audit_max_wait_ms", 8.0) or 0.0))
        self.input_alignment_audit_max_spread_ms = max(0.0, float(getattr(args, "input_alignment_audit_max_spread_ms", 6.0) or 0.0))
        self.input_alignment_audit_history = max(16, int(getattr(args, "input_alignment_audit_history", 512) or 512))
        self._input_alignment_seen_by_sync: Dict[str, Dict[str, Any]] = {}
        self.last_input_alignment_audit: Dict[str, Any] = {}
        self.input_buffer_enabled = bool(getattr(args, "input_buffer_enable", False))
        self.input_buffer_size = max(1, int(getattr(args, "input_buffer_size", 12) or 12))
        self.input_buffer_max_wait_ms = max(0.0, float(getattr(args, "input_buffer_max_wait_ms", 30.0) or 0.0))
        self.input_buffer_max_spread_ms = max(0.0, float(getattr(args, "input_buffer_max_spread_ms", self.input_alignment_audit_max_spread_ms) or self.input_alignment_audit_max_spread_ms))
        self.input_buffer_exposure_end_barrier = bool(getattr(args, "input_buffer_exposure_end_barrier", True))
        self.input_buffers: Dict[str, Deque[Dict[str, Any]]] = {
            cam: deque(maxlen=self.input_buffer_size)
            for cam in self.cams
        }
        self.last_emitted: Dict[str, int] = {}
        self.last_emitted_sync: Dict[str, int] = {}
        self._input_buffer_seq = 0
        self.stats: Dict[str, Any] = {
            "frames_read": 0,
            "frames_inferred": 0,
            "frames_published": 0,
            "records_written": 0,
            "errors": 0,
            "status_writes": 0,
            "latest_writes": 0,
            "empty_polls": 0,
            "full_batches": 0,
            "partial_batches": 0,
            "collect_wait_batches": 0,
            "collect_wait_added_frames": 0,
            "collect_wait_ms_total": 0.0,
            "start_wall_us": now_us(),
            "async_record_postprocess_enabled": bool(self.async_record_postprocess),
            "postprocess_batches_enqueued": 0,
            "postprocess_batches_published": 0,
            "postprocess_queue_depth": 0,
            "postprocess_queue_max_depth": 0,
            "postprocess_backpressure_ms_total": 0.0,
            "postprocess_queue_full_waits": 0,
            "postprocess_dropped_batches": 0,
            "postprocess_errors": 0,
            "postprocess_lag_ms_last": 0.0,
            "postprocess_lag_ms_max": 0.0,
            "published_records_total": 0,
            "published_records_with_persons": 0,
            "published_records_with_polygon": 0,
            "published_records_with_polygon_ratio": 1.0,
            "published_persons_total": 0,
            "published_persons_with_polygon": 0,
            "published_persons_without_polygon": 0,
            "published_persons_with_polygon_ratio": 1.0,
            "input_alignment_audit_enabled": bool(self.input_alignment_audit_enabled),
            "input_alignment_batches": 0,
            "input_alignment_frames": 0,
            "input_alignment_frames_with_sync": 0,
            "input_alignment_frames_with_visual_mid": 0,
            "input_alignment_full_batches": 0,
            "input_alignment_partial_batches": 0,
            "input_alignment_same_sync_batches": 0,
            "input_alignment_same_sync_full_batches": 0,
            "input_alignment_visual_mid_complete_batches": 0,
            "input_alignment_visual_mid_aligned_batches": 0,
            "input_alignment_current_aligned_full_batches": 0,
            "input_alignment_meta_frame_match_batches": 0,
            "input_alignment_meta_frame_mismatch_batches": 0,
            "input_alignment_sync_spread_max": 0,
            "input_alignment_visual_mid_spread_ms_max": 0.0,
            "input_alignment_capture_ts_spread_ms_max": 0.0,
            "input_alignment_exposure_spread_us_max": 0,
            "input_alignment_sync_full_sets_seen": 0,
            "input_alignment_sync_full_sets_within_wait": 0,
            "input_alignment_sync_full_sets_visual_aligned": 0,
            "input_alignment_sync_full_wait_ms_total": 0.0,
            "input_alignment_sync_full_wait_ms_max": 0.0,
            "input_alignment_sync_full_visual_spread_ms_max": 0.0,
            "input_buffer_enabled": bool(self.input_buffer_enabled),
            "input_buffer_size": int(self.input_buffer_size),
            "input_buffer_frames_buffered": 0,
            "input_buffer_batches": 0,
            "input_buffer_aligned_full_batches": 0,
            "input_buffer_fallback_batches": 0,
            "input_buffer_sync_partial_batches": 0,
            "input_buffer_latest_partial_batches": 0,
            "input_buffer_visual_misaligned_full_batches": 0,
            "input_buffer_wait_ms_total": 0.0,
            "input_buffer_wait_ms_max": 0.0,
            "input_buffer_pending_frames": 0,
            "input_buffer_pruned_frames": 0,
        }
        if self.async_record_postprocess:
            queue_size = max(1, int(getattr(args, "async_record_postprocess_queue_size", 2) or 2))
            self.postprocess_queue = queue.Queue(maxsize=queue_size)
            self.postprocess_thread = threading.Thread(
                target=self._postprocess_worker_loop,
                name="global-yolo-record-postprocess",
                daemon=True,
            )
            self.postprocess_thread.start()

    def _remember_micro_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        rounded = _profile_round(profile)
        self.last_micro_profile_ms = dict(rounded)
        self.micro_profile_window.append(dict(rounded))
        return rounded

    def _micro_profile_window_summary(self) -> Dict[str, Any]:
        samples = list(self.micro_profile_window)
        if not samples:
            return {"samples": 0, "ms": {}, "counts": {}}

        def pct(values: List[float], q: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = int(round((len(ordered) - 1) * q))
            return float(ordered[max(0, min(len(ordered) - 1, idx))])

        ms_summary: Dict[str, Any] = {}
        for key in MICRO_PROFILE_MS_FIELDS:
            vals = [float(s.get(key, 0.0) or 0.0) for s in samples]
            ms_summary[key] = {
                "avg": round(sum(vals) / max(1, len(vals)), 3),
                "p95": round(pct(vals, 0.95), 3),
                "max": round(max(vals) if vals else 0.0, 3),
            }
        count_summary: Dict[str, Any] = {}
        for key in MICRO_PROFILE_COUNT_FIELDS:
            vals_i = [int(s.get(key, 0) or 0) for s in samples]
            count_summary[key] = {
                "avg": round(sum(vals_i) / max(1, len(vals_i)), 3),
                "max": int(max(vals_i) if vals_i else 0),
            }
        return {
            "samples": len(samples),
            "ms": ms_summary,
            "counts": count_summary,
        }

    def _alignment_frame_info(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta = item.get("mvs_meta") if isinstance(item.get("mvs_meta"), dict) else {}
        fid = _safe_int_or_none(item.get("frame_id"))
        sync_index = _safe_int_or_none(item.get("sync_index"))
        if sync_index is None or sync_index < 0:
            sync_index = _safe_int_or_none(meta.get("program_sync_index"))
        if sync_index is None or sync_index < 0:
            sync_index = _safe_int_or_none(meta.get("sync_index"))

        meta_ids = [
            _safe_int_or_none(meta.get(key))
            for key in ("shared_frame_id", "frame_id_local")
        ]
        meta_ids = [x for x in meta_ids if x is not None]
        meta_frame_match: Optional[bool] = None
        if fid is not None and meta_ids:
            meta_frame_match = any(int(x) == int(fid) for x in meta_ids)

        exposure_us = _safe_int_or_none(meta.get("exposure_time_us"))
        if exposure_us is None:
            exposure_us = _safe_int_or_none(meta.get("rgb_exposure_time_us"))
        exposure_source = str(meta.get("exposure_source", "") or meta.get("rgb_exposure_source", "") or ("unknown" if exposure_us is None else "mvs_meta"))
        trigger_wall_us = _safe_int_or_none(meta.get("mvs_soft_trigger_est_wall_us"))
        visual_source = "mvs_soft_trigger_est_wall_us"
        if trigger_wall_us is None:
            trigger_wall_us = _safe_int_or_none(meta.get("soft_trigger_cmd_wall_us"))
            visual_source = "soft_trigger_cmd_wall_us"
        if trigger_wall_us is None:
            trigger_wall_us = _safe_int_or_none(item.get("timestamp_us"))
            visual_source = "frame_timestamp_us"

        start_offset_us = _safe_int_or_none(meta.get("rgb_exposure_start_offset_us"))
        exposure_start_us: Optional[int] = None
        if trigger_wall_us is not None:
            exposure_start_us = int(trigger_wall_us) + int(start_offset_us or 0)

        mid_offset_us = _safe_int_or_none(meta.get("rgb_exposure_mid_offset_us"))
        if mid_offset_us is None:
            if start_offset_us is not None and exposure_us is not None:
                mid_offset_us = int(start_offset_us) + int(exposure_us) // 2
                visual_source += "+start_offset+exposure_half"
            elif exposure_us is not None:
                mid_offset_us = int(exposure_us) // 2
                visual_source += "+exposure_half"
        else:
            visual_source += "+rgb_exposure_mid_offset_us"

        visual_mid_us: Optional[int] = None
        if trigger_wall_us is not None and mid_offset_us is not None:
            visual_mid_us = int(trigger_wall_us) + int(mid_offset_us)

        end_offset_us = _safe_int_or_none(meta.get("rgb_exposure_end_offset_us"))
        exposure_end_us: Optional[int] = None
        if trigger_wall_us is not None and end_offset_us is not None:
            exposure_end_us = int(trigger_wall_us) + int(end_offset_us)
        elif exposure_start_us is not None and exposure_us is not None:
            exposure_end_us = int(exposure_start_us) + int(exposure_us)

        return {
            "camera": str(item.get("camera", "")),
            "frame_id": fid,
            "sync_index": sync_index,
            "timestamp_us": _safe_int_or_none(item.get("timestamp_us")),
            "trigger_wall_us": trigger_wall_us,
            "exposure_us": exposure_us,
            "exposure_source": exposure_source,
            "exposure_start_us": exposure_start_us,
            "visual_mid_us": visual_mid_us,
            "exposure_end_us": exposure_end_us,
            "visual_mid_source": visual_source if visual_mid_us is not None else "",
            "meta_frame_match": meta_frame_match,
        }

    def _prune_input_alignment_seen(self) -> None:
        limit = int(self.input_alignment_audit_history)
        if len(self._input_alignment_seen_by_sync) <= limit:
            return
        ordered = sorted(
            self._input_alignment_seen_by_sync.items(),
            key=lambda kv: int(kv[1].get("last_seen_wall_us", 0) or 0),
        )
        for key, _entry in ordered[: max(0, len(ordered) - limit)]:
            self._input_alignment_seen_by_sync.pop(key, None)

    def _input_alignment_window_status(self) -> Dict[str, Any]:
        samples = list(self.input_alignment_window)
        if not samples:
            return {}

        def vals(key: str) -> List[float]:
            out: List[float] = []
            for row in samples:
                value = row.get(key)
                if value is None or value == "":
                    continue
                try:
                    out.append(float(value))
                except Exception:
                    continue
            return out

        visual = vals("visual_mid_spread_ms")
        exposure = vals("exposure_spread_us")
        exposure_end = vals("exposure_end_spread_ms")
        capture = vals("capture_ts_spread_ms")
        return {
            "samples": int(len(samples)),
            "visual_mid_spread_ms_p95": None if not visual else round(float(_percentile(visual, 0.95) or 0.0), 3),
            "exposure_spread_us_p95": None if not exposure else int(round(float(_percentile(exposure, 0.95) or 0.0))),
            "exposure_end_spread_ms_p95": None if not exposure_end else round(float(_percentile(exposure_end, 0.95) or 0.0), 3),
            "capture_ts_spread_ms_p95": None if not capture else round(float(_percentile(capture, 0.95) or 0.0), 3),
        }

    def _per_camera_exposure_status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for cam in self.cams:
            vals = [int(x) for x in self.exposure_windows_by_cam.get(cam, [])]
            if not vals:
                out[cam] = {
                    "samples": 0,
                    "source": self.exposure_source_by_cam.get(cam, ""),
                }
                continue
            out[cam] = {
                "samples": int(len(vals)),
                "source": self.exposure_source_by_cam.get(cam, ""),
                "last_us": int(vals[-1]),
                "avg_us": int(round(sum(vals) / max(1, len(vals)))),
                "p50_us": int(round(float(_percentile([float(v) for v in vals], 0.50) or 0.0))),
                "p95_us": int(round(float(_percentile([float(v) for v in vals], 0.95) or 0.0))),
                "min_us": int(min(vals)),
                "max_us": int(max(vals)),
            }
        return out

    def _input_alignment_status(self) -> Dict[str, Any]:
        s = self.stats
        batches = int(s.get("input_alignment_batches", 0) or 0)
        frames = int(s.get("input_alignment_frames", 0) or 0)
        full_sets = int(s.get("input_alignment_sync_full_sets_seen", 0) or 0)
        wait_total = float(s.get("input_alignment_sync_full_wait_ms_total", 0.0) or 0.0)
        window = self._input_alignment_window_status()
        return {
            "enabled": bool(self.input_alignment_audit_enabled),
            "mode": "active_buffered" if self.input_buffer_enabled else "audit_only",
            "max_wait_ms": round(float(self.input_alignment_audit_max_wait_ms), 3),
            "max_spread_ms": round(float(self.input_alignment_audit_max_spread_ms), 3),
            "batches": batches,
            "frames": frames,
            "full_batch_ratio": _ratio(s.get("input_alignment_full_batches"), batches),
            "partial_batch_ratio": _ratio(s.get("input_alignment_partial_batches"), batches),
            "same_sync_batch_ratio": _ratio(s.get("input_alignment_same_sync_batches"), batches),
            "same_sync_full_batch_ratio": _ratio(s.get("input_alignment_same_sync_full_batches"), batches),
            "visual_mid_complete_batch_ratio": _ratio(s.get("input_alignment_visual_mid_complete_batches"), batches),
            "visual_mid_aligned_batch_ratio": _ratio(s.get("input_alignment_visual_mid_aligned_batches"), batches),
            "current_aligned_full_batch_ratio": _ratio(s.get("input_alignment_current_aligned_full_batches"), batches),
            "meta_frame_match_batch_ratio": _ratio(s.get("input_alignment_meta_frame_match_batches"), batches),
            "frames_with_sync_ratio": _ratio(s.get("input_alignment_frames_with_sync"), frames),
            "frames_with_visual_mid_ratio": _ratio(s.get("input_alignment_frames_with_visual_mid"), frames),
            "sync_spread_max": int(s.get("input_alignment_sync_spread_max", 0) or 0),
            "visual_mid_spread_ms_max": round(float(s.get("input_alignment_visual_mid_spread_ms_max", 0.0) or 0.0), 3),
            "visual_mid_spread_ms_p95": window.get("visual_mid_spread_ms_p95"),
            "capture_ts_spread_ms_max": round(float(s.get("input_alignment_capture_ts_spread_ms_max", 0.0) or 0.0), 3),
            "capture_ts_spread_ms_p95": window.get("capture_ts_spread_ms_p95"),
            "exposure_spread_us_max": int(s.get("input_alignment_exposure_spread_us_max", 0) or 0),
            "exposure_spread_us_p95": window.get("exposure_spread_us_p95"),
            "exposure_end_spread_ms_p95": window.get("exposure_end_spread_ms_p95"),
            "sync_full_sets_seen": full_sets,
            "sync_full_sets_within_wait": int(s.get("input_alignment_sync_full_sets_within_wait", 0) or 0),
            "sync_full_sets_within_wait_ratio": _ratio(s.get("input_alignment_sync_full_sets_within_wait"), full_sets),
            "sync_full_sets_visual_aligned": int(s.get("input_alignment_sync_full_sets_visual_aligned", 0) or 0),
            "sync_full_sets_visual_aligned_ratio": _ratio(s.get("input_alignment_sync_full_sets_visual_aligned"), full_sets),
            "sync_full_wait_ms_avg": round(wait_total / max(1, full_sets), 3),
            "sync_full_wait_ms_max": round(float(s.get("input_alignment_sync_full_wait_ms_max", 0.0) or 0.0), 3),
            "sync_full_visual_spread_ms_max": round(float(s.get("input_alignment_sync_full_visual_spread_ms_max", 0.0) or 0.0), 3),
            "recent_sync_entries": int(len(self._input_alignment_seen_by_sync)),
            "per_camera_exposure_us": self._per_camera_exposure_status(),
            "last_batch": dict(self.last_input_alignment_audit or {}),
        }

    def _audit_input_alignment(self, task: PostprocessTask) -> Dict[str, Any]:
        if not self.input_alignment_audit_enabled:
            return {}
        group_cams = [
            str(x)
            for x in (task.collect_meta.get("group_cams") if isinstance(task.collect_meta, dict) else [])
            if str(x)
        ]
        if not group_cams:
            group_cams = [str(x) for x in task.cam_ids]
        group_size = max(1, int(task.group_size or len(group_cams) or len(task.items)))
        group_key = ",".join(group_cams)
        infos = [self._alignment_frame_info(item) for item in task.items]
        batch_size = len(infos)
        sync_values = [int(x["sync_index"]) for x in infos if x.get("sync_index") is not None and int(x["sync_index"]) >= 0]
        visual_values = [int(x["visual_mid_us"]) for x in infos if x.get("visual_mid_us") is not None]
        exposure_start_values = [int(x["exposure_start_us"]) for x in infos if x.get("exposure_start_us") is not None]
        exposure_end_values = [int(x["exposure_end_us"]) for x in infos if x.get("exposure_end_us") is not None]
        capture_values = [int(x["timestamp_us"]) for x in infos if x.get("timestamp_us") is not None]
        exposure_values = [int(x["exposure_us"]) for x in infos if x.get("exposure_us") is not None]
        meta_match_values = [x.get("meta_frame_match") for x in infos if x.get("meta_frame_match") is not None]
        cam_values = [str(x.get("camera", "")) for x in infos if str(x.get("camera", ""))]
        missing_cams = [cam for cam in group_cams if cam not in set(cam_values)]

        sync_spread = _spread(sync_values)
        visual_spread_us = _spread(visual_values)
        exposure_end_spread_us = _spread(exposure_end_values)
        capture_spread_us = _spread(capture_values)
        exposure_spread = _spread(exposure_values)
        visual_spread_ms = None if visual_spread_us is None else float(visual_spread_us) / 1000.0
        exposure_end_spread_ms = None if exposure_end_spread_us is None else float(exposure_end_spread_us) / 1000.0
        capture_spread_ms = None if capture_spread_us is None else float(capture_spread_us) / 1000.0
        sync_complete = bool(batch_size > 0 and len(sync_values) == batch_size)
        visual_complete = bool(batch_size > 0 and len(visual_values) == batch_size)
        full_batch = bool(batch_size >= group_size)
        same_sync = bool(sync_complete and (sync_spread or 0) == 0)
        visual_aligned = bool(visual_complete and visual_spread_ms is not None and visual_spread_ms <= self.input_alignment_audit_max_spread_ms)
        meta_match_all = bool(batch_size > 0 and len(meta_match_values) == batch_size and all(bool(x) for x in meta_match_values))
        meta_mismatch = any(x is False for x in meta_match_values)
        current_aligned_full = bool(full_batch and same_sync and visual_aligned and not meta_mismatch)

        stats = self.stats
        stats["input_alignment_batches"] = int(stats.get("input_alignment_batches", 0) or 0) + 1
        stats["input_alignment_frames"] = int(stats.get("input_alignment_frames", 0) or 0) + batch_size
        stats["input_alignment_frames_with_sync"] = int(stats.get("input_alignment_frames_with_sync", 0) or 0) + len(sync_values)
        stats["input_alignment_frames_with_visual_mid"] = int(stats.get("input_alignment_frames_with_visual_mid", 0) or 0) + len(visual_values)
        if full_batch:
            stats["input_alignment_full_batches"] = int(stats.get("input_alignment_full_batches", 0) or 0) + 1
        else:
            stats["input_alignment_partial_batches"] = int(stats.get("input_alignment_partial_batches", 0) or 0) + 1
        if same_sync:
            stats["input_alignment_same_sync_batches"] = int(stats.get("input_alignment_same_sync_batches", 0) or 0) + 1
            if full_batch:
                stats["input_alignment_same_sync_full_batches"] = int(stats.get("input_alignment_same_sync_full_batches", 0) or 0) + 1
        if visual_complete:
            stats["input_alignment_visual_mid_complete_batches"] = int(stats.get("input_alignment_visual_mid_complete_batches", 0) or 0) + 1
        if visual_aligned:
            stats["input_alignment_visual_mid_aligned_batches"] = int(stats.get("input_alignment_visual_mid_aligned_batches", 0) or 0) + 1
        if current_aligned_full:
            stats["input_alignment_current_aligned_full_batches"] = int(stats.get("input_alignment_current_aligned_full_batches", 0) or 0) + 1
        if meta_match_all:
            stats["input_alignment_meta_frame_match_batches"] = int(stats.get("input_alignment_meta_frame_match_batches", 0) or 0) + 1
        if meta_mismatch:
            stats["input_alignment_meta_frame_mismatch_batches"] = int(stats.get("input_alignment_meta_frame_mismatch_batches", 0) or 0) + 1
        if sync_spread is not None:
            stats["input_alignment_sync_spread_max"] = max(int(stats.get("input_alignment_sync_spread_max", 0) or 0), int(sync_spread))
        if visual_spread_ms is not None:
            stats["input_alignment_visual_mid_spread_ms_max"] = max(float(stats.get("input_alignment_visual_mid_spread_ms_max", 0.0) or 0.0), float(visual_spread_ms))
        if capture_spread_ms is not None:
            stats["input_alignment_capture_ts_spread_ms_max"] = max(float(stats.get("input_alignment_capture_ts_spread_ms_max", 0.0) or 0.0), float(capture_spread_ms))
        if exposure_spread is not None:
            stats["input_alignment_exposure_spread_us_max"] = max(int(stats.get("input_alignment_exposure_spread_us_max", 0) or 0), int(exposure_spread))
        for info in infos:
            cam = str(info.get("camera", ""))
            exposure = info.get("exposure_us")
            if not cam or exposure is None:
                continue
            try:
                self.exposure_windows_by_cam.setdefault(cam, deque(maxlen=max(16, int(self.input_alignment_audit_history)))).append(int(exposure))
                self.exposure_source_by_cam[cam] = str(info.get("exposure_source", "") or "")
            except Exception:
                continue

        seen_wall_us = now_us()
        would_full_seen = False
        would_within_wait = False
        would_wait_ms: Optional[float] = None
        would_visual_spread_ms: Optional[float] = None
        touched_keys: List[str] = []
        for info in infos:
            sync_index = info.get("sync_index")
            cam = str(info.get("camera", ""))
            if sync_index is None or int(sync_index) < 0 or not cam:
                continue
            key = f"{group_key}:{int(sync_index)}"
            entry = self._input_alignment_seen_by_sync.setdefault(key, {
                "group_key": group_key,
                "group_cams": list(group_cams),
                "sync_index": int(sync_index),
                "cams": {},
                "counted": False,
                "last_seen_wall_us": 0,
            })
            entry["last_seen_wall_us"] = int(seen_wall_us)
            cams_map = entry.setdefault("cams", {})
            if isinstance(cams_map, dict):
                cams_map[cam] = {**info, "seen_wall_us": int(seen_wall_us)}
            touched_keys.append(key)

        for key in sorted(set(touched_keys)):
            entry = self._input_alignment_seen_by_sync.get(key)
            if not isinstance(entry, dict) or bool(entry.get("counted", False)):
                continue
            cams_map = entry.get("cams", {})
            if not isinstance(cams_map, dict):
                continue
            required = [str(x) for x in entry.get("group_cams", group_cams) or group_cams]
            if not required or any(cam not in cams_map for cam in required):
                continue
            entry["counted"] = True
            set_infos = [cams_map[cam] for cam in required]
            wait_values = [int(x.get("seen_wall_us", 0) or 0) for x in set_infos]
            wait_spread_us = _spread(wait_values) or 0
            wait_ms = float(wait_spread_us) / 1000.0
            set_visual_values = [int(x["visual_mid_us"]) for x in set_infos if x.get("visual_mid_us") is not None]
            set_visual_complete = len(set_visual_values) == len(required)
            set_visual_spread_us = _spread(set_visual_values) if set_visual_complete else None
            set_visual_ms = None if set_visual_spread_us is None else float(set_visual_spread_us) / 1000.0
            set_visual_aligned = bool(set_visual_ms is not None and set_visual_ms <= self.input_alignment_audit_max_spread_ms)
            within_wait = bool(wait_ms <= self.input_alignment_audit_max_wait_ms)

            stats["input_alignment_sync_full_sets_seen"] = int(stats.get("input_alignment_sync_full_sets_seen", 0) or 0) + 1
            stats["input_alignment_sync_full_wait_ms_total"] = round(float(stats.get("input_alignment_sync_full_wait_ms_total", 0.0) or 0.0) + wait_ms, 3)
            stats["input_alignment_sync_full_wait_ms_max"] = max(float(stats.get("input_alignment_sync_full_wait_ms_max", 0.0) or 0.0), round(wait_ms, 3))
            if within_wait:
                stats["input_alignment_sync_full_sets_within_wait"] = int(stats.get("input_alignment_sync_full_sets_within_wait", 0) or 0) + 1
            if set_visual_aligned:
                stats["input_alignment_sync_full_sets_visual_aligned"] = int(stats.get("input_alignment_sync_full_sets_visual_aligned", 0) or 0) + 1
            if set_visual_ms is not None:
                stats["input_alignment_sync_full_visual_spread_ms_max"] = max(float(stats.get("input_alignment_sync_full_visual_spread_ms_max", 0.0) or 0.0), round(set_visual_ms, 3))
            would_full_seen = True
            would_within_wait = within_wait
            would_wait_ms = wait_ms
            would_visual_spread_ms = set_visual_ms

        self._prune_input_alignment_seen()
        late_cam = ""
        if visual_values and len(visual_values) == len(infos):
            late_idx = max(range(len(infos)), key=lambda idx: int(infos[idx].get("visual_mid_us") or 0))
            late_cam = str(infos[late_idx].get("camera", "") or "")

        audit = {
            "enabled": True,
            "mode": "active_buffered" if self.input_buffer_enabled else "audit_only",
            "group_cams": group_cams,
            "cam_ids": cam_values,
            "batch_size": int(batch_size),
            "group_size": int(group_size),
            "full_batch": bool(full_batch),
            "missing_cams": missing_cams,
            "late_cam": late_cam,
            "frame_ids": [x.get("frame_id") for x in infos],
            "sync_indices": [x.get("sync_index") for x in infos],
            "sync_spread": None if sync_spread is None else int(sync_spread),
            "same_sync": bool(same_sync),
            "visual_mid_complete": bool(visual_complete),
            "exposure_start_us": [x.get("exposure_start_us") for x in infos],
            "visual_mid_us": [x.get("visual_mid_us") for x in infos],
            "exposure_end_us": [x.get("exposure_end_us") for x in infos],
            "visual_mid_spread_ms": None if visual_spread_ms is None else round(float(visual_spread_ms), 3),
            "exposure_end_spread_ms": None if exposure_end_spread_ms is None else round(float(exposure_end_spread_ms), 3),
            "visual_mid_aligned": bool(visual_aligned),
            "capture_ts_spread_ms": None if capture_spread_ms is None else round(float(capture_spread_ms), 3),
            "exposure_us": [x.get("exposure_us") for x in infos],
            "exposure_source": [x.get("exposure_source") for x in infos],
            "exposure_spread_us": None if exposure_spread is None else int(exposure_spread),
            "meta_frame_match": bool(meta_match_all),
            "meta_frame_mismatch": bool(meta_mismatch),
            "current_aligned_full": bool(current_aligned_full),
            "would_full_seen": bool(would_full_seen),
            "would_full_within_wait": bool(would_within_wait),
            "would_wait_ms": None if would_wait_ms is None else round(float(would_wait_ms), 3),
            "would_visual_mid_spread_ms": None if would_visual_spread_ms is None else round(float(would_visual_spread_ms), 3),
            "max_wait_ms": round(float(self.input_alignment_audit_max_wait_ms), 3),
            "max_spread_ms": round(float(self.input_alignment_audit_max_spread_ms), 3),
            "input_buffer": dict(task.collect_meta.get("input_buffer", {}) if isinstance(task.collect_meta.get("input_buffer"), dict) else {}),
        }
        self.last_input_alignment_audit = dict(audit)
        self.input_alignment_window.append(dict(audit))
        return audit

    def _output_root(self) -> Path:
        if self.shadow:
            root = _resolve_project_path(self.project_root, self.args.shadow_output_dir)
        else:
            root = self.sync_root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _writer_for(self, cam: str) -> JsonlWriter:
        if cam in self.writers:
            return self.writers[cam]
        path = self.output_root / f"human_result_{cam}.jsonl"
        self.writers[cam] = JsonlWriter(path, truncate=bool(self.args.truncate_jsonl_on_start))
        return self.writers[cam]

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        model_path = _nonempty_str(self.model_load_path)
        if not model_path:
            raise RuntimeError("no YOLO model or engine path configured")
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(model_path)
        return self.model

    def _predict(self, frames: List[Any]) -> Tuple[List[Any], float]:
        model = self._load_model()
        t0 = time.time()
        kwargs = {
            "source": frames,
            "imgsz": int(self.args.imgsz),
            "conf": float(self.args.conf),
            "iou": float(self.args.iou),
            "device": str(self.args.device),
            "half": bool(self.args.half),
            "retina_masks": bool(self.args.retina_masks),
            "max_det": int(self.args.max_det),
            "verbose": False,
            "stream": False,
        }
        results = model.predict(**kwargs)
        predict_ms = (time.time() - t0) * 1000.0
        return list(results or []), predict_ms

    def _input_buffer_pending_count(self, group: Sequence[str]) -> int:
        total = 0
        for cam in group:
            last = int(self.last_emitted.get(str(cam), 0) or 0)
            last_sync_raw = self.last_emitted_sync.get(str(cam), -1)
            last_sync = -1 if last_sync_raw is None else int(last_sync_raw)
            for item in self.input_buffers.get(str(cam), []):
                fid = int(item.get("frame_id", 0) or 0)
                sync_index = self._alignment_frame_info(item).get("sync_index")
                sync_ok = sync_index is None or int(sync_index) > last_sync
                if fid > last and sync_ok:
                    total += 1
        return total

    def _buffer_fresh_items(self, items: Sequence[Dict[str, Any]]) -> None:
        if not items:
            return
        seen_wall_us = now_us()
        for item in items:
            cam = str(item.get("camera", ""))
            if not cam:
                continue
            self._input_buffer_seq += 1
            item["_buffer_seen_wall_us"] = int(seen_wall_us)
            item["_buffer_seq"] = int(self._input_buffer_seq)
            self.input_buffers.setdefault(cam, deque(maxlen=self.input_buffer_size)).append(item)
            self.stats["input_buffer_frames_buffered"] = int(self.stats.get("input_buffer_frames_buffered", 0) or 0) + 1
        self.stats["input_buffer_pending_frames"] = int(self._input_buffer_pending_count(self.cams))

    def _eligible_buffer_items(self, cam: str) -> List[Dict[str, Any]]:
        last = int(self.last_emitted.get(str(cam), 0) or 0)
        last_sync_raw = self.last_emitted_sync.get(str(cam), -1)
        last_sync = -1 if last_sync_raw is None else int(last_sync_raw)
        out: List[Dict[str, Any]] = []
        for item in self.input_buffers.get(str(cam), []):
            fid = int(item.get("frame_id", 0) or 0)
            sync_index = self._alignment_frame_info(item).get("sync_index")
            sync_ok = sync_index is None or int(sync_index) > last_sync
            if fid > last and sync_ok:
                out.append(item)
        return out

    def _candidate_meta(
        self,
        group: Sequence[str],
        items: Sequence[Dict[str, Any]],
        decision: str,
        fallback_reason: str,
        wait_ms: float,
    ) -> Dict[str, Any]:
        infos = [self._alignment_frame_info(item) for item in items]
        visual_values = [int(x["visual_mid_us"]) for x in infos if x.get("visual_mid_us") is not None]
        exposure_end_values = [int(x["exposure_end_us"]) for x in infos if x.get("exposure_end_us") is not None]
        visual_spread = _spread(visual_values)
        exposure_end_spread = _spread(exposure_end_values)
        cam_ids = [str(x.get("camera", "")) for x in infos if str(x.get("camera", ""))]
        sync_values = [x.get("sync_index") for x in infos]
        missing = [str(cam) for cam in group if str(cam) not in set(cam_ids)]
        late_cam = ""
        if len(visual_values) == len(infos) and infos:
            late_idx = max(range(len(infos)), key=lambda idx: int(infos[idx].get("visual_mid_us") or 0))
            late_cam = str(infos[late_idx].get("camera", "") or "")
        return {
            "enabled": True,
            "decision": str(decision),
            "fallback_reason": str(fallback_reason),
            "wait_ms": round(float(wait_ms), 3),
            "pending_frames": int(self._input_buffer_pending_count(group)),
            "selected_sync_indices": [None if x is None else int(x) for x in sync_values],
            "selected_sync_index": sync_values[0] if sync_values and all(x == sync_values[0] for x in sync_values) else None,
            "missing_cams": missing,
            "late_cam": late_cam,
            "visual_mid_spread_ms": None if visual_spread is None else round(float(visual_spread) / 1000.0, 3),
            "exposure_end_spread_ms": None if exposure_end_spread is None else round(float(exposure_end_spread) / 1000.0, 3),
            "max_wait_ms": round(float(self.input_buffer_max_wait_ms), 3),
            "max_spread_ms": round(float(self.input_buffer_max_spread_ms), 3),
        }

    def _select_input_buffer_batch(
        self,
        group: Sequence[str],
        allow_fallback: bool,
        waited_ms: float,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        group = [str(cam) for cam in group]
        by_cam = {cam: self._eligible_buffer_items(cam) for cam in group}
        sync_to_cams: Dict[int, Dict[str, Dict[str, Any]]] = {}
        latest_by_cam: Dict[str, Dict[str, Any]] = {}
        for cam, items in by_cam.items():
            if items:
                latest_by_cam[cam] = max(items, key=lambda item: int(item.get("frame_id", 0) or 0))
            for item in items:
                info = self._alignment_frame_info(item)
                sync = info.get("sync_index")
                if sync is None or int(sync) < 0:
                    continue
                sync_map = sync_to_cams.setdefault(int(sync), {})
                prev = sync_map.get(cam)
                if prev is None or int(item.get("frame_id", 0) or 0) >= int(prev.get("frame_id", 0) or 0):
                    sync_map[cam] = item

        now_wall_us = now_us()

        def full_candidates() -> List[Tuple[int, List[Dict[str, Any]], Optional[float], Optional[int]]]:
            out: List[Tuple[int, List[Dict[str, Any]], Optional[float], Optional[int]]] = []
            for sync, cam_map in sync_to_cams.items():
                if any(cam not in cam_map for cam in group):
                    continue
                items = [cam_map[cam] for cam in group]
                infos = [self._alignment_frame_info(item) for item in items]
                visual_values = [int(x["visual_mid_us"]) for x in infos if x.get("visual_mid_us") is not None]
                visual_spread_us = _spread(visual_values) if len(visual_values) == len(items) else None
                visual_spread_ms = None if visual_spread_us is None else float(visual_spread_us) / 1000.0
                exposure_end_values = [int(x["exposure_end_us"]) for x in infos if x.get("exposure_end_us") is not None]
                max_exposure_end_us = max(exposure_end_values) if exposure_end_values else None
                out.append((int(sync), items, visual_spread_ms, max_exposure_end_us))
            return sorted(out, key=lambda row: row[0])

        for sync, items, visual_ms, max_end_us in full_candidates():
            if visual_ms is None or visual_ms > self.input_buffer_max_spread_ms:
                continue
            if self.input_buffer_exposure_end_barrier and max_end_us is not None and now_wall_us < int(max_end_us):
                continue
            meta = self._candidate_meta(group, items, "aligned_full", "", waited_ms)
            return list(items), meta

        if not allow_fallback:
            return [], None

        best_full: Optional[Tuple[int, List[Dict[str, Any]], Optional[float], Optional[int]]] = None
        for candidate in full_candidates():
            best_full = candidate
            break
        if best_full is not None:
            _sync, items, visual_ms, _max_end_us = best_full
            reason = "visual_mid_spread_exceeded" if visual_ms is not None and visual_ms > self.input_buffer_max_spread_ms else "full_sync_timeout"
            meta = self._candidate_meta(group, items, "fallback_full", reason, waited_ms)
            return list(items), meta

        partial_candidates: List[Tuple[int, int, List[Dict[str, Any]]]] = []
        for sync, cam_map in sync_to_cams.items():
            items = [cam_map[cam] for cam in group if cam in cam_map]
            if items:
                partial_candidates.append((-len(items), int(sync), items))
        if partial_candidates:
            _neg_count, _sync, items = sorted(partial_candidates, key=lambda row: (row[0], row[1]))[0]
            meta = self._candidate_meta(group, items, "fallback_sync_partial", "sync_partial_timeout", waited_ms)
            return list(items), meta

        if latest_by_cam:
            items = [latest_by_cam[cam] for cam in group if cam in latest_by_cam]
            meta = self._candidate_meta(group, items, "fallback_latest_partial", "latest_partial_timeout", waited_ms)
            return list(items), meta
        return [], None

    def _consume_input_buffer_items(self, items: Sequence[Dict[str, Any]], group: Optional[Sequence[str]] = None) -> None:
        selected_syncs: List[int] = []
        for item in items:
            cam = str(item.get("camera", ""))
            fid = int(item.get("frame_id", 0) or 0)
            if not cam or fid <= 0:
                continue
            self.last_emitted[cam] = max(int(self.last_emitted.get(cam, 0) or 0), fid)
            sync_index = self._alignment_frame_info(item).get("sync_index")
            if sync_index is not None and int(sync_index) >= 0:
                selected_syncs.append(int(sync_index))
                prev_sync_raw = self.last_emitted_sync.get(cam, -1)
                prev_sync = -1 if prev_sync_raw is None else int(prev_sync_raw)
                self.last_emitted_sync[cam] = max(prev_sync, int(sync_index))
        if group and selected_syncs and all(sync == selected_syncs[0] for sync in selected_syncs):
            sync_watermark = int(selected_syncs[0])
            for cam in group:
                cam_s = str(cam)
                prev_sync_raw = self.last_emitted_sync.get(cam_s, -1)
                prev_sync = -1 if prev_sync_raw is None else int(prev_sync_raw)
                self.last_emitted_sync[cam_s] = max(prev_sync, sync_watermark)
        pruned = 0
        for cam, buf in self.input_buffers.items():
            last = int(self.last_emitted.get(cam, 0) or 0)
            last_sync_raw = self.last_emitted_sync.get(cam, -1)
            last_sync = -1 if last_sync_raw is None else int(last_sync_raw)
            while buf:
                head_fid = int(buf[0].get("frame_id", 0) or 0)
                head_sync = self._alignment_frame_info(buf[0]).get("sync_index")
                head_sync_pruned = head_sync is not None and int(head_sync) <= last_sync
                if head_fid > last and not head_sync_pruned:
                    break
                buf.popleft()
                pruned += 1
        if pruned:
            self.stats["input_buffer_pruned_frames"] = int(self.stats.get("input_buffer_pruned_frames", 0) or 0) + pruned
        self.stats["input_buffer_pending_frames"] = int(self._input_buffer_pending_count(self.cams))

    def _read_group_buffered(self, group: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        collect_t0 = time.time()
        deadline = collect_t0 + max(0.0, float(self.input_buffer_max_wait_ms) / 1000.0)
        initial_count = 0
        added_after_initial = 0
        first_poll = True
        selected: List[Dict[str, Any]] = []
        selected_meta: Optional[Dict[str, Any]] = None
        while True:
            more = self._read_group_once(group)
            if first_poll:
                initial_count = len(more)
                first_poll = False
            else:
                added_after_initial += len(more)
            self._buffer_fresh_items(more)
            waited_ms = max(0.0, (time.time() - collect_t0) * 1000.0)
            selected, selected_meta = self._select_input_buffer_batch(group, allow_fallback=False, waited_ms=waited_ms)
            if selected:
                break
            pending = self._input_buffer_pending_count(group)
            if pending <= 0:
                break
            if time.time() >= deadline:
                selected, selected_meta = self._select_input_buffer_batch(group, allow_fallback=True, waited_ms=waited_ms)
                break
            time.sleep(min(self.batch_collect_retry_s, max(0.0, deadline - time.time())))

        collect_wait_ms = max(0.0, (time.time() - collect_t0) * 1000.0)
        if initial_count > 0 and collect_wait_ms > 0.0:
            self.stats["collect_wait_batches"] = int(self.stats.get("collect_wait_batches", 0)) + 1
            self.stats["collect_wait_added_frames"] = int(self.stats.get("collect_wait_added_frames", 0)) + int(added_after_initial)
            self.stats["collect_wait_ms_total"] = round(float(self.stats.get("collect_wait_ms_total", 0.0)) + collect_wait_ms, 3)
        meta = {
            "group_size": int(len(group)),
            "group_cams": [str(x) for x in group],
            "initial_batch_size": int(initial_count),
            "collect_wait_ms": round(float(collect_wait_ms), 3),
            "collect_added_frames": int(added_after_initial),
            "input_buffer": selected_meta or {
                "enabled": True,
                "decision": "none",
                "fallback_reason": "no_pending_frames" if self._input_buffer_pending_count(group) <= 0 else "no_candidate",
                "wait_ms": round(float(collect_wait_ms), 3),
                "pending_frames": int(self._input_buffer_pending_count(group)),
                "max_wait_ms": round(float(self.input_buffer_max_wait_ms), 3),
                "max_spread_ms": round(float(self.input_buffer_max_spread_ms), 3),
            },
        }
        if selected:
            self._consume_input_buffer_items(selected, group)
            self.stats["input_buffer_batches"] = int(self.stats.get("input_buffer_batches", 0) or 0) + 1
            self.stats["input_buffer_wait_ms_total"] = round(float(self.stats.get("input_buffer_wait_ms_total", 0.0) or 0.0) + collect_wait_ms, 3)
            self.stats["input_buffer_wait_ms_max"] = max(float(self.stats.get("input_buffer_wait_ms_max", 0.0) or 0.0), round(float(collect_wait_ms), 3))
            decision = str((selected_meta or {}).get("decision", ""))
            fallback_reason = str((selected_meta or {}).get("fallback_reason", ""))
            if decision == "aligned_full":
                self.stats["input_buffer_aligned_full_batches"] = int(self.stats.get("input_buffer_aligned_full_batches", 0) or 0) + 1
            if fallback_reason:
                self.stats["input_buffer_fallback_batches"] = int(self.stats.get("input_buffer_fallback_batches", 0) or 0) + 1
            if decision == "fallback_sync_partial":
                self.stats["input_buffer_sync_partial_batches"] = int(self.stats.get("input_buffer_sync_partial_batches", 0) or 0) + 1
            if decision == "fallback_latest_partial":
                self.stats["input_buffer_latest_partial_batches"] = int(self.stats.get("input_buffer_latest_partial_batches", 0) or 0) + 1
            if fallback_reason == "visual_mid_spread_exceeded":
                self.stats["input_buffer_visual_misaligned_full_batches"] = int(self.stats.get("input_buffer_visual_misaligned_full_batches", 0) or 0) + 1
        return selected, meta

    def _read_group_once(self, group: Sequence[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cam in group:
            item = self.readers[cam].read()
            if not item:
                continue
            fid = int(item.get("frame_id", 0) or 0)
            if fid <= 0 or self.last_seen.get(cam) == fid:
                continue
            self.last_seen[cam] = fid
            out.append(item)
        self.stats["frames_read"] = int(self.stats.get("frames_read", 0)) + len(out)
        return out

    def _read_group(self, group: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.input_buffer_enabled:
            return self._read_group_buffered(group)
        collect_t0 = time.time()
        items = self._read_group_once(group)
        initial_count = len(items)
        group_size = len(group)
        wait_start: Optional[float] = None
        if items and self.batch_collect_wait_s > 0.0 and len(items) < group_size:
            got = {str(x.get("camera")) for x in items}
            wait_start = time.time()
            deadline = collect_t0 + self.batch_collect_wait_s
            while len(got) < group_size:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                time.sleep(min(self.batch_collect_retry_s, remaining))
                missing = [cam for cam in group if cam not in got]
                more = self._read_group_once(missing)
                if not more:
                    continue
                items.extend(more)
                got.update(str(x.get("camera")) for x in more)
        collect_wait_ms = max(0.0, (time.time() - wait_start) * 1000.0) if wait_start is not None else 0.0
        added = max(0, len(items) - initial_count)
        if initial_count > 0 and collect_wait_ms > 0.0:
            self.stats["collect_wait_batches"] = int(self.stats.get("collect_wait_batches", 0)) + 1
            self.stats["collect_wait_added_frames"] = int(self.stats.get("collect_wait_added_frames", 0)) + added
            self.stats["collect_wait_ms_total"] = round(float(self.stats.get("collect_wait_ms_total", 0.0)) + collect_wait_ms, 3)
        meta = {
            "group_size": int(group_size),
            "group_cams": [str(x) for x in group],
            "initial_batch_size": int(initial_count),
            "collect_wait_ms": round(float(collect_wait_ms), 3),
            "collect_added_frames": int(added),
        }
        return items, meta

    def _process_batch(self, items: List[Dict[str, Any]], collect_meta: Optional[Dict[str, Any]] = None) -> None:
        if not items:
            return
        observed_t0 = _perf_now()
        collect_meta = collect_meta or {}
        group_size = int(collect_meta.get("group_size") or len(items))
        if len(items) >= group_size:
            self.stats["full_batches"] = int(self.stats.get("full_batches", 0)) + 1
        else:
            self.stats["partial_batches"] = int(self.stats.get("partial_batches", 0)) + 1
        self.batch_index += 1
        micro_profile: Dict[str, Any] = {}
        prepare_t0 = _perf_now()
        frames = [x.get("frame") for x in items]
        cam_ids = [str(x.get("camera")) for x in items]
        frame_shapes = [getattr(f, "shape", ()) for f in frames]
        _profile_add(micro_profile, "frame_prepare_ms", _elapsed_ms(prepare_t0))
        batch_t0 = time.time()
        results, predict_ms = self._predict(frames)
        _profile_add(micro_profile, "predict_ms", float(predict_ms))
        self.stats["frames_inferred"] = int(self.stats.get("frames_inferred", 0)) + len(items)
        task_items: List[Dict[str, Any]] = []
        for item in items:
            meta_item = dict(item)
            meta_item.pop("frame", None)
            task_items.append(meta_item)
        task = PostprocessTask(
            batch_index=int(self.batch_index),
            group_size=int(group_size),
            items=task_items,
            results=list(results or []),
            cam_ids=cam_ids,
            frame_shapes=frame_shapes,
            collect_meta=dict(collect_meta),
            predict_ms=float(predict_ms),
            batch_t0_s=float(batch_t0),
            observed_t0=float(observed_t0),
            micro_profile=micro_profile,
        )
        if self.async_record_postprocess:
            self._enqueue_postprocess_task(task)
            return
        self._process_postprocess_task(task)

    def _enqueue_postprocess_task(self, task: PostprocessTask) -> None:
        if self.postprocess_queue is None:
            self._process_postprocess_task(task)
            return
        if self.postprocess_thread is not None and not self.postprocess_thread.is_alive():
            err = self._postprocess_worker_error or "postprocess worker stopped"
            raise RuntimeError(err)
        enqueue_t0 = _perf_now()
        task.enqueue_wall_us = now_us()
        full_waited = False
        while True:
            if self.postprocess_thread is not None and not self.postprocess_thread.is_alive():
                err = self._postprocess_worker_error or "postprocess worker stopped"
                raise RuntimeError(err)
            backpressure_ms = _elapsed_ms(enqueue_t0)
            task.backpressure_ms = float(backpressure_ms)
            task.micro_profile["async_enqueue_ms"] = float(backpressure_ms)
            task.micro_profile["async_backpressure_ms"] = float(backpressure_ms)
            try:
                self.postprocess_queue.put_nowait(task)
                break
            except queue.Full:
                full_waited = True
                time.sleep(0.001)
        task.backpressure_ms = float(backpressure_ms)
        depth = int(self.postprocess_queue.qsize())
        self.stats["postprocess_batches_enqueued"] = int(self.stats.get("postprocess_batches_enqueued", 0)) + 1
        self.stats["postprocess_queue_depth"] = depth
        self.stats["postprocess_queue_max_depth"] = max(int(self.stats.get("postprocess_queue_max_depth", 0) or 0), depth)
        self.stats["postprocess_backpressure_ms_total"] = round(
            float(self.stats.get("postprocess_backpressure_ms_total", 0.0) or 0.0) + backpressure_ms,
            3,
        )
        if full_waited or backpressure_ms > 0.5:
            self.stats["postprocess_queue_full_waits"] = int(self.stats.get("postprocess_queue_full_waits", 0)) + 1

    def _stop_async_postprocess(self, drain_timeout_s: float = 10.0) -> None:
        if self.postprocess_queue is None or self.postprocess_thread is None:
            return
        deadline = time.time() + max(0.0, float(drain_timeout_s))
        while getattr(self.postprocess_queue, "unfinished_tasks", 0) > 0 and time.time() < deadline:
            self.stats["postprocess_queue_depth"] = int(self.postprocess_queue.qsize())
            time.sleep(0.02)
        if getattr(self.postprocess_queue, "unfinished_tasks", 0) > 0:
            self.stats["postprocess_errors"] = int(self.stats.get("postprocess_errors", 0)) + 1
            self._postprocess_worker_error = "postprocess drain timeout"
        try:
            self.postprocess_queue.put_nowait(None)
        except queue.Full:
            try:
                self.postprocess_queue.get_nowait()
                self.postprocess_queue.task_done()
                self.postprocess_queue.put_nowait(None)
            except Exception:
                pass
        self.postprocess_thread.join(timeout=2.0)

    def _postprocess_worker_loop(self) -> None:
        assert self.postprocess_queue is not None
        while True:
            task = self.postprocess_queue.get()
            try:
                if task is None:
                    return
                self._process_postprocess_task(task)
            except Exception as exc:
                self.stats["postprocess_errors"] = int(self.stats.get("postprocess_errors", 0)) + 1
                self.stats["errors"] = int(self.stats.get("errors", 0)) + 1
                self._postprocess_worker_error = f"{type(exc).__name__}: {exc}"
                self._write_status(state="error", error=self._postprocess_worker_error)
                traceback.print_exc()
            finally:
                self.postprocess_queue.task_done()
                self.stats["postprocess_queue_depth"] = int(self.postprocess_queue.qsize())

    def _process_postprocess_task(self, task: PostprocessTask) -> None:
        post_t0 = time.time()
        if task.enqueue_wall_us:
            lag_ms = max(0.0, float(now_us() - int(task.enqueue_wall_us)) / 1000.0)
            _profile_add(task.micro_profile, "async_queue_lag_ms", lag_ms)
            self.stats["postprocess_lag_ms_last"] = round(float(lag_ms), 3)
            self.stats["postprocess_lag_ms_max"] = max(float(self.stats.get("postprocess_lag_ms_max", 0.0) or 0.0), round(float(lag_ms), 3))
        micro_profile = task.micro_profile
        alignment_audit = self._audit_input_alignment(task)
        if alignment_audit:
            visual_spread = alignment_audit.get("visual_mid_spread_ms")
            capture_spread = alignment_audit.get("capture_ts_spread_ms")
            if visual_spread is not None:
                _profile_add(micro_profile, "input_align_visual_mid_spread_ms", float(visual_spread))
            if capture_spread is not None:
                _profile_add(micro_profile, "input_align_capture_ts_spread_ms", float(capture_spread))
        records: List[Dict[str, Any]] = []
        person_counts: List[int] = []
        post_ms_for_meta = 0.0
        for item, result, shape in zip(task.items, task.results, task.frame_shapes):
            cam = str(item.get("camera"))
            extract_t0 = _perf_now()
            dets = _extract_dets_fast(result, self.args, shape, profile=micro_profile)
            _profile_add(micro_profile, "extract_dets_ms", _elapsed_ms(extract_t0))
            person_counts.append(len(dets))
            _profile_inc(micro_profile, "person_count_total", len(dets))
            infer_meta = {
                "batch_index": task.batch_index,
                "batch_size": len(task.items),
                "predict_ms": round(float(task.predict_ms), 3),
                "postprocess_ms": round(float(post_ms_for_meta), 3),
                "total_ms": round((time.time() - task.batch_t0_s) * 1000.0, 3),
            }
            record_t0 = _perf_now()
            rec = _record_for_cam(
                cam=cam,
                frame_item=item,
                frame_shape=shape,
                dets=dets,
                projector=self.projectors[cam],
                nadir_config=self.nadir_configs.get(cam, {}),
                args=self.args,
                infer_meta=infer_meta,
                profile=micro_profile,
            )
            _profile_add(micro_profile, "record_build_ms", _elapsed_ms(record_t0))
            records.append(rec)
        post_ms = (time.time() - post_t0) * 1000.0
        _profile_add(micro_profile, "postprocess_ms", float(post_ms))
        total_ms = (time.time() - task.batch_t0_s) * 1000.0
        legacy_write_t0 = _perf_now()
        for rec in records:
            rec["yolo_postprocess_batch_ms"] = round(float(post_ms), 3)
            rec["yolo_total_batch_ms"] = round(float(total_ms), 3)
            if bool(self.args.write_legacy_jsonl):
                self._writer_for(str(rec["camera"])).write(rec)
                self.stats["records_written"] = int(self.stats.get("records_written", 0)) + 1
        _profile_add(micro_profile, "legacy_jsonl_write_ms", _elapsed_ms(legacy_write_t0))
        _profile_inc(micro_profile, "records_count", len(records))
        self.stats["frames_published"] = int(self.stats.get("frames_published", 0)) + len(task.items)
        self.stats["postprocess_batches_published"] = int(self.stats.get("postprocess_batches_published", 0)) + 1
        self._update_polygon_contract_stats(records)
        last_batch = self._last_batch_summary(task.cam_ids, len(task.items), task.predict_ms, post_ms, total_ms, person_counts, task.collect_meta)
        if alignment_audit:
            last_batch["input_alignment_audit"] = alignment_audit
        latest_ms = self._write_latest_if_due(task.cam_ids, records, batch_index=task.batch_index)
        _profile_add(micro_profile, "latest_json_write_ms", latest_ms)
        pre_status_profile = _profile_round(micro_profile)
        last_batch["micro_profile_ms"] = pre_status_profile
        status_ms = self._write_status_if_due(state="running", last_batch=last_batch)
        _profile_add(micro_profile, "status_json_write_ms", status_ms)
        _profile_add(micro_profile, "batch_total_observed_ms", _elapsed_ms(task.observed_t0))
        last_batch["micro_profile_ms"] = self._remember_micro_profile(micro_profile)
        perf_row = {
            "wall_us": now_us(),
            "batch_index": task.batch_index,
            "batch_size": len(task.items),
            "group_size": task.group_size,
            "initial_batch_size": int(task.collect_meta.get("initial_batch_size", len(task.items))),
            "collect_wait_ms": float(task.collect_meta.get("collect_wait_ms", 0.0)),
            "collect_added_frames": int(task.collect_meta.get("collect_added_frames", 0)),
            "predict_ms": round(float(task.predict_ms), 3),
            "postprocess_ms": round(float(post_ms), 3),
            "total_ms": round(float(total_ms), 3),
            "predict_fps_total": round(1000.0 * len(task.items) / max(1e-6, task.predict_ms), 3),
            "total_fps_total": round(1000.0 * len(task.items) / max(1e-6, total_ms), 3),
            "backend": self.args.postprocess_backend,
            "model": str(self.model_path),
            "engine": str(self.engine_path),
            "device": str(self.args.device),
            "imgsz": int(self.args.imgsz),
            "half": int(bool(self.args.half)),
            "retina_masks": int(bool(self.args.retina_masks)),
            "cam_ids": "|".join(task.cam_ids),
            "sync_indices": "|".join(str(x.get("sync_index", "")) for x in task.items),
            "frame_ids_local": "|".join(str(x.get("frame_id", "")) for x in task.items),
            "person_counts": "|".join(str(x) for x in person_counts),
        }
        if alignment_audit:
            input_buffer_meta = alignment_audit.get("input_buffer", {}) if isinstance(alignment_audit.get("input_buffer"), dict) else {}
            perf_row.update({
                "input_align_enabled": 1,
                "input_align_full_batch": int(bool(alignment_audit.get("full_batch", False))),
                "input_align_same_sync": int(bool(alignment_audit.get("same_sync", False))),
                "input_align_sync_spread": "" if alignment_audit.get("sync_spread") is None else int(alignment_audit.get("sync_spread") or 0),
                "input_align_visual_mid_complete": int(bool(alignment_audit.get("visual_mid_complete", False))),
                "input_align_visual_mid_aligned": int(bool(alignment_audit.get("visual_mid_aligned", False))),
                "input_align_meta_match": int(bool(alignment_audit.get("meta_frame_match", False))),
                "input_align_current_aligned_full": int(bool(alignment_audit.get("current_aligned_full", False))),
                "input_align_would_full_seen": int(bool(alignment_audit.get("would_full_seen", False))),
                "input_align_would_full_within_wait": int(bool(alignment_audit.get("would_full_within_wait", False))),
                "input_align_would_wait_ms": "" if alignment_audit.get("would_wait_ms") is None else alignment_audit.get("would_wait_ms"),
                "input_align_visual_mid_spread_ms": "" if alignment_audit.get("visual_mid_spread_ms") is None else alignment_audit.get("visual_mid_spread_ms"),
                "input_align_capture_ts_spread_ms": "" if alignment_audit.get("capture_ts_spread_ms") is None else alignment_audit.get("capture_ts_spread_ms"),
                "input_align_exposure_spread_us": "" if alignment_audit.get("exposure_spread_us") is None else int(alignment_audit.get("exposure_spread_us") or 0),
                "input_align_sync_indices": _pipe(alignment_audit.get("sync_indices", []) or []),
                "input_align_cam_ids": _pipe(alignment_audit.get("cam_ids", []) or []),
                "input_align_frame_ids": _pipe(alignment_audit.get("frame_ids", []) or []),
                "input_align_exposure_us": _pipe(alignment_audit.get("exposure_us", []) or []),
                "input_align_exposure_source": _pipe(alignment_audit.get("exposure_source", []) or []),
                "input_align_visual_mid_us": _pipe(alignment_audit.get("visual_mid_us", []) or []),
                "input_align_exposure_end_us": _pipe(alignment_audit.get("exposure_end_us", []) or []),
                "input_align_exposure_end_spread_ms": "" if alignment_audit.get("exposure_end_spread_ms") is None else alignment_audit.get("exposure_end_spread_ms"),
                "input_align_missing_cams": _pipe(alignment_audit.get("missing_cams", []) or []),
                "input_align_late_cam": alignment_audit.get("late_cam", ""),
                "input_buffer_enabled": int(bool(input_buffer_meta.get("enabled", False))),
                "input_buffer_decision": input_buffer_meta.get("decision", ""),
                "input_buffer_fallback_reason": input_buffer_meta.get("fallback_reason", ""),
                "input_buffer_wait_ms": input_buffer_meta.get("wait_ms", ""),
                "input_buffer_pending_frames": input_buffer_meta.get("pending_frames", ""),
            })
            self.exposure_audit_writer.write({
                "wall_us": perf_row["wall_us"],
                "batch_index": task.batch_index,
                "group_cams": _pipe(alignment_audit.get("group_cams", []) or []),
                "group_size": task.group_size,
                "batch_size": len(task.items),
                "full_batch": int(bool(alignment_audit.get("full_batch", False))),
                "current_aligned_full": int(bool(alignment_audit.get("current_aligned_full", False))),
                "same_sync": int(bool(alignment_audit.get("same_sync", False))),
                "visual_mid_aligned": int(bool(alignment_audit.get("visual_mid_aligned", False))),
                "sync_indices": _pipe(alignment_audit.get("sync_indices", []) or []),
                "cam_ids": _pipe(alignment_audit.get("cam_ids", []) or []),
                "frame_ids_local": _pipe(alignment_audit.get("frame_ids", []) or []),
                "missing_cams": _pipe(alignment_audit.get("missing_cams", []) or []),
                "late_cam": alignment_audit.get("late_cam", ""),
                "exposure_us": _pipe(alignment_audit.get("exposure_us", []) or []),
                "exposure_source": _pipe(alignment_audit.get("exposure_source", []) or []),
                "exposure_start_us": _pipe(alignment_audit.get("exposure_start_us", []) or []),
                "visual_mid_us": _pipe(alignment_audit.get("visual_mid_us", []) or []),
                "exposure_end_us": _pipe(alignment_audit.get("exposure_end_us", []) or []),
                "exposure_spread_us": "" if alignment_audit.get("exposure_spread_us") is None else alignment_audit.get("exposure_spread_us"),
                "visual_mid_spread_ms": "" if alignment_audit.get("visual_mid_spread_ms") is None else alignment_audit.get("visual_mid_spread_ms"),
                "exposure_end_spread_ms": "" if alignment_audit.get("exposure_end_spread_ms") is None else alignment_audit.get("exposure_end_spread_ms"),
                "capture_ts_spread_ms": "" if alignment_audit.get("capture_ts_spread_ms") is None else alignment_audit.get("capture_ts_spread_ms"),
                "meta_frame_match": int(bool(alignment_audit.get("meta_frame_match", False))),
                "meta_frame_mismatch": int(bool(alignment_audit.get("meta_frame_mismatch", False))),
                "would_full_seen": int(bool(alignment_audit.get("would_full_seen", False))),
                "would_full_within_wait": int(bool(alignment_audit.get("would_full_within_wait", False))),
                "would_wait_ms": "" if alignment_audit.get("would_wait_ms") is None else alignment_audit.get("would_wait_ms"),
                "would_visual_mid_spread_ms": "" if alignment_audit.get("would_visual_mid_spread_ms") is None else alignment_audit.get("would_visual_mid_spread_ms"),
                "input_buffer_enabled": int(bool(input_buffer_meta.get("enabled", False))),
                "input_buffer_decision": input_buffer_meta.get("decision", ""),
                "input_buffer_fallback_reason": input_buffer_meta.get("fallback_reason", ""),
                "input_buffer_wait_ms": input_buffer_meta.get("wait_ms", ""),
                "input_buffer_pending_frames": input_buffer_meta.get("pending_frames", ""),
            })
        perf_row.update(self.last_micro_profile_ms)
        self.perf_writer.write(perf_row)

    def _update_polygon_contract_stats(self, records: List[Dict[str, Any]]) -> None:
        records_total = len(records)
        records_with_persons = 0
        records_with_polygon = 0
        persons_total = 0
        persons_with_polygon = 0
        for rec in records:
            persons = rec.get("persons", []) if isinstance(rec, dict) else []
            if not isinstance(persons, list):
                persons = []
            if persons:
                records_with_persons += 1
            record_has_polygon = False
            for person in persons:
                if not isinstance(person, dict):
                    continue
                persons_total += 1
                has_polygon = person_has_usable_polygon(person)
                if has_polygon:
                    persons_with_polygon += 1
                    record_has_polygon = True
            if record_has_polygon:
                records_with_polygon += 1
        self.stats["published_records_total"] = int(self.stats.get("published_records_total", 0)) + records_total
        self.stats["published_records_with_persons"] = int(self.stats.get("published_records_with_persons", 0)) + records_with_persons
        self.stats["published_records_with_polygon"] = int(self.stats.get("published_records_with_polygon", 0)) + records_with_polygon
        self.stats["published_persons_total"] = int(self.stats.get("published_persons_total", 0)) + persons_total
        self.stats["published_persons_with_polygon"] = int(self.stats.get("published_persons_with_polygon", 0)) + persons_with_polygon
        persons_all = int(self.stats.get("published_persons_total", 0) or 0)
        persons_polygon_all = int(self.stats.get("published_persons_with_polygon", 0) or 0)
        records_with_persons_all = int(self.stats.get("published_records_with_persons", 0) or 0)
        records_with_polygon_all = int(self.stats.get("published_records_with_polygon", 0) or 0)
        self.stats["published_persons_without_polygon"] = max(0, persons_all - persons_polygon_all)
        self.stats["published_records_with_polygon_ratio"] = round(
            1.0 if records_with_persons_all == 0 else records_with_polygon_all / float(records_with_persons_all),
            4,
        )
        self.stats["published_persons_with_polygon_ratio"] = round(
            1.0 if persons_all == 0 else persons_polygon_all / float(persons_all),
            4,
        )

    def _write_latest_if_due(self, cam_ids: List[str], records: List[Dict[str, Any]], batch_index: Optional[int] = None) -> float:
        if not self._interval_due("_last_latest_write_s", self.latest_interval_s):
            return 0.0
        write_t0 = _perf_now()
        primary_mode = not bool(self.shadow)
        latest = {
            "msg_type": "global_yolo_latest",
            "wall_us": now_us(),
            "shadow_mode": bool(self.shadow),
            "global_yolo_primary": bool(primary_mode),
            "inference_mode": "primary" if primary_mode else "shadow",
            "batch_index": int(self.batch_index if batch_index is None else batch_index),
            "cams": cam_ids,
            "records": records,
        }
        _atomic_write_json(self.latest_json_path, latest)
        self.stats["latest_writes"] = int(self.stats.get("latest_writes", 0)) + 1
        return _elapsed_ms(write_t0)

    def _last_batch_summary(self, cam_ids: List[str], batch_size: int, predict_ms: float, post_ms: float, total_ms: float, person_counts: List[int], collect_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        collect_meta = collect_meta or {}
        return {
            "cams": cam_ids,
            "batch_size": int(batch_size),
            "group_size": int(collect_meta.get("group_size") or batch_size),
            "initial_batch_size": int(collect_meta.get("initial_batch_size", batch_size)),
            "collect_wait_ms": round(float(collect_meta.get("collect_wait_ms", 0.0)), 3),
            "collect_added_frames": int(collect_meta.get("collect_added_frames", 0)),
            "predict_ms": round(float(predict_ms), 3),
            "postprocess_ms": round(float(post_ms), 3),
            "total_ms": round(float(total_ms), 3),
            "person_counts": person_counts,
        }

    def _interval_due(self, attr: str, interval_s: float) -> bool:
        now = time.time()
        last = float(getattr(self, attr, 0.0) or 0.0)
        if interval_s <= 0.0 or last <= 0.0 or now - last >= interval_s:
            setattr(self, attr, now)
            return True
        return False

    def _write_status_if_due(self, state: str, last_batch: Optional[Dict[str, Any]] = None, error: str = "") -> float:
        if self._interval_due("_last_status_write_s", self.status_interval_s):
            return self._write_status(state=state, last_batch=last_batch, error=error)
        return 0.0

    def _write_status(self, state: str, last_batch: Optional[Dict[str, Any]] = None, error: str = "") -> float:
        write_t0 = _perf_now()
        self.stats["status_writes"] = int(self.stats.get("status_writes", 0)) + 1
        if self.postprocess_queue is not None:
            self.stats["postprocess_queue_depth"] = int(self.postprocess_queue.qsize())
        primary_mode = not bool(self.shadow)
        tensorrt_enabled = bool(str(self.engine_path or "").strip())
        mask_postprocess_enabled = bool(self.args.retina_masks or self.args.include_mask_polygons)
        postprocess_draw_enabled = False
        wall_us = now_us()
        start_wall_us = int(self.stats.get("start_wall_us", wall_us) or wall_us)
        elapsed_s = max(1e-6, float(wall_us - start_wall_us) / 1_000_000.0)
        inferred_fps = float(self.stats.get("frames_inferred", 0) or 0) / elapsed_s
        published_fps = float(self.stats.get("frames_published", 0) or 0) / elapsed_s
        polygon_contract = summarize_polygon_contract(
            persons_total=int(self.stats.get("published_persons_total", 0) or 0),
            persons_with_polygon=int(self.stats.get("published_persons_with_polygon", 0) or 0),
            polygon_output_enabled=bool(self.args.include_mask_polygons),
            dropped_batches=int(self.stats.get("postprocess_dropped_batches", 0) or 0),
        )
        polygon_contract_ok = bool(polygon_contract["polygon_contract_ok"])
        async_record_postprocess = {
            "enabled": bool(self.async_record_postprocess),
            "queue_size": int(getattr(self.args, "async_record_postprocess_queue_size", 0) or 0),
            "queue_policy": str(getattr(self.args, "async_record_postprocess_policy", "block")),
            "queue_depth": int(self.stats.get("postprocess_queue_depth", 0) or 0),
            "queue_max_depth": int(self.stats.get("postprocess_queue_max_depth", 0) or 0),
            "lag_ms_last": float(self.stats.get("postprocess_lag_ms_last", 0.0) or 0.0),
            "lag_ms_max": float(self.stats.get("postprocess_lag_ms_max", 0.0) or 0.0),
            "backpressure_ms_total": float(self.stats.get("postprocess_backpressure_ms_total", 0.0) or 0.0),
            "queue_full_waits": int(self.stats.get("postprocess_queue_full_waits", 0) or 0),
            "dropped_batches": int(self.stats.get("postprocess_dropped_batches", 0) or 0),
            "errors": int(self.stats.get("postprocess_errors", 0) or 0),
            "worker_error": str(self._postprocess_worker_error or ""),
            "polygon_contract_ok": bool(polygon_contract_ok),
            "polygon_contract_state": str(polygon_contract["polygon_contract_state"]),
            "polygon_output_enabled": bool(polygon_contract["polygon_output_enabled"]),
            "published_persons_without_polygon": int(polygon_contract["published_persons_without_polygon"]),
            "published_records_with_polygon_ratio": float(self.stats.get("published_records_with_polygon_ratio", 1.0) or 0.0),
            "published_persons_with_polygon_ratio": float(self.stats.get("published_persons_with_polygon_ratio", 1.0) or 0.0),
        }
        input_alignment_audit = self._input_alignment_status()
        input_buffer_status = {
            "enabled": bool(self.input_buffer_enabled),
            "size": int(self.input_buffer_size),
            "max_wait_ms": round(float(self.input_buffer_max_wait_ms), 3),
            "max_spread_ms": round(float(self.input_buffer_max_spread_ms), 3),
            "exposure_end_barrier": bool(self.input_buffer_exposure_end_barrier),
            "pending_frames": int(self._input_buffer_pending_count(self.cams)),
            "batches": int(self.stats.get("input_buffer_batches", 0) or 0),
            "aligned_full_batches": int(self.stats.get("input_buffer_aligned_full_batches", 0) or 0),
            "fallback_batches": int(self.stats.get("input_buffer_fallback_batches", 0) or 0),
            "sync_partial_batches": int(self.stats.get("input_buffer_sync_partial_batches", 0) or 0),
            "latest_partial_batches": int(self.stats.get("input_buffer_latest_partial_batches", 0) or 0),
            "visual_misaligned_full_batches": int(self.stats.get("input_buffer_visual_misaligned_full_batches", 0) or 0),
            "aligned_full_ratio": _ratio(self.stats.get("input_buffer_aligned_full_batches"), self.stats.get("input_buffer_batches")),
            "fallback_ratio": _ratio(self.stats.get("input_buffer_fallback_batches"), self.stats.get("input_buffer_batches")),
            "wait_ms_avg": round(float(self.stats.get("input_buffer_wait_ms_total", 0.0) or 0.0) / max(1, int(self.stats.get("input_buffer_batches", 0) or 0)), 3),
            "wait_ms_max": round(float(self.stats.get("input_buffer_wait_ms_max", 0.0) or 0.0), 3),
        }
        status = {
            "msg_type": "global_yolo_status",
            "state": state,
            "pid": os.getpid(),
            "wall_us": wall_us,
            "shadow_mode": bool(self.shadow),
            "global_yolo_primary": bool(primary_mode),
            "inference_mode": "primary" if primary_mode else "shadow",
            "legacy_yolo_skipped": bool(primary_mode),
            "merge_publisher": False,
            "cams": self.cams,
            "groups": self.groups,
            "output_root": str(self.output_root),
            "write_legacy_jsonl": bool(self.args.write_legacy_jsonl),
            "legacy_jsonl_compat_writer": bool(self.args.write_legacy_jsonl),
            "postprocess_backend": str(self.args.postprocess_backend),
            "postprocess_draw_enabled": bool(postprocess_draw_enabled),
            "cpu_draw_postprocess": False,
            "mask_postprocess_enabled": bool(mask_postprocess_enabled),
            "tensor_rt_enabled": bool(tensorrt_enabled),
            "tensorrt_enabled": bool(tensorrt_enabled),
            "fps": round(float(published_fps), 3),
            "inferred_fps_total": round(float(inferred_fps), 3),
            "published_fps_total": round(float(published_fps), 3),
            "async_record_postprocess_enabled": bool(self.async_record_postprocess),
            "postprocess_queue_depth": int(async_record_postprocess["queue_depth"]),
            "postprocess_lag_ms": float(async_record_postprocess["lag_ms_last"]),
            "postprocess_backpressure_ms": float(async_record_postprocess["backpressure_ms_total"]),
            "postprocess_dropped_batches": int(async_record_postprocess["dropped_batches"]),
            "polygon_contract_ok": bool(polygon_contract_ok),
            "polygon_contract_state": str(polygon_contract["polygon_contract_state"]),
            "polygon_output_enabled": bool(polygon_contract["polygon_output_enabled"]),
            "published_persons_without_polygon": int(polygon_contract["published_persons_without_polygon"]),
            "published_records_with_polygon_ratio": float(async_record_postprocess["published_records_with_polygon_ratio"]),
            "published_persons_with_polygon_ratio": float(async_record_postprocess["published_persons_with_polygon_ratio"]),
            "input_alignment_audit": input_alignment_audit,
            "input_buffer": input_buffer_status,
            "summary": {
                "inferred_fps_total": round(float(inferred_fps), 3),
                "published_fps_total": round(float(published_fps), 3),
                "async_record_postprocess_enabled": bool(self.async_record_postprocess),
                "postprocess_queue_depth": int(async_record_postprocess["queue_depth"]),
                "postprocess_queue_max_depth": int(async_record_postprocess["queue_max_depth"]),
                "postprocess_dropped_batches": int(async_record_postprocess["dropped_batches"]),
                "polygon_contract_ok": bool(polygon_contract_ok),
                "polygon_contract_state": str(polygon_contract["polygon_contract_state"]),
                "polygon_output_enabled": bool(polygon_contract["polygon_output_enabled"]),
                "published_persons_without_polygon": int(polygon_contract["published_persons_without_polygon"]),
                "published_persons_with_polygon_ratio": float(async_record_postprocess["published_persons_with_polygon_ratio"]),
                "input_alignment_audit_enabled": bool(input_alignment_audit.get("enabled", False)),
                "input_alignment_full_batch_ratio": input_alignment_audit.get("full_batch_ratio"),
                "input_alignment_same_sync_batch_ratio": input_alignment_audit.get("same_sync_batch_ratio"),
                "input_alignment_current_aligned_full_batch_ratio": input_alignment_audit.get("current_aligned_full_batch_ratio"),
                "input_alignment_sync_full_sets_within_wait_ratio": input_alignment_audit.get("sync_full_sets_within_wait_ratio"),
                "input_buffer_enabled": bool(input_buffer_status.get("enabled", False)),
                "input_buffer_aligned_full_ratio": input_buffer_status.get("aligned_full_ratio"),
                "input_buffer_fallback_ratio": input_buffer_status.get("fallback_ratio"),
            },
            "footpoint_mode": str(self.args.footpoint_mode),
            "anchor_mode": str(getattr(self.args, "anchor_mode", "auto_topdown_center")),
            "anchor_height_correction_enable": bool(getattr(self.args, "anchor_height_correction_enable", True)),
            "anchor_height_correction_k": float(getattr(self.args, "anchor_height_correction_k", 0.08)),
            "target_fps": float(self.args.target_fps),
            "diagnostics": {
                "status_interval_ms": float(self.args.status_interval_ms),
                "latest_interval_ms": float(self.args.latest_interval_ms),
                "perf_flush_interval_ms": float(self.args.perf_flush_interval_ms),
                "batch_collect_wait_ms": float(self.args.batch_collect_wait_ms),
                "batch_collect_retry_sleep_ms": float(self.args.batch_collect_retry_sleep_ms),
                "input_alignment_audit_enabled": bool(self.input_alignment_audit_enabled),
                "input_alignment_audit_max_wait_ms": float(self.input_alignment_audit_max_wait_ms),
                "input_alignment_audit_max_spread_ms": float(self.input_alignment_audit_max_spread_ms),
                "input_alignment_audit_history": int(self.input_alignment_audit_history),
                "input_buffer": input_buffer_status,
                "postprocess_draw_enabled": bool(postprocess_draw_enabled),
                "cpu_draw_postprocess": False,
                "mask_postprocess_enabled": bool(mask_postprocess_enabled),
                "retina_masks": bool(self.args.retina_masks),
                "include_mask_polygons": bool(self.args.include_mask_polygons),
                "anchor_mode": str(getattr(self.args, "anchor_mode", "auto_topdown_center")),
                "anchor_height_correction_enable": bool(getattr(self.args, "anchor_height_correction_enable", True)),
                "anchor_height_correction_k": float(getattr(self.args, "anchor_height_correction_k", 0.08)),
                "postprocess_gpu_fast": str(self.args.postprocess_backend) in {"gpu_fast", "torch_fast"},
                "tensor_rt_enabled": bool(tensorrt_enabled),
                "micro_profile_enabled": True,
                "micro_profile_window_size": int(getattr(self.args, "micro_profile_window", 300) or 300),
                "async_record_postprocess": async_record_postprocess,
            },
            "async_record_postprocess": async_record_postprocess,
            "micro_profile": {
                "enabled": True,
                "fields_ms": list(MICRO_PROFILE_MS_FIELDS),
                "fields_count": list(MICRO_PROFILE_COUNT_FIELDS),
                "last_ms": dict(self.last_micro_profile_ms),
                "window": self._micro_profile_window_summary(),
            },
            "model": {
                "load_path": str(self.model_load_path),
                "model_path": str(self.model_path),
                "engine_path": str(self.engine_path),
                "source": str(self.model_source),
                "exists": bool(self.model_load_exists),
                "tensor_rt_enabled": bool(tensorrt_enabled),
            },
            "stats": dict(self.stats),
            "last_batch": last_batch or {},
            "reader_errors": {cam: r.last_error for cam, r in self.readers.items() if r.last_error},
            "projectors": {cam: p.status() for cam, p in self.projectors.items()},
            "anchor_nadir": self.nadir_configs,
            "error": error,
        }
        _atomic_write_json(self.status_path, status)
        return _elapsed_ms(write_t0)

    def run(self) -> int:
        self._write_status(state="starting")
        interval = 1.0 / max(1e-6, float(self.args.target_fps))
        group_index = 0
        try:
            while True:
                loop_t0 = time.time()
                group = self.groups[group_index % len(self.groups)]
                group_index += 1
                items, collect_meta = self._read_group(group)
                if items:
                    self._process_batch(items, collect_meta)
                else:
                    self.stats["empty_polls"] = int(self.stats.get("empty_polls", 0)) + 1
                    self._write_status_if_due(state="waiting_frames")
                dt = time.time() - loop_t0
                sleep_s = max(0.0, interval - dt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            self._write_status(state="stopped")
            return 0
        except Exception as exc:
            self.stats["errors"] = int(self.stats.get("errors", 0)) + 1
            err = f"{type(exc).__name__}: {exc}"
            self._write_status(state="error", error=err)
            traceback.print_exc()
            return 1
        finally:
            self._stop_async_postprocess()
            for w in self.writers.values():
                w.close()
            self.perf_writer.close()
            self.exposure_audit_writer.close()
            for r in self.readers.values():
                r.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Global 8-camera YOLO inference server")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", required=True)
    ap.add_argument("--cams", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--mvs-shm-template", default="mvs_latest_{camera}")
    ap.add_argument("--mvs-meta-template", default="{root}/mvs_latest_{camera}.json")
    ap.add_argument("--calib-template", default="calib_true/field_calib_camera{camera}.json")
    ap.add_argument("--model", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.18)
    ap.add_argument("--iou", type=float, default=0.78)
    ap.add_argument("--person-class-id", type=int, default=0)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--retina-masks", action="store_true", default=False)
    ap.add_argument("--target-fps", type=float, default=80.0)
    ap.add_argument("--microbatch-groups", default="A,C,E,G:B,D,F,H")
    ap.add_argument("--batch-collect-wait-ms", type=float, default=0.0, help="After the first fresh frame in a group, wait briefly for missing cameras before inference.")
    ap.add_argument("--batch-collect-retry-sleep-ms", type=float, default=0.5, help="Sleep interval while collecting missing frames inside --batch-collect-wait-ms.")
    ap.add_argument("--postprocess-backend", choices=["gpu_fast", "torch_fast", "cpu_legacy"], default="gpu_fast")
    ap.add_argument("--footpoint-mode", choices=["center", "bottom", "top"], default="center")
    ap.add_argument("--anchor-mode", choices=["bbox_center", "bbox_bottom", "mask_robust_center", "mask_eroded_centroid", "auto_topdown_center"], default="auto_topdown_center")
    ap.add_argument("--anchor-height-correction-enable", action="store_true", default=True)
    ap.add_argument("--no-anchor-height-correction-enable", dest="anchor_height_correction_enable", action="store_false")
    ap.add_argument("--anchor-height-correction-k", type=float, default=0.08)
    ap.add_argument("--anchor-nadir-pixel-template", default="calib_true/nadir_camera{camera}.json")
    ap.add_argument("--include-mask-polygons", action="store_true", default=False)
    ap.add_argument("--mask-polygon-max-points", type=int, default=96)
    ap.add_argument("--write-legacy-jsonl", action="store_true", default=True)
    ap.add_argument("--no-write-legacy-jsonl", dest="write_legacy_jsonl", action="store_false")
    ap.add_argument("--truncate-jsonl-on-start", action="store_true", default=True)
    ap.add_argument("--append-jsonl", dest="truncate_jsonl_on_start", action="store_false")
    ap.add_argument("--shadow-mode", action="store_true", default=False)
    ap.add_argument("--shadow-output-dir", default="sync_ipc/global_yolo_shadow")
    ap.add_argument("--latest-json", default="sync_ipc/global_yolo_latest.json")
    ap.add_argument("--status-json", default="sync_ipc/global_yolo_status.json")
    ap.add_argument("--perf-csv", default="sync_ipc/global_yolo_infer_perf.csv")
    ap.add_argument("--exposure-audit-csv", default="sync_ipc/global_yolo_exposure_audit.csv")
    ap.add_argument("--latest-interval-ms", type=float, default=100.0, help="Minimum interval for diagnostic global_yolo_latest.json writes. 0 writes every batch.")
    ap.add_argument("--status-interval-ms", type=float, default=1000.0, help="Minimum interval for global_yolo_status.json writes. 0 writes every batch.")
    ap.add_argument("--perf-flush-interval-ms", type=float, default=1000.0, help="Minimum flush interval for global_yolo_infer_perf.csv. 0 flushes every row.")
    ap.add_argument("--micro-profile-window", type=int, default=300, help="Rolling batch count used for Global YOLO micro-stage timing summaries in status JSON.")
    ap.add_argument("--async-record-postprocess", action="store_true", default=False, help="Move polygon/anchor/record publication off the main predict loop.")
    ap.add_argument("--async-record-postprocess-queue-size", type=int, default=2, help="Queued predicted batches before the main loop blocks.")
    ap.add_argument("--async-record-postprocess-policy", choices=["block"], default="block", help="Backpressure policy. V24-Z keeps polygon output lossless, so only block is allowed.")
    ap.add_argument("--input-alignment-audit", action="store_true", default=False, help="V24-Z3a audit only: measure visible-frame sync/exposure alignment without changing batching.")
    ap.add_argument("--input-alignment-audit-max-wait-ms", type=float, default=8.0, help="Audit threshold for how long a full same-sync frame set may take to become available.")
    ap.add_argument("--input-alignment-audit-max-spread-ms", type=float, default=6.0, help="Audit threshold for visual exposure-midpoint spread inside a batch or full sync set.")
    ap.add_argument("--input-alignment-audit-history", type=int, default=512, help="Recent sync-index entries kept for audit-only full-set reconstruction.")
    ap.add_argument("--input-buffer-enable", action="store_true", default=False, help="V24-Z4: align YOLO input from a per-camera ring buffer by sync_index and exposure midpoint.")
    ap.add_argument("--no-input-buffer", dest="input_buffer_enable", action="store_false")
    ap.add_argument("--input-buffer-size", type=int, default=12, help="Per-camera frame count retained in the YOLO input ring buffer.")
    ap.add_argument("--input-buffer-max-wait-ms", type=float, default=30.0, help="Maximum time to wait for an aligned full batch before publishing a marked fallback.")
    ap.add_argument("--input-buffer-max-spread-ms", type=float, default=6.0, help="Maximum exposure-midpoint spread for a full aligned batch.")
    ap.add_argument("--input-buffer-exposure-end-barrier", action="store_true", default=True, help="Do not publish an aligned candidate before its latest exposure_end_us.")
    ap.add_argument("--no-input-buffer-exposure-end-barrier", dest="input_buffer_exposure_end_barrier", action="store_false")
    return ap


def _apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    project_root = Path(args.project_root).expanduser().resolve()
    cfg_path = _resolve_project_path(project_root, args.config)
    try:
        cfg = _load_mvs_config(cfg_path)
    except Exception:
        return args
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    discovered_model, _ = _find_model_candidate(cfg)
    if not _nonempty_str(getattr(args, "model", "")) and discovered_model:
        setattr(args, "model", discovered_model)
    for attr, key in [
        ("model", "model"),
        ("device", "device"),
        ("imgsz", "imgsz"),
        ("conf", "conf"),
        ("iou", "iou"),
        ("person_class_id", "person_class_id"),
        ("max_det", "max_det"),
        ("half", "half"),
        ("retina_masks", "retina_masks"),
    ]:
        if key in model_cfg:
            current = getattr(args, attr)
            default = build_arg_parser().get_default(attr)
            if current == default or current in ("", None):
                setattr(args, attr, model_cfg.get(key))
    sync_cfg = cfg.get("sync", {}) if isinstance(cfg.get("sync"), dict) else {}
    if sync_cfg.get("root") and args.sync_root == "sync_ipc":
        setattr(args, "sync_root", str(sync_cfg.get("root")))
    if sync_cfg.get("mvs_frame_pub_name_template") and args.mvs_shm_template == "mvs_latest_{camera}":
        setattr(args, "mvs_shm_template", str(sync_cfg.get("mvs_frame_pub_name_template")))
    if sync_cfg.get("mvs_frame_pub_meta_template") and args.mvs_meta_template == "{root}/mvs_latest_{camera}.json":
        setattr(args, "mvs_meta_template", str(sync_cfg.get("mvs_frame_pub_meta_template")))
    return args


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()
    args = _apply_config_defaults(args)
    server = GlobalYoloServer(args)
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
