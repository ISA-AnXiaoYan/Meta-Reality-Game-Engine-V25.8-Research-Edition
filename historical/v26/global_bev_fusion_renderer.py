#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent GPU BEV fusion process for the 8-camera field view.

This process is deliberately preview/consumer-only.  It reads MVS raw frame
ring buffers, projects a global field plane into each camera in a fragment
shader, blends cameras with center-feather weights, and publishes the latest
BEV frame to shared memory for UE/recording/downstream consumers.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import ctypes
import json
import math
import mmap
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from monotonic_rate_deadline import rate_deadline_due

import numpy as np

try:
    import cv2 as _cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV is optional for this display-only path.
    _cv2 = None

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QSurfaceFormat
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import QApplication
except Exception as exc:  # pragma: no cover
    print(f"[global_bev][ERROR] PySide6 required: {exc}", flush=True)
    raise

try:
    from OpenGL import GL
    from OpenGL.GL import shaders
except Exception as exc:  # pragma: no cover
    print(f"[global_bev][ERROR] PyOpenGL required: {exc}", flush=True)
    raise

from preview_gpu_shm import RawPreviewFrameReader, RAW_PIXEL_BAYER8


GBEV_MAGIC = 0x47424556  # GBEV
GBEV_VERSION = 1
GBEV_HEADER_U64_COUNT = 64
GBEV_HEADER_BYTES = GBEV_HEADER_U64_COUNT * 8
GBEV_CHANNELS = 4
_SF_MAGIC = 0x46524D32
_SF_HEADER_U32_COUNT = 32
_SF_HEADER_BYTES = _SF_HEADER_U32_COUNT * 4
_SF_ACTIVE_SLOT = 8
_SF_SLOT_FIELDS = [(10, 11, 12, 13), (14, 15, 16, 17)]
_SPARSE_MAGIC = 0x455653503031
_SPARSE_HEADER_U64_COUNT = 64
_SPARSE_HEADER_BYTES = _SPARSE_HEADER_U64_COUNT * 8
_SPARSE_POINT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("count", np.uint16),
        ("intensity", np.uint8),
        ("dt_center_us", np.int32),
        ("polarity", np.uint8),
        ("reserved", np.uint8, (4,)),
    ],
    align=False,
)
DEBUG_MODE_CODES = {
    "warp_color": 0,
    "warp_gray": 1,
    "coverage": 2,
    "raw_mosaic": 3,
    "weight_heatmap": 4,
    "cam_index": 5,
    "uv_heatmap": 6,
}
DEBUG_MODE_ORDER = [
    "warp_color",
    "warp_gray",
    "cam_index",
    "coverage",
    "weight_heatmap",
    "uv_heatmap",
    "raw_mosaic",
]
TRANSFORM_MODE_ORDER = [
    "normal",
    "flip_x",
    "flip_y",
    "flip_xy",
    "swap_xy",
    "swap_xy_flip_x",
    "swap_xy_flip_y",
    "swap_xy_flip_xy",
]
TRANSFORM_MODE_CODES = {name: idx for idx, name in enumerate(TRANSFORM_MODE_ORDER)}
DEFAULT_EVENT_LAYER_HOMOGRAPHY_CONFIGS = (
    "ids_8cam_fusion_config_627_runtime.json",
    "ids_8cam_fusion_config.json",
)
DEFAULT_EVENT_LAYER_EVENT_SIZE = (1280, 720)
PERSON_OVERLAY_COORD_MODE_ORDER = [
    "field_xy",
    "raw",
    "neg_y",
    "field_height_plus_y",
    "invert_y",
    "flip_x",
    "flip_xy",
    "swap_xy",
    "swap_xy_neg_y",
]
EVENT_BULLET_OVERLAY_TRANSFORM_MODE_ORDER = [
    "normal",
    "field_xy",
    "raw",
    "none",
    "neg_y",
    "field_height_plus_y",
    "invert_y",
    "flip_x",
    "flip_y",
    "flip_xy",
    "swap_xy",
    "swap_xy_neg_y",
]
EVENT_LAYER_COLOR_MODE_ORDER = ["camera", "red", "yellow"]
EVENT_LAYER_COLOR_MODE_CODES = {name: idx for idx, name in enumerate(EVENT_LAYER_COLOR_MODE_ORDER)}
EVENT_LAYER_SYNC_MODE_ORDER = ["latest", "latest_guarded", "mvs_tick_guarded", "drop_stale", "fade"]
ALIGNMENT_MODE_ORDER = ["current", "shadow", "active"]


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _iso_from_us(ts_us: int) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(ts_us) / 1_000_000.0))
    except Exception:
        return ""


def _join_u64_u32(lo: int, hi: int) -> int:
    return (int(hi) << 32) | int(lo)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


EVENT_LAYER_SYNC_TIMESTAMP_POLICY = "event_window_end_us"


def _event_layer_sync_timestamp_us(meta: Dict[str, Any], fallback: int = 0) -> int:
    """Display sync for accumulated event windows is anchored at window end."""
    if not isinstance(meta, dict):
        return int(fallback)
    return _safe_int(
        meta.get("event_window_end_us", meta.get("event_end_ts_us", meta.get("event_ts_us"))),
        _safe_int(meta.get("timestamp_us"), int(fallback)),
    )


def _dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on", "y"):
        return True
    if s in ("0", "false", "no", "off", "n"):
        return False
    return default


def _dilate_event_layer_frame(frame: np.ndarray, radius_px: float, max_pixels: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sparse upload-side dilation for the display-only BEV event layer."""
    arr = np.asarray(frame)
    if arr.size <= 0:
        return arr, {
            "backend": "off",
            "requested_px": float(radius_px),
            "effective_px": 0.0,
            "src_nonzero_pixels": 0,
            "display_nonzero_pixels": 0,
            "skipped_dense": False,
        }
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    arr = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
    radius = max(0, min(64, int(round(float(radius_px or 0.0)))))
    if arr.ndim == 2:
        luma = arr
    elif arr.ndim == 3 and arr.shape[2] > 0:
        luma = np.max(arr[:, :, : min(3, int(arr.shape[2]))], axis=2)
    else:
        luma = np.asarray(arr).reshape(arr.shape[0], arr.shape[1])
    src_nonzero = int(np.count_nonzero(luma))
    if radius <= 0 or src_nonzero <= 0:
        return arr, {
            "backend": "off" if radius <= 0 else ("cv2_dilate" if _cv2 is not None else "cpu_sparse"),
            "requested_px": float(radius),
            "effective_px": 0.0,
            "src_nonzero_pixels": int(src_nonzero),
            "display_nonzero_pixels": int(src_nonzero),
            "skipped_dense": False,
        }
    if _cv2 is not None:
        try:
            kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
            expanded = _cv2.dilate(arr, kernel, iterations=1, borderType=_cv2.BORDER_REPLICATE)
            if expanded.ndim == 2:
                display_nonzero = int(np.count_nonzero(expanded))
            else:
                display_nonzero = int(np.count_nonzero(np.max(expanded[:, :, : min(3, int(expanded.shape[2]))], axis=2)))
            return np.ascontiguousarray(expanded), {
                "backend": "cv2_dilate",
                "requested_px": float(radius),
                "effective_px": float(radius),
                "src_nonzero_pixels": int(src_nonzero),
                "display_nonzero_pixels": int(display_nonzero),
                "max_pixels": int(max(0, int(max_pixels or 0))),
                "skipped_dense": False,
            }
        except Exception:
            pass
    max_pixels_i = max(0, int(max_pixels or 0))
    if max_pixels_i > 0 and src_nonzero > max_pixels_i:
        return arr, {
            "backend": "cpu_sparse",
            "requested_px": float(radius),
            "effective_px": 0.0,
            "src_nonzero_pixels": int(src_nonzero),
            "display_nonzero_pixels": int(src_nonzero),
            "max_pixels": int(max_pixels_i),
            "skipped_dense": True,
        }
    ys, xs = np.nonzero(luma)
    if ys.size <= 0:
        return arr, {
            "backend": "cpu_sparse",
            "requested_px": float(radius),
            "effective_px": 0.0,
            "src_nonzero_pixels": int(src_nonzero),
            "display_nonzero_pixels": int(src_nonzero),
            "skipped_dense": False,
        }
    h, w = int(arr.shape[0]), int(arr.shape[1])
    expanded = arr.copy()
    if expanded.ndim == 2:
        flat = expanded.reshape(-1)
        values = np.ascontiguousarray(arr[ys, xs])
        for dy in range(-radius, radius + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            row = yy * w
            for dx in range(-radius, radius + 1):
                xx = np.clip(xs + dx, 0, w - 1)
                np.maximum.at(flat, row + xx, values)
    else:
        ch = int(expanded.shape[2])
        flat3 = expanded.reshape(-1, ch)
        values3 = np.ascontiguousarray(arr[ys, xs, :])
        for dy in range(-radius, radius + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            row = yy * w
            for dx in range(-radius, radius + 1):
                xx = np.clip(xs + dx, 0, w - 1)
                idx = row + xx
                for c in range(ch):
                    np.maximum.at(flat3[:, c], idx, values3[:, c])
    if expanded.ndim == 2:
        display_nonzero = int(np.count_nonzero(expanded))
    else:
        display_nonzero = int(np.count_nonzero(np.max(expanded[:, :, : min(3, int(expanded.shape[2]))], axis=2)))
    return np.ascontiguousarray(expanded), {
        "backend": "cpu_sparse",
        "requested_px": float(radius),
        "effective_px": float(radius),
        "src_nonzero_pixels": int(src_nonzero),
        "display_nonzero_pixels": int(display_nonzero),
        "max_pixels": int(max_pixels_i),
        "skipped_dense": False,
    }


def _downsample_event_layer_frame(frame: np.ndarray, scale: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resize the sparse display-only event layer before GL upload.

    The BEV overlay is diagnostic/visual, while the event history ring remains
    the source of truth.  Use area resize here because it preserves sparse
    pixels as dimmer cells at 0.5x/0.25x without the full-frame morphology cost
    of a pre-upload dilation pass.
    """
    arr = np.asarray(frame)
    src_h = int(arr.shape[0]) if arr.ndim >= 2 else 0
    src_w = int(arr.shape[1]) if arr.ndim >= 2 else 0
    scale_f = max(0.25, min(1.0, float(scale or 1.0)))
    meta: Dict[str, Any] = {
        "requested_scale": round(float(scale_f), 4),
        "actual_scale_x": 1.0,
        "actual_scale_y": 1.0,
        "source_width": int(src_w),
        "source_height": int(src_h),
        "texture_width": int(src_w),
        "texture_height": int(src_h),
        "pool_step": 1,
        "backend": "none",
    }
    if arr.size <= 0 or arr.ndim < 2 or src_w <= 1 or src_h <= 1 or scale_f >= 0.999:
        return arr, meta
    out_h = max(1, int(round(float(src_h) * scale_f)))
    out_w = max(1, int(round(float(src_w) * scale_f)))
    if out_h == src_h and out_w == src_w:
        return arr, meta
    step = max(1, int(round(1.0 / scale_f)))
    backend = "stride_sample"
    if _cv2 is not None:
        try:
            pooled = _cv2.resize(arr, (int(out_w), int(out_h)), interpolation=_cv2.INTER_AREA)
            pooled = np.ascontiguousarray(pooled)
            backend = "cv2_area_resize"
        except Exception:
            backend = "stride_sample"
        else:
            meta.update({
                "actual_scale_x": round(float(out_w) / max(1.0, float(src_w)), 6),
                "actual_scale_y": round(float(out_h) / max(1.0, float(src_h)), 6),
                "texture_width": int(out_w),
                "texture_height": int(out_h),
                "pool_step": int(step),
                "backend": backend,
            })
            return pooled, meta
    trim_h = max(1, min(src_h, out_h * step))
    trim_w = max(1, min(src_w, out_w * step))
    trimmed = np.ascontiguousarray(arr[:trim_h, :trim_w])
    pooled = np.ascontiguousarray(trimmed[::step, ::step])
    out_h = int(pooled.shape[0])
    out_w = int(pooled.shape[1])
    meta.update({
        "actual_scale_x": round(float(out_w) / max(1.0, float(src_w)), 6),
        "actual_scale_y": round(float(out_h) / max(1.0, float(src_h)), 6),
        "texture_width": int(out_w),
        "texture_height": int(out_h),
        "pool_step": int(step),
        "backend": backend,
    })
    return pooled, meta


def _normalize_transform_mode(mode: Any, flip_x: bool = False, flip_y: bool = False) -> str:
    m = str(mode or "").strip()
    if m in TRANSFORM_MODE_CODES and not (m == "normal" and (flip_x or flip_y)):
        return m
    if flip_x and flip_y:
        return "flip_xy"
    if flip_x:
        return "flip_x"
    if flip_y:
        return "flip_y"
    return "normal"


def _transform_mode_code(mode: Any) -> int:
    return int(TRANSFORM_MODE_CODES.get(_normalize_transform_mode(mode), 0))


def _event_layer_color_mode(mode: Any) -> str:
    m = str(mode or "camera").strip().lower().replace("-", "_")
    return m if m in EVENT_LAYER_COLOR_MODE_CODES else "camera"


def _event_layer_color_mode_code(mode: Any) -> int:
    return int(EVENT_LAYER_COLOR_MODE_CODES.get(_event_layer_color_mode(mode), 0))


def _event_layer_sync_mode(mode: Any) -> str:
    text = str(mode or "latest_guarded").strip().lower().replace("-", "_")
    if text in set(EVENT_LAYER_SYNC_MODE_ORDER):
        return text
    return "latest_guarded"


def _alignment_mode(mode: Any) -> str:
    text = str(mode or "current").strip().lower().replace("-", "_")
    if text in set(ALIGNMENT_MODE_ORDER):
        return text
    return "current"


def _simple_flip_from_mode(mode: Any) -> Tuple[bool, bool]:
    m = _normalize_transform_mode(mode)
    if m == "flip_x":
        return True, False
    if m == "flip_y":
        return False, True
    if m == "flip_xy":
        return True, True
    return False, False


def _qt_key_value(name: str) -> int:
    v = getattr(Qt, name, None)
    if v is None:
        v = getattr(Qt.Key, name)
    try:
        return int(v)
    except Exception:
        return int(v.value)


def _parse_rgb_triplet(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            _safe_float(value[0], default[0]),
            _safe_float(value[1], default[1]),
            _safe_float(value[2], default[2]),
        )
    parts = [x.strip() for x in str(value or "").replace(";", ",").split(",") if x.strip()]
    if len(parts) >= 3:
        return (
            _safe_float(parts[0], default[0]),
            _safe_float(parts[1], default[1]),
            _safe_float(parts[2], default[2]),
        )
    return default


def _estimate_turf_rgb_from_bayer(img: np.ndarray, pattern: str, max_samples: int = 24000) -> Tuple[Optional[np.ndarray], int]:
    arr = np.asarray(img)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return None, 0
    h, w = arr.shape[:2]
    step = max(2, int(round(np.sqrt(max(1.0, (h * w) / float(max(1, max_samples)))))))
    if step % 2:
        step += 1
    ys = np.arange(0, h - 1, step, dtype=np.int64)
    xs = np.arange(0, w - 1, step, dtype=np.int64)
    if ys.size == 0 or xs.size == 0:
        return None, 0
    a00 = arr[np.ix_(ys, xs)].astype(np.float32) / 255.0
    a01 = arr[np.ix_(ys, xs + 1)].astype(np.float32) / 255.0
    a10 = arr[np.ix_(ys + 1, xs)].astype(np.float32) / 255.0
    a11 = arr[np.ix_(ys + 1, xs + 1)].astype(np.float32) / 255.0
    p = str(pattern or "BG").upper()
    if p == "RG":
        r, g, b = a00, 0.5 * (a01 + a10), a11
    elif p == "GR":
        r, g, b = a01, 0.5 * (a00 + a11), a10
    elif p == "GB":
        r, g, b = a10, 0.5 * (a00 + a11), a01
    else:
        r, g, b = a11, 0.5 * (a01 + a10), a00
    mx = np.maximum(r, b)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Keep the green carpet and reject black covers/shadows and white court lines.
    mask = (g > mx * 1.04) & (g > 0.035) & (luma > 0.045) & (luma < 0.72) & ((g - mx) > 0.010)
    count = int(np.count_nonzero(mask))
    if count < 48:
        mask = (g > mx * 1.01) & (g > 0.025) & (luma > 0.035) & (luma < 0.82)
        count = int(np.count_nonzero(mask))
    if count < 16:
        return None, count
    rgb = np.asarray([np.median(r[mask]), np.median(g[mask]), np.median(b[mask])], dtype=np.float32)
    return np.clip(rgb, 1e-4, 1.0), count


def _parse_cams(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").replace(";", ",").split(",") if x.strip()]


def _shm_path(name_or_path: str) -> str:
    p = str(name_or_path or "").strip()
    if not p:
        raise ValueError("empty shm name")
    if p.startswith("/"):
        return p
    if p.endswith(".mmap"):
        return os.path.join("/dev/shm", p)
    return os.path.join("/dev/shm", p + ".mmap")


class BevSharedFrameWriter:
    def __init__(self, name_or_path: str, width: int, height: int):
        self.path = _shm_path(name_or_path)
        self.width = int(width)
        self.height = int(height)
        self.channels = GBEV_CHANNELS
        self.data_bytes = self.width * self.height * self.channels
        self.total_bytes = GBEV_HEADER_BYTES + 2 * self.data_bytes
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(self.fd, self.total_bytes)
        self.mm = mmap.mmap(self.fd, self.total_bytes, access=mmap.ACCESS_WRITE)
        self.header = np.ndarray((GBEV_HEADER_U64_COUNT,), dtype=np.uint64, buffer=self.mm, offset=0)
        self.slots = [
            np.ndarray((self.height, self.width, self.channels), dtype=np.uint8, buffer=self.mm,
                       offset=GBEV_HEADER_BYTES + i * self.data_bytes)
            for i in range(2)
        ]
        self.header[:] = 0
        self.header[0] = GBEV_MAGIC
        self.header[1] = GBEV_VERSION
        self.header[2] = GBEV_HEADER_BYTES
        self.header[3] = self.width
        self.header[4] = self.height
        self.header[5] = self.channels
        self.header[6] = self.data_bytes
        self.header[7] = 0  # active slot
        self.header[8] = 0  # frame id
        self.header[9] = 0  # timestamp us
        self.header[10] = 0  # target tick/frame
        self.header[11] = 0  # write seq

    def write_rgba(self, rgba: np.ndarray, frame_id: int, timestamp_us: int, target_sync_id: int) -> None:
        arr = np.asarray(rgba)
        if arr.shape != (self.height, self.width, self.channels) or arr.dtype != np.uint8:
            raise ValueError(f"BEV RGBA frame shape mismatch: {arr.shape} {arr.dtype}")
        active = int(self.header[7])
        slot = 1 - active
        seq = int(self.header[11])
        self.header[11] = seq + 1 if seq % 2 == 0 else seq
        np.copyto(self.slots[slot], arr)
        self.header[8] = int(frame_id) & 0xFFFFFFFFFFFFFFFF
        self.header[9] = int(timestamp_us) & 0xFFFFFFFFFFFFFFFF
        self.header[10] = int(target_sync_id) & 0xFFFFFFFFFFFFFFFF
        self.header[12] = int(time.monotonic_ns()) & 0xFFFFFFFFFFFFFFFF
        self.header[7] = int(slot)
        self.header[11] = int(self.header[11]) + 1

    def close(self) -> None:
        try:
            self.header = None
            self.slots = []
            self.mm.close()
        finally:
            os.close(self.fd)


class JsonlTailer:
    def __init__(self, path: Path, initial_tail_bytes: int = 4 * 1024 * 1024):
        self.path = Path(path)
        self.offset = 0
        self.inode = 0
        self.initial_tail_bytes = max(0, int(initial_tail_bytes))
        self.opened = False
        self.error = ""
        self.bad_lines = 0
        self.good_lines = 0
        self.last_size = 0
        self.last_mtime_ns = 0

    def read_new(self, max_lines: int = 4096) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        max_lines = max(1, int(max_lines or 1))
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self.error = "missing"
            return out
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return out

        inode = int(getattr(st, "st_ino", 0) or 0)
        size = int(st.st_size)
        if inode != self.inode or size < self.offset:
            self.inode = inode
            self.offset = 0
            self.opened = False
        self.last_size = size
        self.last_mtime_ns = int(st.st_mtime_ns)

        try:
            with self.path.open("rb") as f:
                if not self.opened and self.initial_tail_bytes > 0 and size > self.initial_tail_bytes:
                    self.offset = max(0, size - self.initial_tail_bytes)
                    f.seek(self.offset)
                    if self.offset > 0:
                        f.readline()
                    self.offset = f.tell()
                else:
                    f.seek(self.offset)
                for raw in f:
                    self.offset = f.tell()
                    if not raw.strip():
                        continue
                    try:
                        obj = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        self.bad_lines += 1
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
                        self.good_lines += 1
                    if len(out) >= max_lines:
                        break
                self.opened = True
            self.error = ""
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        return out


class AsyncStatusWriter:
    """Latest-only JSON writer used to keep file I/O out of the GL paint loop."""

    def __init__(self, status_path: Path, debug_path: Path):
        self.status_path = Path(status_path)
        self.debug_path = Path(debug_path)
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.pending: Optional[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = None
        self.thread = threading.Thread(target=self._run, name="global-bev-status-writer", daemon=True)
        self.started = False
        self.closed = False
        self.seq = 0
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self.last_submit_ms = 0.0
        self.last_write_ms = 0.0
        self.last_write_wall_us = 0
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.started and not self.closed)

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread.start()

    def _snapshot_locked(self) -> Dict[str, Any]:
        state = "closed" if self.closed else ("running" if self.started else "created")
        return {
            "mode": "async",
            "state": state,
            "thread_alive": bool(self.thread.is_alive()) if self.started else False,
            "submitted": int(self.submitted),
            "written": int(self.written),
            "dropped": int(self.dropped),
            "errors": int(self.errors),
            "pending": self.pending is not None,
            "last_submit_ms": round(float(self.last_submit_ms), 3),
            "last_write_ms": round(float(self.last_write_ms), 3),
            "last_write_wall_us": int(self.last_write_wall_us),
            "last_error": str(self.last_error),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()

    def submit(self, payload: Dict[str, Any], debug_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # Deep-copy once on the GL thread so the worker never iterates over
        # dictionaries that paintGL may mutate on the next frame.
        status_obj = copy.deepcopy(payload)
        debug_obj = copy.deepcopy(debug_payload) if debug_payload is not None else None
        submit_ms = (time.perf_counter() - t0) * 1000.0
        with self.lock:
            if self.pending is not None:
                self.dropped += 1
            self.seq += 1
            self.submitted += 1
            self.last_submit_ms = float(submit_ms)
            status_obj["status_writer"] = self._snapshot_locked()
            if debug_obj is not None:
                debug_obj["status_writer"] = dict(status_obj["status_writer"])
            self.pending = (int(self.seq), status_obj, debug_obj)
            snap = self._snapshot_locked()
        self.event.set()
        return snap

    @staticmethod
    def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    def _run(self) -> None:
        while True:
            self.event.wait(0.5)
            self.event.clear()
            while True:
                with self.lock:
                    item = self.pending
                    self.pending = None
                if item is None:
                    break
                _, status_obj, debug_obj = item
                t0 = time.perf_counter()
                try:
                    self._atomic_write_json(self.status_path, status_obj)
                    if debug_obj is not None:
                        self._atomic_write_json(self.debug_path, debug_obj)
                    write_ms = (time.perf_counter() - t0) * 1000.0
                    with self.lock:
                        self.written += 1
                        self.last_write_ms = float(write_ms)
                        self.last_write_wall_us = _now_us()
                        self.last_error = ""
                except Exception as exc:
                    with self.lock:
                        self.errors += 1
                        self.last_write_ms = (time.perf_counter() - t0) * 1000.0
                        self.last_error = f"{type(exc).__name__}: {exc}"
            if self.stop_event.is_set():
                with self.lock:
                    if self.pending is None:
                        self.closed = True
                        return

    def close(self, timeout_s: float = 1.0) -> None:
        self.stop_event.set()
        self.event.set()
        if self.started and self.thread.is_alive():
            self.thread.join(max(0.0, float(timeout_s)))
        with self.lock:
            self.closed = True


class EventSyncMapRingReader:
    """Small cached reader for per-camera Event/MVS sync truth rings."""

    def __init__(self, cam: str, sync_root: str, template: str):
        self.cam = str(cam)
        self.sync_root = str(sync_root or "sync_ipc")
        tmpl = str(template or "event_sync_map_ring_{camera}.json")
        path = tmpl.format(camera=self.cam, cam=self.cam, id=self.cam)
        self.path = Path(path) if os.path.isabs(path) else Path(self.sync_root) / path
        self.mtime_ns = 0
        self.cache: Dict[str, Any] = {}
        self.read_error = ""
        self._usable_mtime_ns = 0
        self._usable_by_event_ts: List[Tuple[int, float, Dict[str, Any]]] = []
        self._usable_by_mvs_sync: List[Tuple[float, int, Dict[str, Any]]] = []
        self._usable_event_ts_keys: List[int] = []
        self._usable_mvs_sync_keys: List[float] = []

    def read(self) -> Dict[str, Any]:
        try:
            st = self.path.stat()
            mtime_ns = int(st.st_mtime_ns)
            if mtime_ns == int(self.mtime_ns) and self.cache:
                return dict(self.cache)
            obj = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict):
                self.cache = obj
                self.mtime_ns = mtime_ns
                self._usable_mtime_ns = 0
                self._usable_by_event_ts = []
                self._usable_by_mvs_sync = []
                self._usable_event_ts_keys = []
                self._usable_mvs_sync_keys = []
                self.read_error = ""
                return dict(obj)
        except Exception as exc:
            cached = dict(self.cache)
            cached["read_error"] = f"{type(exc).__name__}: {exc}"
            cached["path"] = str(self.path)
            return cached
        return {}

    @staticmethod
    def _record_anchor_ts(rec: Dict[str, Any]) -> int:
        return _safe_int(rec.get("mvs_soft_trigger_event_ts_us", rec.get("event_edge_ts_us")), 0)

    def _usable_records(self) -> Tuple[Dict[str, Any], List[Tuple[int, float, Dict[str, Any]]], List[Tuple[float, int, Dict[str, Any]]]]:
        ring = self.read()
        mtime = int(self.mtime_ns)
        if mtime > 0 and mtime == int(self._usable_mtime_ns) and self._usable_by_event_ts:
            return dict(ring), list(self._usable_by_event_ts), list(self._usable_by_mvs_sync)
        records = ring.get("records", []) if isinstance(ring, dict) else []
        by_ts: List[Tuple[int, float, Dict[str, Any]]] = []
        by_sync: List[Tuple[float, int, Dict[str, Any]]] = []
        if isinstance(records, list):
            for raw in records:
                if not isinstance(raw, dict):
                    continue
                mapped = raw.get("mapped_mvs_sync_index")
                if mapped is None:
                    continue
                anchor_ts = self._record_anchor_ts(raw)
                if anchor_ts <= 0:
                    continue
                try:
                    mapped_f = float(mapped)
                except Exception:
                    continue
                by_ts.append((int(anchor_ts), mapped_f, raw))
                by_sync.append((mapped_f, int(anchor_ts), raw))
        by_ts.sort(key=lambda row: row[0])
        by_sync.sort(key=lambda row: row[0])
        self._usable_mtime_ns = mtime
        self._usable_by_event_ts = list(by_ts)
        self._usable_by_mvs_sync = list(by_sync)
        self._usable_event_ts_keys = [int(row[0]) for row in by_ts]
        self._usable_mvs_sync_keys = [float(row[0]) for row in by_sync]
        return dict(ring), by_ts, by_sync

    def map_event_ts(self, event_ts_us: int) -> Dict[str, Any]:
        ring, usable, _ = self._usable_records()
        records = ring.get("records", []) if isinstance(ring, dict) else []
        now_us_i = _now_us()
        ring_wall_us = _safe_int(ring.get("published_wall_us"), 0) if isinstance(ring, dict) else 0
        ring_age_ms = max(0.0, (now_us_i - int(ring_wall_us)) / 1000.0) if ring_wall_us > 1_000_000_000_000 else -1.0
        base = {
            "event_sync_ring_path": str(self.path),
            "event_sync_ring_age_ms": round(float(ring_age_ms), 3) if ring_age_ms >= 0.0 else -1.0,
            "event_sync_ring_record_count": len(records) if isinstance(records, list) else 0,
            "event_sync_ring_records_written": _safe_int(ring.get("records_written"), -1) if isinstance(ring, dict) else -1,
            "event_sync_ring_error": str(ring.get("read_error", "")) if isinstance(ring, dict) else "missing_ring",
            "event_sync_ring_lock_state": str(ring.get("lock_state", "")) if isinstance(ring, dict) else "",
            "event_sync_ring_offset_jump_count": _safe_int(ring.get("offset_jump_count"), -1) if isinstance(ring, dict) else -1,
        }
        if not isinstance(records, list) or not records:
            return {
                **base,
                "source": "event_sync_map_ring_missing",
                "event_sync_quality": "missing",
                "event_mapped_mvs_sync_index": -1,
                "event_mapped_mvs_sync_index_float": None,
            }

        if not usable:
            return {
                **base,
                "source": "event_sync_map_ring_no_usable_records",
                "event_sync_quality": "missing",
                "event_mapped_mvs_sync_index": -1,
                "event_mapped_mvs_sync_index_float": None,
            }
        event_ts_i = int(event_ts_us or 0)
        keys = self._usable_event_ts_keys
        if len(keys) != len(usable):
            keys = [int(row[0]) for row in usable]
        idx = bisect_left(keys, int(event_ts_i))
        before = usable[idx - 1] if idx > 0 else None
        after = usable[idx] if idx < len(usable) else None
        if before is not None and after is not None:
            nearest = before if abs(int(before[0]) - int(event_ts_i)) <= abs(int(after[0]) - int(event_ts_i)) else after
        elif before is not None:
            nearest = before
        elif after is not None:
            nearest = after
        else:
            nearest = usable[0]
        nearest_delta_ms = abs(float(nearest[0] - event_ts_i)) / 1000.0 if event_ts_i > 0 else -1.0

        mapped_float: Optional[float] = None
        source = "event_sync_map_ring_nearest"
        quality = "degraded"
        anchor_rec = nearest[2]
        if before is not None and after is not None:
            if before[0] == after[0]:
                mapped_float = float(before[1])
                source = "event_sync_map_ring_exact"
                quality = "good"
                anchor_rec = before[2]
            else:
                frac = (float(event_ts_i) - float(before[0])) / max(1.0, float(after[0] - before[0]))
                mapped_float = float(before[1]) + float(frac) * (float(after[1]) - float(before[1]))
                source = "event_sync_map_ring_interpolated"
                quality = "good"
                anchor_rec = before[2] if abs(before[0] - event_ts_i) <= abs(after[0] - event_ts_i) else after[2]
        else:
            mapped_float = float(nearest[1])
            if nearest_delta_ms >= 0.0 and nearest_delta_ms <= 40.0:
                quality = "degraded"
            else:
                quality = "missing"

        mapped_i = int(round(float(mapped_float))) if mapped_float is not None else -1
        return {
            **base,
            "source": source,
            "event_sync_quality": quality,
            "event_mapped_mvs_sync_index": int(mapped_i),
            "event_mapped_mvs_sync_index_float": None if mapped_float is None else round(float(mapped_float), 6),
            "event_sync_anchor_mapped_mvs_sync_index": _safe_int(anchor_rec.get("mapped_mvs_sync_index"), -1),
            "event_sync_anchor_event_ts_us": self._record_anchor_ts(anchor_rec),
            "event_sync_estimated_from_event_ts_us": int(event_ts_i),
            "event_sync_nearest_abs_delta_ms": round(float(nearest_delta_ms), 3) if nearest_delta_ms >= 0.0 else -1.0,
            "nearest_mvs_sync_index": _safe_int(anchor_rec.get("nearest_mvs_sync_index"), -1),
            "event_local_sync_index": _safe_int(anchor_rec.get("event_local_sync_index"), -1),
            "event_sync_master": str(anchor_rec.get("sync_master", ring.get("sync_master", "")) or ""),
            "event_sync_lock_state": str(anchor_rec.get("lock_state", ring.get("lock_state", "")) or ""),
            "event_sync_match_mode": str(anchor_rec.get("match_mode", "") or ""),
        }

    def event_ts_for_mvs_sync(self, target_sync: int) -> Dict[str, Any]:
        ring, _, usable = self._usable_records()
        records = ring.get("records", []) if isinstance(ring, dict) else []
        base = {
            "event_sync_ring_path": str(self.path),
            "event_sync_ring_record_count": len(records) if isinstance(records, list) else 0,
            "event_sync_ring_records_written": _safe_int(ring.get("records_written"), -1) if isinstance(ring, dict) else -1,
            "event_sync_ring_error": str(ring.get("read_error", "")) if isinstance(ring, dict) else "missing_ring",
        }
        if int(target_sync) < 0 or not isinstance(records, list) or not records:
            return {**base, "event_ts_us": 0, "source": "missing_target_or_ring"}
        if not usable:
            return {**base, "event_ts_us": 0, "source": "no_usable_records"}
        target = float(target_sync)
        before: Optional[Tuple[float, int, Dict[str, Any]]] = None
        after: Optional[Tuple[float, int, Dict[str, Any]]] = None
        keys = self._usable_mvs_sync_keys
        if len(keys) != len(usable):
            keys = [float(row[0]) for row in usable]
        idx = bisect_left(keys, float(target))
        before = usable[idx - 1] if idx > 0 else None
        after = usable[idx] if idx < len(usable) else None
        if before is not None and after is not None:
            nearest = before if abs(float(before[0]) - target) <= abs(float(after[0]) - target) else after
        elif before is not None:
            nearest = before
        elif after is not None:
            nearest = after
        else:
            nearest = usable[0]
        if before is not None and after is not None:
            if abs(float(after[0] - before[0])) < 1e-9:
                return {**base, "event_ts_us": int(before[1]), "source": "inverse_exact"}
            frac = (target - float(before[0])) / max(1e-9, float(after[0] - before[0]))
            ts = float(before[1]) + float(frac) * float(after[1] - before[1])
            return {**base, "event_ts_us": int(round(ts)), "source": "inverse_interpolated"}
        return {
            **base,
            "event_ts_us": int(nearest[1]),
            "source": "inverse_nearest",
            "nearest_mvs_sync_delta": round(float(nearest[0] - target), 6),
        }


class EventOverlayLayerReader:
    """Reader for event_overlay_latest_* SharedFrameWriter frames.

    The event frame is visualization-only.  Black pixels are treated as
    transparent later in the BEV shader.
    """

    def __init__(
        self,
        cam: str,
        name_template: str,
        sync_root: str,
        sidecar_template: str,
        history_sidecar_template: str = "",
        history_poll_interval_ms: float = 0.0,
    ):
        self.cam = str(cam)
        self.name = str(name_template or "event_overlay_latest_{camera}").format(camera=cam, cam=cam, id=cam)
        self.path = self.name if self.name.startswith("/") else f"/dev/shm/{self.name}.mmap"
        self.sync_root = str(sync_root or "sync_ipc")
        if str(sidecar_template or "").strip():
            self.sidecar_path = os.path.join(
                self.sync_root,
                str(sidecar_template or "event_overlay_{camera}_latest.json").format(camera=cam, cam=cam, id=cam),
            )
        else:
            self.sidecar_path = ""
        if str(history_sidecar_template or "").strip():
            self.history_sidecar_path = os.path.join(
                self.sync_root,
                str(history_sidecar_template).format(camera=cam, cam=cam, id=cam),
            )
        else:
            self.history_sidecar_path = ""
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, os.path.getsize(self.path), access=mmap.ACCESS_READ)
        self.header = np.ndarray((_SF_HEADER_U32_COUNT,), dtype=np.uint32, buffer=self.mm, offset=0)
        if int(self.header[0]) != _SF_MAGIC:
            raise RuntimeError(f"event overlay shared frame magic mismatch: {self.path}")
        self.width = int(self.header[2])
        self.height = int(self.header[3])
        self.channels = int(self.header[4])
        self.data_bytes = int(self.header[9])
        if self.channels <= 0 or self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"bad event overlay shape: {self.width}x{self.height}x{self.channels}")
        self._slots = [
            np.ndarray(
                (self.height, self.width, self.channels),
                dtype=np.uint8,
                buffer=self.mm,
                offset=_SF_HEADER_BYTES + i * self.data_bytes,
            )
            for i in range(2)
        ]
        self.last_frame_id = -1
        self.last_frame: Optional[np.ndarray] = None
        self.last_meta: Dict[str, Any] = {}
        self.last_sidecar_mtime_ns = 0
        self.last_history_mtime_ns = 0
        self.last_history: Dict[str, Any] = {}
        self.last_history_read_mono = 0.0
        self.history_poll_interval_ms = max(0.0, float(history_poll_interval_ms or 0.0))
        self.history_cache_hits = 0
        self.history_reload_count = 0
        self.history_readers: Dict[str, "EventOverlayLayerReader"] = {}
        self.history_reader_cache_limit = 16

    def _read_sidecar(self, force: bool = False) -> Dict[str, Any]:
        if not self.sidecar_path:
            return dict(self.last_meta)
        try:
            st = os.stat(self.sidecar_path)
            if not force and int(st.st_mtime_ns) == int(self.last_sidecar_mtime_ns):
                return dict(self.last_meta)
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self.last_meta = obj
                self.last_sidecar_mtime_ns = int(st.st_mtime_ns)
        except Exception:
            pass
        return dict(self.last_meta)

    def _read_active_with_meta(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                return {
                    "frame": self.last_frame,
                    "frame_id": fid,
                    "timestamp_us": ts_us,
                    "meta": meta,
                    "shape": [int(self.width), int(self.height), int(self.channels)],
                    "new_frame": False,
                }
            frame = self._slots[slot_a].copy()
            slot_b = int(self.header[_SF_ACTIVE_SLOT])
            if slot_a != slot_b:
                continue
            self.last_frame_id = fid
            self.last_frame = frame
            return {
                "frame": frame,
                "frame_id": fid,
                "timestamp_us": ts_us,
                "meta": meta,
                "shape": [int(self.width), int(self.height), int(self.channels)],
                "new_frame": True,
            }
        return None

    def read_latest(self) -> Optional[Dict[str, Any]]:
        meta = self._read_sidecar(False)
        meta.setdefault("event_layer_history_source", "latest")
        return self._read_active_with_meta(meta)

    def _read_history_sidecar(self) -> Dict[str, Any]:
        if not self.history_sidecar_path:
            return {}
        now_mono = time.monotonic()
        poll_interval_s = max(0.0, float(self.history_poll_interval_ms) / 1000.0)
        if (
            poll_interval_s > 0.0
            and self.last_history
            and (now_mono - float(self.last_history_read_mono or 0.0)) < poll_interval_s
        ):
            self.history_cache_hits += 1
            obj = dict(self.last_history)
            obj["_history_cache_hit"] = True
            obj["_history_cache_age_ms"] = max(0.0, (now_mono - float(self.last_history_read_mono or now_mono)) * 1000.0)
            obj["_history_poll_interval_ms"] = float(self.history_poll_interval_ms)
            obj["_history_reload_count"] = int(self.history_reload_count)
            obj["_history_cache_hits"] = int(self.history_cache_hits)
            return obj
        try:
            st = os.stat(self.history_sidecar_path)
            if int(st.st_mtime_ns) == int(self.last_history_mtime_ns):
                self.last_history_read_mono = now_mono
                obj = dict(self.last_history)
                obj["_history_cache_hit"] = True
                obj["_history_cache_age_ms"] = 0.0
                obj["_history_poll_interval_ms"] = float(self.history_poll_interval_ms)
                obj["_history_reload_count"] = int(self.history_reload_count)
                obj["_history_cache_hits"] = int(self.history_cache_hits)
                return obj
            with open(self.history_sidecar_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self.last_history = obj
                self.last_history_mtime_ns = int(st.st_mtime_ns)
                self.last_history_read_mono = now_mono
                self.history_reload_count += 1
                out = dict(obj)
                out["_history_cache_hit"] = False
                out["_history_cache_age_ms"] = 0.0
                out["_history_poll_interval_ms"] = float(self.history_poll_interval_ms)
                out["_history_reload_count"] = int(self.history_reload_count)
                out["_history_cache_hits"] = int(self.history_cache_hits)
                return out
        except Exception as exc:
            obj = dict(self.last_history)
            obj["read_error"] = f"{type(exc).__name__}: {exc}"
            obj["path"] = str(self.history_sidecar_path)
            obj["_history_cache_hit"] = bool(obj)
            obj["_history_cache_age_ms"] = max(0.0, (now_mono - float(self.last_history_read_mono or now_mono)) * 1000.0) if obj else -1.0
            obj["_history_poll_interval_ms"] = float(self.history_poll_interval_ms)
            obj["_history_reload_count"] = int(self.history_reload_count)
            obj["_history_cache_hits"] = int(self.history_cache_hits)
            return obj
        return {}

    def _history_reader_for_entry(self, entry: Dict[str, Any]) -> Optional["EventOverlayLayerReader"]:
        path = str(entry.get("path") or "").strip()
        if not path:
            shm_name = str(entry.get("shm_name") or "").strip()
            if shm_name:
                path = shm_name if shm_name.startswith("/") else f"/dev/shm/{shm_name}.mmap"
        if not path:
            return None
        rd = self.history_readers.get(path)
        if rd is not None:
            try:
                self.history_readers.pop(path, None)
                self.history_readers[path] = rd
            except Exception:
                pass
            return rd
        rd = EventOverlayLayerReader(self.cam, path, self.sync_root, "", "")
        self.history_readers[path] = rd
        while len(self.history_readers) > int(self.history_reader_cache_limit):
            old_path = next(iter(self.history_readers), "")
            old_reader = self.history_readers.pop(old_path, None) if old_path else None
            if old_reader is None:
                break
            try:
                old_reader.close()
            except Exception:
                pass
        return rd

    def read_for_target(self, target_sync: int, score_fn) -> Optional[Dict[str, Any]]:
        history = self._read_history_sidecar()
        entries = history.get("entries", []) if isinstance(history, dict) else []
        history_meta = {
            "event_layer_history_entry_count": len(entries) if isinstance(entries, list) else 0,
            "event_layer_history_slots": _safe_int(history.get("history_slots"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_cursor": _safe_int(history.get("history_cursor"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_write_count": _safe_int(history.get("history_write_count"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_overwrite_count": _safe_int(history.get("history_overwrite_count"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_coverage_ms": _safe_float(history.get("history_coverage_us"), -1.0) / 1000.0 if isinstance(history, dict) else -1.0,
            "event_layer_history_cache_hit": bool(history.get("_history_cache_hit", False)) if isinstance(history, dict) else False,
            "event_layer_history_cache_age_ms": _safe_float(history.get("_history_cache_age_ms"), -1.0) if isinstance(history, dict) else -1.0,
            "event_layer_history_poll_interval_ms": _safe_float(history.get("_history_poll_interval_ms"), 0.0) if isinstance(history, dict) else 0.0,
            "event_layer_history_reload_count": _safe_int(history.get("_history_reload_count"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_cache_hits": _safe_int(history.get("_history_cache_hits"), -1) if isinstance(history, dict) else -1,
        }
        best_entry: Optional[Dict[str, Any]] = None
        best_score: Dict[str, Any] = {}
        best_abs = float("inf")
        if isinstance(entries, list):
            for raw in entries:
                if not isinstance(raw, dict) or not bool(raw.get("valid", True)):
                    continue
                meta = dict(raw)
                score = score_fn(meta, target_sync)
                if not isinstance(score, dict) or not bool(score.get("sync_available", False)):
                    continue
                abs_ms = abs(_safe_float(score.get("event_to_bev_sync_delta_ms"), 1e9))
                if abs_ms < best_abs:
                    best_abs = abs_ms
                    best_entry = meta
                    best_score = dict(score)
        if best_entry is not None:
            try:
                rd = self._history_reader_for_entry(best_entry)
                if rd is not None:
                    meta = dict(best_entry)
                    meta.update(best_score)
                    meta.update(history_meta)
                    meta["event_layer_history_source"] = "history"
                    meta["event_layer_history_sidecar"] = str(self.history_sidecar_path)
                    item = rd._read_active_with_meta(meta)
                    if item is not None:
                        item["history_selected"] = True
                        item["history_entry"] = dict(best_entry)
                        item["history_best_abs_delta_ms"] = round(float(best_abs), 3)
                        return item
            except Exception as exc:
                self.last_history["read_selected_error"] = f"{type(exc).__name__}: {exc}"
        item = self.read_latest()
        if item is not None:
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            meta = dict(meta)
            meta["event_layer_history_source"] = "latest_fallback"
            meta["event_layer_history_available"] = bool(isinstance(entries, list) and len(entries) > 0)
            meta["event_layer_history_entry_count"] = len(entries) if isinstance(entries, list) else 0
            meta.update(history_meta)
            if isinstance(history, dict) and str(history.get("read_error", "")).strip():
                meta["event_layer_history_error"] = str(history.get("read_error", ""))
            item["meta"] = meta
            item["history_selected"] = False
        return item

    def read_accumulated_for_target(
        self,
        target_sync: int,
        score_fn,
        window_ms: float = 40.0,
        max_frames: int = 4,
    ) -> Optional[Dict[str, Any]]:
        history = self._read_history_sidecar()
        entries = history.get("entries", []) if isinstance(history, dict) else []
        if not isinstance(entries, list) or not entries:
            return self.read_for_target(target_sync, score_fn)
        window_ms_f = max(1.0, float(window_ms or 40.0))
        max_frames_i = max(1, int(max_frames or 1))
        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for raw in entries:
            if not isinstance(raw, dict) or not bool(raw.get("valid", True)):
                continue
            meta = dict(raw)
            score = score_fn(meta, target_sync)
            if not isinstance(score, dict) or not bool(score.get("sync_available", False)):
                continue
            abs_ms = abs(_safe_float(score.get("event_to_bev_sync_delta_ms"), 1e9))
            if abs_ms <= window_ms_f:
                scored.append((float(abs_ms), meta, dict(score)))
        if not scored:
            return self.read_for_target(target_sync, score_fn)
        scored.sort(key=lambda row: row[0])
        selected = scored[:max_frames_i]
        frames: List[np.ndarray] = []
        frame_ids: List[int] = []
        slots: List[int] = []
        centers: List[int] = []
        paths: List[str] = []
        best_abs, best_entry, best_score = selected[0]
        best_item: Optional[Dict[str, Any]] = None
        for _, entry, score in selected:
            try:
                rd = self._history_reader_for_entry(entry)
                if rd is None:
                    continue
                meta = dict(entry)
                meta.update(score)
                item = rd._read_active_with_meta(meta)
                if item is None:
                    continue
                frame = np.asarray(item.get("frame"))
                if frame.ndim == 2:
                    frame = frame[:, :, None]
                frames.append(frame)
                frame_ids.append(_safe_int(item.get("frame_id"), -1))
                slots.append(_safe_int(entry.get("slot"), -1))
                centers.append(_safe_int(entry.get("event_window_center_us"), 0))
                paths.append(str(entry.get("path") or entry.get("shm_name") or ""))
                if best_item is None:
                    best_item = item
            except Exception as exc:
                self.last_history["read_accumulated_error"] = f"{type(exc).__name__}: {exc}"
        if not frames or best_item is None:
            return self.read_for_target(target_sync, score_fn)
        if len(frames) == 1:
            combined = frames[0]
        else:
            combined = frames[0].copy()
            for frame in frames[1:]:
                if frame.shape == combined.shape:
                    np.maximum(combined, frame, out=combined)
        history_meta = {
            "event_layer_history_entry_count": len(entries),
            "event_layer_history_slots": _safe_int(history.get("history_slots"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_cursor": _safe_int(history.get("history_cursor"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_write_count": _safe_int(history.get("history_write_count"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_overwrite_count": _safe_int(history.get("history_overwrite_count"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_coverage_ms": _safe_float(history.get("history_coverage_us"), -1.0) / 1000.0 if isinstance(history, dict) else -1.0,
            "event_layer_history_cache_hit": bool(history.get("_history_cache_hit", False)) if isinstance(history, dict) else False,
            "event_layer_history_cache_age_ms": _safe_float(history.get("_history_cache_age_ms"), -1.0) if isinstance(history, dict) else -1.0,
            "event_layer_history_poll_interval_ms": _safe_float(history.get("_history_poll_interval_ms"), 0.0) if isinstance(history, dict) else 0.0,
            "event_layer_history_reload_count": _safe_int(history.get("_history_reload_count"), -1) if isinstance(history, dict) else -1,
            "event_layer_history_cache_hits": _safe_int(history.get("_history_cache_hits"), -1) if isinstance(history, dict) else -1,
        }
        meta = dict(best_entry)
        meta.update(best_score)
        meta.update(history_meta)
        meta["event_layer_history_source"] = "history_accumulated"
        meta["event_layer_history_sidecar"] = str(self.history_sidecar_path)
        meta["event_layer_accumulated_frames"] = int(len(frames))
        meta["event_layer_accumulated_frame_ids"] = list(frame_ids)
        meta["event_layer_accumulated_slots"] = list(slots)
        meta["event_layer_accumulated_centers_us"] = list(centers)
        meta["event_layer_accumulated_paths"] = list(paths)
        meta["event_layer_accumulation_window_ms"] = float(window_ms_f)
        meta["event_layer_accumulation_truncated"] = bool(len(scored) > len(frames))
        meta["overlay_nonzero_pixels"] = int(np.count_nonzero(combined))
        meta["overlay_max_luma"] = int(np.max(combined)) if combined.size else 0
        item = dict(best_item)
        item["frame"] = combined
        item["meta"] = meta
        item["history_selected"] = True
        item["history_entry"] = dict(best_entry)
        item["history_best_abs_delta_ms"] = round(float(best_abs), 3)
        item["accumulated_frame_count"] = int(len(frames))
        item["accumulated_upload_key"] = "|".join(
            f"{slot}:{fid}:{center}" for slot, fid, center in zip(slots, frame_ids, centers)
        )
        item["new_frame"] = True
        return item

    def close(self) -> None:
        try:
            for rd in list(self.history_readers.values()):
                try:
                    rd.close()
                except Exception:
                    pass
            self.history_readers.clear()
            self.header = None
            self._slots = []
            self.mm.close()
            os.close(self.fd)
        except Exception:
            pass


class SparseEventOverlayReader:
    """Reader for sparse event-overlay history mmap slots.

    Each slot is one timestamped point cloud frame generated by the native HAL
    broker after human-mask filtering.  The BEV renderer still uses the V24-A
    event sync-map ring to choose the frame.
    """

    def __init__(
        self,
        cam: str,
        sync_root: str,
        sidecar_template: str,
        poll_interval_ms: float = 0.0,
    ):
        self.cam = str(cam)
        self.sync_root = str(sync_root or "sync_ipc")
        self.sidecar_path = os.path.join(
            self.sync_root,
            str(sidecar_template or "event_overlay_sparse_{camera}_history.json").format(camera=cam, cam=cam, id=cam),
        )
        self.last_sidecar_mtime_ns = 0
        self.last_sidecar: Dict[str, Any] = {}
        self.last_read_mono = 0.0
        self.poll_interval_ms = max(0.0, float(poll_interval_ms or 0.0))
        self.cache_hits = 0
        self.reload_count = 0
        self.last_read_error = ""
        self.mmaps: Dict[str, Dict[str, Any]] = {}
        self.mmap_cache_limit = 16

    def _read_sidecar(self, force: bool = False) -> Dict[str, Any]:
        now_mono = time.monotonic()
        poll_s = max(0.0, float(self.poll_interval_ms) / 1000.0)
        if not force and poll_s > 0.0 and self.last_sidecar and (now_mono - float(self.last_read_mono or 0.0)) < poll_s:
            self.cache_hits += 1
            obj = dict(self.last_sidecar)
            obj["_sparse_cache_hit"] = True
            obj["_sparse_cache_age_ms"] = max(0.0, (now_mono - float(self.last_read_mono or now_mono)) * 1000.0)
            obj["_sparse_poll_interval_ms"] = float(self.poll_interval_ms)
            obj["_sparse_reload_count"] = int(self.reload_count)
            obj["_sparse_cache_hits"] = int(self.cache_hits)
            return obj
        try:
            st = os.stat(self.sidecar_path)
            if not force and int(st.st_mtime_ns) == int(self.last_sidecar_mtime_ns):
                self.last_read_mono = now_mono
                obj = dict(self.last_sidecar)
                obj["_sparse_cache_hit"] = True
                obj["_sparse_cache_age_ms"] = 0.0
                obj["_sparse_poll_interval_ms"] = float(self.poll_interval_ms)
                obj["_sparse_reload_count"] = int(self.reload_count)
                obj["_sparse_cache_hits"] = int(self.cache_hits)
                return obj
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self.last_sidecar = obj
                self.last_sidecar_mtime_ns = int(st.st_mtime_ns)
                self.last_read_mono = now_mono
                self.reload_count += 1
                out = dict(obj)
                out["_sparse_cache_hit"] = False
                out["_sparse_cache_age_ms"] = 0.0
                out["_sparse_poll_interval_ms"] = float(self.poll_interval_ms)
                out["_sparse_reload_count"] = int(self.reload_count)
                out["_sparse_cache_hits"] = int(self.cache_hits)
                return out
        except Exception as exc:
            obj = dict(self.last_sidecar)
            obj["read_error"] = f"{type(exc).__name__}: {exc}"
            self.last_read_error = str(obj["read_error"])
            obj["path"] = str(self.sidecar_path)
            obj["_sparse_cache_hit"] = bool(obj)
            obj["_sparse_cache_age_ms"] = max(0.0, (now_mono - float(self.last_read_mono or now_mono)) * 1000.0) if obj else -1.0
            obj["_sparse_poll_interval_ms"] = float(self.poll_interval_ms)
            obj["_sparse_reload_count"] = int(self.reload_count)
            obj["_sparse_cache_hits"] = int(self.cache_hits)
            return obj
        return {}

    def _path_from_entry(self, entry: Dict[str, Any]) -> str:
        path = str(entry.get("path") or "").strip()
        if path:
            return path
        shm_name = str(entry.get("shm_name") or "").strip()
        if not shm_name:
            return ""
        return shm_name if shm_name.startswith("/") else f"/dev/shm/{shm_name}.mmap"

    def _mmap_for_path(self, path: str) -> Optional[Dict[str, Any]]:
        path = str(path or "").strip()
        if not path:
            return None
        cached = self.mmaps.get(path)
        if cached is not None:
            try:
                self.mmaps.pop(path, None)
                self.mmaps[path] = cached
            except Exception:
                pass
            return cached
        fd = os.open(path, os.O_RDONLY)
        size = os.path.getsize(path)
        mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        os.close(fd)
        header = np.ndarray((_SPARSE_HEADER_U64_COUNT,), dtype=np.uint64, buffer=mm, offset=0)
        if int(header[0]) != int(_SPARSE_MAGIC):
            mm.close()
            raise RuntimeError(f"sparse event overlay magic mismatch: {path}")
        stride = int(header[3])
        if stride != int(_SPARSE_POINT_DTYPE.itemsize):
            mm.close()
            raise RuntimeError(f"sparse point stride mismatch: {path}: {stride}")
        max_points = max(1, int(header[4]))
        need = int(_SPARSE_HEADER_BYTES) + max_points * int(_SPARSE_POINT_DTYPE.itemsize)
        if int(size) < int(need):
            mm.close()
            raise RuntimeError(f"sparse mmap too small: {path}: {size} < {need}")
        points = np.ndarray((max_points,), dtype=_SPARSE_POINT_DTYPE, buffer=mm, offset=_SPARSE_HEADER_BYTES)
        obj = {"path": path, "mm": mm, "header": header, "points": points, "max_points": max_points}
        self.mmaps[path] = obj
        while len(self.mmaps) > int(self.mmap_cache_limit):
            old_path = next(iter(self.mmaps), "")
            old = self.mmaps.pop(old_path, None) if old_path else None
            if old is None:
                break
            try:
                old["mm"].close()
            except Exception:
                pass
        return obj

    def _read_entry(self, entry: Dict[str, Any], meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = self._path_from_entry(entry)
        if not path:
            return None
        mm_obj = self._mmap_for_path(path)
        if mm_obj is None:
            return None
        header = mm_obj["header"]
        points_arr = mm_obj["points"]
        for _ in range(4):
            seq_a = int(header[15])
            if seq_a % 2:
                continue
            point_count = max(0, min(int(header[5]), int(mm_obj["max_points"])))
            frame_id = int(header[8])
            start_us = int(header[9])
            end_us = int(header[10])
            center_us = int(header[11])
            published_wall_us = int(header[12])
            dropped = int(header[13])
            width = int(header[6])
            height = int(header[7])
            pts = points_arr[:point_count].copy()
            seq_b = int(header[15])
            if seq_a == seq_b and seq_b % 2 == 0:
                out_meta = dict(meta)
                out_meta.setdefault("path", path)
                out_meta.setdefault("frame_id", frame_id)
                out_meta.setdefault("event_window_start_us", start_us)
                out_meta.setdefault("event_window_end_us", end_us)
                out_meta.setdefault("event_window_center_us", center_us)
                out_meta.setdefault("published_wall_us", published_wall_us)
                out_meta.setdefault("sparse_point_count", point_count)
                out_meta.setdefault("sparse_dropped_point_count", dropped)
                out_meta.setdefault("overlay_nonzero_pixels", point_count)
                out_meta.setdefault("overlay_max_luma", 255 if point_count > 0 else 0)
                return {
                    "points": pts,
                    "frame_id": frame_id,
                    "timestamp_us": end_us,
                    "meta": out_meta,
                    "path": path,
                    "shape": [int(width), int(height), 1],
                    "new_frame": True,
                }
        return None

    def read_for_target(
        self,
        target_sync: int,
        score_fn,
        center_hint_us: int = 0,
        candidate_limit: int = 8,
        candidate_escalate_enable: bool = True,
        candidate_escalate_limit: int = 32,
        candidate_escalate_abs_ms: float = 10.0,
        candidate_escalate_ratio: float = 0.60,
        max_sync_delta_ms: float = 20.0,
        read_points: bool = True,
    ) -> Optional[Dict[str, Any]]:
        sidecar = self._read_sidecar()
        entries = sidecar.get("entries", []) if isinstance(sidecar, dict) else []
        self.last_read_error = str(sidecar.get("read_error", "") if isinstance(sidecar, dict) else "")
        sparse_meta = {
            "event_layer_sparse_entry_count": len(entries) if isinstance(entries, list) else 0,
            "event_layer_sparse_slots": _safe_int(sidecar.get("sparse_slots"), -1) if isinstance(sidecar, dict) else -1,
            "event_layer_sparse_cursor": _safe_int(sidecar.get("sparse_cursor"), -1) if isinstance(sidecar, dict) else -1,
            "event_layer_sparse_write_count": _safe_int(sidecar.get("sparse_write_count"), -1) if isinstance(sidecar, dict) else -1,
            "event_layer_sparse_overwrite_count": _safe_int(sidecar.get("sparse_overwrite_count"), -1) if isinstance(sidecar, dict) else -1,
            "event_layer_sparse_coverage_ms": _safe_float(sidecar.get("sparse_coverage_us"), -1.0) / 1000.0 if isinstance(sidecar, dict) else -1.0,
            "event_layer_sparse_cache_hit": bool(sidecar.get("_sparse_cache_hit", False)) if isinstance(sidecar, dict) else False,
            "event_layer_sparse_cache_age_ms": _safe_float(sidecar.get("_sparse_cache_age_ms"), -1.0) if isinstance(sidecar, dict) else -1.0,
            "event_layer_sparse_poll_interval_ms": _safe_float(sidecar.get("_sparse_poll_interval_ms"), 0.0) if isinstance(sidecar, dict) else 0.0,
            "event_layer_sparse_reload_count": _safe_int(sidecar.get("_sparse_reload_count"), -1) if isinstance(sidecar, dict) else -1,
            "event_layer_sparse_cache_hits": _safe_int(sidecar.get("_sparse_cache_hits"), -1) if isinstance(sidecar, dict) else -1,
        }
        if not isinstance(entries, list) or not entries:
            if not self.last_read_error:
                self.last_read_error = "no_sparse_entries"
            return None
        valid_entries = [dict(raw) for raw in entries if isinstance(raw, dict) and bool(raw.get("valid", True))]
        if not valid_entries:
            self.last_read_error = "no_valid_sparse_entries"
            return None
        ranked_entries: List[Dict[str, Any]]
        hint_i = int(center_hint_us or 0)
        if hint_i > 0:
            ranked_entries = sorted(valid_entries, key=lambda raw: abs(_event_layer_sync_timestamp_us(raw) - hint_i))
            scan_limit = max(1, min(len(ranked_entries), int(candidate_limit or 8)))
            scan_entries = ranked_entries[:scan_limit]
        else:
            ranked_entries = list(valid_entries)
            scan_entries = list(valid_entries)
        initial_scan_limit = len(scan_entries)
        best_entry: Optional[Dict[str, Any]] = None
        best_score: Dict[str, Any] = {}
        best_abs = float("inf")
        newest_entry: Optional[Dict[str, Any]] = None
        newest_sync_ts = -1
        scanned = 0
        for raw in scan_entries:
            scanned += 1
            sync_ts = _event_layer_sync_timestamp_us(raw)
            if sync_ts >= newest_sync_ts:
                newest_sync_ts = sync_ts
                newest_entry = dict(raw)
            meta = dict(raw)
            score = score_fn(meta, target_sync)
            if not isinstance(score, dict) or not bool(score.get("sync_available", False)):
                continue
            abs_ms = abs(_safe_float(score.get("event_to_bev_sync_delta_ms"), 1e9))
            if abs_ms < best_abs:
                best_abs = abs_ms
                best_entry = meta
                best_score = dict(score)
        escalate = False
        escalate_reason = ""
        threshold_ms = max(
            0.0,
            min(
                float(candidate_escalate_abs_ms or 0.0),
                max(1.0, float(max_sync_delta_ms or 20.0) * max(0.05, float(candidate_escalate_ratio or 0.60))),
            ),
        )
        if bool(candidate_escalate_enable) and hint_i > 0 and len(ranked_entries) > len(scan_entries):
            if best_entry is None:
                escalate = True
                escalate_reason = "no_valid_candidate_in_initial_scan"
            elif float(best_abs) > float(threshold_ms):
                escalate = True
                escalate_reason = "best_delta_near_sync_threshold"
        if escalate:
            final_limit = max(
                len(scan_entries) + 1,
                min(len(ranked_entries), int(candidate_escalate_limit or 32)),
            )
            for raw in ranked_entries[len(scan_entries): final_limit]:
                scanned += 1
                meta = dict(raw)
                score = score_fn(meta, target_sync)
                if not isinstance(score, dict) or not bool(score.get("sync_available", False)):
                    continue
                abs_ms = abs(_safe_float(score.get("event_to_bev_sync_delta_ms"), 1e9))
                if abs_ms < best_abs:
                    best_abs = abs_ms
                    best_entry = meta
                    best_score = dict(score)
            if newest_entry is None:
                newest_entry = ranked_entries[0]
        if best_entry is None and int(target_sync) < 0:
            best_entry = newest_entry
            best_score = {}
            best_abs = -1.0
        if best_entry is None:
            self.last_read_error = "no_sparse_candidate_for_target"
            return None
        meta = dict(best_entry)
        meta.update(best_score)
        meta.update(sparse_meta)
        meta["event_layer_history_source"] = "sparse_history"
        meta["event_layer_sparse_source"] = "sparse_history"
        meta["event_layer_sparse_sidecar"] = str(self.sidecar_path)
        meta["event_layer_sparse_sync_hint_us"] = int(hint_i)
        meta["event_layer_sparse_center_hint_us"] = int(hint_i)
        meta["event_layer_sync_timestamp_policy"] = EVENT_LAYER_SYNC_TIMESTAMP_POLICY
        meta["event_layer_sync_timestamp_us"] = _event_layer_sync_timestamp_us(meta)
        meta["event_layer_sparse_candidate_scan_count"] = int(scanned)
        meta["event_layer_sparse_candidate_scan_initial"] = int(initial_scan_limit)
        meta["event_layer_sparse_candidate_scan_escalated"] = bool(escalate)
        meta["event_layer_sparse_candidate_scan_escalate_reason"] = str(escalate_reason)
        meta["event_layer_sparse_candidate_scan_escalate_threshold_ms"] = round(float(threshold_ms), 3)
        meta["event_layer_sparse_candidate_total_count"] = int(len(valid_entries))
        if not bool(read_points):
            return {
                "points": np.empty((0,), dtype=_SPARSE_POINT_DTYPE),
                "frame_id": _safe_int(best_entry.get("frame_id"), -1),
                "timestamp_us": _event_layer_sync_timestamp_us(best_entry),
                "meta": meta,
                "path": self._path_from_entry(best_entry),
                "shape": [0, 0, 1],
                "new_frame": True,
                "history_selected": True,
                "history_entry": dict(best_entry),
                "history_best_abs_delta_ms": round(float(best_abs), 3) if best_abs >= 0.0 else -1.0,
            }
        try:
            item = self._read_entry(best_entry, meta)
        except Exception as exc:
            self.last_read_error = f"{type(exc).__name__}: {exc}"
            self.last_sidecar_mtime_ns = 0
            self._read_sidecar(force=True)
            return None
        if item is not None:
            item["history_selected"] = True
            item["history_entry"] = dict(best_entry)
            item["history_best_abs_delta_ms"] = round(float(best_abs), 3) if best_abs >= 0.0 else -1.0
            self.last_read_error = ""
        else:
            self.last_read_error = "selected_sparse_slot_unstable"
        return item

    def close(self) -> None:
        for obj in list(self.mmaps.values()):
            try:
                obj["mm"].close()
            except Exception:
                pass
        self.mmaps.clear()


class SparseEventLayerShadowWorker:
    """Background shadow selector for sparse event frames.

    It deliberately does not touch OpenGL or active BEV vertex buffers.  The
    worker only validates whether an independent reader would select the same
    sparse frame/sync delta as the active GUI path.
    """

    def __init__(
        self,
        cams: List[str],
        sync_root: str,
        sparse_sidecar_template: str,
        sync_map_ring_template: str,
        poll_interval_ms: float,
        interval_ms: float,
    ):
        self.cams = [str(cam) for cam in cams]
        self.sync_root = str(sync_root or "sync_ipc")
        self.sparse_sidecar_template = str(sparse_sidecar_template or "event_overlay_sparse_{camera}_history.json")
        self.sync_map_ring_template = str(sync_map_ring_template or "event_sync_map_ring_{camera}.json")
        self.poll_interval_ms = max(0.0, float(poll_interval_ms or 0.0))
        self.interval_s = max(0.05, float(interval_ms or 250.0) / 1000.0)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="bev-event-sparse-shadow", daemon=True)
        self.started = False
        self.target_sync = -1
        self.config: Dict[str, Any] = {}
        self.snapshot_obj: Dict[str, Any] = {
            "enabled": True,
            "mode": "shadow",
            "state": "init",
            "target_sync_id": -1,
            "per_camera": {},
        }
        self.readers: Dict[str, SparseEventOverlayReader] = {
            cam: SparseEventOverlayReader(cam, self.sync_root, self.sparse_sidecar_template, self.poll_interval_ms)
            for cam in self.cams
        }
        self.ring_readers: Dict[str, EventSyncMapRingReader] = {
            cam: EventSyncMapRingReader(cam, self.sync_root, self.sync_map_ring_template)
            for cam in self.cams
        }

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread.start()

    def submit(self, target_sync: int, config: Dict[str, Any]) -> None:
        with self.lock:
            self.target_sync = int(target_sync)
            self.config = dict(config)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.snapshot_obj)

    def stop(self, timeout_s: float = 1.0) -> None:
        self.stop_event.set()
        if self.started and self.thread.is_alive():
            self.thread.join(max(0.0, float(timeout_s)))
        for rd in list(self.readers.values()):
            try:
                rd.close()
            except Exception:
                pass

    def _score_meta(self, cam: str, meta: Dict[str, Any], target_sync: int, tick_ms: float) -> Dict[str, Any]:
        event_ts_us = _event_layer_sync_timestamp_us(meta, _safe_int(meta.get("timestamp_us"), 0))
        mapped_float: Optional[float] = None
        mapped = _safe_int(meta.get("event_mapped_mvs_sync_index", meta.get("mapped_mvs_sync_index")), -1)
        if mapped >= 0:
            mapped_float = float(mapped)
            sync_meta = {
                "source": "event_layer_history_sidecar",
                "event_sync_quality": "good",
                "event_mapped_mvs_sync_index": int(mapped),
                "event_mapped_mvs_sync_index_float": float(mapped),
                "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                "event_sync_estimated_from_event_ts_us": int(event_ts_us),
            }
        else:
            sync_meta = self.ring_readers[str(cam)].map_event_ts(event_ts_us)
            raw_float = sync_meta.get("event_mapped_mvs_sync_index_float")
            try:
                if raw_float is not None:
                    mapped_float = float(raw_float)
            except Exception:
                mapped_float = None
            mapped = _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1)
            if mapped_float is None and mapped >= 0:
                mapped_float = float(mapped)
        if mapped_float is None or int(target_sync) < 0:
            return {
                "sync_available": False,
                "sync_meta": sync_meta,
                "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                "event_layer_sync_timestamp_us": _safe_int(sync_meta.get("event_layer_sync_timestamp_us"), 0) if isinstance(sync_meta, dict) else 0,
                "event_to_bev_sync_delta_ticks": None,
                "event_to_bev_sync_delta_ms": None,
            }
        delta_ticks_f = float(mapped_float) - float(target_sync)
        delta_ms = float(delta_ticks_f) * max(0.001, float(tick_ms or 25.0))
        return {
            "sync_available": True,
            "sync_meta": sync_meta,
            "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "event_layer_sync_timestamp_us": int(event_ts_us),
            "event_to_bev_sync_delta_ticks": int(round(delta_ticks_f)),
            "event_to_bev_sync_delta_ticks_float": round(float(delta_ticks_f), 6),
            "event_to_bev_sync_delta_ms": round(float(delta_ms), 3),
        }

    def _build_once(self, target_sync: int, config: Dict[str, Any]) -> Dict[str, Any]:
        tick_ms = max(0.001, _safe_float(config.get("mvs_sync_tick_ms_est"), 25.0))
        max_sync_ms = max(1.0, _safe_float(config.get("event_layer_max_sync_delta_ms"), 20.0))
        max_sync_ticks = max(0, _safe_int(config.get("event_layer_max_sync_delta_ticks"), 2))
        initial = max(1, _safe_int(config.get("candidate_scan_initial"), 4))
        escalated = max(initial, _safe_int(config.get("candidate_scan_escalated"), 32))
        esc_enable = bool(config.get("candidate_escalate_enable", True))
        esc_abs_ms = max(0.0, _safe_float(config.get("candidate_escalate_abs_ms"), 10.0))
        esc_ratio = max(0.05, min(1.0, _safe_float(config.get("candidate_escalate_ratio"), 0.60)))
        now_us_i = _now_us()
        per_cam: Dict[str, Dict[str, Any]] = {}
        errors = 0
        selected = 0
        synced = 0
        t0 = time.perf_counter()
        for cam in self.cams:
            rd = self.readers.get(cam)
            ring = self.ring_readers.get(cam)
            if rd is None or ring is None:
                continue
            try:
                hint_meta = ring.event_ts_for_mvs_sync(target_sync)
                center_hint_us = _safe_int(hint_meta.get("event_ts_us"), 0) if isinstance(hint_meta, dict) else 0
                item = rd.read_for_target(
                    target_sync,
                    lambda meta, tgt, _cam=cam: self._score_meta(_cam, meta, tgt, tick_ms),
                    center_hint_us=center_hint_us,
                    candidate_limit=initial,
                    candidate_escalate_enable=esc_enable,
                    candidate_escalate_limit=escalated,
                    candidate_escalate_abs_ms=esc_abs_ms,
                    candidate_escalate_ratio=esc_ratio,
                    max_sync_delta_ms=max_sync_ms,
                    read_points=False,
                )
                if item is None:
                    per_cam[cam] = {
                        "draw": False,
                        "reason": "no_sparse_frame",
                        "target_sync_id": int(target_sync),
                    }
                    continue
                meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
                sync_delta_ms = _safe_float(meta.get("event_to_bev_sync_delta_ms"), 999999.0)
                sync_delta_ticks = _safe_int(meta.get("event_to_bev_sync_delta_ticks"), 999999)
                is_synced = (
                    bool(meta.get("sync_available", False))
                    and abs(float(sync_delta_ms)) <= max_sync_ms
                    and abs(int(sync_delta_ticks)) <= max_sync_ticks
                )
                selected += 1
                if is_synced:
                    synced += 1
                per_cam[cam] = {
                    "draw": bool(is_synced),
                    "reason": "within_mvs_sync_delta_ms" if is_synced else "event_mvs_sync_delta_too_large",
                    "frame_id": _safe_int(item.get("frame_id"), -1),
                    "target_sync_id": int(target_sync),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "event_window_start_us": _safe_int(meta.get("event_window_start_us"), 0),
                    "event_window_center_us": _safe_int(meta.get("event_window_center_us"), 0),
                    "event_window_end_us": _safe_int(meta.get("event_window_end_us"), 0),
                    "event_to_bev_sync_delta_ms": round(float(sync_delta_ms), 3),
                    "event_to_bev_sync_delta_ticks": int(sync_delta_ticks),
                    "event_mapped_mvs_sync_index": _safe_int((meta.get("sync_meta") or {}).get("event_mapped_mvs_sync_index") if isinstance(meta.get("sync_meta"), dict) else meta.get("event_mapped_mvs_sync_index"), -1),
                    "event_sync_source": str((meta.get("sync_meta") or {}).get("source", "") if isinstance(meta.get("sync_meta"), dict) else ""),
                    "sparse_point_count": _safe_int(meta.get("sparse_point_count"), -1),
                    "sparse_candidate_scan_count": _safe_int(meta.get("event_layer_sparse_candidate_scan_count"), -1),
                    "sparse_candidate_scan_initial": _safe_int(meta.get("event_layer_sparse_candidate_scan_initial"), -1),
                    "sparse_candidate_scan_escalated": bool(meta.get("event_layer_sparse_candidate_scan_escalated", False)),
                    "sparse_candidate_total_count": _safe_int(meta.get("event_layer_sparse_candidate_total_count"), -1),
                }
            except Exception as exc:
                errors += 1
                per_cam[cam] = {
                    "draw": False,
                    "reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "target_sync_id": int(target_sync),
                }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "enabled": True,
            "mode": "shadow",
            "state": "running",
            "target_sync_id": int(target_sync),
            "updated_wall_us": int(now_us_i),
            "updated_age_ms": 0.0,
            "interval_ms": round(float(self.interval_s * 1000.0), 3),
            "elapsed_ms": round(float(elapsed_ms), 3),
            "selected_cam_count": int(selected),
            "synced_cam_count": int(synced),
            "error_cam_count": int(errors),
            "per_camera": per_cam,
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                target_sync = int(self.target_sync)
                config = dict(self.config)
            if target_sync >= 0:
                try:
                    obj = self._build_once(target_sync, config)
                except Exception as exc:
                    obj = {
                        "enabled": True,
                        "mode": "shadow",
                        "state": "error",
                        "target_sync_id": int(target_sync),
                        "updated_wall_us": _now_us(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "per_camera": {},
                    }
                with self.lock:
                    self.snapshot_obj = obj
            self.stop_event.wait(self.interval_s)


class EventLayerPresentationWorker:
    """CPU-side sparse event presentation bundle builder.

    The worker prepares sparse event vertices for a BEV presentation target, but
    it never touches OpenGL. The GUI thread remains the only owner of GL upload
    and draw calls.
    """

    def __init__(
        self,
        cams: List[str],
        sync_root: str,
        sparse_sidecar_template: str,
        sync_map_ring_template: str,
        poll_interval_ms: float,
        interval_ms: float,
    ):
        self.cams = [str(cam) for cam in cams]
        self.sync_root = str(sync_root or "sync_ipc")
        self.sparse_sidecar_template = str(sparse_sidecar_template or "event_overlay_sparse_{camera}_history.json")
        self.sync_map_ring_template = str(sync_map_ring_template or "event_sync_map_ring_{camera}.json")
        self.poll_interval_ms = max(0.0, float(poll_interval_ms or 0.0))
        self.interval_s = max(0.005, float(interval_ms or 33.333) / 1000.0)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="bev-event-presentation-worker", daemon=True)
        self.started = False
        self.target_sync = -1
        self.config: Dict[str, Any] = {}
        self.bundle_obj: Dict[str, Any] = {
            "enabled": True,
            "mode": "shadow",
            "state": "init",
            "target_sync_id": -1,
            "per_camera": {},
        }
        self.bundle_ring: Dict[int, Dict[str, Any]] = {}
        self.bundle_ring_order: List[int] = []
        self.last_build_targets: List[int] = []
        self.last_skipped_cached_targets: List[int] = []
        self.status_obj: Dict[str, Any] = {
            "enabled": True,
            "mode": "shadow",
            "state": "init",
            "target_sync_id": -1,
            "per_camera": {},
        }
        self.elapsed_samples_ms: List[float] = []
        self.readers: Dict[str, SparseEventOverlayReader] = {
            cam: SparseEventOverlayReader(cam, self.sync_root, self.sparse_sidecar_template, self.poll_interval_ms)
            for cam in self.cams
        }
        self.ring_readers: Dict[str, EventSyncMapRingReader] = {
            cam: EventSyncMapRingReader(cam, self.sync_root, self.sync_map_ring_template)
            for cam in self.cams
        }

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread.start()

    def submit(self, target_sync: int, config: Dict[str, Any]) -> None:
        with self.lock:
            self.target_sync = int(target_sync)
            self.config = dict(config)
            self.interval_s = max(0.005, _safe_float(self.config.get("interval_ms"), self.interval_s * 1000.0) / 1000.0)

    def snapshot(self, target_sync: int = -1, tolerance_ticks: int = 0) -> Dict[str, Any]:
        with self.lock:
            target_i = _safe_int(target_sync, -1)
            if target_i >= 0:
                bundle, lookup = self._select_bundle_locked(target_i, int(tolerance_ticks or 0))
                obj = self._status_from_bundle(bundle)
                obj.update(lookup)
            else:
                obj = copy.deepcopy(self.status_obj)
                obj.update(self._ring_status_locked())
        updated_us = _safe_int(obj.get("updated_wall_us"), 0)
        if updated_us > 0:
            obj["updated_age_ms"] = round(max(0.0, (_now_us() - int(updated_us)) / 1000.0), 3)
        return obj

    def _bundle_copy_locked(self, src: Dict[str, Any]) -> Dict[str, Any]:
        bundle = dict(src)
        per_camera: Dict[str, Any] = {}
        raw_per = src.get("per_camera", {})
        if isinstance(raw_per, dict):
            for cam, rec in raw_per.items():
                if isinstance(rec, dict):
                    per_camera[str(cam)] = dict(rec)
        bundle["per_camera"] = per_camera
        return bundle

    def _ring_status_locked(self) -> Dict[str, Any]:
        targets = sorted(int(x) for x in self.bundle_ring.keys())
        latest_target = _safe_int(self.bundle_obj.get("target_sync_id"), -1) if isinstance(self.bundle_obj, dict) else -1
        return {
            "bundle_ring_count": int(len(targets)),
            "bundle_ring_targets": list(targets[-12:]),
            "cached_target_min": int(targets[0]) if targets else -1,
            "cached_target_max": int(targets[-1]) if targets else -1,
            "latest_bundle_target_sync_id": int(latest_target),
            "last_build_targets": list(self.last_build_targets),
            "last_skipped_cached_targets": list(self.last_skipped_cached_targets),
        }

    def _store_bundle_locked(self, obj: Dict[str, Any], ring_size: int) -> None:
        target = _safe_int(obj.get("target_sync_id"), -1)
        if target < 0:
            return
        self.bundle_ring[int(target)] = obj
        if int(target) in self.bundle_ring_order:
            try:
                self.bundle_ring_order.remove(int(target))
            except ValueError:
                pass
        self.bundle_ring_order.append(int(target))
        max_size = max(1, int(ring_size or 1))
        while len(self.bundle_ring_order) > max_size:
            old = self.bundle_ring_order.pop(0)
            self.bundle_ring.pop(int(old), None)
        self.bundle_obj = obj

    def _select_bundle_locked(self, target_sync: int, tolerance_ticks: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        target = int(target_sync)
        tolerance = max(0, int(tolerance_ticks or 0))
        selected: Optional[Dict[str, Any]] = None
        state = "latest"
        if target >= 0 and self.bundle_ring:
            if target in self.bundle_ring:
                selected = self.bundle_ring.get(target)
                state = "exact"
            else:
                candidates = [
                    int(t)
                    for t in self.bundle_ring.keys()
                    if abs(int(t) - int(target)) <= tolerance
                ]
                if candidates:
                    candidates.sort(key=lambda t: (abs(int(t) - int(target)), -_safe_int(self.bundle_ring.get(int(t), {}).get("updated_wall_us"), 0)))
                    selected = self.bundle_ring.get(int(candidates[0]))
                    state = "near"
                else:
                    selected = self.bundle_obj
                    state = "miss"
        else:
            selected = self.bundle_obj
        if not isinstance(selected, dict):
            selected = {}
        bundle = self._bundle_copy_locked(selected)
        bundle_target = _safe_int(bundle.get("target_sync_id"), -1)
        lookup = {
            "bundle_lookup_state": str(state),
            "bundle_lookup_exact": bool(state == "exact"),
            "bundle_lookup_near": bool(state == "near"),
            "bundle_lookup_miss": bool(state == "miss"),
            "requested_target_sync_id": int(target),
            "bundle_target_delta_ticks": int(bundle_target) - int(target) if target >= 0 and bundle_target >= 0 else -999999,
        }
        lookup.update(self._ring_status_locked())
        bundle.update(lookup)
        return bundle, lookup

    def latest_bundle(self, target_sync: int = -1, tolerance_ticks: int = 0) -> Dict[str, Any]:
        with self.lock:
            bundle, _lookup = self._select_bundle_locked(int(target_sync), int(tolerance_ticks))
            return bundle

    def stop(self, timeout_s: float = 1.0) -> None:
        self.stop_event.set()
        if self.started and self.thread.is_alive():
            self.thread.join(max(0.0, float(timeout_s)))
        for rd in list(self.readers.values()):
            try:
                rd.close()
            except Exception:
                pass

    def _elapsed_stats(self) -> Dict[str, Any]:
        vals = sorted(float(x) for x in self.elapsed_samples_ms if float(x) >= 0.0)
        if not vals:
            return {"build_p50_ms": 0.0, "build_p95_ms": 0.0, "build_max_ms": 0.0, "build_sample_count": 0}
        n = len(vals)

        def _pct(p: float) -> float:
            if n <= 1:
                return vals[0]
            idx = int(round((n - 1) * float(p)))
            idx = max(0, min(n - 1, idx))
            return vals[idx]

        return {
            "build_p50_ms": round(float(_pct(0.50)), 3),
            "build_p95_ms": round(float(_pct(0.95)), 3),
            "build_max_ms": round(float(vals[-1]), 3),
            "build_sample_count": int(n),
        }

    def _field_xy_to_ndc_batch(self, x: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fw = max(1e-6, _safe_float(config.get("field_width_m"), 33.0))
        fh = max(1e-6, _safe_float(config.get("field_height_m"), 22.0))
        origin_x = _safe_float(config.get("field_origin_x_m"), 0.0)
        origin_y = _safe_float(config.get("field_origin_y_m"), 0.0)
        sx = (x.astype(np.float64) - origin_x) / fw
        sy = (y.astype(np.float64) - origin_y) / fh
        valid = np.isfinite(sx) & np.isfinite(sy) & (sx >= -0.05) & (sx <= 1.05) & (sy >= -0.05) & (sy <= 1.05)
        mode = _normalize_transform_mode(config.get("field_transform_mode", "normal"))
        qx = sx.copy()
        qy = sy.copy()
        if mode == "flip_x":
            qx, qy = 1.0 - sx, sy
        elif mode == "flip_y":
            qx, qy = sx, 1.0 - sy
        elif mode == "flip_xy":
            qx, qy = 1.0 - sx, 1.0 - sy
        elif mode == "swap_xy":
            qx, qy = sy, sx
        elif mode == "swap_xy_flip_x":
            qx, qy = sy, 1.0 - sx
        elif mode == "swap_xy_flip_y":
            qx, qy = 1.0 - sy, sx
        elif mode == "swap_xy_flip_xy":
            qx, qy = 1.0 - sy, 1.0 - sx
        ndc_x = qx * 2.0 - 1.0
        ndc_y = (1.0 - qy) * 2.0 - 1.0
        valid &= np.isfinite(ndc_x) & np.isfinite(ndc_y) & (ndc_x >= -1.1) & (ndc_x <= 1.1) & (ndc_y >= -1.1) & (ndc_y <= 1.1)
        return ndc_x.astype(np.float32), ndc_y.astype(np.float32), valid

    def _project_sparse_points_to_ndc(self, cam: str, points: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "input_points": int(len(points)) if hasattr(points, "__len__") else 0,
            "projected_points": 0,
            "visible_points": 0,
            "skip_reason": "",
        }
        if points is None or len(points) <= 0:
            meta["skip_reason"] = "no_points"
            return np.empty((0, 2), dtype=np.float32), meta
        event_to_mvs = config.get("event_to_mvs_by_cam", {})
        img_to_field = config.get("img_to_field_by_cam", {})
        h_event_to_mvs = event_to_mvs.get(cam) if isinstance(event_to_mvs, dict) else None
        h_img_to_field = img_to_field.get(cam) if isinstance(img_to_field, dict) else None
        if h_event_to_mvs is None or h_img_to_field is None:
            meta["skip_reason"] = "missing_homography"
            return np.empty((0, 2), dtype=np.float32), meta
        max_draw = max(1, _safe_int(config.get("max_points_draw"), 20000))
        pts = points[:max_draw]
        xs = pts["x"].astype(np.float64)
        ys = pts["y"].astype(np.float64)
        ones = np.ones_like(xs, dtype=np.float64)
        ev = np.vstack([xs, ys, ones])
        try:
            mvs = np.asarray(h_event_to_mvs, dtype=np.float64).reshape(3, 3) @ ev
            z = mvs[2, :]
            z = np.where(np.abs(z) < 1e-9, np.nan, z)
            mvs_x = mvs[0, :] / z
            mvs_y = mvs[1, :] / z
            mvs_h = np.vstack([mvs_x, mvs_y, ones])
            fld = np.asarray(h_img_to_field, dtype=np.float64).reshape(3, 3) @ mvs_h
            fz = fld[2, :]
            fz = np.where(np.abs(fz) < 1e-9, np.nan, fz)
            fx = fld[0, :] / fz
            fy = fld[1, :] / fz
        except Exception as exc:
            meta["skip_reason"] = f"project_error:{type(exc).__name__}"
            return np.empty((0, 2), dtype=np.float32), meta
        meta["projected_points"] = int(len(fx))
        ndc_x, ndc_y, valid = self._field_xy_to_ndc_batch(fx, fy, config)
        if not np.any(valid):
            meta["skip_reason"] = "all_projected_out_of_field"
            return np.empty((0, 2), dtype=np.float32), meta
        vertices = np.column_stack([ndc_x[valid], ndc_y[valid]]).astype(np.float32, copy=False)
        meta["visible_points"] = int(vertices.shape[0])
        meta["truncated_by_draw_limit"] = bool(len(points) > max_draw)
        meta["max_points_draw"] = int(max_draw)
        meta["skip_reason"] = "ok"
        return vertices, meta

    def _score_meta(self, cam: str, meta: Dict[str, Any], target_sync: int, tick_ms: float) -> Dict[str, Any]:
        event_ts_us = _event_layer_sync_timestamp_us(meta, _safe_int(meta.get("timestamp_us"), 0))
        mapped_float: Optional[float] = None
        mapped = _safe_int(meta.get("event_mapped_mvs_sync_index", meta.get("mapped_mvs_sync_index")), -1)
        if mapped >= 0:
            mapped_float = float(mapped)
            sync_meta = {
                "source": "event_layer_history_sidecar",
                "event_sync_quality": "good",
                "event_mapped_mvs_sync_index": int(mapped),
                "event_mapped_mvs_sync_index_float": float(mapped),
                "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                "event_sync_estimated_from_event_ts_us": int(event_ts_us),
            }
        else:
            sync_meta = self.ring_readers[str(cam)].map_event_ts(event_ts_us)
            raw_float = sync_meta.get("event_mapped_mvs_sync_index_float")
            try:
                if raw_float is not None:
                    mapped_float = float(raw_float)
            except Exception:
                mapped_float = None
            mapped = _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1)
            if mapped_float is None and mapped >= 0:
                mapped_float = float(mapped)
        if mapped_float is None or int(target_sync) < 0:
            return {
                "sync_available": False,
                "sync_meta": sync_meta,
                "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                "event_layer_sync_timestamp_us": _safe_int(sync_meta.get("event_layer_sync_timestamp_us"), 0) if isinstance(sync_meta, dict) else 0,
                "event_to_bev_sync_delta_ticks": None,
                "event_to_bev_sync_delta_ms": None,
            }
        delta_ticks_f = float(mapped_float) - float(target_sync)
        delta_ms = float(delta_ticks_f) * max(0.001, float(tick_ms or 25.0))
        return {
            "sync_available": True,
            "sync_meta": sync_meta,
            "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "event_layer_sync_timestamp_us": int(event_ts_us),
            "event_to_bev_sync_delta_ticks": int(round(delta_ticks_f)),
            "event_to_bev_sync_delta_ticks_float": round(float(delta_ticks_f), 6),
            "event_to_bev_sync_delta_ms": round(float(delta_ms), 3),
        }

    def _sync_decision(
        self,
        *,
        cam: str,
        target_sync: int,
        age_ms: float,
        nonzero: int,
        max_luma: int,
        meta: Dict[str, Any],
        item: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        max_age_ms = max(1.0, _safe_float(config.get("max_age_ms"), 2500.0))
        sync_normal_ms = max(1.0, _safe_float(config.get("sync_normal_ms"), 50.0))
        sync_warn_ms = max(sync_normal_ms, _safe_float(config.get("sync_warn_ms"), 80.0))
        sync_hard_ms = max(sync_warn_ms, _safe_float(config.get("sync_hard_ms"), 80.0))
        max_sync_ticks = max(0, _safe_int(config.get("max_sync_delta_ticks"), 2))
        fade_scale = _clip01(_safe_float(config.get("fade_alpha_scale"), 0.25))
        hard_hold_enable = bool(config.get("hold_last_good_on_hard_outlier", True))
        score = self._score_meta(cam, meta, target_sync, _safe_float(config.get("mvs_sync_tick_ms_est"), 25.0))
        sync_meta = score.get("sync_meta", {}) if isinstance(score.get("sync_meta"), dict) else {}
        sync_delta_ticks = score.get("event_to_bev_sync_delta_ticks", None)
        sync_delta_ms = score.get("event_to_bev_sync_delta_ms", None)
        try:
            sync_delta_ms_f = float(sync_delta_ms) if sync_delta_ms is not None else None
        except Exception:
            sync_delta_ms_f = None

        def _finish(valid: bool, alpha_scale: float, state: str, reason: str, level: str, hold_candidate: bool = False, **extra: Any) -> Dict[str, Any]:
            abs_delta_ms = abs(float(sync_delta_ms_f)) if sync_delta_ms_f is not None else None
            out: Dict[str, Any] = {
                "valid": bool(valid),
                "alpha_scale": float(alpha_scale),
                "state": str(state),
                "reason": str(reason),
                "sync_mode": "mvs_tick_guarded",
                "sync_meta": dict(sync_meta),
                "event_to_bev_sync_delta_ticks": sync_delta_ticks,
                "event_to_bev_sync_delta_ms": sync_delta_ms_f,
                "event_layer_sync_display_state": str(state),
                "event_layer_sync_display_level": str(level),
                "event_layer_sync_display_reason": str(reason),
                "event_layer_sync_abs_delta_ms": None if abs_delta_ms is None else round(float(abs_delta_ms), 3),
                "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                "event_layer_hold_last_good_candidate": bool(hold_candidate),
                "event_layer_hold_last_good_on_hard_outlier": bool(hard_hold_enable),
            }
            out.update(extra)
            return out

        if int(nonzero) <= 0 or int(max_luma) <= 0:
            return _finish(False, 0.0, "empty", "empty", "EMPTY")
        if not bool(score.get("sync_available", False)):
            return _finish(False, 0.0, "missing_event_mvs_sync", "missing_event_mapped_mvs_sync_index", "MISSING")
        sync_quality = str(sync_meta.get("event_sync_quality", "") or "").strip().lower()
        if sync_quality in {"missing", "unusable", "invalid"}:
            return _finish(False, 0.0, "missing_event_mvs_sync", f"sync_quality_{sync_quality}", "MISSING")
        if age_ms >= 0.0 and float(age_ms) > max_age_ms:
            return _finish(False, 0.0, "stale", "event_layer_wall_age_stale", "STALE")
        if sync_delta_ms_f is None:
            return _finish(False, 0.0, "missing_event_mvs_sync_delta_ms", "missing_event_to_bev_sync_delta_ms", "MISSING")
        synced_ticks = bool(abs(int(sync_delta_ticks or 0)) <= max_sync_ticks)
        abs_sync_ms = abs(float(sync_delta_ms_f))
        common = {
            "event_to_bev_sync_delta_ticks": int(sync_delta_ticks or 0),
            "event_to_bev_sync_delta_ms": round(float(sync_delta_ms_f), 3),
            "wall_age_relaxed": False,
            "wall_age_ms": round(float(age_ms), 3) if age_ms >= 0.0 else -1.0,
            "max_age_ms": round(float(max_age_ms), 3),
            "event_layer_ticks_within_limit": bool(synced_ticks),
        }
        if abs_sync_ms <= sync_normal_ms:
            return _finish(True, 1.0, "synced" if synced_ticks else "synced_tick_warn", "within_display_normal_ms" if synced_ticks else "within_display_normal_ms_ticks_outlier", "OK" if synced_ticks else "WARN", **common)
        if abs_sync_ms <= sync_warn_ms:
            return _finish(True, 1.0, "warn_displayed", "event_mvs_sync_delta_warn_display", "WARN", **common)
        if abs_sync_ms <= sync_hard_ms:
            alpha = float(fade_scale) if fade_scale > 0.0 else 1.0
            return _finish(True, alpha, "warn_displayed", "event_mvs_sync_delta_between_warn_and_hard_display", "WARN", **common)
        return _finish(False, 0.0, "hard_outlier", "event_mvs_sync_delta_hard_outlier", "HARD", hold_candidate=hard_hold_enable, **common)

    def _status_from_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        obj = {
            k: v
            for k, v in bundle.items()
            if k not in {"per_camera"}
        }
        per: Dict[str, Any] = {}
        raw_per = bundle.get("per_camera", {})
        if isinstance(raw_per, dict):
            for cam, rec in raw_per.items():
                if not isinstance(rec, dict):
                    continue
                per[str(cam)] = {
                    k: v
                    for k, v in rec.items()
                    if k not in {"vertices", "points"}
                }
        obj["per_camera"] = per
        return obj

    def _target_window(self, target_sync: int, config: Dict[str, Any]) -> List[int]:
        target = int(target_sync)
        if target < 0:
            return []
        lookback = max(0, _safe_int(config.get("prebuild_lookback_ticks"), 0))
        lookahead = max(0, _safe_int(config.get("prebuild_lookahead_ticks"), 0))
        mode = str(config.get("mode", "shadow") or "shadow").strip().lower().replace("-", "_")
        targets: List[int] = []
        for i in range(lookahead, 0, -1):
            targets.append(target + i)
        targets.append(target)
        for i in range(1, lookback + 1):
            targets.append(target - i)
        for i in range(1, lookahead + 1):
            targets.append(target + i)
        out: List[int] = []
        seen = set()
        for raw in targets:
            val = int(raw)
            if val < 0 or val in seen:
                continue
            seen.add(val)
            out.append(val)
        return out

    def _build_once(self, target_sync: int, config: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(config.get("mode", "shadow") or "shadow").strip().lower().replace("-", "_")
        read_points = _safe_bool(config.get("read_points"), bool(mode == "active"))
        payload_policy = "vertices_for_active" if read_points else "metadata_only_no_vertices"
        tick_ms = max(0.001, _safe_float(config.get("mvs_sync_tick_ms_est"), 25.0))
        max_sync_ms = max(1.0, _safe_float(config.get("max_sync_delta_ms"), 50.0))
        initial = max(1, _safe_int(config.get("candidate_scan_initial"), 4))
        escalated = max(initial, _safe_int(config.get("candidate_scan_escalated"), 32))
        esc_enable = bool(config.get("candidate_escalate_enable", True))
        esc_abs_ms = max(0.0, _safe_float(config.get("candidate_escalate_abs_ms"), 10.0))
        esc_ratio = max(0.05, min(1.0, _safe_float(config.get("candidate_escalate_ratio"), 0.60)))
        now_us_i = _now_us()
        per_cam: Dict[str, Dict[str, Any]] = {}
        selected = 0
        synced = 0
        errors = 0
        input_total = 0
        visible_total = 0
        projected_total = 0
        read_elapsed_ms = 0.0
        sync_elapsed_ms = 0.0
        project_elapsed_ms = 0.0
        t0 = time.perf_counter()
        for cam in self.cams:
            rd = self.readers.get(cam)
            ring = self.ring_readers.get(cam)
            if rd is None or ring is None:
                continue
            try:
                hint_meta = ring.event_ts_for_mvs_sync(target_sync)
                center_hint_us = _safe_int(hint_meta.get("event_ts_us"), 0) if isinstance(hint_meta, dict) else 0
                read_t0 = time.perf_counter()
                item = rd.read_for_target(
                    target_sync,
                    lambda meta, tgt, _cam=cam: self._score_meta(_cam, meta, tgt, tick_ms),
                    center_hint_us=center_hint_us,
                    candidate_limit=initial,
                    candidate_escalate_enable=esc_enable,
                    candidate_escalate_limit=escalated,
                    candidate_escalate_abs_ms=esc_abs_ms,
                    candidate_escalate_ratio=esc_ratio,
                    max_sync_delta_ms=max_sync_ms,
                    read_points=bool(read_points),
                )
                read_ms = (time.perf_counter() - read_t0) * 1000.0
                read_elapsed_ms += float(read_ms)
                if item is None:
                    per_cam[cam] = {
                        "draw": False,
                        "reason": "no_sparse_frame",
                        "payload_policy": str(payload_policy),
                        "target_sync_id": int(target_sync),
                        "last_read_error": str(getattr(rd, "last_read_error", "")),
                        "read_ms": round(float(read_ms), 3),
                    }
                    continue
                meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
                points = np.asarray(item.get("points"), dtype=_SPARSE_POINT_DTYPE)
                if read_points:
                    point_count = int(points.shape[0]) if points.ndim == 1 else 0
                else:
                    point_count = _safe_int(
                        meta.get("sparse_point_count", meta.get("overlay_nonzero_pixels", meta.get("event_layer_sparse_point_count"))),
                        0,
                    )
                input_total += int(point_count)
                pub_us = _safe_int(meta.get("published_wall_us"), 0)
                age_ms = max(0.0, (int(now_us_i) - int(pub_us)) / 1000.0) if pub_us > 1_000_000_000_000 else -1.0
                nonzero = _safe_int(meta.get("overlay_nonzero_pixels", point_count), point_count)
                max_luma = _safe_int(meta.get("overlay_max_luma", 255 if point_count > 0 else 0), 255 if point_count > 0 else 0)
                sync_t0 = time.perf_counter()
                decision = self._sync_decision(
                    cam=cam,
                    target_sync=target_sync,
                    age_ms=age_ms,
                    nonzero=nonzero,
                    max_luma=max_luma,
                    meta=meta,
                    item=item,
                    config=config,
                )
                sync_ms = (time.perf_counter() - sync_t0) * 1000.0
                sync_elapsed_ms += float(sync_ms)
                valid = bool(decision.get("valid", False))
                alpha_scale = float(decision.get("alpha_scale", 0.0) or 0.0)
                projection_meta: Dict[str, Any] = {}
                vertices = np.empty((0, 2), dtype=np.float32)
                project_ms = 0.0
                if valid and read_points:
                    project_t0 = time.perf_counter()
                    vertices, projection_meta = self._project_sparse_points_to_ndc(cam, points, config)
                    project_ms = (time.perf_counter() - project_t0) * 1000.0
                    project_elapsed_ms += float(project_ms)
                    if vertices.shape[0] <= 0:
                        valid = False
                        alpha_scale = 0.0
                        decision["state"] = "no_visible_points"
                        decision["reason"] = str(projection_meta.get("skip_reason", "no_visible_points"))
                elif valid:
                    projection_meta = {
                        "input_points": int(point_count),
                        "projected_points": 0,
                        "visible_points": 0,
                        "skip_reason": "shadow_metadata_only",
                    }
                selected += 1
                if valid:
                    synced += 1
                    visible_total += int(vertices.shape[0])
                    projected_total += int(projection_meta.get("projected_points", vertices.shape[0]))
                sync_meta = decision.get("sync_meta", {}) if isinstance(decision.get("sync_meta"), dict) else {}
                event_to_bev_sync_delta_ticks = decision.get("event_to_bev_sync_delta_ticks", None)
                event_to_bev_sync_delta_ms = decision.get("event_to_bev_sync_delta_ms", None)
                display_state = str(decision.get("event_layer_sync_display_state", decision.get("state", "")) or "")
                display_level = str(decision.get("event_layer_sync_display_level", "OK" if valid else "HARD") or "")
                fid = _safe_int(item.get("frame_id"), -1)
                shape = item.get("shape", [0, 0, 1])
                source_w = _safe_int(shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 0, 0)
                source_h = _safe_int(shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 0, 0)
                if source_w <= 0:
                    source_w = _safe_int(meta.get("width", meta.get("event_layer_width", 0)), 0)
                if source_h <= 0:
                    source_h = _safe_int(meta.get("height", meta.get("event_layer_height", 0)), 0)
                rec: Dict[str, Any] = {
                    "backend": "sparse_gl",
                    "draw": bool(valid),
                    "reason": "drawing" if valid and alpha_scale >= 0.999 else ("drawing_faded" if valid else str(decision.get("reason", "invalid"))),
                    "vertices": vertices if valid else np.empty((0, 2), dtype=np.float32),
                    "payload_policy": str(payload_policy),
                    "read_points": bool(read_points),
                    "width": int(source_w),
                    "height": int(source_h),
                    "channels": 1,
                    "frame_id": int(fid),
                    "event_layer_frame_id": int(fid),
                    "age_ms": round(float(age_ms), 3),
                    "event_layer_age_ms": round(float(age_ms), 3),
                    "event_ts_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "published_wall_us": int(pub_us),
                    "bev_render_wall_us": int(now_us_i),
                    "event_layer_sync_state": str(decision.get("state", "")),
                    "event_layer_sync_display_state": str(display_state),
                    "event_layer_sync_display_level": str(display_level),
                    "event_layer_sync_abs_delta_ms": decision.get("event_layer_sync_abs_delta_ms"),
                    "sync_reason": str(decision.get("reason", "")),
                    "sync_mode": str(decision.get("sync_mode", "mvs_tick_guarded")),
                    "event_layer_hold_last_good_candidate": bool(decision.get("event_layer_hold_last_good_candidate", False)),
                    "event_mapped_mvs_sync_index": _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1),
                    "event_mapped_mvs_sync_index_float": sync_meta.get("event_mapped_mvs_sync_index_float"),
                    "event_to_bev_sync_delta_ticks": -999999 if event_to_bev_sync_delta_ticks is None else int(event_to_bev_sync_delta_ticks),
                    "event_to_bev_sync_delta_ms": -999999.0 if event_to_bev_sync_delta_ms is None else round(float(event_to_bev_sync_delta_ms), 3),
                    "event_sync_source": str(sync_meta.get("source", "")),
                    "event_sync_quality": str(sync_meta.get("event_sync_quality", "")),
                    "event_sync_ring_path": str(sync_meta.get("event_sync_ring_path", "")),
                    "event_sync_ring_age_ms": _safe_float(sync_meta.get("event_sync_ring_age_ms"), -1.0),
                    "event_sync_ring_record_count": _safe_int(sync_meta.get("event_sync_ring_record_count"), -1),
                    "event_sync_nearest_abs_delta_ms": _safe_float(sync_meta.get("event_sync_nearest_abs_delta_ms"), -1.0),
                    "alpha_scale": round(float(alpha_scale), 3),
                    "target_sync_id": int(target_sync),
                    "history_selected": bool(item.get("history_selected", False)),
                    "history_source": str(meta.get("event_layer_history_source", "")),
                    "history_best_abs_delta_ms": _safe_float(item.get("history_best_abs_delta_ms"), -1.0),
                    "event_window_center_us": _safe_int(meta.get("event_window_center_us"), 0),
                    "event_window_start_us": _safe_int(meta.get("event_window_start_us"), 0),
                    "event_window_end_us": _safe_int(meta.get("event_window_end_us"), 0),
                    "nonzero_pixels": int(nonzero),
                    "max_luma": int(max_luma),
                    "sparse_point_count": int(point_count),
                    "sparse_projected_points": int(projection_meta.get("projected_points", 0)),
                    "sparse_visible_points": int(vertices.shape[0]) if valid else 0,
                    "sparse_dropped_point_count": _safe_int(meta.get("sparse_dropped_point_count"), _safe_int(meta.get("event_layer_sparse_dropped_point_count"), 0)),
                    "sparse_entry_count": _safe_int(meta.get("event_layer_sparse_entry_count"), -1),
                    "sparse_slots": _safe_int(meta.get("event_layer_sparse_slots"), -1),
                    "sparse_write_count": _safe_int(meta.get("event_layer_sparse_write_count"), -1),
                    "sparse_overwrite_count": _safe_int(meta.get("event_layer_sparse_overwrite_count"), -1),
                    "sparse_coverage_ms": round(_safe_float(meta.get("event_layer_sparse_coverage_ms"), -1.0), 3),
                    "sparse_cache_hit": bool(meta.get("event_layer_sparse_cache_hit", False)),
                    "sparse_cache_age_ms": round(_safe_float(meta.get("event_layer_sparse_cache_age_ms"), -1.0), 3),
                    "sparse_poll_interval_ms": round(_safe_float(meta.get("event_layer_sparse_poll_interval_ms"), 0.0), 3),
                    "sparse_sidecar_path": str(meta.get("event_layer_sparse_sidecar", rd.sidecar_path)),
                    "sparse_center_hint_us": _safe_int(meta.get("event_layer_sparse_center_hint_us"), 0),
                    "sparse_candidate_scan_count": _safe_int(meta.get("event_layer_sparse_candidate_scan_count"), -1),
                    "sparse_candidate_scan_initial": _safe_int(meta.get("event_layer_sparse_candidate_scan_initial"), -1),
                    "sparse_candidate_scan_escalated": bool(meta.get("event_layer_sparse_candidate_scan_escalated", False)),
                    "sparse_candidate_scan_escalate_reason": str(meta.get("event_layer_sparse_candidate_scan_escalate_reason", "")),
                    "sparse_candidate_total_count": _safe_int(meta.get("event_layer_sparse_candidate_total_count"), -1),
                    "read_ms": round(float(read_ms), 3),
                    "sync_gate_ms": round(float(sync_ms), 3),
                    "project_ms": round(float(project_ms), 3),
                    "projection": dict(projection_meta),
                    "meta": meta,
                    "error": "",
                }
                per_cam[cam] = rec
            except Exception as exc:
                errors += 1
                per_cam[cam] = {
                    "draw": False,
                    "reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "target_sync_id": int(target_sync),
                }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.elapsed_samples_ms.append(float(elapsed_ms))
        if len(self.elapsed_samples_ms) > 240:
            self.elapsed_samples_ms = self.elapsed_samples_ms[-240:]
        bundle: Dict[str, Any] = {
            "enabled": True,
            "mode": str(mode),
            "state": "running",
            "payload_policy": str(payload_policy),
            "read_points": bool(read_points),
            "target_sync_id": int(target_sync),
            "updated_wall_us": int(now_us_i),
            "updated_age_ms": 0.0,
            "interval_ms": round(float(self.interval_s * 1000.0), 3),
            "elapsed_ms": round(float(elapsed_ms), 3),
            "selected_cam_count": int(selected),
            "synced_cam_count": int(synced),
            "error_cam_count": int(errors),
            "sparse_input_points": int(input_total),
            "sparse_projected_points": int(projected_total),
            "sparse_visible_points": int(visible_total),
            "sparse_read_elapsed_ms": round(float(read_elapsed_ms), 3),
            "sparse_sync_gate_elapsed_ms": round(float(sync_elapsed_ms), 3),
            "sparse_project_elapsed_ms": round(float(project_elapsed_ms), 3),
            "per_camera": per_cam,
        }
        bundle.update(self._elapsed_stats())
        return bundle

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                target_sync = int(self.target_sync)
                config = dict(self.config)
            if target_sync >= 0:
                targets = self._target_window(target_sync, config)
                max_per_cycle = max(1, _safe_int(config.get("prebuild_max_per_cycle"), 1))
                ring_size = max(1, _safe_int(config.get("bundle_ring_size"), 8))
                interval_ms = max(1.0, _safe_float(config.get("interval_ms"), self.interval_s * 1000.0))
                cache_reuse_ms = max(0.0, _safe_float(config.get("bundle_cache_reuse_ms"), interval_ms * 0.5))
                built: List[Dict[str, Any]] = []
                built_targets: List[int] = []
                skipped_cached: List[int] = []
                for target in targets:
                    if len(built_targets) >= int(max_per_cycle):
                        break
                    with self.lock:
                        cached = self.bundle_ring.get(int(target))
                        cached_age_ms = max(0.0, (_now_us() - _safe_int(cached.get("updated_wall_us"), 0)) / 1000.0) if isinstance(cached, dict) else 999999.0
                    if isinstance(cached, dict) and cached_age_ms <= cache_reuse_ms:
                        skipped_cached.append(int(target))
                        continue
                    try:
                        obj = self._build_once(int(target), config)
                    except Exception as exc:
                        obj = {
                            "enabled": True,
                            "mode": str(config.get("mode", "shadow") or "shadow"),
                            "state": "error",
                            "target_sync_id": int(target),
                            "updated_wall_us": _now_us(),
                            "error": f"{type(exc).__name__}: {exc}",
                            "per_camera": {},
                        }
                    built.append(obj)
                    built_targets.append(int(target))
                with self.lock:
                    for obj in built:
                        obj["bundle_ring_size"] = int(ring_size)
                        obj["prebuild_lookback_ticks"] = max(0, _safe_int(config.get("prebuild_lookback_ticks"), 0))
                        obj["prebuild_lookahead_ticks"] = max(0, _safe_int(config.get("prebuild_lookahead_ticks"), 0))
                        obj["prebuild_max_per_cycle"] = int(max_per_cycle)
                        obj["bundle_cache_reuse_ms"] = round(float(cache_reuse_ms), 3)
                        self._store_bundle_locked(obj, ring_size)
                    self.last_build_targets = list(built_targets)
                    self.last_skipped_cached_targets = list(skipped_cached)
                    selected = self.bundle_ring.get(int(target_sync), self.bundle_obj)
                    status = self._status_from_bundle(selected if isinstance(selected, dict) else {})
                    status.update(self._ring_status_locked())
                    status["bundle_ring_size"] = int(ring_size)
                    status["prebuild_lookback_ticks"] = max(0, _safe_int(config.get("prebuild_lookback_ticks"), 0))
                    status["prebuild_lookahead_ticks"] = max(0, _safe_int(config.get("prebuild_lookahead_ticks"), 0))
                    status["prebuild_max_per_cycle"] = int(max_per_cycle)
                    status["bundle_cache_reuse_ms"] = round(float(cache_reuse_ms), 3)
                    status["requested_target_sync_id"] = int(target_sync)
                    if not built and skipped_cached:
                        status["state"] = str(status.get("state") or "running")
                        status["last_build_skipped_reason"] = "cached_recent"
                    self.status_obj = status
            self.stop_event.wait(self.interval_s)


class DelayedAlignmentWorker:
    """Metadata-only presentation-bundle builder for Global BEV.

    This worker intentionally prepares only CPU-side selection metadata.  It
    never touches OpenGL, MVS image payloads, or sparse point mmap payloads.
    """

    def __init__(
        self,
        cams: List[str],
        raw_template: str,
        sync_root: str,
        sparse_sidecar_template: str,
        sync_map_ring_template: str,
        config: Dict[str, Any],
    ):
        self.cams = [str(cam) for cam in cams]
        self.raw_template = str(raw_template or "mvs_raw_latest_{camera}")
        self.sync_root = str(sync_root or "sync_ipc")
        self.sparse_sidecar_template = str(sparse_sidecar_template or "event_overlay_sparse_{camera}_history.json")
        self.sync_map_ring_template = str(sync_map_ring_template or "event_sync_map_ring_{camera}.json")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.config = dict(config)
        self.interval_s = max(0.05, _safe_float(self.config.get("interval_ms"), 250.0) / 1000.0)
        self.thread = threading.Thread(target=self._run, name="bev-alignment-shadow", daemon=True)
        self.started = False
        self.bundle_seq = 0
        self.mvs_readers: Dict[str, RawPreviewFrameReader] = {}
        self.mvs_reader_errors: Dict[str, str] = {}
        self.sparse_readers: Dict[str, SparseEventOverlayReader] = {
            cam: SparseEventOverlayReader(
                cam,
                self.sync_root,
                self.sparse_sidecar_template,
                _safe_float(self.config.get("event_layer_history_poll_interval_ms"), 0.0),
            )
            for cam in self.cams
        }
        self.ring_readers: Dict[str, EventSyncMapRingReader] = {
            cam: EventSyncMapRingReader(cam, self.sync_root, self.sync_map_ring_template)
            for cam in self.cams
        }
        self.bundle: Dict[str, Any] = {
            "enabled": True,
            "mode": "shadow",
            "state": "init",
            "bundle_id": 0,
            "target_sync_id": -1,
            "per_camera": {},
        }
        self.elapsed_samples_ms: List[float] = []

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self.stop_event.set()
        if self.started and self.thread.is_alive():
            self.thread.join(max(0.0, float(timeout_s)))
        for rd in list(self.mvs_readers.values()):
            try:
                rd.close()
            except Exception:
                pass
        for rd in list(self.sparse_readers.values()):
            try:
                rd.close()
            except Exception:
                pass

    def update_config(self, config: Dict[str, Any]) -> None:
        with self.lock:
            self.config.update(dict(config))
            self.interval_s = max(0.05, _safe_float(self.config.get("interval_ms"), self.interval_s * 1000.0) / 1000.0)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            obj = copy.deepcopy(self.bundle)
        updated_us = _safe_int(obj.get("updated_wall_us"), 0)
        if updated_us > 0:
            obj["updated_age_ms"] = round(max(0.0, (_now_us() - updated_us) / 1000.0), 3)
        return obj

    def _elapsed_stats(self) -> Dict[str, Any]:
        vals = sorted(float(x) for x in self.elapsed_samples_ms if float(x) >= 0.0)
        if not vals:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0}
        n = len(vals)

        def _pct(p: float) -> float:
            if n <= 1:
                return vals[0]
            idx = int(round((n - 1) * float(p)))
            idx = max(0, min(n - 1, idx))
            return vals[idx]

        return {
            "p50_ms": round(float(_pct(0.50)), 3),
            "p95_ms": round(float(_pct(0.95)), 3),
            "max_ms": round(float(vals[-1]), 3),
            "n": int(n),
        }

    def _open_mvs_reader(self, cam: str) -> Optional[RawPreviewFrameReader]:
        rd = self.mvs_readers.get(cam)
        if rd is not None:
            return rd
        name = self.raw_template.format(camera=cam, cam=cam, id=cam)
        try:
            rd = RawPreviewFrameReader(name)
            self.mvs_readers[cam] = rd
            self.mvs_reader_errors.pop(cam, None)
            return rd
        except Exception as exc:
            self.mvs_reader_errors[cam] = f"{type(exc).__name__}: {exc}"
            return None

    def _score_event_meta(self, cam: str, meta: Dict[str, Any], target_sync: int, tick_ms: float) -> Dict[str, Any]:
        event_ts_us = _event_layer_sync_timestamp_us(meta, _safe_int(meta.get("timestamp_us"), 0))
        mapped_float: Optional[float] = None
        mapped = _safe_int(meta.get("event_mapped_mvs_sync_index", meta.get("mapped_mvs_sync_index")), -1)
        if mapped >= 0:
            mapped_float = float(mapped)
            sync_meta = {
                "source": "event_layer_history_sidecar",
                "event_sync_quality": "good",
                "event_mapped_mvs_sync_index": int(mapped),
                "event_mapped_mvs_sync_index_float": float(mapped),
                "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                "event_sync_estimated_from_event_ts_us": int(event_ts_us),
            }
        else:
            sync_meta = self.ring_readers[str(cam)].map_event_ts(event_ts_us)
            raw_float = sync_meta.get("event_mapped_mvs_sync_index_float")
            try:
                if raw_float is not None:
                    mapped_float = float(raw_float)
            except Exception:
                mapped_float = None
            mapped = _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1)
            if mapped_float is None and mapped >= 0:
                mapped_float = float(mapped)
        if mapped_float is None or int(target_sync) < 0:
            return {
                "sync_available": False,
                "sync_meta": sync_meta,
                "event_to_bev_sync_delta_ticks": None,
                "event_to_bev_sync_delta_ms": None,
            }
        delta_ticks_f = float(mapped_float) - float(target_sync)
        delta_ms = float(delta_ticks_f) * max(0.001, float(tick_ms or 25.0))
        return {
            "sync_available": True,
            "sync_meta": sync_meta,
            "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "event_layer_sync_timestamp_us": int(event_ts_us),
            "event_to_bev_sync_delta_ticks": int(round(delta_ticks_f)),
            "event_to_bev_sync_delta_ticks_float": round(float(delta_ticks_f), 6),
            "event_to_bev_sync_delta_ms": round(float(delta_ms), 3),
        }

    def _build_mvs(self, cfg: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
        sync_mode = str(cfg.get("sync_mode", "nearest_trigger") or "nearest_trigger")
        key = "tick_id" if sync_mode == "nearest_trigger" else "frame_id"
        tick_ms = max(0.001, _safe_float(cfg.get("mvs_sync_tick_ms_est"), 25.0))
        delay_ms = max(0.0, _safe_float(cfg.get("delay_ms"), 820.0))
        max_delta_ticks = max(0, _safe_int(cfg.get("max_mvs_sync_delta_ticks"), 2))
        sync_tolerance_ms = max(0.0, _safe_float(cfg.get("sync_tolerance_ms"), 0.0))
        hold_warn_ms = max(sync_tolerance_ms, _safe_float(cfg.get("hold_warn_ms"), 150.0))
        delay_ticks = int(round(delay_ms / tick_ms)) if delay_ms > 0.0 else 0
        snapshots: Dict[str, Dict[str, Any]] = {}
        latest_ids: List[int] = []
        latest_by_cam: Dict[str, int] = {}
        valid_count_by_cam: Dict[str, int] = {}
        per_cam: Dict[str, Dict[str, Any]] = {}
        snapshot_unstable = 0
        snapshot_attempts = max(1, min(8, _safe_int(cfg.get("mvs_snapshot_retries"), 3)))
        for cam in self.cams:
            rd = self._open_mvs_reader(cam)
            if rd is None:
                per_cam[cam] = {
                    "state": "reader_error",
                    "selected": False,
                    "reason": str(self.mvs_reader_errors.get(cam, "reader_error")),
                }
                continue
            snap = None
            attempts_used = 0
            for attempt in range(snapshot_attempts):
                attempts_used = attempt + 1
                snap = rd.read_slots_snapshot()
                if snap is not None:
                    break
                time.sleep(0.0005)
            if snap is None:
                snapshot_unstable += 1
                per_cam[cam] = {
                    "state": "snapshot_unstable",
                    "selected": False,
                    "reason": "read_slots_snapshot_none",
                    "snapshot_attempts": int(attempts_used),
                }
                continue
            snap["snapshot_attempts"] = int(attempts_used)
            snapshots[cam] = snap
            valid = [s for s in snap.get("slots", []) if int(s.get(key, -1)) >= 0 and int(s.get("frame_id", -1)) >= 0]
            valid_count_by_cam[cam] = int(len(valid))
            if valid:
                latest_id = max(int(s.get(key, -1)) for s in valid)
                latest_ids.append(int(latest_id))
                latest_by_cam[cam] = int(latest_id)
        if not latest_ids:
            return -1, 0, {
                "state": "no_latest_ids",
                "target_sync_id": -1,
                "latest_common_sync_id": -1,
                "per_camera": per_cam,
                "valid_slot_count_by_cam": valid_count_by_cam,
            }
        latest_common = int(min(latest_ids))
        target = int(latest_common) - max(0, int(delay_ticks))
        if target < 0:
            target = latest_common
            delay_ticks = 0
        synced = 0
        hold_warn = 0
        missing_history = 0
        metadata_outlier = 0
        selected_ids: Dict[str, int] = {}
        raw_delta_by_cam: Dict[str, int] = {}
        sync_delta_ms_by_cam: Dict[str, float] = {}
        missing_history_cams: List[str] = []
        hold_warn_cams: List[str] = []
        metadata_outlier_cams: List[str] = []
        for cam, snap in snapshots.items():
            valid = [s for s in snap.get("slots", []) if int(s.get(key, -1)) >= 0 and int(s.get("frame_id", -1)) >= 0]
            if not valid:
                missing_history += 1
                per_cam[cam] = {
                    "state": "missing_history",
                    "selected": False,
                    "reason": "no_valid_slots",
                    "valid_slot_count": 0,
                    "slot_count": int(snap.get("slot_count", 0) or 0),
                }
                continue
            ids = [int(s.get(key, -1)) for s in valid]
            min_id = min(ids)
            max_id = max(ids)
            best = min(valid, key=lambda s: abs(int(s.get(key, -1)) - int(target)))
            best_id = int(best.get(key, -1))
            delta = int(best_id) - int(target)
            delta_ms = float(delta) * float(tick_ms)
            outlier_by_ticks = bool(sync_mode == "nearest_trigger" and abs(delta) > max_delta_ticks)
            outlier_by_ms = bool(sync_mode == "nearest_trigger" and sync_tolerance_ms > 0.0 and abs(delta_ms) > sync_tolerance_ms)
            missing = bool(int(target) < min_id or int(target) > max_id)
            if missing:
                state = "missing_history"
                missing_history += 1
                missing_history_cams.append(str(cam))
            elif outlier_by_ticks or outlier_by_ms:
                if abs(delta_ms) <= hold_warn_ms:
                    state = "hold_warn"
                    hold_warn += 1
                    hold_warn_cams.append(str(cam))
                else:
                    state = "metadata_outlier"
                    metadata_outlier += 1
                    metadata_outlier_cams.append(str(cam))
            else:
                state = "synced"
                synced += 1
            selected_ids[cam] = int(best_id)
            raw_delta_by_cam[cam] = int(delta)
            sync_delta_ms_by_cam[cam] = round(float(delta_ms), 3)
            if int(target) < min_id:
                target_position = "before_history"
            elif int(target) > max_id:
                target_position = "after_history"
            else:
                target_position = "inside_history"
            per_cam[cam] = {
                "state": state,
                "selected": bool(state in ("synced", "hold_warn")),
                "sync_valid": bool(state == "synced"),
                "reason": state,
                "latest_id": int(latest_by_cam.get(cam, -1)),
                "selected_id": int(best_id),
                "target_sync_id": int(target),
                "sync_delta": int(delta),
                "sync_delta_ms_est": round(float(delta_ms), 3),
                "target_position": target_position,
                "slot": _safe_int(best.get("slot"), -1),
                "frame_id": _safe_int(best.get("frame_id"), -1),
                "tick_id": _safe_int(best.get("tick_id"), -1),
                "timestamp_us": _safe_int(best.get("timestamp_us"), 0),
                "min_available_id": int(min_id),
                "max_available_id": int(max_id),
                "history_span_ticks": int(max_id - min_id),
                "history_span_ms_est": round(float(max_id - min_id) * float(tick_ms), 3),
                "valid_slot_count": int(len(valid)),
                "slot_count": int(snap.get("slot_count", 0) or 0),
                "header_bytes": int(snap.get("header_bytes", 0) or 0),
                "snapshot_attempts": int(snap.get("snapshot_attempts", 1) or 1),
                "outlier_by_ticks": bool(outlier_by_ticks),
                "outlier_by_ms": bool(outlier_by_ms),
            }
        return int(target), int(delay_ticks), {
            "state": "ok" if synced == len(self.cams) else "degraded",
            "sync_mode": sync_mode,
            "key": key,
            "target_sync_id": int(target),
            "latest_common_sync_id": int(latest_common),
            "delay_ms": float(delay_ms),
            "delay_ticks": int(delay_ticks),
            "tick_ms_est": float(tick_ms),
            "sync_tolerance_ms": float(sync_tolerance_ms),
            "hold_warn_ms": float(hold_warn_ms),
            "max_delta_ticks": int(max_delta_ticks),
            "synced_cam_count": int(synced),
            "hold_warn_cam_count": int(hold_warn),
            "missing_history_cam_count": int(missing_history),
            "metadata_outlier_cam_count": int(metadata_outlier),
            "snapshot_unstable_cam_count": int(snapshot_unstable),
            "selected_id_by_cam": selected_ids,
            "raw_sync_delta_by_cam": raw_delta_by_cam,
            "sync_delta_ms_by_cam": sync_delta_ms_by_cam,
            "mvs_offset_ticks_by_cam": raw_delta_by_cam,
            "mvs_offset_ms_by_cam": sync_delta_ms_by_cam,
            "latest_id_by_cam": latest_by_cam,
            "valid_slot_count_by_cam": valid_count_by_cam,
            "missing_history_cams": missing_history_cams,
            "hold_warn_cams": hold_warn_cams,
            "metadata_outlier_cams": metadata_outlier_cams,
            "per_camera": per_cam,
        }

    def _build_event(self, target_sync: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
        tick_ms = max(0.001, _safe_float(cfg.get("mvs_sync_tick_ms_est"), 25.0))
        max_sync_ms = max(1.0, _safe_float(cfg.get("event_layer_max_sync_delta_ms"), 20.0))
        max_sync_ticks = max(0, _safe_int(cfg.get("event_layer_max_sync_delta_ticks"), 2))
        initial = max(1, _safe_int(cfg.get("candidate_scan_initial"), 4))
        escalated = max(initial, _safe_int(cfg.get("candidate_scan_escalated"), 32))
        esc_enable = bool(cfg.get("candidate_escalate_enable", True))
        esc_abs_ms = max(0.0, _safe_float(cfg.get("candidate_escalate_abs_ms"), 10.0))
        esc_ratio = max(0.05, min(1.0, _safe_float(cfg.get("candidate_escalate_ratio"), 0.60)))
        per_cam: Dict[str, Dict[str, Any]] = {}
        synced = 0
        selected = 0
        no_sparse = 0
        outlier = 0
        event_delta_ms_by_cam: Dict[str, float] = {}
        event_delta_ticks_by_cam: Dict[str, int] = {}
        sync_outlier_cams: List[str] = []
        for cam in self.cams:
            rd = self.sparse_readers.get(cam)
            ring = self.ring_readers.get(cam)
            if rd is None or ring is None or int(target_sync) < 0:
                no_sparse += 1
                per_cam[cam] = {"state": "no_sparse_frame", "selected": False, "reason": "reader_or_target_missing"}
                continue
            try:
                hint_meta = ring.event_ts_for_mvs_sync(target_sync)
                center_hint_us = _safe_int(hint_meta.get("event_ts_us"), 0) if isinstance(hint_meta, dict) else 0
                item = rd.read_for_target(
                    target_sync,
                    lambda meta, tgt, _cam=cam: self._score_event_meta(_cam, meta, tgt, tick_ms),
                    center_hint_us=center_hint_us,
                    candidate_limit=initial,
                    candidate_escalate_enable=esc_enable,
                    candidate_escalate_limit=escalated,
                    candidate_escalate_abs_ms=esc_abs_ms,
                    candidate_escalate_ratio=esc_ratio,
                    max_sync_delta_ms=max_sync_ms,
                    read_points=False,
                )
                if item is None:
                    no_sparse += 1
                    per_cam[cam] = {
                        "state": "no_sparse_frame",
                        "selected": False,
                        "target_sync_id": int(target_sync),
                        "center_hint_us": int(center_hint_us),
                    }
                    continue
                meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
                sync_delta_ms = _safe_float(meta.get("event_to_bev_sync_delta_ms"), 999999.0)
                sync_delta_ticks = _safe_int(meta.get("event_to_bev_sync_delta_ticks"), 999999)
                is_synced = (
                    bool(meta.get("sync_available", False))
                    and abs(float(sync_delta_ms)) <= max_sync_ms
                    and abs(int(sync_delta_ticks)) <= max_sync_ticks
                )
                selected += 1
                event_delta_ms_by_cam[cam] = round(float(sync_delta_ms), 3)
                event_delta_ticks_by_cam[cam] = int(sync_delta_ticks)
                if is_synced:
                    synced += 1
                    state = "synced"
                else:
                    outlier += 1
                    sync_outlier_cams.append(str(cam))
                    state = "sync_delta_too_large"
                sync_meta = meta.get("sync_meta") if isinstance(meta.get("sync_meta"), dict) else {}
                per_cam[cam] = {
                    "state": state,
                    "selected": True,
                    "sync_valid": bool(is_synced),
                    "reason": state,
                    "frame_id": _safe_int(item.get("frame_id"), -1),
                    "target_sync_id": int(target_sync),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "event_window_start_us": _safe_int(meta.get("event_window_start_us"), 0),
                    "event_window_center_us": _safe_int(meta.get("event_window_center_us"), 0),
                    "event_window_end_us": _safe_int(meta.get("event_window_end_us"), 0),
                    "event_to_bev_sync_delta_ms": round(float(sync_delta_ms), 3),
                    "event_to_bev_sync_delta_ticks": int(sync_delta_ticks),
                    "event_mapped_mvs_sync_index": _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1),
                    "event_sync_source": str(sync_meta.get("source", "")),
                    "sparse_point_count": _safe_int(meta.get("sparse_point_count"), -1),
                    "sparse_candidate_scan_count": _safe_int(meta.get("event_layer_sparse_candidate_scan_count"), -1),
                    "sparse_candidate_scan_initial": _safe_int(meta.get("event_layer_sparse_candidate_scan_initial"), -1),
                    "sparse_candidate_scan_escalated": bool(meta.get("event_layer_sparse_candidate_scan_escalated", False)),
                    "sparse_candidate_total_count": _safe_int(meta.get("event_layer_sparse_candidate_total_count"), -1),
                    "center_hint_us": int(center_hint_us),
                }
            except Exception as exc:
                no_sparse += 1
                per_cam[cam] = {
                    "state": "error",
                    "selected": False,
                    "reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "target_sync_id": int(target_sync),
                }
        return {
            "state": "ok" if synced == len(self.cams) else "degraded",
            "target_sync_id": int(target_sync),
            "selected_cam_count": int(selected),
            "synced_cam_count": int(synced),
            "no_sparse_cam_count": int(no_sparse),
            "sync_outlier_cam_count": int(outlier),
            "sync_outlier_cams": sync_outlier_cams,
            "max_sync_delta_ms": float(max_sync_ms),
            "max_sync_delta_ticks": int(max_sync_ticks),
            "event_delta_ms_by_cam": event_delta_ms_by_cam,
            "event_delta_ticks_by_cam": event_delta_ticks_by_cam,
            "per_camera": per_cam,
        }

    def _build_once(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        target_sync, delay_ticks, mvs_obj = self._build_mvs(cfg)
        event_obj = self._build_event(target_sync, cfg)
        self.bundle_seq += 1
        overall = "ok"
        if str(mvs_obj.get("state")) != "ok" or str(event_obj.get("state")) != "ok":
            overall = "degraded"
        if int(target_sync) < 0:
            overall = "no_target"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.elapsed_samples_ms.append(float(elapsed_ms))
        if len(self.elapsed_samples_ms) > 120:
            del self.elapsed_samples_ms[:len(self.elapsed_samples_ms) - 120]
        worker_perf = self._elapsed_stats()
        return {
            "enabled": True,
            "mode": str(cfg.get("mode", "shadow")),
            "state": "running",
            "overall_state": overall,
            "bundle_id": int(self.bundle_seq),
            "target_sync_id": int(target_sync),
            "latest_common_sync_id": _safe_int(mvs_obj.get("latest_common_sync_id"), -1),
            "delay_ms": _safe_float(mvs_obj.get("delay_ms"), _safe_float(cfg.get("delay_ms"), 820.0)),
            "delay_ticks": int(delay_ticks),
            "build_wall_us": _now_us(),
            "updated_wall_us": _now_us(),
            "updated_age_ms": 0.0,
            "interval_ms": round(float(self.interval_s * 1000.0), 3),
            "elapsed_ms": round(float(elapsed_ms), 3),
            "elapsed_p50_ms": worker_perf["p50_ms"],
            "elapsed_p95_ms": worker_perf["p95_ms"],
            "elapsed_max_ms": worker_perf["max_ms"],
            "elapsed_sample_count": worker_perf["n"],
            "worker_elapsed_p95_ms": worker_perf["p95_ms"],
            "cpu_prep_only": True,
            "opengl_thread": "gui",
            "payload_policy": "metadata_only_no_mvs_image_no_sparse_points",
            "mvs": mvs_obj,
            "event_layer": event_obj,
            "mvs_offset_ticks_by_cam": dict(mvs_obj.get("mvs_offset_ticks_by_cam", {})),
            "mvs_offset_ms_by_cam": dict(mvs_obj.get("mvs_offset_ms_by_cam", {})),
            "event_delta_ms_by_cam": dict(event_obj.get("event_delta_ms_by_cam", {})),
            "event_delta_ticks_by_cam": dict(event_obj.get("event_delta_ticks_by_cam", {})),
            "quality": {
                "overall_state": overall,
                "mvs_synced_cam_count": _safe_int(mvs_obj.get("synced_cam_count"), 0),
                "mvs_hold_warn_cam_count": _safe_int(mvs_obj.get("hold_warn_cam_count"), 0),
                "mvs_missing_history_cam_count": _safe_int(mvs_obj.get("missing_history_cam_count"), 0),
                "mvs_metadata_outlier_cam_count": _safe_int(mvs_obj.get("metadata_outlier_cam_count"), 0),
                "event_synced_cam_count": _safe_int(event_obj.get("synced_cam_count"), 0),
                "event_no_sparse_cam_count": _safe_int(event_obj.get("no_sparse_cam_count"), 0),
                "event_sync_outlier_cam_count": _safe_int(event_obj.get("sync_outlier_cam_count"), 0),
            },
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                cfg = dict(self.config)
            try:
                obj = self._build_once(cfg)
            except Exception as exc:
                obj = {
                    "enabled": True,
                    "mode": str(cfg.get("mode", "shadow")),
                    "state": "error",
                    "overall_state": "error",
                    "bundle_id": int(self.bundle_seq),
                    "target_sync_id": -1,
                    "updated_wall_us": _now_us(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "mvs": {"per_camera": {}},
                    "event_layer": {"per_camera": {}},
                }
            with self.lock:
                self.bundle = obj
            self.stop_event.wait(self.interval_s)


def _as_point_array(data: Dict[str, Any], primary_key: str, fallback_key: str = "") -> np.ndarray:
    pts = data.get(primary_key, None)
    if pts is None and fallback_key:
        pts = data.get(fallback_key, None)
    if pts is None:
        pts = []
    return np.asarray(pts, dtype=np.float32)


def _parse_homography(data: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[np.ndarray], str]:
    for key in keys:
        raw = data.get(key, None)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape == (3, 3):
            return arr.astype(np.float32), key
        if arr.size == 9:
            return arr.reshape(3, 3).astype(np.float32), key
    return None, ""


def _field_range(points: np.ndarray) -> Dict[str, Any]:
    if points.ndim != 2 or points.shape[0] <= 0 or points.shape[1] < 2:
        return {}
    xs = points[:, 0].astype(np.float64)
    ys = points[:, 1].astype(np.float64)
    return {
        "x_min": float(np.min(xs)),
        "x_max": float(np.max(xs)),
        "y_min": float(np.min(ys)),
        "y_max": float(np.max(ys)),
        "width": float(np.max(xs) - np.min(xs)),
        "height": float(np.max(ys) - np.min(ys)),
    }


def _load_homography(calib_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = json.loads(calib_path.read_text(encoding="utf-8-sig"))
    img_pts = _as_point_array(data, "image_points")
    field_pts = _as_point_array(data, "field_points", "local_points")
    H_field_to_img, source = _parse_homography(data, ("field_to_image_homography", "homography_local_to_image"))
    if H_field_to_img is None:
        H_img_to_field, source = _parse_homography(data, ("image_to_field_homography", "homography_image_to_local"))
        if H_img_to_field is not None:
            H_field_to_img = np.linalg.inv(np.asarray(H_img_to_field, dtype=np.float64)).astype(np.float32)
            source = f"inverse({source})"
    if H_field_to_img is None:
        if img_pts.shape[0] < 4 or field_pts.shape[0] < 4:
            raise RuntimeError(f"calibration needs field_to_image_homography or >=4 point pairs: {calib_path}")
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"OpenCV required for fallback findHomography: {exc}") from exc
        H_img_to_field, _ = cv2.findHomography(img_pts, field_pts, method=0)
        if H_img_to_field is None:
            raise RuntimeError(f"findHomography failed: {calib_path}")
        H_field_to_img = np.linalg.inv(np.asarray(H_img_to_field, dtype=np.float64)).astype(np.float32)
        source = "fallback_findHomography(image_points,field_points)"
    return H_field_to_img, {
        "path": str(calib_path),
        "point_count": int(min(len(img_pts), len(field_pts))),
        "matrix_source": str(source),
        "matrix_upload_mode": "row_vec3_no_transpose",
        "field_range_m": _field_range(field_pts),
        "source_files": data.get("source_files", {}),
        "ransac": data.get("ransac", {}),
        "error_field_unit_all": data.get("error_field_unit_all", {}),
        "error_field_unit_inliers": data.get("error_field_unit_inliers", {}),
        "error_px_back_project_all": data.get("error_px_back_project_all", {}),
        "error_px_back_project_inliers": data.get("error_px_back_project_inliers", {}),
        "unit": str(data.get("unit", "m")),
        "type": str(data.get("type", "")),
    }


VERT_SRC = """
#version 120
varying vec2 v_uv;
void main() {
    v_uv = gl_MultiTexCoord0.xy;
    gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);
}
"""


FRAG_SRC = """
#version 120
varying vec2 v_uv;
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;
uniform sampler2D u_tex6;
uniform sampler2D u_tex7;
uniform sampler2D u_event_tex0;
uniform sampler2D u_event_tex1;
uniform sampler2D u_event_tex2;
uniform sampler2D u_event_tex3;
uniform sampler2D u_event_tex4;
uniform sampler2D u_event_tex5;
uniform sampler2D u_event_tex6;
uniform sampler2D u_event_tex7;
uniform vec3 u_field_to_img_r0[8];
uniform vec3 u_field_to_img_r1[8];
uniform vec3 u_field_to_img_r2[8];
uniform vec3 u_mvs_to_event_r0[8];
uniform vec3 u_mvs_to_event_r1[8];
uniform vec3 u_mvs_to_event_r2[8];
uniform vec2 u_img_size[8];
uniform int u_valid[8];
uniform vec2 u_event_img_size[8];
uniform int u_event_valid[8];
uniform int u_bayer_code[8];
uniform vec2 u_field_origin_m;
uniform vec2 u_field_size_m;
uniform float u_feather_power;
uniform int u_debug_mode;
uniform int u_texture_y_flip;
uniform float u_gain;
uniform float u_black_level;
uniform float u_gamma;
uniform vec3 u_color_gain;
uniform vec3 u_cam_rgb_gain[8];
uniform vec3 u_turf_target_rgb;
uniform float u_turf_match_strength;
uniform int u_grid_enable;
uniform int u_field_transform_mode;
uniform int u_event_layer_enable;
uniform float u_event_layer_alpha;
uniform float u_event_layer_gain;
uniform float u_event_layer_min_value;
uniform int u_event_layer_color_mode;
uniform float u_event_alpha_scale[8];

vec2 raw_uv(vec2 ip, vec2 img_size) {
    vec2 p = clamp(ip, vec2(0.0), img_size - vec2(1.0));
    vec2 uv = (floor(p) + vec2(0.5)) / img_size;
    if (u_texture_y_flip != 0) {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}

vec2 texture_uv_from_source(vec2 src, vec2 img_size) {
    vec2 p = clamp(src, vec2(0.0), img_size - vec2(1.0));
    vec2 uv = (floor(p) + vec2(0.5)) / img_size;
    if (u_texture_y_flip != 0) {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}

vec2 source_pix_from_image_pix(vec2 pix, vec2 img_size) {
    vec2 p = clamp(pix, vec2(0.0), img_size - vec2(1.0));
    return p;
}

float raw_at_src(sampler2D tex, vec2 src, vec2 img_size) {
    return texture2D(tex, texture_uv_from_source(src, img_size)).r;
}

float raw_mosaic_value(vec2 uv) {
    vec2 grid = vec2(4.0, 2.0);
    vec2 cell = floor(uv * grid);
    vec2 local = fract(uv * grid);
    if (u_texture_y_flip != 0) {
        local.y = 1.0 - local.y;
    }
    float idx = cell.y * 4.0 + cell.x;
    if (idx < 0.5) return (u_valid[0] != 0) ? texture2D(u_tex0, local).r : 0.0;
    if (idx < 1.5) return (u_valid[1] != 0) ? texture2D(u_tex1, local).r : 0.0;
    if (idx < 2.5) return (u_valid[2] != 0) ? texture2D(u_tex2, local).r : 0.0;
    if (idx < 3.5) return (u_valid[3] != 0) ? texture2D(u_tex3, local).r : 0.0;
    if (idx < 4.5) return (u_valid[4] != 0) ? texture2D(u_tex4, local).r : 0.0;
    if (idx < 5.5) return (u_valid[5] != 0) ? texture2D(u_tex5, local).r : 0.0;
    if (idx < 6.5) return (u_valid[6] != 0) ? texture2D(u_tex6, local).r : 0.0;
    return (u_valid[7] != 0) ? texture2D(u_tex7, local).r : 0.0;
}

vec3 tone_rgb(vec3 rgb) {
    rgb = max(vec3(0.0), rgb - vec3(max(0.0, u_black_level)));
    rgb *= max(0.01, u_gain);
    rgb = clamp(rgb, 0.0, 1.0);
    rgb = pow(rgb, vec3(1.0 / max(0.05, u_gamma)));
    return clamp(rgb, 0.0, 1.0);
}

vec3 heat_color(float x) {
    x = clamp(x, 0.0, 1.0);
    return clamp(vec3(1.5 * x, 1.5 * (1.0 - abs(x - 0.5) * 2.0), 1.5 * (1.0 - x)), 0.0, 1.0);
}

float grid_line(vec2 field_xy, float step_m, float width_m) {
    vec2 f = fract(field_xy / step_m);
    vec2 d2 = min(f, vec2(1.0) - f) * step_m;
    float d = min(d2.x, d2.y);
    return 1.0 - smoothstep(0.0, width_m, d);
}

vec3 apply_grid(vec3 rgb, vec2 field_xy) {
    if (u_grid_enable == 0) return rgb;
    float minor = grid_line(field_xy, 1.0, 0.035);
    float major = grid_line(field_xy, 5.0, 0.070);
    rgb = mix(rgb, vec3(0.70, 0.78, 0.82), 0.18 * minor);
    rgb = mix(rgb, vec3(0.10, 0.92, 1.00), 0.38 * major);
    return clamp(rgb, 0.0, 1.0);
}

vec3 apply_turf_color_match(vec3 rgb) {
    if (u_turf_match_strength <= 0.0001) return rgb;
    float green_delta = rgb.g - max(rgb.r, rgb.b);
    float green_key = smoothstep(0.015, 0.110, green_delta) * smoothstep(0.045, 0.180, rgb.g);
    float line_reject = 1.0 - smoothstep(0.58, 0.95, max(max(rgb.r, rgb.g), rgb.b));
    float k = clamp(green_key * line_reject * u_turf_match_strength, 0.0, 1.0);
    float lum = max(0.001, dot(rgb, vec3(0.2126, 0.7152, 0.0722)));
    float tgt_lum = max(0.001, dot(u_turf_target_rgb, vec3(0.2126, 0.7152, 0.0722)));
    vec3 target_hue = clamp(u_turf_target_rgb * (lum / tgt_lum), 0.0, 1.0);
    return clamp(mix(rgb, target_hue, k), 0.0, 1.0);
}

vec2 display_to_sample_field_xy(vec2 uv) {
    vec2 q = vec2(uv.x, 1.0 - uv.y);
    vec2 s = q;
    if (u_field_transform_mode == 1) {
        s = vec2(1.0 - q.x, q.y);
    } else if (u_field_transform_mode == 2) {
        s = vec2(q.x, 1.0 - q.y);
    } else if (u_field_transform_mode == 3) {
        s = vec2(1.0 - q.x, 1.0 - q.y);
    } else if (u_field_transform_mode == 4) {
        s = vec2(q.y, q.x);
    } else if (u_field_transform_mode == 5) {
        s = vec2(1.0 - q.y, q.x);
    } else if (u_field_transform_mode == 6) {
        s = vec2(q.y, 1.0 - q.x);
    } else if (u_field_transform_mode == 7) {
        s = vec2(1.0 - q.y, 1.0 - q.x);
    }
    return u_field_origin_m + clamp(s, vec2(0.0), vec2(1.0)) * u_field_size_m;
}

float bayer_color(vec2 src, int pattern) {
    float py = mod(floor(src.y), 2.0);
    float px = mod(floor(src.x), 2.0);
    if (pattern == 0) {       // RGGB
        if (py < 0.5 && px < 0.5) return 0.0;
        if (py > 0.5 && px > 0.5) return 2.0;
        return 1.0;
    }
    if (pattern == 1) {       // BGGR
        if (py < 0.5 && px < 0.5) return 2.0;
        if (py > 0.5 && px > 0.5) return 0.0;
        return 1.0;
    }
    if (pattern == 2) {       // GRBG
        if (py < 0.5 && px > 0.5) return 0.0;
        if (py > 0.5 && px < 0.5) return 2.0;
        return 1.0;
    }
    // GBRG
    if (py > 0.5 && px < 0.5) return 0.0;
    if (py < 0.5 && px > 0.5) return 2.0;
    return 1.0;
}

vec3 demosaic_rgb(sampler2D tex, int pattern, vec2 pix, vec2 img_size) {
    vec2 src = source_pix_from_image_pix(pix, img_size);
    float c = bayer_color(src, pattern);
    float v = raw_at_src(tex, src, img_size);
    float r;
    float g;
    float b;
    if (c < 0.5) {
        r = v;
        g = 0.25 * (
            raw_at_src(tex, src + vec2(0.0, -1.0), img_size) +
            raw_at_src(tex, src + vec2(0.0, 1.0), img_size) +
            raw_at_src(tex, src + vec2(-1.0, 0.0), img_size) +
            raw_at_src(tex, src + vec2(1.0, 0.0), img_size)
        );
        b = 0.25 * (
            raw_at_src(tex, src + vec2(-1.0, -1.0), img_size) +
            raw_at_src(tex, src + vec2(1.0, -1.0), img_size) +
            raw_at_src(tex, src + vec2(-1.0, 1.0), img_size) +
            raw_at_src(tex, src + vec2(1.0, 1.0), img_size)
        );
    } else if (c > 1.5) {
        b = v;
        g = 0.25 * (
            raw_at_src(tex, src + vec2(0.0, -1.0), img_size) +
            raw_at_src(tex, src + vec2(0.0, 1.0), img_size) +
            raw_at_src(tex, src + vec2(-1.0, 0.0), img_size) +
            raw_at_src(tex, src + vec2(1.0, 0.0), img_size)
        );
        r = 0.25 * (
            raw_at_src(tex, src + vec2(-1.0, -1.0), img_size) +
            raw_at_src(tex, src + vec2(1.0, -1.0), img_size) +
            raw_at_src(tex, src + vec2(-1.0, 1.0), img_size) +
            raw_at_src(tex, src + vec2(1.0, 1.0), img_size)
        );
    } else {
        g = v;
        float lc = bayer_color(src + vec2(-1.0, 0.0), pattern);
        float rc = bayer_color(src + vec2(1.0, 0.0), pattern);
        if (lc < 0.5 || rc < 0.5) {
            r = 0.5 * (
                raw_at_src(tex, src + vec2(-1.0, 0.0), img_size) +
                raw_at_src(tex, src + vec2(1.0, 0.0), img_size)
            );
            b = 0.5 * (
                raw_at_src(tex, src + vec2(0.0, -1.0), img_size) +
                raw_at_src(tex, src + vec2(0.0, 1.0), img_size)
            );
        } else {
            b = 0.5 * (
                raw_at_src(tex, src + vec2(-1.0, 0.0), img_size) +
                raw_at_src(tex, src + vec2(1.0, 0.0), img_size)
            );
            r = 0.5 * (
                raw_at_src(tex, src + vec2(0.0, -1.0), img_size) +
                raw_at_src(tex, src + vec2(0.0, 1.0), img_size)
            );
        }
    }
    return clamp(vec3(r, g, b) * u_color_gain, 0.0, 1.0);
}

vec4 sample_cam(sampler2D tex, vec3 h0, vec3 h1, vec3 h2, vec2 img_size, int valid, int bayer_code, vec2 field_xy, vec3 cam_color, vec3 cam_gain) {
    if (valid == 0) return vec4(0.0);
    vec3 f = vec3(field_xy, 1.0);
    vec3 p = vec3(dot(h0, f), dot(h1, f), dot(h2, f));
    if (abs(p.z) < 1e-6) return vec4(0.0);
    vec2 pix = p.xy / p.z;
    vec2 uv = pix / max(img_size, vec2(1.0));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * 2.0;
    float w = pow(smoothstep(0.0, 1.0, clamp(edge, 0.0, 1.0)), max(0.1, u_feather_power));
    vec2 src_pix = source_pix_from_image_pix(pix, img_size);
    vec3 rgb = demosaic_rgb(tex, bayer_code, pix, img_size);
    if (u_debug_mode == 1) {
        rgb = vec3(raw_at_src(tex, src_pix, img_size));
    } else if (u_debug_mode == 2 || u_debug_mode == 5) {
        rgb = cam_color;
    } else if (u_debug_mode == 6) {
        rgb = vec3(uv.x, uv.y, 1.0 - uv.x);
    }
    if (u_debug_mode == 0) {
        rgb = apply_turf_color_match(clamp(rgb * cam_gain, 0.0, 1.0));
        rgb = tone_rgb(rgb);
    } else if (u_debug_mode == 1) {
        rgb = tone_rgb(rgb);
    }
    return vec4(rgb * w, w);
}

void accum(inout vec3 sum_rgb, inout float sum_w, vec4 c) {
    sum_rgb += c.rgb;
    sum_w += c.a;
}

vec4 sample_event_overlay(
    sampler2D tex,
    vec3 h0, vec3 h1, vec3 h2,
    vec3 eh0, vec3 eh1, vec3 eh2,
    vec2 mvs_img_size,
    vec2 event_img_size,
    int valid,
    vec2 field_xy,
    vec3 cam_color,
    float alpha_scale
) {
    if (u_event_layer_enable == 0 || valid == 0) return vec4(0.0);
    vec3 f = vec3(field_xy, 1.0);
    vec3 p = vec3(dot(h0, f), dot(h1, f), dot(h2, f));
    if (abs(p.z) < 1e-6) return vec4(0.0);
    vec2 mvs_pix = p.xy / p.z;
    vec2 mvs_uv = mvs_pix / max(mvs_img_size, vec2(1.0));
    if (mvs_uv.x < 0.0 || mvs_uv.x > 1.0 || mvs_uv.y < 0.0 || mvs_uv.y > 1.0) return vec4(0.0);
    vec3 mp = vec3(mvs_pix, 1.0);
    vec3 ep = vec3(dot(eh0, mp), dot(eh1, mp), dot(eh2, mp));
    if (abs(ep.z) < 1e-6) return vec4(0.0);
    vec2 event_pix = ep.xy / ep.z;
    vec2 uv = event_pix / max(event_img_size, vec2(1.0));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec4(0.0);
    vec4 raw = texture2D(tex, texture_uv_from_source(event_pix, event_img_size));
    float lum = max(max(raw.r, raw.g), raw.b);
    lum = clamp((lum - u_event_layer_min_value) / max(0.001, 1.0 - u_event_layer_min_value), 0.0, 1.0);
    lum = clamp(lum * max(0.01, u_event_layer_gain), 0.0, 1.0);
    if (lum <= 0.001) return vec4(0.0);
    vec3 layer_color = cam_color;
    if (u_event_layer_color_mode == 1) {
        layer_color = vec3(1.0, 0.05, 0.02);
    } else if (u_event_layer_color_mode == 2) {
        layer_color = vec3(1.0, 0.85, 0.05);
    }
    vec3 rgb = (u_event_layer_color_mode == 0) ? max(raw.rgb, layer_color * lum) : (layer_color * lum);
    return vec4(clamp(rgb, 0.0, 1.0), lum * clamp(alpha_scale, 0.0, 1.0));
}

void accum_event(inout vec3 sum_rgb, inout float sum_a, vec4 c) {
    sum_rgb += c.rgb * c.a;
    sum_a += c.a;
}

vec3 apply_event_layer(vec3 base_rgb, vec2 field_xy) {
    if (u_event_layer_enable == 0 || u_event_layer_alpha <= 0.0001) return base_rgb;
    vec3 ev_rgb = vec3(0.0);
    float ev_a = 0.0;
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex0, u_field_to_img_r0[0], u_field_to_img_r1[0], u_field_to_img_r2[0], u_mvs_to_event_r0[0], u_mvs_to_event_r1[0], u_mvs_to_event_r2[0], u_img_size[0], u_event_img_size[0], u_event_valid[0], field_xy, vec3(1.0, 0.15, 0.10), u_event_alpha_scale[0]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex1, u_field_to_img_r0[1], u_field_to_img_r1[1], u_field_to_img_r2[1], u_mvs_to_event_r0[1], u_mvs_to_event_r1[1], u_mvs_to_event_r2[1], u_img_size[1], u_event_img_size[1], u_event_valid[1], field_xy, vec3(1.0, 0.65, 0.10), u_event_alpha_scale[1]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex2, u_field_to_img_r0[2], u_field_to_img_r1[2], u_field_to_img_r2[2], u_mvs_to_event_r0[2], u_mvs_to_event_r1[2], u_mvs_to_event_r2[2], u_img_size[2], u_event_img_size[2], u_event_valid[2], field_xy, vec3(0.25, 0.95, 0.25), u_event_alpha_scale[2]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex3, u_field_to_img_r0[3], u_field_to_img_r1[3], u_field_to_img_r2[3], u_mvs_to_event_r0[3], u_mvs_to_event_r1[3], u_mvs_to_event_r2[3], u_img_size[3], u_event_img_size[3], u_event_valid[3], field_xy, vec3(0.10, 0.80, 1.0), u_event_alpha_scale[3]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex4, u_field_to_img_r0[4], u_field_to_img_r1[4], u_field_to_img_r2[4], u_mvs_to_event_r0[4], u_mvs_to_event_r1[4], u_mvs_to_event_r2[4], u_img_size[4], u_event_img_size[4], u_event_valid[4], field_xy, vec3(0.30, 0.35, 1.0), u_event_alpha_scale[4]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex5, u_field_to_img_r0[5], u_field_to_img_r1[5], u_field_to_img_r2[5], u_mvs_to_event_r0[5], u_mvs_to_event_r1[5], u_mvs_to_event_r2[5], u_img_size[5], u_event_img_size[5], u_event_valid[5], field_xy, vec3(0.85, 0.30, 1.0), u_event_alpha_scale[5]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex6, u_field_to_img_r0[6], u_field_to_img_r1[6], u_field_to_img_r2[6], u_mvs_to_event_r0[6], u_mvs_to_event_r1[6], u_mvs_to_event_r2[6], u_img_size[6], u_event_img_size[6], u_event_valid[6], field_xy, vec3(1.0, 0.25, 0.65), u_event_alpha_scale[6]));
    accum_event(ev_rgb, ev_a, sample_event_overlay(u_event_tex7, u_field_to_img_r0[7], u_field_to_img_r1[7], u_field_to_img_r2[7], u_mvs_to_event_r0[7], u_mvs_to_event_r1[7], u_mvs_to_event_r2[7], u_img_size[7], u_event_img_size[7], u_event_valid[7], field_xy, vec3(0.80, 0.80, 0.80), u_event_alpha_scale[7]));
    if (ev_a <= 0.0001) return base_rgb;
    vec3 layer_rgb = ev_rgb / max(0.001, ev_a);
    float a = clamp(ev_a * u_event_layer_alpha, 0.0, clamp(u_event_layer_alpha, 0.0, 1.0));
    return clamp(mix(base_rgb, layer_rgb, a), 0.0, 1.0);
}

void main() {
    if (u_debug_mode == 3) {
        float g = raw_mosaic_value(v_uv);
        gl_FragColor = vec4(g, g, g, 1.0);
        return;
    }
    vec2 field_xy = display_to_sample_field_xy(v_uv);
    vec4 c0 = sample_cam(u_tex0, u_field_to_img_r0[0], u_field_to_img_r1[0], u_field_to_img_r2[0], u_img_size[0], u_valid[0], u_bayer_code[0], field_xy, vec3(1.0, 0.15, 0.10), u_cam_rgb_gain[0]);
    vec4 c1 = sample_cam(u_tex1, u_field_to_img_r0[1], u_field_to_img_r1[1], u_field_to_img_r2[1], u_img_size[1], u_valid[1], u_bayer_code[1], field_xy, vec3(1.0, 0.65, 0.10), u_cam_rgb_gain[1]);
    vec4 c2 = sample_cam(u_tex2, u_field_to_img_r0[2], u_field_to_img_r1[2], u_field_to_img_r2[2], u_img_size[2], u_valid[2], u_bayer_code[2], field_xy, vec3(0.25, 0.95, 0.25), u_cam_rgb_gain[2]);
    vec4 c3 = sample_cam(u_tex3, u_field_to_img_r0[3], u_field_to_img_r1[3], u_field_to_img_r2[3], u_img_size[3], u_valid[3], u_bayer_code[3], field_xy, vec3(0.10, 0.80, 1.0), u_cam_rgb_gain[3]);
    vec4 c4 = sample_cam(u_tex4, u_field_to_img_r0[4], u_field_to_img_r1[4], u_field_to_img_r2[4], u_img_size[4], u_valid[4], u_bayer_code[4], field_xy, vec3(0.30, 0.35, 1.0), u_cam_rgb_gain[4]);
    vec4 c5 = sample_cam(u_tex5, u_field_to_img_r0[5], u_field_to_img_r1[5], u_field_to_img_r2[5], u_img_size[5], u_valid[5], u_bayer_code[5], field_xy, vec3(0.85, 0.30, 1.0), u_cam_rgb_gain[5]);
    vec4 c6 = sample_cam(u_tex6, u_field_to_img_r0[6], u_field_to_img_r1[6], u_field_to_img_r2[6], u_img_size[6], u_valid[6], u_bayer_code[6], field_xy, vec3(1.0, 0.25, 0.65), u_cam_rgb_gain[6]);
    vec4 c7 = sample_cam(u_tex7, u_field_to_img_r0[7], u_field_to_img_r1[7], u_field_to_img_r2[7], u_img_size[7], u_valid[7], u_bayer_code[7], field_xy, vec3(0.80, 0.80, 0.80), u_cam_rgb_gain[7]);
    if (u_debug_mode == 5) {
        vec4 best = c0;
        if (c1.a > best.a) best = c1;
        if (c2.a > best.a) best = c2;
        if (c3.a > best.a) best = c3;
        if (c4.a > best.a) best = c4;
        if (c5.a > best.a) best = c5;
        if (c6.a > best.a) best = c6;
        if (c7.a > best.a) best = c7;
        if (best.a <= 0.0001) {
            gl_FragColor = vec4(apply_event_layer(vec3(0.03, 0.035, 0.04), field_xy), 1.0);
        } else {
            gl_FragColor = vec4(apply_event_layer(apply_grid(best.rgb / best.a, field_xy), field_xy), 1.0);
        }
        return;
    }
    vec3 sum_rgb = vec3(0.0);
    float sum_w = 0.0;
    accum(sum_rgb, sum_w, c0);
    accum(sum_rgb, sum_w, c1);
    accum(sum_rgb, sum_w, c2);
    accum(sum_rgb, sum_w, c3);
    accum(sum_rgb, sum_w, c4);
    accum(sum_rgb, sum_w, c5);
    accum(sum_rgb, sum_w, c6);
    accum(sum_rgb, sum_w, c7);
    if (u_debug_mode == 4) {
        gl_FragColor = vec4(apply_event_layer(apply_grid(heat_color(sum_w / 2.0), field_xy), field_xy), 1.0);
        return;
    }
    if (sum_w <= 0.0001) {
        gl_FragColor = vec4(apply_event_layer(vec3(0.03, 0.035, 0.04), field_xy), 1.0);
    } else {
        gl_FragColor = vec4(apply_event_layer(apply_grid(sum_rgb / sum_w, field_xy), field_xy), 1.0);
    }
}
"""


DISPLAY_FRAG_SRC = """
#version 120
varying vec2 v_uv;
uniform sampler2D u_tex;
void main() {
    gl_FragColor = texture2D(u_tex, v_uv);
}
"""


class GlobalBevWidget(QOpenGLWidget):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.cams = _parse_cams(args.cams)[:8]
        self.out_w = max(1, int(round(float(args.field_width_m) / max(1e-6, float(args.meters_per_pixel)))))
        self.out_h = max(1, int(round(float(args.field_height_m) / max(1e-6, float(args.meters_per_pixel)))))
        self.resize(int(args.window_width), int(args.window_height))
        self.setWindowTitle("Global BEV Fusion")
        self.setFocusPolicy(Qt.StrongFocus)
        self._set_transform_mode(_normalize_transform_mode(getattr(args, "field_transform_mode", "normal"), bool(args.field_flip_x), bool(args.field_flip_y)))
        self.readers: Dict[str, RawPreviewFrameReader] = {}
        self.reader_errors: Dict[str, str] = {}
        self.calib: Dict[str, Dict[str, Any]] = {}
        self.H: Dict[str, np.ndarray] = {}
        self.coverage_diag: Dict[str, Any] = {}
        self.textures: Dict[str, int] = {}
        self.tex_shape: Dict[str, Tuple[int, int]] = {}
        self.texture_diag: Dict[str, Dict[str, Any]] = {}
        self.event_layer_readers: Dict[str, EventOverlayLayerReader] = {}
        self.event_layer_sparse_readers: Dict[str, SparseEventOverlayReader] = {}
        self.event_layer_textures: Dict[str, int] = {}
        self.event_layer_shape: Dict[str, Tuple[int, int, int]] = {}
        self.event_layer_last_uploaded: Dict[str, int] = {}
        self.event_layer_last_upload_key: Dict[str, str] = {}
        self.event_layer_items: Dict[str, Dict[str, Any]] = {}
        self.event_layer_valid: Dict[str, bool] = {}
        self.event_layer_alpha_scale: Dict[str, float] = {}
        self.event_layer_mvs_to_event: Dict[str, np.ndarray] = {}
        self.event_layer_event_to_mvs: Dict[str, np.ndarray] = {}
        self.event_layer_sparse_vertices: Dict[str, np.ndarray] = {}
        self.event_layer_sparse_point_alpha: Dict[str, np.ndarray] = {}
        self.event_layer_sparse_vertex_mono: Dict[str, float] = {}
        self.fbo_stats: Dict[str, Any] = {}
        self.display_diag: Dict[str, Any] = {}
        self.stability_diag: Dict[str, Any] = {}
        self.turf_diag: Dict[str, Any] = {}
        self.last_uploaded: Dict[str, Tuple[int, int, int, int, int, int]] = {}
        self.render_items: Dict[str, Dict[str, Any]] = {}
        self.render_item_mono: Dict[str, float] = {}
        self.mvs_upload_diag: Dict[str, Any] = {
            "budget_enable": bool(getattr(args, "mvs_upload_budget_enable", True)),
            "budget_ms": float(getattr(args, "mvs_upload_budget_ms", 10.0) or 0.0),
            "elapsed_ms": 0.0,
            "uploaded_cams": [],
            "duplicate_skip_cams": [],
            "held_by_budget_cams": [],
            "forced_upload_cams": [],
            "unstable_skip_cams": [],
            "unsupported_pixel_format_cams": [],
        }
        self._mvs_upload_budget_cursor = 0
        self._event_layer_budget_cursor = 0
        self._budget_diag_state: Dict[str, Dict[str, Any]] = {}
        self.selected: Dict[str, Dict[str, Any]] = {}
        self.turf_rgb_stats: Dict[str, np.ndarray] = {}
        self.turf_rgb_counts: Dict[str, int] = {}
        self.cam_rgb_gain: Dict[str, np.ndarray] = {}
        self.turf_target_rgb = np.asarray(
            _parse_rgb_triplet(
                getattr(args, "turf_target_rgb", ""),
                (float(args.turf_target_r), float(args.turf_target_g), float(args.turf_target_b)),
            ),
            dtype=np.float32,
        )
        self.frame_id = 0
        self.render_fps = 0.0
        self.output_fps = 0.0
        self._render_count = 0
        self._output_count = 0
        self._fps_t0 = time.time()
        self._last_output_t = 0.0
        self._last_status_t = 0.0
        self._last_status_debug_t = 0.0
        self._last_coverage_t = 0.0
        self._last_fbo_stats_t = 0.0
        self._last_overlay_text_cache_t = 0.0
        self._last_error = ""
        self.fbo = 0
        self.fbo_tex = 0
        self.pbos: List[int] = []
        self.pbo_index = 0
        self.pbo_ready = False
        self.pbo_ready_count = 0
        self.program = 0
        self.display_program = 0
        self.writer: Optional[BevSharedFrameWriter] = None
        self.perf_timings: Dict[str, List[float]] = {}
        self.perf_diag: Dict[str, Any] = {}
        self.readback_diag: Dict[str, Any] = {}
        self._turf_last_sample_mono: Dict[str, float] = {}
        self._person_overlay_cached_tracks: List[Dict[str, Any]] = []
        self._person_overlay_cache_t = 0.0
        self._overlay_text_pixmap: Optional[QPixmap] = None
        self._overlay_text_pixmap_size: Tuple[int, int] = (0, 0)
        self.sidecar_path = Path(args.sidecar_json)
        self.status_path = Path(args.status_json)
        self.status_debug_path = self.status_path.with_name(f"{self.status_path.stem}_debug{self.status_path.suffix}")
        self.control_path = Path(args.control_json)
        self.runtime_state_path = Path(args.runtime_state_json)
        self.status_writer: Optional[AsyncStatusWriter] = None
        if bool(getattr(args, "async_status_writer_enable", True)):
            self.status_writer = AsyncStatusWriter(self.status_path, self.status_debug_path)
            self.status_writer.start()
        self.global_person_path = Path(args.global_person_latest_json)
        self.global_person_diag: Dict[str, Any] = {
            "path": str(self.global_person_path),
            "enabled": bool(args.global_person_overlay_enable),
            "presentation_mode": "latest_fallback",
            "presentation_note": "BEV background/event layer may be delayed; global person overlay is latest until a person-history source is added.",
        }
        self.global_person_tracks: List[Dict[str, Any]] = []
        self.global_person_meta: Dict[str, Any] = {}
        self._global_person_mtime_ns = 0
        self._person_overlay_slots: Dict[str, Dict[str, Any]] = {}
        self._person_overlay_next_slot_id = 1
        self.manual_hit_command_path = Path(getattr(args, "manual_hit_command_json", "sync_ipc/manual_hit_commands.jsonl"))
        self.manual_hit_click_enable = bool(getattr(args, "manual_hit_click_enable", False))
        self.manual_hit_click_radius_px = max(4.0, float(getattr(args, "manual_hit_click_radius_px", 42.0) or 42.0))
        self.manual_hit_targets: List[Dict[str, Any]] = []
        self.manual_hit_target_cache_us = 0
        self.manual_hit_command_seq = 0
        self.manual_hit_last_result: Dict[str, Any] = {
            "state": "init",
            "click_enable": bool(self.manual_hit_click_enable),
            "command_jsonl": str(self.manual_hit_command_path),
        }
        self.event_bullet_path = Path(args.event_bullet_jsonl)
        if not self.event_bullet_path.is_absolute():
            self.event_bullet_path = Path(args.project_root) / self.event_bullet_path
        self.event_bullet_tailer = JsonlTailer(self.event_bullet_path)
        self.event_bullet_records: List[Dict[str, Any]] = []
        self.event_bullet_diag: Dict[str, Any] = {
            "enabled": bool(args.event_bullet_overlay_enable),
            "path": str(self.event_bullet_path),
            "coord_source": "mvs_pixel_to_field",
            "render_target": "qt_overlay",
            "presentation_mode": "latest_fallback",
            "presentation_note": "BEV background/event layer may be delayed; event bullet QT overlay is latest/life-window filtered.",
            "transform_mode": str(getattr(args, "event_bullet_overlay_transform_mode", "normal") or "normal"),
            "skip_reason": "init",
            "records_buffered": 0,
            "recent_records": 0,
            "draw_points": 0,
        }
        self._event_bullet_last_poll_ms = 0.0
        self._event_bullet_img_to_field_cache: Dict[str, np.ndarray] = {}
        self._event_bullet_projection_cache_hits = 0
        self._event_bullet_projection_cache_misses = 0
        self.max_texture_units = 0
        self.event_layer_diag: Dict[str, Any] = {
            "enabled": bool(getattr(args, "event_layer_enable", False)),
            "path_template": str(getattr(args, "event_layer_template", "event_overlay_latest_{camera}") or "event_overlay_latest_{camera}"),
            "sidecar_template": str(getattr(args, "event_layer_sidecar_template", "event_overlay_{camera}_latest.json") or "event_overlay_{camera}_latest.json"),
            "backend": str(getattr(args, "event_layer_backend", "texture") or "texture"),
            "sparse_sidecar_template": str(getattr(args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
            "homography_config": str(getattr(args, "event_layer_homography_config", "") or ""),
            "coord_source": "field_xy -> MVS pixel -> inverse(event_to_mvs_homography) -> event pixel",
            "render_target": "bev_fbo",
            "transparent_background": True,
            "alpha": float(getattr(args, "event_layer_alpha", 0.45) or 0.0),
            "gain": float(getattr(args, "event_layer_gain", 2.4) or 2.4),
            "min_value": float(getattr(args, "event_layer_min_value", 0.03) or 0.03),
            "max_age_ms": float(getattr(args, "event_layer_max_age_ms", 150.0) or 150.0),
            "sync_mode": _event_layer_sync_mode(getattr(args, "event_layer_sync_mode", "latest_guarded")),
            "available_sync_modes": list(EVENT_LAYER_SYNC_MODE_ORDER),
            "max_sync_delta_ms": float(getattr(args, "event_layer_max_sync_delta_ms", 120.0) or 120.0),
            "max_sync_delta_ticks": int(getattr(args, "event_layer_max_sync_delta_ticks", 2) or 2),
            "sync_normal_ms": float(getattr(args, "event_layer_sync_normal_ms", 50.0) or 50.0),
            "sync_warn_ms": float(getattr(args, "event_layer_sync_warn_ms", 80.0) or 80.0),
            "sync_hard_ms": float(getattr(args, "event_layer_sync_hard_ms", 80.0) or 80.0),
            "hold_last_good_on_hard_outlier": bool(getattr(args, "event_layer_hold_last_good_on_hard_outlier", True)),
            "hold_last_good_ms": float(getattr(args, "event_layer_hold_last_good_ms", 250.0) or 0.0),
            "sync_display_state_by_cam": {},
            "sync_display_level_by_cam": {},
            "sync_warn_cams": [],
            "sync_hard_outlier_cams": [],
            "sync_hold_cams": [],
            "sync_hidden_cams": [],
            "history_enable": bool(getattr(args, "event_layer_history_enable", False)),
            "history_sidecar_template": str(getattr(args, "event_layer_history_sidecar_template", "") or ""),
            "sync_map_ring_template": str(getattr(args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
            "sync_truth_mode": "history_ring",
            "sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "exact_sync_cam_count": 0,
            "interpolated_sync_cam_count": 0,
            "estimated_sync_cam_count": 0,
            "missing_sync_cam_count": 0,
            "accumulate_enable": bool(getattr(args, "event_layer_accumulate_enable", True)),
            "accumulate_window_ms": float(getattr(args, "event_layer_accumulate_window_ms", 40.0) or 40.0),
            "accumulate_max_frames": int(getattr(args, "event_layer_accumulate_max_frames", 4) or 4),
            "texture_scale": float(getattr(args, "event_layer_texture_scale", 1.0) or 1.0),
            "history_poll_interval_ms": float(getattr(args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
            "display_only": True,
            "judgement_sync_threshold_ms": 100.0,
            "bev_visual_sync_target_ms": float(getattr(args, "event_layer_max_sync_delta_ms", 120.0) or 120.0),
            "event_frame_dropped_count": 0,
            "fade_alpha_scale": float(getattr(args, "event_layer_fade_alpha_scale", 0.25) or 0.25),
            "color_mode": _event_layer_color_mode(getattr(args, "event_layer_color_mode", "camera")),
            "available_color_modes": list(EVENT_LAYER_COLOR_MODE_ORDER),
            "display_budget_enable": bool(getattr(args, "event_layer_display_budget_enable", True)),
            "display_budget_ms": float(getattr(args, "event_layer_display_budget_ms", 10.0) or 0.0),
            "display_budget_elapsed_ms": 0.0,
            "display_budget_held_cams": [],
            "display_budget_held_cam_count": 0,
            "display_budget_order": [],
            "display_budget_overrun": False,
            "display_budget_overrun_rate": 0.0,
            "display_budget_held_cam_count_p95": 0.0,
            "sparse_read_elapsed_ms": 0.0,
            "sparse_sync_gate_elapsed_ms": 0.0,
            "sparse_project_elapsed_ms": 0.0,
            "display_hold_ms": float(getattr(args, "event_layer_display_hold_ms", getattr(args, "frame_hold_ms", 280.0)) or 0.0),
            "dilate_px": float(getattr(args, "event_layer_dilate_px", 16.0) or 0.0),
            "requested_dilate_px": float(getattr(args, "event_layer_dilate_px", 16.0) or 0.0),
            "effective_dilate_px": float(getattr(args, "event_layer_dilate_px", 16.0) or 0.0),
            "dilate_backend": "cv2_dilate" if _cv2 is not None else "cpu_sparse_upload",
            "dilate_max_pixels": int(getattr(args, "event_layer_dilate_max_pixels", 2500) or 0),
            "dilate_authority": "launch_profile_or_control",
            "runtime_state_event_layer_display_restore_enable": bool(getattr(args, "runtime_state_event_layer_display_restore_enable", False)),
            "runtime_state_loaded_path": str(getattr(args, "runtime_state_loaded_path", "")),
            "runtime_state_skipped_event_layer_display_keys": list(getattr(args, "runtime_state_skipped_event_layer_display_keys", []) or []),
            "shadow": {
                "enabled": bool(getattr(args, "event_layer_sparse_shadow_enable", False)),
                "mode": "shadow",
                "state": "init",
                "target_sync_id": -1,
                "matching_target": False,
                "mismatch_cam_count": 0,
                "per_camera": {},
            },
            "presentation_worker": {
                "enabled": bool(getattr(args, "event_layer_presentation_worker_enable", False)),
                "mode": str(getattr(args, "event_layer_presentation_worker_mode", "off") or "off"),
                "state": "disabled",
                "cpu_prep_only": True,
                "opengl_thread": "gui",
                "fallback_to_gui_direct": True,
                "active_supported": True,
                "target_sync_id": -1,
                "bundle_ring_size": int(getattr(args, "event_layer_presentation_bundle_ring_size", 8) or 8),
                "bundle_ring_count": 0,
                "bundle_hit_count": 0,
                "bundle_miss_count": 0,
                "bundle_hit_ratio": 0.0,
                "consume_exact_count": 0,
                "consume_near_count": 0,
                "prebuild_lookback_ticks": int(getattr(args, "event_layer_presentation_prebuild_lookback_ticks", 2) or 2),
                "prebuild_lookahead_ticks": int(getattr(args, "event_layer_presentation_prebuild_lookahead_ticks", 2) or 2),
                "prebuild_max_per_cycle": int(getattr(args, "event_layer_presentation_prebuild_max_per_cycle", 1) or 1),
                "per_camera": {},
            },
            "performance_guard": {
                "auto_degrade_enable": bool(getattr(args, "event_layer_auto_degrade_enable", True)),
                "fps_threshold": float(getattr(args, "event_layer_auto_degrade_fps_threshold", 28.0) or 28.0),
                "upload_p95_threshold_ms": float(getattr(args, "event_layer_auto_degrade_upload_p95_ms", 8.0) or 8.0),
                "min_dilate_px": float(getattr(args, "event_layer_auto_degrade_min_dilate_px", 4.0) or 4.0),
                "state": "normal",
            },
            "skip_reason": "init",
            "draw_cams": [],
            "per_camera": {},
        }
        self.mvs_sync_diag: Dict[str, Any] = {
            "sync_mode": str(getattr(args, "sync_mode", "nearest_trigger")),
            "target_sync_id": -1,
            "max_delta_ticks": int(getattr(args, "max_mvs_sync_delta_ticks", 2) or 2),
            "outlier_policy": str(getattr(args, "mvs_sync_outlier_policy", "hide") or "hide"),
            "tick_ms_est": float(getattr(args, "mvs_sync_tick_ms_est", 25.0) or 25.0),
            "mixed_frame_detected": False,
            "hidden_by_mvs_sync_delta_cams": [],
            "per_camera": {},
        }
        self.event_sync_health_cache: Dict[str, Dict[str, Any]] = {}
        self.event_sync_health_mtime_ns: Dict[str, int] = {}
        self.event_sync_ring_readers: Dict[str, EventSyncMapRingReader] = {}
        self.event_layer_sparse_shadow: Optional[SparseEventLayerShadowWorker] = None
        self.event_layer_presentation_worker: Optional[EventLayerPresentationWorker] = None
        self.alignment_worker: Optional[DelayedAlignmentWorker] = None
        self.event_layer_active_history: Dict[int, Dict[str, Any]] = {}
        self.event_layer_active_history_order: List[int] = []
        if bool(getattr(args, "event_layer_sparse_shadow_enable", False)):
            try:
                self.event_layer_sparse_shadow = SparseEventLayerShadowWorker(
                    self.cams,
                    str(getattr(args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
                    str(getattr(args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
                    float(getattr(args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
                    float(getattr(args, "event_layer_sparse_shadow_interval_ms", 250.0) or 250.0),
                )
                self.event_layer_sparse_shadow.start()
                self.event_layer_diag["shadow"]["state"] = "started"
            except Exception as exc:
                self.event_layer_sparse_shadow = None
                self.event_layer_diag["shadow"] = {
                    "enabled": False,
                    "mode": "shadow",
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        self.alignment_diag: Dict[str, Any] = {
            "enabled": _alignment_mode(getattr(args, "alignment_mode", "current")) in ("shadow", "active"),
            "mode": _alignment_mode(getattr(args, "alignment_mode", "current")),
            "available_modes": list(ALIGNMENT_MODE_ORDER),
            "state": "disabled",
            "cpu_prep_only": True,
            "opengl_thread": "gui",
            "delay_ms": float(getattr(args, "alignment_delay_ms", getattr(args, "presentation_delay_ms", 820.0)) or 0.0),
            "shadow_interval_ms": float(getattr(args, "alignment_shadow_interval_ms", 250.0) or 250.0),
            "hold_warn_ms": float(getattr(args, "alignment_hold_warn_ms", 150.0) or 150.0),
            "active_supported": False,
            "active_note": "shadow-only until remote validation passes; OpenGL upload/draw remains in GUI thread",
            "mvs": {"per_camera": {}},
            "event_layer": {"per_camera": {}},
        }
        if _alignment_mode(getattr(args, "alignment_mode", "current")) in ("shadow", "active"):
            try:
                self.alignment_worker = DelayedAlignmentWorker(
                    self.cams,
                    str(getattr(args, "raw_template", "mvs_raw_latest_{camera}") or "mvs_raw_latest_{camera}"),
                    str(getattr(args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
                    str(getattr(args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
                    self._alignment_worker_config(),
                )
                self.alignment_worker.start()
                self.alignment_diag["state"] = "started"
            except Exception as exc:
                self.alignment_worker = None
                self.alignment_diag.update({
                    "enabled": False,
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        self._event_layer_effective_dilate_px = float(getattr(args, "event_layer_dilate_px", 16.0) or 0.0)
        self._event_layer_dilate_last_change_us = 0
        self._event_layer_dilate_state = "normal"
        self.presentation_diag: Dict[str, Any] = {
            "enabled": bool(getattr(args, "presentation_delay_enable", False)),
            "delay_ms": float(getattr(args, "presentation_delay_ms", 0.0) or 0.0),
            "sync_tolerance_ms": float(getattr(args, "sync_tolerance_ms", 0.0) or 0.0),
            "target_sync_id": -1,
            "latest_common_sync_id": -1,
            "delay_ticks": 0,
            "scope": "global_bev_only",
        }
        self.current_target_sync_id = -1
        self._load_event_layer_homographies()
        self._ensure_event_layer_presentation_worker()
        self.control_diag: Dict[str, Any] = {
            "path": str(self.control_path),
            "runtime_state_json": str(self.runtime_state_path),
            "available_transform_modes": list(TRANSFORM_MODE_ORDER),
            "available_debug_modes": list(DEBUG_MODE_ORDER),
            "available_person_overlay_coord_modes": list(PERSON_OVERLAY_COORD_MODE_ORDER),
            "available_event_bullet_overlay_transform_modes": list(EVENT_BULLET_OVERLAY_TRANSFORM_MODE_ORDER),
            "hotkeys": "M/N transform, Q person_xy, D debug, T texture_y_flip, G grid, R reset_transform",
        }
        self._control_mtime_ns = 0
        self._control_seq = 0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update)
        self.timer.start(max(1, int(round(1000.0 / max(1.0, float(args.display_fps))))))

    def _perf_start(self) -> float:
        return time.perf_counter()

    def _perf_end(self, name: str, t0: float) -> None:
        try:
            ms = (time.perf_counter() - float(t0)) * 1000.0
            samples = self.perf_timings.setdefault(str(name), [])
            samples.append(float(ms))
            max_samples = max(30, int(getattr(self.args, "perf_timing_window", 180) or 180))
            if len(samples) > max_samples:
                del samples[:len(samples) - max_samples]
        except Exception:
            pass

    def _perf_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, samples in sorted(self.perf_timings.items()):
            if not samples:
                continue
            vals = sorted(float(x) for x in samples)
            n = len(vals)
            def _pct(p: float) -> float:
                if n <= 1:
                    return vals[0]
                idx = int(round((n - 1) * float(p)))
                idx = max(0, min(n - 1, idx))
                return vals[idx]
            out[name] = {
                "last_ms": round(float(samples[-1]), 3),
                "p50_ms": round(float(_pct(0.50)), 3),
                "p95_ms": round(float(_pct(0.95)), 3),
                "max_ms": round(float(vals[-1]), 3),
                "n": int(n),
            }
        target_ms = 1000.0 / max(1.0, float(getattr(self.args, "display_fps", 30.0) or 30.0))
        return {
            "target_display_fps": float(getattr(self.args, "display_fps", 30.0)),
            "target_output_fps": float(getattr(self.args, "output_fps", 20.0)),
            "target_frame_ms": round(float(target_ms), 3),
            "window": max(30, int(getattr(self.args, "perf_timing_window", 180) or 180)),
            "stages": out,
        }

    def _budget_order(self, keys: Iterable[str], cursor_attr: str) -> List[str]:
        ordered = [str(k) for k in keys if str(k)]
        if not ordered:
            return []
        try:
            start = int(getattr(self, cursor_attr, 0) or 0) % len(ordered)
        except Exception:
            start = 0
        setattr(self, cursor_attr, (start + 1) % len(ordered))
        return ordered[start:] + ordered[:start]

    def _update_budget_diag(self, name: str, elapsed_ms: float, budget_ms: float, held_count: int) -> Dict[str, Any]:
        state = self._budget_diag_state.setdefault(
            str(name),
            {
                "sample_count": 0,
                "overrun_count": 0,
                "elapsed_samples": [],
                "held_samples": [],
            },
        )
        samples = state.setdefault("elapsed_samples", [])
        held_samples = state.setdefault("held_samples", [])
        samples.append(float(elapsed_ms))
        held_samples.append(int(held_count))
        max_samples = max(30, int(getattr(self.args, "perf_timing_window", 180) or 180))
        if len(samples) > max_samples:
            del samples[:len(samples) - max_samples]
        if len(held_samples) > max_samples:
            del held_samples[:len(held_samples) - max_samples]
        state["sample_count"] = int(state.get("sample_count", 0) or 0) + 1
        overrun = bool(float(budget_ms) > 0.0 and float(elapsed_ms) > float(budget_ms))
        if overrun:
            state["overrun_count"] = int(state.get("overrun_count", 0) or 0) + 1

        def _p95(vals: List[float]) -> float:
            if not vals:
                return 0.0
            ordered_vals = sorted(float(v) for v in vals)
            idx = int(round((len(ordered_vals) - 1) * 0.95))
            idx = max(0, min(len(ordered_vals) - 1, idx))
            return float(ordered_vals[idx])

        sample_count = max(1, int(state.get("sample_count", 0) or 0))
        return {
            "sample_count": int(sample_count),
            "overrun": bool(overrun),
            "overrun_count": int(state.get("overrun_count", 0) or 0),
            "overrun_rate": round(float(state.get("overrun_count", 0) or 0) / float(sample_count), 4),
            "elapsed_p95_ms": round(float(_p95([float(v) for v in samples])), 3),
            "held_cam_count_p95": round(float(_p95([float(v) for v in held_samples])), 3),
        }

    def _set_transform_mode(self, mode: Any) -> str:
        raw = str(mode or "").strip()
        if raw in TRANSFORM_MODE_CODES:
            m = raw
        else:
            m = _normalize_transform_mode(raw, bool(getattr(self.args, "field_flip_x", False)), bool(getattr(self.args, "field_flip_y", False)))
        self.args.field_transform_mode = m
        fx, fy = _simple_flip_from_mode(m)
        self.args.field_flip_x = fx
        self.args.field_flip_y = fy
        return m

    def _set_debug_mode(self, mode: Any) -> str:
        m = str(mode or "").strip()
        if m not in DEBUG_MODE_CODES:
            m = str(getattr(self.args, "debug_mode", "warp_color"))
        if m not in DEBUG_MODE_CODES:
            m = "warp_color"
        self.args.debug_mode = m
        self.args.debug_mode_code = int(DEBUG_MODE_CODES[m])
        return m

    def _cycle_transform_mode(self, delta: int) -> str:
        cur = _normalize_transform_mode(getattr(self.args, "field_transform_mode", "normal"))
        idx = TRANSFORM_MODE_ORDER.index(cur) if cur in TRANSFORM_MODE_ORDER else 0
        return self._set_transform_mode(TRANSFORM_MODE_ORDER[(idx + int(delta)) % len(TRANSFORM_MODE_ORDER)])

    def _cycle_debug_mode(self, delta: int) -> str:
        cur = str(getattr(self.args, "debug_mode", "warp_color"))
        idx = DEBUG_MODE_ORDER.index(cur) if cur in DEBUG_MODE_ORDER else 0
        return self._set_debug_mode(DEBUG_MODE_ORDER[(idx + int(delta)) % len(DEBUG_MODE_ORDER)])

    def _cycle_person_overlay_coord_mode(self, delta: int) -> str:
        cur = str(getattr(self.args, "global_person_overlay_coord_mode", "field_xy") or "field_xy").strip().lower().replace("-", "_")
        idx = PERSON_OVERLAY_COORD_MODE_ORDER.index(cur) if cur in PERSON_OVERLAY_COORD_MODE_ORDER else 0
        mode = PERSON_OVERLAY_COORD_MODE_ORDER[(idx + int(delta)) % len(PERSON_OVERLAY_COORD_MODE_ORDER)]
        self.args.global_person_overlay_coord_mode = mode
        return mode

    def _field_xy_to_widget_xy(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        fw = max(1e-6, float(self.args.field_width_m))
        fh = max(1e-6, float(self.args.field_height_m))
        sx = (float(x) - float(self.args.field_origin_x_m)) / fw
        sy = (float(y) - float(self.args.field_origin_y_m)) / fh
        if sx < -0.05 or sx > 1.05 or sy < -0.05 or sy > 1.05:
            return None
        m = _normalize_transform_mode(getattr(self.args, "field_transform_mode", "normal"))
        # Inverse of shader display_to_sample normalized transform.
        qx, qy = sx, sy
        if m == "flip_x":
            qx, qy = 1.0 - sx, sy
        elif m == "flip_y":
            qx, qy = sx, 1.0 - sy
        elif m == "flip_xy":
            qx, qy = 1.0 - sx, 1.0 - sy
        elif m == "swap_xy":
            qx, qy = sy, sx
        elif m == "swap_xy_flip_x":
            qx, qy = sy, 1.0 - sx
        elif m == "swap_xy_flip_y":
            qx, qy = 1.0 - sy, sx
        elif m == "swap_xy_flip_xy":
            qx, qy = 1.0 - sy, 1.0 - sx
        uvx = qx
        uvy = 1.0 - qy
        return uvx * float(self.width()), uvy * float(self.height())

    def _transform_person_xy_for_overlay(self, x: float, y: float) -> Tuple[float, float]:
        mode = str(getattr(self.args, "global_person_overlay_coord_mode", "field_xy") or "field_xy").strip().lower().replace("-", "_")
        ox = float(self.args.field_origin_x_m)
        oy = float(self.args.field_origin_y_m)
        fw = float(self.args.field_width_m)
        fh = float(self.args.field_height_m)
        if mode in {"field_xy", "raw", "none", "normal"}:
            return float(x), float(y)
        if mode == "neg_y":
            return float(x), float(-y)
        if mode in {"field_height_plus_y", "height_plus_y"}:
            return float(x), float(oy + fh + y)
        if mode in {"invert_y", "flip_y"}:
            return float(x), float(oy + fh - (y - oy))
        if mode == "flip_x":
            return float(ox + fw - (x - ox)), float(y)
        if mode == "flip_xy":
            return float(ox + fw - (x - ox)), float(oy + fh - (y - oy))
        if mode == "swap_xy":
            return float(y), float(x)
        if mode == "swap_xy_neg_y":
            return float(-y), float(x)
        return float(x), float(y)

    def _transform_event_bullet_xy_for_overlay(self, x: float, y: float) -> Tuple[float, float]:
        mode = str(getattr(self.args, "event_bullet_overlay_transform_mode", "normal") or "normal").strip().lower().replace("-", "_")
        ox = float(self.args.field_origin_x_m)
        oy = float(self.args.field_origin_y_m)
        fw = float(self.args.field_width_m)
        fh = float(self.args.field_height_m)
        if mode in {"field_xy", "raw", "none", "normal"}:
            return float(x), float(y)
        if mode == "neg_y":
            return float(x), float(-y)
        if mode in {"field_height_plus_y", "height_plus_y"}:
            return float(x), float(oy + fh + y)
        if mode in {"invert_y", "flip_y"}:
            return float(x), float(oy + fh - (y - oy))
        if mode == "flip_x":
            return float(ox + fw - (x - ox)), float(y)
        if mode == "flip_xy":
            return float(ox + fw - (x - ox)), float(oy + fh - (y - oy))
        if mode == "swap_xy":
            return float(y), float(x)
        if mode == "swap_xy_neg_y":
            return float(-y), float(x)
        return float(x), float(y)

    def _alignment_worker_config(self) -> Dict[str, Any]:
        return {
            "mode": _alignment_mode(getattr(self.args, "alignment_mode", "current")),
            "delay_ms": float(getattr(self.args, "alignment_delay_ms", getattr(self.args, "presentation_delay_ms", 820.0)) or 0.0),
            "interval_ms": float(getattr(self.args, "alignment_shadow_interval_ms", 250.0) or 250.0),
            "hold_warn_ms": float(getattr(self.args, "alignment_hold_warn_ms", 150.0) or 150.0),
            "sync_mode": str(getattr(self.args, "sync_mode", "nearest_trigger") or "nearest_trigger"),
            "max_mvs_sync_delta_ticks": int(getattr(self.args, "max_mvs_sync_delta_ticks", 2) or 2),
            "sync_tolerance_ms": float(getattr(self.args, "sync_tolerance_ms", 0.0) or 0.0),
            "mvs_sync_tick_ms_est": float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0) or 25.0),
            "mvs_snapshot_retries": int(getattr(self.args, "unstable_copy_retries", 3) or 3),
            "event_layer_max_sync_delta_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 20.0) or 20.0),
            "event_layer_max_sync_delta_ticks": int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2) or 2),
            "event_layer_sync_normal_ms": float(getattr(self.args, "event_layer_sync_normal_ms", 50.0) or 50.0),
            "event_layer_sync_warn_ms": float(getattr(self.args, "event_layer_sync_warn_ms", 80.0) or 80.0),
            "event_layer_sync_hard_ms": float(getattr(self.args, "event_layer_sync_hard_ms", 80.0) or 80.0),
            "event_layer_hold_last_good_on_hard_outlier": bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True)),
            "event_layer_hold_last_good_ms": float(getattr(self.args, "event_layer_hold_last_good_ms", 250.0) or 0.0),
            "event_layer_history_poll_interval_ms": float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
            "candidate_scan_initial": int(getattr(self.args, "event_layer_sparse_candidate_scan_initial", 4) or 4),
            "candidate_scan_escalated": int(getattr(self.args, "event_layer_sparse_candidate_scan_escalated", 32) or 32),
            "candidate_escalate_enable": bool(getattr(self.args, "event_layer_sparse_candidate_escalate_enable", True)),
            "candidate_escalate_abs_ms": float(getattr(self.args, "event_layer_sparse_candidate_escalate_abs_ms", 10.0) or 10.0),
            "candidate_escalate_ratio": float(getattr(self.args, "event_layer_sparse_candidate_escalate_ratio", 0.60) or 0.60),
        }

    def _ensure_alignment_worker_mode(self) -> None:
        mode = _alignment_mode(getattr(self.args, "alignment_mode", "current"))
        self.alignment_diag["mode"] = mode
        self.alignment_diag["enabled"] = bool(mode in ("shadow", "active"))
        if mode == "current":
            if self.alignment_worker is not None:
                try:
                    self.alignment_worker.stop(timeout_s=1.0)
                except Exception:
                    pass
                self.alignment_worker = None
            self.alignment_diag.update({"state": "disabled", "mvs": {"per_camera": {}}, "event_layer": {"per_camera": {}}})
            return
        if self.alignment_worker is None:
            try:
                self.alignment_worker = DelayedAlignmentWorker(
                    self.cams,
                    str(getattr(self.args, "raw_template", "mvs_raw_latest_{camera}") or "mvs_raw_latest_{camera}"),
                    str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(self.args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
                    str(getattr(self.args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
                    self._alignment_worker_config(),
                )
                self.alignment_worker.start()
                self.alignment_diag["state"] = "started"
            except Exception as exc:
                self.alignment_diag.update({"state": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _refresh_alignment_diag(self, active_target_sync: int, selected: Dict[str, Dict[str, Any]]) -> None:
        self._ensure_alignment_worker_mode()
        mode = _alignment_mode(getattr(self.args, "alignment_mode", "current"))
        if self.alignment_worker is not None:
            self.alignment_worker.update_config(self._alignment_worker_config())
            snap = self.alignment_worker.snapshot()
        else:
            snap = {
                "enabled": False,
                "mode": mode,
                "state": "disabled",
                "target_sync_id": -1,
                "mvs": {"per_camera": {}},
                "event_layer": {"per_camera": {}},
            }
        active_key = "tick_id" if self.args.sync_mode == "nearest_trigger" else "frame_id"
        active_selected = {
            cam: _safe_int(item.get(active_key), -1)
            for cam, item in selected.items()
            if isinstance(item, dict)
        }
        mvs_per = ((snap.get("mvs") or {}).get("per_camera") or {}) if isinstance(snap.get("mvs"), dict) else {}
        mismatch: List[str] = []
        shadow_selected: Dict[str, int] = {}
        if isinstance(mvs_per, dict):
            for cam, rec in mvs_per.items():
                if not isinstance(rec, dict):
                    continue
                sid = _safe_int(rec.get("selected_id"), -1)
                shadow_selected[str(cam)] = int(sid)
                if str(cam) in active_selected and sid >= 0 and int(active_selected[str(cam)]) != int(sid):
                    mismatch.append(str(cam))
        snap["compare_to_active"] = {
            "active_target_sync_id": int(active_target_sync),
            "shadow_target_sync_id": _safe_int(snap.get("target_sync_id"), -1),
            "target_match": bool(int(active_target_sync) == _safe_int(snap.get("target_sync_id"), -1)),
            "active_selected_id_by_cam": active_selected,
            "shadow_selected_id_by_cam": shadow_selected,
            "mismatch_cams": list(mismatch),
            "mismatch_cam_count": int(len(mismatch)),
        }
        snap["worker_interval_ms"] = _safe_float(snap.get("interval_ms"), float(getattr(self.args, "alignment_shadow_interval_ms", 250.0) or 250.0))
        snap["worker_elapsed_p95_ms"] = _safe_float(snap.get("worker_elapsed_p95_ms", snap.get("elapsed_p95_ms")), 0.0)
        snap["gui_refresh_p95_ms"] = round(float(self._perf_stage_p95_ms("refresh_alignment")), 3)
        snap["gui_upload_selected_p95_ms"] = round(float(self._perf_stage_p95_ms("upload_selected")), 3)
        snap["gui_upload_event_layer_p95_ms"] = round(float(self._perf_stage_p95_ms("upload_event_layer_textures")), 3)
        snap["gui_event_sparse_read_p95_ms"] = round(float(self._perf_stage_p95_ms("event_layer_sparse_read")), 3)
        snap["gui_event_sparse_sync_gate_p95_ms"] = round(float(self._perf_stage_p95_ms("event_layer_sparse_sync_gate")), 3)
        snap["gui_event_sparse_project_p95_ms"] = round(float(self._perf_stage_p95_ms("event_layer_sparse_project")), 3)
        snap["gui_render_fbo_p95_ms"] = round(float(self._perf_stage_p95_ms("render_fbo")), 3)
        snap["gui_paint_total_p95_ms"] = round(float(self._perf_stage_p95_ms("paint_total")), 3)
        snap["mvs_upload_budget_ms"] = _safe_float(self.mvs_upload_diag.get("budget_ms"), 0.0)
        snap["mvs_upload_elapsed_ms"] = _safe_float(self.mvs_upload_diag.get("elapsed_ms"), 0.0)
        snap["mvs_upload_held_by_budget_cam_count"] = _safe_int(self.mvs_upload_diag.get("held_by_budget_cam_count"), 0)
        snap["mvs_upload_budget_overrun_rate"] = _safe_float(self.mvs_upload_diag.get("budget_overrun_rate"), 0.0)
        snap["mvs_upload_held_cam_count_p95"] = _safe_float(self.mvs_upload_diag.get("held_cam_count_p95"), 0.0)
        snap["event_layer_display_budget_ms"] = _safe_float(self.event_layer_diag.get("display_budget_ms"), 0.0)
        snap["event_layer_display_budget_elapsed_ms"] = _safe_float(self.event_layer_diag.get("display_budget_elapsed_ms"), 0.0)
        snap["event_layer_display_budget_held_cam_count"] = _safe_int(self.event_layer_diag.get("display_budget_held_cam_count"), 0)
        snap["event_layer_display_budget_overrun_rate"] = _safe_float(self.event_layer_diag.get("display_budget_overrun_rate"), 0.0)
        snap["event_layer_display_budget_held_cam_count_p95"] = _safe_float(self.event_layer_diag.get("display_budget_held_cam_count_p95"), 0.0)
        snap["event_layer_sparse_read_elapsed_ms"] = _safe_float(self.event_layer_diag.get("sparse_read_elapsed_ms"), 0.0)
        snap["event_layer_sparse_sync_gate_elapsed_ms"] = _safe_float(self.event_layer_diag.get("sparse_sync_gate_elapsed_ms"), 0.0)
        snap["event_layer_sparse_project_elapsed_ms"] = _safe_float(self.event_layer_diag.get("sparse_project_elapsed_ms"), 0.0)
        snap["performance"] = {
            "worker_interval_ms": snap["worker_interval_ms"],
            "worker_elapsed_p95_ms": snap["worker_elapsed_p95_ms"],
            "gui_refresh_p95_ms": snap["gui_refresh_p95_ms"],
            "gui_upload_selected_p95_ms": snap["gui_upload_selected_p95_ms"],
            "gui_upload_event_layer_p95_ms": snap["gui_upload_event_layer_p95_ms"],
            "gui_event_sparse_read_p95_ms": snap["gui_event_sparse_read_p95_ms"],
            "gui_event_sparse_sync_gate_p95_ms": snap["gui_event_sparse_sync_gate_p95_ms"],
            "gui_event_sparse_project_p95_ms": snap["gui_event_sparse_project_p95_ms"],
            "gui_render_fbo_p95_ms": snap["gui_render_fbo_p95_ms"],
            "gui_paint_total_p95_ms": snap["gui_paint_total_p95_ms"],
            "mvs_upload_budget_ms": snap["mvs_upload_budget_ms"],
            "mvs_upload_elapsed_ms": snap["mvs_upload_elapsed_ms"],
            "mvs_upload_held_by_budget_cam_count": snap["mvs_upload_held_by_budget_cam_count"],
            "mvs_upload_budget_overrun_rate": snap["mvs_upload_budget_overrun_rate"],
            "mvs_upload_held_cam_count_p95": snap["mvs_upload_held_cam_count_p95"],
            "event_layer_display_budget_ms": snap["event_layer_display_budget_ms"],
            "event_layer_display_budget_elapsed_ms": snap["event_layer_display_budget_elapsed_ms"],
            "event_layer_display_budget_held_cam_count": snap["event_layer_display_budget_held_cam_count"],
            "event_layer_display_budget_overrun_rate": snap["event_layer_display_budget_overrun_rate"],
            "event_layer_display_budget_held_cam_count_p95": snap["event_layer_display_budget_held_cam_count_p95"],
            "event_layer_sparse_read_elapsed_ms": snap["event_layer_sparse_read_elapsed_ms"],
            "event_layer_sparse_sync_gate_elapsed_ms": snap["event_layer_sparse_sync_gate_elapsed_ms"],
            "event_layer_sparse_project_elapsed_ms": snap["event_layer_sparse_project_elapsed_ms"],
        }
        snap["active_supported"] = False
        snap["active_note"] = "OpenGL upload/draw remains in GUI; active bundle switch is gated behind shadow remote validation."
        self.alignment_diag = snap

    def _apply_control_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        if "field_transform_mode" in obj:
            applied["field_transform_mode"] = self._set_transform_mode(obj.get("field_transform_mode"))
        elif "transform_mode" in obj:
            applied["field_transform_mode"] = self._set_transform_mode(obj.get("transform_mode"))
        elif "field_flip_x" in obj or "field_flip_y" in obj:
            applied["field_transform_mode"] = self._set_transform_mode(
                _normalize_transform_mode(
                    "",
                    _safe_bool(obj.get("field_flip_x"), bool(getattr(self.args, "field_flip_x", False))),
                    _safe_bool(obj.get("field_flip_y"), bool(getattr(self.args, "field_flip_y", False))),
                )
            )
        if "debug_mode" in obj:
            applied["debug_mode"] = self._set_debug_mode(obj.get("debug_mode"))
        if "texture_y_flip" in obj:
            self.args.texture_y_flip = _safe_bool(obj.get("texture_y_flip"), bool(self.args.texture_y_flip))
            applied["texture_y_flip"] = bool(self.args.texture_y_flip)
        if "grid_enable" in obj:
            self.args.grid_enable = _safe_bool(obj.get("grid_enable"), bool(self.args.grid_enable))
            applied["grid_enable"] = bool(self.args.grid_enable)
        for key in ("gain", "black_level", "gamma", "feather_power", "turf_match_strength", "turf_gain_alpha", "frame_hold_ms"):
            if key in obj:
                setattr(self.args, key, _safe_float(obj.get(key), float(getattr(self.args, key))))
                applied[key] = float(getattr(self.args, key))
        if "alignment_mode" in obj or "global_bev_alignment_mode" in obj:
            raw = obj.get("alignment_mode", obj.get("global_bev_alignment_mode"))
            mode = _alignment_mode(raw)
            if mode == "active":
                mode = "shadow"
                self.alignment_diag["active_request_downgraded"] = True
                self.alignment_diag["active_request_downgrade_reason"] = "active bundle switch is gated until shadow remote validation passes"
            self.args.alignment_mode = mode
            applied["alignment_mode"] = mode
        for key in ("alignment_delay_ms", "alignment_shadow_interval_ms", "alignment_hold_warn_ms"):
            if key in obj or f"global_bev_{key}" in obj:
                raw = obj.get(key, obj.get(f"global_bev_{key}"))
                val = max(0.0, _safe_float(raw, float(getattr(self.args, key))))
                setattr(self.args, key, float(val))
                applied[key] = float(val)
        if "unstable_copy_retries" in obj:
            self.args.unstable_copy_retries = max(1, int(_safe_float(obj.get("unstable_copy_retries"), float(self.args.unstable_copy_retries))))
            applied["unstable_copy_retries"] = int(self.args.unstable_copy_retries)
        for key in ("turf_match_enable", "turf_auto_match"):
            if key in obj:
                setattr(self.args, key, _safe_bool(obj.get(key), bool(getattr(self.args, key))))
                applied[key] = bool(getattr(self.args, key))
        if "global_person_overlay_coord_mode" in obj:
            mode = str(obj.get("global_person_overlay_coord_mode") or "field_xy").strip().lower().replace("-", "_")
            self.args.global_person_overlay_coord_mode = mode
            applied["global_person_overlay_coord_mode"] = mode
        if "event_bullet_overlay_transform_mode" in obj:
            mode = str(obj.get("event_bullet_overlay_transform_mode") or "normal").strip().lower().replace("-", "_")
            if mode not in set(EVENT_BULLET_OVERLAY_TRANSFORM_MODE_ORDER):
                mode = "normal"
            self.args.event_bullet_overlay_transform_mode = mode
            applied["event_bullet_overlay_transform_mode"] = mode
        if "event_bullet_overlay_enable" in obj:
            enabled = _safe_bool(obj.get("event_bullet_overlay_enable"), bool(getattr(self.args, "event_bullet_overlay_enable", False)))
            self.args.event_bullet_overlay_enable = bool(enabled)
            self.event_bullet_diag["enabled"] = bool(enabled)
            if not enabled:
                self.event_bullet_records = []
                self.event_bullet_diag.update({
                    "records_buffered": 0,
                    "recent_records": 0,
                    "draw_points": 0,
                    "draw_segments": 0,
                    "skip_reason": "disabled_by_control",
                })
            applied["event_bullet_overlay_enable"] = bool(enabled)
        if "event_layer_enable" in obj or "event_overlay_layer_enable" in obj or "show_event_layer" in obj:
            raw = obj.get("event_layer_enable", obj.get("event_overlay_layer_enable", obj.get("show_event_layer")))
            enabled = _safe_bool(raw, bool(getattr(self.args, "event_layer_enable", False)))
            self.args.event_layer_enable = bool(enabled)
            self.event_layer_diag["enabled"] = bool(enabled)
            if not enabled:
                self.event_layer_valid = {cam: False for cam in self.cams}
                self.event_layer_diag.update({
                    "draw_cams": [],
                    "draw_cam_count": 0,
                    "skip_reason": "disabled_by_control",
                })
            applied["event_layer_enable"] = bool(enabled)
        for key in ("event_layer_alpha", "event_layer_gain", "event_layer_min_value", "event_layer_max_age_ms"):
            if key in obj:
                cur = float(getattr(self.args, key))
                val = _safe_float(obj.get(key), cur)
                if key == "event_layer_alpha":
                    val = _clip01(val)
                elif key in ("event_layer_gain", "event_layer_max_age_ms"):
                    val = max(0.0, val)
                else:
                    val = max(0.0, min(0.95, val))
                setattr(self.args, key, float(val))
                diag_key = {
                    "event_layer_alpha": "alpha",
                    "event_layer_gain": "gain",
                    "event_layer_min_value": "min_value",
                    "event_layer_max_age_ms": "max_age_ms",
                }.get(key, key)
                self.event_layer_diag[diag_key] = float(getattr(self.args, key))
                applied[key] = float(getattr(self.args, key))
        if "event_layer_sync_mode" in obj or "event_overlay_layer_sync_mode" in obj:
            raw = obj.get("event_layer_sync_mode", obj.get("event_overlay_layer_sync_mode"))
            mode = _event_layer_sync_mode(raw)
            self.args.event_layer_sync_mode = mode
            self.event_layer_diag["sync_mode"] = mode
            applied["event_layer_sync_mode"] = mode
        for key in (
            "event_layer_max_sync_delta_ms", "event_layer_max_sync_delta_ticks", "event_layer_fade_alpha_scale",
            "event_layer_sync_normal_ms", "event_layer_sync_warn_ms", "event_layer_sync_hard_ms",
            "event_layer_hold_last_good_ms",
            "max_mvs_sync_delta_ticks", "mvs_sync_tick_ms_est", "sync_tolerance_ms",
            "presentation_delay_ms",
        ):
            if key in obj:
                cur = float(getattr(self.args, key))
                val = _safe_float(obj.get(key), cur)
                if key == "event_layer_fade_alpha_scale":
                    val = _clip01(val)
                elif key in ("event_layer_max_sync_delta_ticks", "max_mvs_sync_delta_ticks"):
                    val = max(0.0, val)
                else:
                    val = max(1.0, val)
                if key in ("event_layer_max_sync_delta_ticks", "max_mvs_sync_delta_ticks"):
                    setattr(self.args, key, int(val))
                else:
                    setattr(self.args, key, float(val))
                self.event_layer_diag[{
                    "event_layer_max_sync_delta_ms": "max_sync_delta_ms",
                    "event_layer_max_sync_delta_ticks": "max_sync_delta_ticks",
                    "event_layer_fade_alpha_scale": "fade_alpha_scale",
                    "event_layer_sync_normal_ms": "sync_normal_ms",
                    "event_layer_sync_warn_ms": "sync_warn_ms",
                    "event_layer_sync_hard_ms": "sync_hard_ms",
                    "event_layer_hold_last_good_ms": "hold_last_good_ms",
                }.get(key, key)] = float(getattr(self.args, key))
                applied[key] = float(getattr(self.args, key))
                if key in ("presentation_delay_ms", "sync_tolerance_ms"):
                    self.presentation_diag[key.replace("presentation_", "") if key.startswith("presentation_") else key] = float(getattr(self.args, key))
        for alias, key in (
            ("global_bev_event_layer_sync_normal_ms", "event_layer_sync_normal_ms"),
            ("global_bev_event_layer_sync_warn_ms", "event_layer_sync_warn_ms"),
            ("global_bev_event_layer_sync_hard_ms", "event_layer_sync_hard_ms"),
            ("global_bev_event_layer_hold_last_good_ms", "event_layer_hold_last_good_ms"),
        ):
            if alias in obj and key not in obj:
                cur = float(getattr(self.args, key))
                val = max(0.0, _safe_float(obj.get(alias), cur))
                if key != "event_layer_hold_last_good_ms":
                    val = max(1.0, val)
                setattr(self.args, key, float(val))
                self.event_layer_diag[{
                    "event_layer_sync_normal_ms": "sync_normal_ms",
                    "event_layer_sync_warn_ms": "sync_warn_ms",
                    "event_layer_sync_hard_ms": "sync_hard_ms",
                    "event_layer_hold_last_good_ms": "hold_last_good_ms",
                }[key]] = float(getattr(self.args, key))
                applied[alias] = float(getattr(self.args, key))
        if "event_layer_hold_last_good_on_hard_outlier" in obj or "global_bev_event_layer_hold_last_good_on_hard_outlier" in obj:
            raw = obj.get(
                "event_layer_hold_last_good_on_hard_outlier",
                obj.get("global_bev_event_layer_hold_last_good_on_hard_outlier"),
            )
            self.args.event_layer_hold_last_good_on_hard_outlier = _safe_bool(
                raw,
                bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True)),
            )
            self.event_layer_diag["hold_last_good_on_hard_outlier"] = bool(self.args.event_layer_hold_last_good_on_hard_outlier)
            applied["event_layer_hold_last_good_on_hard_outlier"] = bool(self.args.event_layer_hold_last_good_on_hard_outlier)
        if any(
            key in obj
            for key in (
                "event_layer_sync_normal_ms",
                "event_layer_sync_warn_ms",
                "event_layer_sync_hard_ms",
                "global_bev_event_layer_sync_normal_ms",
                "global_bev_event_layer_sync_warn_ms",
                "global_bev_event_layer_sync_hard_ms",
            )
        ):
            normal_ms, warn_ms, hard_ms = self._event_layer_sync_display_thresholds()
            self.args.event_layer_sync_normal_ms = float(normal_ms)
            self.args.event_layer_sync_warn_ms = float(warn_ms)
            self.args.event_layer_sync_hard_ms = float(hard_ms)
            self.event_layer_diag["sync_normal_ms"] = float(normal_ms)
            self.event_layer_diag["sync_warn_ms"] = float(warn_ms)
            self.event_layer_diag["sync_hard_ms"] = float(hard_ms)
            applied["event_layer_sync_normal_ms"] = float(normal_ms)
            applied["event_layer_sync_warn_ms"] = float(warn_ms)
            applied["event_layer_sync_hard_ms"] = float(hard_ms)
        if "presentation_delay_enable" in obj or "global_bev_presentation_delay_enable" in obj:
            raw = obj.get("presentation_delay_enable", obj.get("global_bev_presentation_delay_enable"))
            self.args.presentation_delay_enable = _safe_bool(raw, bool(getattr(self.args, "presentation_delay_enable", False)))
            self.presentation_diag["enabled"] = bool(self.args.presentation_delay_enable)
            applied["presentation_delay_enable"] = bool(self.args.presentation_delay_enable)
        if "event_layer_history_enable" in obj or "global_bev_event_layer_history_enable" in obj:
            raw = obj.get("event_layer_history_enable", obj.get("global_bev_event_layer_history_enable"))
            self.args.event_layer_history_enable = _safe_bool(raw, bool(getattr(self.args, "event_layer_history_enable", False)))
            self.event_layer_diag["history_enable"] = bool(self.args.event_layer_history_enable)
            applied["event_layer_history_enable"] = bool(self.args.event_layer_history_enable)
        if "event_layer_backend" in obj or "global_bev_event_layer_backend" in obj:
            raw = obj.get("event_layer_backend", obj.get("global_bev_event_layer_backend"))
            backend = str(raw or "texture").strip().lower().replace("-", "_")
            if backend in {"sparse", "sparse_points", "opengl_sparse", "opengl_sparse_points"}:
                backend = "sparse_gl"
            if backend in {"texture", "sparse_gl"}:
                self.args.event_layer_backend = backend
                self.event_layer_diag["backend"] = backend
                applied["event_layer_backend"] = backend
        if "mvs_sync_outlier_policy" in obj:
            policy = str(obj.get("mvs_sync_outlier_policy") or "hide").strip().lower()
            if policy not in ("hide", "warn"):
                policy = "hide"
            self.args.mvs_sync_outlier_policy = policy
            self.mvs_sync_diag["outlier_policy"] = policy
            applied["mvs_sync_outlier_policy"] = policy
        if "event_layer_color_mode" in obj or "event_overlay_layer_color_mode" in obj:
            raw = obj.get("event_layer_color_mode", obj.get("event_overlay_layer_color_mode"))
            mode = _event_layer_color_mode(raw)
            self.args.event_layer_color_mode = mode
            self.event_layer_diag["color_mode"] = mode
            applied["event_layer_color_mode"] = mode
        if "event_layer_dilate_px" in obj or "event_overlay_layer_dilate_px" in obj:
            raw = obj.get("event_layer_dilate_px", obj.get("event_overlay_layer_dilate_px"))
            val = max(0.0, min(64.0, _safe_float(raw, float(getattr(self.args, "event_layer_dilate_px", 16.0)))))
            self.args.event_layer_dilate_px = float(val)
            self.event_layer_diag["dilate_px"] = float(val)
            applied["event_layer_dilate_px"] = float(val)
        if "event_layer_dilate_max_pixels" in obj:
            val_i = max(0, int(_safe_float(obj.get("event_layer_dilate_max_pixels"), float(getattr(self.args, "event_layer_dilate_max_pixels", 2500)))))
            self.args.event_layer_dilate_max_pixels = int(val_i)
            self.event_layer_diag["dilate_max_pixels"] = int(val_i)
            applied["event_layer_dilate_max_pixels"] = int(val_i)
        if "event_layer_sparse_max_points_draw" in obj:
            val_i = max(1, min(200000, _safe_int(obj.get("event_layer_sparse_max_points_draw"), int(getattr(self.args, "event_layer_sparse_max_points_draw", 20000)))))
            self.args.event_layer_sparse_max_points_draw = int(val_i)
            self.event_layer_diag["sparse_max_points_draw"] = int(val_i)
            applied["event_layer_sparse_max_points_draw"] = int(val_i)
        if "event_layer_presentation_worker_enable" in obj or "global_bev_event_layer_presentation_worker_enable" in obj:
            raw = obj.get("event_layer_presentation_worker_enable", obj.get("global_bev_event_layer_presentation_worker_enable"))
            self.args.event_layer_presentation_worker_enable = _safe_bool(
                raw,
                bool(getattr(self.args, "event_layer_presentation_worker_enable", False)),
            )
            if not bool(self.args.event_layer_presentation_worker_enable):
                self.args.event_layer_presentation_worker_mode = "off"
            elif str(getattr(self.args, "event_layer_presentation_worker_mode", "off") or "off") == "off":
                self.args.event_layer_presentation_worker_mode = "shadow"
            applied["event_layer_presentation_worker_enable"] = bool(self.args.event_layer_presentation_worker_enable)
        if "event_layer_presentation_worker_mode" in obj or "global_bev_event_layer_presentation_worker_mode" in obj:
            raw = obj.get("event_layer_presentation_worker_mode", obj.get("global_bev_event_layer_presentation_worker_mode"))
            mode = str(raw or "shadow").strip().lower().replace("-", "_")
            if mode not in {"off", "shadow", "active"}:
                mode = "shadow"
            self.args.event_layer_presentation_worker_mode = mode
            self.args.event_layer_presentation_worker_enable = bool(mode != "off")
            applied["event_layer_presentation_worker_mode"] = mode
        for key, lo, hi, default in (
            ("event_layer_presentation_worker_interval_ms", 5.0, 1000.0, 33.333),
            ("event_layer_presentation_bundle_max_age_ms", 1.0, 2000.0, 120.0),
            ("event_layer_presentation_target_tolerance_ticks", 0.0, 8.0, 1.0),
            ("event_layer_presentation_bundle_ring_size", 1.0, 64.0, 8.0),
            ("event_layer_presentation_prebuild_lookback_ticks", 0.0, 16.0, 2.0),
            ("event_layer_presentation_prebuild_lookahead_ticks", 0.0, 16.0, 2.0),
            ("event_layer_presentation_prebuild_max_per_cycle", 1.0, 8.0, 1.0),
            ("event_layer_presentation_bundle_cache_reuse_ms", 0.0, 1000.0, 16.0),
        ):
            alias = f"global_bev_{key}"
            if key in obj or alias in obj:
                raw = obj.get(key, obj.get(alias))
                val = max(float(lo), min(float(hi), _safe_float(raw, float(getattr(self.args, key, default)))))
                if key in {
                    "event_layer_presentation_target_tolerance_ticks",
                    "event_layer_presentation_bundle_ring_size",
                    "event_layer_presentation_prebuild_lookback_ticks",
                    "event_layer_presentation_prebuild_lookahead_ticks",
                    "event_layer_presentation_prebuild_max_per_cycle",
                }:
                    setattr(self.args, key, int(val))
                    applied[key] = int(getattr(self.args, key))
                else:
                    setattr(self.args, key, float(val))
                    applied[key] = float(getattr(self.args, key))
        if any(
            key in obj
            for key in (
                "event_layer_presentation_worker_enable",
                "global_bev_event_layer_presentation_worker_enable",
                "event_layer_presentation_worker_mode",
                "global_bev_event_layer_presentation_worker_mode",
                "event_layer_presentation_worker_interval_ms",
                "global_bev_event_layer_presentation_worker_interval_ms",
                "event_layer_presentation_bundle_max_age_ms",
                "global_bev_event_layer_presentation_bundle_max_age_ms",
                "event_layer_presentation_target_tolerance_ticks",
                "global_bev_event_layer_presentation_target_tolerance_ticks",
                "event_layer_presentation_bundle_ring_size",
                "global_bev_event_layer_presentation_bundle_ring_size",
                "event_layer_presentation_prebuild_lookback_ticks",
                "global_bev_event_layer_presentation_prebuild_lookback_ticks",
                "event_layer_presentation_prebuild_lookahead_ticks",
                "global_bev_event_layer_presentation_prebuild_lookahead_ticks",
                "event_layer_presentation_prebuild_max_per_cycle",
                "global_bev_event_layer_presentation_prebuild_max_per_cycle",
                "event_layer_presentation_bundle_cache_reuse_ms",
                "global_bev_event_layer_presentation_bundle_cache_reuse_ms",
                "event_layer_backend",
            )
        ):
            self._ensure_event_layer_presentation_worker()
            diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
            diag.update({
                "enabled": bool(getattr(self.args, "event_layer_presentation_worker_enable", False)),
                "mode": self._event_layer_presentation_mode(),
                "interval_ms": float(getattr(self.args, "event_layer_presentation_worker_interval_ms", 33.333) or 33.333),
                "bundle_max_age_ms": float(getattr(self.args, "event_layer_presentation_bundle_max_age_ms", 120.0) or 120.0),
                "target_tolerance_ticks": int(getattr(self.args, "event_layer_presentation_target_tolerance_ticks", 1) or 1),
                "bundle_ring_size": int(getattr(self.args, "event_layer_presentation_bundle_ring_size", 8) or 8),
                "prebuild_lookback_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookback_ticks", 2) or 2),
                "prebuild_lookahead_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookahead_ticks", 2) or 2),
                "prebuild_max_per_cycle": int(getattr(self.args, "event_layer_presentation_prebuild_max_per_cycle", 1) or 1),
                "bundle_cache_reuse_ms": float(getattr(self.args, "event_layer_presentation_bundle_cache_reuse_ms", 16.0) or 16.0),
            })
            self.event_layer_diag["presentation_worker"] = diag
        if "mvs_upload_budget_enable" in obj:
            self.args.mvs_upload_budget_enable = _safe_bool(
                obj.get("mvs_upload_budget_enable"),
                bool(getattr(self.args, "mvs_upload_budget_enable", True)),
            )
            self.mvs_upload_diag["budget_enable"] = bool(self.args.mvs_upload_budget_enable)
            applied["mvs_upload_budget_enable"] = bool(self.args.mvs_upload_budget_enable)
        if "mvs_upload_budget_ms" in obj:
            self.args.mvs_upload_budget_ms = max(
                0.0,
                _safe_float(obj.get("mvs_upload_budget_ms"), float(getattr(self.args, "mvs_upload_budget_ms", 10.0))),
            )
            self.mvs_upload_diag["budget_ms"] = float(self.args.mvs_upload_budget_ms)
            applied["mvs_upload_budget_ms"] = float(self.args.mvs_upload_budget_ms)
        if "event_layer_display_budget_enable" in obj:
            self.args.event_layer_display_budget_enable = _safe_bool(
                obj.get("event_layer_display_budget_enable"),
                bool(getattr(self.args, "event_layer_display_budget_enable", True)),
            )
            self.event_layer_diag["display_budget_enable"] = bool(self.args.event_layer_display_budget_enable)
            applied["event_layer_display_budget_enable"] = bool(self.args.event_layer_display_budget_enable)
        for key in ("event_layer_display_budget_ms", "event_layer_display_hold_ms"):
            if key in obj:
                setattr(self.args, key, max(0.0, _safe_float(obj.get(key), float(getattr(self.args, key, 10.0)))))
                diag_key = "display_budget_ms" if key == "event_layer_display_budget_ms" else "display_hold_ms"
                self.event_layer_diag[diag_key] = float(getattr(self.args, key))
                applied[key] = float(getattr(self.args, key))
        if "event_layer_auto_degrade_enable" in obj:
            self.args.event_layer_auto_degrade_enable = _safe_bool(
                obj.get("event_layer_auto_degrade_enable"),
                bool(getattr(self.args, "event_layer_auto_degrade_enable", True)),
            )
            applied["event_layer_auto_degrade_enable"] = bool(self.args.event_layer_auto_degrade_enable)
        for key in (
            "event_layer_auto_degrade_fps_threshold",
            "event_layer_auto_degrade_upload_p95_ms",
            "event_layer_auto_degrade_min_dilate_px",
        ):
            if key in obj:
                setattr(self.args, key, max(0.0, _safe_float(obj.get(key), float(getattr(self.args, key)))))
                applied[key] = float(getattr(self.args, key))
        if "event_layer_accumulate_enable" in obj:
            self.args.event_layer_accumulate_enable = _safe_bool(
                obj.get("event_layer_accumulate_enable"),
                bool(getattr(self.args, "event_layer_accumulate_enable", True)),
            )
            self.event_layer_diag["accumulate_enable"] = bool(self.args.event_layer_accumulate_enable)
            applied["event_layer_accumulate_enable"] = bool(self.args.event_layer_accumulate_enable)
        if "event_layer_accumulate_window_ms" in obj:
            self.args.event_layer_accumulate_window_ms = max(
                1.0,
                _safe_float(obj.get("event_layer_accumulate_window_ms"), float(getattr(self.args, "event_layer_accumulate_window_ms", 40.0))),
            )
            self.event_layer_diag["accumulate_window_ms"] = float(self.args.event_layer_accumulate_window_ms)
            applied["event_layer_accumulate_window_ms"] = float(self.args.event_layer_accumulate_window_ms)
        if "event_layer_accumulate_max_frames" in obj:
            self.args.event_layer_accumulate_max_frames = max(
                1,
                min(16, int(_safe_float(obj.get("event_layer_accumulate_max_frames"), float(getattr(self.args, "event_layer_accumulate_max_frames", 4))))),
            )
            self.event_layer_diag["accumulate_max_frames"] = int(self.args.event_layer_accumulate_max_frames)
            applied["event_layer_accumulate_max_frames"] = int(self.args.event_layer_accumulate_max_frames)
        if "event_layer_texture_scale" in obj:
            self.args.event_layer_texture_scale = max(
                0.25,
                min(1.0, _safe_float(obj.get("event_layer_texture_scale"), float(getattr(self.args, "event_layer_texture_scale", 1.0)))),
            )
            self.event_layer_diag["texture_scale"] = float(self.args.event_layer_texture_scale)
            applied["event_layer_texture_scale"] = float(self.args.event_layer_texture_scale)
        if "event_layer_history_poll_interval_ms" in obj:
            val = max(
                0.0,
                min(1000.0, _safe_float(obj.get("event_layer_history_poll_interval_ms"), float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0)))),
            )
            self.args.event_layer_history_poll_interval_ms = float(val)
            for rd in list(self.event_layer_readers.values()):
                try:
                    rd.history_poll_interval_ms = float(val)
                except Exception:
                    pass
            for rd in list(self.event_layer_sparse_readers.values()):
                try:
                    rd.poll_interval_ms = float(val)
                except Exception:
                    pass
            self.event_layer_diag["history_poll_interval_ms"] = float(val)
            applied["event_layer_history_poll_interval_ms"] = float(val)
        if "global_person_overlay_max_speed_mps" in obj:
            self.args.global_person_overlay_max_speed_mps = max(
                0.0,
                _safe_float(obj.get("global_person_overlay_max_speed_mps"), float(self.args.global_person_overlay_max_speed_mps)),
            )
            applied["global_person_overlay_max_speed_mps"] = float(self.args.global_person_overlay_max_speed_mps)
        if "global_person_overlay_lost_max_age_ms" in obj:
            self.args.global_person_overlay_lost_max_age_ms = _safe_float(
                obj.get("global_person_overlay_lost_max_age_ms"),
                float(self.args.global_person_overlay_lost_max_age_ms),
            )
            applied["global_person_overlay_lost_max_age_ms"] = float(self.args.global_person_overlay_lost_max_age_ms)
        if "global_person_overlay_display_stabilize_enable" in obj:
            self.args.global_person_overlay_display_stabilize_enable = _safe_bool(
                obj.get("global_person_overlay_display_stabilize_enable"),
                bool(getattr(self.args, "global_person_overlay_display_stabilize_enable", True)),
            )
            applied["global_person_overlay_display_stabilize_enable"] = bool(self.args.global_person_overlay_display_stabilize_enable)
        for key in (
            "global_person_overlay_display_alpha",
            "global_person_overlay_reid_gate_m",
            "global_person_overlay_reid_ttl_ms",
            "global_person_overlay_hold_ms",
        ):
            if key in obj:
                setattr(self.args, key, _safe_float(obj.get(key), float(getattr(self.args, key))))
                applied[key] = float(getattr(self.args, key))
        if "manual_hit_click_enable" in obj or "global_bev_manual_hit_click_enable" in obj:
            raw = obj.get("manual_hit_click_enable", obj.get("global_bev_manual_hit_click_enable"))
            self.manual_hit_click_enable = _safe_bool(raw, bool(getattr(self, "manual_hit_click_enable", False)))
            applied["manual_hit_click_enable"] = bool(self.manual_hit_click_enable)
        if "manual_hit_click_radius_px" in obj or "global_bev_manual_hit_click_radius_px" in obj:
            raw = obj.get("manual_hit_click_radius_px", obj.get("global_bev_manual_hit_click_radius_px"))
            self.manual_hit_click_radius_px = max(4.0, min(160.0, _safe_float(raw, float(self.manual_hit_click_radius_px))))
            applied["manual_hit_click_radius_px"] = float(self.manual_hit_click_radius_px)
        if "turf_target_rgb" in obj:
            vals = _parse_rgb_triplet(obj.get("turf_target_rgb"), tuple(float(x) for x in self.turf_target_rgb))
            self.turf_target_rgb = np.asarray(vals, dtype=np.float32)
            self.args.turf_target_r, self.args.turf_target_g, self.args.turf_target_b = vals
            applied["turf_target_rgb"] = [float(x) for x in self.turf_target_rgb]
        if isinstance(obj.get("color_gain"), (list, tuple)) and len(obj.get("color_gain")) >= 3:
            vals = obj.get("color_gain")
            self.args.color_gain_r = _safe_float(vals[0], float(self.args.color_gain_r))
            self.args.color_gain_g = _safe_float(vals[1], float(self.args.color_gain_g))
            self.args.color_gain_b = _safe_float(vals[2], float(self.args.color_gain_b))
            applied["color_gain"] = [float(self.args.color_gain_r), float(self.args.color_gain_g), float(self.args.color_gain_b)]
        for suffix in ("r", "g", "b"):
            key = f"color_gain_{suffix}"
            if key in obj:
                setattr(self.args, key, _safe_float(obj.get(key), float(getattr(self.args, key))))
                applied[key] = float(getattr(self.args, key))
        return applied

    def _runtime_state_obj(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "saved_us": _now_us(),
            "field_transform_mode": str(self.args.field_transform_mode),
            "debug_mode": str(self.args.debug_mode),
            "texture_y_flip": bool(self.args.texture_y_flip),
            "grid_enable": bool(self.args.grid_enable),
            "global_person_overlay_coord_mode": str(getattr(self.args, "global_person_overlay_coord_mode", "field_xy")),
            "event_bullet_overlay_enable": bool(getattr(self.args, "event_bullet_overlay_enable", False)),
            "event_bullet_overlay_transform_mode": str(getattr(self.args, "event_bullet_overlay_transform_mode", "normal")),
            "manual_hit_click_enable": bool(getattr(self, "manual_hit_click_enable", False)),
            "manual_hit_click_radius_px": float(getattr(self, "manual_hit_click_radius_px", 42.0)),
            "event_layer_enable": bool(getattr(self.args, "event_layer_enable", False)),
            "event_layer_backend": str(getattr(self.args, "event_layer_backend", "texture") or "texture"),
            "event_layer_sparse_sidecar_template": str(getattr(self.args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
            "event_layer_sparse_max_points_draw": int(getattr(self.args, "event_layer_sparse_max_points_draw", 20000) or 20000),
            "event_layer_presentation_worker_enable": bool(getattr(self.args, "event_layer_presentation_worker_enable", False)),
            "event_layer_presentation_worker_mode": self._event_layer_presentation_mode(),
            "event_layer_presentation_worker_interval_ms": float(getattr(self.args, "event_layer_presentation_worker_interval_ms", 33.333)),
            "event_layer_presentation_bundle_max_age_ms": float(getattr(self.args, "event_layer_presentation_bundle_max_age_ms", 120.0)),
            "event_layer_presentation_target_tolerance_ticks": int(getattr(self.args, "event_layer_presentation_target_tolerance_ticks", 1)),
            "event_layer_presentation_bundle_ring_size": int(getattr(self.args, "event_layer_presentation_bundle_ring_size", 8)),
            "event_layer_presentation_prebuild_lookback_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookback_ticks", 2)),
            "event_layer_presentation_prebuild_lookahead_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookahead_ticks", 2)),
            "event_layer_presentation_prebuild_max_per_cycle": int(getattr(self.args, "event_layer_presentation_prebuild_max_per_cycle", 1)),
            "event_layer_presentation_bundle_cache_reuse_ms": float(getattr(self.args, "event_layer_presentation_bundle_cache_reuse_ms", 16.0)),
            "event_layer_alpha": float(getattr(self.args, "event_layer_alpha", 0.45)),
            "event_layer_gain": float(getattr(self.args, "event_layer_gain", 2.4)),
            "event_layer_min_value": float(getattr(self.args, "event_layer_min_value", 0.03)),
            "event_layer_max_age_ms": float(getattr(self.args, "event_layer_max_age_ms", 150.0)),
            "event_layer_sync_mode": _event_layer_sync_mode(getattr(self.args, "event_layer_sync_mode", "latest_guarded")),
            "event_layer_max_sync_delta_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 120.0)),
            "event_layer_max_sync_delta_ticks": int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2)),
            "event_layer_sync_normal_ms": float(getattr(self.args, "event_layer_sync_normal_ms", 50.0)),
            "event_layer_sync_warn_ms": float(getattr(self.args, "event_layer_sync_warn_ms", 80.0)),
            "event_layer_sync_hard_ms": float(getattr(self.args, "event_layer_sync_hard_ms", 80.0)),
            "event_layer_hold_last_good_on_hard_outlier": bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True)),
            "event_layer_hold_last_good_ms": float(getattr(self.args, "event_layer_hold_last_good_ms", 250.0)),
            "event_layer_fade_alpha_scale": float(getattr(self.args, "event_layer_fade_alpha_scale", 0.25)),
            "event_layer_color_mode": _event_layer_color_mode(getattr(self.args, "event_layer_color_mode", "camera")),
            "event_layer_dilate_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
            "event_layer_dilate_max_pixels": int(getattr(self.args, "event_layer_dilate_max_pixels", 2500)),
            "event_layer_auto_degrade_enable": bool(getattr(self.args, "event_layer_auto_degrade_enable", True)),
            "event_layer_auto_degrade_fps_threshold": float(getattr(self.args, "event_layer_auto_degrade_fps_threshold", 28.0)),
            "event_layer_auto_degrade_upload_p95_ms": float(getattr(self.args, "event_layer_auto_degrade_upload_p95_ms", 8.0)),
            "event_layer_auto_degrade_min_dilate_px": float(getattr(self.args, "event_layer_auto_degrade_min_dilate_px", 4.0)),
            "event_layer_accumulate_enable": bool(getattr(self.args, "event_layer_accumulate_enable", True)),
            "event_layer_accumulate_window_ms": float(getattr(self.args, "event_layer_accumulate_window_ms", 40.0)),
            "event_layer_accumulate_max_frames": int(getattr(self.args, "event_layer_accumulate_max_frames", 4)),
            "event_layer_texture_scale": float(getattr(self.args, "event_layer_texture_scale", 1.0)),
            "event_layer_history_poll_interval_ms": float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0)),
            "event_layer_sync_map_ring_template": str(getattr(self.args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
            "alignment_mode": _alignment_mode(getattr(self.args, "alignment_mode", "current")),
            "alignment_delay_ms": float(getattr(self.args, "alignment_delay_ms", getattr(self.args, "presentation_delay_ms", 820.0))),
            "alignment_shadow_interval_ms": float(getattr(self.args, "alignment_shadow_interval_ms", 250.0)),
            "alignment_hold_warn_ms": float(getattr(self.args, "alignment_hold_warn_ms", 150.0)),
            "gain": float(self.args.gain),
            "black_level": float(self.args.black_level),
            "gamma": float(self.args.gamma),
            "color_gain": [float(self.args.color_gain_r), float(self.args.color_gain_g), float(self.args.color_gain_b)],
            "turf_match_enable": bool(getattr(self.args, "turf_match_enable", False)),
            "turf_auto_match": bool(getattr(self.args, "turf_auto_match", False)),
            "turf_match_strength": float(getattr(self.args, "turf_match_strength", 0.0)),
            "frame_hold_ms": float(getattr(self.args, "frame_hold_ms", 250.0)),
            "max_mvs_sync_delta_ticks": int(getattr(self.args, "max_mvs_sync_delta_ticks", 2)),
            "mvs_sync_outlier_policy": str(getattr(self.args, "mvs_sync_outlier_policy", "hide") or "hide"),
            "mvs_sync_tick_ms_est": float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0)),
        }

    def _save_runtime_state(self, reason: str = "manual") -> None:
        if not bool(getattr(self.args, "runtime_state_enable", True)):
            return
        try:
            obj = self._runtime_state_obj()
            obj["reason"] = str(reason)
            self.runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.runtime_state_path.with_suffix(self.runtime_state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.runtime_state_path)
            self.control_diag["runtime_state_saved_us"] = int(obj["saved_us"])
            self.control_diag["runtime_state_reason"] = str(reason)
        except Exception as exc:
            self.control_diag["runtime_state_error"] = f"{type(exc).__name__}: {exc}"

    def _poll_control_json(self) -> None:
        path = self.control_path
        if not str(path):
            return
        try:
            st = path.stat()
        except FileNotFoundError:
            return
        except Exception as exc:
            self.control_diag["error"] = f"{type(exc).__name__}: {exc}"
            return
        if int(st.st_mtime_ns) <= int(self._control_mtime_ns):
            return
        self._control_mtime_ns = int(st.st_mtime_ns)
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(obj, dict):
                raise ValueError("control JSON root must be an object")
            applied = self._apply_control_obj(obj)
            self._control_seq += 1
            self.control_diag.update({
                "path": str(path),
                "seq": int(self._control_seq),
                "mtime_ns": int(self._control_mtime_ns),
                "updated_us": _now_us(),
                "applied": applied,
                "error": "",
            })
            if applied:
                print(f"[global_bev][control] applied {applied}", flush=True)
        except Exception as exc:
            self.control_diag.update({
                "path": str(path),
                "mtime_ns": int(self._control_mtime_ns),
                "updated_us": _now_us(),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[global_bev][control][WARN] {self.control_diag['error']}", flush=True)

    def _poll_global_person_json(self) -> None:
        if not bool(getattr(self.args, "global_person_overlay_enable", False)):
            return
        path = self.global_person_path
        try:
            st = path.stat()
        except FileNotFoundError:
            self.global_person_diag.update({"enabled": True, "missing": True, "path": str(path)})
            return
        except Exception as exc:
            self.global_person_diag.update({"enabled": True, "error": f"{type(exc).__name__}: {exc}", "path": str(path)})
            return
        if int(st.st_mtime_ns) <= int(self._global_person_mtime_ns):
            return
        self._global_person_mtime_ns = int(st.st_mtime_ns)
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            tracks = obj.get("tracks", []) if isinstance(obj, dict) else []
            if not isinstance(tracks, list):
                tracks = []
            self.global_person_tracks = [t for t in tracks if isinstance(t, dict)]
            self.global_person_meta = {
                "timestamp_us": obj.get("timestamp_us") if isinstance(obj, dict) else None,
                "sync_index": obj.get("sync_index") if isinstance(obj, dict) else None,
                "track_count": obj.get("track_count") if isinstance(obj, dict) else None,
                "coord_transform": obj.get("coord_transform", {}) if isinstance(obj, dict) else {},
                "stats": obj.get("stats", {}) if isinstance(obj, dict) else {},
            }
            self.global_person_diag.update({
                "enabled": True,
                "missing": False,
                "path": str(path),
                "mtime_ns": int(self._global_person_mtime_ns),
                "track_count": len(self.global_person_tracks),
                "sync_index": int(obj.get("sync_index", -1)) if isinstance(obj, dict) else -1,
                "coord_transform": self.global_person_meta.get("coord_transform", {}),
                "stats": self.global_person_meta.get("stats", {}),
                "overlay_coord_mode": str(getattr(self.args, "global_person_overlay_coord_mode", "field_xy")),
                "presentation_mode": "latest_fallback",
                "presentation_target_sync_id": int(getattr(self, "current_target_sync_id", -1)),
                "presentation_delay_ms": float(self.presentation_diag.get("delay_ms", 0.0) or 0.0),
                "updated_us": _now_us(),
                "error": "",
            })
        except Exception as exc:
            self.global_person_diag.update({"enabled": True, "error": f"{type(exc).__name__}: {exc}", "path": str(path)})

    def _poll_event_bullet_jsonl(self) -> None:
        if not bool(getattr(self.args, "event_bullet_overlay_enable", False)):
            self.event_bullet_diag.update({"enabled": False})
            return
        now_ms = time.monotonic() * 1000.0
        poll_ms = max(5.0, float(getattr(self.args, "event_bullet_overlay_poll_ms", 40.0) or 40.0))
        if now_ms - float(self._event_bullet_last_poll_ms) < poll_ms:
            return
        self._event_bullet_last_poll_ms = now_ms

        max_records = max(32, int(getattr(self.args, "event_bullet_overlay_max_records", 4096) or 4096))
        new_records = self.event_bullet_tailer.read_new(max_lines=max_records)
        accepted = 0
        for rec in new_records:
            if str(rec.get("type", "")) != "overlay_bullet_point":
                continue
            self.event_bullet_records.append(rec)
            accepted += 1

        now_wall = time.time()
        life_ms = max(1.0, float(getattr(self.args, "event_bullet_overlay_life_ms", 2500.0) or 2500.0))
        keep_sec = max(10.0, life_ms * 4.0 / 1000.0)
        kept: List[Dict[str, Any]] = []
        for rec in self.event_bullet_records[-max_records * 2:]:
            wall_time = _safe_float(rec.get("wall_time"), 0.0)
            if wall_time > 1.0 and now_wall - wall_time > keep_sec:
                continue
            kept.append(rec)
        if len(kept) > max_records:
            kept = kept[-max_records:]
        self.event_bullet_records = kept

        try:
            st = self.event_bullet_path.stat()
            file_age_ms = max(0.0, (time.time_ns() - int(st.st_mtime_ns)) / 1e6)
            missing = False
        except FileNotFoundError:
            file_age_ms = -1.0
            missing = True
        except Exception:
            file_age_ms = -1.0
            missing = False
        latest_rec = None
        latest_wall = 0.0
        for rec in self.event_bullet_records:
            wall = _safe_float(rec.get("wall_time"), 0.0)
            if wall >= latest_wall:
                latest_wall = wall
                latest_rec = rec
        latest_age_ms = max(0.0, (now_wall - latest_wall) * 1000.0) if latest_wall > 1.0 else -1.0
        self.event_bullet_diag.update({
            "enabled": True,
            "path": str(self.event_bullet_path),
            "missing": bool(missing),
            "file_size_bytes": int(self.event_bullet_tailer.last_size),
            "file_age_ms": round(float(file_age_ms), 3),
            "tail_offset": int(self.event_bullet_tailer.offset),
            "records_buffered": int(len(self.event_bullet_records)),
            "records_accepted_this_poll": int(accepted),
            "tail_good_lines": int(self.event_bullet_tailer.good_lines),
            "tail_bad_lines": int(self.event_bullet_tailer.bad_lines),
            "latest_age_ms": round(float(latest_age_ms), 3),
            "last_record_camera": str((latest_rec or {}).get("camera_alias") or (latest_rec or {}).get("pair_id") or ""),
            "last_record_bullet_id": _safe_int((latest_rec or {}).get("bullet_id"), 0),
            "last_record_point_index": _safe_int((latest_rec or {}).get("point_index"), -1),
            "last_record_age_ms": round(float(latest_age_ms), 3),
            "life_ms": float(life_ms),
            "error": str(self.event_bullet_tailer.error or ""),
        })

    def _event_bullet_img_to_field(self, cam: str) -> Optional[np.ndarray]:
        cam = str(cam or "").strip()
        if not cam:
            return None
        cached = self._event_bullet_img_to_field_cache.get(cam)
        if cached is not None:
            return cached
        H = self.H.get(cam)
        if H is None:
            return None
        try:
            inv = np.linalg.inv(np.asarray(H, dtype=np.float64).reshape(3, 3))
        except Exception as exc:
            self.event_bullet_diag["homography_error"] = f"{cam}: {type(exc).__name__}: {exc}"
            return None
        self._event_bullet_img_to_field_cache[cam] = inv
        return inv

    def _event_bullet_record_field_xy(self, rec: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], str]:
        transform_mode = str(getattr(self.args, "event_bullet_overlay_transform_mode", "normal") or "normal").strip().lower().replace("-", "_")
        cached = rec.get("_bev_field_xy")
        cached_mode = str(rec.get("_bev_field_xy_transform_mode", "normal") or "normal")
        if cached_mode == transform_mode and isinstance(cached, (list, tuple)) and len(cached) >= 2:
            try:
                fx = float(cached[0])
                fy = float(cached[1])
                if math.isfinite(fx) and math.isfinite(fy):
                    self._event_bullet_projection_cache_hits += 1
                    return (fx, fy), ""
            except Exception:
                pass
        raw_cached = rec.get("_bev_raw_field_xy")
        if isinstance(raw_cached, (list, tuple)) and len(raw_cached) >= 2:
            try:
                raw_fx = float(raw_cached[0])
                raw_fy = float(raw_cached[1])
                if math.isfinite(raw_fx) and math.isfinite(raw_fy):
                    fx, fy = self._transform_event_bullet_xy_for_overlay(raw_fx, raw_fy)
                    rec["_bev_field_xy"] = [float(fx), float(fy)]
                    rec["_bev_field_xy_transform_mode"] = transform_mode
                    self._event_bullet_projection_cache_hits += 1
                    return (fx, fy), ""
            except Exception:
                pass
        cam = str(rec.get("camera_alias") or rec.get("pair_id") or "").strip()
        inv = self._event_bullet_img_to_field(cam)
        if inv is None:
            return None, "no_calib"
        x = _safe_float(rec.get("x"), float("nan"))
        y = _safe_float(rec.get("y"), float("nan"))
        if not (math.isfinite(x) and math.isfinite(y)):
            return None, "bad_xy"
        try:
            p = inv @ np.asarray([x, y, 1.0], dtype=np.float64)
            if abs(float(p[2])) < 1e-9:
                return None, "bad_homogeneous"
            fx = float(p[0] / p[2])
            fy = float(p[1] / p[2])
        except Exception:
            return None, "project_error"
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return None, "nonfinite_field"
        rec["_bev_raw_field_xy"] = [float(fx), float(fy)]
        fx, fy = self._transform_event_bullet_xy_for_overlay(fx, fy)
        rec["_bev_field_xy"] = [float(fx), float(fy)]
        rec["_bev_field_xy_transform_mode"] = transform_mode
        self._event_bullet_projection_cache_misses += 1
        return (fx, fy), ""

    def _event_bullet_color(self, cam: str, bullet_id: int) -> QColor:
        palette = {
            "A": QColor(255, 80, 80),
            "B": QColor(255, 180, 70),
            "C": QColor(255, 245, 90),
            "D": QColor(100, 255, 130),
            "E": QColor(80, 230, 255),
            "F": QColor(120, 160, 255),
            "G": QColor(210, 120, 255),
            "H": QColor(255, 120, 205),
        }
        return palette.get(str(cam or "").strip(), QColor(255, 230, 90))

    def _draw_event_bullet_overlay(self) -> None:
        if not bool(getattr(self.args, "event_bullet_overlay_enable", False)):
            self.event_bullet_diag.update({"enabled": False, "skip_reason": "disabled"})
            return
        self._poll_event_bullet_jsonl()
        now_wall = time.time()
        life_ms = max(1.0, float(getattr(self.args, "event_bullet_overlay_life_ms", 2500.0) or 2500.0))
        min_conf = max(0.0, float(getattr(self.args, "event_bullet_overlay_min_confidence", 0.0) or 0.0))
        max_gap_ms = max(1.0, float(getattr(self.args, "event_bullet_overlay_max_segment_gap_ms", 80.0) or 80.0))
        radius = max(1.0, float(getattr(self.args, "event_bullet_overlay_dilate_px", 4.0) or 4.0))
        line_width = max(1.0, float(getattr(self.args, "event_bullet_overlay_line_width", 2.5) or 2.5))
        groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        skip: Dict[str, int] = {}
        recent_records = 0
        for rec in self.event_bullet_records:
            wall_time = _safe_float(rec.get("wall_time"), 0.0)
            age_ms = max(0.0, (now_wall - wall_time) * 1000.0) if wall_time > 1.0 else 0.0
            if wall_time > 1.0 and age_ms > life_ms:
                continue
            if _safe_float(rec.get("confidence"), 1.0) < min_conf:
                skip["low_conf"] = skip.get("low_conf", 0) + 1
                continue
            cam = str(rec.get("camera_alias") or rec.get("pair_id") or "").strip()
            bid = _safe_int(rec.get("bullet_id"), 0)
            groups.setdefault((cam, bid), []).append(rec)
            recent_records += 1

        painter = QPainter(self)
        draw_points = 0
        draw_segments = 0
        projected_points = 0
        visible_points = 0
        try:
            for (cam, bid), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                recs.sort(key=lambda r: (_safe_int(r.get("point_ts_us"), 0), _safe_int(r.get("point_index"), 0), _safe_int(r.get("bullet_seq_id"), 0)))
                color = self._event_bullet_color(cam, bid)
                last_pt: Optional[Tuple[float, float]] = None
                last_ts = 0
                last_draw_xy: Optional[Tuple[float, float]] = None
                for rec in recs:
                    field_xy, reason = self._event_bullet_record_field_xy(rec)
                    if field_xy is None:
                        skip[reason] = skip.get(reason, 0) + 1
                        continue
                    projected_points += 1
                    widget_xy = self._field_xy_to_widget_xy(field_xy[0], field_xy[1])
                    if widget_xy is None:
                        skip["out_of_field"] = skip.get("out_of_field", 0) + 1
                        last_pt = None
                        last_draw_xy = None
                        continue
                    visible_points += 1
                    wall_time = _safe_float(rec.get("wall_time"), 0.0)
                    age_ms = max(0.0, (now_wall - wall_time) * 1000.0) if wall_time > 1.0 else 0.0
                    alpha = int(max(60, min(245, 245.0 * (1.0 - min(1.0, age_ms / max(1.0, life_ms))) + 35.0)))
                    pt_color = QColor(color.red(), color.green(), color.blue(), alpha)
                    painter.setPen(QPen(pt_color, line_width))
                    painter.setBrush(QColor(color.red(), color.green(), color.blue(), max(45, min(150, alpha // 2))))
                    px, py = widget_xy
                    ts = _safe_int(rec.get("point_ts_us"), 0)
                    if last_pt is not None and last_draw_xy is not None and ts > 0 and last_ts > 0 and (ts - last_ts) / 1000.0 <= max_gap_ms:
                        painter.drawLine(int(last_draw_xy[0]), int(last_draw_xy[1]), int(px), int(py))
                        draw_segments += 1
                    painter.drawEllipse(int(px - radius), int(py - radius), int(radius * 2), int(radius * 2))
                    draw_points += 1
                    last_pt = field_xy
                    last_ts = ts
                    last_draw_xy = (px, py)
                if last_draw_xy is not None and bool(getattr(self.args, "event_bullet_overlay_label_enable", True)):
                    label = f"{cam} B{bid}"
                    painter.setPen(QColor(255, 255, 220, 230))
                    painter.drawText(int(last_draw_xy[0] + radius + 4), int(last_draw_xy[1] - radius - 4), label)
        finally:
            painter.end()

        if draw_points > 0:
            skip_reason = "drawing"
        elif int(self.event_bullet_diag.get("records_buffered", 0) or 0) <= 0:
            skip_reason = "no_buffered_records"
        elif recent_records <= 0:
            skip_reason = "no_recent_records"
        elif projected_points <= 0:
            skip_reason = "no_projected_points"
        elif visible_points <= 0:
            skip_reason = "all_projected_out_of_field"
        else:
            skip_reason = "no_draw_points"
        projection_fail = {
            str(k): int(v)
            for k, v in skip.items()
            if str(k) not in {"low_conf", "out_of_field"}
        }
        self.event_bullet_diag.update({
            "enabled": True,
            "recent_records": int(recent_records),
            "group_count": int(len(groups)),
            "projected_points": int(projected_points),
            "visible_points": int(visible_points),
            "draw_points": int(draw_points),
            "draw_segments": int(draw_segments),
            "skip_counts": dict(skip),
            "projection_fail_count_by_reason": projection_fail,
            "skip_reason": skip_reason,
            "dilate_px": float(radius),
            "line_width": float(line_width),
            "max_segment_gap_ms": float(max_gap_ms),
            "coord_source": "overlay_bullet_point.x_y_mvs_pixel + inverse_global_bev_field_to_img",
            "transform_mode": str(getattr(self.args, "event_bullet_overlay_transform_mode", "normal") or "normal"),
            "render_target": "qt_overlay",
            "presentation_mode": "latest_fallback",
            "presentation_target_sync_id": int(getattr(self, "current_target_sync_id", -1)),
            "presentation_delay_ms": float(self.presentation_diag.get("delay_ms", 0.0) or 0.0),
            "projection_cache_hits": int(self._event_bullet_projection_cache_hits),
            "projection_cache_misses": int(self._event_bullet_projection_cache_misses),
        })

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        try:
            if event.button() == Qt.LeftButton and bool(getattr(self, "manual_hit_click_enable", False)):
                pos = event.position() if hasattr(event, "position") else event.pos()
                px = float(pos.x())
                py = float(pos.y())
                self._handle_manual_hit_click(px, py)
                event.accept()
                self.update()
                return
        except Exception as exc:
            self.manual_hit_last_result = {
                "state": "click_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_time_us": _now_us(),
            }
            print(f"[global_bev][manual_hit][WARN] click failed: {self.manual_hit_last_result['error']}", flush=True)
        return super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = int(event.key())
        handled = True
        if key == _qt_key_value("Key_M"):
            self._cycle_transform_mode(1)
        elif key == _qt_key_value("Key_N"):
            self._cycle_transform_mode(-1)
        elif key == _qt_key_value("Key_Q"):
            self._cycle_person_overlay_coord_mode(1)
        elif key == _qt_key_value("Key_D"):
            self._cycle_debug_mode(1)
        elif key == _qt_key_value("Key_T"):
            self.args.texture_y_flip = not bool(self.args.texture_y_flip)
        elif key == _qt_key_value("Key_G"):
            self.args.grid_enable = not bool(self.args.grid_enable)
        elif key == _qt_key_value("Key_R"):
            self._set_transform_mode("normal")
        else:
            handled = False
        if handled:
            self._save_runtime_state("hotkey")
            self.control_diag.update({
                "updated_us": _now_us(),
                "last_hotkey": int(key),
                "applied": {
                    "field_transform_mode": str(self.args.field_transform_mode),
                    "global_person_overlay_coord_mode": str(getattr(self.args, "global_person_overlay_coord_mode", "field_xy")),
                    "debug_mode": str(self.args.debug_mode),
                    "texture_y_flip": bool(self.args.texture_y_flip),
                    "grid_enable": bool(self.args.grid_enable),
                },
            })
            print(f"[global_bev][hotkey] {self.control_diag['applied']}", flush=True)
            event.accept()
            self.update()
            return
        return super().keyPressEvent(event)

    def _event_layer_homography_paths(self) -> List[Path]:
        project_root = Path(str(getattr(self.args, "project_root", "."))).expanduser()
        raw = str(getattr(self.args, "event_layer_homography_config", "") or "").strip()
        names: List[str] = []
        if raw:
            names.extend([p.strip() for p in raw.split(",") if p.strip()])
        for fallback in DEFAULT_EVENT_LAYER_HOMOGRAPHY_CONFIGS:
            if fallback not in names:
                names.append(fallback)
        paths: List[Path] = []
        seen = set()
        for name in names:
            p = Path(name).expanduser()
            if not p.is_absolute():
                p = project_root / p
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
        return paths

    def _load_event_layer_homographies(self) -> None:
        self.event_layer_mvs_to_event.clear()
        self.event_layer_event_to_mvs.clear()
        candidate_paths = self._event_layer_homography_paths()
        self.event_layer_diag.update({
            "homography_candidate_configs": [str(p) for p in candidate_paths],
            "homography_loaded_by_cam": {},
            "missing_homography_cams": list(self.cams),
            "homography_loaded_count": 0,
        })
        cfg: Dict[str, Any] = {}
        cfg_path: Optional[Path] = None
        errors: List[str] = []
        for path in candidate_paths:
            try:
                if not path.exists():
                    continue
                obj = json.loads(path.read_text(encoding="utf-8-sig"))
                event_overlay = ((obj.get("mvs_config") or {}).get("event_overlay") or {}) if isinstance(obj, dict) else {}
                per_route = event_overlay.get("per_route") or {}
                if isinstance(per_route, dict) and per_route:
                    cfg = event_overlay
                    cfg_path = path
                    break
                errors.append(f"{path}: no mvs_config.event_overlay.per_route")
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if cfg_path is None:
            self.event_layer_diag.update({
                "homography_error": "; ".join(errors) if errors else "no event-layer homography config found",
                "homography_config": "",
            })
            return

        per_route = cfg.get("per_route") or {}
        loaded: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        default_w, default_h = DEFAULT_EVENT_LAYER_EVENT_SIZE
        event_w_default = int(_safe_float(cfg.get("event_width"), float(default_w)))
        event_h_default = int(_safe_float(cfg.get("event_height"), float(default_h)))
        for cam in self.cams:
            try:
                route = per_route.get(cam) or {}
                if not isinstance(route, dict):
                    raise ValueError("route is not an object")
                homos = route.get("homographies_event_to_mvs") or {}
                if not isinstance(homos, dict) or not homos:
                    raise KeyError("homographies_event_to_mvs missing")
                active = str(route.get("active_homography") or next(iter(homos.keys())))
                raw_h = homos.get(active)
                if raw_h is None:
                    raise KeyError(f"active homography {active!r} missing")
                h_event_to_mvs = np.asarray(raw_h, dtype=np.float64)
                if h_event_to_mvs.size == 9:
                    h_event_to_mvs = h_event_to_mvs.reshape(3, 3)
                if h_event_to_mvs.shape != (3, 3):
                    raise ValueError(f"homography shape {h_event_to_mvs.shape}")
                if abs(float(h_event_to_mvs[2, 2])) > 1e-12:
                    h_event_to_mvs = h_event_to_mvs / float(h_event_to_mvs[2, 2])
                h_mvs_to_event = np.linalg.inv(h_event_to_mvs)
                self.event_layer_event_to_mvs[cam] = h_event_to_mvs.astype(np.float32)
                self.event_layer_mvs_to_event[cam] = h_mvs_to_event.astype(np.float32)

                event_w = max(1, int(_safe_float(route.get("event_width", cfg.get("event_width")), float(event_w_default))))
                event_h = max(1, int(_safe_float(route.get("event_height", cfg.get("event_height")), float(event_h_default))))
                pts = np.asarray(
                    [[0.0, 0.0, 1.0], [float(event_w), 0.0, 1.0], [float(event_w), float(event_h), 1.0], [0.0, float(event_h), 1.0], [float(event_w) * 0.5, float(event_h) * 0.5, 1.0]],
                    dtype=np.float64,
                ).T
                q = h_event_to_mvs @ pts
                z = q[2:3, :]
                z = np.where(np.abs(z) < 1e-12, 1e-12, z)
                q = q[:2, :] / z
                xs = q[0, :4]
                ys = q[1, :4]
                loaded[cam] = {
                    "config": str(cfg_path),
                    "active_homography": active,
                    "event_size": [int(event_w), int(event_h)],
                    "event_to_mvs_first_row": [float(x) for x in h_event_to_mvs[0, :]],
                    "mvs_to_event_first_row": [float(x) for x in h_mvs_to_event[0, :]],
                    "event_fov_mvs_bbox": [round(float(xs.min()), 3), round(float(ys.min()), 3), round(float(xs.max()), 3), round(float(ys.max()), 3)],
                    "event_center_mvs": [round(float(q[0, 4]), 3), round(float(q[1, 4]), 3)],
                }
            except Exception as exc:
                missing.append(cam)
                loaded[cam] = {"error": f"{type(exc).__name__}: {exc}"}

        self.event_layer_diag.update({
            "homography_config": str(cfg_path),
            "homography_loaded_by_cam": loaded,
            "homography_loaded_count": int(len(self.event_layer_mvs_to_event)),
            "missing_homography_cams": list(missing),
            "homography_error": "",
        })

    def _open_inputs(self) -> None:
        for cam in self.cams:
            if cam not in self.readers:
                name = str(self.args.raw_template).format(camera=cam, cam=cam, id=cam)
                try:
                    self.readers[cam] = RawPreviewFrameReader(name)
                    self.reader_errors.pop(cam, None)
                    print(f"[global_bev][{cam}] opened raw ring {name} slots={self.readers[cam].slot_count}", flush=True)
                except Exception as exc:
                    self.reader_errors[cam] = f"{type(exc).__name__}: {exc}"
            if cam not in self.H:
                p = Path(str(self.args.calib_template).format(camera=cam, cam=cam, id=cam))
                if not p.is_absolute():
                    p = Path(self.args.project_root) / p
                try:
                    H, meta = _load_homography(p)
                    self.H[cam] = H
                    self.calib[cam] = meta
                    print(
                        f"[global_bev][{cam}] loaded calib {p} points={meta['point_count']} "
                        f"matrix_source={meta.get('matrix_source', '')}",
                        flush=True,
                    )
                except Exception as exc:
                    self.calib[cam] = {"path": str(p), "error": f"{type(exc).__name__}: {exc}"}

    def initializeGL(self) -> None:
        vendor = GL.glGetString(GL.GL_VENDOR)
        renderer = GL.glGetString(GL.GL_RENDERER)
        version = GL.glGetString(GL.GL_VERSION)
        print(f"[global_bev] GL_VENDOR={vendor!r} GL_RENDERER={renderer!r} GL_VERSION={version!r}", flush=True)
        try:
            self.max_texture_units = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_IMAGE_UNITS))
        except Exception:
            self.max_texture_units = 0
        print(f"[global_bev] GL_MAX_TEXTURE_IMAGE_UNITS={self.max_texture_units}", flush=True)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glClearColor(0.02, 0.025, 0.03, 1.0)
        self.program = shaders.compileProgram(shaders.compileShader(VERT_SRC, GL.GL_VERTEX_SHADER), shaders.compileShader(FRAG_SRC, GL.GL_FRAGMENT_SHADER))
        self.display_program = shaders.compileProgram(shaders.compileShader(VERT_SRC, GL.GL_VERTEX_SHADER), shaders.compileShader(DISPLAY_FRAG_SRC, GL.GL_FRAGMENT_SHADER))
        self._create_fbo()
        self.writer = BevSharedFrameWriter(str(self.args.output_shm), self.out_w, self.out_h)
        self._open_inputs()

    def _create_fbo(self) -> None:
        self.fbo_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.out_w, self.out_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        self.fbo = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.fbo_tex, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"BEV FBO incomplete: 0x{int(status):x}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._qt_default_fbo())
        try:
            pbo_count = max(2, int(getattr(self.args, "pbo_count", 3) or 3))
            self.pbos = [int(x) for x in GL.glGenBuffers(pbo_count)]
            for pbo in self.pbos:
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
                GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, self.out_w * self.out_h * 4, None, GL.GL_STREAM_READ)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
            self.pbo_index = 0
            self.pbo_ready_count = 0
            self.pbo_ready = False
        except Exception as exc:
            self.pbos = []
            print(f"[global_bev][WARN] PBO readback unavailable, fallback glReadPixels: {exc}", flush=True)

    def _ensure_texture(self, cam: str, width: int, height: int) -> int:
        tex = int(self.textures.get(cam, 0) or 0)
        if tex <= 0:
            tex = int(GL.glGenTextures(1))
            self.textures[cam] = tex
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        shape = (int(width), int(height))
        if self.tex_shape.get(cam) != shape:
            self.tex_shape[cam] = shape
            self.last_uploaded.pop(cam, None)
        return tex

    def _choose_frames(self) -> Tuple[int, Dict[str, Dict[str, Any]]]:
        snapshots: Dict[str, Dict[str, Any]] = {}
        latest_ids: List[int] = []
        latest_by_cam: Dict[str, int] = {}
        valid_count_by_cam: Dict[str, int] = {}
        for cam, rd in list(self.readers.items()):
            snap = rd.read_slots_snapshot()
            if snap is None:
                continue
            snapshots[cam] = snap
            valid = [s for s in snap["slots"] if int(s.get("frame_id", -1)) >= 0]
            valid_count_by_cam[cam] = int(len(valid))
            if valid:
                key = "tick_id" if self.args.sync_mode == "nearest_trigger" else "frame_id"
                latest_id = max(int(s.get(key, -1)) for s in valid)
                latest_ids.append(latest_id)
                latest_by_cam[cam] = int(latest_id)
        if not latest_ids:
            self.mvs_sync_diag.update({
                "target_sync_id": -1,
                "latest_id_by_cam": dict(latest_by_cam),
                "valid_slot_count_by_cam": dict(valid_count_by_cam),
                "raw_selected_id_by_cam": {},
                "selected_id_by_cam": {},
                "raw_sync_span_ticks": -1,
                "selected_sync_span_ticks": -1,
                "mixed_frame_detected": False,
                "hidden_by_mvs_sync_delta_cams": [],
                "per_camera": {},
                "skip_reason": "no_latest_ids",
            })
            return -1, {}
        latest_common = min(latest_ids)
        selected: Dict[str, Dict[str, Any]] = {}
        key = "tick_id" if self.args.sync_mode == "nearest_trigger" else "frame_id"
        max_delta = max(0, int(getattr(self.args, "max_mvs_sync_delta_ticks", 2) or 2))
        policy = str(getattr(self.args, "mvs_sync_outlier_policy", "hide") or "hide").strip().lower()
        if policy not in ("hide", "warn"):
            policy = "hide"
        tick_ms_est = max(0.001, float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0) or 25.0))
        sync_tolerance_ms = max(0.0, float(getattr(self.args, "sync_tolerance_ms", 0.0) or 0.0))
        presentation_enable = bool(getattr(self.args, "presentation_delay_enable", False))
        presentation_delay_ms = max(0.0, float(getattr(self.args, "presentation_delay_ms", 0.0) or 0.0))
        delay_ticks = int(round(presentation_delay_ms / tick_ms_est)) if presentation_enable and presentation_delay_ms > 0.0 else 0
        target = int(latest_common) - max(0, int(delay_ticks))
        if target < 0:
            target = int(latest_common)
            delay_ticks = 0
        self.current_target_sync_id = int(target)
        raw_selected_ids: Dict[str, int] = {}
        raw_delta_by_cam: Dict[str, int] = {}
        selected_ids: Dict[str, int] = {}
        hidden_cams: List[str] = []
        per_cam_diag: Dict[str, Dict[str, Any]] = {}
        for cam, snap in snapshots.items():
            valid = [s for s in snap["slots"] if int(s.get(key, -1)) >= 0 and int(s.get("frame_id", -1)) >= 0]
            if not valid:
                per_cam_diag[cam] = {
                    "valid_slot_count": 0,
                    "selected": False,
                    "reason": "no_valid_slots",
                }
                continue
            best = min(valid, key=lambda s: abs(int(s.get(key, -1)) - target))
            best_id = int(best.get(key, -1))
            delta = int(best_id) - int(target)
            raw_selected_ids[cam] = int(best_id)
            raw_delta_by_cam[cam] = int(delta)
            delta_ms_est = float(delta) * float(tick_ms_est)
            outlier_by_ticks = bool(self.args.sync_mode == "nearest_trigger" and abs(delta) > max_delta)
            outlier_by_ms = bool(self.args.sync_mode == "nearest_trigger" and sync_tolerance_ms > 0.0 and abs(delta_ms_est) > sync_tolerance_ms)
            is_outlier = bool(outlier_by_ticks or outlier_by_ms)
            reason = "selected"
            if is_outlier and policy == "hide":
                reason = "hidden_by_mvs_sync_delta_ms" if outlier_by_ms else "hidden_by_mvs_sync_delta"
            per_cam_diag[cam] = {
                "latest_id": int(latest_by_cam.get(cam, -1)),
                "selected_id": int(best_id),
                "sync_delta": int(delta),
                "sync_delta_ms_est": round(float(delta_ms_est), 3),
                "valid_slot_count": int(len(valid)),
                "slot_count": int(snap.get("slot_count", 0) or 0),
                "header_bytes": int(snap.get("header_bytes", 0) or 0),
                "outlier": bool(is_outlier),
                "outlier_by_ticks": bool(outlier_by_ticks),
                "outlier_by_ms": bool(outlier_by_ms),
                "selected": bool(not is_outlier or policy == "warn"),
                "reason": reason,
            }
            if is_outlier and policy == "hide":
                hidden_cams.append(cam)
                self.render_items.pop(cam, None)
                self.render_item_mono.pop(cam, None)
                continue
            selected[cam] = dict(best)
            selected[cam]["width"] = int(snap["width"])
            selected[cam]["height"] = int(snap["height"])
            selected[cam]["stride"] = int(snap["stride"])
            selected[cam]["pixel_format"] = int(snap["pixel_format"])
            selected[cam]["bayer_pattern"] = str(snap["bayer_pattern"])
            selected[cam]["sync_delta"] = int(delta)
            selected[cam]["hidden_by_mvs_sync_delta"] = False
            selected[cam]["snapshot_write_seq"] = int(snap.get("write_seq", -1))
            selected_ids[cam] = int(best_id)
        raw_values = list(raw_selected_ids.values())
        selected_values = list(selected_ids.values())
        raw_span = (max(raw_values) - min(raw_values)) if raw_values else -1
        selected_span = (max(selected_values) - min(selected_values)) if selected_values else -1
        self.mvs_sync_diag.update({
            "sync_mode": str(self.args.sync_mode),
            "target_sync_id": int(target),
            "latest_common_sync_id": int(latest_common),
            "key": str(key),
            "max_delta_ticks": int(max_delta),
            "sync_tolerance_ms": float(sync_tolerance_ms),
            "outlier_policy": str(policy),
            "tick_ms_est": float(tick_ms_est),
            "latest_id_by_cam": dict(latest_by_cam),
            "valid_slot_count_by_cam": dict(valid_count_by_cam),
            "raw_selected_id_by_cam": dict(raw_selected_ids),
            "raw_sync_delta_by_cam": dict(raw_delta_by_cam),
            "selected_id_by_cam": dict(selected_ids),
            "raw_sync_span_ticks": int(raw_span),
            "raw_sync_span_ms_est": round(float(raw_span) * float(tick_ms_est), 3) if raw_span >= 0 else -1.0,
            "selected_sync_span_ticks": int(selected_span),
            "selected_sync_span_ms_est": round(float(selected_span) * float(tick_ms_est), 3) if selected_span >= 0 else -1.0,
            "mixed_frame_detected": bool(raw_span > max_delta),
            "hidden_by_mvs_sync_delta_cams": list(hidden_cams),
            "hidden_by_mvs_sync_delta_count": int(len(hidden_cams)),
            "selected_cam_count": int(len(selected_ids)),
            "snapshot_cam_count": int(len(snapshots)),
            "per_camera": dict(per_cam_diag),
            "skip_reason": "ok" if selected else ("all_hidden_by_mvs_sync_delta" if hidden_cams else "no_selected"),
        })
        self.presentation_diag.update({
            "enabled": bool(presentation_enable),
            "delay_ms": float(presentation_delay_ms),
            "delay_ticks": int(delay_ticks),
            "tick_ms_est": float(tick_ms_est),
            "requested_delay_ms": round(float(delay_ticks) * float(tick_ms_est), 3),
            "latest_common_sync_id": int(latest_common),
            "target_sync_id": int(target),
            "target_lag_ticks": int(latest_common) - int(target),
            "effective_lag_ms_est": round(float(int(latest_common) - int(target)) * float(tick_ms_est), 3),
            "sync_tolerance_ms": float(sync_tolerance_ms),
            "scope": "global_bev_only",
        })
        return int(target), selected

    def _current_render_items(self) -> Dict[str, Dict[str, Any]]:
        now = time.monotonic()
        hold_s = max(0.0, float(getattr(self.args, "frame_hold_ms", 250.0)) / 1000.0)
        out: Dict[str, Dict[str, Any]] = {}
        held = 0
        expired = 0
        for cam, item in list(self.render_items.items()):
            tex = int(self.textures.get(cam, 0) or 0)
            age_s = now - float(self.render_item_mono.get(cam, 0.0))
            if tex > 0 and age_s <= hold_s:
                out[cam] = dict(item)
                out[cam]["hold_age_ms"] = float(age_s * 1000.0)
                if age_s > 0.050:
                    held += 1
            else:
                expired += 1
        self.stability_diag.update({
            "render_cams": sorted(out.keys()),
            "held_cams": int(held),
            "expired_cams": int(expired),
            "frame_hold_ms": float(getattr(self.args, "frame_hold_ms", 250.0)),
        })
        return out

    def _update_turf_stats(self, cam: str, img: np.ndarray, pattern: str) -> None:
        if not (bool(getattr(self.args, "turf_match_enable", False)) or bool(getattr(self.args, "turf_auto_match", False))):
            return
        rgb, count = _estimate_turf_rgb_from_bayer(img, pattern)
        self.turf_rgb_counts[cam] = int(count)
        if rgb is None:
            return
        alpha = float(np.clip(_safe_float(getattr(self.args, "turf_stat_alpha", 0.05), 0.05), 0.001, 1.0))
        prev = self.turf_rgb_stats.get(cam)
        self.turf_rgb_stats[cam] = rgb if prev is None else (prev * (1.0 - alpha) + rgb * alpha)

    def _update_turf_gains(self) -> None:
        target = np.asarray([
            float(getattr(self.args, "turf_target_r", self.turf_target_rgb[0])),
            float(getattr(self.args, "turf_target_g", self.turf_target_rgb[1])),
            float(getattr(self.args, "turf_target_b", self.turf_target_rgb[2])),
        ], dtype=np.float32)
        valid = [v for v in self.turf_rgb_stats.values() if np.all(np.isfinite(v)) and float(np.max(v)) > 1e-4]
        if bool(getattr(self.args, "turf_auto_match", False)) and len(valid) >= 2:
            measured_target = np.median(np.stack(valid, axis=0), axis=0).astype(np.float32)
            alpha = float(np.clip(_safe_float(getattr(self.args, "turf_target_alpha", 0.02), 0.02), 0.001, 1.0))
            self.turf_target_rgb = self.turf_target_rgb * (1.0 - alpha) + measured_target * alpha
            target = self.turf_target_rgb
        else:
            self.turf_target_rgb = target
        gmin = max(0.05, float(getattr(self.args, "turf_gain_min", 0.72)))
        gmax = max(gmin, float(getattr(self.args, "turf_gain_max", 1.35)))
        alpha = float(np.clip(_safe_float(getattr(self.args, "turf_gain_alpha", 0.06), 0.06), 0.001, 1.0))
        for cam in self.cams:
            stat = self.turf_rgb_stats.get(cam)
            if stat is None:
                self.cam_rgb_gain.setdefault(cam, np.ones((3,), dtype=np.float32))
                continue
            raw_gain = np.clip(target / np.maximum(stat, 0.015), gmin, gmax).astype(np.float32)
            prev = self.cam_rgb_gain.get(cam, np.ones((3,), dtype=np.float32))
            self.cam_rgb_gain[cam] = prev * (1.0 - alpha) + raw_gain * alpha
        self.turf_diag = {
            "enable": bool(getattr(self.args, "turf_match_enable", False)),
            "auto_match": bool(getattr(self.args, "turf_auto_match", False)),
            "match_strength": float(getattr(self.args, "turf_match_strength", 0.0)),
            "target_rgb": [float(x) for x in self.turf_target_rgb],
            "stats_rgb": {cam: [float(x) for x in v] for cam, v in self.turf_rgb_stats.items()},
            "sample_counts": {cam: int(v) for cam, v in self.turf_rgb_counts.items()},
            "cam_rgb_gain": {cam: [float(x) for x in self.cam_rgb_gain.get(cam, np.ones((3,), dtype=np.float32))] for cam in self.cams},
        }

    def _texture_format_pair(self) -> Tuple[int, int, str]:
        fmt_name = str(getattr(self.args, "texture_format", "r8") or "r8").lower()
        if fmt_name in ("luminance", "legacy", "gl_luminance"):
            return int(GL.GL_LUMINANCE), int(GL.GL_LUMINANCE), "GL_LUMINANCE"
        red = int(getattr(GL, "GL_RED", 0x1903))
        r8 = int(getattr(GL, "GL_R8", 0x8229))
        return r8, red, "GL_R8/GL_RED"

    def _drain_gl_error(self) -> int:
        last = int(GL.GL_NO_ERROR)
        for _ in range(16):
            err = int(GL.glGetError())
            if err == int(GL.GL_NO_ERROR):
                break
            last = err
        return int(last)

    def _upload_raw_texture(self, cam: str, tex: int, img: np.ndarray, width: int, height: int, first_upload: bool) -> Tuple[str, int]:
        internal, upload_format, fmt_name = self._texture_format_pair()
        ptr = ctypes.c_void_p(int(img.ctypes.data))
        self._drain_gl_error()
        try:
            if first_upload:
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, upload_format, GL.GL_UNSIGNED_BYTE, ptr)
            else:
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height, upload_format, GL.GL_UNSIGNED_BYTE, ptr)
            err = self._drain_gl_error()
            if err == int(GL.GL_NO_ERROR):
                return fmt_name, 0
            raise RuntimeError(f"gl error after {fmt_name} upload: 0x{err:x}")
        except Exception as exc:
            if fmt_name == "GL_LUMINANCE":
                raise
            print(f"[global_bev][{cam}][WARN] {fmt_name} upload failed, fallback GL_LUMINANCE: {exc}", flush=True)
            self._drain_gl_error()
            if first_upload:
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_LUMINANCE, width, height, 0, GL.GL_LUMINANCE, GL.GL_UNSIGNED_BYTE, ptr)
            else:
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height, GL.GL_LUMINANCE, GL.GL_UNSIGNED_BYTE, ptr)
            return "GL_LUMINANCE_FALLBACK", self._drain_gl_error()

    def _mvs_upload_key(self, item: Dict[str, Any], width: int, height: int) -> Tuple[int, int, int, int, int, int]:
        return (
            _safe_int(item.get("tick_id"), -1),
            _safe_int(item.get("frame_id"), -1),
            _safe_int(item.get("slot"), -1),
            int(width),
            int(height),
            _safe_int(item.get("pixel_format"), RAW_PIXEL_BAYER8),
        )

    def _upload_selected(self, selected: Dict[str, Dict[str, Any]]) -> None:
        start = time.perf_counter()
        budget_enable = bool(getattr(self.args, "mvs_upload_budget_enable", True))
        budget_ms = max(0.0, float(getattr(self.args, "mvs_upload_budget_ms", 10.0) or 0.0))
        unstable_skips = 0
        uploaded_cams: List[str] = []
        duplicate_skip_cams: List[str] = []
        held_by_budget_cams: List[str] = []
        forced_upload_cams: List[str] = []
        unstable_skip_cams: List[str] = []
        unsupported_pixel_format_cams: List[str] = []
        upload_order = self._budget_order(selected.keys(), "_mvs_upload_budget_cursor")
        for cam in upload_order:
            item = selected.get(cam)
            if not isinstance(item, dict):
                continue
            pixel_format = _safe_int(item.get("pixel_format"), RAW_PIXEL_BAYER8)
            if int(pixel_format) != RAW_PIXEL_BAYER8:
                unsupported_pixel_format_cams.append(str(cam))
                continue
            fid = _safe_int(item.get("frame_id"), -1)
            slot = _safe_int(item.get("slot"), -1)
            width = _safe_int(item.get("width"), 0)
            height = _safe_int(item.get("height"), 0)
            frame = np.asarray(item["frame"])
            if width <= 0:
                width = int(frame.shape[1])
            if height <= 0:
                height = int(frame.shape[0])
            upload_key = self._mvs_upload_key(item, width, height)
            if self.last_uploaded.get(cam) == upload_key:
                if cam in self.textures:
                    self.render_items[cam] = dict(item)
                    self.render_item_mono[cam] = time.monotonic()
                duplicate_skip_cams.append(str(cam))
                self.texture_diag[cam] = {
                    "ok": True,
                    "skipped": "duplicate_upload_key",
                    "upload_key": list(upload_key),
                    "tick_id": int(upload_key[0]),
                    "frame_id": int(fid),
                    "slot": int(slot),
                    "width": int(width),
                    "height": int(height),
                    "pixel_format": int(pixel_format),
                }
                continue
            has_previous_texture = bool(int(self.textures.get(cam, 0) or 0) > 0 and cam in self.render_items)
            elapsed_before_ms = (time.perf_counter() - start) * 1000.0
            if budget_enable and budget_ms > 0.0 and elapsed_before_ms >= budget_ms and has_previous_texture:
                held_by_budget_cams.append(str(cam))
                hold_age_ms = max(0.0, (time.monotonic() - float(self.render_item_mono.get(cam, 0.0))) * 1000.0)
                self.texture_diag[cam] = {
                    "ok": True,
                    "skipped": "held_by_upload_budget",
                    "upload_key": list(upload_key),
                    "tick_id": int(upload_key[0]),
                    "frame_id": int(fid),
                    "slot": int(slot),
                    "width": int(width),
                    "height": int(height),
                    "pixel_format": int(pixel_format),
                    "elapsed_before_ms": round(float(elapsed_before_ms), 3),
                    "budget_ms": round(float(budget_ms), 3),
                    "hold_age_ms": round(float(hold_age_ms), 3),
                }
                continue
            if not has_previous_texture and budget_enable and budget_ms > 0.0 and elapsed_before_ms >= budget_ms:
                forced_upload_cams.append(str(cam))
            rd = self.readers.get(cam)
            copied = None
            if rd is not None and hasattr(rd, "copy_slot_frame"):
                copied = rd.copy_slot_frame(slot, width=width, height=height, max_retries=int(getattr(self.args, "unstable_copy_retries", 3)))
            if copied is None:
                unstable_skips += 1
                unstable_skip_cams.append(str(cam))
                self.texture_diag[cam] = {
                    "ok": False,
                    "skipped": "unstable_raw_slot_copy",
                    "slot": int(slot),
                    "frame_id": int(fid),
                    "tick_id": int(upload_key[0]),
                    "upload_key": list(upload_key),
                    "snapshot_write_seq": int(item.get("snapshot_write_seq", -1)),
                }
                continue
            img, stable_seq = copied
            tex = self._ensure_texture(cam, width, height)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            first_upload = self.last_uploaded.get(cam) is None
            ok = True
            upload_format = ""
            gl_error = 0
            try:
                upload_format, gl_error = self._upload_raw_texture(cam, tex, img, width, height, first_upload)
                ok = gl_error == int(GL.GL_NO_ERROR)
            except Exception as exc:
                ok = False
                upload_format = "FAILED"
                gl_error = self._drain_gl_error()
                self._last_error = f"{cam} texture upload failed: {type(exc).__name__}: {exc}"
                print(f"[global_bev][{cam}][WARN] texture upload failed: {self._last_error}", flush=True)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            self.texture_diag[cam] = {
                "ok": bool(ok),
                "format": upload_format,
                "gl_error": int(gl_error),
                "width": int(width),
                "height": int(height),
                "slot": int(slot),
                "frame_id": int(fid),
                "tick_id": int(upload_key[0]),
                "pixel_format": int(pixel_format),
                "upload_key": list(upload_key),
                "stable_write_seq": int(stable_seq),
                "min": int(np.min(img)) if img.size else 0,
                "max": int(np.max(img)) if img.size else 0,
                "mean": float(np.mean(img)) if img.size else 0.0,
            }
            if ok:
                uploaded_cams.append(str(cam))
                self.last_uploaded[cam] = upload_key
                self.render_items[cam] = dict(item)
                self.render_item_mono[cam] = time.monotonic()
                turf_fps = max(0.0, float(getattr(self.args, "turf_stat_fps", 5.0) or 0.0))
                now_mono = time.monotonic()
                last_turf = float(self._turf_last_sample_mono.get(cam, 0.0))
                if turf_fps <= 0.0 or now_mono - last_turf >= 1.0 / max(1e-6, turf_fps):
                    self._turf_last_sample_mono[cam] = now_mono
                    self._update_turf_stats(cam, img, str(item.get("bayer_pattern", "BG")))
                    self.texture_diag[cam]["turf_stats_sampled"] = True
                else:
                    self.texture_diag[cam]["turf_stats_sampled"] = False
        if unstable_skips:
            prev = int(self.stability_diag.get("unstable_copy_skips", 0))
            self.stability_diag["unstable_copy_skips"] = prev + int(unstable_skips)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        budget_stats = self._update_budget_diag("mvs_upload", elapsed_ms, budget_ms if budget_enable else 0.0, len(held_by_budget_cams))
        self.mvs_upload_diag = {
            "budget_enable": bool(budget_enable),
            "budget_ms": round(float(budget_ms), 3),
            "elapsed_ms": round(float(elapsed_ms), 3),
            "upload_order": list(upload_order),
            "budget_overrun": bool(budget_stats.get("overrun", False)),
            "budget_overrun_rate": float(budget_stats.get("overrun_rate", 0.0)),
            "budget_overrun_count": int(budget_stats.get("overrun_count", 0)),
            "budget_sample_count": int(budget_stats.get("sample_count", 0)),
            "elapsed_p95_ms": float(budget_stats.get("elapsed_p95_ms", 0.0)),
            "held_cam_count_p95": float(budget_stats.get("held_cam_count_p95", 0.0)),
            "uploaded_cams": list(uploaded_cams),
            "uploaded_cam_count": int(len(uploaded_cams)),
            "duplicate_skip_cams": list(duplicate_skip_cams),
            "duplicate_skip_cam_count": int(len(duplicate_skip_cams)),
            "held_by_budget_cams": list(held_by_budget_cams),
            "held_by_budget_cam_count": int(len(held_by_budget_cams)),
            "forced_upload_cams": list(forced_upload_cams),
            "forced_upload_cam_count": int(len(forced_upload_cams)),
            "unstable_skip_cams": list(unstable_skip_cams),
            "unstable_skip_cam_count": int(len(unstable_skip_cams)),
            "unsupported_pixel_format_cams": list(unsupported_pixel_format_cams),
            "unsupported_pixel_format_cam_count": int(len(unsupported_pixel_format_cams)),
        }
        self._update_turf_gains()

    def _open_event_layer_inputs(self) -> None:
        if not bool(getattr(self.args, "event_layer_enable", False)):
            self.event_layer_diag.update({"enabled": False, "skip_reason": "disabled"})
            return
        per_cam = self.event_layer_diag.setdefault("per_camera", {})
        if not isinstance(per_cam, dict):
            per_cam = {}
            self.event_layer_diag["per_camera"] = per_cam
        for cam in self.cams:
            if cam in self.event_layer_readers:
                continue
            try:
                rd = EventOverlayLayerReader(
                    cam,
                    str(getattr(self.args, "event_layer_template", "event_overlay_latest_{camera}") or "event_overlay_latest_{camera}"),
                    str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(self.args, "event_layer_sidecar_template", "event_overlay_{camera}_latest.json") or "event_overlay_{camera}_latest.json"),
                    str(getattr(self.args, "event_layer_history_sidecar_template", "") or "") if bool(getattr(self.args, "event_layer_history_enable", False)) else "",
                    float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
                )
                self.event_layer_readers[cam] = rd
                per_cam[cam] = {
                    "open": True,
                    "path": str(rd.path),
                    "sidecar_path": str(rd.sidecar_path),
                    "width": int(rd.width),
                    "height": int(rd.height),
                    "channels": int(rd.channels),
                    "error": "",
                }
            except Exception as exc:
                per_cam[cam] = {
                    "open": False,
                    "path": str(getattr(self.args, "event_layer_template", "event_overlay_latest_{camera}")).format(camera=cam, cam=cam, id=cam),
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def _event_layer_backend(self) -> str:
        backend = str(getattr(self.args, "event_layer_backend", "texture") or "texture").strip().lower().replace("-", "_")
        if backend in {"sparse", "sparse_points", "opengl_sparse", "opengl_sparse_points"}:
            return "sparse_gl"
        if backend not in {"texture", "sparse_gl"}:
            return "texture"
        return backend

    def _open_event_layer_sparse_inputs(self) -> None:
        if not bool(getattr(self.args, "event_layer_enable", False)):
            self.event_layer_diag.update({"enabled": False, "skip_reason": "disabled"})
            return
        per_cam = self.event_layer_diag.setdefault("per_camera", {})
        if not isinstance(per_cam, dict):
            per_cam = {}
            self.event_layer_diag["per_camera"] = per_cam
        for cam in self.cams:
            if cam in self.event_layer_sparse_readers:
                continue
            try:
                rd = SparseEventOverlayReader(
                    cam,
                    str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(self.args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
                    float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
                )
                self.event_layer_sparse_readers[cam] = rd
                per_cam[cam] = {
                    "open": True,
                    "backend": "sparse_gl",
                    "sidecar_path": str(rd.sidecar_path),
                    "error": "",
                }
            except Exception as exc:
                per_cam[cam] = {
                    "open": False,
                    "backend": "sparse_gl",
                    "sidecar_path": str(getattr(self.args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json")).format(camera=cam, cam=cam, id=cam),
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def _field_xy_to_ndc_batch(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fw = max(1e-6, float(self.args.field_width_m))
        fh = max(1e-6, float(self.args.field_height_m))
        sx = (x.astype(np.float64) - float(self.args.field_origin_x_m)) / fw
        sy = (y.astype(np.float64) - float(self.args.field_origin_y_m)) / fh
        valid = np.isfinite(sx) & np.isfinite(sy) & (sx >= -0.05) & (sx <= 1.05) & (sy >= -0.05) & (sy <= 1.05)
        m = _normalize_transform_mode(getattr(self.args, "field_transform_mode", "normal"))
        qx = sx.copy()
        qy = sy.copy()
        if m == "flip_x":
            qx, qy = 1.0 - sx, sy
        elif m == "flip_y":
            qx, qy = sx, 1.0 - sy
        elif m == "flip_xy":
            qx, qy = 1.0 - sx, 1.0 - sy
        elif m == "swap_xy":
            qx, qy = sy, sx
        elif m == "swap_xy_flip_x":
            qx, qy = sy, 1.0 - sx
        elif m == "swap_xy_flip_y":
            qx, qy = 1.0 - sy, sx
        elif m == "swap_xy_flip_xy":
            qx, qy = 1.0 - sy, 1.0 - sx
        ndc_x = qx * 2.0 - 1.0
        # Match the shader display_to_sample_field_xy() mapping: screen/FBO uv.y
        # is inverted before converting back to field y.
        ndc_y = (1.0 - qy) * 2.0 - 1.0
        valid &= np.isfinite(ndc_x) & np.isfinite(ndc_y) & (ndc_x >= -1.1) & (ndc_x <= 1.1) & (ndc_y >= -1.1) & (ndc_y <= 1.1)
        return ndc_x.astype(np.float32), ndc_y.astype(np.float32), valid

    def _project_sparse_points_to_ndc(self, cam: str, points: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "input_points": int(len(points)) if hasattr(points, "__len__") else 0,
            "projected_points": 0,
            "visible_points": 0,
            "skip_reason": "",
        }
        if points is None or len(points) <= 0:
            meta["skip_reason"] = "no_points"
            return np.empty((0, 2), dtype=np.float32), meta
        h_event_to_mvs = self.event_layer_event_to_mvs.get(cam)
        h_img_to_field = self._event_bullet_img_to_field(cam)
        if h_event_to_mvs is None or h_img_to_field is None:
            meta["skip_reason"] = "missing_homography"
            return np.empty((0, 2), dtype=np.float32), meta
        max_draw = max(1, int(getattr(self.args, "event_layer_sparse_max_points_draw", 20000) or 20000))
        pts = points[:max_draw]
        xs = pts["x"].astype(np.float64)
        ys = pts["y"].astype(np.float64)
        ones = np.ones_like(xs, dtype=np.float64)
        ev = np.vstack([xs, ys, ones])
        try:
            mvs = np.asarray(h_event_to_mvs, dtype=np.float64).reshape(3, 3) @ ev
            z = mvs[2, :]
            z = np.where(np.abs(z) < 1e-9, np.nan, z)
            mvs_x = mvs[0, :] / z
            mvs_y = mvs[1, :] / z
            mvs_h = np.vstack([mvs_x, mvs_y, ones])
            fld = np.asarray(h_img_to_field, dtype=np.float64).reshape(3, 3) @ mvs_h
            fz = fld[2, :]
            fz = np.where(np.abs(fz) < 1e-9, np.nan, fz)
            fx = fld[0, :] / fz
            fy = fld[1, :] / fz
        except Exception as exc:
            meta["skip_reason"] = f"project_error:{type(exc).__name__}"
            return np.empty((0, 2), dtype=np.float32), meta
        meta["projected_points"] = int(len(fx))
        ndc_x, ndc_y, valid = self._field_xy_to_ndc_batch(fx, fy)
        if not np.any(valid):
            meta["skip_reason"] = "all_projected_out_of_field"
            return np.empty((0, 2), dtype=np.float32), meta
        vertices = np.column_stack([ndc_x[valid], ndc_y[valid]]).astype(np.float32, copy=False)
        meta["visible_points"] = int(vertices.shape[0])
        meta["truncated_by_draw_limit"] = bool(len(points) > max_draw)
        meta["max_points_draw"] = int(max_draw)
        meta["skip_reason"] = "ok"
        return vertices, meta

    def _event_layer_color_rgba(self, cam: str, alpha_scale: float) -> Tuple[float, float, float, float]:
        mode = _event_layer_color_mode(getattr(self.args, "event_layer_color_mode", "camera"))
        if mode == "red":
            rgb = (1.0, 0.08, 0.04)
        elif mode == "yellow":
            rgb = (1.0, 0.86, 0.08)
        else:
            palette = {
                "A": (1.0, 0.15, 0.10),
                "B": (1.0, 0.65, 0.10),
                "C": (0.25, 0.95, 0.25),
                "D": (0.10, 0.80, 1.0),
                "E": (0.30, 0.35, 1.0),
                "F": (0.85, 0.30, 1.0),
                "G": (1.0, 0.25, 0.65),
                "H": (0.80, 0.80, 0.80),
            }
            rgb = palette.get(str(cam), (1.0, 0.86, 0.08))
        a = _clip01(float(getattr(self.args, "event_layer_alpha", 0.45) or 0.0) * float(alpha_scale))
        return float(rgb[0]), float(rgb[1]), float(rgb[2]), float(a)

    def _event_layer_presentation_mode(self) -> str:
        if not bool(getattr(self.args, "event_layer_presentation_worker_enable", False)):
            return "off"
        mode = str(getattr(self.args, "event_layer_presentation_worker_mode", "shadow") or "shadow").strip().lower().replace("-", "_")
        if mode not in {"off", "shadow", "active"}:
            return "shadow"
        return mode

    def _ensure_event_layer_presentation_worker(self) -> None:
        mode = self._event_layer_presentation_mode()
        enabled = bool(mode in {"shadow", "active"} and self._event_layer_backend() == "sparse_gl")
        diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
        diag.update({
            "enabled": bool(enabled),
            "mode": mode,
            "cpu_prep_only": True,
            "opengl_thread": "gui",
            "fallback_to_gui_direct": True,
            "active_supported": True,
            "bundle_ring_size": int(getattr(self.args, "event_layer_presentation_bundle_ring_size", 8) or 8),
            "prebuild_lookback_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookback_ticks", 2) or 2),
            "prebuild_lookahead_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookahead_ticks", 2) or 2),
            "prebuild_max_per_cycle": int(getattr(self.args, "event_layer_presentation_prebuild_max_per_cycle", 1) or 1),
            "bundle_cache_reuse_ms": float(getattr(self.args, "event_layer_presentation_bundle_cache_reuse_ms", 16.0) or 16.0),
        })
        if not enabled:
            if self.event_layer_presentation_worker is not None:
                try:
                    self.event_layer_presentation_worker.stop(timeout_s=1.0)
                except Exception:
                    pass
                self.event_layer_presentation_worker = None
            diag.update({
                "state": "disabled" if mode == "off" else "backend_not_sparse_gl",
                "per_camera": {},
            })
            self.event_layer_diag["presentation_worker"] = diag
            return
        if self.event_layer_presentation_worker is None:
            try:
                self.event_layer_presentation_worker = EventLayerPresentationWorker(
                    self.cams,
                    str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
                    str(getattr(self.args, "event_layer_sparse_sidecar_template", "event_overlay_sparse_{camera}_history.json") or "event_overlay_sparse_{camera}_history.json"),
                    str(getattr(self.args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
                    float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
                    float(getattr(self.args, "event_layer_presentation_worker_interval_ms", 33.333) or 33.333),
                )
                self.event_layer_presentation_worker.start()
                diag["state"] = "started"
            except Exception as exc:
                self.event_layer_presentation_worker = None
                diag.update({
                    "enabled": False,
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "per_camera": {},
                })
        else:
            diag["state"] = str(diag.get("state") or "running")
            try:
                self.event_layer_presentation_worker.interval_s = max(
                    0.001,
                    float(getattr(self.args, "event_layer_presentation_worker_interval_ms", 33.333) or 33.333) / 1000.0,
                )
            except Exception:
                pass
        self.event_layer_diag["presentation_worker"] = diag

    def _event_layer_presentation_worker_config(self) -> Dict[str, Any]:
        event_to_mvs: Dict[str, np.ndarray] = {}
        img_to_field: Dict[str, np.ndarray] = {}
        for cam in self.cams:
            h_event = self.event_layer_event_to_mvs.get(cam)
            h_field = self._event_bullet_img_to_field(cam)
            if h_event is not None:
                event_to_mvs[str(cam)] = np.asarray(h_event, dtype=np.float64).reshape(3, 3).copy()
            if h_field is not None:
                img_to_field[str(cam)] = np.asarray(h_field, dtype=np.float64).reshape(3, 3).copy()
        return {
            "mode": self._event_layer_presentation_mode(),
            "read_points": bool(self._event_layer_presentation_mode() == "active"),
            "interval_ms": float(getattr(self.args, "event_layer_presentation_worker_interval_ms", 33.333) or 33.333),
            "bundle_ring_size": int(getattr(self.args, "event_layer_presentation_bundle_ring_size", 8) or 8),
            "prebuild_lookback_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookback_ticks", 2) or 2),
            "prebuild_lookahead_ticks": int(getattr(self.args, "event_layer_presentation_prebuild_lookahead_ticks", 2) or 2),
            "prebuild_max_per_cycle": int(getattr(self.args, "event_layer_presentation_prebuild_max_per_cycle", 1) or 1),
            "bundle_cache_reuse_ms": float(getattr(self.args, "event_layer_presentation_bundle_cache_reuse_ms", 16.0) or 16.0),
            "mvs_sync_tick_ms_est": float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0) or 25.0),
            "max_sync_delta_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 50.0) or 50.0),
            "max_sync_delta_ticks": int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2) or 2),
            "sync_normal_ms": float(getattr(self.args, "event_layer_sync_normal_ms", 50.0) or 50.0),
            "sync_warn_ms": float(getattr(self.args, "event_layer_sync_warn_ms", 80.0) or 80.0),
            "sync_hard_ms": float(getattr(self.args, "event_layer_sync_hard_ms", 80.0) or 80.0),
            "max_age_ms": float(getattr(self.args, "event_layer_max_age_ms", 2500.0) or 2500.0),
            "fade_alpha_scale": float(getattr(self.args, "event_layer_fade_alpha_scale", 0.25) or 0.25),
            "hold_last_good_on_hard_outlier": bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True)),
            "hold_last_good_ms": float(getattr(self.args, "event_layer_hold_last_good_ms", 250.0) or 250.0),
            "candidate_scan_initial": int(getattr(self.args, "event_layer_sparse_candidate_scan_initial", 4) or 4),
            "candidate_scan_escalated": int(getattr(self.args, "event_layer_sparse_candidate_scan_escalated", 32) or 32),
            "candidate_escalate_enable": bool(getattr(self.args, "event_layer_sparse_candidate_escalate_enable", True)),
            "candidate_escalate_abs_ms": float(getattr(self.args, "event_layer_sparse_candidate_escalate_abs_ms", 10.0) or 10.0),
            "candidate_escalate_ratio": float(getattr(self.args, "event_layer_sparse_candidate_escalate_ratio", 0.60) or 0.60),
            "field_width_m": float(getattr(self.args, "field_width_m", 33.0) or 33.0),
            "field_height_m": float(getattr(self.args, "field_height_m", 22.0) or 22.0),
            "field_origin_x_m": float(getattr(self.args, "field_origin_x_m", 0.0) or 0.0),
            "field_origin_y_m": float(getattr(self.args, "field_origin_y_m", 0.0) or 0.0),
            "field_transform_mode": str(getattr(self.args, "field_transform_mode", "normal") or "normal"),
            "max_points_draw": int(getattr(self.args, "event_layer_sparse_max_points_draw", 20000) or 20000),
            "event_to_mvs_by_cam": event_to_mvs,
            "img_to_field_by_cam": img_to_field,
        }

    def _event_layer_presentation_submit(self, target_sync: int) -> None:
        self._ensure_event_layer_presentation_worker()
        worker = getattr(self, "event_layer_presentation_worker", None)
        mode = self._event_layer_presentation_mode()
        if worker is None or mode == "off" or self._event_layer_backend() != "sparse_gl":
            diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
            diag.update({
                "enabled": bool(worker is not None and mode != "off"),
                "mode": mode,
                "state": "disabled" if mode == "off" else ("unavailable" if worker is None else "backend_not_sparse_gl"),
                "target_sync_id": int(target_sync),
                "cpu_prep_only": True,
                "opengl_thread": "gui",
                "fallback_to_gui_direct": True,
            })
            self.event_layer_diag["presentation_worker"] = diag
            return
        try:
            worker.submit(int(target_sync), self._event_layer_presentation_worker_config())
        except Exception as exc:
            diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
            diag.update({
                "enabled": True,
                "mode": mode,
                "state": "submit_error",
                "target_sync_id": int(target_sync),
                "error": f"{type(exc).__name__}: {exc}",
                "cpu_prep_only": True,
                "opengl_thread": "gui",
                "fallback_to_gui_direct": True,
            })
            self.event_layer_diag["presentation_worker"] = diag

    def _event_layer_presentation_compare_and_diag(self, target_sync: int, active_per_cam: Dict[str, Any]) -> None:
        worker = getattr(self, "event_layer_presentation_worker", None)
        mode = self._event_layer_presentation_mode()
        if worker is None or mode == "off":
            return
        prev_diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
        compare_tolerance = 0 if mode == "shadow" else max(0, int(getattr(self.args, "event_layer_presentation_target_tolerance_ticks", 1) or 1))
        try:
            obj = worker.snapshot(int(target_sync), int(compare_tolerance))
        except Exception as exc:
            self.event_layer_diag["presentation_worker"] = {
                "enabled": True,
                "mode": mode,
                "state": "snapshot_error",
                "target_sync_id": int(target_sync),
                "error": f"{type(exc).__name__}: {exc}",
                "cpu_prep_only": True,
                "opengl_thread": "gui",
                "fallback_to_gui_direct": True,
            }
            return
        worker_state = str(obj.get("state", "") or "")
        obj["mode"] = mode
        obj["worker_state"] = worker_state
        obj["current_target_sync_id"] = int(target_sync)
        obj["cpu_prep_only"] = True
        obj["opengl_thread"] = "gui"
        obj["fallback_to_gui_direct"] = True
        obj["active_supported"] = True
        worker_target = _safe_int(obj.get("target_sync_id"), -1)
        lookup_state = str(obj.get("bundle_lookup_state", "") or "").strip().lower()
        target_delta_ticks = int(worker_target) - int(target_sync) if worker_target >= 0 else -999999
        matching_target = bool(
            worker_target in self.event_layer_active_history
            and worker_target >= 0
            and abs(int(target_delta_ticks)) <= int(compare_tolerance)
            and lookup_state not in {"miss"}
        )
        active_compare = self.event_layer_active_history.get(worker_target, active_per_cam if worker_target == int(target_sync) else {})
        per_worker = obj.get("per_camera", {}) if isinstance(obj.get("per_camera"), dict) else {}
        compare_per_cam: Dict[str, Any] = {}
        mismatch_count = 0
        display_mismatch_count = 0
        frame_mismatch_count = 0
        draw_mismatch_count = 0
        delta_mismatch_count = 0
        for cam in self.cams:
            active = active_compare.get(cam, {}) if isinstance(active_compare, dict) else {}
            built = per_worker.get(cam, {}) if isinstance(per_worker, dict) else {}
            if not isinstance(active, dict) or not isinstance(built, dict):
                continue
            active_frame = _safe_int(active.get("frame_id", active.get("event_layer_frame_id")), -1)
            built_frame = _safe_int(built.get("frame_id", built.get("event_layer_frame_id")), -1)
            active_draw = bool(active.get("draw", False))
            built_draw = bool(built.get("draw", False))
            active_delta = _safe_float(active.get("event_to_bev_sync_delta_ms"), 999999.0)
            built_delta = _safe_float(built.get("event_to_bev_sync_delta_ms"), 999999.0)
            frame_match = bool(active_frame == built_frame) if matching_target else False
            draw_match = bool(active_draw == built_draw) if matching_target else False
            delta_diff_ms = abs(float(active_delta) - float(built_delta)) if abs(active_delta) < 900000.0 and abs(built_delta) < 900000.0 else -1.0
            delta_match = bool(delta_diff_ms >= 0.0 and delta_diff_ms <= 1.0) if matching_target else False
            if matching_target and not active_draw and not built_draw and active_frame == built_frame and delta_diff_ms < 0.0:
                delta_match = True
            display_mismatch = bool(matching_target and not (frame_match and draw_match))
            delta_mismatch = bool(matching_target and not delta_match)
            mismatch = bool(display_mismatch)
            if mismatch:
                mismatch_count += 1
                display_mismatch_count += 1
            if matching_target and not frame_match:
                frame_mismatch_count += 1
            if matching_target and not draw_match:
                draw_mismatch_count += 1
            if delta_mismatch:
                delta_mismatch_count += 1
            compare_per_cam[str(cam)] = {
                "active_frame_id": int(active_frame),
                "worker_frame_id": int(built_frame),
                "active_draw": bool(active_draw),
                "worker_draw": bool(built_draw),
                "active_delta_ms": round(float(active_delta), 3) if abs(active_delta) < 900000.0 else None,
                "worker_delta_ms": round(float(built_delta), 3) if abs(built_delta) < 900000.0 else None,
                "delta_diff_ms": round(float(delta_diff_ms), 3) if delta_diff_ms >= 0.0 else None,
                "frame_match": bool(frame_match),
                "draw_match": bool(draw_match),
                "delta_match": bool(delta_match),
                "mismatch": bool(mismatch),
                "display_mismatch": bool(display_mismatch),
                "delta_mismatch": bool(delta_mismatch),
            }
        obj["matching_target"] = bool(matching_target)
        obj["compare_lookup_state"] = str(lookup_state or "latest")
        obj["compare_target_delta_ticks"] = int(target_delta_ticks)
        obj["compare_tolerance_ticks"] = int(compare_tolerance)
        compare_exact_count = _safe_int(prev_diag.get("compare_exact_count"), 0)
        compare_near_count = _safe_int(prev_diag.get("compare_near_count"), 0)
        compare_miss_count = _safe_int(prev_diag.get("compare_miss_count"), 0)
        if lookup_state == "exact":
            compare_exact_count += 1
        elif lookup_state == "near":
            compare_near_count += 1
        elif lookup_state == "miss":
            compare_miss_count += 1
        compare_total = compare_exact_count + compare_near_count + compare_miss_count
        obj["compare_exact_count"] = int(compare_exact_count)
        obj["compare_near_count"] = int(compare_near_count)
        obj["compare_miss_count"] = int(compare_miss_count)
        obj["compare_hit_ratio"] = round(float(compare_exact_count + compare_near_count) / float(compare_total), 4) if compare_total > 0 else 0.0
        obj["compare_state"] = "match" if matching_target and mismatch_count == 0 else ("pending_target" if not matching_target else "mismatch")
        obj["display_compare_state"] = str(obj["compare_state"])
        obj["delta_compare_state"] = "match" if matching_target and delta_mismatch_count == 0 else ("pending_target" if not matching_target else "warn")
        obj["mismatch_cam_count"] = int(mismatch_count)
        obj["display_mismatch_cam_count"] = int(display_mismatch_count)
        obj["frame_mismatch_cam_count"] = int(frame_mismatch_count)
        obj["draw_mismatch_cam_count"] = int(draw_mismatch_count)
        obj["delta_mismatch_cam_count"] = int(delta_mismatch_count)
        obj["diagnostic_mismatch_cam_count"] = int(delta_mismatch_count)
        obj["compare_per_camera"] = compare_per_cam
        for key in (
            "fallback_count",
            "fallback_reason",
            "last_fallback_reason",
            "last_fallback_wall_us",
            "active_consume_count",
            "active_consume_elapsed_ms",
            "last_consume_state",
            "last_active_consume_wall_us",
            "bundle_lookup_state",
            "bundle_target_delta_ticks",
            "bundle_hit_count",
            "bundle_miss_count",
            "bundle_hit_ratio",
            "consume_exact_count",
            "consume_near_count",
            "bundle_max_age_ms",
            "target_tolerance_ticks",
        ):
            if key in prev_diag:
                obj[key] = prev_diag.get(key)
        if mode == "active":
            prev_state = str(prev_diag.get("state", "") or "")
            if prev_state in {"active_consumed", "fallback_to_gui_direct"}:
                obj["state"] = prev_state
            elif worker_state:
                obj["state"] = worker_state
        self.event_layer_diag["presentation_worker"] = obj

    def _try_consume_event_layer_presentation_bundle(
        self,
        *,
        render_context: Optional[Dict[str, Dict[str, Any]]],
        target_sync: int,
        budget_t0: float,
        effective_dilate_px: float,
        perf_guard: Dict[str, Any],
    ) -> bool:
        worker = getattr(self, "event_layer_presentation_worker", None)
        mode = self._event_layer_presentation_mode()
        if worker is None or mode != "active":
            return False
        max_age_ms = max(1.0, float(getattr(self.args, "event_layer_presentation_bundle_max_age_ms", 120.0) or 120.0))
        tolerance_ticks = max(0, int(getattr(self.args, "event_layer_presentation_target_tolerance_ticks", 1) or 1))
        bundle = worker.latest_bundle(int(target_sync), int(tolerance_ticks))
        bundle_target = _safe_int(bundle.get("target_sync_id"), -1)
        updated_us = _safe_int(bundle.get("updated_wall_us"), 0)
        age_ms = max(0.0, (_now_us() - int(updated_us)) / 1000.0) if updated_us > 1_000_000_000_000 else 999999.0
        target_delta = int(bundle_target) - int(target_sync)
        lookup_state = str(bundle.get("bundle_lookup_state", "latest") or "latest")
        prev_diag_for_lookup = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
        bundle_hit_count = _safe_int(prev_diag_for_lookup.get("bundle_hit_count"), 0)
        bundle_miss_count = _safe_int(prev_diag_for_lookup.get("bundle_miss_count"), 0)
        consume_exact_count = _safe_int(prev_diag_for_lookup.get("consume_exact_count"), 0)
        consume_near_count = _safe_int(prev_diag_for_lookup.get("consume_near_count"), 0)
        if bundle_target < 0 or age_ms > max_age_ms or abs(target_delta) > tolerance_ticks:
            diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
            fallback_count = _safe_int(diag.get("fallback_count"), 0) + 1
            bundle_miss_count = _safe_int(diag.get("bundle_miss_count"), 0) + 1
            total_lookup = bundle_hit_count + bundle_miss_count
            diag.update({
                "enabled": True,
                "mode": mode,
                "state": "fallback_to_gui_direct",
                "fallback_reason": "bundle_stale_or_target_mismatch",
                "last_fallback_reason": "bundle_stale_or_target_mismatch",
                "last_fallback_wall_us": _now_us(),
                "fallback_count": int(fallback_count),
                "target_sync_id": int(bundle_target),
                "current_target_sync_id": int(target_sync),
                "target_delta_ticks": int(target_delta),
                "bundle_lookup_state": str(lookup_state),
                "bundle_target_delta_ticks": int(target_delta),
                "bundle_hit_count": int(bundle_hit_count),
                "bundle_miss_count": int(bundle_miss_count),
                "bundle_hit_ratio": round(float(bundle_hit_count) / float(total_lookup), 4) if total_lookup > 0 else 0.0,
                "consume_exact_count": int(consume_exact_count),
                "consume_near_count": int(consume_near_count),
                "updated_age_ms": round(float(age_ms), 3),
                "bundle_max_age_ms": round(float(max_age_ms), 3),
                "target_tolerance_ticks": int(tolerance_ticks),
                "cpu_prep_only": True,
                "opengl_thread": "gui",
                "fallback_to_gui_direct": True,
            })
            self.event_layer_diag["presentation_worker"] = diag
            return False
        per_raw = bundle.get("per_camera", {}) if isinstance(bundle.get("per_camera"), dict) else {}
        if not per_raw:
            diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
            fallback_count = _safe_int(diag.get("fallback_count"), 0) + 1
            bundle_miss_count = _safe_int(diag.get("bundle_miss_count"), 0) + 1
            total_lookup = bundle_hit_count + bundle_miss_count
            diag.update({
                "enabled": True,
                "mode": mode,
                "state": "fallback_to_gui_direct",
                "fallback_reason": "bundle_empty",
                "last_fallback_reason": "bundle_empty",
                "last_fallback_wall_us": _now_us(),
                "fallback_count": int(fallback_count),
                "target_sync_id": int(bundle_target),
                "current_target_sync_id": int(target_sync),
                "target_delta_ticks": int(target_delta),
                "bundle_lookup_state": str(lookup_state),
                "bundle_target_delta_ticks": int(target_delta),
                "bundle_hit_count": int(bundle_hit_count),
                "bundle_miss_count": int(bundle_miss_count),
                "bundle_hit_ratio": round(float(bundle_hit_count) / float(total_lookup), 4) if total_lookup > 0 else 0.0,
                "consume_exact_count": int(consume_exact_count),
                "consume_near_count": int(consume_near_count),
                "updated_age_ms": round(float(age_ms), 3),
                "bundle_max_age_ms": round(float(max_age_ms), 3),
                "target_tolerance_ticks": int(tolerance_ticks),
                "cpu_prep_only": True,
                "opengl_thread": "gui",
                "fallback_to_gui_direct": True,
            })
            self.event_layer_diag["presentation_worker"] = diag
            return False
        per_cam = self.event_layer_diag.setdefault("per_camera", {})
        if not isinstance(per_cam, dict):
            per_cam = {}
            self.event_layer_diag["per_camera"] = per_cam
        render_context = render_context or {}
        bev_render_wall_us = _now_us()
        hard_hold_ms = max(0.0, float(getattr(self.args, "event_layer_hold_last_good_ms", 250.0) or 0.0))
        hold_ms = max(0.0, float(getattr(self.args, "event_layer_display_hold_ms", getattr(self.args, "frame_hold_ms", 280.0)) or 0.0))
        draw_cams: List[str] = []
        sync_warn_cams: List[str] = []
        sync_hard_outlier_cams: List[str] = []
        sync_hold_cams: List[str] = []
        sync_hidden_cams: List[str] = []
        sync_display_state_by_cam: Dict[str, str] = {}
        sync_display_level_by_cam: Dict[str, str] = {}
        next_sparse_vertices: Dict[str, np.ndarray] = {}
        next_sparse_alpha: Dict[str, np.ndarray] = {}
        visible_total = 0
        input_total = 0
        projected_total = 0
        stale = delayed = empty = errors = hidden_by_sync = fade_count = 0
        for cam in self.cams:
            rec = per_raw.get(cam, {}) if isinstance(per_raw, dict) else {}
            if not isinstance(rec, dict):
                continue
            status_rec = {k: v for k, v in rec.items() if k not in {"vertices", "points"}}
            valid = bool(status_rec.get("draw", False))
            vertices = np.asarray(rec.get("vertices"), dtype=np.float32) if rec.get("vertices") is not None else np.empty((0, 2), dtype=np.float32)
            alpha_scale = _clip01(_safe_float(status_rec.get("alpha_scale"), 1.0 if valid else 0.0))
            prev_vertices = self.event_layer_sparse_vertices.get(cam)
            prev_valid = bool(self.event_layer_valid.get(cam, False))
            prev_age_ms = max(0.0, (time.monotonic() - float(self.event_layer_sparse_vertex_mono.get(cam, 0.0))) * 1000.0)
            if (
                not valid
                and bool(status_rec.get("event_layer_hold_last_good_candidate", False))
                and prev_valid
                and prev_vertices is not None
                and np.asarray(prev_vertices).ndim == 2
                and np.asarray(prev_vertices).shape[0] > 0
                and (hard_hold_ms <= 0.0 or prev_age_ms <= hard_hold_ms)
            ):
                vertices = np.asarray(prev_vertices, dtype=np.float32)
                valid = True
                alpha_scale = float(self.event_layer_alpha_scale.get(cam, 1.0) or 1.0)
                status_rec["draw"] = True
                status_rec["reason"] = "held_last_good_hard_outlier"
                status_rec["event_layer_sync_display_state"] = "hard_outlier_hold_last_good"
                status_rec["event_layer_sync_display_level"] = "WARN"
                status_rec["event_layer_hold_last_good"] = True
                status_rec["event_layer_hold_last_good_age_ms"] = round(float(prev_age_ms), 3)
                sync_hold_cams.append(str(cam))
            if valid and vertices.ndim == 2 and vertices.shape[0] > 0:
                next_sparse_vertices[cam] = vertices.astype(np.float32, copy=False)
                self.event_layer_sparse_vertex_mono[cam] = time.monotonic()
                draw_cams.append(str(cam))
                visible_total += int(vertices.shape[0])
                if alpha_scale < 0.999:
                    fade_count += 1
                    delayed += 1
            else:
                valid = False
                reason = str(status_rec.get("reason", "invalid"))
                state = str(status_rec.get("event_layer_sync_state", reason))
                if state == "stale" or reason == "event_layer_wall_age_stale":
                    stale += 1
                elif state in {"empty", "no_visible_points"} or reason == "empty":
                    empty += 1
                elif reason == "error":
                    errors += 1
                else:
                    hidden_by_sync += 1
                sync_hidden_cams.append(str(cam))
            self.event_layer_valid[cam] = bool(valid)
            self.event_layer_alpha_scale[cam] = float(alpha_scale if valid else 0.0)
            display_state = str(status_rec.get("event_layer_sync_display_state", status_rec.get("event_layer_sync_state", "")) or "")
            display_level = str(status_rec.get("event_layer_sync_display_level", "OK" if valid else "HARD") or "")
            sync_display_state_by_cam[str(cam)] = display_state
            sync_display_level_by_cam[str(cam)] = display_level
            if display_level.upper() == "WARN" and str(cam) not in sync_warn_cams:
                sync_warn_cams.append(str(cam))
            if display_state in {"hard_outlier", "hard_outlier_hold_last_good"} and str(cam) not in sync_hard_outlier_cams:
                sync_hard_outlier_cams.append(str(cam))
            input_total += max(0, _safe_int(status_rec.get("sparse_point_count"), 0))
            projected_total += max(0, _safe_int(status_rec.get("sparse_projected_points"), 0))
            mvs_item = render_context.get(cam, {}) if isinstance(render_context, dict) else {}
            mvs_ts_us = _safe_int(mvs_item.get("timestamp_us"), 0) if isinstance(mvs_item, dict) else 0
            mvs_age_ms = max(0.0, (bev_render_wall_us - int(mvs_ts_us)) / 1000.0) if mvs_ts_us > 1_000_000_000_000 else -1.0
            status_rec.update({
                "open": True,
                "backend": "sparse_gl",
                "draw": bool(valid),
                "target_sync_id": int(target_sync),
                "worker_target_sync_id": int(bundle_target),
                "presentation_source": "worker_active",
                "mvs_frame_id": _safe_int(mvs_item.get("frame_id"), -1) if isinstance(mvs_item, dict) else -1,
                "mvs_tick_id": _safe_int(mvs_item.get("tick_id"), -1) if isinstance(mvs_item, dict) else -1,
                "mvs_frame_age_ms": round(float(mvs_age_ms), 3) if mvs_age_ms >= 0.0 else -1.0,
            })
            per_cam[cam] = dict(status_rec)
            self.event_layer_items[cam] = dict(status_rec)
        self.event_layer_sparse_vertices = next_sparse_vertices
        self.event_layer_sparse_point_alpha = next_sparse_alpha
        elapsed_ms = (time.perf_counter() - budget_t0) * 1000.0
        budget_stats = self._update_budget_diag("event_layer_display", elapsed_ms, 0.0, 0)
        skip_reason = "drawing" if draw_cams else ("errors" if errors else ("stale" if stale else ("empty" if empty else "hidden_by_sync")))
        exact_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_source", "")) == "event_sync_map_ring_exact")
        interpolated_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_source", "")) == "event_sync_map_ring_interpolated")
        estimated_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and "estimated" in str(rec.get("event_sync_source", "")))
        missing_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_quality", "")).lower() == "missing")
        self.event_layer_diag.update({
            "enabled": True,
            "backend": "sparse_gl",
            "draw_cams": list(draw_cams),
            "draw_cam_count": int(len(draw_cams)),
            "shader_draw_cams": list(draw_cams),
            "shader_draw_cam_count": int(len(draw_cams)),
            "uploaded_this_tick": 0,
            "uploaded_cams": [],
            "sparse_input_points": int(input_total),
            "sparse_projected_points": int(projected_total),
            "sparse_visible_points": int(visible_total),
            "display_budget_enable": bool(getattr(self.args, "event_layer_display_budget_enable", True)),
            "display_budget_ms": float(getattr(self.args, "event_layer_display_budget_ms", 10.0) or 0.0),
            "display_budget_elapsed_ms": round(float(elapsed_ms), 3),
            "display_budget_order": [],
            "display_budget_overrun": bool(budget_stats.get("overrun", False)),
            "display_budget_overrun_rate": float(budget_stats.get("overrun_rate", 0.0)),
            "display_budget_overrun_count": int(budget_stats.get("overrun_count", 0)),
            "display_budget_sample_count": int(budget_stats.get("sample_count", 0)),
            "display_budget_elapsed_p95_ms": float(budget_stats.get("elapsed_p95_ms", 0.0)),
            "display_budget_held_cam_count_p95": float(budget_stats.get("held_cam_count_p95", 0.0)),
            "display_budget_held_cams": [],
            "display_budget_held_cam_count": 0,
            "display_budget_forced_cams": [],
            "display_budget_forced_cam_count": 0,
            "sparse_read_elapsed_ms": 0.0,
            "sparse_sync_gate_elapsed_ms": 0.0,
            "sparse_project_elapsed_ms": 0.0,
            "sparse_cpu_prepare_elapsed_ms": round(float(elapsed_ms), 3),
            "display_hold_ms": round(float(hold_ms), 3),
            "sync_display_state_by_cam": dict(sync_display_state_by_cam),
            "sync_display_level_by_cam": dict(sync_display_level_by_cam),
            "sync_warn_cams": list(dict.fromkeys(sync_warn_cams)),
            "sync_warn_cam_count": int(len(dict.fromkeys(sync_warn_cams))),
            "sync_hard_outlier_cams": list(dict.fromkeys(sync_hard_outlier_cams)),
            "sync_hard_outlier_cam_count": int(len(dict.fromkeys(sync_hard_outlier_cams))),
            "sync_hold_cams": list(dict.fromkeys(sync_hold_cams)),
            "sync_hold_cam_count": int(len(dict.fromkeys(sync_hold_cams))),
            "sync_hidden_cams": list(dict.fromkeys(sync_hidden_cams)),
            "sync_hidden_cam_count": int(len(dict.fromkeys(sync_hidden_cams))),
            "stale_cam_count": int(stale),
            "delayed_cam_count": int(delayed),
            "hidden_by_sync_cam_count": int(hidden_by_sync),
            "fade_cam_count": int(fade_count),
            "empty_cam_count": int(empty),
            "error_cam_count": int(errors),
            "skip_reason": skip_reason,
            "sync_truth_mode": "history_ring",
            "sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "exact_sync_cam_count": int(exact_sync_count),
            "interpolated_sync_cam_count": int(interpolated_sync_count),
            "estimated_sync_cam_count": int(estimated_sync_count),
            "missing_sync_cam_count": int(missing_sync_count),
            "display_only": True,
            "judgement_sync_threshold_ms": 100.0,
            "bev_visual_sync_target_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 50.0)),
            "fade_alpha_scale": float(getattr(self.args, "event_layer_fade_alpha_scale", 0.25)),
            "color_mode": _event_layer_color_mode(getattr(self.args, "event_layer_color_mode", "camera")),
            "available_color_modes": list(EVENT_LAYER_COLOR_MODE_ORDER),
            "dilate_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
            "requested_dilate_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
            "effective_dilate_px": float(effective_dilate_px),
            "dilate_backend": "opengl_points",
            "dilate_max_pixels": int(getattr(self.args, "event_layer_dilate_max_pixels", 2500) or 0),
            "performance_guard": dict(perf_guard),
            "max_texture_units": int(self.max_texture_units or 0),
            "transparent_background": True,
            "render_target": "bev_fbo",
        })
        diag = worker.snapshot(int(target_sync), int(tolerance_ticks))
        prev_diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
        bundle_hit_count = _safe_int(prev_diag.get("bundle_hit_count"), 0) + 1
        bundle_miss_count = _safe_int(prev_diag.get("bundle_miss_count"), 0)
        consume_exact_count = _safe_int(prev_diag.get("consume_exact_count"), 0) + (1 if lookup_state == "exact" else 0)
        consume_near_count = _safe_int(prev_diag.get("consume_near_count"), 0) + (1 if lookup_state == "near" else 0)
        total_lookup = bundle_hit_count + bundle_miss_count
        diag.update({
            "mode": mode,
            "state": "active_consumed",
            "worker_state": str(diag.get("state", "") or ""),
            "active_consume_count": _safe_int(prev_diag.get("active_consume_count"), 0) + 1,
            "last_consume_state": "active_consumed",
            "last_active_consume_wall_us": _now_us(),
            "fallback_count": _safe_int(prev_diag.get("fallback_count"), 0),
            "bundle_lookup_state": str(lookup_state),
            "bundle_target_delta_ticks": int(target_delta),
            "bundle_hit_count": int(bundle_hit_count),
            "bundle_miss_count": int(bundle_miss_count),
            "bundle_hit_ratio": round(float(bundle_hit_count) / float(total_lookup), 4) if total_lookup > 0 else 0.0,
            "consume_exact_count": int(consume_exact_count),
            "consume_near_count": int(consume_near_count),
            "current_target_sync_id": int(target_sync),
            "target_delta_ticks": int(target_delta),
            "updated_age_ms": round(float(age_ms), 3),
            "active_consume_elapsed_ms": round(float(elapsed_ms), 3),
            "cpu_prep_only": True,
            "opengl_thread": "gui",
            "fallback_to_gui_direct": True,
        })
        self.event_layer_diag["presentation_worker"] = diag
        self._event_layer_shadow_submit_and_diag(int(target_sync), per_cam)
        active_diag = dict(self.event_layer_diag.get("presentation_worker", {})) if isinstance(self.event_layer_diag.get("presentation_worker"), dict) else {}
        active_diag.update({
            "mode": mode,
            "state": "active_consumed",
            "last_consume_state": "active_consumed",
            "active_consume_count": _safe_int(active_diag.get("active_consume_count"), _safe_int(prev_diag.get("active_consume_count"), 0) + 1),
            "last_active_consume_wall_us": _now_us(),
            "fallback_count": _safe_int(active_diag.get("fallback_count"), _safe_int(prev_diag.get("fallback_count"), 0)),
            "bundle_lookup_state": str(lookup_state),
            "bundle_target_delta_ticks": int(target_delta),
            "bundle_hit_count": int(bundle_hit_count),
            "bundle_miss_count": int(bundle_miss_count),
            "bundle_hit_ratio": round(float(bundle_hit_count) / float(total_lookup), 4) if total_lookup > 0 else 0.0,
            "consume_exact_count": int(consume_exact_count),
            "consume_near_count": int(consume_near_count),
            "current_target_sync_id": int(target_sync),
            "target_delta_ticks": int(target_delta),
            "updated_age_ms": round(float(age_ms), 3),
            "active_consume_elapsed_ms": round(float(elapsed_ms), 3),
            "cpu_prep_only": True,
            "opengl_thread": "gui",
            "fallback_to_gui_direct": True,
        })
        self.event_layer_diag["presentation_worker"] = active_diag
        return True

    def _event_layer_shadow_submit_and_diag(self, target_sync: int, active_per_cam: Dict[str, Any]) -> None:
        target_i = int(target_sync)
        active_compact: Dict[str, Any] = {}
        for cam in self.cams:
            rec = active_per_cam.get(cam, {}) if isinstance(active_per_cam, dict) else {}
            if not isinstance(rec, dict):
                continue
            active_compact[cam] = {
                "frame_id": _safe_int(rec.get("frame_id", rec.get("event_layer_frame_id")), -1),
                "draw": bool(rec.get("draw", False)),
                "event_to_bev_sync_delta_ms": _safe_float(rec.get("event_to_bev_sync_delta_ms"), 999999.0),
                "reason": str(rec.get("reason", rec.get("sync_reason", ""))),
            }
        self.event_layer_active_history[target_i] = active_compact
        self.event_layer_active_history_order.append(target_i)
        while len(self.event_layer_active_history_order) > 180:
            old = self.event_layer_active_history_order.pop(0)
            self.event_layer_active_history.pop(int(old), None)
        worker = getattr(self, "event_layer_sparse_shadow", None)
        shadow_obj: Dict[str, Any]
        if worker is None or not bool(getattr(self.args, "event_layer_sparse_shadow_enable", False)):
            shadow_obj = {
                "enabled": False,
                "mode": "shadow",
                "state": "disabled",
                "target_sync_id": int(target_i),
                "current_target_sync_id": int(target_i),
                "matching_target": False,
                "mismatch_cam_count": 0,
                "per_camera": {},
            }
            self.event_layer_diag["shadow"] = shadow_obj
            self._event_layer_presentation_compare_and_diag(target_i, active_compact)
            return
        config = {
            "mvs_sync_tick_ms_est": float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0) or 25.0),
            "event_layer_max_sync_delta_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 20.0) or 20.0),
            "event_layer_max_sync_delta_ticks": int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2) or 2),
            "event_layer_sync_normal_ms": float(getattr(self.args, "event_layer_sync_normal_ms", 50.0) or 50.0),
            "event_layer_sync_warn_ms": float(getattr(self.args, "event_layer_sync_warn_ms", 80.0) or 80.0),
            "event_layer_sync_hard_ms": float(getattr(self.args, "event_layer_sync_hard_ms", 80.0) or 80.0),
            "candidate_scan_initial": int(getattr(self.args, "event_layer_sparse_candidate_scan_initial", 4) or 4),
            "candidate_scan_escalated": int(getattr(self.args, "event_layer_sparse_candidate_scan_escalated", 32) or 32),
            "candidate_escalate_enable": bool(getattr(self.args, "event_layer_sparse_candidate_escalate_enable", True)),
            "candidate_escalate_abs_ms": float(getattr(self.args, "event_layer_sparse_candidate_escalate_abs_ms", 10.0) or 10.0),
            "candidate_escalate_ratio": float(getattr(self.args, "event_layer_sparse_candidate_escalate_ratio", 0.60) or 0.60),
        }
        try:
            worker.submit(int(target_i), config)
            shadow_obj = worker.snapshot()
        except Exception as exc:
            self.event_layer_diag["shadow"] = {
                "enabled": True,
                "mode": "shadow",
                "state": "error",
                "target_sync_id": int(target_i),
                "current_target_sync_id": int(target_i),
                "error": f"{type(exc).__name__}: {exc}",
                "per_camera": {},
            }
            return
        shadow_target = _safe_int(shadow_obj.get("target_sync_id"), -1)
        matching_target = bool(shadow_target in self.event_layer_active_history)
        active_compare = self.event_layer_active_history.get(shadow_target, active_compact if shadow_target == target_i else {})
        shadow_per_cam = shadow_obj.get("per_camera", {}) if isinstance(shadow_obj.get("per_camera"), dict) else {}
        compare_per_cam: Dict[str, Any] = {}
        mismatch_count = 0
        frame_mismatch_count = 0
        draw_mismatch_count = 0
        delta_mismatch_count = 0
        for cam in self.cams:
            active = active_compare.get(cam, {}) if isinstance(active_compare, dict) else {}
            shadow = shadow_per_cam.get(cam, {}) if isinstance(shadow_per_cam, dict) else {}
            if not isinstance(active, dict) or not isinstance(shadow, dict):
                continue
            active_frame = _safe_int(active.get("frame_id", active.get("event_layer_frame_id")), -1)
            shadow_frame = _safe_int(shadow.get("frame_id"), -1)
            active_draw = bool(active.get("draw", False))
            shadow_draw = bool(shadow.get("draw", False))
            active_delta = _safe_float(active.get("event_to_bev_sync_delta_ms"), 999999.0)
            shadow_delta = _safe_float(shadow.get("event_to_bev_sync_delta_ms"), 999999.0)
            frame_match = bool(active_frame == shadow_frame) if matching_target else False
            draw_match = bool(active_draw == shadow_draw) if matching_target else False
            delta_diff_ms = abs(float(active_delta) - float(shadow_delta)) if abs(active_delta) < 900000.0 and abs(shadow_delta) < 900000.0 else -1.0
            delta_match = bool(delta_diff_ms >= 0.0 and delta_diff_ms <= 1.0) if matching_target else bool(False)
            if matching_target and not active_draw and not shadow_draw and active_frame == shadow_frame and delta_diff_ms < 0.0:
                delta_match = True
            mismatch = bool(matching_target and not (frame_match and draw_match and delta_match))
            if mismatch:
                mismatch_count += 1
            if matching_target and not frame_match:
                frame_mismatch_count += 1
            if matching_target and not draw_match:
                draw_mismatch_count += 1
            if matching_target and not delta_match:
                delta_mismatch_count += 1
            compare_per_cam[cam] = {
                "active_frame_id": int(active_frame),
                "shadow_frame_id": int(shadow_frame),
                "active_draw": bool(active_draw),
                "shadow_draw": bool(shadow_draw),
                "active_delta_ms": round(float(active_delta), 3) if abs(active_delta) < 900000.0 else None,
                "shadow_delta_ms": round(float(shadow_delta), 3) if abs(shadow_delta) < 900000.0 else None,
                "delta_diff_ms": round(float(delta_diff_ms), 3) if delta_diff_ms >= 0.0 else None,
                "frame_match": bool(frame_match),
                "draw_match": bool(draw_match),
                "delta_match": bool(delta_match),
                "mismatch": bool(mismatch),
            }
        updated_us = _safe_int(shadow_obj.get("updated_wall_us"), 0)
        shadow_age_ms = max(0.0, (_now_us() - int(updated_us)) / 1000.0) if updated_us > 1_000_000_000_000 else -1.0
        shadow_obj["updated_age_ms"] = round(float(shadow_age_ms), 3) if shadow_age_ms >= 0.0 else -1.0
        shadow_obj["current_target_sync_id"] = int(target_i)
        shadow_obj["active_history_count"] = int(len(self.event_layer_active_history_order))
        shadow_obj["matching_target"] = bool(matching_target)
        shadow_obj["compare_state"] = "match" if matching_target and mismatch_count == 0 else ("pending_target" if not matching_target else "mismatch")
        shadow_obj["mismatch_cam_count"] = int(mismatch_count)
        shadow_obj["frame_mismatch_cam_count"] = int(frame_mismatch_count)
        shadow_obj["draw_mismatch_cam_count"] = int(draw_mismatch_count)
        shadow_obj["delta_mismatch_cam_count"] = int(delta_mismatch_count)
        shadow_obj["compare_per_camera"] = compare_per_cam
        self.event_layer_diag["shadow"] = shadow_obj
        self._event_layer_presentation_compare_and_diag(target_i, active_compact)

    def _upload_event_layer_sparse_points(
        self,
        render_context: Optional[Dict[str, Dict[str, Any]]] = None,
        target_sync: int = -1,
    ) -> None:
        if not bool(getattr(self.args, "event_layer_enable", False)):
            self.event_layer_valid = {cam: False for cam in self.cams}
            self.event_layer_alpha_scale = {cam: 0.0 for cam in self.cams}
            self.event_layer_sparse_vertices = {}
            self.event_layer_diag.update({"enabled": False, "backend": "sparse_gl", "draw_cams": [], "draw_cam_count": 0, "skip_reason": "disabled"})
            return
        self._open_event_layer_sparse_inputs()
        per_cam = self.event_layer_diag.setdefault("per_camera", {})
        if not isinstance(per_cam, dict):
            per_cam = {}
            self.event_layer_diag["per_camera"] = per_cam
        max_age_ms = max(1.0, float(getattr(self.args, "event_layer_max_age_ms", 150.0) or 150.0))
        effective_dilate_px, perf_guard = self._event_layer_effective_dilate()
        bev_render_wall_us = _now_us()
        render_context = render_context or {}
        budget_enable = bool(getattr(self.args, "event_layer_display_budget_enable", True))
        budget_ms = max(0.0, float(getattr(self.args, "event_layer_display_budget_ms", 10.0) or 0.0))
        hold_ms = max(0.0, float(getattr(self.args, "event_layer_display_hold_ms", getattr(self.args, "frame_hold_ms", 280.0)) or 0.0))
        budget_t0 = time.perf_counter()
        draw_cams: List[str] = []
        stale = 0
        delayed = 0
        empty = 0
        errors = 0
        hidden_by_sync = 0
        fade_count = 0
        projected_total = 0
        visible_total = 0
        input_total = 0
        sparse_dropped_total = 0
        held_by_budget_cams: List[str] = []
        forced_by_budget_cams: List[str] = []
        budget_expired_hold_cams: List[str] = []
        sync_normal_ms, sync_warn_ms, sync_hard_ms = self._event_layer_sync_display_thresholds()
        hard_hold_enable = bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True))
        hard_hold_ms = max(0.0, float(getattr(self.args, "event_layer_hold_last_good_ms", 250.0) or 0.0))
        sync_display_state_by_cam: Dict[str, str] = {}
        sync_display_level_by_cam: Dict[str, str] = {}
        sync_warn_cams: List[str] = []
        sync_hard_outlier_cams: List[str] = []
        sync_hold_cams: List[str] = []
        sync_hidden_cams: List[str] = []
        next_sparse_vertices: Dict[str, np.ndarray] = {}
        next_sparse_alpha: Dict[str, np.ndarray] = {}
        display_budget_order = self._budget_order(self.cams, "_event_layer_budget_cursor")
        self._event_layer_presentation_submit(int(target_sync))
        if self._try_consume_event_layer_presentation_bundle(
            render_context=render_context,
            target_sync=int(target_sync),
            budget_t0=budget_t0,
            effective_dilate_px=float(effective_dilate_px),
            perf_guard=perf_guard,
        ):
            return
        sparse_read_elapsed_ms = 0.0
        sparse_sync_gate_elapsed_ms = 0.0
        sparse_project_elapsed_ms = 0.0
        for cam in display_budget_order:
            rd = self.event_layer_sparse_readers.get(cam)
            if rd is None:
                self.event_layer_valid[cam] = False
                self.event_layer_alpha_scale[cam] = 0.0
                continue
            try:
                elapsed_before_ms = (time.perf_counter() - budget_t0) * 1000.0
                prev_vertices = self.event_layer_sparse_vertices.get(cam)
                prev_valid = bool(self.event_layer_valid.get(cam, False))
                prev_age_ms = max(
                    0.0,
                    (time.monotonic() - float(self.event_layer_sparse_vertex_mono.get(cam, 0.0))) * 1000.0,
                )
                can_hold_previous = bool(
                    prev_valid
                    and prev_vertices is not None
                    and np.asarray(prev_vertices).ndim == 2
                    and np.asarray(prev_vertices).shape[0] > 0
                    and (hold_ms <= 0.0 or prev_age_ms <= hold_ms)
                )
                if budget_enable and budget_ms > 0.0 and elapsed_before_ms >= budget_ms and can_hold_previous:
                    held_by_budget_cams.append(str(cam))
                    prev_arr = np.asarray(prev_vertices, dtype=np.float32)
                    prev_item = self.event_layer_items.get(cam, {})
                    if not isinstance(prev_item, dict):
                        prev_item = {}
                    next_sparse_vertices[cam] = prev_arr
                    prev_alpha_arr = self.event_layer_sparse_point_alpha.get(cam)
                    if prev_alpha_arr is not None:
                        next_sparse_alpha[cam] = prev_alpha_arr
                    alpha_scale = float(self.event_layer_alpha_scale.get(cam, 1.0) or 1.0)
                    self.event_layer_valid[cam] = True
                    self.event_layer_alpha_scale[cam] = float(alpha_scale)
                    draw_cams.append(str(cam))
                    sync_display_state_by_cam[str(cam)] = "held_by_display_budget"
                    sync_display_level_by_cam[str(cam)] = "WARN"
                    visible_total += int(prev_arr.shape[0])
                    projected_total += int(prev_arr.shape[0])
                    per_cam[cam] = {
                        "open": True,
                        "backend": "sparse_gl",
                        "draw": True,
                        "reason": "held_by_display_budget",
                        "event_layer_sync_state": "held_by_display_budget",
                        "event_layer_sync_display_state": "held_by_display_budget",
                        "event_layer_sync_display_level": "WARN",
                        "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                        "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                        "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                        "event_layer_hold_last_good": True,
                        "event_layer_hold_last_good_reason": "display_budget",
                        "frame_id": _safe_int(prev_item.get("frame_id"), -1),
                        "event_layer_frame_id": _safe_int(prev_item.get("frame_id"), -1),
                        "event_to_bev_sync_delta_ticks": _safe_int(prev_item.get("event_to_bev_sync_delta_ticks"), -999999),
                        "event_to_bev_sync_delta_ms": _safe_float(prev_item.get("event_to_bev_sync_delta_ms"), 999999.0),
                        "event_mapped_mvs_sync_index": _safe_int(prev_item.get("event_mapped_mvs_sync_index"), -1),
                        "sync_reason": str(prev_item.get("sync_reason", "held_by_display_budget")),
                        "sync_mode": str(prev_item.get("sync_mode", getattr(self.args, "event_layer_sync_mode", ""))),
                        "target_sync_id": int(target_sync),
                        "history_selected": bool(prev_item.get("history_selected", False)),
                        "history_source": str(prev_item.get("history_source", "")),
                        "sparse_visible_points": int(prev_arr.shape[0]),
                        "display_budget_enable": bool(budget_enable),
                        "display_budget_ms": round(float(budget_ms), 3),
                        "display_budget_elapsed_before_ms": round(float(elapsed_before_ms), 3),
                        "display_hold_age_ms": round(float(prev_age_ms), 3),
                        "display_hold_ms": round(float(hold_ms), 3),
                        "error": "",
                    }
                    continue
                if budget_enable and budget_ms > 0.0 and elapsed_before_ms >= budget_ms:
                    if prev_vertices is not None and not can_hold_previous:
                        budget_expired_hold_cams.append(str(cam))
                    else:
                        forced_by_budget_cams.append(str(cam))
                read_t0 = self._perf_start()
                hint_meta = self._event_sync_ring_reader(cam).event_ts_for_mvs_sync(target_sync)
                center_hint_us = _safe_int(hint_meta.get("event_ts_us"), 0) if isinstance(hint_meta, dict) else 0
                item = rd.read_for_target(
                    target_sync,
                    lambda meta, tgt, _cam=cam: self._event_layer_score_meta(_cam, meta, tgt),
                    center_hint_us=center_hint_us,
                    candidate_limit=int(getattr(self.args, "event_layer_sparse_candidate_scan_initial", 4) or 4),
                    candidate_escalate_enable=bool(getattr(self.args, "event_layer_sparse_candidate_escalate_enable", True)),
                    candidate_escalate_limit=int(getattr(self.args, "event_layer_sparse_candidate_scan_escalated", 32) or 32),
                    candidate_escalate_abs_ms=float(getattr(self.args, "event_layer_sparse_candidate_escalate_abs_ms", 10.0) or 10.0),
                    candidate_escalate_ratio=float(getattr(self.args, "event_layer_sparse_candidate_escalate_ratio", 0.60) or 0.60),
                    max_sync_delta_ms=float(getattr(self.args, "event_layer_max_sync_delta_ms", 20.0) or 20.0),
                )
                read_ms = (time.perf_counter() - float(read_t0)) * 1000.0
                self._perf_end("event_layer_sparse_read", read_t0)
                sparse_read_elapsed_ms += float(read_ms)
                if item is None:
                    self.event_layer_valid[cam] = False
                    self.event_layer_alpha_scale[cam] = 0.0
                    sync_display_state_by_cam[str(cam)] = "missing"
                    sync_display_level_by_cam[str(cam)] = "MISSING"
                    sync_hidden_cams.append(str(cam))
                    per_cam[cam] = {
                        "open": True,
                        "backend": "sparse_gl",
                        "draw": False,
                        "reason": "no_sparse_frame",
                        "event_layer_sync_state": "missing",
                        "event_layer_sync_display_state": "missing",
                        "event_layer_sync_display_level": "MISSING",
                        "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                        "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                        "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                        "sidecar_path": str(rd.sidecar_path),
                        "last_read_error": str(getattr(rd, "last_read_error", "")),
                        "read_ms": round(float(read_ms), 3),
                    }
                    continue
                meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
                points = np.asarray(item.get("points"), dtype=_SPARSE_POINT_DTYPE)
                point_count = int(points.shape[0]) if points.ndim == 1 else 0
                input_total += point_count
                fid = _safe_int(item.get("frame_id"), -1)
                shape = item.get("shape", [0, 0, 1])
                source_w = _safe_int(shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else 0, 0)
                source_h = _safe_int(shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else 0, 0)
                pub_us = _safe_int(meta.get("published_wall_us"), 0)
                age_ms = max(0.0, (bev_render_wall_us - int(pub_us)) / 1000.0) if pub_us > 1_000_000_000_000 else -1.0
                nonzero = _safe_int(meta.get("overlay_nonzero_pixels", point_count), point_count)
                max_luma = _safe_int(meta.get("overlay_max_luma", 255 if point_count > 0 else 0), 255 if point_count > 0 else 0)
                sync_t0 = self._perf_start()
                decision = self._event_layer_sync_decision(
                    cam=cam,
                    target_sync=target_sync,
                    age_ms=age_ms,
                    nonzero=nonzero,
                    max_luma=max_luma,
                    meta=meta,
                    item=item,
                )
                sync_ms = (time.perf_counter() - float(sync_t0)) * 1000.0
                self._perf_end("event_layer_sparse_sync_gate", sync_t0)
                sparse_sync_gate_elapsed_ms += float(sync_ms)
                valid = bool(decision.get("valid", False))
                alpha_scale = float(decision.get("alpha_scale", 0.0) or 0.0)
                h_event_to_mvs_for_cache = self.event_layer_event_to_mvs.get(cam)
                h_img_to_field_for_cache = self._event_bullet_img_to_field(cam)
                homography_ready = bool(cam in self.H and h_event_to_mvs_for_cache is not None and h_img_to_field_for_cache is not None)
                projection_meta: Dict[str, Any] = {}
                vertices = np.empty((0, 2), dtype=np.float32)
                project_ms = 0.0
                if valid and not homography_ready:
                    valid = False
                    alpha_scale = 0.0
                    decision["state"] = "no_homography"
                    decision["reason"] = "missing_field_or_event_to_mvs_homography"
                if valid:
                    proj_t0 = self._perf_start()
                    vertices, projection_meta = self._project_sparse_points_to_ndc(cam, points)
                    project_ms = (time.perf_counter() - float(proj_t0)) * 1000.0
                    self._perf_end("event_layer_sparse_project", proj_t0)
                    sparse_project_elapsed_ms += float(project_ms)
                    if vertices.shape[0] <= 0:
                        valid = False
                        alpha_scale = 0.0
                        decision["state"] = "no_visible_points"
                        decision["reason"] = str(projection_meta.get("skip_reason", "no_visible_points"))
                held_last_good = False
                hold_source_frame_id = -1
                hold_source_sync_delta_ms = -999999.0
                hold_reason = ""
                if (
                    not valid
                    and bool(decision.get("event_layer_hold_last_good_candidate", False))
                    and bool(hard_hold_enable)
                ):
                    hard_can_hold_previous = bool(
                        prev_valid
                        and prev_vertices is not None
                        and np.asarray(prev_vertices).ndim == 2
                        and np.asarray(prev_vertices).shape[0] > 0
                        and (hard_hold_ms <= 0.0 or prev_age_ms <= hard_hold_ms)
                    )
                    sync_hard_outlier_cams.append(str(cam))
                    if hard_can_hold_previous:
                        prev_arr = np.asarray(prev_vertices, dtype=np.float32)
                        prev_alpha_arr = self.event_layer_sparse_point_alpha.get(cam)
                        if prev_alpha_arr is not None:
                            next_sparse_alpha[cam] = prev_alpha_arr
                        prev_item = self.event_layer_items.get(cam, {})
                        if not isinstance(prev_item, dict):
                            prev_item = {}
                        vertices = prev_arr
                        projection_meta = {
                            "projected_points": int(prev_arr.shape[0]),
                            "skip_reason": "held_last_good_hard_outlier",
                        }
                        hold_source_frame_id = _safe_int(prev_item.get("frame_id"), -1)
                        hold_source_sync_delta_ms = _safe_float(prev_item.get("event_to_bev_sync_delta_ms"), -999999.0)
                        hold_reason = "hard_outlier_hold_last_good"
                        decision["event_layer_sync_original_state"] = str(decision.get("state", ""))
                        decision["event_layer_sync_original_reason"] = str(decision.get("reason", ""))
                        decision["state"] = "hard_outlier_hold_last_good"
                        decision["reason"] = hold_reason
                        decision["event_layer_sync_display_state"] = "hard_outlier_hold_last_good"
                        decision["event_layer_sync_display_level"] = "HOLD"
                        decision["event_layer_hold_last_good"] = True
                        decision["event_layer_hold_last_good_age_ms"] = round(float(prev_age_ms), 3)
                        decision["event_layer_hold_last_good_ms"] = round(float(hard_hold_ms), 3)
                        decision["event_layer_hold_last_good_source_frame_id"] = int(hold_source_frame_id)
                        valid = True
                        held_last_good = True
                        alpha_scale = float(self.event_layer_alpha_scale.get(cam, 1.0) or 1.0)
                        sync_hold_cams.append(str(cam))
                    else:
                        sync_hidden_cams.append(str(cam))
                if not valid:
                    if str(decision.get("state")) == "delayed":
                        delayed += 1
                        hidden_by_sync += 1
                        reason = str(decision.get("reason") or "delayed")
                    elif age_ms > max_age_ms:
                        stale += 1
                        reason = "stale"
                    elif nonzero == 0 or max_luma == 0:
                        empty += 1
                        reason = "empty"
                    else:
                        hidden_by_sync += 1
                        reason = str(decision.get("reason") or "invalid")
                else:
                    draw_cams.append(cam)
                    reason = hold_reason or ("drawing" if alpha_scale >= 0.999 else "drawing_faded")
                    if alpha_scale < 0.999 and not held_last_good:
                        fade_count += 1
                        delayed += 1
                    next_sparse_vertices[cam] = vertices
                    if not held_last_good:
                        self.event_layer_sparse_vertex_mono[cam] = time.monotonic()
                    visible_total += int(vertices.shape[0])
                    projected_total += int(projection_meta.get("projected_points", vertices.shape[0]))
                self.event_layer_valid[cam] = bool(valid)
                self.event_layer_alpha_scale[cam] = float(alpha_scale if valid else 0.0)
                sync_meta = decision.get("sync_meta", {}) if isinstance(decision.get("sync_meta"), dict) else {}
                event_to_bev_sync_delta_ticks = decision.get("event_to_bev_sync_delta_ticks", None)
                event_to_bev_sync_delta_ticks_i = int(event_to_bev_sync_delta_ticks) if event_to_bev_sync_delta_ticks is not None else -999999
                event_to_bev_sync_delta_ms = decision.get("event_to_bev_sync_delta_ms", None)
                event_to_bev_sync_delta_ms_f = _safe_float(event_to_bev_sync_delta_ms, -999999.0) if event_to_bev_sync_delta_ms is not None else -999999.0
                display_state = str(decision.get("event_layer_sync_display_state", decision.get("state", "")) or "")
                display_level = str(decision.get("event_layer_sync_display_level", "OK" if valid else "HARD") or "")
                sync_display_state_by_cam[str(cam)] = display_state
                sync_display_level_by_cam[str(cam)] = display_level
                if display_level.upper() == "WARN" and str(cam) not in sync_warn_cams:
                    sync_warn_cams.append(str(cam))
                if display_state in {"hard_outlier", "hard_outlier_hold_last_good"} and str(cam) not in sync_hard_outlier_cams:
                    sync_hard_outlier_cams.append(str(cam))
                if not valid and str(cam) not in sync_hidden_cams:
                    sync_hidden_cams.append(str(cam))
                sparse_dropped = _safe_int(meta.get("sparse_dropped_point_count"), _safe_int(meta.get("event_layer_sparse_dropped_point_count"), 0))
                sparse_dropped_total += max(0, int(sparse_dropped))
                mvs_item = render_context.get(cam, {}) if isinstance(render_context, dict) else {}
                mvs_ts_us = _safe_int(mvs_item.get("timestamp_us"), 0) if isinstance(mvs_item, dict) else 0
                mvs_age_ms = max(0.0, (bev_render_wall_us - int(mvs_ts_us)) / 1000.0) if mvs_ts_us > 1_000_000_000_000 else -1.0
                self.event_layer_items[cam] = {
                    "backend": "sparse_gl",
                    "width": int(source_w),
                    "height": int(source_h),
                    "channels": 1,
                    "frame_id": int(fid),
                    "age_ms": round(float(age_ms), 3),
                    "event_ts_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "published_wall_us": int(pub_us),
                    "bev_render_wall_us": int(bev_render_wall_us),
                    "nonzero_pixels": int(nonzero),
                    "max_luma": int(max_luma),
                    "sync_state": str(decision.get("state", "")),
                    "sync_reason": str(decision.get("reason", "")),
                    "sync_mode": str(decision.get("sync_mode", "")),
                    "sync_display_state": str(display_state),
                    "sync_display_level": str(display_level),
                    "event_layer_sync_display_state": str(display_state),
                    "event_layer_sync_display_level": str(display_level),
                    "event_layer_sync_abs_delta_ms": decision.get("event_layer_sync_abs_delta_ms"),
                    "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                    "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                    "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                    "event_layer_hold_last_good": bool(decision.get("event_layer_hold_last_good", False)),
                    "event_layer_hold_last_good_age_ms": decision.get("event_layer_hold_last_good_age_ms", -1.0),
                    "event_layer_hold_last_good_ms": decision.get("event_layer_hold_last_good_ms", round(float(hard_hold_ms), 3)),
                    "event_layer_hold_last_good_source_frame_id": decision.get("event_layer_hold_last_good_source_frame_id", int(hold_source_frame_id)),
                    "event_layer_hold_last_good_source_sync_delta_ms": round(float(hold_source_sync_delta_ms), 3),
                    "event_mapped_mvs_sync_index": _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1),
                    "event_to_bev_sync_delta_ticks": int(event_to_bev_sync_delta_ticks_i),
                    "event_to_bev_sync_delta_ms": round(float(event_to_bev_sync_delta_ms_f), 3),
                    "event_sync": dict(sync_meta),
                    "history_selected": bool(item.get("history_selected", False)),
                    "history_source": str(meta.get("event_layer_history_source", "")),
                    "history_best_abs_delta_ms": _safe_float(item.get("history_best_abs_delta_ms"), -1.0),
                    "alpha_scale": round(float(alpha_scale), 3),
                    "sparse_point_count": int(point_count),
                    "sparse_visible_points": int(vertices.shape[0]) if valid else 0,
                    "sparse_dropped_point_count": int(sparse_dropped),
                    "read_ms": round(float(read_ms), 3),
                    "sync_gate_ms": round(float(sync_ms), 3),
                    "project_ms": round(float(project_ms), 3),
                    "projection": dict(projection_meta),
                    "meta": meta,
                }
                per_cam[cam] = {
                    "open": True,
                    "backend": "sparse_gl",
                    "draw": bool(valid),
                    "reason": reason,
                    "sidecar_path": str(rd.sidecar_path),
                    "path": str(item.get("path", "")),
                    "width": int(source_w),
                    "height": int(source_h),
                    "frame_id": int(fid),
                    "age_ms": round(float(age_ms), 3),
                    "event_layer_age_ms": round(float(age_ms), 3),
                    "event_layer_frame_id": int(fid),
                    "event_layer_sync_state": str(decision.get("state", "")),
                    "event_layer_sync_display_state": str(display_state),
                    "event_layer_sync_display_level": str(display_level),
                    "event_layer_sync_abs_delta_ms": decision.get("event_layer_sync_abs_delta_ms"),
                    "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                    "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                    "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                    "event_layer_hold_last_good": bool(decision.get("event_layer_hold_last_good", False)),
                    "event_layer_hold_last_good_age_ms": decision.get("event_layer_hold_last_good_age_ms", -1.0),
                    "event_layer_hold_last_good_ms": decision.get("event_layer_hold_last_good_ms", round(float(hard_hold_ms), 3)),
                    "event_layer_hold_last_good_source_frame_id": decision.get("event_layer_hold_last_good_source_frame_id", int(hold_source_frame_id)),
                    "event_layer_hold_last_good_source_sync_delta_ms": round(float(hold_source_sync_delta_ms), 3),
                    "sync_reason": str(decision.get("reason", "")),
                    "sync_mode": str(decision.get("sync_mode", "")),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0)),
                    "event_mapped_mvs_sync_index": _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1),
                    "event_mapped_mvs_sync_index_float": sync_meta.get("event_mapped_mvs_sync_index_float"),
                    "event_to_bev_sync_delta_ticks": int(event_to_bev_sync_delta_ticks_i),
                    "event_to_bev_sync_delta_ms": round(float(event_to_bev_sync_delta_ms_f), 3),
                    "event_sync_source": str(sync_meta.get("source", "")),
                    "event_sync_quality": str(sync_meta.get("event_sync_quality", "")),
                    "event_sync_ring_path": str(sync_meta.get("event_sync_ring_path", "")),
                    "event_sync_ring_age_ms": _safe_float(sync_meta.get("event_sync_ring_age_ms"), -1.0),
                    "event_sync_ring_record_count": _safe_int(sync_meta.get("event_sync_ring_record_count"), -1),
                    "event_sync_nearest_abs_delta_ms": _safe_float(sync_meta.get("event_sync_nearest_abs_delta_ms"), -1.0),
                    "homography_ready": bool(homography_ready),
                    "alpha_scale": round(float(alpha_scale), 3),
                    "target_sync_id": int(target_sync),
                    "history_selected": bool(item.get("history_selected", False)),
                    "history_source": str(meta.get("event_layer_history_source", "")),
                    "history_best_abs_delta_ms": _safe_float(item.get("history_best_abs_delta_ms"), -1.0),
                    "event_window_center_us": _safe_int(meta.get("event_window_center_us"), 0),
                    "event_window_start_us": _safe_int(meta.get("event_window_start_us"), 0),
                    "event_window_end_us": _safe_int(meta.get("event_window_end_us"), 0),
                    "mvs_frame_id": _safe_int(mvs_item.get("frame_id"), -1) if isinstance(mvs_item, dict) else -1,
                    "mvs_tick_id": _safe_int(mvs_item.get("tick_id"), -1) if isinstance(mvs_item, dict) else -1,
                    "mvs_frame_age_ms": round(float(mvs_age_ms), 3) if mvs_age_ms >= 0.0 else -1.0,
                    "nonzero_pixels": int(nonzero),
                    "max_luma": int(max_luma),
                    "sparse_point_count": int(point_count),
                    "sparse_projected_points": int(projection_meta.get("projected_points", 0)),
                    "sparse_visible_points": int(vertices.shape[0]) if valid else 0,
                    "sparse_dropped_point_count": int(sparse_dropped),
                    "sparse_entry_count": _safe_int(meta.get("event_layer_sparse_entry_count"), -1),
                    "sparse_slots": _safe_int(meta.get("event_layer_sparse_slots"), -1),
                    "sparse_write_count": _safe_int(meta.get("event_layer_sparse_write_count"), -1),
                    "sparse_overwrite_count": _safe_int(meta.get("event_layer_sparse_overwrite_count"), -1),
                    "sparse_coverage_ms": round(_safe_float(meta.get("event_layer_sparse_coverage_ms"), -1.0), 3),
                    "sparse_cache_hit": bool(meta.get("event_layer_sparse_cache_hit", False)),
                    "sparse_cache_age_ms": round(_safe_float(meta.get("event_layer_sparse_cache_age_ms"), -1.0), 3),
                    "sparse_poll_interval_ms": round(_safe_float(meta.get("event_layer_sparse_poll_interval_ms"), 0.0), 3),
                    "sparse_sidecar_path": str(meta.get("event_layer_sparse_sidecar", rd.sidecar_path)),
                    "sparse_center_hint_us": _safe_int(meta.get("event_layer_sparse_center_hint_us"), 0),
                    "sparse_candidate_scan_count": _safe_int(meta.get("event_layer_sparse_candidate_scan_count"), -1),
                    "sparse_candidate_scan_initial": _safe_int(meta.get("event_layer_sparse_candidate_scan_initial"), -1),
                    "sparse_candidate_scan_escalated": bool(meta.get("event_layer_sparse_candidate_scan_escalated", False)),
                    "sparse_candidate_scan_escalate_reason": str(meta.get("event_layer_sparse_candidate_scan_escalate_reason", "")),
                    "sparse_candidate_scan_escalate_threshold_ms": _safe_float(meta.get("event_layer_sparse_candidate_scan_escalate_threshold_ms"), -1.0),
                    "sparse_candidate_total_count": _safe_int(meta.get("event_layer_sparse_candidate_total_count"), -1),
                    "read_ms": round(float(read_ms), 3),
                    "sync_gate_ms": round(float(sync_ms), 3),
                    "project_ms": round(float(project_ms), 3),
                    "error": "",
                }
            except Exception as exc:
                errors += 1
                self.event_layer_valid[cam] = False
                self.event_layer_alpha_scale[cam] = 0.0
                sync_display_state_by_cam[str(cam)] = "error"
                sync_display_level_by_cam[str(cam)] = "ERROR"
                sync_hidden_cams.append(str(cam))
                per_cam[cam] = {
                    "open": True,
                    "backend": "sparse_gl",
                    "draw": False,
                    "reason": "error",
                    "event_layer_sync_state": "error",
                    "event_layer_sync_display_state": "error",
                    "event_layer_sync_display_level": "ERROR",
                    "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                    "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                    "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        self.event_layer_sparse_vertices = next_sparse_vertices
        self.event_layer_sparse_point_alpha = next_sparse_alpha
        display_budget_elapsed_ms = (time.perf_counter() - budget_t0) * 1000.0
        budget_stats = self._update_budget_diag(
            "event_layer_display",
            display_budget_elapsed_ms,
            budget_ms if budget_enable else 0.0,
            len(held_by_budget_cams),
        )
        if draw_cams:
            skip_reason = "drawing"
        elif errors:
            skip_reason = "errors"
        elif stale:
            skip_reason = "stale"
        elif delayed:
            skip_reason = "delayed"
        elif empty:
            skip_reason = "empty"
        elif hidden_by_sync:
            skip_reason = "hidden_by_sync"
        else:
            skip_reason = "no_sparse_readers"
        exact_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_source", "")) == "event_sync_map_ring_exact")
        interpolated_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_source", "")) == "event_sync_map_ring_interpolated")
        estimated_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and "estimated" in str(rec.get("event_sync_source", "")))
        missing_sync_count = sum(
            1
            for rec in per_cam.values()
            if isinstance(rec, dict) and (
                str(rec.get("event_sync_quality", "")).lower() == "missing"
                or str(rec.get("event_sync_source", "")).endswith("_missing")
                or str(rec.get("event_sync_source", "")) == "event_sync_map_ring_no_usable_records"
            )
        )
        self.event_layer_diag.update({
            "enabled": True,
            "backend": "sparse_gl",
            "draw_cams": list(draw_cams),
            "draw_cam_count": int(len(draw_cams)),
            "shader_draw_cams": list(draw_cams),
            "shader_draw_cam_count": int(len(draw_cams)),
            "uploaded_this_tick": 0,
            "uploaded_cams": [],
            "sparse_input_points": int(input_total),
            "sparse_projected_points": int(projected_total),
            "sparse_visible_points": int(visible_total),
            "sparse_dropped_points": int(sparse_dropped_total),
            "display_budget_enable": bool(budget_enable),
            "display_budget_ms": round(float(budget_ms), 3),
            "display_budget_elapsed_ms": round(float(display_budget_elapsed_ms), 3),
            "display_budget_order": list(display_budget_order),
            "display_budget_overrun": bool(budget_stats.get("overrun", False)),
            "display_budget_overrun_rate": float(budget_stats.get("overrun_rate", 0.0)),
            "display_budget_overrun_count": int(budget_stats.get("overrun_count", 0)),
            "display_budget_sample_count": int(budget_stats.get("sample_count", 0)),
            "display_budget_elapsed_p95_ms": float(budget_stats.get("elapsed_p95_ms", 0.0)),
            "display_budget_held_cam_count_p95": float(budget_stats.get("held_cam_count_p95", 0.0)),
            "display_budget_held_cams": list(held_by_budget_cams),
            "display_budget_held_cam_count": int(len(held_by_budget_cams)),
            "display_budget_forced_cams": list(forced_by_budget_cams),
            "display_budget_forced_cam_count": int(len(forced_by_budget_cams)),
            "display_budget_expired_hold_cams": list(budget_expired_hold_cams),
            "display_budget_expired_hold_cam_count": int(len(budget_expired_hold_cams)),
            "sparse_read_elapsed_ms": round(float(sparse_read_elapsed_ms), 3),
            "sparse_sync_gate_elapsed_ms": round(float(sparse_sync_gate_elapsed_ms), 3),
            "sparse_project_elapsed_ms": round(float(sparse_project_elapsed_ms), 3),
            "sparse_cpu_prepare_elapsed_ms": round(float(display_budget_elapsed_ms), 3),
            "display_hold_ms": round(float(hold_ms), 3),
            "sync_normal_ms": round(float(sync_normal_ms), 3),
            "sync_warn_ms": round(float(sync_warn_ms), 3),
            "sync_hard_ms": round(float(sync_hard_ms), 3),
            "hold_last_good_on_hard_outlier": bool(hard_hold_enable),
            "hold_last_good_ms": round(float(hard_hold_ms), 3),
            "sync_display_state_by_cam": dict(sync_display_state_by_cam),
            "sync_display_level_by_cam": dict(sync_display_level_by_cam),
            "sync_warn_cams": list(dict.fromkeys(sync_warn_cams)),
            "sync_warn_cam_count": int(len(dict.fromkeys(sync_warn_cams))),
            "sync_hard_outlier_cams": list(dict.fromkeys(sync_hard_outlier_cams)),
            "sync_hard_outlier_cam_count": int(len(dict.fromkeys(sync_hard_outlier_cams))),
            "sync_hold_cams": list(dict.fromkeys(sync_hold_cams)),
            "sync_hold_cam_count": int(len(dict.fromkeys(sync_hold_cams))),
            "sync_hidden_cams": list(dict.fromkeys(sync_hidden_cams)),
            "sync_hidden_cam_count": int(len(dict.fromkeys(sync_hidden_cams))),
            "stale_cam_count": int(stale),
            "delayed_cam_count": int(delayed),
            "hidden_by_sync_cam_count": int(hidden_by_sync),
            "fade_cam_count": int(fade_count),
            "empty_cam_count": int(empty),
            "error_cam_count": int(errors),
            "skip_reason": skip_reason,
            "alpha": float(getattr(self.args, "event_layer_alpha", 0.45)),
            "gain": float(getattr(self.args, "event_layer_gain", 2.4)),
            "min_value": float(getattr(self.args, "event_layer_min_value", 0.03)),
            "max_age_ms": float(max_age_ms),
            "sync_mode": _event_layer_sync_mode(getattr(self.args, "event_layer_sync_mode", "latest_guarded")),
            "available_sync_modes": list(EVENT_LAYER_SYNC_MODE_ORDER),
            "max_sync_delta_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 120.0)),
            "max_sync_delta_ticks": int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2)),
            "history_enable": bool(getattr(self.args, "event_layer_history_enable", False)),
            "history_selected_cam_count": int(sum(1 for rec in per_cam.values() if isinstance(rec, dict) and bool(rec.get("history_selected", False)))),
            "sparse_candidate_scan_initial": int(getattr(self.args, "event_layer_sparse_candidate_scan_initial", 4) or 4),
            "sparse_candidate_scan_escalated": int(getattr(self.args, "event_layer_sparse_candidate_scan_escalated", 32) or 32),
            "sparse_candidate_escalate_enable": bool(getattr(self.args, "event_layer_sparse_candidate_escalate_enable", True)),
            "sparse_candidate_escalated_cam_count": int(sum(1 for rec in per_cam.values() if isinstance(rec, dict) and bool(rec.get("sparse_candidate_scan_escalated", False)))),
            "sync_truth_mode": "history_ring",
            "sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "sync_map_ring_template": str(getattr(self.args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
            "exact_sync_cam_count": int(exact_sync_count),
            "interpolated_sync_cam_count": int(interpolated_sync_count),
            "estimated_sync_cam_count": int(estimated_sync_count),
            "missing_sync_cam_count": int(missing_sync_count),
            "accumulate_enable": False,
            "texture_scale": 0.0,
            "display_only": True,
            "judgement_sync_threshold_ms": 100.0,
            "bev_visual_sync_target_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 120.0)),
            "fade_alpha_scale": float(getattr(self.args, "event_layer_fade_alpha_scale", 0.25)),
            "color_mode": _event_layer_color_mode(getattr(self.args, "event_layer_color_mode", "camera")),
            "available_color_modes": list(EVENT_LAYER_COLOR_MODE_ORDER),
            "dilate_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
            "requested_dilate_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
            "effective_dilate_px": float(effective_dilate_px),
            "dilate_backend": "opengl_points",
            "dilate_max_pixels": int(getattr(self.args, "event_layer_dilate_max_pixels", 2500) or 0),
            "dilate_authority": "launch_profile_or_control",
            "runtime_state_event_layer_display_restore_enable": bool(getattr(self.args, "runtime_state_event_layer_display_restore_enable", False)),
            "runtime_state_loaded_path": str(getattr(self.args, "runtime_state_loaded_path", "")),
            "runtime_state_skipped_event_layer_display_keys": list(getattr(self.args, "runtime_state_skipped_event_layer_display_keys", []) or []),
            "performance_guard": dict(perf_guard),
            "max_texture_units": int(self.max_texture_units or 0),
            "transparent_background": True,
            "render_target": "bev_fbo",
        })
        self._event_layer_shadow_submit_and_diag(int(target_sync), per_cam)

    def _ensure_event_layer_texture(self, cam: str, width: int, height: int, channels: int) -> int:
        tex = int(self.event_layer_textures.get(cam, 0) or 0)
        if tex <= 0:
            tex = int(GL.glGenTextures(1))
            self.event_layer_textures[cam] = tex
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        shape = (int(width), int(height), int(channels))
        if self.event_layer_shape.get(cam) != shape:
            self.event_layer_shape[cam] = shape
            self.event_layer_last_uploaded.pop(cam, None)
            self.event_layer_last_upload_key.pop(cam, None)
        return tex

    def _event_layer_upload_formats(self, channels: int) -> Tuple[int, int, str]:
        ch = max(1, int(channels or 1))
        if ch == 1:
            return int(GL.GL_LUMINANCE), int(GL.GL_LUMINANCE), "GL_LUMINANCE"
        if ch == 4:
            return int(GL.GL_RGBA), int(getattr(GL, "GL_BGRA", GL.GL_RGBA)), "GL_BGRA"
        return int(GL.GL_RGB), int(getattr(GL, "GL_BGR", GL.GL_RGB)), "GL_BGR"

    def _event_layer_age_ms(self, rd: EventOverlayLayerReader, item: Dict[str, Any]) -> float:
        meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
        pub_us = _safe_int(meta.get("published_wall_us"), 0)
        if pub_us > 1_000_000_000_000:
            return max(0.0, (_now_us() - int(pub_us)) / 1000.0)
        try:
            st = os.stat(rd.sidecar_path)
            return max(0.0, (time.time_ns() - int(st.st_mtime_ns)) / 1e6)
        except Exception:
            return -1.0

    def _event_sync_health_path(self, cam: str) -> Path:
        root = Path(str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"))
        return root / f"event_sync_health_{cam}.json"

    def _read_event_sync_health(self, cam: str) -> Dict[str, Any]:
        path = self._event_sync_health_path(cam)
        try:
            st = path.stat()
            mtime_ns = int(st.st_mtime_ns)
            if int(self.event_sync_health_mtime_ns.get(cam, 0)) == mtime_ns:
                return dict(self.event_sync_health_cache.get(cam, {}))
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict):
                self.event_sync_health_cache[cam] = obj
                self.event_sync_health_mtime_ns[cam] = mtime_ns
                return dict(obj)
        except Exception as exc:
            cached = dict(self.event_sync_health_cache.get(cam, {}))
            cached["read_error"] = f"{type(exc).__name__}: {exc}"
            cached["path"] = str(path)
            return cached
        return {}

    def _event_sync_ring_reader(self, cam: str) -> EventSyncMapRingReader:
        rd = self.event_sync_ring_readers.get(str(cam))
        if rd is None:
            rd = EventSyncMapRingReader(
                str(cam),
                str(getattr(self.args, "sync_root", "sync_ipc") or "sync_ipc"),
                str(getattr(self.args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
            )
            self.event_sync_ring_readers[str(cam)] = rd
        return rd

    def _event_layer_score_meta(self, cam: str, meta: Dict[str, Any], target_sync: int) -> Dict[str, Any]:
        sync_meta = self._event_layer_sync_meta(cam, meta)
        mapped_float_raw = sync_meta.get("event_mapped_mvs_sync_index_float")
        mapped_float = None
        try:
            if mapped_float_raw is not None:
                mapped_float = float(mapped_float_raw)
        except Exception:
            mapped_float = None
        mapped_i = _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1)
        if mapped_float is None and mapped_i >= 0:
            mapped_float = float(mapped_i)
        if mapped_float is None or int(target_sync) < 0:
            return {
                "sync_available": False,
                "sync_meta": sync_meta,
                "event_to_bev_sync_delta_ticks": None,
                "event_to_bev_sync_delta_ms": None,
            }
        tick_ms = max(0.001, float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0) or 25.0))
        delta_ticks_f = float(mapped_float) - float(target_sync)
        delta_ms = float(delta_ticks_f) * float(tick_ms)
        return {
            "sync_available": True,
            "sync_meta": sync_meta,
            "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "event_layer_sync_timestamp_us": _safe_int(sync_meta.get("event_layer_sync_timestamp_us"), 0) if isinstance(sync_meta, dict) else 0,
            "event_to_bev_sync_delta_ticks": int(round(delta_ticks_f)),
            "event_to_bev_sync_delta_ticks_float": round(float(delta_ticks_f), 6),
            "event_to_bev_sync_delta_ms": round(float(delta_ms), 3),
        }

    def _event_layer_sync_meta(self, cam: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        source = "event_layer_sidecar"
        history_source = str(meta.get("event_layer_history_source", "") or "")
        is_history_frame = bool(
            history_source.startswith("history")
            or "event_window_center_us" in meta
            or "event_window_start_us" in meta
            or "event_window_end_us" in meta
            or "slot" in meta
        )
        mapped = _safe_int(meta.get("event_mapped_mvs_sync_index", meta.get("mapped_mvs_sync_index")), -1)
        nearest = _safe_int(meta.get("nearest_mvs_sync_index"), -1)
        local = _safe_int(meta.get("event_local_sync_index"), -1)
        sync_master = str(meta.get("event_sync_master", meta.get("sync_master", "")) or "")
        lock_state = str(meta.get("event_sync_lock_state", meta.get("lock_state", "")) or "")
        match_mode = str(meta.get("event_sync_match_mode", meta.get("match_mode", "")) or "")
        health_age_ms = -1.0
        health_path = ""
        health_error = ""
        mapped_float: Optional[float] = float(mapped) if mapped >= 0 else None
        event_ts_us = _event_layer_sync_timestamp_us(meta, _safe_int(meta.get("timestamp_us"), 0))
        if is_history_frame:
            if mapped >= 0:
                mapped_float = float(mapped) if mapped_float is None else mapped_float
                return {
                    "source": "event_layer_history_sidecar",
                    "event_sync_quality": "good",
                    "event_mapped_mvs_sync_index": int(mapped),
                    "event_mapped_mvs_sync_index_float": round(float(mapped_float), 6),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": int(event_ts_us),
                    "event_sync_anchor_mapped_mvs_sync_index": int(mapped),
                    "event_sync_anchor_event_ts_us": int(event_ts_us),
                    "event_sync_estimated_from_event_ts_us": int(event_ts_us),
                    "nearest_mvs_sync_index": int(nearest),
                    "event_local_sync_index": int(local),
                    "event_sync_master": str(sync_master),
                    "event_sync_lock_state": str(lock_state),
                    "event_sync_match_mode": str(match_mode),
                    "event_sync_health_age_ms": -1.0,
                    "event_sync_health_path": "",
                    "event_sync_health_error": "",
                }
            ring_meta = self._event_sync_ring_reader(cam).map_event_ts(event_ts_us)
            ring_meta["event_layer_sync_timestamp_policy"] = EVENT_LAYER_SYNC_TIMESTAMP_POLICY
            ring_meta["event_layer_sync_timestamp_us"] = int(event_ts_us)
            ring_meta.setdefault("event_sync_health_age_ms", -1.0)
            ring_meta.setdefault("event_sync_health_path", "")
            ring_meta.setdefault("event_sync_health_error", "")
            return ring_meta
        health = self._read_event_sync_health(cam)
        health_path = str(health.get("path") or self._event_sync_health_path(cam))
        health_error = str(health.get("read_error", "") or "")
        health_wall_us = _safe_int(health.get("wall_time_us"), 0)
        if health_wall_us > 1_000_000_000_000:
            health_age_ms = max(0.0, (_now_us() - int(health_wall_us)) / 1000.0)
        latest = health.get("latest_map", {})
        anchor_ts_us = -1
        anchor_mapped = -1
        if isinstance(latest, dict):
            nearest = _safe_int(latest.get("nearest_mvs_sync_index"), nearest)
            local = _safe_int(latest.get("event_local_sync_index"), local)
            sync_master = str(latest.get("sync_master", health.get("sync_master", sync_master)) or "")
            lock_state = str(latest.get("lock_state", health.get("lock_state", lock_state)) or "")
            match_mode = str(latest.get("match_mode", match_mode) or "")
            anchor_mapped = _safe_int(latest.get("mapped_mvs_sync_index"), -1)
            anchor_ts_us = _safe_int(
                latest.get("mvs_soft_trigger_event_ts_us", latest.get("event_edge_ts_us")),
                -1,
            )
            if mapped < 0 and anchor_mapped >= 0:
                mapped = int(anchor_mapped)
                source = "event_sync_health.latest_map"
            if mapped_float is None and anchor_mapped >= 0 and anchor_ts_us > 0 and event_ts_us > 0:
                tick_us = max(1.0, float(getattr(self.args, "mvs_sync_tick_ms_est", 25.0) or 25.0) * 1000.0)
                mapped_float = float(anchor_mapped) + (float(event_ts_us) - float(anchor_ts_us)) / tick_us
                mapped = int(round(mapped_float))
                source = "event_sync_health.estimated_from_event_ts"
        else:
            sync_master = str(health.get("sync_master", sync_master) or "")
            lock_state = str(health.get("lock_state", lock_state) or "")
            source = "event_sync_health"
        return {
            "source": str(source),
            "event_mapped_mvs_sync_index": int(mapped),
            "event_mapped_mvs_sync_index_float": None if mapped_float is None else round(float(mapped_float), 6),
            "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "event_layer_sync_timestamp_us": int(event_ts_us),
            "event_sync_anchor_mapped_mvs_sync_index": int(anchor_mapped),
            "event_sync_anchor_event_ts_us": int(anchor_ts_us),
            "event_sync_estimated_from_event_ts_us": int(event_ts_us),
            "nearest_mvs_sync_index": int(nearest),
            "event_local_sync_index": int(local),
            "event_sync_master": str(sync_master),
            "event_sync_lock_state": str(lock_state),
            "event_sync_match_mode": str(match_mode),
            "event_sync_health_age_ms": round(float(health_age_ms), 3) if health_age_ms >= 0.0 else -1.0,
            "event_sync_health_path": str(health_path or self._event_sync_health_path(cam)),
            "event_sync_health_error": str(health_error),
        }

    def _perf_stage_p95_ms(self, name: str) -> float:
        samples = self.perf_timings.get(str(name), [])
        if not samples:
            return 0.0
        try:
            arr = np.asarray(samples, dtype=np.float32)
            return float(np.percentile(arr, 95))
        except Exception:
            return 0.0

    def _event_layer_effective_dilate(self) -> Tuple[float, Dict[str, Any]]:
        requested = max(0.0, float(getattr(self.args, "event_layer_dilate_px", 16.0) or 0.0))
        enabled = bool(getattr(self.args, "event_layer_auto_degrade_enable", True))
        fps_threshold = max(1.0, float(getattr(self.args, "event_layer_auto_degrade_fps_threshold", 28.0) or 28.0))
        upload_threshold = max(0.1, float(getattr(self.args, "event_layer_auto_degrade_upload_p95_ms", 8.0) or 8.0))
        min_px = max(0.0, min(requested, float(getattr(self.args, "event_layer_auto_degrade_min_dilate_px", 4.0) or 4.0)))
        backend = self._event_layer_backend()
        upload_p95 = self._perf_stage_p95_ms("upload_event_layer_textures")
        guard_stage = "upload_event_layer_textures"
        guard_p95 = float(upload_p95)
        guard_reason = "high_upload_cost"
        if backend == "sparse_gl":
            guard_stage = "render_fbo"
            guard_p95 = self._perf_stage_p95_ms("render_fbo")
            guard_reason = "high_sparse_draw_cost"
        render_fps = float(self.render_fps or 0.0)
        reason = "normal"
        effective = requested
        if enabled and requested > 0.0:
            low_fps = render_fps > 0.0 and render_fps < fps_threshold
            high_guard_cost = guard_p95 > upload_threshold
            if backend == "sparse_gl" and not high_guard_cost:
                low_fps = False
            if low_fps or high_guard_cost:
                severe = (
                    (render_fps > 0.0 and render_fps < max(1.0, fps_threshold - 3.0))
                    or guard_p95 > upload_threshold * 2.0
                )
                effective = min_px if severe else max(min_px, requested * 0.5)
                if abs(effective - requested) < 0.001 and min_px <= 0.001:
                    effective = 0.0
                reason = "low_render_fps" if low_fps else guard_reason
                if low_fps and high_guard_cost:
                    reason = f"low_render_fps_and_{guard_reason}"
        stabilize = bool(getattr(self.args, "event_layer_dilate_stabilize_enable", True))
        prev_effective = float(getattr(self, "_event_layer_effective_dilate_px", requested) or 0.0)
        changed = False
        if stabilize and backend == "sparse_gl":
            if abs(effective - prev_effective) > 0.001:
                changed = True
                self._event_layer_dilate_last_change_us = _now_us()
            self._event_layer_effective_dilate_px = float(effective)
            self._event_layer_dilate_state = "degraded" if abs(effective - requested) > 0.001 else "normal"
        else:
            self._event_layer_effective_dilate_px = float(effective)
            self._event_layer_dilate_state = "degraded" if abs(effective - requested) > 0.001 else "normal"
        meta = {
            "auto_degrade_enable": bool(enabled),
            "dilate_stabilize_enable": bool(stabilize),
            "backend": str(backend),
            "requested_dilate_px": float(requested),
            "effective_dilate_px": float(effective),
            "state": str(getattr(self, "_event_layer_dilate_state", "normal")),
            "reason": reason,
            "changed_this_tick": bool(changed),
            "last_change_us": int(getattr(self, "_event_layer_dilate_last_change_us", 0) or 0),
            "render_fps": round(float(render_fps), 3),
            "fps_threshold": float(fps_threshold),
            "upload_p95_ms": round(float(upload_p95), 3),
            "guard_stage": str(guard_stage),
            "guard_p95_ms": round(float(guard_p95), 3),
            "upload_p95_threshold_ms": float(upload_threshold),
            "min_dilate_px": float(min_px),
            "sparse_gl_low_fps_degrade_enable": False if backend == "sparse_gl" else True,
        }
        return float(effective), meta

    def _event_layer_sync_display_thresholds(self) -> Tuple[float, float, float]:
        normal_ms = max(
            1.0,
            float(
                getattr(
                    self.args,
                    "event_layer_sync_normal_ms",
                    getattr(self.args, "event_layer_max_sync_delta_ms", 50.0),
                )
                or 50.0
            ),
        )
        warn_ms = max(
            normal_ms,
            float(getattr(self.args, "event_layer_sync_warn_ms", 80.0) or 80.0),
        )
        hard_ms = max(
            warn_ms,
            float(getattr(self.args, "event_layer_sync_hard_ms", warn_ms) or warn_ms),
        )
        return float(normal_ms), float(warn_ms), float(hard_ms)

    def _event_layer_sync_decision(
        self,
        *,
        cam: str,
        target_sync: int,
        age_ms: float,
        nonzero: int,
        max_luma: int,
        meta: Dict[str, Any],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        mode = _event_layer_sync_mode(getattr(self.args, "event_layer_sync_mode", "latest_guarded"))
        max_age_ms = max(1.0, float(getattr(self.args, "event_layer_max_age_ms", 150.0) or 150.0))
        history_frame = bool(item.get("history_selected", False)) or str(meta.get("event_layer_history_source", "")) == "history"
        if history_frame and bool(getattr(self.args, "presentation_delay_enable", False)):
            max_age_ms = max(max_age_ms, max(1500.0, float(getattr(self.args, "presentation_delay_ms", 0.0) or 0.0) + 300.0))
        max_sync_ms = max(1.0, float(getattr(self.args, "event_layer_max_sync_delta_ms", 120.0) or 120.0))
        max_sync_ticks = max(0, int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2) or 2))
        sync_normal_ms, sync_warn_ms, sync_hard_ms = self._event_layer_sync_display_thresholds()
        hard_hold_enable = bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True))
        fade_scale = _clip01(float(getattr(self.args, "event_layer_fade_alpha_scale", 0.25) or 0.25))
        has_pixels = bool(nonzero != 0 and max_luma != 0)
        score = self._event_layer_score_meta(cam, meta, target_sync)
        sync_meta = score.get("sync_meta", {}) if isinstance(score.get("sync_meta"), dict) else self._event_layer_sync_meta(cam, meta)
        mapped_sync = _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1)
        sync_delta_ticks = score.get("event_to_bev_sync_delta_ticks", None)
        if sync_delta_ticks is None:
            sync_delta_ticks = int(mapped_sync) - int(target_sync) if mapped_sync >= 0 and int(target_sync) >= 0 else None
        sync_delta_ms = score.get("event_to_bev_sync_delta_ms", None)
        try:
            sync_delta_ms_f = float(sync_delta_ms) if sync_delta_ms is not None else None
        except Exception:
            sync_delta_ms_f = None

        def _finish(
            *,
            valid: bool,
            alpha_scale: float,
            state: str,
            reason: str,
            display_state: Optional[str] = None,
            display_level: Optional[str] = None,
            hold_candidate: bool = False,
            **extra: Any,
        ) -> Dict[str, Any]:
            abs_delta_ms = abs(float(sync_delta_ms_f)) if sync_delta_ms_f is not None else None
            if display_state is None:
                display_state = state
            if display_level is None:
                display_level = "OK" if valid else "HARD"
            out: Dict[str, Any] = {
                "valid": bool(valid),
                "alpha_scale": float(alpha_scale),
                "state": str(state),
                "reason": str(reason),
                "sync_mode": mode,
                "sync_meta": sync_meta,
                "event_to_bev_sync_delta_ticks": sync_delta_ticks,
                "event_to_bev_sync_delta_ms": sync_delta_ms_f,
                "event_layer_sync_display_state": str(display_state),
                "event_layer_sync_display_level": str(display_level),
                "event_layer_sync_display_reason": str(reason),
                "event_layer_sync_abs_delta_ms": None if abs_delta_ms is None else round(float(abs_delta_ms), 3),
                "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                "event_layer_hold_last_good_candidate": bool(hold_candidate),
                "event_layer_hold_last_good_on_hard_outlier": bool(hard_hold_enable),
            }
            out.update(extra)
            return out

        if not has_pixels:
            return _finish(
                valid=False,
                alpha_scale=0.0,
                state="empty",
                reason="empty",
                display_state="empty",
                display_level="EMPTY",
            )
        if age_ms < 0.0:
            latest_valid = mode == "latest"
            return _finish(
                valid=latest_valid,
                alpha_scale=1.0 if latest_valid else 0.0,
                state="unknown_age" if latest_valid else "missing_time",
                reason="missing_published_wall_us",
                display_state="unknown_age" if latest_valid else "missing_time",
                display_level="WARN" if latest_valid else "MISSING",
            )
        within_age = bool(age_ms <= max_age_ms)
        if mode == "mvs_tick_guarded":
            if not bool(score.get("sync_available", False)) or mapped_sync < 0 or int(target_sync) < 0:
                return _finish(
                    valid=False,
                    alpha_scale=0.0,
                    state="missing_event_mvs_sync",
                    reason="missing_event_mapped_mvs_sync_index",
                    display_state="missing_event_mvs_sync",
                    display_level="MISSING",
                )
            sync_quality = str(sync_meta.get("event_sync_quality", "") or "").strip().lower()
            if sync_quality in {"missing", "unusable", "invalid"}:
                return _finish(
                    valid=False,
                    alpha_scale=0.0,
                    state="missing_event_mvs_sync",
                    reason=f"sync_quality_{sync_quality}",
                    display_state="missing_event_mvs_sync",
                    display_level="MISSING",
                )
            if not within_age:
                return _finish(
                    valid=False,
                    alpha_scale=0.0,
                    state="stale",
                    reason="event_layer_wall_age_stale",
                    display_state="stale",
                    display_level="STALE",
                    event_to_bev_sync_delta_ticks=int(sync_delta_ticks or 0),
                    event_to_bev_sync_delta_ms=None if sync_delta_ms_f is None else round(float(sync_delta_ms_f), 3),
                )
            if sync_delta_ms_f is None:
                return _finish(
                    valid=False,
                    alpha_scale=0.0,
                    state="missing_event_mvs_sync_delta_ms",
                    reason="missing_event_to_bev_sync_delta_ms",
                    display_state="missing_event_mvs_sync_delta_ms",
                    display_level="MISSING",
                    event_to_bev_sync_delta_ticks=int(sync_delta_ticks or 0),
                )
            synced_ticks = bool(abs(int(sync_delta_ticks or 0)) <= max_sync_ticks)
            abs_sync_ms = abs(float(sync_delta_ms_f))
            common = {
                "event_to_bev_sync_delta_ticks": int(sync_delta_ticks or 0),
                "event_to_bev_sync_delta_ms": round(float(sync_delta_ms_f), 3),
                "wall_age_relaxed": False,
                "wall_age_ms": round(float(age_ms), 3),
                "max_age_ms": round(float(max_age_ms), 3),
                "legacy_max_sync_delta_ms": round(float(max_sync_ms), 3),
                "legacy_max_sync_delta_ticks": int(max_sync_ticks),
                "event_layer_ticks_within_limit": bool(synced_ticks),
            }
            if abs_sync_ms <= sync_normal_ms:
                tick_reason = "within_display_normal_ms" if synced_ticks else "within_display_normal_ms_ticks_outlier"
                return _finish(
                    valid=True,
                    alpha_scale=1.0,
                    state="synced" if synced_ticks else "synced_tick_warn",
                    reason=tick_reason,
                    display_state="normal" if synced_ticks else "normal_tick_warn",
                    display_level="OK" if synced_ticks else "WARN",
                    **common,
                )
            if abs_sync_ms <= sync_warn_ms:
                return _finish(
                    valid=True,
                    alpha_scale=1.0,
                    state="warn_displayed",
                    reason="event_mvs_sync_delta_warn_display",
                    display_state="warn_displayed",
                    display_level="WARN",
                    **common,
                )
            if abs_sync_ms <= sync_hard_ms:
                return _finish(
                    valid=True,
                    alpha_scale=float(fade_scale) if fade_scale > 0.0 else 1.0,
                    state="warn_displayed",
                    reason="event_mvs_sync_delta_between_warn_and_hard_display",
                    display_state="warn_displayed",
                    display_level="WARN",
                    **common,
                )
            return _finish(
                valid=False,
                alpha_scale=0.0,
                state="hard_outlier",
                reason="event_mvs_sync_delta_hard_outlier",
                display_state="hard_outlier",
                display_level="HARD",
                hold_candidate=hard_hold_enable,
                **common,
            )
        synced = bool(age_ms <= max_sync_ms)
        if mode == "latest":
            state = "synced" if synced else ("delayed" if within_age else "stale")
            return _finish(
                valid=bool(within_age),
                alpha_scale=1.0,
                state=state,
                reason="legacy_latest_age_gate" if within_age else "stale",
                display_state="normal" if within_age else "stale",
                display_level="OK" if within_age else "STALE",
            )
        if synced:
            return _finish(
                valid=True,
                alpha_scale=1.0,
                state="synced",
                reason="within_sync_delta",
                display_state="normal",
                display_level="OK",
            )
        if mode == "fade" and within_age:
            return _finish(
                valid=True,
                alpha_scale=float(fade_scale),
                state="delayed",
                reason="fade_delayed_event_layer",
                display_state="warn_displayed",
                display_level="WARN",
            )
        state = "stale" if not within_age else "delayed"
        return _finish(
            valid=False,
            alpha_scale=0.0,
            state=state,
            reason="drop_stale" if mode == "drop_stale" else "sync_delta_too_large",
            display_state=state,
            display_level="STALE" if not within_age else "HARD",
        )

    def _upload_event_layer_textures(
        self,
        render_context: Optional[Dict[str, Dict[str, Any]]] = None,
        target_sync: int = -1,
    ) -> None:
        if self._event_layer_backend() == "sparse_gl":
            self._upload_event_layer_sparse_points(render_context=render_context, target_sync=target_sync)
            return
        if not bool(getattr(self.args, "event_layer_enable", False)):
            self.event_layer_valid = {cam: False for cam in self.cams}
            self.event_layer_alpha_scale = {cam: 0.0 for cam in self.cams}
            self.event_layer_diag.update({"enabled": False, "backend": self._event_layer_backend(), "draw_cams": [], "draw_cam_count": 0, "skip_reason": "disabled"})
            return
        if int(self.max_texture_units or 0) < 16:
            self.event_layer_valid = {cam: False for cam in self.cams}
            self.event_layer_alpha_scale = {cam: 0.0 for cam in self.cams}
            self.event_layer_diag.update({
                "enabled": True,
                "backend": "texture",
                "draw_cams": [],
                "draw_cam_count": 0,
                "skip_reason": "insufficient_texture_units",
                "max_texture_units": int(self.max_texture_units or 0),
            })
            return
        self._open_event_layer_inputs()
        max_age_ms = max(1.0, float(getattr(self.args, "event_layer_max_age_ms", 150.0) or 150.0))
        per_cam = self.event_layer_diag.setdefault("per_camera", {})
        if not isinstance(per_cam, dict):
            per_cam = {}
            self.event_layer_diag["per_camera"] = per_cam
        draw_cams: List[str] = []
        uploaded = 0
        stale = 0
        delayed = 0
        empty = 0
        errors = 0
        hidden_by_sync = 0
        fade_count = 0
        uploaded_cams: List[str] = []
        accumulated_frames_total = 0
        accumulation_truncated = 0
        history_overwrite_total = 0
        history_coverage_ms_min = -1.0
        effective_dilate_px, perf_guard = self._event_layer_effective_dilate()
        bev_render_wall_us = _now_us()
        render_context = render_context or {}
        sync_normal_ms, sync_warn_ms, sync_hard_ms = self._event_layer_sync_display_thresholds()
        hard_hold_enable = bool(getattr(self.args, "event_layer_hold_last_good_on_hard_outlier", True))
        hard_hold_ms = max(0.0, float(getattr(self.args, "event_layer_hold_last_good_ms", 250.0) or 0.0))
        sync_display_state_by_cam: Dict[str, str] = {}
        sync_display_level_by_cam: Dict[str, str] = {}
        sync_warn_cams: List[str] = []
        sync_hard_outlier_cams: List[str] = []
        sync_hidden_cams: List[str] = []
        for cam in self.cams:
            rd = self.event_layer_readers.get(cam)
            if rd is None:
                self.event_layer_valid[cam] = False
                self.event_layer_alpha_scale[cam] = 0.0
                continue
            try:
                read_t0 = self._perf_start()
                if bool(getattr(self.args, "event_layer_history_enable", False)):
                    if bool(getattr(self.args, "event_layer_accumulate_enable", True)):
                        item = rd.read_accumulated_for_target(
                            target_sync,
                            lambda meta, tgt, _cam=cam: self._event_layer_score_meta(_cam, meta, tgt),
                            float(getattr(self.args, "event_layer_accumulate_window_ms", 40.0) or 40.0),
                            int(getattr(self.args, "event_layer_accumulate_max_frames", 4) or 4),
                        )
                    else:
                        item = rd.read_for_target(target_sync, lambda meta, tgt, _cam=cam: self._event_layer_score_meta(_cam, meta, tgt))
                else:
                    item = rd.read_latest()
                read_ms = (time.perf_counter() - float(read_t0)) * 1000.0
                self._perf_end("event_layer_read", read_t0)
                if item is None:
                    self.event_layer_valid[cam] = False
                    self.event_layer_alpha_scale[cam] = 0.0
                    sync_display_state_by_cam[str(cam)] = "missing"
                    sync_display_level_by_cam[str(cam)] = "MISSING"
                    sync_hidden_cams.append(str(cam))
                    rec = per_cam.setdefault(cam, {})
                    rec.update({
                        "open": True,
                        "draw": False,
                        "reason": "no_frame",
                        "event_layer_sync_state": "missing",
                        "event_layer_sync_display_state": "missing",
                        "event_layer_sync_display_level": "MISSING",
                        "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                        "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                        "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                        "read_ms": round(float(read_ms), 3),
                    })
                    continue
                meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
                frame = np.asarray(item.get("frame"))
                fid = _safe_int(item.get("frame_id"), -1)
                source_h = int(frame.shape[0]) if frame.ndim >= 2 else 0
                source_w = int(frame.shape[1]) if frame.ndim >= 2 else 0
                texture_scale_req = max(0.25, min(1.0, float(getattr(self.args, "event_layer_texture_scale", 1.0) or 1.0)))
                downsample_t0 = self._perf_start()
                frame, downsample_meta = _downsample_event_layer_frame(frame, texture_scale_req)
                downsample_ms = (time.perf_counter() - float(downsample_t0)) * 1000.0
                self._perf_end("event_layer_downsample", downsample_t0)
                texture_scale_x = _safe_float(downsample_meta.get("actual_scale_x"), 1.0)
                texture_scale_y = _safe_float(downsample_meta.get("actual_scale_y"), 1.0)
                texture_scale_min = max(0.25, min(1.0, min(float(texture_scale_x), float(texture_scale_y))))
                dilate_meta = self.event_layer_items.get(cam, {}).get("dilate", {})
                dilate_px = float(effective_dilate_px) * float(texture_scale_min)
                dilate_max_pixels = int(getattr(self.args, "event_layer_dilate_max_pixels", 2500) or 0)
                dilate_key = f"{dilate_px:.3f}/{dilate_max_pixels}/scale{texture_scale_x:.4f}x{texture_scale_y:.4f}"
                history_entry = item.get("history_entry", {}) if isinstance(item.get("history_entry"), dict) else {}
                upload_key_parts = [
                    str(meta.get("event_layer_history_source", "latest")),
                    str(history_entry.get("slot", meta.get("slot", ""))),
                    str(history_entry.get("path", meta.get("path", ""))),
                    str(meta.get("event_window_center_us", "")),
                    str(meta.get("event_window_start_us", "")),
                    str(meta.get("event_window_end_us", "")),
                    str(item.get("accumulated_upload_key", "")),
                    str(fid),
                    str(source_w),
                    str(source_h),
                    str(downsample_meta.get("pool_step", 1)),
                    str(dilate_key),
                ]
                upload_key = "::".join(upload_key_parts)
                needs_upload = self.event_layer_last_upload_key.get(cam) != upload_key
                dilate_ms = 0.0
                if needs_upload:
                    dilate_t0 = self._perf_start()
                    frame, dilate_meta = _dilate_event_layer_frame(
                        frame,
                        dilate_px,
                        dilate_max_pixels,
                    )
                    dilate_ms = (time.perf_counter() - float(dilate_t0)) * 1000.0
                    self._perf_end("event_layer_dilate", dilate_t0)
                if frame.ndim == 2:
                    frame = frame[:, :, None]
                h, w = int(frame.shape[0]), int(frame.shape[1])
                ch = int(frame.shape[2]) if frame.ndim == 3 else 1
                tex = self._ensure_event_layer_texture(cam, w, h, ch)
                tex_upload_ms = 0.0
                if needs_upload:
                    internal, upload_format, fmt_name = self._event_layer_upload_formats(ch)
                    tex_t0 = self._perf_start()
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                    ptr = ctypes.c_void_p(int(frame.ctypes.data))
                    first_upload = self.event_layer_last_uploaded.get(cam) is None
                    if first_upload:
                        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, upload_format, GL.GL_UNSIGNED_BYTE, ptr)
                    else:
                        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, upload_format, GL.GL_UNSIGNED_BYTE, ptr)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    self.event_layer_last_uploaded[cam] = fid
                    self.event_layer_last_upload_key[cam] = upload_key
                    tex_upload_ms = (time.perf_counter() - float(tex_t0)) * 1000.0
                    self._perf_end("event_layer_texture_upload", tex_t0)
                    uploaded += 1
                    uploaded_cams.append(cam)
                else:
                    fmt_name = str(per_cam.get(cam, {}).get("upload_format", "cached"))
                age_ms = self._event_layer_age_ms(rd, item)
                nonzero = _safe_int(meta.get("overlay_nonzero_pixels"), -1)
                max_luma = _safe_int(meta.get("overlay_max_luma"), -1)
                accumulated_frame_count = _safe_int(
                    item.get("accumulated_frame_count", meta.get("event_layer_accumulated_frames")),
                    1,
                )
                accumulated_frames_total += max(1, int(accumulated_frame_count))
                if bool(meta.get("event_layer_accumulation_truncated", False)):
                    accumulation_truncated += 1
                history_overwrite = _safe_int(meta.get("event_layer_history_overwrite_count"), 0)
                if history_overwrite > 0:
                    history_overwrite_total += int(history_overwrite)
                history_coverage = _safe_float(meta.get("event_layer_history_coverage_ms"), -1.0)
                if history_coverage >= 0.0:
                    history_coverage_ms_min = history_coverage if history_coverage_ms_min < 0.0 else min(history_coverage_ms_min, history_coverage)
                sync_t0 = self._perf_start()
                decision = self._event_layer_sync_decision(
                    cam=cam,
                    target_sync=target_sync,
                    age_ms=age_ms,
                    nonzero=nonzero,
                    max_luma=max_luma,
                    meta=meta,
                    item=item,
                )
                self._perf_end("event_layer_sync_gate", sync_t0)
                valid = bool(decision.get("valid", False))
                alpha_scale = float(decision.get("alpha_scale", 0.0) or 0.0)
                homography_ready = bool(cam in self.H and self.event_layer_mvs_to_event.get(cam) is not None)
                if valid and not homography_ready:
                    valid = False
                    alpha_scale = 0.0
                    decision["state"] = "no_homography"
                    decision["reason"] = "missing_field_or_mvs_to_event_homography"
                if not valid:
                    if str(decision.get("state")) == "delayed":
                        delayed += 1
                        hidden_by_sync += 1
                        reason = str(decision.get("reason") or "delayed")
                    elif age_ms > max_age_ms:
                        stale += 1
                        reason = "stale"
                    elif nonzero == 0 or max_luma == 0:
                        empty += 1
                        reason = "empty"
                    else:
                        hidden_by_sync += 1
                        reason = str(decision.get("reason") or "invalid")
                else:
                    draw_cams.append(cam)
                    reason = "drawing" if alpha_scale >= 0.999 else "drawing_faded"
                    if alpha_scale < 0.999:
                        fade_count += 1
                        delayed += 1
                self.event_layer_valid[cam] = bool(valid)
                self.event_layer_alpha_scale[cam] = float(alpha_scale if valid else 0.0)
                mvs_item = render_context.get(cam, {}) if isinstance(render_context, dict) else {}
                mvs_ts_us = _safe_int(mvs_item.get("timestamp_us"), 0) if isinstance(mvs_item, dict) else 0
                mvs_age_ms = max(0.0, (bev_render_wall_us - int(mvs_ts_us)) / 1000.0) if mvs_ts_us > 1_000_000_000_000 else -1.0
                pub_us = _safe_int(meta.get("published_wall_us"), 0)
                event_to_bev_wall_delta_ms = max(0.0, (bev_render_wall_us - int(pub_us)) / 1000.0) if pub_us > 1_000_000_000_000 else -1.0
                event_ts_us = _event_layer_sync_timestamp_us(meta, _safe_int(item.get("timestamp_us"), 0))
                sync_meta = decision.get("sync_meta", {})
                if not isinstance(sync_meta, dict):
                    sync_meta = {}
                event_mapped_mvs_sync_index = _safe_int(sync_meta.get("event_mapped_mvs_sync_index"), -1)
                event_to_bev_sync_delta_ticks = decision.get("event_to_bev_sync_delta_ticks", None)
                if event_to_bev_sync_delta_ticks is None:
                    event_to_bev_sync_delta_ticks_i = -999999
                else:
                    event_to_bev_sync_delta_ticks_i = int(event_to_bev_sync_delta_ticks)
                event_to_bev_sync_delta_ms = decision.get("event_to_bev_sync_delta_ms", None)
                event_to_bev_sync_delta_ms_f = _safe_float(event_to_bev_sync_delta_ms, -999999.0) if event_to_bev_sync_delta_ms is not None else -999999.0
                display_state = str(decision.get("event_layer_sync_display_state", decision.get("state", "")) or "")
                display_level = str(decision.get("event_layer_sync_display_level", "OK" if valid else "HARD") or "")
                sync_display_state_by_cam[str(cam)] = display_state
                sync_display_level_by_cam[str(cam)] = display_level
                if display_level.upper() == "WARN" and str(cam) not in sync_warn_cams:
                    sync_warn_cams.append(str(cam))
                if display_state in {"hard_outlier", "hard_outlier_hold_last_good"} and str(cam) not in sync_hard_outlier_cams:
                    sync_hard_outlier_cams.append(str(cam))
                if not valid and str(cam) not in sync_hidden_cams:
                    sync_hidden_cams.append(str(cam))
                self.event_layer_items[cam] = {
                    "width": int(w),
                    "height": int(h),
                    "channels": int(ch),
                    "source_width": int(source_w),
                    "source_height": int(source_h),
                    "texture_scale_requested": float(texture_scale_req),
                    "texture_scale_x": float(texture_scale_x),
                    "texture_scale_y": float(texture_scale_y),
                    "downsample": dict(downsample_meta) if isinstance(downsample_meta, dict) else {},
                    "downsample_ms": round(float(downsample_ms), 3),
                    "frame_id": int(fid),
                    "age_ms": round(float(age_ms), 3),
                    "event_ts_us": int(event_ts_us),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": int(event_ts_us),
                    "published_wall_us": int(pub_us),
                    "bev_render_wall_us": int(bev_render_wall_us),
                    "event_to_bev_wall_delta_ms": round(float(event_to_bev_wall_delta_ms), 3) if event_to_bev_wall_delta_ms >= 0.0 else -1.0,
                    "nonzero_pixels": int(nonzero),
                    "max_luma": int(max_luma),
                    "sync_state": str(decision.get("state", "")),
                    "sync_reason": str(decision.get("reason", "")),
                    "sync_mode": str(decision.get("sync_mode", "")),
                    "sync_display_state": str(display_state),
                    "sync_display_level": str(display_level),
                    "event_layer_sync_display_state": str(display_state),
                    "event_layer_sync_display_level": str(display_level),
                    "event_layer_sync_abs_delta_ms": decision.get("event_layer_sync_abs_delta_ms"),
                    "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                    "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                    "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                    "event_layer_hold_last_good_on_hard_outlier": bool(hard_hold_enable),
                    "event_layer_hold_last_good_ms": round(float(hard_hold_ms), 3),
                    "wall_age_relaxed": bool(decision.get("wall_age_relaxed", False)),
                    "event_mapped_mvs_sync_index": int(event_mapped_mvs_sync_index),
                    "event_to_bev_sync_delta_ticks": int(event_to_bev_sync_delta_ticks_i),
                    "event_to_bev_sync_delta_ms": round(float(event_to_bev_sync_delta_ms_f), 3),
                    "event_sync": dict(sync_meta),
                    "history_selected": bool(item.get("history_selected", False)),
                    "history_source": str(meta.get("event_layer_history_source", "")),
                    "history_best_abs_delta_ms": _safe_float(item.get("history_best_abs_delta_ms"), -1.0),
                    "accumulated_frame_count": int(accumulated_frame_count),
                    "accumulated_upload_key": str(item.get("accumulated_upload_key", upload_key)),
                    "upload_key": str(upload_key),
                    "alpha_scale": round(float(alpha_scale), 3),
                    "dilate": dict(dilate_meta) if isinstance(dilate_meta, dict) else {},
                    "dilate_key": str(dilate_key),
                    "meta": meta,
                }
                per_cam[cam] = {
                    "open": True,
                    "draw": bool(valid),
                    "reason": reason,
                    "path": str(rd.path),
                    "sidecar_path": str(rd.sidecar_path),
                    "width": int(w),
                    "height": int(h),
                    "channels": int(ch),
                    "source_width": int(source_w),
                    "source_height": int(source_h),
                    "texture_scale_requested": float(texture_scale_req),
                    "texture_scale_x": round(float(texture_scale_x), 6),
                    "texture_scale_y": round(float(texture_scale_y), 6),
                    "downsample_backend": str(downsample_meta.get("backend", "")) if isinstance(downsample_meta, dict) else "",
                    "downsample_pool_step": _safe_int(downsample_meta.get("pool_step"), 1) if isinstance(downsample_meta, dict) else 1,
                    "downsample_ms": round(float(downsample_ms), 3),
                    "frame_id": int(fid),
                    "age_ms": round(float(age_ms), 3),
                    "event_layer_age_ms": round(float(age_ms), 3),
                    "event_layer_frame_id": int(fid),
                    "event_ts_us": int(event_ts_us),
                    "event_layer_sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
                    "event_layer_sync_timestamp_us": int(event_ts_us),
                    "published_wall_us": int(pub_us),
                    "bev_render_wall_us": int(bev_render_wall_us),
                    "event_to_bev_wall_delta_ms": round(float(event_to_bev_wall_delta_ms), 3) if event_to_bev_wall_delta_ms >= 0.0 else -1.0,
                    "event_layer_sync_state": str(decision.get("state", "")),
                    "event_layer_sync_display_state": str(display_state),
                    "event_layer_sync_display_level": str(display_level),
                    "event_layer_sync_abs_delta_ms": decision.get("event_layer_sync_abs_delta_ms"),
                    "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                    "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                    "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                    "event_layer_hold_last_good_on_hard_outlier": bool(hard_hold_enable),
                    "event_layer_hold_last_good_ms": round(float(hard_hold_ms), 3),
                    "sync_reason": str(decision.get("reason", "")),
                    "sync_mode": str(decision.get("sync_mode", "")),
                    "wall_age_relaxed": bool(decision.get("wall_age_relaxed", False)),
                    "event_mapped_mvs_sync_index": int(event_mapped_mvs_sync_index),
                    "event_mapped_mvs_sync_index_float": sync_meta.get("event_mapped_mvs_sync_index_float"),
                    "event_local_sync_index": _safe_int(sync_meta.get("event_local_sync_index"), -1),
                    "nearest_mvs_sync_index": _safe_int(sync_meta.get("nearest_mvs_sync_index"), -1),
                    "event_to_bev_sync_delta_ticks": int(event_to_bev_sync_delta_ticks_i),
                    "event_to_bev_sync_delta_ms": round(float(event_to_bev_sync_delta_ms_f), 3),
                    "event_sync_source": str(sync_meta.get("source", "")),
                    "event_sync_quality": str(sync_meta.get("event_sync_quality", "")),
                    "event_sync_ring_path": str(sync_meta.get("event_sync_ring_path", "")),
                    "event_sync_ring_age_ms": _safe_float(sync_meta.get("event_sync_ring_age_ms"), -1.0),
                    "event_sync_ring_record_count": _safe_int(sync_meta.get("event_sync_ring_record_count"), -1),
                    "event_sync_ring_records_written": _safe_int(sync_meta.get("event_sync_ring_records_written"), -1),
                    "event_sync_nearest_abs_delta_ms": _safe_float(sync_meta.get("event_sync_nearest_abs_delta_ms"), -1.0),
                    "event_sync_ring_error": str(sync_meta.get("event_sync_ring_error", "")),
                    "event_sync_master": str(sync_meta.get("event_sync_master", "")),
                    "event_sync_lock_state": str(sync_meta.get("event_sync_lock_state", "")),
                    "event_sync_match_mode": str(sync_meta.get("event_sync_match_mode", "")),
                    "event_sync_health_age_ms": _safe_float(sync_meta.get("event_sync_health_age_ms"), -1.0),
                    "event_sync_health_path": str(sync_meta.get("event_sync_health_path", "")),
                    "event_sync_health_error": str(sync_meta.get("event_sync_health_error", "")),
                    "homography_ready": bool(homography_ready),
                    "alpha_scale": round(float(alpha_scale), 3),
                    "new_frame": bool(item.get("new_frame", False)),
                    "target_sync_id": int(target_sync),
                    "history_selected": bool(item.get("history_selected", False)),
                    "history_source": str(meta.get("event_layer_history_source", "")),
                    "history_entry_count": _safe_int(meta.get("event_layer_history_entry_count"), -1),
                    "history_slots": _safe_int(meta.get("event_layer_history_slots"), -1),
                    "history_write_count": _safe_int(meta.get("event_layer_history_write_count"), -1),
                    "history_overwrite_count": _safe_int(meta.get("event_layer_history_overwrite_count"), -1),
                    "history_coverage_ms": round(_safe_float(meta.get("event_layer_history_coverage_ms"), -1.0), 3),
                    "history_cache_hit": bool(meta.get("event_layer_history_cache_hit", False)),
                    "history_cache_age_ms": round(_safe_float(meta.get("event_layer_history_cache_age_ms"), -1.0), 3),
                    "history_poll_interval_ms": round(_safe_float(meta.get("event_layer_history_poll_interval_ms"), 0.0), 3),
                    "history_reload_count": _safe_int(meta.get("event_layer_history_reload_count"), -1),
                    "history_cache_hits": _safe_int(meta.get("event_layer_history_cache_hits"), -1),
                    "history_best_abs_delta_ms": _safe_float(item.get("history_best_abs_delta_ms"), -1.0),
                    "history_sidecar_path": str(meta.get("event_layer_history_sidecar", "")),
                    "accumulated_frame_count": int(accumulated_frame_count),
                    "accumulated_frame_ids": list(meta.get("event_layer_accumulated_frame_ids", [])) if isinstance(meta.get("event_layer_accumulated_frame_ids", []), list) else [],
                    "accumulated_slots": list(meta.get("event_layer_accumulated_slots", [])) if isinstance(meta.get("event_layer_accumulated_slots", []), list) else [],
                    "accumulation_window_ms": _safe_float(meta.get("event_layer_accumulation_window_ms"), -1.0),
                    "accumulation_truncated": bool(meta.get("event_layer_accumulation_truncated", False)),
                    "upload_key": str(upload_key),
                    "texture_uploaded": bool(needs_upload),
                    "event_window_center_us": _safe_int(meta.get("event_window_center_us"), 0),
                    "event_window_start_us": _safe_int(meta.get("event_window_start_us"), 0),
                    "event_window_end_us": _safe_int(meta.get("event_window_end_us"), 0),
                    "mvs_frame_id": _safe_int(mvs_item.get("frame_id"), -1) if isinstance(mvs_item, dict) else -1,
                    "mvs_tick_id": _safe_int(mvs_item.get("tick_id"), -1) if isinstance(mvs_item, dict) else -1,
                    "mvs_frame_age_ms": round(float(mvs_age_ms), 3) if mvs_age_ms >= 0.0 else -1.0,
                    "nonzero_pixels": int(nonzero),
                    "max_luma": int(max_luma),
                    "dilate_px": float((dilate_meta or {}).get("effective_px", 0.0)) if isinstance(dilate_meta, dict) else 0.0,
                    "dilate_requested_px": float((dilate_meta or {}).get("requested_px", effective_dilate_px)) if isinstance(dilate_meta, dict) else float(effective_dilate_px),
                    "dilate_config_requested_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
                    "dilate_backend": str((dilate_meta or {}).get("backend", "cpu_sparse")) if isinstance(dilate_meta, dict) else "cpu_sparse",
                    "dilate_skipped_dense": bool((dilate_meta or {}).get("skipped_dense", False)) if isinstance(dilate_meta, dict) else False,
                    "display_nonzero_pixels": int((dilate_meta or {}).get("display_nonzero_pixels", nonzero)) if isinstance(dilate_meta, dict) else int(nonzero),
                    "upload_format": fmt_name,
                    "read_ms": round(float(read_ms), 3),
                    "dilate_ms": round(float(dilate_ms), 3),
                    "tex_upload_ms": round(float(tex_upload_ms), 3),
                    "error": "",
                }
            except Exception as exc:
                errors += 1
                self.event_layer_valid[cam] = False
                self.event_layer_alpha_scale[cam] = 0.0
                sync_display_state_by_cam[str(cam)] = "error"
                sync_display_level_by_cam[str(cam)] = "ERROR"
                sync_hidden_cams.append(str(cam))
                per_cam[cam] = {
                    "open": True,
                    "draw": False,
                    "reason": "error",
                    "event_layer_sync_state": "error",
                    "event_layer_sync_display_state": "error",
                    "event_layer_sync_display_level": "ERROR",
                    "event_layer_sync_normal_ms": round(float(sync_normal_ms), 3),
                    "event_layer_sync_warn_ms": round(float(sync_warn_ms), 3),
                    "event_layer_sync_hard_ms": round(float(sync_hard_ms), 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if draw_cams:
            skip_reason = "drawing"
        elif errors:
            skip_reason = "errors"
        elif stale:
            skip_reason = "stale"
        elif delayed:
            skip_reason = "delayed"
        elif empty:
            skip_reason = "empty"
        elif hidden_by_sync:
            skip_reason = "hidden_by_sync"
        else:
            skip_reason = "no_open_readers"
        dilate_backends = sorted({
            str(rec.get("dilate_backend", ""))
            for rec in per_cam.values()
            if isinstance(rec, dict) and str(rec.get("dilate_backend", "")).strip()
        })
        history_selected = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and bool(rec.get("history_selected", False)))
        history_fallback = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("history_source", "")) == "latest_fallback")
        history_cache_hit_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and bool(rec.get("history_cache_hit", False)))
        exact_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_source", "")) == "event_sync_map_ring_exact")
        interpolated_sync_count = sum(1 for rec in per_cam.values() if isinstance(rec, dict) and str(rec.get("event_sync_source", "")) == "event_sync_map_ring_interpolated")
        estimated_sync_count = sum(
            1
            for rec in per_cam.values()
            if isinstance(rec, dict) and (
                "estimated" in str(rec.get("event_sync_source", ""))
                or str(rec.get("event_sync_source", "")) == "event_sync_health.latest_map"
            )
        )
        missing_sync_count = sum(
            1
            for rec in per_cam.values()
            if isinstance(rec, dict) and (
                str(rec.get("event_sync_quality", "")).lower() == "missing"
                or str(rec.get("event_sync_source", "")).endswith("_missing")
                or str(rec.get("event_sync_source", "")) == "event_sync_map_ring_no_usable_records"
            )
        )
        history_cache_age_values = [
            _safe_float(rec.get("history_cache_age_ms"), -1.0)
            for rec in per_cam.values()
            if isinstance(rec, dict) and _safe_float(rec.get("history_cache_age_ms"), -1.0) >= 0.0
        ]
        self.event_layer_diag.update({
            "enabled": True,
            "backend": "texture",
            "draw_cams": list(draw_cams),
            "draw_cam_count": int(len(draw_cams)),
            "uploaded_this_tick": int(uploaded),
            "uploaded_cams": list(uploaded_cams),
            "accumulated_event_frames": int(accumulated_frames_total),
            "accumulation_truncated_cam_count": int(accumulation_truncated),
            "event_frame_dropped_count": 0,
            "event_history_overwrite_count": int(history_overwrite_total),
            "event_history_coverage_ms_min": round(float(history_coverage_ms_min), 3) if history_coverage_ms_min >= 0.0 else -1.0,
            "stale_cam_count": int(stale),
            "delayed_cam_count": int(delayed),
            "hidden_by_sync_cam_count": int(hidden_by_sync),
            "fade_cam_count": int(fade_count),
            "empty_cam_count": int(empty),
            "error_cam_count": int(errors),
            "skip_reason": skip_reason,
            "alpha": float(getattr(self.args, "event_layer_alpha", 0.45)),
            "gain": float(getattr(self.args, "event_layer_gain", 2.4)),
            "min_value": float(getattr(self.args, "event_layer_min_value", 0.03)),
            "max_age_ms": float(max_age_ms),
            "sync_mode": _event_layer_sync_mode(getattr(self.args, "event_layer_sync_mode", "latest_guarded")),
            "available_sync_modes": list(EVENT_LAYER_SYNC_MODE_ORDER),
            "max_sync_delta_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 120.0)),
            "max_sync_delta_ticks": int(getattr(self.args, "event_layer_max_sync_delta_ticks", 2)),
            "sync_normal_ms": round(float(sync_normal_ms), 3),
            "sync_warn_ms": round(float(sync_warn_ms), 3),
            "sync_hard_ms": round(float(sync_hard_ms), 3),
            "hold_last_good_on_hard_outlier": bool(hard_hold_enable),
            "hold_last_good_ms": round(float(hard_hold_ms), 3),
            "sync_display_state_by_cam": dict(sync_display_state_by_cam),
            "sync_display_level_by_cam": dict(sync_display_level_by_cam),
            "sync_warn_cams": list(dict.fromkeys(sync_warn_cams)),
            "sync_warn_cam_count": int(len(dict.fromkeys(sync_warn_cams))),
            "sync_hard_outlier_cams": list(dict.fromkeys(sync_hard_outlier_cams)),
            "sync_hard_outlier_cam_count": int(len(dict.fromkeys(sync_hard_outlier_cams))),
            "sync_hold_cams": [],
            "sync_hold_cam_count": 0,
            "sync_hidden_cams": list(dict.fromkeys(sync_hidden_cams)),
            "sync_hidden_cam_count": int(len(dict.fromkeys(sync_hidden_cams))),
            "history_enable": bool(getattr(self.args, "event_layer_history_enable", False)),
            "history_selected_cam_count": int(history_selected),
            "history_latest_fallback_cam_count": int(history_fallback),
            "history_cache_hit_cam_count": int(history_cache_hit_count),
            "history_cache_age_ms_max": round(float(max(history_cache_age_values)), 3) if history_cache_age_values else -1.0,
            "history_poll_interval_ms": float(getattr(self.args, "event_layer_history_poll_interval_ms", 0.0) or 0.0),
            "sync_truth_mode": "history_ring",
            "sync_timestamp_policy": EVENT_LAYER_SYNC_TIMESTAMP_POLICY,
            "sync_map_ring_template": str(getattr(self.args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json"),
            "exact_sync_cam_count": int(exact_sync_count),
            "interpolated_sync_cam_count": int(interpolated_sync_count),
            "estimated_sync_cam_count": int(estimated_sync_count),
            "missing_sync_cam_count": int(missing_sync_count),
            "accumulate_enable": bool(getattr(self.args, "event_layer_accumulate_enable", True)),
            "accumulate_window_ms": float(getattr(self.args, "event_layer_accumulate_window_ms", 40.0) or 40.0),
            "accumulate_max_frames": int(getattr(self.args, "event_layer_accumulate_max_frames", 4) or 4),
            "texture_scale": float(getattr(self.args, "event_layer_texture_scale", 1.0) or 1.0),
            "display_only": True,
            "judgement_sync_threshold_ms": 100.0,
            "bev_visual_sync_target_ms": float(getattr(self.args, "event_layer_max_sync_delta_ms", 120.0)),
            "fade_alpha_scale": float(getattr(self.args, "event_layer_fade_alpha_scale", 0.25)),
            "color_mode": _event_layer_color_mode(getattr(self.args, "event_layer_color_mode", "camera")),
            "available_color_modes": list(EVENT_LAYER_COLOR_MODE_ORDER),
            "dilate_px": float(getattr(self.args, "event_layer_dilate_px", 16.0)),
            "effective_dilate_px": float(effective_dilate_px),
            "dilate_backend": ",".join(dilate_backends) if dilate_backends else ("cv2_dilate" if _cv2 is not None else "cpu_sparse_upload"),
            "dilate_max_pixels": int(getattr(self.args, "event_layer_dilate_max_pixels", 2500) or 0),
            "dilate_skipped_dense_cam_count": int(sum(1 for rec in per_cam.values() if isinstance(rec, dict) and bool(rec.get("dilate_skipped_dense", False)))),
            "performance_guard": dict(perf_guard),
            "max_texture_units": int(self.max_texture_units or 0),
            "transparent_background": True,
            "render_target": "bev_fbo",
        })

    def _draw_sparse_event_layer(self) -> None:
        if self._event_layer_backend() != "sparse_gl" or not bool(getattr(self.args, "event_layer_enable", False)):
            return
        vertices_by_cam = dict(getattr(self, "event_layer_sparse_vertices", {}) or {})
        if not vertices_by_cam:
            self.event_layer_diag["sparse_gl_draw_points"] = 0
            return
        point_size = max(1.0, float(self.event_layer_diag.get("effective_dilate_px", getattr(self.args, "event_layer_dilate_px", 16.0)) or 1.0))
        point_size = min(128.0, point_size)
        draw_points = 0
        draw_cams: List[str] = []
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        try:
            for unit in range(max(1, min(16, int(self.max_texture_units or 1)))):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glActiveTexture(GL.GL_TEXTURE0)
        except Exception:
            pass
        GL.glPointSize(float(point_size))
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        try:
            for cam in self.cams:
                if not bool(self.event_layer_valid.get(cam, False)):
                    continue
                vertices = vertices_by_cam.get(cam)
                if vertices is None:
                    continue
                arr = np.asarray(vertices, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] <= 0:
                    continue
                rgba = self._event_layer_color_rgba(cam, float(self.event_layer_alpha_scale.get(cam, 1.0)))
                if rgba[3] <= 0.0:
                    continue
                GL.glColor4f(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
                GL.glVertexPointer(2, GL.GL_FLOAT, 0, arr)
                GL.glDrawArrays(GL.GL_POINTS, 0, int(arr.shape[0]))
                draw_points += int(arr.shape[0])
                draw_cams.append(cam)
        finally:
            try:
                GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            except Exception:
                pass
            GL.glDisable(GL.GL_BLEND)
            GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        self.event_layer_diag["sparse_gl_draw_points"] = int(draw_points)
        self.event_layer_diag["sparse_gl_draw_cams"] = list(draw_cams)
        self.event_layer_diag["sparse_gl_point_size_px"] = float(point_size)

    def _quad(self) -> None:
        GL.glBegin(GL.GL_QUADS)
        GL.glMultiTexCoord2f(GL.GL_TEXTURE0, 0.0, 0.0); GL.glVertex2f(-1.0, -1.0)
        GL.glMultiTexCoord2f(GL.GL_TEXTURE0, 1.0, 0.0); GL.glVertex2f(1.0, -1.0)
        GL.glMultiTexCoord2f(GL.GL_TEXTURE0, 1.0, 1.0); GL.glVertex2f(1.0, 1.0)
        GL.glMultiTexCoord2f(GL.GL_TEXTURE0, 0.0, 1.0); GL.glVertex2f(-1.0, 1.0)
        GL.glEnd()

    def _reset_draw_state(self) -> None:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def _qt_default_fbo(self) -> int:
        try:
            return int(self.defaultFramebufferObject())
        except Exception:
            return 0

    def _drain_gl_errors(self) -> int:
        last = int(getattr(GL, "GL_NO_ERROR", 0))
        try:
            for _ in range(16):
                err = int(GL.glGetError())
                if err == int(GL.GL_NO_ERROR):
                    break
                last = err
        except Exception:
            pass
        return last

    def _display_to_qt_fbo(self, qt_fbo: int, view_w: int, view_h: int) -> Dict[str, Any]:
        diag: Dict[str, Any] = {"method": "blit", "blit_ok": False, "fallback_ok": False}
        self._drain_gl_errors()
        try:
            self._reset_draw_state()
            GL.glUseProgram(0)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(qt_fbo))
            GL.glViewport(0, 0, int(view_w), int(view_h))
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBlitFramebuffer(
                0, 0, int(self.out_w), int(self.out_h),
                0, 0, int(view_w), int(view_h),
                GL.GL_COLOR_BUFFER_BIT,
                GL.GL_LINEAR,
            )
            err = int(GL.glGetError())
            diag["blit_gl_error"] = err
            if err != int(GL.GL_NO_ERROR):
                raise RuntimeError(f"glBlitFramebuffer failed: 0x{err:x}")
            diag["blit_ok"] = True
            return diag
        except Exception as exc:
            diag["method"] = "shader_fallback"
            diag["blit_error"] = f"{type(exc).__name__}: {exc}"
            self._drain_gl_errors()
            try:
                self._reset_draw_state()
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(qt_fbo))
                GL.glViewport(0, 0, int(view_w), int(view_h))
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                GL.glUseProgram(self.display_program)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.fbo_tex)
                GL.glUniform1i(GL.glGetUniformLocation(self.display_program, "u_tex"), 0)
                self._quad()
                GL.glUseProgram(0)
                err = int(GL.glGetError())
                diag["fallback_gl_error"] = err
                diag["fallback_ok"] = err == int(GL.GL_NO_ERROR)
            except Exception as fb_exc:
                diag["fallback_error"] = f"{type(fb_exc).__name__}: {fb_exc}"
            return diag

    def _set_mat3_rows(self, prefix: str, idx: int, H: np.ndarray) -> None:
        arr = np.asarray(H, dtype=np.float32).reshape(3, 3)
        for row in range(3):
            loc = GL.glGetUniformLocation(self.program, f"{prefix}{row}[{idx}]")
            GL.glUniform3f(loc, float(arr[row, 0]), float(arr[row, 1]), float(arr[row, 2]))

    def _set_homography_rows(self, idx: int, H: np.ndarray) -> None:
        self._set_mat3_rows("u_field_to_img_r", idx, H)

    def _set_mvs_to_event_rows(self, idx: int, H: np.ndarray) -> None:
        self._set_mat3_rows("u_mvs_to_event_r", idx, H)

    def _field_window(self) -> Dict[str, Any]:
        ox = float(self.args.field_origin_x_m)
        oy = float(self.args.field_origin_y_m)
        w = float(self.args.field_width_m)
        h = float(self.args.field_height_m)
        return {
            "origin_m": [ox, oy],
            "size_m": [w, h],
            "x_min": ox,
            "x_max": ox + w,
            "y_min": oy,
            "y_max": oy + h,
        }

    def _calib_field_range(self) -> Dict[str, Any]:
        ranges = [m.get("field_range_m", {}) for m in self.calib.values() if isinstance(m.get("field_range_m", {}), dict)]
        ranges = [r for r in ranges if {"x_min", "x_max", "y_min", "y_max"}.issubset(r.keys())]
        if not ranges:
            return {}
        xmin = min(float(r["x_min"]) for r in ranges)
        xmax = max(float(r["x_max"]) for r in ranges)
        ymin = min(float(r["y_min"]) for r in ranges)
        ymax = max(float(r["y_max"]) for r in ranges)
        return {
            "x_min": xmin,
            "x_max": xmax,
            "y_min": ymin,
            "y_max": ymax,
            "width": xmax - xmin,
            "height": ymax - ymin,
        }

    def _compute_coverage_diag(self, selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        nx = max(8, int(self.args.coverage_grid_x))
        ny = max(8, int(self.args.coverage_grid_y))
        ox = float(self.args.field_origin_x_m)
        oy = float(self.args.field_origin_y_m)
        fw = float(self.args.field_width_m)
        fh = float(self.args.field_height_m)
        total = nx * ny
        any_mask = np.zeros((ny, nx), dtype=bool)
        per_cam: Dict[str, Any] = {}
        xs_field = ox + np.linspace(0.0, 1.0, nx, dtype=np.float64) * fw
        ys_field = oy + (1.0 - np.linspace(0.0, 1.0, ny, dtype=np.float64)) * fh
        grid_x, grid_y = np.meshgrid(xs_field, ys_field)
        points = np.vstack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.ones((total,), dtype=np.float64),
        ])
        for cam, item in selected.items():
            H = self.H.get(cam)
            if H is None:
                continue
            Hf = np.asarray(H, dtype=np.float64).reshape(3, 3)
            img_w = max(1.0, float(item.get("width", 1)))
            img_h = max(1.0, float(item.get("height", 1)))
            p = Hf @ points
            z = p[2, :]
            finite = np.isfinite(z) & (np.abs(z) >= 1e-9)
            px = np.full((total,), np.nan, dtype=np.float64)
            py = np.full((total,), np.nan, dtype=np.float64)
            px[finite] = p[0, finite] / z[finite]
            py[finite] = p[1, finite] / z[finite]
            valid = (
                finite
                & np.isfinite(px)
                & np.isfinite(py)
                & (px >= 0.0)
                & (px <= img_w)
                & (py >= 0.0)
                & (py <= img_h)
            ).reshape(ny, nx)
            any_mask |= valid
            ys, xs = np.where(valid)
            entry: Dict[str, Any] = {"valid_sample_ratio": float(np.count_nonzero(valid) / float(total))}
            if len(xs):
                entry["bbox_grid_xy"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
            per_cam[cam] = entry
        ys, xs = np.where(any_mask)
        out: Dict[str, Any] = {
            "grid": [int(nx), int(ny)],
            "any_camera_valid_ratio": float(np.count_nonzero(any_mask) / float(total)),
            "per_camera": per_cam,
            "field_window_m": self._field_window(),
            "field_range_from_calib_m": self._calib_field_range(),
        }
        if len(xs):
            out["any_camera_bbox_grid_xy"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
        return out

    def _render_fbo(self, target_sync: int, selected: Dict[str, Dict[str, Any]]) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.out_w, self.out_h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self.program)
        GL.glUniform2f(GL.glGetUniformLocation(self.program, "u_field_origin_m"), float(self.args.field_origin_x_m), float(self.args.field_origin_y_m))
        GL.glUniform2f(GL.glGetUniformLocation(self.program, "u_field_size_m"), float(self.args.field_width_m), float(self.args.field_height_m))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_feather_power"), float(self.args.feather_power))
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "u_debug_mode"), int(self.args.debug_mode_code))
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "u_texture_y_flip"), 1 if bool(self.args.texture_y_flip) else 0)
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_gain"), float(self.args.gain))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_black_level"), float(self.args.black_level))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_gamma"), float(self.args.gamma))
        GL.glUniform3f(
            GL.glGetUniformLocation(self.program, "u_color_gain"),
            float(self.args.color_gain_r),
            float(self.args.color_gain_g),
            float(self.args.color_gain_b),
        )
        GL.glUniform3f(
            GL.glGetUniformLocation(self.program, "u_turf_target_rgb"),
            float(self.turf_target_rgb[0]),
            float(self.turf_target_rgb[1]),
            float(self.turf_target_rgb[2]),
        )
        GL.glUniform1f(
            GL.glGetUniformLocation(self.program, "u_turf_match_strength"),
            float(self.args.turf_match_strength) if bool(self.args.turf_match_enable) else 0.0,
        )
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "u_grid_enable"), 1 if bool(self.args.grid_enable) else 0)
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "u_field_transform_mode"), _transform_mode_code(self.args.field_transform_mode))
        event_layer_enabled = (
            self._event_layer_backend() == "texture"
            and bool(getattr(self.args, "event_layer_enable", False))
            and int(self.max_texture_units or 0) >= 16
        )
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "u_event_layer_enable"), 1 if event_layer_enabled else 0)
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_event_layer_alpha"), _clip01(float(getattr(self.args, "event_layer_alpha", 0.45) or 0.0)))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_event_layer_gain"), max(0.01, float(getattr(self.args, "event_layer_gain", 2.4) or 2.4)))
        GL.glUniform1f(GL.glGetUniformLocation(self.program, "u_event_layer_min_value"), max(0.0, min(0.95, float(getattr(self.args, "event_layer_min_value", 0.03) or 0.03))))
        GL.glUniform1i(GL.glGetUniformLocation(self.program, "u_event_layer_color_mode"), _event_layer_color_mode_code(getattr(self.args, "event_layer_color_mode", "camera")))
        for idx in range(8):
            cam = self.cams[idx] if idx < len(self.cams) else ""
            tex = int(self.textures.get(cam, 0) or 0)
            GL.glActiveTexture(GL.GL_TEXTURE0 + idx)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex if tex > 0 else 0)
            GL.glUniform1i(GL.glGetUniformLocation(self.program, f"u_tex{idx}"), idx)
            gain = self.cam_rgb_gain.get(cam, np.ones((3,), dtype=np.float32))
            GL.glUniform3f(GL.glGetUniformLocation(self.program, f"u_cam_rgb_gain[{idx}]"), float(gain[0]), float(gain[1]), float(gain[2]))
            valid = 1 if cam in selected and cam in self.H and tex > 0 else 0
            GL.glUniform1i(GL.glGetUniformLocation(self.program, f"u_valid[{idx}]"), int(valid))
            if valid:
                item = selected[cam]
                GL.glUniform2f(GL.glGetUniformLocation(self.program, f"u_img_size[{idx}]"), float(item["width"]), float(item["height"]))
                self._set_homography_rows(idx, self.H[cam])
                bayer_code = {"RG": 0, "BG": 1, "GR": 2, "GB": 3}.get(str(item.get("bayer_pattern", "BG")).upper(), 1)
                GL.glUniform1i(GL.glGetUniformLocation(self.program, f"u_bayer_code[{idx}]"), int(bayer_code))
            else:
                GL.glUniform2f(GL.glGetUniformLocation(self.program, f"u_img_size[{idx}]"), 1.0, 1.0)
                self._set_homography_rows(idx, self.H[cam] if cam in self.H else np.eye(3, dtype=np.float32))
                GL.glUniform1i(GL.glGetUniformLocation(self.program, f"u_bayer_code[{idx}]"), 1)
        draw_cams: List[str] = []
        missing_transform_cams: List[str] = []
        for idx in range(8):
            cam = self.cams[idx] if idx < len(self.cams) else ""
            tex = int(self.event_layer_textures.get(cam, 0) or 0)
            GL.glActiveTexture(GL.GL_TEXTURE0 + 8 + idx)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex if tex > 0 else 0)
            GL.glUniform1i(GL.glGetUniformLocation(self.program, f"u_event_tex{idx}"), 8 + idx)
            GL.glUniform1f(
                GL.glGetUniformLocation(self.program, f"u_event_alpha_scale[{idx}]"),
                _clip01(float(self.event_layer_alpha_scale.get(cam, 1.0) if cam else 0.0)),
            )
            item = self.event_layer_items.get(cam, {})
            w = max(1.0, float(item.get("width", 1.0)))
            h = max(1.0, float(item.get("height", 1.0)))
            h_mvs_to_event = self.event_layer_mvs_to_event.get(cam)
            if h_mvs_to_event is not None:
                event_h = np.asarray(h_mvs_to_event, dtype=np.float32)
                sx = max(0.25, min(1.0, _safe_float(item.get("texture_scale_x"), 1.0)))
                sy = max(0.25, min(1.0, _safe_float(item.get("texture_scale_y"), 1.0)))
                if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                    event_h = event_h.copy()
                    event_h[0, :] *= float(sx)
                    event_h[1, :] *= float(sy)
                self._set_mvs_to_event_rows(idx, event_h)
            else:
                self._set_mvs_to_event_rows(idx, np.eye(3, dtype=np.float32))
            valid = 1 if event_layer_enabled and tex > 0 and bool(self.event_layer_valid.get(cam, False)) and cam in self.H and h_mvs_to_event is not None else 0
            if valid:
                draw_cams.append(cam)
            elif event_layer_enabled and tex > 0 and bool(self.event_layer_valid.get(cam, False)) and cam in self.H and h_mvs_to_event is None:
                missing_transform_cams.append(cam)
            GL.glUniform1i(GL.glGetUniformLocation(self.program, f"u_event_valid[{idx}]"), int(valid))
            GL.glUniform2f(GL.glGetUniformLocation(self.program, f"u_event_img_size[{idx}]"), float(w), float(h))
        if event_layer_enabled:
            self.event_layer_diag["shader_draw_cams"] = list(draw_cams)
            self.event_layer_diag["shader_draw_cam_count"] = int(len(draw_cams))
            self.event_layer_diag["shader_missing_homography_cams"] = list(missing_transform_cams)
        self._quad()
        GL.glUseProgram(0)
        if self._event_layer_backend() == "sparse_gl":
            self._draw_sparse_event_layer()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._qt_default_fbo())

    def _readback_output(self, target_sync: int) -> None:
        if self.writer is None:
            return
        rb_t0 = time.perf_counter()
        nbytes = self.out_w * self.out_h * 4
        rgba = None
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        readback_mode = "glReadPixels"
        if self.pbos:
            cur_idx = int(self.pbo_index) % max(1, len(self.pbos))
            cur = self.pbos[cur_idx]
            read_idx = (cur_idx + 1) % len(self.pbos)
            prev = self.pbos[read_idx]
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, cur)
            GL.glReadPixels(0, 0, self.out_w, self.out_h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            readback_mode = f"pbo_{len(self.pbos)}"
            if self.pbo_ready_count >= len(self.pbos):
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, prev)
                ptr = GL.glMapBuffer(GL.GL_PIXEL_PACK_BUFFER, GL.GL_READ_ONLY)
                if ptr:
                    addr = int(getattr(ptr, "value", ptr) or 0)
                    if addr:
                        raw = ctypes.string_at(addr, nbytes)
                        rgba = np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 4))
                    GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
            self.pbo_ready_count = min(len(self.pbos), int(self.pbo_ready_count) + 1)
            self.pbo_ready = self.pbo_ready_count >= len(self.pbos)
            self.pbo_index = (cur_idx + 1) % len(self.pbos)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        if rgba is None:
            raw = GL.glReadPixels(0, 0, self.out_w, self.out_h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 4))
            readback_mode = "direct_glReadPixels"
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._qt_default_fbo())
        now = time.time()
        stats_interval_s = max(0.05, float(getattr(self.args, "fbo_stats_interval_ms", 1000.0) or 1000.0) / 1000.0)
        if now - float(self._last_fbo_stats_t) >= stats_interval_s:
            stats_t0 = time.perf_counter()
            stride = max(1, int(getattr(self.args, "fbo_stats_stride", 8) or 8))
            rgb = rgba[::stride, ::stride, :3].astype(np.float32)
            luma = rgb.mean(axis=2) if rgb.size else np.zeros((0, 0), dtype=np.float32)
            if luma.size:
                h = luma.shape[0]
                bg = np.asarray([8.0, 9.0, 10.0], dtype=np.float32)
                dist = np.abs(rgb - bg.reshape(1, 1, 3)).mean(axis=2)
                self.fbo_stats = {
                    "mean": float(np.mean(luma)),
                    "max": int(np.max(luma)),
                    "p01": float(np.percentile(luma, 1)),
                    "p05": float(np.percentile(luma, 5)),
                    "p50": float(np.percentile(luma, 50)),
                    "p95": float(np.percentile(luma, 95)),
                    "p99": float(np.percentile(luma, 99)),
                    "non_bg_ratio": float(np.count_nonzero(dist > 8.0) / float(dist.size)),
                    "top_mean": float(np.mean(luma[:max(1, h // 5), :])),
                    "mid_mean": float(np.mean(luma[max(0, (h * 2) // 5):max(1, (h * 3) // 5), :])),
                    "bottom_mean": float(np.mean(luma[max(0, (h * 4) // 5):, :])),
                    "stats_age_ms": 0.0,
                    "stats_cost_ms": round(float((time.perf_counter() - stats_t0) * 1000.0), 3),
                    "stats_stride": int(stride),
                    "stats_sample_shape": [int(luma.shape[1]), int(luma.shape[0])],
                }
            self._last_fbo_stats_t = now
        elif self.fbo_stats:
            self.fbo_stats["stats_age_ms"] = round(float((now - float(self._last_fbo_stats_t)) * 1000.0), 3)
        rgba = np.flipud(rgba)
        self.writer.write_rgba(rgba, self.frame_id, _now_us(), target_sync)
        self._output_count += 1
        self._write_sidecar(target_sync)
        self.readback_diag = {
            "mode": readback_mode,
            "pbo_count": int(len(self.pbos)),
            "pbo_ready": bool(self.pbo_ready),
            "pbo_ready_count": int(self.pbo_ready_count),
            "output_bytes": int(nbytes),
            "elapsed_ms": round(float((time.perf_counter() - rb_t0) * 1000.0), 3),
            "fbo_stats_interval_ms": float(getattr(self.args, "fbo_stats_interval_ms", 1000.0)),
            "flip_copy": "np.flipud_view_then_shm_copy",
        }

    def _write_sidecar(self, target_sync: int) -> None:
        obj = {
            "schema_version": 1,
            "frame_id": int(self.frame_id),
            "timestamp_us": _now_us(),
            "shm": str(_shm_path(str(self.args.output_shm))),
            "width": int(self.out_w),
            "height": int(self.out_h),
            "channels": 4,
            "pixel_format": "RGBA8",
            "field_origin_m": [float(self.args.field_origin_x_m), float(self.args.field_origin_y_m)],
            "field_size_m": [float(self.args.field_width_m), float(self.args.field_height_m)],
            "field_window_m": self._field_window(),
            "field_range_from_calib_m": self._calib_field_range(),
            "field_transform": {
                "mode": str(self.args.field_transform_mode),
                "mode_code": _transform_mode_code(self.args.field_transform_mode),
                "flip_x": bool(self.args.field_flip_x),
                "flip_y": bool(self.args.field_flip_y),
                "available_modes": list(TRANSFORM_MODE_ORDER),
                "note": "Display-space transform before sampling calibrated field coordinates.",
            },
            "meters_per_pixel": float(self.args.meters_per_pixel),
            "sync_mode": str(self.args.sync_mode),
            "target_sync_id": int(target_sync),
            "presentation": dict(self.presentation_diag),
            "mvs_sync": dict(self.mvs_sync_diag),
            "debug_mode": str(self.args.debug_mode),
            "texture_y_flip": bool(self.args.texture_y_flip),
            "texture_format": str(self.args.texture_format),
            "tone": {
                "gain": float(self.args.gain),
                "black_level": float(self.args.black_level),
                "gamma": float(self.args.gamma),
                "color_gain": [
                    float(self.args.color_gain_r),
                    float(self.args.color_gain_g),
                    float(self.args.color_gain_b),
                ],
                "grid_enable": bool(self.args.grid_enable),
            },
            "turf_color_match": dict(self.turf_diag),
            "stability": dict(self.stability_diag),
            "coverage": dict(self.coverage_diag),
            "texture_upload": dict(self.texture_diag),
            "mvs_texture_upload": dict(self.mvs_upload_diag),
            "fbo_stats": dict(self.fbo_stats),
            "readback": dict(self.readback_diag),
            "display": dict(self.display_diag),
            "control": dict(self.control_diag),
            "alignment": dict(self.alignment_diag),
            "alignment_shadow": dict(self.alignment_diag),
            "global_person": dict(self.global_person_diag),
            "global_person_overlay": dict(self.global_person_diag),
            "manual_hit": self._manual_hit_diag(),
            "event_layer": dict(self.event_layer_diag),
            "event_layer_overlay": dict(self.event_layer_diag),
            "event_bullet": dict(self.event_bullet_diag),
            "event_bullet_overlay": dict(self.event_bullet_diag),
            "performance": dict(self.perf_diag),
            "cameras": {cam: {k: v for k, v in item.items() if k != "frame"} for cam, item in self.selected.items()},
        }
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.sidecar_path.with_suffix(self.sidecar_path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.sidecar_path)

    def _write_status(self) -> None:
        now = time.time()
        status_published_us = _now_us()
        dt = max(1e-6, now - self._fps_t0)
        if dt >= 1.0:
            self.render_fps = self._render_count / dt
            self.output_fps = self._output_count / dt
            self._render_count = 0
            self._output_count = 0
            self._fps_t0 = now
        coverage_interval_s = max(0.1, float(getattr(self.args, "coverage_interval_ms", 2000.0) or 2000.0) / 1000.0)
        if self.selected and now - float(self._last_coverage_t) >= coverage_interval_s:
            try:
                self.coverage_diag = self._compute_coverage_diag(self.selected)
                self.coverage_diag["interval_ms"] = float(coverage_interval_s * 1000.0)
                self.coverage_diag["updated_us"] = _now_us()
                self._last_coverage_t = now
            except Exception as exc:
                self.coverage_diag = {"error": f"{type(exc).__name__}: {exc}"}
                self._last_coverage_t = now
        self.perf_diag = self._perf_summary()
        obj = {
            "schema_version": 1,
            "state": "running",
            "frame_id": int(self.frame_id),
            "published_wall_us": int(status_published_us),
            "published_wall_iso": _iso_from_us(int(status_published_us)),
            "render_fps": float(self.render_fps),
            "output_fps": float(self.output_fps),
            "output_shm": str(_shm_path(str(self.args.output_shm))),
            "sidecar_json": str(self.sidecar_path),
            "width": int(self.out_w),
            "height": int(self.out_h),
            "sync_mode": str(self.args.sync_mode),
            "target_sync_id": int(self.current_target_sync_id),
            "presentation": dict(self.presentation_diag),
            "mvs_sync": dict(self.mvs_sync_diag),
            "debug_mode": str(self.args.debug_mode),
            "texture_y_flip": bool(self.args.texture_y_flip),
            "texture_format": str(self.args.texture_format),
            "tone": {
                "gain": float(self.args.gain),
                "black_level": float(self.args.black_level),
                "gamma": float(self.args.gamma),
                "color_gain": [
                    float(self.args.color_gain_r),
                    float(self.args.color_gain_g),
                    float(self.args.color_gain_b),
                ],
                "grid_enable": bool(self.args.grid_enable),
            },
            "turf_color_match": dict(self.turf_diag),
            "stability": dict(self.stability_diag),
            "field_window_m": self._field_window(),
            "field_range_from_calib_m": self._calib_field_range(),
            "field_transform": {
                "mode": str(self.args.field_transform_mode),
                "mode_code": _transform_mode_code(self.args.field_transform_mode),
                "flip_x": bool(self.args.field_flip_x),
                "flip_y": bool(self.args.field_flip_y),
                "available_modes": list(TRANSFORM_MODE_ORDER),
                "note": "Display-space transform before sampling calibrated field coordinates.",
            },
            "coverage": dict(self.coverage_diag),
            "texture_upload": dict(self.texture_diag),
            "mvs_texture_upload": dict(self.mvs_upload_diag),
            "fbo_stats": dict(self.fbo_stats),
            "readback": dict(self.readback_diag),
            "display": dict(self.display_diag),
            "control": dict(self.control_diag),
            "alignment": dict(self.alignment_diag),
            "alignment_shadow": dict(self.alignment_diag),
            "global_person": dict(self.global_person_diag),
            "global_person_overlay": dict(self.global_person_diag),
            "manual_hit": self._manual_hit_diag(),
            "event_layer": dict(self.event_layer_diag),
            "event_layer_overlay": dict(self.event_layer_diag),
            "event_bullet": dict(self.event_bullet_diag),
            "event_bullet_overlay": dict(self.event_bullet_diag),
            "performance": dict(self.perf_diag),
            "status_writer": self.status_writer.snapshot() if self.status_writer is not None else {
                "mode": "sync",
                "state": "disabled",
                "thread_alive": False,
                "submitted": 0,
                "written": 0,
                "dropped": 0,
                "errors": 0,
                "pending": False,
                "last_submit_ms": 0.0,
                "last_write_ms": 0.0,
                "last_write_wall_us": 0,
                "last_error": "",
            },
            "selected_summary": {
                cam: {
                    "frame_id": _safe_int(item.get("frame_id"), -1),
                    "tick_id": _safe_int(item.get("tick_id"), -1),
                    "slot": _safe_int(item.get("slot"), -1),
                    "age_ms": _safe_float(item.get("age_ms"), -1.0),
                    "sync_delta_ms": _safe_float(item.get("selected_sync_delta_ms"), -999999.0),
                }
                for cam, item in self.selected.items()
            },
            "reader_errors": dict(self.reader_errors),
            "status_debug_json": str(self.status_debug_path),
            "last_error": str(self._last_error),
        }
        debug_obj: Optional[Dict[str, Any]] = None
        debug_interval_s = max(0.0, float(getattr(self.args, "status_debug_interval_ms", 0.0) or 0.0) / 1000.0)
        if debug_interval_s > 0.0 and now - float(self._last_status_debug_t) >= debug_interval_s:
            debug_obj = dict(obj)
            debug_obj["debug_status"] = True
            debug_obj["selected"] = {cam: {k: v for k, v in item.items() if k != "frame"} for cam, item in self.selected.items()}
            debug_obj["calibration"] = dict(self.calib)
            self._last_status_debug_t = now
        if self.status_writer is not None and self.status_writer.enabled:
            self.status_writer.submit(obj, debug_obj)
        else:
            self._write_status_payload_sync(obj, debug_obj)

    def _write_status_payload_sync(self, obj: Dict[str, Any], debug_obj: Optional[Dict[str, Any]] = None) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.status_path)
        if debug_obj is not None:
            debug_tmp = self.status_debug_path.with_suffix(self.status_debug_path.suffix + ".tmp")
            debug_tmp.write_text(json.dumps(debug_obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(debug_tmp, self.status_debug_path)

    def paintGL(self) -> None:
        total_t0 = self._perf_start()
        try:
            t0 = self._perf_start()
            self._poll_control_json()
            self._perf_end("poll_control", t0)
            t0 = self._perf_start()
            self._poll_global_person_json()
            self._perf_end("poll_global_person", t0)
            t0 = self._perf_start()
            self._open_inputs()
            self._perf_end("open_inputs", t0)
            t0 = self._perf_start()
            target_sync, selected = self._choose_frames()
            self._perf_end("choose_frames", t0)
            t0 = self._perf_start()
            self._refresh_alignment_diag(target_sync, selected)
            self._perf_end("refresh_alignment", t0)
            if selected:
                t0 = self._perf_start()
                self._upload_selected(selected)
                self._perf_end("upload_selected", t0)
                t0 = self._perf_start()
                self._upload_event_layer_textures(render_context=selected, target_sync=target_sync)
                self._perf_end("upload_event_layer_textures", t0)
                t0 = self._perf_start()
                render_selected = self._current_render_items()
                self._perf_end("current_render_items", t0)
                self.selected = render_selected
                self._reset_draw_state()
                t0 = self._perf_start()
                self._render_fbo(target_sync, render_selected)
                self._perf_end("render_fbo", t0)
                self.frame_id += 1
                self._render_count += 1
                output_due, next_output_deadline = rate_deadline_due(
                    self._last_output_t,
                    time.monotonic(),
                    float(self.args.output_fps),
                )
                if output_due:
                    t0 = self._perf_start()
                    self._readback_output(target_sync)
                    self._perf_end("readback_output", t0)
                    self._last_output_t = next_output_deadline
            else:
                t0 = self._perf_start()
                render_selected = self._current_render_items()
                self._perf_end("current_render_items", t0)
                if render_selected:
                    self.selected = render_selected
                    t0 = self._perf_start()
                    self._upload_event_layer_textures(render_context=render_selected, target_sync=-1)
                    self._perf_end("upload_event_layer_textures", t0)
                    self._reset_draw_state()
                    t0 = self._perf_start()
                    self._render_fbo(-1, render_selected)
                    self._perf_end("render_fbo", t0)
                    self.frame_id += 1
                    self._render_count += 1
            qt_fbo = self._qt_default_fbo()
            self._reset_draw_state()
            dpr = float(self.devicePixelRatioF())
            view_w = max(1, int(round(float(self.width()) * max(1.0, dpr))))
            view_h = max(1, int(round(float(self.height()) * max(1.0, dpr))))
            self.display_diag = {
                "logical_width": int(self.width()),
                "logical_height": int(self.height()),
                "device_pixel_ratio": float(dpr),
                "viewport_width": int(view_w),
                "viewport_height": int(view_h),
                "qt_default_fbo": int(qt_fbo),
                "draw_target_fbo": int(qt_fbo),
                "fbo_texture": int(self.fbo_tex),
            }
            t0 = self._perf_start()
            self.display_diag.update(self._display_to_qt_fbo(qt_fbo, view_w, view_h))
            self._perf_end("display_to_qt_fbo", t0)
            t0 = self._perf_start()
            self._draw_global_person_overlay()
            self._perf_end("draw_global_person_overlay", t0)
            t0 = self._perf_start()
            self._draw_event_bullet_overlay()
            self._perf_end("draw_event_bullet_overlay", t0)
            t0 = self._perf_start()
            self._draw_overlay_text()
            self._perf_end("draw_overlay_text", t0)
            if time.time() - self._last_status_t >= max(0.1, float(self.args.status_interval_ms) / 1000.0):
                t0 = self._perf_start()
                self._write_status()
                status_stage = "status_submit" if self.status_writer is not None and self.status_writer.enabled else "write_status"
                self._perf_end(status_stage, t0)
                self._last_status_t = time.time()
            self._perf_end("paint_total", total_t0)
        except Exception as exc:
            self._perf_end("paint_total", total_t0)
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[global_bev][WARN] paint failed: {self._last_error}", flush=True)

    def _manual_hit_diag(self) -> Dict[str, Any]:
        age_ms = -1.0
        cache_us = int(getattr(self, "manual_hit_target_cache_us", 0) or 0)
        if cache_us > 0:
            age_ms = max(0.0, (_now_us() - cache_us) / 1000.0)
        return {
            "click_enable": bool(getattr(self, "manual_hit_click_enable", False)),
            "click_source": "global_bev_window",
            "command_jsonl": str(getattr(self, "manual_hit_command_path", "")),
            "click_radius_px": float(getattr(self, "manual_hit_click_radius_px", 42.0)),
            "target_cache_count": len(getattr(self, "manual_hit_targets", []) or []),
            "target_cache_age_ms": round(float(age_ms), 3) if age_ms >= 0.0 else -1.0,
            "command_seq": int(getattr(self, "manual_hit_command_seq", 0) or 0),
            "last_command": dict(getattr(self, "manual_hit_last_result", {}) or {}),
        }

    def _manual_hit_target_from_track(
        self,
        tr: Dict[str, Any],
        raw_xy: Tuple[float, float],
        overlay_xy: Tuple[float, float],
        widget_xy: Tuple[float, float],
    ) -> Dict[str, Any]:
        slot_id = _safe_int(tr.get("display_slot_id"), -1)
        source_gid = _safe_int(tr.get("source_global_id", tr.get("global_id")), -1)
        target_gid = int(slot_id if slot_id >= 0 else source_gid)
        semantics = "global_bev_display_id" if slot_id >= 0 else "global_runtime_id"
        display_player_id = str(
            tr.get("display_player_id")
            or tr.get("global_player_id")
            or (f"P{slot_id:02d}" if slot_id >= 0 else f"P{target_gid:02d}")
        )
        return {
            "target_global_id_int": int(target_gid),
            "target_id_semantics": semantics,
            "target_display_slot_id": int(slot_id),
            "target_display_player_id": display_player_id,
            "target_source_global_id_int": int(source_gid),
            "target_source_global_id": str(source_gid) if source_gid >= 0 else "",
            "source_global_player_id": str(tr.get("source_global_player_id") or tr.get("global_player_id") or ""),
            "state": str(tr.get("state", "")),
            "age_ms": _safe_float(tr.get("age_ms"), -1.0),
            "confidence": _safe_float(tr.get("confidence"), 0.0),
            "speed_mps": _safe_float(tr.get("speed_mps"), 0.0),
            "field_xy": [round(float(raw_xy[0]), 4), round(float(raw_xy[1]), 4)],
            "overlay_field_xy": [round(float(overlay_xy[0]), 4), round(float(overlay_xy[1]), 4)],
            "widget_xy": [round(float(widget_xy[0]), 2), round(float(widget_xy[1]), 2)],
            "radius_px": float(getattr(self, "manual_hit_click_radius_px", 42.0)),
            "display_source_key": str(tr.get("display_source_key", "")),
            "source_cameras": list(tr.get("source_cameras", [])) if isinstance(tr.get("source_cameras"), list) else [],
            "local_ids": dict(tr.get("local_ids", {}) or {}) if isinstance(tr.get("local_ids"), dict) else {},
        }

    def _select_manual_hit_target(self, px: float, py: float) -> Tuple[Optional[Dict[str, Any]], float]:
        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for target in list(getattr(self, "manual_hit_targets", []) or []):
            xy = target.get("widget_xy")
            if not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
                continue
            dd = math.hypot(float(px) - _safe_float(xy[0]), float(py) - _safe_float(xy[1]))
            if dd < best_dist:
                best_dist = float(dd)
                best = target
        radius = max(4.0, float(getattr(self, "manual_hit_click_radius_px", 42.0) or 42.0))
        if best is None or best_dist > radius:
            return None, float(best_dist)
        return dict(best), float(best_dist)

    def _append_manual_hit_command(self, target: Dict[str, Any], click_xy: Tuple[float, float], distance_px: float) -> Dict[str, Any]:
        self.manual_hit_command_seq += 1
        now_us = _now_us()
        command_id = f"GLOBAL-BEV-CLICK-{now_us}-{int(self.manual_hit_command_seq)}"
        cmd = {
            "schema_version": 1,
            "type": "manual_hit_command",
            "source": "global_bev_window_click",
            "click_source": "global_bev_window",
            "command_id": command_id,
            "wall_time": time.time(),
            "wall_time_us": int(now_us),
            "camera": "global_bev",
            "click_xy": [round(float(click_xy[0]), 2), round(float(click_xy[1]), 2)],
            "click_distance_px": round(float(distance_px), 3),
            "field_xy": list(target.get("field_xy", [])),
            "overlay_field_xy": list(target.get("overlay_field_xy", [])),
            "target_global_id_int": _safe_int(target.get("target_global_id_int"), -1),
            "target_global_id": str(target.get("target_global_id_int", "")),
            "target_id_semantics": str(target.get("target_id_semantics") or "global_bev_display_id"),
            "target_display_slot_id": _safe_int(target.get("target_display_slot_id"), -1),
            "target_display_player_id": str(target.get("target_display_player_id") or ""),
            "target_source_global_id_int": _safe_int(target.get("target_source_global_id_int"), -1),
            "target_source_global_id": str(target.get("target_source_global_id") or ""),
            "target_state": str(target.get("state") or ""),
            "target_age_ms": _safe_float(target.get("age_ms"), -1.0),
            "target_confidence": _safe_float(target.get("confidence"), 0.0),
            "target_widget_xy": list(target.get("widget_xy", [])),
            "target_source_cameras": list(target.get("source_cameras", [])) if isinstance(target.get("source_cameras"), list) else [],
            "target_local_ids": dict(target.get("local_ids", {}) or {}) if isinstance(target.get("local_ids"), dict) else {},
            "bev_presentation_delay_enable": bool(self.presentation_diag.get("enabled", False)),
            "bev_presentation_delay_ms": _safe_float(self.presentation_diag.get("delay_ms"), 0.0),
            "bev_presentation_effective_lag_ms_est": _safe_float(self.presentation_diag.get("effective_lag_ms_est"), -1.0),
            "bev_presentation_target_sync_id": _safe_int(self.presentation_diag.get("target_sync_id"), -1),
            "bev_presentation_latest_common_sync_id": _safe_int(self.presentation_diag.get("latest_common_sync_id"), -1),
            "target_binding_time_semantics": "global_bev_delayed_background_with_latest_person_fallback",
        }
        self.manual_hit_command_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manual_hit_command_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cmd, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.manual_hit_last_result = {
            "state": "command_written",
            "command_id": command_id,
            "target_display_player_id": cmd.get("target_display_player_id"),
            "target_global_id_int": cmd.get("target_global_id_int"),
            "target_id_semantics": cmd.get("target_id_semantics"),
            "click_xy": cmd.get("click_xy"),
            "click_distance_px": cmd.get("click_distance_px"),
            "wall_time_us": int(now_us),
        }
        print(f"[global_bev][manual_hit] wrote {command_id} target={cmd.get('target_display_player_id')}", flush=True)
        return cmd

    def _handle_manual_hit_click(self, px: float, py: float) -> bool:
        target, dist_px = self._select_manual_hit_target(px, py)
        if target is None:
            self.manual_hit_last_result = {
                "state": "no_target",
                "click_xy": [round(float(px), 2), round(float(py), 2)],
                "nearest_distance_px": round(float(dist_px), 3) if math.isfinite(dist_px) else -1.0,
                "target_cache_count": len(getattr(self, "manual_hit_targets", []) or []),
                "wall_time_us": _now_us(),
            }
            print(f"[global_bev][manual_hit] click ignored: no target near ({px:.1f},{py:.1f})", flush=True)
            return False
        try:
            self._append_manual_hit_command(target, (px, py), dist_px)
            return True
        except Exception as exc:
            self.manual_hit_last_result = {
                "state": "write_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "click_xy": [round(float(px), 2), round(float(py), 2)],
                "wall_time_us": _now_us(),
            }
            print(f"[global_bev][manual_hit][WARN] write failed: {self.manual_hit_last_result['error']}", flush=True)
            return False

    def _draw_overlay_text(self) -> None:
        fps = max(0.5, float(getattr(self.args, "overlay_text_fps", 5.0) or 5.0))
        pix_w = max(1, min(max(1, int(self.width()) - 16), 1120))
        pix_h = 198
        now = time.monotonic()
        need_rebuild = (
            self._overlay_text_pixmap is None
            or self._overlay_text_pixmap_size != (pix_w, pix_h)
            or now - float(self._last_overlay_text_cache_t) >= 1.0 / fps
        )
        if need_rebuild:
            pix = QPixmap(pix_w, pix_h)
            pix.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pix)
            try:
                painter.setFont(QFont("DejaVu Sans Mono", 11))
                painter.setPen(QColor(180, 240, 255))
                painter.drawText(8, 20, f"Global BEV {self.out_w}x{self.out_h} sync={self.args.sync_mode} render={self.render_fps:.1f} output={self.output_fps:.1f}")
                painter.drawText(8, 42, f"selected={','.join(sorted(self.selected.keys()))} shm={_shm_path(str(self.args.output_shm))}")
                cover = self.coverage_diag.get("any_camera_valid_ratio", None)
                if cover is not None:
                    painter.drawText(8, 64, f"field={self.args.field_width_m:g}x{self.args.field_height_m:g}m origin=({self.args.field_origin_x_m:g},{self.args.field_origin_y_m:g}) cover={float(cover):.3f} mode={self.args.debug_mode} yflip={int(bool(self.args.texture_y_flip))}")
                painter.drawText(
                    8,
                    86,
                    f"gain={float(self.args.gain):.2f} black={float(self.args.black_level):.3f} "
                    f"gamma={float(self.args.gamma):.2f} cg=({float(self.args.color_gain_r):.2f},"
                    f"{float(self.args.color_gain_g):.2f},{float(self.args.color_gain_b):.2f}) "
                    f"grid={int(bool(self.args.grid_enable))}",
                )
                painter.drawText(
                    8,
                    108,
                    f"transform={self.args.field_transform_mode} turf={int(bool(self.args.turf_match_enable))}/{float(self.args.turf_match_strength):.2f} "
                    f"person_xy={getattr(self.args, 'global_person_overlay_coord_mode', 'field_xy')} "
                    f"hold={float(self.args.frame_hold_ms):.0f}ms hotkeys=M/N Q D T G R",
                )
                bullet = self.event_bullet_diag if isinstance(self.event_bullet_diag, dict) else {}
                painter.drawText(
                    8,
                    130,
                    f"event_bullet={int(bool(bullet.get('enabled', False)))} "
                    f"buf={int(bullet.get('records_buffered', 0) or 0)} recent={int(bullet.get('recent_records', 0) or 0)} "
                    f"draw={int(bullet.get('draw_points', 0) or 0)}/{int(bullet.get('draw_segments', 0) or 0)} "
                    f"age={float(bullet.get('latest_age_ms', -1.0) or -1.0):.0f}ms",
                )
                layer = self.event_layer_diag if isinstance(self.event_layer_diag, dict) else {}
                painter.drawText(
                    8,
                    152,
                    f"event_layer={int(bool(layer.get('enabled', False)))} "
                    f"draw_cams={int(layer.get('shader_draw_cam_count', layer.get('draw_cam_count', 0)) or 0)} "
                    f"alpha={float(layer.get('alpha', getattr(self.args, 'event_layer_alpha', 0.45)) or 0.0):.2f} "
                    f"age_gate={float(layer.get('max_age_ms', getattr(self.args, 'event_layer_max_age_ms', 150.0)) or 0.0):.0f}ms "
                    f"reason={str(layer.get('skip_reason', ''))[:36]}",
                )
                if self._last_error:
                    painter.setPen(QColor(255, 160, 120))
                    painter.drawText(8, 174, self._last_error[:120])
            finally:
                painter.end()
            self._overlay_text_pixmap = pix
            self._overlay_text_pixmap_size = (pix_w, pix_h)
            self._last_overlay_text_cache_t = now
        painter = QPainter(self)
        try:
            if self._overlay_text_pixmap is not None:
                painter.drawPixmap(8, 8, self._overlay_text_pixmap)
        finally:
            painter.end()

    def _draw_global_person_overlay(self) -> None:
        if not bool(getattr(self.args, "global_person_overlay_enable", False)):
            self.manual_hit_targets = []
            self.manual_hit_target_cache_us = _now_us()
            return
        fps = max(1.0, float(getattr(self.args, "person_overlay_build_fps", 15.0) or 15.0))
        now = time.monotonic()
        if now - float(self._person_overlay_cache_t) >= 1.0 / fps or not self._person_overlay_cached_tracks:
            tracks = self._stable_global_person_overlay_tracks()
            self._person_overlay_cached_tracks = list(tracks)
            self._person_overlay_cache_t = now
            self.global_person_diag["overlay_build_fps_limit"] = float(fps)
            self.global_person_diag["overlay_cache_age_ms"] = 0.0
        else:
            tracks = list(self._person_overlay_cached_tracks)
            self.global_person_diag["overlay_build_fps_limit"] = float(fps)
            self.global_person_diag["overlay_cache_age_ms"] = round(float((now - float(self._person_overlay_cache_t)) * 1000.0), 3)
        if not tracks:
            self.manual_hit_targets = []
            self.manual_hit_target_cache_us = _now_us()
            return
        targets: List[Dict[str, Any]] = []
        painter = QPainter(self)
        try:
            painter.setFont(QFont("DejaVu Sans Mono", 12))
            for tr in tracks[:64]:
                state = str(tr.get("state", ""))
                if state not in {"tracking", "tentative", "lost"}:
                    continue
                try:
                    xy = tr.get("field_xy")
                    if not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
                        xy = [tr.get("x_m"), tr.get("y_m")]
                    x = float(xy[0])
                    y = float(xy[1])
                except Exception:
                    continue
                speed = _safe_float(tr.get("speed_mps"), 0.0)
                max_speed = float(getattr(self.args, "global_person_overlay_max_speed_mps", 14.0))
                if max_speed > 0.0 and speed > max_speed:
                    continue
                age_ms = _safe_float(tr.get("age_ms"), 0.0)
                lost_max = float(getattr(self.args, "global_person_overlay_lost_max_age_ms", 500.0))
                if state == "lost" and lost_max >= 0.0 and age_ms > lost_max:
                    continue
                raw_xy = (float(x), float(y))
                overlay_xy = self._transform_person_xy_for_overlay(x, y)
                pt = self._field_xy_to_widget_xy(overlay_xy[0], overlay_xy[1])
                if pt is None:
                    continue
                px, py = pt
                targets.append(self._manual_hit_target_from_track(tr, raw_xy, overlay_xy, (px, py)))
                if state == "lost":
                    color = QColor(255, 145, 145, 190)
                elif state == "tentative":
                    color = QColor(255, 95, 120, 220)
                else:
                    color = QColor(255, 55, 55, 235)
                painter.setPen(color)
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70))
                painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)
                label = str(tr.get("display_player_id") or tr.get("global_player_id") or f"P{int(tr.get('global_id', 0)):02d}")
                painter.drawText(int(px + 9), int(py - 8), f"{label} {speed:.1f}m/s")
        finally:
            painter.end()
            self.manual_hit_targets = targets
            self.manual_hit_target_cache_us = _now_us()

    def _person_track_xy(self, tr: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        try:
            xy = tr.get("field_xy")
            if not (isinstance(xy, (list, tuple)) and len(xy) >= 2):
                xy = [tr.get("x_m"), tr.get("y_m")]
            x = float(xy[0])
            y = float(xy[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            return x, y
        except Exception:
            return None

    def _person_track_source_key(self, tr: Dict[str, Any]) -> str:
        gid = tr.get("global_id")
        if gid is not None:
            try:
                return f"gid:{int(gid)}"
            except Exception:
                pass
        label = str(tr.get("global_player_id", "") or "").strip()
        if label:
            return f"label:{label}"
        local_ids = tr.get("local_ids")
        if isinstance(local_ids, dict) and local_ids:
            return "local:" + ",".join(f"{k}:{v}" for k, v in sorted(local_ids.items()))
        return f"obj:{id(tr)}"

    def _make_person_display_label(self, slot_id: int) -> str:
        return f"P{int(slot_id):02d}"

    def _stable_global_person_overlay_tracks(self) -> List[Dict[str, Any]]:
        if not bool(getattr(self.args, "global_person_overlay_display_stabilize_enable", True)):
            return list(self.global_person_tracks)
        now = time.monotonic()
        hold_ms = max(0.0, float(getattr(self.args, "global_person_overlay_hold_ms", 900.0)))
        reid_ttl_ms = max(0.0, float(getattr(self.args, "global_person_overlay_reid_ttl_ms", 2200.0)))
        reid_gate = max(0.0, float(getattr(self.args, "global_person_overlay_reid_gate_m", 1.2)))
        alpha = _clip01(float(getattr(self.args, "global_person_overlay_display_alpha", 0.45)))
        used_keys = set()
        relinked = 0
        created = 0

        for tr in self.global_person_tracks:
            if not isinstance(tr, dict):
                continue
            xy = self._person_track_xy(tr)
            if xy is None:
                continue
            key = self._person_track_source_key(tr)
            slot = self._person_overlay_slots.get(key)
            if slot is None:
                best_key = ""
                best_dist = 1e9
                for old_key, old_slot in self._person_overlay_slots.items():
                    if old_key in used_keys:
                        continue
                    age_ms = (now - float(old_slot.get("last_seen_mono", now))) * 1000.0
                    if age_ms > reid_ttl_ms:
                        continue
                    old_xy = (float(old_slot.get("x_m", xy[0])), float(old_slot.get("y_m", xy[1])))
                    dd = _dist_xy(xy, old_xy)
                    if dd <= reid_gate and dd < best_dist:
                        best_dist = dd
                        best_key = old_key
                if best_key:
                    slot = self._person_overlay_slots.pop(best_key)
                    slot["source_key"] = key
                    slot["relinked_count"] = int(slot.get("relinked_count", 0) or 0) + 1
                    self._person_overlay_slots[key] = slot
                    relinked += 1
                else:
                    slot_id = int(self._person_overlay_next_slot_id)
                    self._person_overlay_next_slot_id += 1
                    slot = {
                        "slot_id": slot_id,
                        "display_player_id": self._make_person_display_label(slot_id),
                        "source_key": key,
                        "created_mono": now,
                        "relinked_count": 0,
                    }
                    self._person_overlay_slots[key] = slot
                    created += 1

            old_x = slot.get("draw_x_m")
            old_y = slot.get("draw_y_m")
            if old_x is None or old_y is None:
                draw_x, draw_y = float(xy[0]), float(xy[1])
            else:
                draw_x = float(old_x) * (1.0 - alpha) + float(xy[0]) * alpha
                draw_y = float(old_y) * (1.0 - alpha) + float(xy[1]) * alpha
            slot.update({
                "source_key": key,
                "source_global_id": tr.get("global_id"),
                "source_global_player_id": tr.get("global_player_id"),
                "x_m": float(xy[0]),
                "y_m": float(xy[1]),
                "draw_x_m": float(draw_x),
                "draw_y_m": float(draw_y),
                "last_seen_mono": now,
                "track": dict(tr),
                "state": str(tr.get("state", "tracking") or "tracking"),
                "speed_mps": _safe_float(tr.get("speed_mps"), 0.0),
            })
            used_keys.add(key)

        for key in list(self._person_overlay_slots.keys()):
            slot = self._person_overlay_slots[key]
            age_ms = (now - float(slot.get("last_seen_mono", now))) * 1000.0
            if key not in used_keys and age_ms > max(hold_ms, reid_ttl_ms):
                self._person_overlay_slots.pop(key, None)

        out: List[Dict[str, Any]] = []
        display_tracks: List[Dict[str, Any]] = []
        display_id_by_source_global_id: Dict[str, Dict[str, Any]] = {}
        for key, slot in sorted(self._person_overlay_slots.items(), key=lambda kv: int(kv[1].get("slot_id", 0))):
            age_ms = (now - float(slot.get("last_seen_mono", now))) * 1000.0
            if key not in used_keys and age_ms > hold_ms:
                continue
            base = dict(slot.get("track", {}) if isinstance(slot.get("track"), dict) else {})
            state = str(slot.get("state", base.get("state", "tracking")) or "tracking")
            if key not in used_keys:
                state = "lost"
            base.update({
                "display_player_id": str(slot.get("display_player_id", "")),
                "display_slot_id": int(slot.get("slot_id", 0) or 0),
                "display_source_key": str(slot.get("source_key", key)),
                "display_relinked_count": int(slot.get("relinked_count", 0) or 0),
                "state": state,
                "x_m": round(float(slot.get("draw_x_m", slot.get("x_m", 0.0))), 4),
                "y_m": round(float(slot.get("draw_y_m", slot.get("y_m", 0.0))), 4),
                "field_xy": [
                    round(float(slot.get("draw_x_m", slot.get("x_m", 0.0))), 4),
                    round(float(slot.get("draw_y_m", slot.get("y_m", 0.0))), 4),
                ],
                "age_ms": round(float(age_ms), 2),
                "speed_mps": round(float(slot.get("speed_mps", base.get("speed_mps", 0.0))), 4),
            })
            out.append(base)
            display_record = {
                "display_player_id": str(base.get("display_player_id", "")),
                "display_slot_id": int(base.get("display_slot_id", 0) or 0),
                "display_source_key": str(base.get("display_source_key", "")),
                "source_global_id": base.get("global_id"),
                "source_global_player_id": base.get("global_player_id"),
                "state": str(base.get("state", "")),
                "age_ms": _safe_float(base.get("age_ms"), 0.0),
                "confidence": _safe_float(base.get("confidence"), 0.0),
                "source_cameras": list(base.get("source_cameras", [])) if isinstance(base.get("source_cameras"), list) else [],
                "field_xy": list(base.get("field_xy", [])) if isinstance(base.get("field_xy"), (list, tuple)) else [],
                "local_ids": dict(base.get("local_ids", {}) or {}) if isinstance(base.get("local_ids"), dict) else {},
            }
            display_tracks.append(display_record)
            try:
                source_gid = int(base.get("global_id"))
            except Exception:
                source_gid = -1
            if source_gid >= 0:
                display_id_by_source_global_id[str(source_gid)] = dict(display_record)

        self.global_person_diag.update({
            "display_stabilize_enable": True,
            "display_slot_count": len(self._person_overlay_slots),
            "display_draw_count": len(out),
            "display_relinked_this_poll": int(relinked),
            "display_created_this_poll": int(created),
            "display_alpha": float(alpha),
            "display_reid_gate_m": float(reid_gate),
            "display_hold_ms": float(hold_ms),
            "display_id_source": "global_bev_overlay_display_stabilizer",
            "display_tracks": display_tracks,
            "display_id_by_source_global_id": display_id_by_source_global_id,
        })
        return out

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_runtime_state("close")
        try:
            if self.status_writer is not None:
                timeout_s = max(0.0, float(getattr(self.args, "status_writer_stop_timeout_ms", 1000.0) or 1000.0) / 1000.0)
                self.status_writer.close(timeout_s=timeout_s)
        except Exception:
            pass
        try:
            if self.event_layer_sparse_shadow is not None:
                self.event_layer_sparse_shadow.stop(timeout_s=1.0)
        except Exception:
            pass
        try:
            if self.event_layer_presentation_worker is not None:
                self.event_layer_presentation_worker.stop(timeout_s=1.0)
        except Exception:
            pass
        try:
            if self.alignment_worker is not None:
                self.alignment_worker.stop(timeout_s=1.0)
        except Exception:
            pass
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        return super().closeEvent(event)


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent GPU global BEV fusion renderer")
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--cams", required=True)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--raw-template", default="mvs_raw_latest_{camera}")
    ap.add_argument("--calib-template", default="calib_true/field_calib_camera{camera}.json")
    ap.add_argument("--sync-mode", choices=["nearest_trigger", "nearest_frame"], default="nearest_trigger")
    ap.add_argument("--max-mvs-sync-delta-ticks", type=int, default=2)
    ap.add_argument("--mvs-sync-tolerance-ms", dest="sync_tolerance_ms", type=float, default=0.0)
    ap.add_argument("--mvs-sync-outlier-policy", choices=["hide", "warn"], default="hide")
    ap.add_argument("--mvs-sync-tick-ms-est", type=float, default=25.0)
    ap.add_argument("--presentation-delay-enable", action="store_true", default=False)
    ap.add_argument("--no-presentation-delay-enable", dest="presentation_delay_enable", action="store_false")
    ap.add_argument("--presentation-delay-ms", type=float, default=0.0)
    ap.add_argument("--field-width-m", type=float, default=33.0)
    ap.add_argument("--field-height-m", type=float, default=22.0)
    ap.add_argument("--field-origin-x-m", type=float, default=0.0)
    ap.add_argument("--field-origin-y-m", type=float, default=0.0)
    ap.add_argument("--meters-per-pixel", type=float, default=0.02)
    ap.add_argument("--display-fps", type=float, default=30.0)
    ap.add_argument("--output-fps", type=float, default=20.0)
    ap.add_argument("--window-width", type=int, default=1200)
    ap.add_argument("--window-height", type=int, default=800)
    ap.add_argument("--output-shm", default="global_bev_latest")
    ap.add_argument("--sidecar-json", default="/dev/shm/global_bev_latest.json")
    ap.add_argument("--status-json", default="sync_ipc/global_bev_status.json")
    ap.add_argument("--control-json", default="sync_ipc/global_bev_control.json")
    ap.add_argument("--runtime-state-json", default="sync_ipc/global_bev_runtime_state.json")
    ap.add_argument("--runtime-state-enable", action="store_true", default=True)
    ap.add_argument("--no-runtime-state-enable", dest="runtime_state_enable", action="store_false")
    ap.add_argument("--global-person-overlay-enable", action="store_true")
    ap.add_argument("--global-person-latest-json", default="sync_ipc/global_person_latest.json")
    ap.add_argument("--global-person-overlay-coord-mode", choices=["field_xy", "raw", "neg_y", "field_height_plus_y", "invert_y", "flip_x", "flip_xy", "swap_xy", "swap_xy_neg_y"], default="field_xy")
    ap.add_argument("--global-person-overlay-max-speed-mps", type=float, default=14.0)
    ap.add_argument("--global-person-overlay-lost-max-age-ms", type=float, default=500.0)
    ap.add_argument("--global-person-overlay-display-stabilize-enable", action="store_true", default=True)
    ap.add_argument("--no-global-person-overlay-display-stabilize-enable", dest="global_person_overlay_display_stabilize_enable", action="store_false")
    ap.add_argument("--global-person-overlay-display-alpha", type=float, default=0.45)
    ap.add_argument("--global-person-overlay-reid-gate-m", type=float, default=1.2)
    ap.add_argument("--global-person-overlay-reid-ttl-ms", type=float, default=2200.0)
    ap.add_argument("--global-person-overlay-hold-ms", type=float, default=900.0)
    ap.add_argument("--manual-hit-click-enable", action="store_true", default=False)
    ap.add_argument("--manual-hit-command-json", default="sync_ipc/manual_hit_commands.jsonl")
    ap.add_argument("--manual-hit-click-radius-px", type=float, default=42.0)
    ap.add_argument("--event-bullet-overlay-enable", action="store_true", default=False)
    ap.add_argument("--no-event-bullet-overlay-enable", dest="event_bullet_overlay_enable", action="store_false")
    ap.add_argument("--event-bullet-jsonl", default="sync_ipc/overlay_bullet_point_all.jsonl")
    ap.add_argument("--event-bullet-overlay-life-ms", type=float, default=10000.0)
    ap.add_argument("--event-bullet-overlay-max-records", type=int, default=4096)
    ap.add_argument("--event-bullet-overlay-poll-ms", type=float, default=40.0)
    ap.add_argument("--event-bullet-overlay-dilate-px", type=float, default=4.0)
    ap.add_argument("--event-bullet-overlay-line-width", type=float, default=2.5)
    ap.add_argument("--event-bullet-overlay-min-confidence", type=float, default=0.0)
    ap.add_argument("--event-bullet-overlay-max-segment-gap-ms", type=float, default=80.0)
    ap.add_argument("--event-bullet-overlay-label-enable", action="store_true", default=True)
    ap.add_argument("--no-event-bullet-overlay-label-enable", dest="event_bullet_overlay_label_enable", action="store_false")
    ap.add_argument("--event-bullet-overlay-transform-mode", choices=EVENT_BULLET_OVERLAY_TRANSFORM_MODE_ORDER, default="normal")
    ap.add_argument("--event-layer-enable", action="store_true", default=False)
    ap.add_argument("--no-event-layer-enable", dest="event_layer_enable", action="store_false")
    ap.add_argument("--event-layer-template", default="event_overlay_latest_{camera}")
    ap.add_argument("--event-layer-sidecar-template", default="event_overlay_{camera}_latest.json")
    ap.add_argument("--event-layer-backend", choices=["texture", "sparse_gl"], default="texture")
    ap.add_argument("--event-layer-sparse-sidecar-template", default="event_overlay_sparse_{camera}_history.json")
    ap.add_argument("--event-layer-sparse-max-points-draw", type=int, default=20000)
    ap.add_argument("--event-layer-sparse-candidate-scan-initial", type=int, default=4)
    ap.add_argument("--event-layer-sparse-candidate-scan-escalated", type=int, default=32)
    ap.add_argument("--event-layer-sparse-candidate-escalate-abs-ms", type=float, default=10.0)
    ap.add_argument("--event-layer-sparse-candidate-escalate-ratio", type=float, default=0.60)
    ap.add_argument("--event-layer-sparse-candidate-escalate-enable", action="store_true", default=True)
    ap.add_argument("--no-event-layer-sparse-candidate-escalate-enable", dest="event_layer_sparse_candidate_escalate_enable", action="store_false")
    ap.add_argument("--event-layer-sparse-shadow-enable", action="store_true", default=False)
    ap.add_argument("--no-event-layer-sparse-shadow-enable", dest="event_layer_sparse_shadow_enable", action="store_false")
    ap.add_argument("--event-layer-sparse-shadow-interval-ms", type=float, default=250.0)
    ap.add_argument("--event-layer-presentation-worker-enable", action="store_true", default=False)
    ap.add_argument("--no-event-layer-presentation-worker-enable", dest="event_layer_presentation_worker_enable", action="store_false")
    ap.add_argument("--event-layer-presentation-worker-mode", choices=["off", "shadow", "active"], default="off")
    ap.add_argument("--event-layer-presentation-worker-interval-ms", type=float, default=33.333)
    ap.add_argument("--event-layer-presentation-bundle-max-age-ms", type=float, default=120.0)
    ap.add_argument("--event-layer-presentation-target-tolerance-ticks", type=int, default=1)
    ap.add_argument("--event-layer-presentation-bundle-ring-size", type=int, default=8)
    ap.add_argument("--event-layer-presentation-prebuild-lookback-ticks", type=int, default=2)
    ap.add_argument("--event-layer-presentation-prebuild-lookahead-ticks", type=int, default=2)
    ap.add_argument("--event-layer-presentation-prebuild-max-per-cycle", type=int, default=1)
    ap.add_argument("--event-layer-presentation-bundle-cache-reuse-ms", type=float, default=16.0)
    ap.add_argument("--event-layer-homography-config", default="ids_8cam_fusion_config_627_runtime.json")
    ap.add_argument("--event-layer-alpha", type=float, default=0.45)
    ap.add_argument("--event-layer-gain", type=float, default=2.4)
    ap.add_argument("--event-layer-min-value", type=float, default=0.03)
    ap.add_argument("--event-layer-max-age-ms", type=float, default=150.0)
    ap.add_argument("--event-layer-sync-mode", choices=EVENT_LAYER_SYNC_MODE_ORDER, default="latest_guarded")
    ap.add_argument("--event-layer-max-sync-delta-ms", type=float, default=120.0)
    ap.add_argument("--event-layer-max-sync-delta-ticks", type=int, default=2)
    ap.add_argument("--event-layer-sync-normal-ms", type=float, default=50.0)
    ap.add_argument("--event-layer-sync-warn-ms", type=float, default=80.0)
    ap.add_argument("--event-layer-sync-hard-ms", type=float, default=80.0)
    ap.add_argument("--event-layer-hold-last-good-on-hard-outlier", action="store_true", default=True)
    ap.add_argument("--no-event-layer-hold-last-good-on-hard-outlier", dest="event_layer_hold_last_good_on_hard_outlier", action="store_false")
    ap.add_argument("--event-layer-hold-last-good-ms", type=float, default=250.0)
    ap.add_argument("--event-layer-history-enable", action="store_true", default=False)
    ap.add_argument("--no-event-layer-history-enable", dest="event_layer_history_enable", action="store_false")
    ap.add_argument("--event-layer-history-sidecar-template", default="")
    ap.add_argument("--event-layer-sync-map-ring-template", default="event_sync_map_ring_{camera}.json")
    ap.add_argument("--event-layer-fade-alpha-scale", type=float, default=0.25)
    ap.add_argument("--event-layer-color-mode", choices=EVENT_LAYER_COLOR_MODE_ORDER, default="camera")
    ap.add_argument("--event-layer-dilate-px", type=float, default=16.0)
    ap.add_argument("--event-layer-dilate-max-pixels", type=int, default=2500)
    ap.add_argument("--event-layer-dilate-stabilize-enable", action="store_true", default=True)
    ap.add_argument("--no-event-layer-dilate-stabilize-enable", dest="event_layer_dilate_stabilize_enable", action="store_false")
    ap.add_argument("--event-layer-auto-degrade-enable", action="store_true", default=True)
    ap.add_argument("--no-event-layer-auto-degrade-enable", dest="event_layer_auto_degrade_enable", action="store_false")
    ap.add_argument("--event-layer-auto-degrade-fps-threshold", type=float, default=28.0)
    ap.add_argument("--event-layer-auto-degrade-upload-p95-ms", type=float, default=8.0)
    ap.add_argument("--event-layer-auto-degrade-min-dilate-px", type=float, default=4.0)
    ap.add_argument("--event-layer-display-budget-enable", action="store_true", default=True)
    ap.add_argument("--no-event-layer-display-budget-enable", dest="event_layer_display_budget_enable", action="store_false")
    ap.add_argument("--event-layer-display-budget-ms", type=float, default=10.0)
    ap.add_argument("--event-layer-display-hold-ms", type=float, default=280.0)
    ap.add_argument("--event-layer-accumulate-enable", action="store_true", default=True)
    ap.add_argument("--no-event-layer-accumulate-enable", dest="event_layer_accumulate_enable", action="store_false")
    ap.add_argument("--event-layer-accumulate-window-ms", type=float, default=40.0)
    ap.add_argument("--event-layer-accumulate-max-frames", type=int, default=4)
    ap.add_argument("--event-layer-texture-scale", type=float, default=1.0)
    ap.add_argument("--event-layer-history-poll-interval-ms", type=float, default=0.0)
    ap.add_argument("--alignment-mode", choices=ALIGNMENT_MODE_ORDER, default="current")
    ap.add_argument("--alignment-delay-ms", type=float, default=820.0)
    ap.add_argument("--alignment-shadow-interval-ms", type=float, default=250.0)
    ap.add_argument("--alignment-hold-warn-ms", type=float, default=150.0)
    ap.add_argument("--status-interval-ms", type=float, default=1000.0)
    ap.add_argument("--async-status-writer-enable", action="store_true", default=True)
    ap.add_argument("--no-async-status-writer-enable", dest="async_status_writer_enable", action="store_false")
    ap.add_argument("--status-writer-stop-timeout-ms", type=float, default=1000.0)
    ap.add_argument("--status-debug-interval-ms", type=float, default=0.0)
    ap.add_argument("--runtime-state-event-layer-display-restore-enable", action="store_true", default=False)
    ap.add_argument("--no-runtime-state-event-layer-display-restore-enable", dest="runtime_state_event_layer_display_restore_enable", action="store_false")
    ap.add_argument("--perf-timing-window", type=int, default=180)
    ap.add_argument("--pbo-count", type=int, default=3)
    ap.add_argument("--fbo-stats-interval-ms", type=float, default=1000.0)
    ap.add_argument("--fbo-stats-stride", type=int, default=8)
    ap.add_argument("--coverage-interval-ms", type=float, default=2000.0)
    ap.add_argument("--turf-stat-fps", type=float, default=5.0)
    ap.add_argument("--person-overlay-build-fps", type=float, default=15.0)
    ap.add_argument("--overlay-text-fps", type=float, default=5.0)
    ap.add_argument("--feather-power", type=float, default=1.7)
    ap.add_argument("--debug-gray", action="store_true")
    ap.add_argument("--debug-mode", choices=sorted(DEBUG_MODE_CODES.keys()), default="warp_color")
    ap.add_argument("--texture-y-flip", action="store_true")
    ap.add_argument("--no-texture-y-flip", dest="texture_y_flip", action="store_false")
    ap.add_argument("--texture-format", choices=["r8", "luminance"], default="r8")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--black-level", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--color-gain-r", type=float, default=1.0)
    ap.add_argument("--color-gain-g", type=float, default=1.0)
    ap.add_argument("--color-gain-b", type=float, default=1.0)
    ap.add_argument("--turf-match-enable", action="store_true")
    ap.add_argument("--turf-auto-match", action="store_true")
    ap.add_argument("--turf-match-strength", type=float, default=0.28)
    ap.add_argument("--turf-target-r", type=float, default=0.18)
    ap.add_argument("--turf-target-g", type=float, default=0.42)
    ap.add_argument("--turf-target-b", type=float, default=0.15)
    ap.add_argument("--turf-target-rgb", default="")
    ap.add_argument("--turf-stat-alpha", type=float, default=0.05)
    ap.add_argument("--turf-target-alpha", type=float, default=0.02)
    ap.add_argument("--turf-gain-alpha", type=float, default=0.06)
    ap.add_argument("--turf-gain-min", type=float, default=0.72)
    ap.add_argument("--turf-gain-max", type=float, default=1.35)
    ap.add_argument("--frame-hold-ms", type=float, default=250.0)
    ap.add_argument("--unstable-copy-retries", type=int, default=3)
    ap.add_argument("--mvs-upload-budget-enable", action="store_true", default=True)
    ap.add_argument("--no-mvs-upload-budget-enable", dest="mvs_upload_budget_enable", action="store_false")
    ap.add_argument("--mvs-upload-budget-ms", type=float, default=10.0)
    ap.add_argument("--grid-enable", action="store_true")
    ap.add_argument("--field-transform-mode", choices=TRANSFORM_MODE_ORDER, default="normal")
    ap.add_argument("--field-flip-x", action="store_true")
    ap.add_argument("--field-flip-y", action="store_true")
    ap.add_argument("--coverage-grid-x", type=int, default=66)
    ap.add_argument("--coverage-grid-y", type=int, default=44)
    args = ap.parse_args()
    if args.debug_gray:
        args.debug_mode = "warp_gray"
    args.debug_mode_code = int(DEBUG_MODE_CODES.get(str(args.debug_mode), 0))
    args.field_transform_mode = _normalize_transform_mode(args.field_transform_mode, bool(args.field_flip_x), bool(args.field_flip_y))
    args.field_flip_x, args.field_flip_y = _simple_flip_from_mode(args.field_transform_mode)

    project_root = Path(args.project_root).expanduser().resolve()
    os.chdir(project_root)
    args.project_root = str(project_root)
    if not Path(args.status_json).is_absolute():
        args.status_json = str(project_root / args.status_json)
    if not Path(args.control_json).is_absolute():
        args.control_json = str(project_root / args.control_json)
    if not Path(args.runtime_state_json).is_absolute():
        args.runtime_state_json = str(project_root / args.runtime_state_json)
    if not Path(args.sync_root).is_absolute():
        args.sync_root = str(project_root / args.sync_root)
    if not Path(args.global_person_latest_json).is_absolute():
        args.global_person_latest_json = str(project_root / args.global_person_latest_json)
    if not Path(args.manual_hit_command_json).is_absolute():
        args.manual_hit_command_json = str(project_root / args.manual_hit_command_json)
    if str(getattr(args, "event_layer_homography_config", "") or "").strip() and not Path(args.event_layer_homography_config).is_absolute():
        args.event_layer_homography_config = str(project_root / args.event_layer_homography_config)
    if bool(getattr(args, "runtime_state_enable", True)):
        state_path = Path(args.runtime_state_json)
        try:
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8-sig"))
                if isinstance(state, dict):
                    base_keys = (
                        "field_transform_mode", "debug_mode", "texture_y_flip", "grid_enable",
                        "global_person_overlay_coord_mode", "event_bullet_overlay_transform_mode",
                        "gain", "black_level", "gamma",
                        "turf_match_enable", "turf_auto_match", "turf_match_strength", "frame_hold_ms",
                        "mvs_upload_budget_enable", "mvs_upload_budget_ms",
                    )
                    event_layer_display_keys = (
                        "event_layer_enable", "event_layer_alpha", "event_layer_gain", "event_layer_min_value", "event_layer_max_age_ms",
                        "event_layer_backend", "event_layer_sparse_sidecar_template", "event_layer_sparse_max_points_draw",
                        "event_layer_fade_alpha_scale",
                        "event_layer_sync_normal_ms", "event_layer_sync_warn_ms", "event_layer_sync_hard_ms",
                        "event_layer_hold_last_good_on_hard_outlier", "event_layer_hold_last_good_ms",
                        "event_layer_display_budget_enable", "event_layer_display_budget_ms", "event_layer_display_hold_ms",
                        "event_layer_color_mode",
                        "event_layer_dilate_px", "event_layer_dilate_max_pixels",
                        "event_layer_auto_degrade_enable", "event_layer_auto_degrade_fps_threshold",
                        "event_layer_auto_degrade_upload_p95_ms", "event_layer_auto_degrade_min_dilate_px",
                        "event_layer_accumulate_enable", "event_layer_accumulate_window_ms",
                        "event_layer_accumulate_max_frames", "event_layer_texture_scale",
                        "event_layer_history_poll_interval_ms", "event_layer_sync_map_ring_template",
                    )
                    restore_event_layer_display = bool(getattr(args, "runtime_state_event_layer_display_restore_enable", False))
                    restored_keys: List[str] = []
                    skipped_keys: List[str] = []
                    restore_keys = list(base_keys) + (list(event_layer_display_keys) if restore_event_layer_display else [])
                    for key in restore_keys:
                        if key in state:
                            setattr(args, key, state.get(key))
                            restored_keys.append(str(key))
                    if not restore_event_layer_display:
                        skipped_keys = [str(key) for key in event_layer_display_keys if key in state]
                    cg = state.get("color_gain")
                    if isinstance(cg, (list, tuple)) and len(cg) >= 3:
                        args.color_gain_r = float(cg[0])
                        args.color_gain_g = float(cg[1])
                        args.color_gain_b = float(cg[2])
                    args.runtime_state_loaded_path = str(state_path)
                    args.runtime_state_restored_keys = list(restored_keys)
                    args.runtime_state_skipped_event_layer_display_keys = list(skipped_keys)
                    print(
                        f"[global_bev] loaded runtime state: {state_path} "
                        f"restored={len(restored_keys)} skipped_event_layer_display={len(skipped_keys)}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[global_bev][WARN] failed to load runtime state {state_path}: {type(exc).__name__}: {exc}", flush=True)
    if str(args.debug_mode) not in DEBUG_MODE_CODES:
        args.debug_mode = "warp_color"
    args.debug_mode_code = int(DEBUG_MODE_CODES.get(str(args.debug_mode), 0))
    args.field_transform_mode = _normalize_transform_mode(args.field_transform_mode, bool(args.field_flip_x), bool(args.field_flip_y))
    args.field_flip_x, args.field_flip_y = _simple_flip_from_mode(args.field_transform_mode)
    args.max_mvs_sync_delta_ticks = max(0, int(float(getattr(args, "max_mvs_sync_delta_ticks", 2) or 2)))
    args.sync_tolerance_ms = max(0.0, float(getattr(args, "sync_tolerance_ms", 0.0) or 0.0))
    args.mvs_sync_outlier_policy = str(getattr(args, "mvs_sync_outlier_policy", "hide") or "hide").strip().lower()
    if args.mvs_sync_outlier_policy not in ("hide", "warn"):
        args.mvs_sync_outlier_policy = "hide"
    args.mvs_sync_tick_ms_est = max(0.0, float(getattr(args, "mvs_sync_tick_ms_est", 25.0) or 25.0))
    args.presentation_delay_enable = bool(_safe_bool(getattr(args, "presentation_delay_enable", False), False))
    args.presentation_delay_ms = max(0.0, float(getattr(args, "presentation_delay_ms", 0.0) or 0.0))
    args.event_bullet_overlay_transform_mode = str(getattr(args, "event_bullet_overlay_transform_mode", "normal") or "normal").strip().lower().replace("-", "_")
    if args.event_bullet_overlay_transform_mode not in set(EVENT_BULLET_OVERLAY_TRANSFORM_MODE_ORDER):
        args.event_bullet_overlay_transform_mode = "normal"
    args.event_layer_enable = bool(_safe_bool(getattr(args, "event_layer_enable", False), False))
    args.event_layer_alpha = _clip01(float(getattr(args, "event_layer_alpha", 0.45) or 0.0))
    args.event_layer_gain = max(0.01, float(getattr(args, "event_layer_gain", 2.4) or 2.4))
    args.event_layer_min_value = max(0.0, min(0.95, float(getattr(args, "event_layer_min_value", 0.03) or 0.03)))
    args.event_layer_max_age_ms = max(1.0, float(getattr(args, "event_layer_max_age_ms", 150.0) or 150.0))
    if bool(args.presentation_delay_enable) and bool(_safe_bool(getattr(args, "event_layer_history_enable", False), False)):
        history_poll_ms = max(0.0, float(getattr(args, "event_layer_history_poll_interval_ms", 0.0) or 0.0))
        min_history_age_ms = max(2500.0, float(args.presentation_delay_ms) + max(1000.0, history_poll_ms * 2.0 + 500.0))
        if float(args.event_layer_max_age_ms) < min_history_age_ms:
            print(
                f"[global_bev] lift event_layer_max_age_ms {args.event_layer_max_age_ms:.0f}->{min_history_age_ms:.0f} "
                f"for delayed presentation history",
                flush=True,
            )
            args.event_layer_max_age_ms = float(min_history_age_ms)
    args.event_layer_sync_mode = _event_layer_sync_mode(getattr(args, "event_layer_sync_mode", "latest_guarded"))
    args.event_layer_max_sync_delta_ms = max(1.0, float(getattr(args, "event_layer_max_sync_delta_ms", 120.0) or 120.0))
    args.event_layer_max_sync_delta_ticks = max(0, int(float(getattr(args, "event_layer_max_sync_delta_ticks", 2) or 2)))
    args.event_layer_sync_normal_ms = max(1.0, float(getattr(args, "event_layer_sync_normal_ms", args.event_layer_max_sync_delta_ms) or args.event_layer_max_sync_delta_ms))
    args.event_layer_sync_warn_ms = max(float(args.event_layer_sync_normal_ms), float(getattr(args, "event_layer_sync_warn_ms", 80.0) or 80.0))
    args.event_layer_sync_hard_ms = max(float(args.event_layer_sync_warn_ms), float(getattr(args, "event_layer_sync_hard_ms", args.event_layer_sync_warn_ms) or args.event_layer_sync_warn_ms))
    args.event_layer_hold_last_good_on_hard_outlier = bool(_safe_bool(getattr(args, "event_layer_hold_last_good_on_hard_outlier", True), True))
    args.event_layer_hold_last_good_ms = max(0.0, float(getattr(args, "event_layer_hold_last_good_ms", 250.0) or 0.0))
    args.event_layer_history_enable = bool(_safe_bool(getattr(args, "event_layer_history_enable", False), False))
    args.event_layer_history_sidecar_template = str(getattr(args, "event_layer_history_sidecar_template", "") or "")
    args.event_layer_sync_map_ring_template = str(getattr(args, "event_layer_sync_map_ring_template", "event_sync_map_ring_{camera}.json") or "event_sync_map_ring_{camera}.json")
    args.event_layer_fade_alpha_scale = _clip01(float(getattr(args, "event_layer_fade_alpha_scale", 0.25) or 0.25))
    args.event_layer_color_mode = _event_layer_color_mode(getattr(args, "event_layer_color_mode", "camera"))
    args.event_layer_dilate_px = max(0.0, min(64.0, float(getattr(args, "event_layer_dilate_px", 16.0) or 0.0)))
    args.event_layer_dilate_max_pixels = max(0, int(float(getattr(args, "event_layer_dilate_max_pixels", 2500) or 0)))
    args.event_layer_auto_degrade_enable = bool(_safe_bool(getattr(args, "event_layer_auto_degrade_enable", True), True))
    args.event_layer_auto_degrade_fps_threshold = max(1.0, float(getattr(args, "event_layer_auto_degrade_fps_threshold", 28.0) or 28.0))
    args.event_layer_auto_degrade_upload_p95_ms = max(0.1, float(getattr(args, "event_layer_auto_degrade_upload_p95_ms", 8.0) or 8.0))
    args.event_layer_auto_degrade_min_dilate_px = max(0.0, min(64.0, float(getattr(args, "event_layer_auto_degrade_min_dilate_px", 4.0) or 4.0)))
    args.event_layer_display_budget_enable = bool(_safe_bool(getattr(args, "event_layer_display_budget_enable", True), True))
    args.event_layer_display_budget_ms = max(0.0, float(getattr(args, "event_layer_display_budget_ms", 10.0) or 0.0))
    args.event_layer_display_hold_ms = max(0.0, float(getattr(args, "event_layer_display_hold_ms", getattr(args, "frame_hold_ms", 280.0)) or 0.0))
    args.event_layer_accumulate_enable = bool(_safe_bool(getattr(args, "event_layer_accumulate_enable", True), True))
    args.event_layer_accumulate_window_ms = max(1.0, float(getattr(args, "event_layer_accumulate_window_ms", 40.0) or 40.0))
    args.event_layer_accumulate_max_frames = max(1, min(16, int(float(getattr(args, "event_layer_accumulate_max_frames", 4) or 4))))
    args.event_layer_texture_scale = max(0.25, min(1.0, float(getattr(args, "event_layer_texture_scale", 1.0) or 1.0)))
    args.event_layer_history_poll_interval_ms = max(0.0, min(1000.0, float(getattr(args, "event_layer_history_poll_interval_ms", 0.0) or 0.0)))
    args.event_layer_backend = "sparse_gl" if str(getattr(args, "event_layer_backend", "texture") or "texture").strip().lower().replace("-", "_") in {"sparse", "sparse_points", "opengl_sparse", "opengl_sparse_points", "sparse_gl"} else "texture"
    args.event_layer_sparse_max_points_draw = max(1, min(200000, int(float(getattr(args, "event_layer_sparse_max_points_draw", 20000) or 20000))))
    args.event_layer_sparse_candidate_scan_initial = max(1, min(128, int(float(getattr(args, "event_layer_sparse_candidate_scan_initial", 4) or 4))))
    args.event_layer_sparse_candidate_scan_escalated = max(
        int(args.event_layer_sparse_candidate_scan_initial),
        min(256, int(float(getattr(args, "event_layer_sparse_candidate_scan_escalated", 32) or 32))),
    )
    args.event_layer_sparse_candidate_escalate_abs_ms = max(0.0, float(getattr(args, "event_layer_sparse_candidate_escalate_abs_ms", 10.0) or 10.0))
    args.event_layer_sparse_candidate_escalate_ratio = max(0.05, min(1.0, float(getattr(args, "event_layer_sparse_candidate_escalate_ratio", 0.60) or 0.60)))
    args.event_layer_sparse_candidate_escalate_enable = bool(_safe_bool(getattr(args, "event_layer_sparse_candidate_escalate_enable", True), True))
    args.event_layer_sparse_shadow_enable = bool(_safe_bool(getattr(args, "event_layer_sparse_shadow_enable", False), False))
    args.event_layer_sparse_shadow_interval_ms = max(50.0, min(5000.0, float(getattr(args, "event_layer_sparse_shadow_interval_ms", 250.0) or 250.0)))
    args.event_layer_presentation_worker_enable = bool(_safe_bool(getattr(args, "event_layer_presentation_worker_enable", False), False))
    args.event_layer_presentation_worker_mode = str(getattr(args, "event_layer_presentation_worker_mode", "off") or "off").strip().lower().replace("-", "_")
    if args.event_layer_presentation_worker_mode not in {"off", "shadow", "active"}:
        args.event_layer_presentation_worker_mode = "shadow"
    if not bool(args.event_layer_presentation_worker_enable):
        args.event_layer_presentation_worker_mode = "off"
    args.event_layer_presentation_worker_interval_ms = max(5.0, min(1000.0, float(getattr(args, "event_layer_presentation_worker_interval_ms", 33.333) or 33.333)))
    args.event_layer_presentation_bundle_max_age_ms = max(1.0, min(2000.0, float(getattr(args, "event_layer_presentation_bundle_max_age_ms", 120.0) or 120.0)))
    args.event_layer_presentation_target_tolerance_ticks = max(0, min(8, int(float(getattr(args, "event_layer_presentation_target_tolerance_ticks", 1) or 1))))
    args.event_layer_presentation_bundle_ring_size = max(1, min(64, int(float(getattr(args, "event_layer_presentation_bundle_ring_size", 8) or 8))))
    args.event_layer_presentation_prebuild_lookback_ticks = max(0, min(16, int(float(getattr(args, "event_layer_presentation_prebuild_lookback_ticks", 2) or 2))))
    args.event_layer_presentation_prebuild_lookahead_ticks = max(0, min(16, int(float(getattr(args, "event_layer_presentation_prebuild_lookahead_ticks", 2) or 2))))
    args.event_layer_presentation_prebuild_max_per_cycle = max(1, min(8, int(float(getattr(args, "event_layer_presentation_prebuild_max_per_cycle", 1) or 1))))
    args.event_layer_presentation_bundle_cache_reuse_ms = max(0.0, min(1000.0, float(getattr(args, "event_layer_presentation_bundle_cache_reuse_ms", 16.0) or 16.0)))
    args.event_layer_dilate_stabilize_enable = bool(_safe_bool(getattr(args, "event_layer_dilate_stabilize_enable", True), True))
    args.alignment_mode = _alignment_mode(getattr(args, "alignment_mode", "current"))
    if args.alignment_mode == "active":
        print("[global_bev][WARN] alignment active requested but gated; using shadow mode", flush=True)
        args.alignment_mode = "shadow"
    args.alignment_delay_ms = max(0.0, float(getattr(args, "alignment_delay_ms", getattr(args, "presentation_delay_ms", 820.0)) or 0.0))
    args.alignment_shadow_interval_ms = max(50.0, min(5000.0, float(getattr(args, "alignment_shadow_interval_ms", 250.0) or 250.0)))
    args.alignment_hold_warn_ms = max(0.0, float(getattr(args, "alignment_hold_warn_ms", 150.0) or 150.0))
    args.runtime_state_event_layer_display_restore_enable = bool(_safe_bool(getattr(args, "runtime_state_event_layer_display_restore_enable", False), False))
    args.status_debug_interval_ms = max(0.0, float(getattr(args, "status_debug_interval_ms", 0.0) or 0.0))
    args.async_status_writer_enable = bool(_safe_bool(getattr(args, "async_status_writer_enable", True), True))
    args.status_writer_stop_timeout_ms = max(0.0, min(5000.0, float(getattr(args, "status_writer_stop_timeout_ms", 1000.0) or 1000.0)))
    args.mvs_upload_budget_enable = bool(_safe_bool(getattr(args, "mvs_upload_budget_enable", True), True))
    args.mvs_upload_budget_ms = max(0.0, float(getattr(args, "mvs_upload_budget_ms", 10.0) or 0.0))
    args.manual_hit_click_radius_px = max(4.0, min(160.0, float(getattr(args, "manual_hit_click_radius_px", 42.0) or 42.0)))

    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication([])
    w = GlobalBevWidget(args)
    def _shutdown_global_bev() -> None:
        try:
            if w.status_writer is not None:
                w.status_writer.close(timeout_s=max(0.0, float(args.status_writer_stop_timeout_ms) / 1000.0))
        except Exception:
            pass
        try:
            if w.alignment_worker is not None:
                w.alignment_worker.stop(timeout_s=1.0)
        except Exception:
            pass
        try:
            if w.event_layer_presentation_worker is not None:
                w.event_layer_presentation_worker.stop(timeout_s=1.0)
        except Exception:
            pass
    app.aboutToQuit.connect(_shutdown_global_bev)
    w.show()
    print(
        f"[global_bev] started cams={','.join(w.cams)} field={args.field_width_m}x{args.field_height_m}m "
        f"mpp={args.meters_per_pixel} output={w.out_w}x{w.out_h} mode={args.debug_mode} "
        f"transform={args.field_transform_mode} bullet_transform={args.event_bullet_overlay_transform_mode} "
        f"event_layer={int(bool(args.event_layer_enable))}/{args.event_layer_alpha:.2f}/{args.event_layer_max_age_ms:.0f}ms/{args.event_layer_color_mode}/dil{args.event_layer_dilate_px:.0f}px "
        f"event_backend={args.event_layer_backend} event_tex_scale={args.event_layer_texture_scale:.2f} event_hist_poll={args.event_layer_history_poll_interval_ms:.0f}ms "
        f"event_presentation_worker={int(bool(args.event_layer_presentation_worker_enable))}/{args.event_layer_presentation_worker_mode}@{args.event_layer_presentation_worker_interval_ms:.1f}ms "
        f"ring={args.event_layer_presentation_bundle_ring_size}/lb{args.event_layer_presentation_prebuild_lookback_ticks}/la{args.event_layer_presentation_prebuild_lookahead_ticks}/max{args.event_layer_presentation_prebuild_max_per_cycle} "
        f"mvs_upload_budget={int(bool(args.mvs_upload_budget_enable))}/{args.mvs_upload_budget_ms:.1f}ms "
        f"event_display_budget={int(bool(args.event_layer_display_budget_enable))}/{args.event_layer_display_budget_ms:.1f}ms "
        f"alignment={args.alignment_mode}/{args.alignment_delay_ms:.0f}ms/{args.alignment_shadow_interval_ms:.0f}ms "
        f"texture={args.texture_format} yflip={int(bool(args.texture_y_flip))} "
        f"control={args.control_json} shm={_shm_path(str(args.output_shm))}",
        flush=True,
    )
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
