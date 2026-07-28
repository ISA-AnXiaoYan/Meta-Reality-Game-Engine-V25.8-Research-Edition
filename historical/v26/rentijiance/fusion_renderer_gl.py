#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_renderer_gl.py

Standalone Qt/OpenGL fusion preview for the MVS + event-camera system.

Purpose
-------
This renderer replaces cv2.imshow for preview only.  It does not capture cameras
and does not change the hardware synchronization path.  The MVS process publishes
raw MVS frames to /dev/shm/mvs_latest_{camera}.mmap; the event process already
publishes event overlays to /dev/shm/event_overlay_latest_{camera}.mmap.

Rendering model
---------------
- MVS frame:     background texture, normally 25 fps.
- Event overlay: foreground texture, normally 250 fps.
- Homography:   applied in the fragment shader, event -> MVS.
- Blending:     done by the GPU, not by OpenCV.

Dependencies
------------
pip install PySide6 PyOpenGL

Example
-------
python3 fusion_renderer_gl.py \
  --config hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json \
  --cams A,B,C,D,E \
  --preview-fps 120
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import datetime
import json
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from latest_frame_shm import SharedFrameReader
except Exception:
    try:
        _parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        from latest_frame_shm import SharedFrameReader
    except Exception as exc:  # pragma: no cover
        print(f"[fusion_gl][ERROR] cannot import latest_frame_shm.SharedFrameReader: {exc}", file=sys.stderr)
        raise

try:
    from PySide6.QtCore import QTimer, Qt, QPointF
    from PySide6.QtGui import QColor, QFont, QPainter, QSurfaceFormat, QPolygonF, QPen
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import QApplication
except Exception as exc:  # pragma: no cover
    print("[fusion_gl][ERROR] PySide6 is required. Install with: pip install PySide6", file=sys.stderr)
    raise

try:
    from OpenGL import GL
    from OpenGL.GL import shaders
except Exception as exc:  # pragma: no cover
    print("[fusion_gl][ERROR] PyOpenGL is required. Install with: pip install PyOpenGL", file=sys.stderr)
    raise




def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _record_cfg_enabled(cfg: Dict[str, Any]) -> bool:
    return _as_bool((cfg or {}).get("enable", False), False)


def _record_command_file(record_cfg: Dict[str, Any], sync_cfg: Dict[str, Any]) -> str:
    root = str((sync_cfg or {}).get("root", "./sync_ipc") or "./sync_ipc")
    default = os.path.join(root, "record_control.json")
    return os.path.abspath(str((record_cfg or {}).get("command_file", default) or default))


def _record_status_file(record_cfg: Dict[str, Any], sync_cfg: Dict[str, Any]) -> str:
    root = str((sync_cfg or {}).get("root", "./sync_ipc") or "./sync_ipc")
    default = os.path.join(root, "record_status.json")
    return os.path.abspath(str((record_cfg or {}).get("status_file", default) or default))


def _sync_template_path(sync_cfg: Dict[str, Any], key: str, camera: str, default_template: str) -> str:
    root = str((sync_cfg or {}).get("root", "./sync_ipc") or "./sync_ipc")
    template = str((sync_cfg or {}).get(key, default_template) or default_template)
    path = template.format(root=root, camera=str(camera), cam=str(camera), id=str(camera))
    return os.path.abspath(os.path.expanduser(path))


def _safe_session_id(session_id: Optional[str] = None) -> str:
    raw = str(session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:96] or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_record_targets(targets: Optional[Any]) -> List[str]:
    if targets is None:
        return []
    if isinstance(targets, str):
        raw_items = [x.strip() for x in targets.replace(";", ",").split(",")]
    elif isinstance(targets, (list, tuple, set)):
        raw_items = [str(x).strip() for x in targets]
    else:
        raw_items = [str(targets).strip()]
    out: List[str] = []
    for item in raw_items:
        if not item:
            continue
        norm = item.lower().replace("-", "_").strip()
        if norm in {"all", "both", "*"}:
            norm = "all"
        elif norm in {"event", "events", "event_raw", "raw", "raw_event", "eventcam", "event_camera"}:
            norm = "event_raw"
        elif norm in {"mvs", "mvs_raw", "mvs_video", "video", "hik", "hik_mvs"}:
            norm = "mvs_raw"
        if norm not in out:
            out.append(norm)
    return out


def _write_one_key_record_command(
    command_file: str,
    cmd: str,
    session_id: Optional[str] = None,
    source: str = "fusion_gl_renderer",
    targets: Optional[Any] = None,
    mode: str = "",
) -> bool:
    payload: Dict[str, Any] = {
        "seq": int(time.time() * 1000),
        "cmd": str(cmd).lower().strip(),
        "wall_us": _now_us(),
        "source": str(source),
    }
    norm_targets = _normalize_record_targets(targets)
    if norm_targets:
        payload["targets"] = norm_targets
    if mode:
        payload["mode"] = str(mode)
    if session_id:
        payload["session_id"] = _safe_session_id(session_id)
    try:
        path = Path(command_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(
            f"[fusion_gl][onekey_record] command {payload['cmd']} targets={payload.get('targets','all')} "
            f"-> {path} session={payload.get('session_id','')}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[fusion_gl][onekey_record][ERROR] write command failed: {command_file}: {exc}", flush=True)
        return False

# ----------------------------- config helpers -----------------------------

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


def _json_error_context(path: Path, exc: json.JSONDecodeError, radius: int = 4) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return f"{path}: {exc}"
    lo = max(0, int(exc.lineno) - radius - 1)
    hi = min(len(lines), int(exc.lineno) + radius)
    out = [f"{path}: JSON syntax error: {exc.msg} at line {exc.lineno}, column {exc.colno}"]
    for i in range(lo, hi):
        mark = " >>>" if (i + 1) == exc.lineno else "    "
        out.append(f"{mark} {i + 1:5d}: {lines[i]}")
    return "\n".join(out)


def _candidate_config_paths(config_arg: str) -> List[Path]:
    """Resolve config robustly for the project's two-layer layout.

    The project usually has both:
      project_root/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json
      project_root/rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json

    In practice the renderer is often launched from rentijiance/.  If the local
    JSON is stale or accidentally edited into invalid JSON, try the parent
    project-root JSON before failing.
    """
    raw = Path(config_arg).expanduser()
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        # Prefer the project-root config over rentijiance/local copies.
        # The renderer is often launched as:
        #   python3 test607/rentijiance/fusion_renderer_gl.py --config hik_...json
        # from a parent directory.  In that case script_dir/raw points to
        # test607/rentijiance/hik_...json, whose relative ./sync_ipc would be
        # wrong for human_result_*.jsonl and hit_candidate_*.jsonl.
        if script_dir.name == "rentijiance":
            candidates.append(script_dir.parent / raw.name)
            candidates.append(script_dir.parent / raw)
        if cwd.name == "rentijiance":
            candidates.append(cwd.parent / raw.name)
            candidates.append(cwd.parent / raw)
        candidates.append(cwd / raw)
        candidates.append(script_dir / raw)
    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_config(config_arg: str) -> Tuple[Dict[str, Any], str]:
    errors: List[str] = []
    missing: List[str] = []
    for p in _candidate_config_paths(config_arg):
        if not p.exists():
            missing.append(str(p))
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg, str(p.resolve())
        except json.JSONDecodeError as exc:
            errors.append(_json_error_context(p, exc))
        except Exception as exc:
            errors.append(f"{p}: cannot read config: {exc}")
    msg = ["Could not load a valid JSON config."]
    if errors:
        msg.append("\nTried existing config files but they had errors:")
        msg.extend(errors)
    if missing:
        msg.append("\nAlso looked for these paths:")
        msg.extend([f"  - {x}" for x in missing])
    raise RuntimeError("\n".join(msg))


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _routes_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    routes = cfg.get("routes")
    if isinstance(routes, list) and routes:
        return [r for r in routes if isinstance(r, dict) and _as_bool(r.get("enabled", True), True)]
    active = str(cfg.get("active_profile", "") or "")
    profiles = cfg.get("profiles") or {}
    prof = profiles.get(active) if isinstance(profiles, dict) else None
    if isinstance(prof, dict):
        routes = prof.get("routes") or []
        return [r for r in routes if isinstance(r, dict) and _as_bool(r.get("enabled", True), True)]
    return []


def _parse_rgb_color(value: Any) -> Tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        vals = [float(value[i]) for i in range(3)]
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        return float(vals[0]), float(vals[1]), float(vals[2])
    s = str(value or "yellow").strip().lower()
    table = {
        "yellow": (1.0, 1.0, 0.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0),
        "orange": (1.0, 0.55, 0.0),
    }
    return table.get(s, table["yellow"])




def _person_palette_bgr(idx: int) -> Tuple[int, int, int]:
    # BGR palette chosen to match the old OpenCV-style human region overlay:
    # high-contrast magenta/cyan/yellow regions, not green event tint.
    palette = [
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (0, 255, 255),   # yellow
        (0, 128, 255),   # orange
        (255, 128, 0),   # blue-ish
        (180, 255, 180),
        (255, 180, 255),
        (128, 255, 255),
        (255, 128, 128),
        (160, 160, 255),
        (180, 220, 80),
        (0, 220, 255),
    ]
    return palette[int(idx) % len(palette)]


def _person_qcolor(stable_id: Any, alpha: int = 255) -> QColor:
    try:
        idx = max(0, int(stable_id) - 1)
    except Exception:
        idx = 0
    b, g, r = _person_palette_bgr(idx)
    return QColor(int(r), int(g), int(b), int(max(0, min(255, alpha))))


def _extract_mask_polygons(person: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    # human_result has used a few field names across versions. Accept all of
    # them so the OpenGL renderer draws the actual recognized region whenever
    # polygons are available, instead of silently showing nothing.
    raw = (
        person.get("mask_polygons")
        or person.get("mask_polygon")
        or person.get("polygons")
        or person.get("segments")
        or person.get("segment")
        or person.get("contours")
        or []
    )
    polys: List[List[Tuple[float, float]]] = []
    if not isinstance(raw, list):
        return polys
    # Accept either [[x,y], ...] or [ [[x,y], ...], [[x,y], ...] ].
    if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 2 and not isinstance(raw[0][0], (list, tuple)):
        raw = [raw]
    for poly in raw:
        if not isinstance(poly, (list, tuple)):
            continue
        pts: List[Tuple[float, float]] = []
        for pt in poly:
            if isinstance(pt, dict):
                if "x" in pt and "y" in pt:
                    try:
                        pts.append((float(pt["x"]), float(pt["y"])))
                    except Exception:
                        pass
                continue
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    pts.append((float(pt[0]), float(pt[1])))
                except Exception:
                    pass
        if len(pts) >= 3:
            polys.append(pts)
    return polys

def _homography_from_overlay_cfg(rcfg: Dict[str, Any], event_shape: Tuple[int, int], mvs_shape: Tuple[int, int]) -> np.ndarray:
    """Return event->MVS homography in top-left image coordinates."""
    H_value = None
    hs = rcfg.get("homographies_event_to_mvs")
    if isinstance(hs, dict) and hs:
        active = str(rcfg.get("active_homography", "flight") or "flight")
        H_value = hs.get(active)
        if H_value is None and len(hs) == 1:
            H_value = next(iter(hs.values()))
    if H_value is None:
        H_value = rcfg.get("homography_event_to_mvs")
    if H_value is not None:
        H = np.asarray(H_value, dtype=np.float64)
        if H.shape == (3, 3) and np.all(np.isfinite(H)):
            return H

    # Fallback: reproduce the old center_offset/scale placement as a homography.
    ev_h, ev_w = event_shape
    mv_h, mv_w = mvs_shape
    scale = float(rcfg.get("scale", 1.0) or 1.0)
    scale = max(0.01, scale)
    xoff = float(rcfg.get("x", rcfg.get("offset_x", 0)) or 0)
    yoff = float(rcfg.get("y", rcfg.get("offset_y", 0)) or 0)
    mode = str(rcfg.get("position_mode", "center_offset") or "center_offset").lower()
    if mode in ("top_left", "topleft", "xy"):
        x0 = xoff
        y0 = yoff
    else:
        x0 = (mv_w - ev_w * scale) * 0.5 + xoff
        y0 = (mv_h - ev_h * scale) * 0.5 + yoff
    return np.array([[scale, 0.0, x0], [0.0, scale, y0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ----------------------------- shared readers ------------------------------

class LazySharedReader:
    def __init__(self, name: str):
        self.name = str(name)
        self.reader: Optional[SharedFrameReader] = None
        self.last_try_s = 0.0
        self.last_error = "not opened"

    def read_latest(self) -> Optional[Dict[str, Any]]:
        if self.reader is None:
            now = time.time()
            if now - self.last_try_s < 0.5:
                return None
            self.last_try_s = now
            try:
                self.reader = SharedFrameReader(self.name)
                self.last_error = ""
                print(f"[fusion_gl] opened shared frame: {self.name}", flush=True)
            except Exception as exc:
                self.last_error = f"open failed: {type(exc).__name__}: {exc}"
                return None
        try:
            item = self.reader.read_latest()
            self.last_error = ""
            return item
        except Exception as exc:
            self.last_error = f"read failed: {type(exc).__name__}: {exc}"
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
            return None

    def close(self) -> None:
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None


class JsonlTailer:
    """Small bounded JSONL tailer used by the renderer for human/hit overlays.

    The old preview opened JSONL files at offset 0 and parsed the full history.
    With mask polygons and five cameras this can stall the Qt/OpenGL thread for
    seconds when the renderer starts late or a file is rotated.  For preview we
    only need the most recent records, so open near EOF, pre-load a bounded tail,
    and cap per-tick parsing work.
    """

    def __init__(self, path: str, max_records: int = 256, *, start_at_end: bool = True, startup_tail_records: Optional[int] = None, max_lines_per_poll: int = 256, max_bytes_per_poll: int = 1_000_000):
        self.path = os.path.abspath(str(path))
        self.max_records = int(max_records)
        self.start_at_end = bool(start_at_end)
        self.startup_tail_records = int(startup_tail_records if startup_tail_records is not None else max_records)
        self.max_lines_per_poll = int(max(16, max_lines_per_poll))
        self.max_bytes_per_poll = int(max(64 * 1024, max_bytes_per_poll))
        self.fp = None
        self.offset = 0
        self.records: Deque[Dict[str, Any]] = deque(maxlen=max(1, self.max_records))
        self.last_error = "not opened"
        self.last_try_s = 0.0
        self._inode_key = None

    def _parse_json_line(self, line: str) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _load_tail_records(self, st: os.stat_result) -> None:
        if self.startup_tail_records <= 0 or st.st_size <= 0:
            return
        try:
            need = max(1, int(self.startup_tail_records))
            # Read a bounded suffix.  Human JSONL lines are usually <2KB, but
            # leave enough room for bigger masks without ever reading the full file.
            max_bytes = min(int(st.st_size), max(256 * 1024, need * 8192))
            with open(self.path, "rb") as bf:
                bf.seek(max(0, st.st_size - max_bytes), os.SEEK_SET)
                data = bf.read(max_bytes)
            lines = data.splitlines()
            if st.st_size > max_bytes and lines:
                lines = lines[1:]  # first line may be partial
            for raw in lines[-need:]:
                try:
                    obj = self._parse_json_line(raw.decode("utf-8", errors="replace"))
                    if obj is not None:
                        self.records.append(obj)
                except Exception:
                    continue
        except Exception:
            # Tail pre-load is best effort.  poll() will still read new records.
            pass

    def _open_from_current_position(self) -> bool:
        try:
            st = os.stat(self.path)
            self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
            self._inode_key = (st.st_dev, st.st_ino)
            if self.start_at_end:
                self._load_tail_records(st)
                self.fp.seek(0, os.SEEK_END)
                self.offset = self.fp.tell()
            else:
                self.offset = 0
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"open failed: {type(exc).__name__}: {exc}"
            return False

    def _open_if_needed(self) -> bool:
        if self.fp is not None:
            return True
        now = time.time()
        if now - self.last_try_s < 0.5:
            return False
        self.last_try_s = now
        return self._open_from_current_position()

    def poll(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._open_if_needed():
            return out
        try:
            # Handle truncate/rotation on restart.
            st = os.stat(self.path)
            key = (st.st_dev, st.st_ino)
            if key != self._inode_key or st.st_size < self.offset:
                try:
                    self.fp.close()
                except Exception:
                    pass
                self.fp = None
                self.records.clear()
                self._open_from_current_position()
                return out
            self.fp.seek(self.offset)
            lines = 0
            bytes_read = 0
            while lines < self.max_lines_per_poll and bytes_read < self.max_bytes_per_poll:
                line = self.fp.readline()
                if not line:
                    break
                self.offset = self.fp.tell()
                lines += 1
                bytes_read += len(line.encode("utf-8", errors="ignore"))
                obj = self._parse_json_line(line)
                if obj is not None:
                    self.records.append(obj)
                    out.append(obj)
            self.last_error = ""
            return out
        except Exception as exc:
            self.last_error = f"poll failed: {type(exc).__name__}: {exc}"
            try:
                if self.fp is not None:
                    self.fp.close()
            except Exception:
                pass
            self.fp = None
            return out

    def close(self) -> None:
        try:
            if self.fp is not None:
                self.fp.close()
        except Exception:
            pass
        self.fp = None


class TriggerCsvTailer:
    """Bounded tailer for event_trigger_{camera}.csv diagnostics and sync lookup.

    Current schema is normally:
      all_index,sync_index,event_ts_us,polarity,trigger_id,slice_ts_us,wall_time_us
    but this parser remains tolerant of older numeric-only rows.
    """

    def __init__(self, path: str, keep: int = 128, *, start_at_end: bool = True, startup_tail_rows: Optional[int] = None, max_lines_per_poll: int = 512):
        self.path = os.path.abspath(str(path))
        self.keep = int(max(4, keep))
        self.start_at_end = bool(start_at_end)
        self.startup_tail_rows = int(startup_tail_rows if startup_tail_rows is not None else self.keep)
        self.max_lines_per_poll = int(max(32, max_lines_per_poll))
        self.fp = None
        self.offset = 0
        self.rows: Deque[Dict[str, Any]] = deque(maxlen=self.keep)
        self.last_error = "not opened"
        self.last_try_s = 0.0
        self._inode_key = None
        self._line_no = 0

    def _load_tail_rows(self, st: os.stat_result) -> None:
        if self.startup_tail_rows <= 0 or st.st_size <= 0:
            return
        try:
            need = max(1, int(self.startup_tail_rows))
            max_bytes = min(int(st.st_size), max(128 * 1024, need * 256))
            with open(self.path, "rb") as bf:
                bf.seek(max(0, st.st_size - max_bytes), os.SEEK_SET)
                data = bf.read(max_bytes)
            lines = data.splitlines()
            if st.st_size > max_bytes and lines:
                lines = lines[1:]
            # Approximate line numbers are fine for display diagnostics.
            base = max(0, len(lines) - need)
            for i, raw in enumerate(lines[-need:], start=base + 1):
                try:
                    rec = self._parse_line(raw.decode("utf-8", errors="replace"), i)
                    if rec is not None:
                        self.rows.append(rec)
                except Exception:
                    continue
        except Exception:
            pass

    def _open_from_current_position(self) -> bool:
        try:
            st = os.stat(self.path)
            self.fp = open(self.path, "r", encoding="utf-8", errors="replace")
            self._inode_key = (st.st_dev, st.st_ino)
            if self.start_at_end:
                self._load_tail_rows(st)
                self.fp.seek(0, os.SEEK_END)
                self.offset = self.fp.tell()
            else:
                self.offset = 0
                self._line_no = 0
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"open failed: {type(exc).__name__}: {exc}"
            return False

    def _open_if_needed(self) -> bool:
        if self.fp is not None:
            return True
        now = time.time()
        if now - self.last_try_s < 0.5:
            return False
        self.last_try_s = now
        return self._open_from_current_position()

    @staticmethod
    def _parse_line(line: str, line_no: int) -> Optional[Dict[str, Any]]:
        raw = line.strip()
        if not raw:
            return None
        parts = [x.strip() for x in raw.split(',')]
        nums: List[Optional[float]] = []
        numeric_count = 0
        for x in parts:
            try:
                v = float(x)
                nums.append(v)
                numeric_count += 1
            except Exception:
                nums.append(None)
        if numeric_count == 0:
            return None
        wall_us = None
        vals = [v for v in nums if isinstance(v, float)]
        if vals:
            last = vals[-1]
            if last > 1e14:  # epoch microseconds, not event sensor time
                wall_us = int(last)
        rec: Dict[str, Any] = {"raw": raw, "parts": parts, "nums": nums, "line_no": int(line_no), "wall_us": wall_us}
        try:
            if len(nums) >= 6 and nums[1] is not None:
                sync_i = int(nums[1])
                if sync_i >= 0:
                    rec["sync_index"] = sync_i
                    # slice_ts_us is what the event overlay publisher stores as
                    # SharedFrame timestamp in the current pipeline.
                    if nums[5] is not None:
                        rec["slice_ts_us"] = int(nums[5])
                    if len(nums) >= 3 and nums[2] is not None:
                        rec["event_ts_us"] = int(nums[2])
        except Exception:
            pass
        return rec

    def poll(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._open_if_needed():
            return out
        try:
            st = os.stat(self.path)
            key = (st.st_dev, st.st_ino)
            if key != self._inode_key or st.st_size < self.offset:
                try:
                    self.fp.close()
                except Exception:
                    pass
                self.fp = None
                self.rows.clear()
                self._open_from_current_position()
                return out
            self.fp.seek(self.offset)
            lines = 0
            while lines < self.max_lines_per_poll:
                line = self.fp.readline()
                if not line:
                    break
                self.offset = self.fp.tell()
                self._line_no += 1
                lines += 1
                rec = self._parse_line(line, self._line_no)
                if rec is not None:
                    self.rows.append(rec)
                    out.append(rec)
            self.last_error = ""
            return out
        except Exception as exc:
            self.last_error = f"poll failed: {type(exc).__name__}: {exc}"
            try:
                if self.fp is not None:
                    self.fp.close()
            except Exception:
                pass
            self.fp = None
            return out

    def latest(self) -> Optional[Dict[str, Any]]:
        if self.rows:
            return self.rows[-1]
        return None

    def period_ms(self) -> Optional[float]:
        prev = None
        for rec in reversed(self.rows):
            # Use only sync-indexed trigger rows.  The CSV also contains the
            # opposite edge with sync_index=-1; mixing both makes the displayed
            # period look like ~30ms instead of the actual 40ms soft-trigger cadence.
            if not isinstance(rec.get("sync_index"), int):
                continue
            w = rec.get("wall_us")
            if isinstance(w, int):
                if prev is not None:
                    dt = (prev - w) / 1000.0
                    if dt >= 0:
                        return dt
                prev = w
        return None

    def infer_sync_near(self, mvs_sync: Optional[int], tolerance: int = 5) -> Optional[int]:
        if mvs_sync is None:
            return None
        rec = self.latest()
        if not rec:
            return None
        if isinstance(rec.get("sync_index"), int):
            d = int(rec["sync_index"]) - int(mvs_sync)
            if abs(d) <= int(tolerance):
                return int(rec["sync_index"])
        best = None
        best_abs = None
        for v in rec.get("nums") or []:
            if not isinstance(v, float):
                continue
            iv = int(v)
            if abs(float(iv) - float(v)) > 1e-6:
                continue
            d = abs(iv - int(mvs_sync))
            if d <= int(tolerance) and (best_abs is None or d < best_abs):
                best = iv
                best_abs = d
        return best

    def event_ts_for_sync(self, sync_index: int) -> Optional[int]:
        try:
            target = int(sync_index)
        except Exception:
            return None
        for rec in reversed(self.rows):
            if rec.get("sync_index") == target:
                v = rec.get("slice_ts_us", rec.get("event_ts_us"))
                try:
                    return int(v)
                except Exception:
                    return None
        return None

    def infer_sync_for_event_ts(self, event_ts_us: int, max_diff_us: int = 120_000) -> Tuple[Optional[int], Optional[float]]:
        try:
            ts = int(event_ts_us)
        except Exception:
            return None, None
        best_sync = None
        best_abs = None
        for rec in reversed(self.rows):
            s = rec.get("sync_index")
            if not isinstance(s, int):
                continue
            cand = rec.get("slice_ts_us", rec.get("event_ts_us"))
            if cand is None:
                continue
            try:
                d = abs(int(cand) - ts)
            except Exception:
                continue
            if best_abs is None or d < best_abs:
                best_abs = d
                best_sync = int(s)
        if best_abs is None or best_abs > int(max_diff_us):
            return None, (float(best_abs) / 1000.0 if best_abs is not None else None)
        return best_sync, float(best_abs) / 1000.0

    def close(self) -> None:
        try:
            if self.fp is not None:
                self.fp.close()
        except Exception:
            pass
        self.fp = None

def _latest_human_record(records: Deque[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    return records[-1]


def _record_sync_index(rec: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(rec, dict):
        return None
    for key in ("sync_index_hint", "program_sync_index", "mvs_sync_index", "sync_index", "frame_id_local", "mvs_frame_num"):
        try:
            v = rec.get(key)
            if v is not None:
                return int(v)
        except Exception:
            pass
    meta = rec.get("mvs_meta") if isinstance(rec.get("mvs_meta"), dict) else None
    if meta:
        for key in ("program_sync_index", "mvs_sync_index", "sync_index"):
            try:
                v = meta.get(key)
                if v is not None:
                    return int(v)
            except Exception:
                pass
    return None


def _person_center_from_dict(p: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    box = p.get("bbox_xyxy") or p.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + max(1.0, float(box[2]))
                y2 = y1 + max(1.0, float(box[3]))
            return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        except Exception:
            return None
    return None


def _shift_xy_list(obj: Any, dx: float, dy: float) -> Any:
    if isinstance(obj, list):
        if len(obj) >= 2 and all(isinstance(obj[i], (int, float)) for i in (0, 1)):
            out = list(obj)
            out[0] = float(out[0]) + dx
            out[1] = float(out[1]) + dy
            return out
        return [_shift_xy_list(v, dx, dy) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_shift_xy_list(list(obj), dx, dy))
    return obj


def _shift_person_for_display(person: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    q = copy.deepcopy(person)
    for key in ("bbox_xyxy", "bbox"):
        box = q.get(key)
        if isinstance(box, list) and len(box) >= 4:
            try:
                q[key] = [float(box[0]) + dx, float(box[1]) + dy, float(box[2]) + dx, float(box[3]) + dy]
            except Exception:
                pass
    for key in ("foot_pixel", "mask_polygons", "mask_polygon", "polygons", "segments", "segment", "contours"):
        if key in q:
            q[key] = _shift_xy_list(q[key], dx, dy)
    q["display_shift_xy"] = [round(float(dx), 2), round(float(dy), 2)]
    return q


@dataclass
class CameraState:
    cam_id: str
    mvs_name: str
    event_name: str
    overlay_cfg: Dict[str, Any]
    mvs_reader: LazySharedReader
    event_reader: LazySharedReader
    mvs_rgb: Optional[np.ndarray] = None
    event_rgb: Optional[np.ndarray] = None
    mvs_frame_id: int = -1
    event_frame_id: int = -1
    mvs_ts_us: int = 0
    event_ts_us: int = 0
    mvs_shape_hw: Tuple[int, int] = (0, 0)
    event_shape_hw: Tuple[int, int] = (0, 0)
    mvs_tex: int = 0
    event_tex: int = 0
    mvs_tex_shape: Tuple[int, int] = (0, 0)
    event_tex_shape: Tuple[int, int] = (0, 0)
    last_mvs_wall: float = 0.0
    last_event_wall: float = 0.0
    status: str = "waiting"
    meta_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    trigger_path: Optional[str] = None
    trigger_tailer: Optional[TriggerCsvTailer] = None
    trigger_record: Optional[Dict[str, Any]] = None
    trigger_period_ms: Optional[float] = None
    human_path: Optional[str] = None
    human_tailer: Optional[JsonlTailer] = None
    human_record: Optional[Dict[str, Any]] = None
    human_latest_record: Optional[Dict[str, Any]] = None
    human_record_wall: float = 0.0
    hit_path: Optional[str] = None
    hit_tailer: Optional[JsonlTailer] = None
    recent_hits: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=32))
    # Display-only frame history.  This is used by the OpenGL preview to
    # render a slightly delayed view so latest MVS frames do not visually
    # outrun YOLO human masks / event overlays.  It does not feed
    # hit_judge, event processing, or any synchronization file.
    mvs_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))
    event_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=512))
    mvs_latest_frame_id: int = -1
    event_latest_frame_id: int = -1
    event_display_sync: Optional[int] = None
    event_match_diff_ms: float = -1.0
    mvs_uploaded_tex_frame_id: int = -1
    event_uploaded_tex_frame_id: int = -1
    display_delay_age_ms: float = 0.0
    display_delay_active: bool = False
    mvs_meta_status: str = ""


# ----------------------------- OpenGL widget -------------------------------

VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

BG_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""

EVENT_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_event_tex;
uniform mat3 u_mvs_to_event;
uniform vec2 u_mvs_size;
uniform vec2 u_event_size;
uniform float u_black_threshold;
uniform float u_opacity;
uniform float u_intensity_gamma;
uniform int u_colorize;
uniform vec3 u_color;
void main() {
    // v_uv is OpenGL bottom-left coordinates. Convert to image top-left MVS pixel.
    float mvs_x = v_uv.x * u_mvs_size.x;
    float mvs_y = (1.0 - v_uv.y) * u_mvs_size.y;
    vec3 ep = u_mvs_to_event * vec3(mvs_x, mvs_y, 1.0);
    if (abs(ep.z) < 1e-6) discard;
    vec2 e = ep.xy / ep.z;
    if (e.x < 0.0 || e.y < 0.0 || e.x >= u_event_size.x || e.y >= u_event_size.y) discard;

    // Convert top-left event pixel to OpenGL UV.
    vec2 euv = vec2(e.x / u_event_size.x, 1.0 - e.y / u_event_size.y);
    vec3 rgb = texture(u_event_tex, euv).rgb;
    float gray = max(max(rgb.r, rgb.g), rgb.b);
    float norm = clamp((gray - u_black_threshold) / max(1e-6, 1.0 - u_black_threshold), 0.0, 1.0);
    // Low-intrusion live overlay: alpha follows event intensity, so sparse
    // bullet/event pixels remain visible while the MVS image is not tinted.
    float alpha = pow(norm, max(0.1, u_intensity_gamma)) * u_opacity;
    if (alpha <= 0.001) discard;
    vec3 out_rgb = rgb;
    if (u_colorize != 0) {
        out_rgb = u_color * max(0.35, gray);
    }
    fragColor = vec4(out_rgb, alpha);
}
"""


class FusionGLWidget(QOpenGLWidget):
    def __init__(self, cameras: List[CameraState], preview_fps: float = 120.0, event_stale_ms: float = 200.0, record_cfg: Optional[Dict[str, Any]] = None, sync_cfg: Optional[Dict[str, Any]] = None, show_human_overlay: bool = True, show_hit_overlay: bool = True, human_overlay_cfg: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.cameras = cameras
        self.record_cfg = record_cfg if isinstance(record_cfg, dict) else {}
        self.sync_cfg = sync_cfg if isinstance(sync_cfg, dict) else {}
        self.show_human_overlay = bool(show_human_overlay)
        self.show_hit_overlay = bool(show_hit_overlay)
        self.human_overlay_cfg = human_overlay_cfg if isinstance(human_overlay_cfg, dict) else {}
        self.human_draw_mask = _as_bool(self.human_overlay_cfg.get("draw_mask", True), True)
        self.human_draw_contour = _as_bool(self.human_overlay_cfg.get("draw_contour", True), True)
        self.human_draw_box = _as_bool(self.human_overlay_cfg.get("draw_box", False), False)
        self.human_draw_label = _as_bool(self.human_overlay_cfg.get("draw_label", True), True)
        try:
            self.human_mask_alpha = float(self.human_overlay_cfg.get("mask_alpha", 0.35) or 0.35)
        except Exception:
            self.human_mask_alpha = 0.35
        try:
            self.human_contour_thickness = max(1, int(self.human_overlay_cfg.get("contour_thickness", 2) or 2))
        except Exception:
            self.human_contour_thickness = 2
        self.human_display_mode = str(self.human_overlay_cfg.get("display_mode", "predict_to_current") or "predict_to_current")
        self.human_prediction_enable = _as_bool(self.human_overlay_cfg.get("prediction_enable", True), True)
        self.human_prediction_max_frames = int(self.human_overlay_cfg.get("prediction_max_frames", self.human_overlay_cfg.get("history_frames", 12)) or 12)
        self.human_prediction_max_frames = max(0, min(60, self.human_prediction_max_frames))
        self.human_prediction_max_shift_px = float(self.human_overlay_cfg.get("prediction_max_shift_px", 220.0) or 220.0)
        self.human_prediction_max_record_age_ms = float(self.human_overlay_cfg.get("max_lock_delay_ms", 500.0) or 500.0)
        self.human_draw_sync_debug = _as_bool(self.human_overlay_cfg.get("draw_sync_debug", False), False)
        self.display_sync_mode = str(self.human_overlay_cfg.get("display_sync_mode", "live_low_latency") or "live_low_latency")
        try:
            self.display_delay_ms = float(self.human_overlay_cfg.get("display_delay_ms", self.human_overlay_cfg.get("mvs_display_delay_ms", 0.0)) or 0.0)
        except Exception:
            self.display_delay_ms = 0.0
        try:
            self.display_buffer_ms = float(self.human_overlay_cfg.get("display_buffer_ms", max(600.0, self.display_delay_ms + 300.0)) or max(600.0, self.display_delay_ms + 300.0))
        except Exception:
            self.display_buffer_ms = max(600.0, self.display_delay_ms + 300.0)
        self.display_delay_enabled = (
            self.display_delay_ms > 1.0
            and self.display_sync_mode.lower() in {"debug_aligned", "delayed_preview", "delay_mvs", "delayed", "mvs_delay"}
        )
        if self.display_delay_enabled:
            print(f"[fusion_gl] display delay enabled: mode={self.display_sync_mode} delay={self.display_delay_ms:.1f}ms buffer={self.display_buffer_ms:.1f}ms (preview only; hit_judge unchanged)", flush=True)
        self.record_enabled = _record_cfg_enabled(self.record_cfg)
        self.record_command_file = _record_command_file(self.record_cfg, self.sync_cfg)
        self.record_status_file = _record_status_file(self.record_cfg, self.sync_cfg)
        self._recording_hint: Optional[bool] = None
        self._record_session_hint: str = ""
        self._record_mode_hints: Dict[str, bool] = {"all": False, "event_raw": False, "mvs_raw": False}
        self._record_status_text = "REC disabled" if not self.record_enabled else "REC ready: S=ALL O=EVENT P=MVS"
        self.preview_fps = max(1.0, float(preview_fps or 120.0))
        self.event_stale_ms = float(event_stale_ms or 200.0)
        self.display_audit_enable = _as_bool(self.human_overlay_cfg.get("display_audit_enable", True), True)
        try:
            self.display_audit_interval_ms = max(20.0, float(self.human_overlay_cfg.get("display_audit_interval_ms", 100.0) or 100.0))
        except Exception:
            self.display_audit_interval_ms = 100.0
        audit_dir_raw = str(self.human_overlay_cfg.get("display_audit_dir", "") or "")
        if audit_dir_raw:
            self.display_audit_dir = Path(audit_dir_raw).expanduser()
        else:
            self.display_audit_dir = Path(str(self.sync_cfg.get("root", "./sync_ipc") or "./sync_ipc"))
        if not self.display_audit_dir.is_absolute():
            self.display_audit_dir = Path.cwd() / self.display_audit_dir
        self._audit_last_wall: Dict[str, float] = {}
        self._audit_header_written: set = set()
        self.human_sync_match_enable = _as_bool(self.human_overlay_cfg.get("human_sync_match_enable", self.human_overlay_cfg.get("sync_match_enable", True)), True)
        try:
            self.human_sync_match_max_diff_frames = max(0, int(self.human_overlay_cfg.get("human_sync_match_max_diff_frames", 4) or 4))
        except Exception:
            self.human_sync_match_max_diff_frames = 4
        self.event_sync_match_enable = _as_bool(self.human_overlay_cfg.get("event_sync_match_enable", True), True)
        try:
            self.event_sync_match_max_diff_ms = max(1.0, float(self.human_overlay_cfg.get("event_sync_match_max_diff_ms", self.human_overlay_cfg.get("event_overlay_max_match_diff_ms", 80.0)) or 80.0))
        except Exception:
            self.event_sync_match_max_diff_ms = 80.0
        self.bg_prog = None
        self.event_prog = None
        self.vbo = 0
        self.vao = 0
        self._last_poll_s = 0.0
        self._paint_count = 0
        self._paint_t0 = time.time()
        self._fps_value = 0.0
        self.setWindowTitle("MVS + Event OpenGL Fusion Preview")
        self.resize(1600, 1000)
        self.setFocusPolicy(Qt.StrongFocus)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(max(1, int(round(1000.0 / self.preview_fps))))

    def _on_timer(self) -> None:
        self.poll_frames()
        self._refresh_record_status_text()
        self.update()

    def initializeGL(self) -> None:
        GL.glClearColor(0.02, 0.02, 0.025, 1.0)
        self.bg_prog = shaders.compileProgram(shaders.compileShader(VERT_SRC, GL.GL_VERTEX_SHADER), shaders.compileShader(BG_FRAG_SRC, GL.GL_FRAGMENT_SHADER))
        self.event_prog = shaders.compileProgram(shaders.compileShader(VERT_SRC, GL.GL_VERTEX_SHADER), shaders.compileShader(EVENT_FRAG_SRC, GL.GL_FRAGMENT_SHADER))
        verts = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, int(verts.nbytes), verts, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        for cam in self.cameras:
            cam.mvs_tex = GL.glGenTextures(1)
            cam.event_tex = GL.glGenTextures(1)
            for tex in (cam.mvs_tex, cam.event_tex):
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        for cam in self.cameras:
            cam.mvs_reader.close()
            cam.event_reader.close()
            if cam.human_tailer is not None:
                cam.human_tailer.close()
            if cam.hit_tailer is not None:
                cam.hit_tailer.close()
        super().closeEvent(event)

    def _read_record_status(self) -> Tuple[Optional[bool], str]:
        """Read MVS recorder status written by the capture process if available."""
        path = Path(self.record_status_file)
        if not path.exists():
            return self._recording_hint, self._record_session_hint
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rec = bool(data.get("recording", False))
            sid = str(data.get("session_id", "") or "")
            self._recording_hint = rec
            self._record_session_hint = sid
            return rec, sid
        except Exception:
            return self._recording_hint, self._record_session_hint

    def _refresh_record_status_text(self) -> None:
        if not self.record_enabled:
            self._record_status_text = "REC disabled"
            return
        rec, sid = self._read_record_status()
        if rec is True:
            self._record_status_text = f"REC ON {sid}" if sid else "REC ON"
        elif rec is False:
            self._record_status_text = "REC OFF"
        else:
            self._record_status_text = "REC ready: S=ALL O=EVENT P=MVS"

    def _toggle_record_targets(self, mode_key: str, targets: List[str], label: str) -> None:
        if not self.record_enabled:
            self._record_status_text = "REC disabled in one_key_record.enable=false"
            print(f"[fusion_gl][onekey_record] disabled; command_file would be {self.record_command_file}", flush=True)
            return

        rec: Optional[bool] = None
        if mode_key in {"all", "mvs_raw"}:
            rec, _sid = self._read_record_status()

        if rec is None:
            rec = bool(self._record_mode_hints.get(mode_key, False))

        if rec is True:
            ok = _write_one_key_record_command(
                self.record_command_file,
                "stop",
                source=f"fusion_gl_renderer_key_{mode_key}",
                targets=targets,
                mode=mode_key,
            )
            if ok:
                self._record_mode_hints[mode_key] = False
                if mode_key == "all":
                    self._record_mode_hints["event_raw"] = False
                    self._record_mode_hints["mvs_raw"] = False
                    self._recording_hint = False
                elif mode_key == "mvs_raw":
                    self._recording_hint = False
                self._record_status_text = f"REC {label} stopping..."
        else:
            sid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ok = _write_one_key_record_command(
                self.record_command_file,
                "start",
                session_id=sid,
                source=f"fusion_gl_renderer_key_{mode_key}",
                targets=targets,
                mode=mode_key,
            )
            if ok:
                self._record_mode_hints[mode_key] = True
                if mode_key == "all":
                    self._record_mode_hints["event_raw"] = True
                    self._record_mode_hints["mvs_raw"] = True
                    self._recording_hint = True
                    self._record_session_hint = sid
                elif mode_key == "mvs_raw":
                    self._recording_hint = True
                    self._record_session_hint = sid
                self._record_status_text = f"REC {label} starting {sid}..."

    def _toggle_one_key_recording(self) -> None:
        self._toggle_record_targets("all", ["event_raw", "mvs_raw"], "ALL")

    def _qt_key_value(self, name: str):
        value = getattr(Qt, name, None)
        if value is None and hasattr(Qt, "Key"):
            value = getattr(Qt.Key, name, None)
        return value

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        text = str(event.text() or "").lower()
        esc_key = self._qt_key_value("Key_Escape")
        key_s = self._qt_key_value("Key_S")
        key_o = self._qt_key_value("Key_O")
        key_p = self._qt_key_value("Key_P")
        if text == "s" or (key_s is not None and key == key_s):
            self._toggle_record_targets("all", ["event_raw", "mvs_raw"], "ALL")
            self.update()
            event.accept()
            return
        if text == "o" or (key_o is not None and key == key_o):
            self._toggle_record_targets("event_raw", ["event_raw"], "EVENT RAW")
            self.update()
            event.accept()
            return
        if text == "p" or (key_p is not None and key == key_p):
            self._toggle_record_targets("mvs_raw", ["mvs_raw"], "MVS RAW")
            self.update()
            event.accept()
            return
        if text == "h":
            self.show_human_overlay = not self.show_human_overlay
            print(f"[fusion_gl] human overlay: {self.show_human_overlay}", flush=True)
            self.update()
            event.accept()
            return
        if text == "j":
            self.show_hit_overlay = not self.show_hit_overlay
            print(f"[fusion_gl] hit overlay: {self.show_hit_overlay}", flush=True)
            self.update()
            event.accept()
            return
        if text == "q" or (esc_key is not None and key == esc_key):
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def _append_display_history(self, hist: Deque[Dict[str, Any]], rec: Dict[str, Any], now: float) -> None:
        """Append a display-frame record and drop very old frames.

        This history is for preview alignment only.  The acquisition, JSONL
        output, event-camera processing, and hit_judge continue to use their
        original real-time paths.  If the GL timer sees the same shared-memory
        frame again, update its metadata snapshot instead of appending a duplicate;
        this handles the small race where MVS shm is visible before its JSON meta
        sidecar has been atomically replaced.
        """
        if not isinstance(rec, dict):
            return
        frame_id = rec.get("frame_id")
        ts_us = rec.get("timestamp_us")
        if hist:
            last = hist[-1]
            if frame_id is not None and last.get("frame_id") == frame_id and last.get("timestamp_us") == ts_us:
                if "meta" in rec:
                    new_meta = rec.get("meta")
                    old_meta = last.get("meta")
                    if isinstance(new_meta, dict) and new_meta:
                        last["meta"] = new_meta
                        last["meta_status"] = rec.get("meta_status", last.get("meta_status", ""))
                    elif not isinstance(old_meta, dict) or not old_meta:
                        last["meta"] = new_meta
                        last["meta_status"] = rec.get("meta_status", last.get("meta_status", ""))
                return
        hist.append(rec)
        cutoff = float(now) - max(0.2, float(self.display_buffer_ms) / 1000.0)
        while len(hist) > 1 and float(hist[0].get("wall", 0.0) or 0.0) < cutoff:
            hist.popleft()

    def _read_mvs_meta_snapshot(self, cam: CameraState, frame_id: int, timestamp_us: int) -> Tuple[Dict[str, Any], str]:
        """Read the MVS sidecar meta only if it matches the shm frame being queued."""
        if not cam.meta_path:
            return {}, "no_meta_path"
        try:
            p = Path(cam.meta_path)
            if not p.exists():
                return {}, "meta_missing"
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return {}, f"meta_read_error:{type(exc).__name__}"
        if not isinstance(meta, dict):
            return {}, "meta_not_dict"
        meta["_display_meta_read_wall_us"] = _now_us()
        meta_fid = None
        for key in ("shared_frame_id", "frame_id_local", "frame_id"):
            try:
                if meta.get(key) is not None:
                    meta_fid = int(meta.get(key))
                    break
            except Exception:
                pass
        if meta_fid is not None and int(frame_id) >= 0 and meta_fid != int(frame_id):
            return {}, f"meta_mismatch:meta_fid={meta_fid}:shm_fid={int(frame_id)}"
        return meta, "ok"


    def _display_mvs_sync(self, cam: CameraState) -> Optional[int]:
        meta = cam.meta if isinstance(cam.meta, dict) else {}
        for key in ("program_sync_index", "mvs_sync_index", "sync_index"):
            try:
                if meta.get(key) is not None:
                    return int(meta.get(key))
            except Exception:
                pass
        # In this pipeline frame_id_local is usually sync_index + 1 because
        # sync.sync_index_offset=-1.  Use this only as a last-resort display hint.
        try:
            if int(cam.mvs_frame_id) >= 1:
                return int(cam.mvs_frame_id) - 1
        except Exception:
            pass
        return None

    def _select_human_record_for_mvs(self, cam: CameraState) -> Optional[Dict[str, Any]]:
        if cam.human_tailer is None:
            return cam.human_latest_record or cam.human_record
        records = list(cam.human_tailer.records)
        if not records:
            return cam.human_latest_record or cam.human_record
        if not self.human_sync_match_enable:
            return records[-1]
        target_sync = self._display_mvs_sync(cam)
        if target_sync is None:
            return records[-1]
        best = None
        best_abs = None
        best_future = None
        best_future_abs = None
        for rec in records:
            s = _record_sync_index(rec)
            if s is None:
                continue
            d = int(s) - int(target_sync)
            ad = abs(d)
            if d <= 0:
                if best_abs is None or ad < best_abs:
                    best = rec
                    best_abs = ad
            elif best_future_abs is None or ad < best_future_abs:
                best_future = rec
                best_future_abs = ad
        max_diff = int(self.human_sync_match_max_diff_frames)
        if best is not None and (best_abs is None or best_abs <= max_diff):
            return best
        # If the exact/older result has not arrived yet, allow a very small future
        # fallback rather than drawing a much older frozen mask.
        if best_future is not None and best_future_abs is not None and best_future_abs <= max(1, max_diff):
            return best_future
        return records[-1]

    def _select_event_history_for_mvs(self, cam: CameraState, now: float) -> Optional[Dict[str, Any]]:
        fallback = self._select_display_history(cam.event_history, now)
        cam.event_display_sync = None
        cam.event_match_diff_ms = -1.0
        if not self.event_sync_match_enable or cam.trigger_tailer is None:
            return fallback
        target_sync = self._display_mvs_sync(cam)
        if target_sync is None:
            return fallback
        target_event_ts = cam.trigger_tailer.event_ts_for_sync(int(target_sync))
        if target_event_ts is None:
            return fallback
        max_diff_us = int(float(self.event_sync_match_max_diff_ms) * 1000.0)
        best = None
        best_abs = None
        for rec in reversed(cam.event_history):
            try:
                ts = int(rec.get("timestamp_us", 0) or 0)
            except Exception:
                continue
            if ts <= 0:
                continue
            d = abs(ts - int(target_event_ts))
            if best_abs is None or d < best_abs:
                best = rec
                best_abs = d
                if d == 0:
                    break
        if best is not None and best_abs is not None and best_abs <= max_diff_us:
            try:
                best["display_age_ms"] = max(0.0, (float(now) - float(best.get("wall", now))) * 1000.0)
            except Exception:
                pass
            best["event_display_sync"] = int(target_sync)
            best["event_match_diff_ms"] = float(best_abs) / 1000.0
            cam.event_display_sync = int(target_sync)
            cam.event_match_diff_ms = float(best_abs) / 1000.0
            return best
        return fallback

    def _append_display_audit(self, cam: CameraState, now: float) -> None:
        if not self.display_audit_enable:
            return
        last = float(self._audit_last_wall.get(cam.cam_id, 0.0) or 0.0)
        if (now - last) * 1000.0 < self.display_audit_interval_ms:
            return
        self._audit_last_wall[cam.cam_id] = now
        try:
            self.display_audit_dir.mkdir(parents=True, exist_ok=True)
            path = self.display_audit_dir / f"display_audit_{cam.cam_id}.csv"
            meta = cam.meta if isinstance(cam.meta, dict) else {}
            human = cam.human_record if isinstance(cam.human_record, dict) else {}
            try:
                mvs_sync = int(meta.get("program_sync_index")) if meta.get("program_sync_index") is not None else ""
            except Exception:
                mvs_sync = ""
            human_sync = _record_sync_index(human) if human else None
            try:
                event_age_ms = (now - float(cam.last_event_wall)) * 1000.0 if cam.last_event_wall else ""
            except Exception:
                event_age_ms = ""
            recent_hit_count = 0
            for h in list(cam.recent_hits):
                try:
                    if now - float(h.get("judge_wall_time", now) or now) <= 3.0:
                        recent_hit_count += 1
                except Exception:
                    pass
            row = {
                "wall_us": _now_us(),
                "cam_id": cam.cam_id,
                "display_mvs_frame_id": cam.mvs_frame_id,
                "display_mvs_sync": mvs_sync,
                "display_mvs_ts_us": cam.mvs_ts_us,
                "display_event_frame_id": cam.event_frame_id,
                "display_event_ts_us": cam.event_ts_us,
                "display_event_sync": cam.event_display_sync if cam.event_display_sync is not None else "",
                "event_sync_delta": (int(cam.event_display_sync) - int(mvs_sync)) if (cam.event_display_sync is not None and mvs_sync != "") else "",
                "event_match_diff_ms": f"{cam.event_match_diff_ms:.1f}" if cam.event_match_diff_ms >= 0 else "",
                "event_age_ms": f"{event_age_ms:.1f}" if isinstance(event_age_ms, float) else event_age_ms,
                "human_sync": human_sync if human_sync is not None else "",
                "human_sync_delta": (int(human_sync) - int(mvs_sync)) if (human_sync is not None and mvs_sync != "") else "",
                "human_frame_id": human.get("frame_id_local", human.get("frame_id", "")) if human else "",
                "hit_recent_count": recent_hit_count,
                "mvs_meta_status": cam.mvs_meta_status,
                "display_delay_active": int(bool(cam.display_delay_active)),
                "display_delay_age_ms": f"{cam.display_delay_age_ms:.1f}",
            }
            fields = list(row.keys())
            need_header = path not in self._audit_header_written and (not path.exists() or path.stat().st_size <= 0)
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if need_header:
                    w.writeheader()
                w.writerow(row)
            self._audit_header_written.add(path)
        except Exception:
            # Audit must never affect preview or hit judgment.
            pass

    def _select_display_history(self, hist: Deque[Dict[str, Any]], now: float) -> Optional[Dict[str, Any]]:
        """Return the frame to draw for the current preview tick.

        When display_delay_enabled=false, this is simply the latest frame.
        When enabled, choose the newest frame whose read wall-time is <=
        now-display_delay_ms.  If the buffer is not warm yet, fall back to the
        oldest available frame instead of showing nothing.
        """
        if not hist:
            return None
        if not self.display_delay_enabled:
            return hist[-1]
        target = float(now) - float(self.display_delay_ms) / 1000.0
        best = None
        for rec in reversed(hist):
            try:
                if float(rec.get("wall", 0.0) or 0.0) <= target:
                    best = rec
                    break
            except Exception:
                continue
        if best is None:
            best = hist[0]
        try:
            best["display_age_ms"] = max(0.0, (float(now) - float(best.get("wall", now))) * 1000.0)
        except Exception:
            pass
        return best

    def poll_frames(self) -> None:
        for cam in self.cameras:
            now_poll = time.time()
            m = cam.mvs_reader.read_latest()
            if m is not None and m.get("frame") is not None:
                frame = np.asarray(m["frame"])
                rgb = None
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    rgb = frame[:, :, :3][:, :, ::-1].copy()
                elif frame.ndim == 2:
                    rgb = np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8)
                if rgb is not None:
                    latest_frame_id = int(m.get("frame_id", cam.mvs_latest_frame_id) or 0)
                    latest_ts_us = int(m.get("timestamp_us", 0) or 0)
                    cam.mvs_latest_frame_id = latest_frame_id
                    cam.last_mvs_wall = now_poll
                    meta_snapshot, meta_status = self._read_mvs_meta_snapshot(cam, latest_frame_id, latest_ts_us)
                    self._append_display_history(
                        cam.mvs_history,
                        {
                            "wall": now_poll,
                            "rgb": rgb,
                            "frame_id": latest_frame_id,
                            "timestamp_us": latest_ts_us,
                            "meta": meta_snapshot,
                            "meta_status": meta_status,
                        },
                        now_poll,
                    )
            m_disp = self._select_display_history(cam.mvs_history, now_poll)
            if isinstance(m_disp, dict) and m_disp.get("rgb") is not None:
                cam.mvs_rgb = m_disp.get("rgb")
                cam.mvs_frame_id = int(m_disp.get("frame_id", cam.mvs_frame_id) or 0)
                cam.mvs_ts_us = int(m_disp.get("timestamp_us", 0) or 0)
                cam.mvs_shape_hw = tuple(cam.mvs_rgb.shape[:2]) if cam.mvs_rgb is not None else (0, 0)
                cam.display_delay_age_ms = float(m_disp.get("display_age_ms", 0.0) or 0.0)
                cam.display_delay_active = bool(self.display_delay_enabled)
                cam.meta = dict(m_disp.get("meta") or {})
                cam.mvs_meta_status = str(m_disp.get("meta_status", "") or "")

            # Keep sync maps fresh before selecting event/human display records.
            if cam.trigger_tailer is not None:
                new_trig = cam.trigger_tailer.poll()
                if new_trig:
                    cam.trigger_record = new_trig[-1]
                    cam.trigger_period_ms = cam.trigger_tailer.period_ms()
                elif cam.trigger_record is None:
                    cam.trigger_record = cam.trigger_tailer.latest()
                    cam.trigger_period_ms = cam.trigger_tailer.period_ms()

            if cam.human_tailer is not None:
                new_h = cam.human_tailer.poll()
                if new_h:
                    cam.human_latest_record = new_h[-1]
                    cam.human_record_wall = time.time()
                elif cam.human_latest_record is None:
                    cam.human_latest_record = _latest_human_record(cam.human_tailer.records)
                # Display mask is now chosen for the displayed MVS sync, not the
                # latest detector output.  This is preview-only and does not
                # change human_result_*.jsonl or hit_judge behavior.
                cam.human_record = self._select_human_record_for_mvs(cam)

            e = cam.event_reader.read_latest()
            if e is not None and e.get("frame") is not None:
                frame = np.asarray(e["frame"])
                rgb = None
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    rgb = frame[:, :, :3][:, :, ::-1].copy()
                elif frame.ndim == 2:
                    rgb = np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8)
                if rgb is not None:
                    latest_frame_id = int(e.get("frame_id", cam.event_latest_frame_id) or 0)
                    latest_ts_us = int(e.get("timestamp_us", 0) or 0)
                    cam.event_latest_frame_id = latest_frame_id
                    cam.last_event_wall = now_poll
                    self._append_display_history(
                        cam.event_history,
                        {
                            "wall": now_poll,
                            "rgb": rgb,
                            "frame_id": latest_frame_id,
                            "timestamp_us": latest_ts_us,
                        },
                        now_poll,
                    )
            e_disp = self._select_event_history_for_mvs(cam, now_poll)
            if isinstance(e_disp, dict) and e_disp.get("rgb") is not None:
                cam.event_rgb = e_disp.get("rgb")
                cam.event_frame_id = int(e_disp.get("frame_id", cam.event_frame_id) or 0)
                cam.event_ts_us = int(e_disp.get("timestamp_us", 0) or 0)
                cam.event_shape_hw = tuple(cam.event_rgb.shape[:2]) if cam.event_rgb is not None else (0, 0)
                if "event_display_sync" in e_disp:
                    try:
                        cam.event_display_sync = int(e_disp.get("event_display_sync"))
                    except Exception:
                        pass
                if "event_match_diff_ms" in e_disp:
                    try:
                        cam.event_match_diff_ms = float(e_disp.get("event_match_diff_ms"))
                    except Exception:
                        pass
            if not cam.mvs_history and cam.meta_path:
                try:
                    p = Path(cam.meta_path)
                    if p.exists():
                        cam.meta = json.loads(p.read_text(encoding="utf-8"))
                        cam.mvs_meta_status = "fallback_latest_meta"
                except Exception:
                    pass
            if cam.hit_tailer is not None:
                for obj in cam.hit_tailer.poll():
                    cam.recent_hits.append(obj)
            mvs_age = (time.time() - cam.last_mvs_wall) * 1000.0 if cam.last_mvs_wall else 1e9
            ev_age = (time.time() - cam.last_event_wall) * 1000.0 if cam.last_event_wall else 1e9
            if cam.mvs_rgb is None:
                cam.status = f"NO MVS shm={cam.mvs_name} err={cam.mvs_reader.last_error}"
            elif cam.event_rgb is None:
                cam.status = f"MVS only; NO EVENT shm={cam.event_name} err={cam.event_reader.last_error}"
            elif ev_age > self.event_stale_ms:
                cam.status = f"EVENT STALE {ev_age:.0f}ms"
            else:
                meta = cam.meta if isinstance(cam.meta, dict) else {}
                mvs_frame_num = meta.get("mvs_frame_num", meta.get("nFrameNum", cam.mvs_frame_id))
                mvs_sync = self._display_mvs_sync(cam)
                try:
                    mvs_frame_i = int(mvs_frame_num) if mvs_frame_num is not None else None
                except Exception:
                    mvs_frame_i = None
                fs_delta = ""
                if mvs_frame_i is not None and mvs_sync is not None:
                    fs_delta = f" f-s={mvs_frame_i - mvs_sync:+d}"
                event_age = f" ev_age={ev_age:.0f}ms" if ev_age < 999999 else ""
                ev_sync_txt = ""
                if cam.event_display_sync is not None:
                    ev_sync_txt = f" ev_sync={cam.event_display_sync}"
                    if cam.event_match_diff_ms >= 0:
                        ev_sync_txt += f" ediff={cam.event_match_diff_ms:.1f}ms"
                trig_txt = " trig=NA"
                if cam.trigger_tailer is not None:
                    rec = cam.trigger_record
                    if rec is None:
                        err = cam.trigger_tailer.last_error
                        trig_txt = " trig=none" if not err else f" trig_err={err[:24]}"
                    else:
                        wall_us = rec.get("wall_us")
                        age_txt = ""
                        if isinstance(wall_us, int) and wall_us > 0:
                            age_txt = f" age={max(0.0, (_now_us() - wall_us) / 1000.0):.0f}ms"
                        per_txt = ""
                        if cam.trigger_period_ms is not None:
                            per_txt = f" per={cam.trigger_period_ms:.1f}"
                        near_txt = ""
                        near = cam.trigger_tailer.infer_sync_near(mvs_sync) if mvs_sync is not None else None
                        if near is not None:
                            near_txt = f" trig_sync={near} diff={near - int(mvs_sync):+d}"
                        trig_txt = f" trig#{rec.get('line_no','-')}{age_txt}{per_txt}{near_txt}"
                disp_txt = ""
                if cam.display_delay_active:
                    disp_txt = f" disp_delay={cam.display_delay_age_ms:.0f}ms"
                meta_warn = "" if cam.mvs_meta_status in {"", "ok"} else f" meta={cam.mvs_meta_status}"
                cam.status = f"MVS#{mvs_frame_num} mvs_sync={mvs_sync if mvs_sync is not None else '-'}{fs_delta}{disp_txt}{meta_warn} | EVT#{cam.event_frame_id}{event_age}{ev_sync_txt} |{trig_txt}"
            self._append_display_audit(cam, now_poll)

    def _update_texture(self, tex: int, rgb: np.ndarray, shape_attr: str, cam: CameraState) -> None:
        if rgb is None:
            return
        rgb = np.ascontiguousarray(np.flipud(rgb.astype(np.uint8)))
        h, w = rgb.shape[:2]
        old_shape = getattr(cam, shape_attr)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if old_shape != (h, w):
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
            setattr(cam, shape_attr, (h, w))
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _bind_quad(self, prog: int) -> None:
        GL.glUseProgram(prog)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        stride = 4 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(8))

    def _draw_background(self, cam: CameraState) -> None:
        if cam.mvs_rgb is None:
            return
        if int(cam.mvs_uploaded_tex_frame_id) != int(cam.mvs_frame_id):
            self._update_texture(cam.mvs_tex, cam.mvs_rgb, "mvs_tex_shape", cam)
            cam.mvs_uploaded_tex_frame_id = int(cam.mvs_frame_id)
        GL.glDisable(GL.GL_BLEND)
        self._bind_quad(self.bg_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, cam.mvs_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.bg_prog, "u_tex"), 0)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _draw_event_overlay(self, cam: CameraState) -> None:
        if cam.mvs_rgb is None or cam.event_rgb is None:
            return
        mv_h, mv_w = cam.mvs_rgb.shape[:2]
        ev_h, ev_w = cam.event_rgb.shape[:2]
        if mv_h <= 0 or mv_w <= 0 or ev_h <= 0 or ev_w <= 0:
            return
        try:
            H_ev_to_mvs = _homography_from_overlay_cfg(cam.overlay_cfg, (ev_h, ev_w), (mv_h, mv_w))
            H_mvs_to_ev = np.linalg.inv(H_ev_to_mvs).astype(np.float32)
        except Exception:
            return
        if int(cam.event_uploaded_tex_frame_id) != int(cam.event_frame_id):
            self._update_texture(cam.event_tex, cam.event_rgb, "event_tex_shape", cam)
            cam.event_uploaded_tex_frame_id = int(cam.event_frame_id)
        GL.glEnable(GL.GL_BLEND)
        self._bind_quad(self.event_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, cam.event_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.event_prog, "u_event_tex"), 0)
        GL.glUniformMatrix3fv(GL.glGetUniformLocation(self.event_prog, "u_mvs_to_event"), 1, GL.GL_TRUE, H_mvs_to_ev)
        GL.glUniform2f(GL.glGetUniformLocation(self.event_prog, "u_mvs_size"), float(mv_w), float(mv_h))
        GL.glUniform2f(GL.glGetUniformLocation(self.event_prog, "u_event_size"), float(ev_w), float(ev_h))
        # Live preview must not tint the whole MVS image.
        # The event process may publish an "overlay" image whose background is not
        # perfectly black.  Use a much higher OpenGL-only key threshold by default
        # so only bright event/bullet pixels survive.  If event_overlay_pub_mode is
        # switched to "bullet", the background is black and this is also safe.
        black_raw = cam.overlay_cfg.get("opengl_black_threshold", cam.overlay_cfg.get("opengl_key_threshold", None))
        if black_raw is None:
            black_raw = cam.overlay_cfg.get("black_threshold", cam.overlay_cfg.get("chroma_black_threshold", 220))
        black = float(black_raw or 0.0) / 255.0
        opacity = float(cam.overlay_cfg.get("opengl_opacity", cam.overlay_cfg.get("opacity", cam.overlay_cfg.get("alpha", 0.28))) or 0.28)
        intensity_gamma = float(cam.overlay_cfg.get("opengl_intensity_gamma", 2.0) or 2.0)
        GL.glUniform1f(GL.glGetUniformLocation(self.event_prog, "u_black_threshold"), max(0.0, min(1.0, black)))
        GL.glUniform1f(GL.glGetUniformLocation(self.event_prog, "u_opacity"), max(0.0, min(1.0, opacity)))
        GL.glUniform1f(GL.glGetUniformLocation(self.event_prog, "u_intensity_gamma"), max(0.1, min(5.0, intensity_gamma)))
        # Do not recolor the whole event texture by default.  Preserve the event
        # process' sparse/bullet colors, unless explicitly requested.
        colorize = 1 if _as_bool(cam.overlay_cfg.get("opengl_colorize", False), False) else 0
        GL.glUniform1i(GL.glGetUniformLocation(self.event_prog, "u_colorize"), colorize)
        r, g, b = _parse_rgb_color(cam.overlay_cfg.get("opengl_color", cam.overlay_cfg.get("color", "yellow")))
        GL.glUniform3f(GL.glGetUniformLocation(self.event_prog, "u_color"), r, g, b)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _layout_viewports(self) -> Dict[str, Tuple[int, int, int, int]]:
        n = max(1, len(self.cameras))
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        W, H = max(1, self.width()), max(1, self.height())
        cell_w = W / cols
        cell_h = H / rows
        out: Dict[str, Tuple[int, int, int, int]] = {}
        for i, cam in enumerate(self.cameras):
            col = i % cols
            row_top = i // cols
            x0 = int(round(col * cell_w))
            y_top = int(round(row_top * cell_h))
            cw = int(round(cell_w))
            ch = int(round(cell_h))
            aspect = 16.0 / 9.0
            if cam.mvs_rgb is not None:
                h, w = cam.mvs_rgb.shape[:2]
                if h > 0:
                    aspect = float(w) / float(h)
            vw = cw
            vh = int(round(vw / aspect))
            if vh > ch:
                vh = ch
                vw = int(round(vh * aspect))
            vx = x0 + max(0, (cw - vw) // 2)
            # OpenGL viewport y origin is bottom-left. Convert top-row y to bottom-left.
            vy = H - (y_top + max(0, (ch + vh) // 2))
            out[cam.cam_id] = (vx, max(0, vy), max(1, vw), max(1, vh))
        return out

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        viewports = self._layout_viewports()
        for cam in self.cameras:
            vx, vy, vw, vh = viewports.get(cam.cam_id, (0, 0, self.width(), self.height()))
            GL.glViewport(vx, vy, vw, vh)
            self._draw_background(cam)
            self._draw_event_overlay(cam)
        GL.glUseProgram(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        self._paint_count += 1
        now = time.time()
        if now - self._paint_t0 >= 1.0:
            self._fps_value = self._paint_count / max(1e-6, now - self._paint_t0)
            self._paint_t0 = now
            self._paint_count = 0
        self._draw_text_overlay(viewports)

    def _mvs_to_qt(self, cam: CameraState, x: float, y: float, qx: int, qy: int, vw: int, vh: int) -> Tuple[float, float]:
        fw = float((cam.human_record or {}).get("frame_w") or (cam.mvs_rgb.shape[1] if cam.mvs_rgb is not None else vw) or vw)
        fh = float((cam.human_record or {}).get("frame_h") or (cam.mvs_rgb.shape[0] if cam.mvs_rgb is not None else vh) or vh)
        return qx + float(x) / max(1.0, fw) * vw, qy + float(y) / max(1.0, fh) * vh

    def _draw_event_frame_overlay(self, painter: QPainter, cam: CameraState, qx: int, qy: int, vw: int, vh: int) -> None:
        """Draw the projected event-camera image boundary on top of the MVS view.

        The event texture itself is drawn by OpenGL, but its true projected
        footprint can be hard to see when the event image is sparse.  This
        QPainter overlay draws the four corners of the event image after the
        same event->MVS homography used by the GL shader, so the operator can
        see the actual event-camera coverage in the fusion view.
        """
        if cam.mvs_rgb is None or cam.event_rgb is None:
            return
        cfg = cam.overlay_cfg if isinstance(cam.overlay_cfg, dict) else {}
        draw_border = _as_bool(
            cfg.get(
                "opengl_draw_event_border",
                cfg.get("draw_event_border", cfg.get("draw_border", True)),
            ),
            True,
        )
        if not draw_border:
            return
        mv_h, mv_w = cam.mvs_rgb.shape[:2]
        ev_h, ev_w = cam.event_rgb.shape[:2]
        if mv_h <= 0 or mv_w <= 0 or ev_h <= 0 or ev_w <= 0:
            return
        try:
            H = _homography_from_overlay_cfg(cfg, (ev_h, ev_w), (mv_h, mv_w)).astype(float)
        except Exception:
            return

        # Use the outer pixel rectangle of the event image.  Mapping these
        # through H gives the same quadrilateral that the event texture occupies
        # in MVS coordinates.
        event_corners = [
            (0.0, 0.0),
            (float(ev_w - 1), 0.0),
            (float(ev_w - 1), float(ev_h - 1)),
            (0.0, float(ev_h - 1)),
        ]
        qpoly = QPolygonF()
        mvs_pts = []
        for ex, ey in event_corners:
            p = H @ np.array([ex, ey, 1.0], dtype=float)
            if not np.all(np.isfinite(p)) or abs(float(p[2])) < 1e-9:
                return
            mx = float(p[0] / p[2])
            my = float(p[1] / p[2])
            mvs_pts.append((mx, my))
            sx, sy = self._mvs_to_qt(cam, mx, my, qx, qy, vw, vh)
            qpoly.append(QPointF(float(sx), float(sy)))
        if qpoly.size() < 4:
            return

        color_value = cfg.get("opengl_event_border_color", cfg.get("event_border_color", "cyan"))
        r, g, b = _parse_rgb_color(color_value)
        alpha_raw = cfg.get("opengl_event_border_alpha", cfg.get("event_border_alpha", 0.95))
        width_raw = cfg.get("opengl_event_border_width", cfg.get("event_border_width", 4))
        try:
            alpha = int(max(0, min(255, round(float(alpha_raw) * 255.0 if float(alpha_raw) <= 1.0 else float(alpha_raw)))))
        except Exception:
            alpha = 242
        try:
            width = max(1, int(round(float(width_raw))))
        except Exception:
            width = 4

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)

        # Dark halo first, then bright border, so it remains visible over both
        # bright and dark backgrounds.
        painter.setPen(QPen(QColor(0, 0, 0, min(220, alpha)), width + 4))
        painter.drawPolygon(qpoly)
        painter.setPen(QPen(QColor(int(r * 255), int(g * 255), int(b * 255), alpha), width))
        painter.drawPolygon(qpoly)

        if _as_bool(cfg.get("opengl_event_border_label", cfg.get("event_border_label", True)), True):
            try:
                tx = min(p.x() for p in qpoly)
                ty = min(p.y() for p in qpoly)
                label = f"EV {cam.cam_id} {ev_w}x{ev_h}"
                painter.fillRect(int(round(tx)), max(0, int(round(ty)) - 19), min(170, max(76, len(label) * 8)), 17, QColor(0, 0, 0, 170))
                painter.setPen(QColor(int(r * 255), int(g * 255), int(b * 255), alpha))
                painter.drawText(int(round(tx)) + 4, max(12, int(round(ty)) - 6), label)
            except Exception:
                pass

    def _human_record_for_display(self, cam: CameraState) -> Optional[Dict[str, Any]]:
        """Return a display-only human record predicted to the current MVS frame.

        The hit_judge still consumes the original human_result_*.jsonl.  This method
        only changes what the OpenGL preview draws, reducing the visible lag caused
        by YOLO inference time when players run quickly.
        """
        rec = cam.human_record
        if not isinstance(rec, dict):
            return rec
        if not self.human_prediction_enable or self.human_display_mode in {"detected_frame", "raw", "none"}:
            return rec
        persons = rec.get("persons") or []
        if not isinstance(persons, list) or not persons:
            return rec

        # Do not predict very stale detector output forever; a frozen mask is worse than no mask.
        try:
            record_wall_us = int(rec.get("record_wall_us", 0) or 0)
            if record_wall_us > 0:
                age_ms = (time.time() * 1_000_000.0 - float(record_wall_us)) / 1000.0
                if age_ms > self.human_prediction_max_record_age_ms:
                    return rec
        except Exception:
            pass

        target_sync = None
        for key in ("program_sync_index", "mvs_sync_index", "sync_index", "mvs_frame_num", "frame_id"):
            try:
                if key in cam.meta and cam.meta.get(key) is not None:
                    target_sync = int(cam.meta.get(key))
                    break
            except Exception:
                pass
        if target_sync is None:
            target_sync = int(cam.mvs_frame_id) if int(cam.mvs_frame_id) >= 0 else None
        source_sync = _record_sync_index(rec)
        if target_sync is None or source_sync is None:
            return rec
        delta_sync = int(target_sync) - int(source_sync)
        if delta_sync <= 0:
            return rec
        if delta_sync > self.human_prediction_max_frames:
            return rec

        history = list(cam.human_tailer.records) if cam.human_tailer is not None else []
        prev_by_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for old in reversed(history):
            old_sync = _record_sync_index(old)
            if old_sync is None or old_sync >= source_sync:
                continue
            for op in (old.get("persons") or []):
                if not isinstance(op, dict):
                    continue
                sid = str(op.get("stable_id", op.get("person_id", "")))
                if sid and sid not in prev_by_id:
                    prev_by_id[sid] = (int(old_sync), op)
            if len(prev_by_id) >= len(persons):
                break

        pred = copy.deepcopy(rec)
        pred_persons = []
        for p in persons:
            if not isinstance(p, dict):
                continue
            sid = str(p.get("stable_id", p.get("person_id", "")))
            cur_c = _person_center_from_dict(p)
            dx = dy = 0.0
            if sid in prev_by_id and cur_c is not None:
                old_sync, old_p = prev_by_id[sid]
                old_c = _person_center_from_dict(old_p)
                ds = max(1, int(source_sync) - int(old_sync))
                if old_c is not None:
                    vx = (float(cur_c[0]) - float(old_c[0])) / float(ds)
                    vy = (float(cur_c[1]) - float(old_c[1])) / float(ds)
                    dx = vx * float(delta_sync)
                    dy = vy * float(delta_sync)
                    mag = float((dx * dx + dy * dy) ** 0.5)
                    if mag > self.human_prediction_max_shift_px > 0:
                        scale = self.human_prediction_max_shift_px / max(1e-6, mag)
                        dx *= scale
                        dy *= scale
            pred_persons.append(_shift_person_for_display(p, dx, dy))
        pred["persons"] = pred_persons
        pred["display_prediction"] = {
            "enabled": True,
            "target_sync": int(target_sync),
            "source_sync": int(source_sync),
            "delta_sync": int(delta_sync),
            "mode": self.human_display_mode,
        }
        return pred

    def _draw_human_overlay(self, painter: QPainter, cam: CameraState, qx: int, qy: int, vw: int, vh: int) -> None:
        rec = self._human_record_for_display(cam)
        if not self.show_human_overlay or not isinstance(rec, dict):
            return
        persons = rec.get("persons") or []
        if not isinstance(persons, list):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setFont(QFont("Monospace", 9))
        if self.human_draw_sync_debug and isinstance(rec.get("display_prediction"), dict):
            dp = rec.get("display_prediction") or {}
            try:
                txt = f"HUMAN pred d={int(dp.get('delta_sync', 0))} src={dp.get('source_sync')} tgt={dp.get('target_sync')}"
                painter.fillRect(qx + 8, qy + 34, min(360, max(180, len(txt) * 8)), 18, QColor(0, 0, 0, 150))
                painter.setPen(QColor(255, 255, 0, 230))
                painter.drawText(qx + 12, qy + 48, txt)
            except Exception:
                pass
        for p in persons:
            if not isinstance(p, dict):
                continue
            sid = p.get("stable_id", p.get("person_id", "?"))
            color = _person_qcolor(sid, 255)
            fill_color = _person_qcolor(sid, int(max(0.0, min(1.0, self.human_mask_alpha)) * 255.0))
            contour_color = _person_qcolor(sid, 235)

            box = p.get("bbox_xyxy") or p.get("bbox")
            x1 = y1 = x2 = y2 = None
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                    if x2 <= x1 or y2 <= y1:
                        x2 = x1 + max(1.0, float(box[2]))
                        y2 = y1 + max(1.0, float(box[3]))
                except Exception:
                    x1 = y1 = x2 = y2 = None

            polys = _extract_mask_polygons(p)
            drew_mask = False
            if polys and (self.human_draw_mask or self.human_draw_contour):
                for poly in polys:
                    qpoly = QPolygonF()
                    for px, py in poly:
                        sx, sy = self._mvs_to_qt(cam, px, py, qx, qy, vw, vh)
                        qpoly.append(QPointF(float(sx), float(sy)))
                    if qpoly.size() < 3:
                        continue
                    if self.human_draw_mask:
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(fill_color)
                        painter.drawPolygon(qpoly)
                    if self.human_draw_contour:
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(contour_color)
                        painter.drawPolygon(qpoly)
                    drew_mask = True

            # Draw bbox only when explicitly requested or when there is no polygon to show.
            if x1 is not None and (self.human_draw_box or not drew_mask):
                sx1, sy1 = self._mvs_to_qt(cam, x1, y1, qx, qy, vw, vh)
                sx2, sy2 = self._mvs_to_qt(cam, x2, y2, qx, qy, vw, vh)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(color)
                painter.drawRect(int(round(sx1)), int(round(sy1)), int(round(sx2 - sx1)), int(round(sy2 - sy1)))

            if self.human_draw_label:
                label_x = x1
                label_y = y1
                if label_x is None and polys:
                    label_x, label_y = polys[0][0]
                if label_x is None:
                    continue
                sx, sy = self._mvs_to_qt(cam, float(label_x), float(label_y), qx, qy, vw, vh)
                conf = p.get("confidence", "")
                label = f"P{sid}"
                local_xy = p.get("local_xy")
                if isinstance(local_xy, (list, tuple)) and len(local_xy) >= 2:
                    try:
                        label += f" F({float(local_xy[0]):.2f},{float(local_xy[1]):.2f})"
                    except Exception:
                        pass
                elif conf != "":
                    try:
                        label += f" {float(conf):.2f}"
                    except Exception:
                        pass
                painter.fillRect(int(round(sx)), max(0, int(round(sy)) - 18), min(150, max(50, len(label) * 8)), 16, QColor(0, 0, 0, 150))
                painter.setPen(color)
                painter.drawText(int(round(sx)) + 3, max(12, int(round(sy)) - 5), label)

    def _draw_hit_overlay(self, painter: QPainter, cam: CameraState, qx: int, qy: int, vw: int, vh: int) -> None:
        """
        只显示 hit_judge_server 真正判定为命中的 hit_candidate。

        保留显示：
          1. MVS 画面
          2. 人体 bbox / mask
          3. 普通子弹轨迹 / bullet_effect / event overlay

        过滤掉：
          1. human_timeout
          2. no_geometry_match
          3. debug 记录
          4. 普通 bullet_effect 误检轨迹
          5. 同一 bullet_id 重复 candidate
        """
        if not self.show_hit_overlay:
            return

        now = time.time()
        painter.setFont(QFont("Monospace", 10))

        # 同一发子弹可能连续几帧都生成 candidate。
        # 这里按 bullet_id 去重，只显示分数最高；分数相同显示更新的一条。
        best_by_bullet = {}

        for hit in list(cam.recent_hits):
            if not isinstance(hit, dict):
                continue

            try:
                judge_wall = float(hit.get("judge_wall_time", now) or now)
            except Exception:
                judge_wall = now

            age = now - judge_wall

            # HIT 显示保留 3 秒，避免旧命中一直挂在画面上。
            if age > 3.0:
                continue

            # 只接受 hit_candidate 里的候选。
            # 如果 source 字段存在，必须来自 hit_judge_server。
            source = str(hit.get("source", "") or "")
            if source and source != "hit_judge_server":
                continue

            # hit_candidate 应该带有几何命中方式。
            # no_geometry_match / human_timeout 通常没有这些字段，因此不会被画成 HIT。
            geometry_mode = str(
                hit.get("geometry_mode")
                or hit.get("match_method")
                or hit.get("geometry_best_method")
                or hit.get("method")
                or ""
            )

            ok_modes = {
                # Legacy direct geometry modes. These are still readable for old
                # candidate files, but hit_judge v7 normally no longer emits them
                # as final hits when impact-behavior mode is enabled.
                "point_in_mask",
                "point_in_bbox",
                "segment_intersect_mask",
                "segment_intersect_bbox",
                "segment_intersect",
                "near_bbox",

                # v7 impact-behavior final-hit modes. Without these two, the
                # judge can write a valid hit_candidate but the GL overlay will
                # silently ignore it, so no HIT effect is shown on screen.
                "hit_absorb",
                "hit_bounce",
            }

            if geometry_mode not in ok_modes:
                continue

            # HIT 位置必须是 MVS 坐标。
            pt = (
                hit.get("event_point")
                or hit.get("hit_point")
                or hit.get("event_point_mvs")
                or hit.get("point_mvs")
            )

            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue

            # 如果有 evidence_level，只接受可靠证据。v7 的真实最终击中
            # 使用 evidence_level="impact_behavior"，这里必须放行，否则
            # hit_absorb / hit_bounce 会被显示层过滤掉。
            evidence = str(hit.get("evidence_level", "") or "").lower()
            if evidence and evidence not in {"strong", "medium", "impact_behavior"}:
                continue

            bullet_id = hit.get("bullet_id", hit.get("global_bullet_id", None))
            if bullet_id is None:
                # 老格式没有 bullet_id 时，用点坐标兜底分组。
                try:
                    bullet_id = f"pt:{float(pt[0]):.1f},{float(pt[1]):.1f}"
                except Exception:
                    bullet_id = f"obj:{id(hit)}"

            try:
                score = float(
                    hit.get(
                        "total_score",
                        hit.get("score", hit.get("candidate_score", 0.0))
                    ) or 0.0
                )
            except Exception:
                score = 0.0

            key = str(bullet_id)
            old = best_by_bullet.get(key)

            if old is None:
                best_by_bullet[key] = (score, judge_wall, hit)
            else:
                old_score, old_wall, _old_hit = old
                if score > old_score + 1e-6:
                    best_by_bullet[key] = (score, judge_wall, hit)
                elif abs(score - old_score) <= 1e-6 and judge_wall > old_wall:
                    best_by_bullet[key] = (score, judge_wall, hit)

        if not best_by_bullet:
            return

        for _key, (_score, _wall, hit) in best_by_bullet.items():
            pt = (
                hit.get("event_point")
                or hit.get("hit_point")
                or hit.get("event_point_mvs")
                or hit.get("point_mvs")
            )

            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue

            try:
                mvs_x = float(pt[0])
                mvs_y = float(pt[1])
            except Exception:
                continue

            sx, sy = self._mvs_to_qt(cam, mvs_x, mvs_y, qx, qy, vw, vh)

            geometry_mode = str(
                hit.get("geometry_mode")
                or hit.get("match_method")
                or hit.get("geometry_best_method")
                or hit.get("method")
                or ""
            )

            # 红色 HIT 圈。impact-behavior 命中稍微画大一点，
            # 方便现场确认确实触发了画面层特效。
            painter.setPen(QColor(255, 40, 40, 245))
            r = 16 if geometry_mode in {"hit_absorb", "hit_bounce"} else 10
            painter.drawEllipse(
                int(round(sx - r)),
                int(round(sy - r)),
                2 * r,
                2 * r,
            )
            if geometry_mode in {"hit_absorb", "hit_bounce"}:
                r2 = r + 10
                painter.drawEllipse(
                    int(round(sx - r2)),
                    int(round(sy - r2)),
                    2 * r2,
                    2 * r2,
                )

            bullet_id = hit.get("bullet_id", "?")
            person_id = (
                hit.get("stable_id")
                or hit.get("person_id")
                or hit.get("track_id")
                or "?"
            )

            try:
                score = float(
                    hit.get(
                        "total_score",
                        hit.get("score", hit.get("candidate_score", 0.0))
                    ) or 0.0
                )
                score_txt = f"{score:.2f}"
            except Exception:
                score_txt = str(
                    hit.get("total_score", hit.get("score", ""))
                    or ""
                )

            geometry_mode = str(
                hit.get("geometry_mode")
                or hit.get("match_method")
                or hit.get("geometry_best_method")
                or hit.get("method")
                or ""
            )

            label = f"HIT b{bullet_id} P{person_id} {score_txt} {geometry_mode}"

            label_w = min(420, max(160, len(label) * 8))
            painter.fillRect(
                int(round(sx)) + 12,
                int(round(sy)) - 18,
                label_w,
                18,
                QColor(0, 0, 0, 180),
            )
            painter.drawText(
                int(round(sx)) + 16,
                int(round(sy)) - 5,
                label,
            )

    def _draw_text_overlay(self, viewports: Dict[str, Tuple[int, int, int, int]]) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(QFont("Monospace", 10))
        W, H = self.width(), self.height()
        for cam in self.cameras:
            vx, vy, vw, vh = viewports.get(cam.cam_id, (0, 0, W, H))
            # Convert GL bottom-left viewport to Qt top-left rect.
            qx = vx
            qy = H - vy - vh
            painter.fillRect(qx + 6, qy + 6, min(vw - 12, 760), 66, QColor(0, 0, 0, 150))
            painter.setPen(QColor(255, 230, 80))
            painter.drawText(qx + 14, qy + 24, f"{cam.cam_id}  GL preview={self._fps_value:.1f}fps")
            painter.setPen(QColor(255, 80, 80) if self._recording_hint else QColor(180, 220, 255))
            painter.drawText(qx + 14, qy + 44, f"S=ALL  O=Event RAW  P=MVS video | {self._record_status_text}")
            painter.setPen(QColor(200, 255, 200) if cam.mvs_rgb is not None else QColor(255, 120, 120))
            human_n = 0
            if isinstance(cam.human_record, dict):
                try:
                    human_n = int(cam.human_record.get("person_count", len(cam.human_record.get("persons") or [])) or 0)
                except Exception:
                    human_n = 0
            human_extra = ""
            if human_n == 0 and cam.human_tailer is not None and cam.human_record is None:
                err = getattr(cam.human_tailer, "last_error", "") or ""
                if err:
                    human_extra = " human_json:" + err[:46]
            painter.drawText(qx + 14, qy + 64, (cam.status + f"  human={human_n}" + human_extra)[:300])
            self._draw_event_frame_overlay(painter, cam, qx, qy, vw, vh)
            self._draw_human_overlay(painter, cam, qx, qy, vw, vh)
            self._draw_hit_overlay(painter, cam, qx, qy, vw, vh)
        painter.end()


# ----------------------------- application --------------------------------

def build_camera_states(cfg: Dict[str, Any], args: argparse.Namespace) -> List[CameraState]:
    routes = _routes_from_config(cfg)
    if not routes:
        raise RuntimeError("No enabled routes found in config")
    wanted = None
    if args.cams:
        wanted = {x.strip() for x in str(args.cams).split(",") if x.strip()}
    sync_cfg = cfg.get("sync") or {}
    root = os.path.abspath(str(sync_cfg.get("root", "./sync_ipc") or "./sync_ipc"))
    overlay_base = dict(cfg.get("event_overlay") or {})
    per_route = overlay_base.get("per_route") if isinstance(overlay_base.get("per_route"), dict) else {}
    renderer_cfg = cfg.get("opengl_renderer") if isinstance(cfg.get("opengl_renderer"), dict) else {}
    human_overlay_cfg0 = renderer_cfg.get("human_overlay") if isinstance(renderer_cfg.get("human_overlay"), dict) else {}
    try:
        human_keep_records = max(32, int(human_overlay_cfg0.get("history_frames", 160) or 160) + 16)
    except Exception:
        human_keep_records = 176
    mvs_template = str(args.mvs_name_template or sync_cfg.get("mvs_frame_pub_name_template", "mvs_latest_{camera}") or "mvs_latest_{camera}")
    event_template = str(args.event_name_template or overlay_base.get("name_template", "event_overlay_latest_{camera}") or "event_overlay_latest_{camera}")
    meta_template = str(sync_cfg.get("mvs_frame_pub_meta_template", "{root}/mvs_latest_{camera}.json") or "{root}/mvs_latest_{camera}.json")
    cams: List[CameraState] = []
    for r in routes:
        cam_id = str(r.get("id", "") or "").strip()
        if not cam_id:
            continue
        if wanted is not None and cam_id not in wanted:
            continue
        rcfg = _deep_merge(overlay_base, per_route.get(cam_id, {}) if isinstance(per_route, dict) else {})
        # Per-route config should not contain nested per_route again.
        rcfg.pop("per_route", None)
        mvs_name = mvs_template.format(camera=cam_id, cam=cam_id, id=cam_id)
        event_name = event_template.format(camera=cam_id, cam=cam_id, id=cam_id)
        meta_path = meta_template.format(root=root, camera=cam_id, cam=cam_id, id=cam_id)
        trigger_path = _sync_template_path(sync_cfg, "event_trigger_csv_template", cam_id, "{root}/event_trigger_{camera}.csv")
        human_path = _sync_template_path(sync_cfg, "human_jsonl_template", cam_id, "{root}/human_result_{camera}.jsonl")
        hit_cfg = cfg.get("hit_judge") if isinstance(cfg.get("hit_judge"), dict) else {}
        hit_template = str(hit_cfg.get("hit_jsonl_template", "{root}/hit_candidate_{camera}.jsonl") if isinstance(hit_cfg, dict) else "{root}/hit_candidate_{camera}.jsonl")
        hit_path = hit_template.format(root=root, camera=cam_id, cam=cam_id, id=cam_id)
        if not os.path.isabs(hit_path):
            hit_path = os.path.abspath(hit_path)
        cams.append(CameraState(
            cam_id=cam_id,
            mvs_name=mvs_name,
            event_name=event_name,
            overlay_cfg=rcfg,
            mvs_reader=LazySharedReader(mvs_name),
            event_reader=LazySharedReader(event_name),
            meta_path=meta_path,
            trigger_path=trigger_path,
            trigger_tailer=TriggerCsvTailer(trigger_path, keep=160, start_at_end=True, startup_tail_rows=160),
            human_path=human_path,
            human_tailer=JsonlTailer(human_path, max_records=human_keep_records, start_at_end=True, startup_tail_records=human_keep_records, max_lines_per_poll=64),
            hit_path=hit_path,
            hit_tailer=JsonlTailer(hit_path, max_records=64, start_at_end=True, startup_tail_records=0, max_lines_per_poll=128),
        ))
    if not cams:
        raise RuntimeError("No cameras selected")
    return cams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qt/OpenGL MVS + Event fusion preview without cv2.imshow")
    p.add_argument("--config", default="hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json", help="MVS fusion JSON config")
    p.add_argument("--cams", default="", help="Comma-separated camera ids, e.g. A,B,C,D,E,F. Empty means all enabled routes.")
    p.add_argument("--preview-fps", type=float, default=120.0, help="Qt/OpenGL window refresh target. 120 is enough for most monitors.")
    p.add_argument("--event-stale-ms", type=float, default=250.0, help="Mark event overlay stale after this age.")
    p.add_argument("--mvs-name-template", default="", help="Override MVS shared-memory name template")
    p.add_argument("--event-name-template", default="", help="Override event shared-memory name template")
    p.add_argument("--no-human-overlay", action="store_true", help="Do not draw YOLO human boxes from human_result_*.jsonl")
    p.add_argument("--no-hit-overlay", action="store_true", help="Do not draw hit candidates from hit_candidate_*.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg, cfg_path = _load_json_config(args.config)
    os.chdir(os.path.dirname(cfg_path) or os.getcwd())
    print(f"[fusion_gl] using config: {cfg_path}", flush=True)
    cams = build_camera_states(cfg, args)
    print("[fusion_gl] cameras:", ", ".join([c.cam_id for c in cams]), flush=True)
    print("[fusion_gl] MVS shm:", ", ".join([c.mvs_name for c in cams]), flush=True)
    print("[fusion_gl] EVT shm:", ", ".join([c.event_name for c in cams]), flush=True)
    print("[fusion_gl] trigger csv:", ", ".join([str(c.trigger_path) for c in cams]), flush=True)
    print("[fusion_gl] human jsonl:", ", ".join([str(c.human_path) for c in cams]), flush=True)
    print("[fusion_gl] hit jsonl:", ", ".join([str(c.hit_path) for c in cams]), flush=True)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    human_overlay_cfg = copy.deepcopy(((cfg.get("common") or {}).get("output") or {}) if isinstance(cfg.get("common"), dict) else {})
    ogl_cfg = cfg.get("opengl_renderer") if isinstance(cfg.get("opengl_renderer"), dict) else {}
    if isinstance(ogl_cfg.get("human_overlay"), dict):
        human_overlay_cfg = _deep_merge(human_overlay_cfg, ogl_cfg.get("human_overlay") or {})
    # Display delay is a preview-only OpenGL setting.  Keep it in the renderer
    # config, but pass it through the existing human_overlay_cfg channel so the
    # widget can use it without changing any capture/hit_judge APIs.
    for _k in ("display_sync_mode", "display_delay_ms", "display_buffer_ms", "display_audit_enable", "display_audit_interval_ms", "display_audit_dir"):
        if _k in ogl_cfg and _k not in human_overlay_cfg:
            human_overlay_cfg[_k] = ogl_cfg.get(_k)
    w = FusionGLWidget(cams, preview_fps=args.preview_fps, event_stale_ms=args.event_stale_ms, record_cfg=cfg.get("one_key_record") or {}, sync_cfg=cfg.get("sync") or {}, show_human_overlay=(not args.no_human_overlay), show_hit_overlay=(not args.no_hit_overlay), human_overlay_cfg=human_overlay_cfg)

    # Make Ctrl+C reliable.  Qt's event loop can otherwise swallow SIGINT until
    # the next Python callback, which made the renderer hard to stop from a
    # terminal.  The timer keeps the interpreter awake for signal delivery.
    def _request_quit(signum=None, frame=None):
        try:
            print("[fusion_gl] received interrupt, closing...", flush=True)
        except Exception:
            pass
        try:
            w.close()
        except Exception:
            pass
        try:
            app.quit()
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _request_quit)
        signal.signal(signal.SIGTERM, _request_quit)
    except Exception:
        pass
    _sig_timer = QTimer()
    _sig_timer.timeout.connect(lambda: None)
    _sig_timer.start(100)

    w.show()
    try:
        rc = int(app.exec())
    except KeyboardInterrupt:
        _request_quit()
        rc = 130
    try:
        _sig_timer.stop()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
