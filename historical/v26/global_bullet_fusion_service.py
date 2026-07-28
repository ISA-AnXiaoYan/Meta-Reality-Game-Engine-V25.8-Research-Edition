#!/usr/bin/env python3
"""Sidecar that projects live bullet points into field coordinates.

This is intentionally conservative. It consumes hit_judge's
overlay_bullet_point_all.jsonl stream, projects MVS pixel coordinates through
per-camera field calibration, and publishes compact field-track diagnostics for
downstream damage attribution. If calibration is missing, the point is skipped
and the status JSON explains why.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


Matrix3 = List[List[float]]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _resolve_path(sync_root: Path, project_root: Path, value: str) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    if str(value).startswith("sync_ipc/") or str(value).startswith("sync_ipc\\"):
        return project_root / p
    return sync_root / p


def _flatten_matrix(value: Any) -> Optional[Matrix3]:
    if value is None:
        return None
    vals: List[float] = []
    if isinstance(value, list) and len(value) == 3 and all(isinstance(r, list) for r in value):
        try:
            return [[float(value[r][c]) for c in range(3)] for r in range(3)]
        except Exception:
            return None
    if isinstance(value, list):
        try:
            vals = [float(x) for x in value]
        except Exception:
            return None
    if len(vals) != 9:
        return None
    return [vals[0:3], vals[3:6], vals[6:9]]


def _det3(m: Matrix3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _invert3(m: Matrix3) -> Matrix3:
    det = _det3(m)
    if abs(det) < 1e-12:
        raise ValueError("singular homography")
    inv_det = 1.0 / det
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]


def _apply_h(m: Matrix3, x: float, y: float) -> Optional[Tuple[float, float]]:
    z = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(z) < 1e-12:
        return None
    fx = (m[0][0] * x + m[0][1] * y + m[0][2]) / z
    fy = (m[1][0] * x + m[1][1] * y + m[1][2]) / z
    if not (math.isfinite(fx) and math.isfinite(fy)):
        return None
    return fx, fy


def _load_img_to_field(calib_path: Path) -> Tuple[Matrix3, Dict[str, Any]]:
    data = json.loads(calib_path.read_text(encoding="utf-8-sig"))
    for key in ("image_to_field_homography", "homography_image_to_local"):
        h = _flatten_matrix(data.get(key))
        if h is not None:
            return h, {"path": str(calib_path), "matrix_source": key}
    for key in ("field_to_image_homography", "homography_local_to_image"):
        h = _flatten_matrix(data.get(key))
        if h is not None:
            return _invert3(h), {"path": str(calib_path), "matrix_source": f"inverse({key})"}
    raise RuntimeError(f"no supported homography in {calib_path}")


def _transform_xy(x: float, y: float, mode: str, origin_x: float, origin_y: float, width: float, height: float) -> Tuple[float, float]:
    m = str(mode or "normal").strip().lower().replace("-", "_")
    if m in {"field_xy", "raw", "none", "normal"}:
        return x, y
    if m == "neg_y":
        return x, -y
    if m in {"field_height_plus_y", "height_plus_y"}:
        return x, origin_y + height + y
    if m in {"invert_y", "flip_y"}:
        return x, origin_y + height - (y - origin_y)
    if m == "flip_x":
        return origin_x + width - (x - origin_x), y
    if m == "flip_xy":
        return origin_x + width - (x - origin_x), origin_y + height - (y - origin_y)
    if m == "swap_xy":
        return y, x
    if m == "swap_xy_neg_y":
        return -y, x
    return x, y


class JsonlTailer:
    def __init__(self, path: Path, read_existing: bool = False) -> None:
        self.path = path
        self.offset = 0
        self.error = ""
        self._initialized = False
        self.read_existing = read_existing

    def read_new(self, max_lines: int = 1000) -> List[Dict[str, Any]]:
        if not self.path.exists():
            self.error = "missing"
            return []
        try:
            size = self.path.stat().st_size
            if not self._initialized:
                self.offset = 0 if self.read_existing else size
                self._initialized = True
            if size < self.offset:
                self.offset = 0
            out: List[Dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8", errors="replace") as fp:
                fp.seek(self.offset)
                for _ in range(max_lines):
                    line = fp.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        pass
                self.offset = fp.tell()
            self.error = ""
            return out
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []


@dataclass
class BulletTrack:
    key: str
    camera: str
    camera_sn: str
    bullet_id: int
    points: Deque[Tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=64))
    last_record: Dict[str, Any] = field(default_factory=dict)

    def add_point(self, ts_us: int, x: float, y: float, rec: Dict[str, Any]) -> None:
        if self.points and ts_us < self.points[-1][0]:
            ts_us = self.points[-1][0]
        self.points.append((int(ts_us), float(x), float(y)))
        self.last_record = dict(rec)

    def to_json(self) -> Dict[str, Any]:
        pts = list(self.points)
        first = pts[0]
        last = pts[-1]
        prev = pts[-2] if len(pts) >= 2 else first
        dt = max(0.0, (last[0] - prev[0]) / 1_000_000.0)
        vx = (last[1] - prev[1]) / dt if dt > 1e-6 else 0.0
        vy = (last[2] - prev[2]) / dt if dt > 1e-6 else 0.0
        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            direction = [round(vx / speed, 6), round(vy / speed, 6)]
        else:
            direction = [0.0, 0.0]
        return {
            "type": "global_bullet_track",
            "schema_version": 1,
            "track_id": self.key,
            "camera": self.camera,
            "camera_sn": self.camera_sn,
            "bullet_id": int(self.bullet_id),
            "point_count": len(pts),
            "first_ts_us": int(first[0]),
            "last_ts_us": int(last[0]),
            "field_xy": [round(float(last[1]), 4), round(float(last[2]), 4)],
            "prev_field_xy": [round(float(prev[1]), 4), round(float(prev[2]), 4)],
            "vx_mps": round(float(vx), 4),
            "vy_mps": round(float(vy), 4),
            "speed_mps": round(float(speed), 4),
            "direction_xy": direction,
            "age_ms": round(max(0.0, (_now_ms() - int(last[0] / 1000))), 2),
            "source_record": {
                "point_index": _safe_int(self.last_record.get("point_index"), -1),
                "status": str(self.last_record.get("status", "")),
                "confidence": _safe_float(self.last_record.get("confidence"), 1.0),
                "x_mvs": _safe_float(self.last_record.get("x")),
                "y_mvs": _safe_float(self.last_record.get("y")),
            },
        }


class GlobalBulletFusionService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.sync_root = Path(args.sync_root).resolve()
        self.input_path = _resolve_path(self.sync_root, self.project_root, args.input_jsonl)
        self.output_path = _resolve_path(self.sync_root, self.project_root, args.output_jsonl)
        self.latest_path = _resolve_path(self.sync_root, self.project_root, args.latest_json)
        self.status_path = _resolve_path(self.sync_root, self.project_root, args.status_json)
        self.tailer = JsonlTailer(self.input_path, bool(args.read_existing))
        self.out_fp = None
        self.tracks: Dict[str, BulletTrack] = {}
        self.calib: Dict[str, Matrix3] = {}
        self.calib_meta: Dict[str, Dict[str, Any]] = {}
        self.stats = Counter()
        self.latest_tracks: List[Dict[str, Any]] = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode_enabled:
            self.out_fp = self.output_path.open("a", encoding="utf-8", buffering=1)

    @property
    def mode_enabled(self) -> bool:
        return str(self.args.mode) not in {"disabled", "off"}

    def _calib_path(self, cam: str) -> Path:
        p = Path(str(self.args.calib_template).format(camera=cam, cam=cam, id=cam))
        return p if p.is_absolute() else self.project_root / p

    def _img_to_field(self, cam: str) -> Optional[Matrix3]:
        cam = str(cam or "").strip()
        if not cam:
            return None
        if cam in self.calib:
            return self.calib[cam]
        path = self._calib_path(cam)
        try:
            h, meta = _load_img_to_field(path)
            self.calib[cam] = h
            self.calib_meta[cam] = meta
            return h
        except Exception as exc:
            self.calib_meta[cam] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            self.stats["missing_calib"] += 1
            return None

    def _track_key(self, rec: Dict[str, Any], cam: str) -> Tuple[str, int]:
        bullet_id = _safe_int(rec.get("bullet_id"), 0)
        camera_sn = str(rec.get("camera_sn") or "")
        global_id = str(rec.get("global_bullet_id") or "").strip()
        if global_id:
            return f"global:{global_id}", bullet_id
        return f"{cam}:{camera_sn}:{bullet_id}", bullet_id

    def _process_record(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cam = str(rec.get("camera_alias") or rec.get("pair_id") or rec.get("camera") or "").strip()
        if not cam:
            self.stats["drop_no_camera"] += 1
            return None
        x = _safe_float(rec.get("x"))
        y = _safe_float(rec.get("y"))
        if not (math.isfinite(x) and math.isfinite(y)):
            self.stats["drop_bad_xy"] += 1
            return None
        h = self._img_to_field(cam)
        if h is None:
            self.stats["drop_no_calib"] += 1
            return None
        raw = _apply_h(h, x, y)
        if raw is None:
            self.stats["drop_project_failed"] += 1
            return None
        fx, fy = _transform_xy(
            raw[0],
            raw[1],
            str(self.args.coord_transform_mode),
            float(self.args.field_origin_x_m),
            float(self.args.field_origin_y_m),
            float(self.args.field_width_m),
            float(self.args.field_height_m),
        )
        key, bullet_id = self._track_key(rec, cam)
        tr = self.tracks.get(key)
        if tr is None:
            tr = BulletTrack(key=key, camera=cam, camera_sn=str(rec.get("camera_sn") or ""), bullet_id=bullet_id)
            self.tracks[key] = tr
            self.stats["created_tracks"] += 1
        ts_us = _safe_int(rec.get("point_ts_us") or rec.get("ts_us") or rec.get("event_ts") or int(time.time() * 1_000_000), int(time.time() * 1_000_000))
        tr.add_point(ts_us, fx, fy, rec)
        out = tr.to_json()
        out["raw_field_xy"] = [round(float(raw[0]), 4), round(float(raw[1]), 4)]
        out["coord_transform_mode"] = str(self.args.coord_transform_mode)
        out["source"] = "global_bullet_fusion_service"
        out["wall_time"] = time.time()
        self.stats["accepted_points"] += 1
        return out

    def _prune(self) -> None:
        ttl_ms = max(100.0, float(self.args.track_ttl_ms))
        now_us = int(time.time() * 1_000_000)
        old = [k for k, tr in self.tracks.items() if tr.points and (now_us - tr.points[-1][0]) / 1000.0 > ttl_ms]
        for k in old:
            self.tracks.pop(k, None)
        if old:
            self.stats["retired_tracks"] += len(old)

    def _publish_status(self) -> None:
        active = [tr.to_json() for tr in self.tracks.values() if len(tr.points) >= int(self.args.min_points)]
        active.sort(key=lambda d: int(d.get("last_ts_us", 0)), reverse=True)
        self.latest_tracks = active[: int(self.args.latest_max_tracks)]
        latest = {
            "schema_version": 1,
            "msg_type": "global_bullet_fusion_latest",
            "mode": str(self.args.mode),
            "timestamp_ms": _now_ms(),
            "active_track_count": len(active),
            "tracks": self.latest_tracks,
        }
        status = {
            "schema_version": 1,
            "source": "global_bullet_fusion_service",
            "mode": str(self.args.mode),
            "input_jsonl": str(self.input_path),
            "output_jsonl": str(self.output_path),
            "latest_json": str(self.latest_path),
            "status_json": str(self.status_path),
            "track_count": len(self.tracks),
            "active_track_count": len(active),
            "stats": dict(self.stats),
            "tail_error": str(self.tailer.error),
            "calibration": dict(self.calib_meta),
            "timestamp_ms": _now_ms(),
        }
        _atomic_write_json(self.latest_path, latest)
        _atomic_write_json(self.status_path, status)

    def run(self) -> None:
        print(
            f"[global_bullet_fusion] start mode={self.args.mode} input={self.input_path} output={self.output_path}",
            flush=True,
        )
        next_status = 0.0
        while True:
            if not self.mode_enabled:
                self._publish_status()
                time.sleep(max(0.1, float(self.args.interval_sec)))
                continue
            records = self.tailer.read_new(max_lines=int(self.args.max_lines_per_poll))
            self.stats["input_records"] += len(records)
            for rec in records:
                out = self._process_record(rec)
                if out is not None and self.out_fp is not None:
                    self.out_fp.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._prune()
            now = time.time()
            if now >= next_status:
                self._publish_status()
                next_status = now + max(0.1, float(self.args.status_interval_sec))
            time.sleep(max(0.005, float(self.args.interval_sec)))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fuse live bullet points into global field tracks")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--mode", choices=["disabled", "shadow_no_publish", "shadow", "publish", "enforce"], default="disabled")
    ap.add_argument("--input-jsonl", default="overlay_bullet_point_all.jsonl")
    ap.add_argument("--output-jsonl", default="global_bullet_tracks.jsonl")
    ap.add_argument("--latest-json", default="global_bullet_fusion_latest.json")
    ap.add_argument("--status-json", default="global_bullet_fusion_status.json")
    ap.add_argument("--calib-template", default="calib_true/field_calib_camera{camera}.json")
    ap.add_argument("--coord-transform-mode", default="flip_y")
    ap.add_argument("--field-origin-x-m", type=float, default=0.0)
    ap.add_argument("--field-origin-y-m", type=float, default=0.0)
    ap.add_argument("--field-width-m", type=float, default=33.0)
    ap.add_argument("--field-height-m", type=float, default=22.0)
    ap.add_argument("--interval-sec", type=float, default=0.04)
    ap.add_argument("--status-interval-sec", type=float, default=1.0)
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--track-ttl-ms", type=float, default=1500.0)
    ap.add_argument("--latest-max-tracks", type=int, default=128)
    ap.add_argument("--max-lines-per-poll", type=int, default=2000)
    ap.add_argument("--read-existing", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    GlobalBulletFusionService(args).run()


if __name__ == "__main__":
    main()
