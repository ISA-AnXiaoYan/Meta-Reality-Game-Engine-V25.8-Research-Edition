#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent human-mask evidence ticker for Event/Hit final gating.

This service deliberately runs outside the 4ms Event worker loop.  It samples
the latest Global YOLO human_result JSONL files at a configurable low rate and
publishes per-camera evidence snapshots for hit_judge/status consumers.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional on non-vision hosts
    cv2 = None
    np = None


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _mono_ns() -> int:
    return int(time.monotonic_ns())


def _json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


MASK_RASTER_MAGIC = b"HMRSTR1\0"
MASK_RASTER_VERSION = 1
MASK_RASTER_HEADER = struct.Struct("<8sIIQIIIQ")


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").replace(";", ",").split(",") if x.strip()]


def _resolve_shm_template(template: str, cam: str) -> str:
    raw = str(template or "/dev/shm/event_human_mask_raster_{camera}.mmap")
    raw = raw.format(camera=cam, cam=cam, id=cam, alias=cam)
    if raw.startswith("/"):
        return raw
    if raw.endswith(".mmap"):
        return os.path.join("/dev/shm", raw)
    return os.path.join("/dev/shm", raw + ".mmap")


def _safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _event_overlay_homography_for_camera(cfg: Dict[str, Any], cam: str):
    if np is None:
        return None, "numpy_unavailable"
    candidates: List[tuple[str, Dict[str, Any]]] = []
    if isinstance(cfg.get("event_overlay"), dict):
        candidates.append(("event_overlay", cfg.get("event_overlay") or {}))
    for section_name in ("mvs_config", "mvs", "global_bev"):
        section = cfg.get(section_name)
        if isinstance(section, dict) and isinstance(section.get("event_overlay"), dict):
            candidates.append((f"{section_name}.event_overlay", section.get("event_overlay") or {}))
    for source, overlay_base in candidates:
        per_route = overlay_base.get("per_route") if isinstance(overlay_base.get("per_route"), dict) else {}
        rcfg: Dict[str, Any] = {}
        rcfg.update(overlay_base)
        if isinstance(per_route, dict) and isinstance(per_route.get(str(cam)), dict):
            rcfg.update(per_route.get(str(cam), {}))
        hs = rcfg.get("homographies_event_to_mvs") if isinstance(rcfg.get("homographies_event_to_mvs"), dict) else {}
        active = str(rcfg.get("active_homography", "") or "")
        mat = None
        chosen = ""
        if active and active in hs:
            mat = hs.get(active)
            chosen = active
        elif "flight" in hs:
            mat = hs.get("flight")
            chosen = "flight"
        elif "ground" in hs:
            mat = hs.get("ground")
            chosen = "ground"
        elif hs:
            chosen, mat = next(iter(hs.items()))
        if mat is None:
            continue
        try:
            arr = np.array(mat, dtype=np.float64).reshape(3, 3)
            return arr, f"{source}.{cam}.{chosen}"
        except Exception:
            continue
    return np.eye(3, dtype=np.float64), "identity_fallback"


def _apply_h(H, x: float, y: float) -> List[int]:
    if np is None or H is None:
        return [int(round(x)), int(round(y))]
    den = float(H[2, 0] * x + H[2, 1] * y + H[2, 2])
    if abs(den) < 1e-9:
        return [int(round(x)), int(round(y))]
    ex = float((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den)
    ey = float((H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den)
    return [int(round(ex)), int(round(ey))]


def _person_polygons(person: Dict[str, Any]) -> List[Any]:
    polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
    out: List[Any] = []
    if isinstance(polys, dict):
        polys = list(polys.values())
    if isinstance(polys, list):
        for poly in polys:
            if isinstance(poly, list) and len(poly) >= 3:
                out.append(poly)
    if out:
        return out
    box = person.get("bbox_xyxy") or person.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if x2 > x1 and y2 > y1:
                return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        except Exception:
            pass
    return []


class MaskRasterPublisher:
    def __init__(self, template: str, width: int, height: int):
        self.template = str(template or "/dev/shm/event_human_mask_raster_{camera}.mmap")
        self.width = max(1, int(width or 1280))
        self.height = max(1, int(height or 720))
        self.data_bytes = self.width * self.height
        self.total_bytes = MASK_RASTER_HEADER.size + self.data_bytes
        self._items: Dict[str, Any] = {}
        self._seq: Dict[str, int] = {}

    def _open(self, cam: str):
        item = self._items.get(cam)
        if item is not None:
            return item
        path = _resolve_shm_template(self.template, cam)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, self.total_bytes)
        mm = mmap.mmap(fd, self.total_bytes, access=mmap.ACCESS_WRITE)
        item = (fd, mm, path)
        self._items[cam] = item
        self._seq[cam] = 0
        return item

    def publish(self, cam: str, mask, publish_wall_us: int) -> Dict[str, Any]:
        fd, mm, path = self._open(str(cam))
        seq = int(self._seq.get(str(cam), 0)) + 2
        if seq % 2:
            seq += 1
        if np is None:
            raise RuntimeError("numpy unavailable")
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.shape != (self.height, self.width):
            raise ValueError(f"mask shape {arr.shape} != {(self.height, self.width)}")
        odd = seq - 1
        mm.seek(0)
        mm.write(MASK_RASTER_HEADER.pack(MASK_RASTER_MAGIC, MASK_RASTER_VERSION, MASK_RASTER_HEADER.size,
                                         odd, self.width, self.height, self.data_bytes, 0))
        mm.seek(MASK_RASTER_HEADER.size)
        mm.write(arr.reshape(-1).tobytes(order="C"))
        mm.seek(0)
        mm.write(MASK_RASTER_HEADER.pack(MASK_RASTER_MAGIC, MASK_RASTER_VERSION, MASK_RASTER_HEADER.size,
                                         seq, self.width, self.height, self.data_bytes, int(publish_wall_us)))
        self._seq[str(cam)] = seq
        return {"path": path, "seq": seq, "width": self.width, "height": self.height, "data_bytes": self.data_bytes}

    def close(self) -> None:
        for fd, mm, _path in list(self._items.values()):
            try:
                mm.close()
            except Exception:
                pass
            try:
                os.close(fd)
            except Exception:
                pass
        self._items.clear()


def _count_polygons(persons: List[Dict[str, Any]]) -> int:
    n = 0
    for person in persons:
        if not isinstance(person, dict):
            continue
        polys = person.get("mask_polygons") or person.get("polygons") or person.get("poly") or []
        if isinstance(polys, list):
            if polys and isinstance(polys[0], list) and polys[0] and isinstance(polys[0][0], (int, float)):
                n += 1
            else:
                n += len([p for p in polys if isinstance(p, list) and p])
    return n


class JsonlTailer:
    def __init__(self, path: Path, tail_bytes: int = 4_000_000):
        self.path = path
        self.tail_bytes = max(4096, int(tail_bytes))
        self.pos = 0
        self.inode: Optional[int] = None
        self.last_error = ""

    def _reset_if_needed(self) -> bool:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self.last_error = "missing"
            self.pos = 0
            self.inode = None
            return False
        except Exception as exc:
            self.last_error = f"stat:{type(exc).__name__}:{exc}"
            return False
        inode = int(getattr(st, "st_ino", 0) or 0)
        if self.inode != inode or self.pos > int(st.st_size):
            self.inode = inode
            self.pos = max(0, int(st.st_size) - int(self.tail_bytes))
        return True

    def poll(self) -> List[Dict[str, Any]]:
        if not self._reset_if_needed():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fp:
                fp.seek(self.pos)
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
                self.pos = fp.tell()
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"read:{type(exc).__name__}:{exc}"
        return out


class EventMaskEvidenceService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cams = [x.strip() for x in str(args.cams or "").replace(";", ",").split(",") if x.strip()]
        self.sync_root = Path(args.sync_root).expanduser()
        if not self.sync_root.is_absolute():
            self.sync_root = (Path.cwd() / self.sync_root).resolve()
        self.control_path = self._resolve_path(args.control_json)
        self.status_path = self._resolve_path(args.status_json)
        self.evidence_template = str(args.evidence_json_template or "{root}/event_mask_evidence_{camera}.json")
        self.tailers: Dict[str, JsonlTailer] = {}
        self.latest_records: Dict[str, Dict[str, Any]] = {}
        self.latest_evidence: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {
            "ticks": 0,
            "records_loaded": 0,
            "control_seq": 0,
            "control_error": "",
            "last_tick_us": 0,
            "last_tick_mono_ns": 0,
        }
        self.period_ms = max(4.0, float(args.period_ms or 16.0))
        self.stale_ms = max(1.0, float(args.stale_ms or 120.0))
        self.enabled = bool(args.enable)
        self.running = True
        self.mask_raster_enable = bool(getattr(args, "mask_raster_enable", False))
        self.mask_raster_template = str(getattr(args, "mask_raster_template", "") or "/dev/shm/event_human_mask_raster_{camera}.mmap")
        self.mask_raster_width = max(1, int(getattr(args, "mask_raster_width", 1280) or 1280))
        self.mask_raster_height = max(1, int(getattr(args, "mask_raster_height", 720) or 720))
        self.mask_raster_dilate_px = max(0, int(getattr(args, "mask_raster_dilate_px", 0) or 0))
        self.mask_raster_erode_px = max(0, int(getattr(args, "mask_raster_erode_px", 0) or 0))
        self.mask_raster_min_conf = float(getattr(args, "mask_raster_min_conf", 0.0) or 0.0)
        self.mask_raster_publisher: Optional[MaskRasterPublisher] = None
        self.mask_raster_meta: Dict[str, Dict[str, Any]] = {}
        self.mask_raster_h_mvs_to_event: Dict[str, Any] = {}
        self.mask_raster_h_source: Dict[str, str] = {}
        if self.mask_raster_enable:
            self.mask_raster_publisher = MaskRasterPublisher(
                self.mask_raster_template,
                self.mask_raster_width,
                self.mask_raster_height,
            )
            self._load_mask_raster_homographies(str(getattr(args, "mask_raster_mvs_config", "") or ""))
        for cam in self.cams:
            self.tailers[cam] = JsonlTailer(self._human_path(cam), tail_bytes=int(args.tail_bytes))

    def _resolve_path(self, p: str) -> Path:
        raw = str(p or "")
        raw = raw.format(root=str(self.sync_root))
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _human_path(self, cam: str) -> Path:
        raw = str(self.args.human_jsonl_template or "{root}/human_result_{camera}.jsonl")
        raw = raw.format(root=str(self.sync_root), camera=cam, cam=cam, id=cam)
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _evidence_path(self, cam: str) -> Path:
        raw = self.evidence_template.format(root=str(self.sync_root), camera=cam, cam=cam, id=cam)
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _poll_control(self) -> None:
        try:
            if not self.control_path.exists():
                self.stats["control_error"] = ""
                return
            obj = json.loads(self.control_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("control root must be object")
            seq = _safe_int(obj.get("seq"), 0)
            if seq and seq == _safe_int(self.stats.get("control_seq"), 0):
                return
            if "event_human_mask_evidence_period_ms" in obj:
                self.period_ms = min(1000.0, max(4.0, _safe_float(obj.get("event_human_mask_evidence_period_ms"), self.period_ms)))
            elif "period_ms" in obj:
                self.period_ms = min(1000.0, max(4.0, _safe_float(obj.get("period_ms"), self.period_ms)))
            if "event_human_mask_evidence_stale_ms" in obj:
                self.stale_ms = max(1.0, _safe_float(obj.get("event_human_mask_evidence_stale_ms"), self.stale_ms))
            if "event_human_mask_evidence_enable" in obj:
                self.enabled = bool(obj.get("event_human_mask_evidence_enable"))
            if "mask_raster_enable" in obj:
                self.mask_raster_enable = bool(obj.get("mask_raster_enable"))
                if self.mask_raster_enable and self.mask_raster_publisher is None:
                    self.mask_raster_publisher = MaskRasterPublisher(
                        self.mask_raster_template,
                        self.mask_raster_width,
                        self.mask_raster_height,
                    )
            if "mask_raster_dilate_px" in obj:
                self.mask_raster_dilate_px = max(0, _safe_int(obj.get("mask_raster_dilate_px"), self.mask_raster_dilate_px))
            if "mask_raster_erode_px" in obj:
                self.mask_raster_erode_px = max(0, _safe_int(obj.get("mask_raster_erode_px"), self.mask_raster_erode_px))
            if "mask_raster_min_conf" in obj:
                self.mask_raster_min_conf = max(0.0, _safe_float(obj.get("mask_raster_min_conf"), self.mask_raster_min_conf))
            self.stats["control_seq"] = seq
            self.stats["control_error"] = ""
            self.stats["control_applied_us"] = _now_us()
        except Exception as exc:
            self.stats["control_error"] = f"{type(exc).__name__}: {exc}"

    def _load_mask_raster_homographies(self, config_path: str) -> None:
        cfg: Dict[str, Any] = {}
        if config_path:
            path = Path(config_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            cfg = _safe_json_load(path)
        for cam in self.cams:
            h_event_to_mvs, h_source = _event_overlay_homography_for_camera(cfg, cam)
            self.mask_raster_h_source[cam] = str(h_source or "")
            if np is None or h_event_to_mvs is None:
                self.mask_raster_h_mvs_to_event[cam] = None
                continue
            try:
                self.mask_raster_h_mvs_to_event[cam] = np.linalg.inv(h_event_to_mvs)
            except Exception:
                self.mask_raster_h_mvs_to_event[cam] = np.eye(3, dtype=np.float64)
                self.mask_raster_h_source[cam] = f"{h_source}:inverse_failed_identity"

    def _empty_mask_raster_meta(self, cam: str, reason: str) -> Dict[str, Any]:
        return {
            "enabled": bool(self.mask_raster_enable),
            "path": _resolve_shm_template(self.mask_raster_template, cam),
            "width": int(self.mask_raster_width),
            "height": int(self.mask_raster_height),
            "dilate_px": int(self.mask_raster_dilate_px),
            "erode_px": int(self.mask_raster_erode_px),
            "min_conf": float(self.mask_raster_min_conf),
            "seq": _safe_int(self.mask_raster_meta.get(cam, {}).get("seq"), 0),
            "mask_pixels": 0,
            "used_persons": 0,
            "used_polygons": 0,
            "projected_total_points": 0,
            "projected_in_bounds_points": 0,
            "raw_bbox_xyxy": None,
            "projected_bbox_xyxy": None,
            "projection_state": "empty",
            "homography_source": str(self.mask_raster_h_source.get(cam, "")),
            "publish_count": _safe_int(self.mask_raster_meta.get(cam, {}).get("publish_count"), 0),
            "error": str(reason or ""),
        }

    def _build_mask_raster(self, cam: str, rec: Optional[Dict[str, Any]]) -> tuple[Any, Dict[str, Any]]:
        if np is None or cv2 is None:
            raise RuntimeError("numpy/cv2 unavailable")
        mask = np.zeros((self.mask_raster_height, self.mask_raster_width), dtype=np.uint8)
        meta = self._empty_mask_raster_meta(cam, "")
        if not isinstance(rec, dict):
            meta["error"] = "no_record"
            return mask, meta

        persons = rec.get("persons") if isinstance(rec.get("persons"), list) else []
        h_mvs_to_event = self.mask_raster_h_mvs_to_event.get(cam)
        used_persons = 0
        used_polygons = 0
        projected_total_points = 0
        projected_in_bounds_points = 0
        raw_xs: List[float] = []
        raw_ys: List[float] = []
        event_xs: List[int] = []
        event_ys: List[int] = []
        for person in persons:
            if not isinstance(person, dict):
                continue
            conf = _safe_float(person.get("confidence", person.get("score", 1.0)), 1.0)
            if conf < self.mask_raster_min_conf:
                continue
            polys = _person_polygons(person)
            if not polys:
                continue
            person_used = False
            for poly in polys:
                pts: List[List[int]] = []
                for p in poly:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        raw_x = _safe_float(p[0])
                        raw_y = _safe_float(p[1])
                        raw_xs.append(raw_x)
                        raw_ys.append(raw_y)
                        q = _apply_h(h_mvs_to_event, raw_x, raw_y)
                        pts.append(q)
                        projected_total_points += 1
                        event_xs.append(int(q[0]))
                        event_ys.append(int(q[1]))
                        if 0 <= int(q[0]) < self.mask_raster_width and 0 <= int(q[1]) < self.mask_raster_height:
                            projected_in_bounds_points += 1
                if len(pts) < 3:
                    continue
                arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [arr], 255)
                used_polygons += 1
                person_used = True
            if person_used:
                used_persons += 1

        if self.mask_raster_erode_px > 0 and int(mask.max()) > 0:
            k = max(1, int(self.mask_raster_erode_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.erode(mask, kernel, iterations=1)
        if self.mask_raster_dilate_px > 0 and int(mask.max()) > 0:
            k = max(1, int(self.mask_raster_dilate_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)

        meta["mask_pixels"] = int(np.count_nonzero(mask))
        meta["used_persons"] = int(used_persons)
        meta["used_polygons"] = int(used_polygons)
        meta["projected_total_points"] = int(projected_total_points)
        meta["projected_in_bounds_points"] = int(projected_in_bounds_points)
        if raw_xs and raw_ys:
            meta["raw_bbox_xyxy"] = [round(min(raw_xs), 2), round(min(raw_ys), 2), round(max(raw_xs), 2), round(max(raw_ys), 2)]
        if event_xs and event_ys:
            meta["projected_bbox_xyxy"] = [int(min(event_xs)), int(min(event_ys)), int(max(event_xs)), int(max(event_ys))]
        if used_polygons <= 0:
            meta["projection_state"] = "no_polygon"
        elif projected_in_bounds_points <= 0 and int(meta["mask_pixels"]) <= 0:
            meta["projection_state"] = "out_of_event_frame"
        elif int(meta["mask_pixels"]) <= 0:
            meta["projection_state"] = "projected_but_empty"
        else:
            meta["projection_state"] = "ok"
        return mask, meta

    def _publish_mask_raster(self, cam: str, rec: Optional[Dict[str, Any]], now_us: int) -> Dict[str, Any]:
        if not self.mask_raster_enable:
            meta = self._empty_mask_raster_meta(cam, "disabled")
            self.mask_raster_meta[cam] = meta
            return meta
        if self.mask_raster_publisher is None:
            self.mask_raster_publisher = MaskRasterPublisher(
                self.mask_raster_template,
                self.mask_raster_width,
                self.mask_raster_height,
            )
        try:
            mask, meta = self._build_mask_raster(cam, rec)
            pub = self.mask_raster_publisher.publish(cam, mask, now_us)
            prev_count = _safe_int(self.mask_raster_meta.get(cam, {}).get("publish_count"), 0)
            meta.update(pub)
            meta["publish_count"] = prev_count + 1
            meta["publish_wall_us"] = int(now_us)
            meta["error"] = ""
        except Exception as exc:
            meta = self._empty_mask_raster_meta(cam, f"{type(exc).__name__}: {exc}")
        self.mask_raster_meta[cam] = meta
        return meta

    def _normalize_record(self, cam: str, rec: Dict[str, Any], now_us: int, mono_ns: int) -> Dict[str, Any]:
        persons = rec.get("persons") if isinstance(rec.get("persons"), list) else []
        sync_index = rec.get("sync_index_hint")
        if sync_index is None and isinstance(rec.get("mvs_meta"), dict):
            sync_index = rec.get("mvs_meta", {}).get("program_sync_index")
        capture_wall_us = _safe_int(rec.get("capture_wall_us"), 0)
        record_wall_us = _safe_int(
            rec.get("record_wall_us", rec.get("wall_time_us", rec.get("ts_wall_us", rec.get("timestamp_us")))),
            0,
        )
        age_ms = ((now_us - record_wall_us) / 1000.0) if record_wall_us > 0 else -1.0
        polygon_count = _count_polygons(persons)
        bbox_count = 0
        for p in persons:
            if isinstance(p, dict) and (p.get("bbox_xyxy") or p.get("bbox")):
                bbox_count += 1
        return {
            "schema_version": 1,
            "source": "event_mask_evidence_service",
            "camera": str(cam),
            "pid": int(os.getpid()),
            "wall_time_us": int(now_us),
            "monotonic_ns": int(mono_ns),
            "enabled": bool(self.enabled),
            "period_ms": float(self.period_ms),
            "stale_ms": float(self.stale_ms),
            "fresh": bool(age_ms < 0 or age_ms <= float(self.stale_ms)),
            "age_ms": round(float(age_ms), 3),
            "sync_index": _safe_int(sync_index, -1),
            "human_frame_id": _safe_int(rec.get("frame_id", rec.get("human_frame_id")), -1),
            "frame_w": _safe_int(rec.get("frame_w"), 0),
            "frame_h": _safe_int(rec.get("frame_h"), 0),
            "capture_wall_us": int(capture_wall_us),
            "record_wall_us": int(record_wall_us),
            "infer_source": str(rec.get("infer_source", "")),
            "person_count": int(len(persons)),
            "polygon_count": int(polygon_count),
            "bbox_count": int(bbox_count),
            "persons": persons,
            "control_seq": _safe_int(self.stats.get("control_seq"), 0),
        }

    def _tick(self) -> None:
        now_us = _now_us()
        mono_ns = _mono_ns()
        for cam, tailer in self.tailers.items():
            records = tailer.poll()
            if records:
                self.stats["records_loaded"] = _safe_int(self.stats.get("records_loaded"), 0) + len(records)
                self.latest_records[cam] = records[-1]
            rec = self.latest_records.get(cam)
            if rec is None:
                evidence = {
                    "schema_version": 1,
                    "source": "event_mask_evidence_service",
                    "camera": str(cam),
                    "pid": int(os.getpid()),
                    "wall_time_us": int(now_us),
                    "monotonic_ns": int(mono_ns),
                    "enabled": bool(self.enabled),
                    "period_ms": float(self.period_ms),
                    "stale_ms": float(self.stale_ms),
                    "fresh": False,
                    "age_ms": -1.0,
                    "sync_index": -1,
                    "person_count": 0,
                    "polygon_count": 0,
                    "bbox_count": 0,
                    "persons": [],
                    "reason": tailer.last_error or "no_record",
                    "control_seq": _safe_int(self.stats.get("control_seq"), 0),
                }
            else:
                evidence = self._normalize_record(cam, rec, now_us, mono_ns)
            if self.mask_raster_enable:
                evidence["mask_raster"] = self._publish_mask_raster(cam, rec, now_us)
            self.latest_evidence[cam] = evidence
            _json_atomic(self._evidence_path(cam), evidence)
        self.stats["ticks"] = _safe_int(self.stats.get("ticks"), 0) + 1
        self.stats["last_tick_us"] = int(now_us)
        self.stats["last_tick_mono_ns"] = int(mono_ns)
        self._write_status(now_us, mono_ns)

    def _write_status(self, now_us: int, mono_ns: int) -> None:
        per_cam = {}
        fresh = 0
        for cam in self.cams:
            ev = self.latest_evidence.get(cam, {})
            if bool(ev.get("fresh")):
                fresh += 1
            per_cam[cam] = {
                "fresh": bool(ev.get("fresh", False)),
                "age_ms": _safe_float(ev.get("age_ms"), -1.0),
                "sync_index": _safe_int(ev.get("sync_index"), -1),
                "person_count": _safe_int(ev.get("person_count"), 0),
                "polygon_count": _safe_int(ev.get("polygon_count"), 0),
                "bbox_count": _safe_int(ev.get("bbox_count"), 0),
                "infer_source": str(ev.get("infer_source", "")),
                "mask_raster": dict(self.mask_raster_meta.get(cam, self._empty_mask_raster_meta(cam, "not_published"))),
            }
        status = {
            "schema_version": 1,
            "source": "event_mask_evidence_service",
            "state": "running" if self.running else "stopping",
            "pid": int(os.getpid()),
            "wall_time_us": int(now_us),
            "monotonic_ns": int(mono_ns),
            "enabled": bool(self.enabled),
            "period_ms": float(self.period_ms),
            "effective_fps": 1000.0 / max(1e-3, float(self.period_ms)),
            "stale_ms": float(self.stale_ms),
            "control_json": str(self.control_path),
            "control_seq": _safe_int(self.stats.get("control_seq"), 0),
            "control_error": str(self.stats.get("control_error", "")),
            "camera_count": int(len(self.cams)),
            "fresh_count": int(fresh),
            "mask_raster": {
                "enabled": bool(self.mask_raster_enable),
                "template": str(self.mask_raster_template),
                "width": int(self.mask_raster_width),
                "height": int(self.mask_raster_height),
                "dilate_px": int(self.mask_raster_dilate_px),
                "erode_px": int(self.mask_raster_erode_px),
                "min_conf": float(self.mask_raster_min_conf),
            },
            "stats": dict(self.stats),
            "per_camera": per_cam,
        }
        _json_atomic(self.status_path, status)

    def run(self) -> int:
        print(
            f"[event_mask_evidence] start cams={','.join(self.cams)} period_ms={self.period_ms:.1f} "
            f"control={self.control_path} status={self.status_path}",
            flush=True,
        )
        next_tick = time.monotonic()
        while self.running:
            self._poll_control()
            if self.enabled:
                self._tick()
            else:
                self._write_status(_now_us(), _mono_ns())
            now = time.monotonic()
            next_tick = max(next_tick + self.period_ms / 1000.0, now + 0.001)
            time.sleep(max(0.001, min(0.25, next_tick - now)))
        self._write_status(_now_us(), _mono_ns())
        if self.mask_raster_publisher is not None:
            self.mask_raster_publisher.close()
        return 0

    def stop(self, *_args) -> None:
        self.running = False


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent Event human-mask evidence ticker")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--sync-root", default="sync_ipc")
    ap.add_argument("--human-jsonl-template", default="{root}/human_result_{camera}.jsonl")
    ap.add_argument("--evidence-json-template", default="{root}/event_mask_evidence_{camera}.json")
    ap.add_argument("--status-json", default="{root}/event_mask_evidence_status.json")
    ap.add_argument("--control-json", default="{root}/event_mask_control.json")
    ap.add_argument("--period-ms", type=float, default=16.0)
    ap.add_argument("--stale-ms", type=float, default=120.0)
    ap.add_argument("--tail-bytes", type=int, default=4_000_000)
    ap.add_argument("--enable", action="store_true", default=True)
    ap.add_argument("--disable", dest="enable", action="store_false")
    ap.add_argument("--mask-raster-enable", dest="mask_raster_enable", action="store_true", default=False)
    ap.add_argument("--disable-mask-raster", dest="mask_raster_enable", action="store_false")
    ap.add_argument("--mask-raster-template", default="/dev/shm/event_human_mask_raster_{camera}.mmap")
    ap.add_argument("--mask-raster-width", type=int, default=1280)
    ap.add_argument("--mask-raster-height", type=int, default=720)
    ap.add_argument("--mask-raster-dilate-px", type=int, default=8)
    ap.add_argument("--mask-raster-erode-px", type=int, default=0)
    ap.add_argument("--mask-raster-min-conf", type=float, default=0.2)
    ap.add_argument("--mask-raster-mvs-config", default="")
    args = ap.parse_args()
    svc = EventMaskEvidenceService(args)
    try:
        signal.signal(signal.SIGINT, svc.stop)
        signal.signal(signal.SIGTERM, svc.stop)
    except Exception:
        pass
    return svc.run()


if __name__ == "__main__":
    raise SystemExit(main())
