#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge multiple Global YOLO primary workers into one diagnostic stream.

The worker processes still publish legacy-compatible human_result_{camera}.jsonl
files directly, so this merger is intentionally lightweight: it keeps the
global_yolo_latest/status files coherent for diagnostics, overlays, and tests.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence


def now_us() -> int:
    return int(time.time() * 1_000_000)


def _parse_cams(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").replace(";", ",").split(",") if x.strip()]


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


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


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as fp:
            obj = json.load(fp)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _record_cam(record: Dict[str, Any]) -> str:
    return str(record.get("camera") or record.get("cam") or "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _uptime_fps(count: int, start_wall_us: int, wall_us: int) -> float:
    elapsed_s = max(1e-6, float(wall_us - int(start_wall_us or wall_us)) / 1_000_000.0)
    return float(count) / elapsed_s


def _ratio(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den or 0)
        if d <= 0.0:
            return float(default)
        return round(float(num or 0) / d, 4)
    except Exception:
        return float(default)


def _last_batch_fps(last_batch: Dict[str, Any], key: str = "total_ms") -> float:
    if not isinstance(last_batch, dict):
        return 0.0
    batch_size = _safe_int(last_batch.get("batch_size"), 0)
    ms = _safe_float(last_batch.get(key), 0.0)
    if batch_size <= 0 or ms <= 0.0:
        return 0.0
    return 1000.0 * float(batch_size) / ms


class GlobalYoloMergePublisher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.cams = _parse_cams(args.cams)
        self.latest_paths = [_resolve_project_path(self.project_root, p) for p in _split_csv(args.worker_latest_jsons)]
        self.status_paths = [_resolve_project_path(self.project_root, p) for p in _split_csv(args.worker_status_jsons)]
        self.latest_out = _resolve_project_path(self.project_root, args.latest_json)
        self.status_out = _resolve_project_path(self.project_root, args.status_json)
        self.poll_s = max(0.001, float(args.poll_ms or 20.0) / 1000.0)
        self.status_interval_s = max(0.0, float(args.status_interval_ms or 1000.0) / 1000.0)
        self.latest_interval_s = max(0.0, float(args.latest_interval_ms or 100.0) / 1000.0)
        self.stale_us = max(1, int(float(args.worker_stale_ms or 500.0) * 1000.0))
        self._last_status_write_s = 0.0
        self._last_latest_write_s = 0.0
        self._last_latest_by_worker: List[Dict[str, Any]] = [{} for _ in self.latest_paths]
        self._last_status_by_worker: List[Dict[str, Any]] = [{} for _ in self.status_paths]
        self._last_merged_records: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "start_wall_us": now_us(),
            "latest_writes": 0,
            "status_writes": 0,
            "polls": 0,
            "worker_latest_reads": 0,
            "worker_status_reads": 0,
            "errors": 0,
        }

    def _due(self, attr: str, interval_s: float) -> bool:
        now = time.time()
        last = float(getattr(self, attr, 0.0) or 0.0)
        if interval_s <= 0.0 or last <= 0.0 or now - last >= interval_s:
            setattr(self, attr, now)
            return True
        return False

    def _read_workers(self) -> None:
        for idx, path in enumerate(self.latest_paths):
            obj = _read_json(path)
            if obj:
                self._last_latest_by_worker[idx] = obj
                self.stats["worker_latest_reads"] = int(self.stats.get("worker_latest_reads", 0)) + 1
        for idx, path in enumerate(self.status_paths):
            obj = _read_json(path)
            if obj:
                self._last_status_by_worker[idx] = obj
                self.stats["worker_status_reads"] = int(self.stats.get("worker_status_reads", 0)) + 1

    def _merge_latest(self) -> Dict[str, Any]:
        wall_us = now_us()
        records_by_cam: Dict[str, Dict[str, Any]] = {}
        worker_summaries: List[Dict[str, Any]] = []
        for idx, obj in enumerate(self._last_latest_by_worker):
            obj_wall = int(obj.get("wall_us", 0) or 0) if obj else 0
            stale = not obj_wall or wall_us - obj_wall > self.stale_us
            recs = obj.get("records", []) if isinstance(obj, dict) else []
            if not isinstance(recs, list):
                recs = []
            worker_summaries.append({
                "worker": idx,
                "path": str(self.latest_paths[idx]),
                "wall_us": obj_wall,
                "age_ms": round((wall_us - obj_wall) / 1000.0, 3) if obj_wall else None,
                "stale": bool(stale),
                "batch_index": obj.get("batch_index") if isinstance(obj, dict) else None,
                "cams": obj.get("cams", []) if isinstance(obj, dict) else [],
                "records": len(recs),
            })
            if stale:
                continue
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                cam = _record_cam(rec)
                if not cam:
                    continue
                prev = records_by_cam.get(cam)
                rec_wall = int(rec.get("wall_us", obj_wall) or obj_wall)
                prev_wall = int(prev.get("wall_us", 0) or 0) if prev else -1
                if prev is None or rec_wall >= prev_wall:
                    records_by_cam[cam] = rec

        cam_order = {cam: i for i, cam in enumerate(self.cams)}
        records = sorted(records_by_cam.values(), key=lambda r: cam_order.get(_record_cam(r), 999))
        self._last_merged_records = records
        return {
            "msg_type": "global_yolo_latest",
            "wall_us": wall_us,
            "source": "global_yolo_merge_publisher",
            "shadow_mode": False,
            "global_yolo_primary": True,
            "inference_mode": "primary_merge",
            "merge_publisher": True,
            "worker_count": len(self.latest_paths),
            "cams": [cam for cam in self.cams if cam in records_by_cam],
            "records": records,
            "workers": worker_summaries,
        }

    def _merge_status(self, state: str = "running", error: str = "") -> Dict[str, Any]:
        wall_us = now_us()
        workers: List[Dict[str, Any]] = []
        frames_inferred = 0
        records_written = 0
        frames_published = 0
        errors = int(self.stats.get("errors", 0))
        worker_states: List[str] = []
        groups: List[Any] = []
        last_batches: List[Any] = []
        primary_flags: List[bool] = []
        postprocess_draw_enabled = False
        cpu_draw_postprocess = False
        mask_postprocess_enabled = False
        tensorrt_enabled = False
        postprocess_backends: List[str] = []
        for idx, status in enumerate(self._last_status_by_worker):
            status_wall = int(status.get("wall_us", 0) or 0) if status else 0
            status_stale_us = max(self.stale_us, int(max(self.status_interval_s * 3.0, 1.0) * 1_000_000))
            stale = not status_wall or wall_us - status_wall > status_stale_us
            st = str(status.get("state", "missing") if isinstance(status, dict) else "missing")
            stats = status.get("stats", {}) if isinstance(status, dict) else {}
            if not isinstance(stats, dict):
                stats = {}
            diagnostics = status.get("diagnostics", {}) if isinstance(status, dict) else {}
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            model = status.get("model", {}) if isinstance(status, dict) else {}
            if not isinstance(model, dict):
                model = {}
            primary = bool(status.get("global_yolo_primary", not bool(status.get("shadow_mode", False)))) if isinstance(status, dict) else False
            primary_flags.append(primary)
            worker_postprocess_draw_enabled = bool(
                status.get("postprocess_draw_enabled", False) if isinstance(status, dict) else False
            ) or bool(diagnostics.get("postprocess_draw_enabled", False))
            worker_cpu_draw_postprocess = bool(
                status.get("cpu_draw_postprocess", False) if isinstance(status, dict) else False
            ) or bool(diagnostics.get("cpu_draw_postprocess", False))
            worker_mask_postprocess_enabled = bool(
                status.get("mask_postprocess_enabled", False) if isinstance(status, dict) else False
            ) or bool(diagnostics.get("mask_postprocess_enabled", False))
            worker_tensorrt_enabled = bool(
                status.get("tensor_rt_enabled", False) if isinstance(status, dict) else False
            ) or bool(status.get("tensorrt_enabled", False) if isinstance(status, dict) else False) or bool(model.get("tensor_rt_enabled", False)) or bool(str(model.get("engine_path", "") or "").strip())
            postprocess_draw_enabled = postprocess_draw_enabled or worker_postprocess_draw_enabled
            cpu_draw_postprocess = cpu_draw_postprocess or worker_cpu_draw_postprocess
            mask_postprocess_enabled = mask_postprocess_enabled or worker_mask_postprocess_enabled
            tensorrt_enabled = tensorrt_enabled or worker_tensorrt_enabled
            backend = str(status.get("postprocess_backend", "") if isinstance(status, dict) else "")
            if backend:
                postprocess_backends.append(backend)
            frames_inferred += int(stats.get("frames_inferred", 0) or 0)
            records_written += int(stats.get("records_written", 0) or 0)
            errors += int(stats.get("errors", 0) or 0)
            worker_states.append(st)
            if isinstance(status.get("groups"), list):
                groups.extend(status.get("groups") or [])
            if isinstance(status.get("last_batch"), dict) and status.get("last_batch"):
                last_batches.append(status.get("last_batch"))
            worker_frames_inferred = _safe_int(stats.get("frames_inferred"), 0)
            worker_records_written = _safe_int(stats.get("records_written"), 0)
            worker_frames_published = _safe_int(stats.get("frames_published"), worker_records_written)
            worker_start_wall_us = _safe_int(stats.get("start_wall_us"), wall_us)
            worker_last_batch = status.get("last_batch", {}) if isinstance(status, dict) else {}
            worker_inferred_fps = _uptime_fps(worker_frames_inferred, worker_start_wall_us, wall_us)
            worker_published_fps = _uptime_fps(worker_frames_published, worker_start_wall_us, wall_us)
            worker_last_total_fps = _last_batch_fps(worker_last_batch, "total_ms")
            worker_last_predict_fps = _last_batch_fps(worker_last_batch, "predict_ms")
            workers.append({
                "worker": idx,
                "path": str(self.status_paths[idx]) if idx < len(self.status_paths) else "",
                "state": st,
                "stale": bool(stale),
                "age_ms": round((wall_us - status_wall) / 1000.0, 3) if status_wall else None,
                "cams": status.get("cams", []) if isinstance(status, dict) else [],
                "groups": status.get("groups", []) if isinstance(status, dict) else [],
                "global_yolo_primary": bool(primary),
                "postprocess_backend": backend,
                "postprocess_draw_enabled": bool(worker_postprocess_draw_enabled),
                "cpu_draw_postprocess": bool(worker_cpu_draw_postprocess),
                "mask_postprocess_enabled": bool(worker_mask_postprocess_enabled),
                "tensor_rt_enabled": bool(worker_tensorrt_enabled),
                "inferred_fps": round(float(worker_inferred_fps), 3),
                "published_fps": round(float(worker_published_fps), 3),
                "last_total_fps": round(float(worker_last_total_fps), 3),
                "last_predict_fps": round(float(worker_last_predict_fps), 3),
                "input_alignment_audit": status.get("input_alignment_audit", {}) if isinstance(status.get("input_alignment_audit"), dict) else {},
                "input_buffer": status.get("input_buffer", {}) if isinstance(status.get("input_buffer"), dict) else {},
                "stats": stats,
                "last_batch": worker_last_batch,
                "error": status.get("error", "") if isinstance(status, dict) else "",
            })
            frames_published += worker_frames_published
        if error:
            state = "error"
        elif not workers:
            state = "missing_workers"
        elif any(w["state"] == "error" for w in workers):
            state = "error"
        elif any(w["stale"] for w in workers):
            state = "degraded"
        start_candidates = [
            _safe_int((w.get("stats") or {}).get("start_wall_us"), 0)
            for w in workers
            if isinstance(w.get("stats"), dict)
        ]
        start_candidates = [x for x in start_candidates if x > 0]
        start_wall_us = min(start_candidates) if start_candidates else int(self.stats.get("start_wall_us", wall_us) or wall_us)
        inferred_fps_total = _uptime_fps(frames_inferred, start_wall_us, wall_us)
        published_fps_total = _uptime_fps(frames_published, start_wall_us, wall_us)
        last_total_fps_total = sum(_safe_float(w.get("last_total_fps"), 0.0) for w in workers if not bool(w.get("stale")))
        last_predict_fps_total = sum(_safe_float(w.get("last_predict_fps"), 0.0) for w in workers if not bool(w.get("stale")))
        async_enabled = any(bool((w.get("stats") or {}).get("async_record_postprocess_enabled")) for w in workers if isinstance(w.get("stats"), dict))
        queue_depth = sum(_safe_int((w.get("stats") or {}).get("postprocess_queue_depth"), 0) for w in workers if isinstance(w.get("stats"), dict))
        queue_max_depth = max([_safe_int((w.get("stats") or {}).get("postprocess_queue_max_depth"), 0) for w in workers if isinstance(w.get("stats"), dict)] or [0])
        queue_lag_ms = max([_safe_float((w.get("stats") or {}).get("postprocess_lag_ms_last"), 0.0) for w in workers if isinstance(w.get("stats"), dict)] or [0.0])
        backpressure_ms = sum(_safe_float((w.get("stats") or {}).get("postprocess_backpressure_ms_total"), 0.0) for w in workers if isinstance(w.get("stats"), dict))
        dropped_batches = sum(_safe_int((w.get("stats") or {}).get("postprocess_dropped_batches"), 0) for w in workers if isinstance(w.get("stats"), dict))
        record_polygon_ratios = [
            _safe_float((w.get("stats") or {}).get("published_records_with_polygon_ratio"), 0.0)
            for w in workers
            if isinstance(w.get("stats"), dict) and (w.get("stats") or {}).get("published_records_with_persons") is not None
        ]
        record_polygon_ratio = min(record_polygon_ratios) if record_polygon_ratios else None
        polygon_ratios = [
            _safe_float((w.get("stats") or {}).get("published_persons_with_polygon_ratio"), 0.0)
            for w in workers
            if isinstance(w.get("stats"), dict) and (w.get("stats") or {}).get("published_persons_total") is not None
        ]
        polygon_ratio = min(polygon_ratios) if polygon_ratios else None
        polygon_contract_ok = bool(dropped_batches == 0 and (polygon_ratio is None or polygon_ratio >= 0.90))
        align_stats = [w.get("stats") or {} for w in workers if isinstance(w.get("stats"), dict)]
        align_batches = sum(_safe_int(s.get("input_alignment_batches"), 0) for s in align_stats)
        align_frames = sum(_safe_int(s.get("input_alignment_frames"), 0) for s in align_stats)
        align_full = sum(_safe_int(s.get("input_alignment_full_batches"), 0) for s in align_stats)
        align_partial = sum(_safe_int(s.get("input_alignment_partial_batches"), 0) for s in align_stats)
        align_same_sync = sum(_safe_int(s.get("input_alignment_same_sync_batches"), 0) for s in align_stats)
        align_same_sync_full = sum(_safe_int(s.get("input_alignment_same_sync_full_batches"), 0) for s in align_stats)
        align_visual_complete = sum(_safe_int(s.get("input_alignment_visual_mid_complete_batches"), 0) for s in align_stats)
        align_visual_aligned = sum(_safe_int(s.get("input_alignment_visual_mid_aligned_batches"), 0) for s in align_stats)
        align_current_full = sum(_safe_int(s.get("input_alignment_current_aligned_full_batches"), 0) for s in align_stats)
        align_meta_match = sum(_safe_int(s.get("input_alignment_meta_frame_match_batches"), 0) for s in align_stats)
        align_frames_with_sync = sum(_safe_int(s.get("input_alignment_frames_with_sync"), 0) for s in align_stats)
        align_frames_with_visual_mid = sum(_safe_int(s.get("input_alignment_frames_with_visual_mid"), 0) for s in align_stats)
        align_full_sets = sum(_safe_int(s.get("input_alignment_sync_full_sets_seen"), 0) for s in align_stats)
        align_full_sets_within_wait = sum(_safe_int(s.get("input_alignment_sync_full_sets_within_wait"), 0) for s in align_stats)
        align_full_sets_visual = sum(_safe_int(s.get("input_alignment_sync_full_sets_visual_aligned"), 0) for s in align_stats)
        align_wait_total = sum(_safe_float(s.get("input_alignment_sync_full_wait_ms_total"), 0.0) for s in align_stats)
        input_alignment_enabled = any(bool((w.get("input_alignment_audit") or {}).get("enabled")) for w in workers if isinstance(w.get("input_alignment_audit"), dict))
        input_buffer_enabled = any(bool((w.get("input_buffer") or {}).get("enabled")) for w in workers if isinstance(w.get("input_buffer"), dict))
        input_buffer_batches = sum(_safe_int((w.get("stats") or {}).get("input_buffer_batches"), 0) for w in workers if isinstance(w.get("stats"), dict))
        input_buffer_aligned = sum(_safe_int((w.get("stats") or {}).get("input_buffer_aligned_full_batches"), 0) for w in workers if isinstance(w.get("stats"), dict))
        input_buffer_fallback = sum(_safe_int((w.get("stats") or {}).get("input_buffer_fallback_batches"), 0) for w in workers if isinstance(w.get("stats"), dict))
        input_buffer_pending = sum(_safe_int((w.get("input_buffer") or {}).get("pending_frames"), 0) for w in workers if isinstance(w.get("input_buffer"), dict))
        input_alignment_audit = {
            "enabled": bool(input_alignment_enabled),
            "mode": "active_buffered" if input_buffer_enabled else "audit_only",
            "batches": int(align_batches),
            "frames": int(align_frames),
            "full_batch_ratio": _ratio(align_full, align_batches),
            "partial_batch_ratio": _ratio(align_partial, align_batches),
            "same_sync_batch_ratio": _ratio(align_same_sync, align_batches),
            "same_sync_full_batch_ratio": _ratio(align_same_sync_full, align_batches),
            "visual_mid_complete_batch_ratio": _ratio(align_visual_complete, align_batches),
            "visual_mid_aligned_batch_ratio": _ratio(align_visual_aligned, align_batches),
            "current_aligned_full_batch_ratio": _ratio(align_current_full, align_batches),
            "meta_frame_match_batch_ratio": _ratio(align_meta_match, align_batches),
            "frames_with_sync_ratio": _ratio(align_frames_with_sync, align_frames),
            "frames_with_visual_mid_ratio": _ratio(align_frames_with_visual_mid, align_frames),
            "sync_spread_max": max([_safe_int(s.get("input_alignment_sync_spread_max"), 0) for s in align_stats] or [0]),
            "visual_mid_spread_ms_max": round(max([_safe_float(s.get("input_alignment_visual_mid_spread_ms_max"), 0.0) for s in align_stats] or [0.0]), 3),
            "visual_mid_spread_ms_p95": round(max([_safe_float((w.get("input_alignment_audit") or {}).get("visual_mid_spread_ms_p95"), 0.0) for w in workers if isinstance(w.get("input_alignment_audit"), dict)] or [0.0]), 3),
            "capture_ts_spread_ms_max": round(max([_safe_float(s.get("input_alignment_capture_ts_spread_ms_max"), 0.0) for s in align_stats] or [0.0]), 3),
            "capture_ts_spread_ms_p95": round(max([_safe_float((w.get("input_alignment_audit") or {}).get("capture_ts_spread_ms_p95"), 0.0) for w in workers if isinstance(w.get("input_alignment_audit"), dict)] or [0.0]), 3),
            "exposure_spread_us_max": max([_safe_int(s.get("input_alignment_exposure_spread_us_max"), 0) for s in align_stats] or [0]),
            "exposure_spread_us_p95": max([_safe_int((w.get("input_alignment_audit") or {}).get("exposure_spread_us_p95"), 0) for w in workers if isinstance(w.get("input_alignment_audit"), dict)] or [0]),
            "exposure_end_spread_ms_p95": round(max([_safe_float((w.get("input_alignment_audit") or {}).get("exposure_end_spread_ms_p95"), 0.0) for w in workers if isinstance(w.get("input_alignment_audit"), dict)] or [0.0]), 3),
            "sync_full_sets_seen": int(align_full_sets),
            "sync_full_sets_within_wait": int(align_full_sets_within_wait),
            "sync_full_sets_within_wait_ratio": _ratio(align_full_sets_within_wait, align_full_sets),
            "sync_full_sets_visual_aligned": int(align_full_sets_visual),
            "sync_full_sets_visual_aligned_ratio": _ratio(align_full_sets_visual, align_full_sets),
            "sync_full_wait_ms_avg": round(align_wait_total / max(1, align_full_sets), 3),
            "sync_full_wait_ms_max": round(max([_safe_float(s.get("input_alignment_sync_full_wait_ms_max"), 0.0) for s in align_stats] or [0.0]), 3),
            "sync_full_visual_spread_ms_max": round(max([_safe_float(s.get("input_alignment_sync_full_visual_spread_ms_max"), 0.0) for s in align_stats] or [0.0]), 3),
            "workers": [
                {
                    "worker": w.get("worker"),
                    "cams": w.get("cams"),
                    "audit": w.get("input_alignment_audit", {}),
                    "input_buffer": w.get("input_buffer", {}),
                }
                for w in workers
            ],
        }
        input_buffer = {
            "enabled": bool(input_buffer_enabled),
            "worker_count": len([w for w in workers if bool((w.get("input_buffer") or {}).get("enabled"))]),
            "batches": int(input_buffer_batches),
            "aligned_full_batches": int(input_buffer_aligned),
            "fallback_batches": int(input_buffer_fallback),
            "aligned_full_ratio": _ratio(input_buffer_aligned, input_buffer_batches),
            "fallback_ratio": _ratio(input_buffer_fallback, input_buffer_batches),
            "pending_frames": int(input_buffer_pending),
            "workers": [
                {
                    "worker": w.get("worker"),
                    "cams": w.get("cams"),
                    "input_buffer": w.get("input_buffer", {}),
                }
                for w in workers
            ],
        }
        summary = {
            "inferred_fps_total": round(float(inferred_fps_total), 3),
            "published_fps_total": round(float(published_fps_total), 3),
            "last_total_fps_total": round(float(last_total_fps_total), 3),
            "last_predict_fps_total": round(float(last_predict_fps_total), 3),
            "worker_count": len(workers),
            "stale_worker_count": sum(1 for w in workers if bool(w.get("stale"))),
            "error_count": int(errors),
            "async_record_postprocess_enabled": bool(async_enabled),
            "postprocess_queue_depth": int(queue_depth),
            "postprocess_queue_max_depth": int(queue_max_depth),
            "postprocess_dropped_batches": int(dropped_batches),
            "polygon_contract_ok": bool(polygon_contract_ok),
            "published_records_with_polygon_ratio": None if record_polygon_ratio is None else round(float(record_polygon_ratio), 4),
            "published_persons_with_polygon_ratio": None if polygon_ratio is None else round(float(polygon_ratio), 4),
            "input_alignment_audit_enabled": bool(input_alignment_enabled),
            "input_alignment_full_batch_ratio": input_alignment_audit.get("full_batch_ratio"),
            "input_alignment_same_sync_batch_ratio": input_alignment_audit.get("same_sync_batch_ratio"),
            "input_alignment_current_aligned_full_batch_ratio": input_alignment_audit.get("current_aligned_full_batch_ratio"),
            "input_alignment_sync_full_sets_within_wait_ratio": input_alignment_audit.get("sync_full_sets_within_wait_ratio"),
            "input_buffer_enabled": bool(input_buffer_enabled),
            "input_buffer_aligned_full_ratio": input_buffer.get("aligned_full_ratio"),
            "input_buffer_fallback_ratio": input_buffer.get("fallback_ratio"),
        }
        status_out = {
            "msg_type": "global_yolo_status",
            "state": state,
            "pid": os.getpid(),
            "wall_us": wall_us,
            "source": "global_yolo_merge_publisher",
            "shadow_mode": False,
            "global_yolo_primary": bool(primary_flags and all(primary_flags)),
            "inference_mode": "primary_merge",
            "legacy_yolo_skipped": bool(primary_flags and all(primary_flags)),
            "merge_publisher": True,
            "worker_count": len(workers),
            "worker_states": worker_states,
            "cams": self.cams,
            "groups": groups,
            "postprocess_backends": sorted(set(postprocess_backends)),
            "postprocess_draw_enabled": bool(postprocess_draw_enabled),
            "cpu_draw_postprocess": bool(cpu_draw_postprocess),
            "mask_postprocess_enabled": bool(mask_postprocess_enabled),
            "tensor_rt_enabled": bool(tensorrt_enabled),
            "tensorrt_enabled": bool(tensorrt_enabled),
            "fps": round(float(published_fps_total), 3),
            "inferred_fps_total": round(float(inferred_fps_total), 3),
            "published_fps_total": round(float(published_fps_total), 3),
            "last_total_fps_total": round(float(last_total_fps_total), 3),
            "async_record_postprocess_enabled": bool(async_enabled),
            "postprocess_queue_depth": int(queue_depth),
            "postprocess_lag_ms": round(float(queue_lag_ms), 3),
            "postprocess_backpressure_ms": round(float(backpressure_ms), 3),
            "postprocess_dropped_batches": int(dropped_batches),
            "polygon_contract_ok": bool(polygon_contract_ok),
            "published_records_with_polygon_ratio": None if record_polygon_ratio is None else round(float(record_polygon_ratio), 4),
            "published_persons_with_polygon_ratio": None if polygon_ratio is None else round(float(polygon_ratio), 4),
            "input_alignment_audit": input_alignment_audit,
            "input_buffer": input_buffer,
            "summary": summary,
            "workers": workers,
            "stats": {
                **self.stats,
                "frames_inferred": frames_inferred,
                "records_written": records_written,
                "frames_published": frames_published,
                "errors": errors,
                "merged_records": len(self._last_merged_records),
            },
            "last_batch": last_batches[-1] if last_batches else {},
            "last_batches": last_batches,
            "diagnostics": {
                "poll_ms": float(self.args.poll_ms),
                "latest_interval_ms": float(self.args.latest_interval_ms),
                "status_interval_ms": float(self.args.status_interval_ms),
                "worker_stale_ms": float(self.args.worker_stale_ms),
                "worker_latest_jsons": [str(p) for p in self.latest_paths],
                "worker_status_jsons": [str(p) for p in self.status_paths],
                "postprocess_draw_enabled": bool(postprocess_draw_enabled),
                "cpu_draw_postprocess": bool(cpu_draw_postprocess),
                "mask_postprocess_enabled": bool(mask_postprocess_enabled),
                "tensor_rt_enabled": bool(tensorrt_enabled),
            },
            "error": error,
        }
        return status_out

    def run(self) -> int:
        try:
            _atomic_write_json(self.status_out, self._merge_status(state="starting"))
            while True:
                self.stats["polls"] = int(self.stats.get("polls", 0)) + 1
                self._read_workers()
                if self._due("_last_latest_write_s", self.latest_interval_s):
                    _atomic_write_json(self.latest_out, self._merge_latest())
                    self.stats["latest_writes"] = int(self.stats.get("latest_writes", 0)) + 1
                if self._due("_last_status_write_s", self.status_interval_s):
                    _atomic_write_json(self.status_out, self._merge_status())
                    self.stats["status_writes"] = int(self.stats.get("status_writes", 0)) + 1
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            _atomic_write_json(self.status_out, self._merge_status(state="stopped"))
            return 0
        except Exception as exc:
            self.stats["errors"] = int(self.stats.get("errors", 0)) + 1
            err = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(self.status_out, self._merge_status(state="error", error=err))
            traceback.print_exc()
            return 1


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Merge multiple Global YOLO worker status/latest files")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cams", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--worker-latest-jsons", required=True)
    ap.add_argument("--worker-status-jsons", required=True)
    ap.add_argument("--latest-json", default="sync_ipc/global_yolo_latest.json")
    ap.add_argument("--status-json", default="sync_ipc/global_yolo_status.json")
    ap.add_argument("--poll-ms", type=float, default=20.0)
    ap.add_argument("--latest-interval-ms", type=float, default=100.0)
    ap.add_argument("--status-interval-ms", type=float, default=1000.0)
    ap.add_argument("--worker-stale-ms", type=float, default=500.0)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    return GlobalYoloMergePublisher(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
