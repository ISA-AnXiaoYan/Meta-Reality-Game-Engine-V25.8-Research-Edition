#!/usr/bin/env python3
"""Analyze why event-side human masks lag the current MVS/event sync.

The script is read-only. It correlates:
  - human_result_{camera}.jsonl records
  - event_human_mask_stats_{camera}.jsonl records
  - event_worker_{camera}_status.json snapshots
  - global_yolo_worker_*_perf.csv
  - optional global_trigger_timeline.jsonl

It intentionally does not require or change the event filter policy.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HUMAN_RE = re.compile(r"human_result_([^/\\]+)\.jsonl$", re.IGNORECASE)
MASK_STATS_RE = re.compile(r"event_human_mask_stats_([^/\\]+)\.jsonl$", re.IGNORECASE)
WORKER_STATUS_RE = re.compile(r"event_worker_([^/\\]+)_status\.json$", re.IGNORECASE)
YOLO_PERF_RE = re.compile(r"global_yolo_worker_([0-9]+)_perf\.csv$", re.IGNORECASE)


def is_num(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def pct(values: Iterable[Any], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if is_num(v))
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 3)
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(vals[lo], 3)
    frac = pos - lo
    return round(vals[lo] * (1.0 - frac) + vals[hi] * frac, 3)


def avg(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if is_num(v)]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def load_json_line(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def iter_files(paths: List[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            yield root
        elif root.is_dir():
            for child in root.rglob("*"):
                if child.is_file():
                    yield child


def discover_run_window(paths: List[Path]) -> Tuple[Optional[int], Optional[int], List[str]]:
    starts: List[int] = []
    ends: List[int] = []
    sources: List[str] = []
    candidates: List[Path] = []
    for root in paths:
        candidates.append(root if root.is_dir() else root.parent)
        if root.is_dir():
            for child in root.rglob("start_epoch.txt"):
                candidates.append(child.parent)
    seen: set[str] = set()
    for folder in candidates:
        key = str(folder.resolve()) if folder.exists() else str(folder)
        if key in seen:
            continue
        seen.add(key)
        start_path = folder / "start_epoch.txt"
        end_path = folder / "end_epoch.txt"
        if not start_path.exists():
            continue
        try:
            start_s = int(float(start_path.read_text(encoding="utf-8-sig").strip()))
        except Exception:
            continue
        end_s = 0
        if end_path.exists():
            try:
                end_s = int(float(end_path.read_text(encoding="utf-8-sig").strip()))
            except Exception:
                end_s = 0
        starts.append(start_s * 1_000_000)
        if end_s > 0:
            ends.append(end_s * 1_000_000)
        sources.append(str(folder))
    if not starts:
        return None, None, []
    return min(starts), (max(ends) if ends else None), sources


def in_time_window(wall_us: int, start_us: Optional[int], end_us: Optional[int], margin_us: int) -> bool:
    if wall_us <= 0:
        return True
    if start_us is not None and wall_us < start_us - margin_us:
        return False
    if end_us is not None and wall_us > end_us + margin_us:
        return False
    return True


def sync_index_from_record(rec: Dict[str, Any]) -> Optional[int]:
    for key in ("sync_index_hint", "sync_index", "mvs_frame_num", "frame_id_local"):
        if key in rec and rec.get(key) is not None:
            try:
                return int(float(rec.get(key)))
            except Exception:
                pass
    meta = rec.get("mvs_meta") if isinstance(rec.get("mvs_meta"), dict) else {}
    for key in ("program_sync_index", "sync_index", "mvs_sync_index"):
        if key in meta and meta.get(key) is not None:
            try:
                return int(float(meta.get(key)))
            except Exception:
                pass
    return None


def wall_us_from_record(rec: Dict[str, Any], *keys: str) -> int:
    meta = rec.get("mvs_meta") if isinstance(rec.get("mvs_meta"), dict) else {}
    for key in keys:
        for src in (rec, meta):
            value = src.get(key) if isinstance(src, dict) else None
            if value is not None:
                try:
                    return int(float(value))
                except Exception:
                    pass
    return 0


def record_wall_for_window(rec: Dict[str, Any]) -> int:
    return wall_us_from_record(
        rec,
        "record_wall_us",
        "publish_wall_us",
        "receiver_wall_us",
        "wall_us",
        "ts_wall_us",
        "wall_time_us",
        "last_jsonl_poll_wall_us",
        "capture_wall_us",
        "timestamp_us",
        "host_read_wall_us",
        "soft_trigger_cmd_done_wall_us",
        "soft_trigger_cmd_wall_us",
    )


def timeline_index_and_wall(rec: Dict[str, Any]) -> Tuple[Optional[int], int]:
    idx = None
    for key in ("program_sync_index", "sync_index", "trigger_seq", "tick_id", "frame_id"):
        if rec.get(key) is not None:
            try:
                idx = int(float(rec.get(key)))
                break
            except Exception:
                pass
    wall = wall_us_from_record(
        rec,
        "soft_trigger_cmd_done_wall_us",
        "trigger_wall_us",
        "wall_time_us",
        "wall_us",
        "ts_wall_us",
        "record_wall_us",
    )
    return idx, wall


def reason_family(reason: Any) -> str:
    text = str(reason or "")
    return text.split(":", 1)[0] if text else ""


def summarize_human_records(records: List[Dict[str, Any]], trigger_wall_by_sync: Dict[int, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(records)}
    if not records:
        return out
    capture_to_record = []
    trigger_to_capture = []
    trigger_to_record = []
    yolo_predict = []
    yolo_total = []
    yolo_collect = []
    batch_sizes = []
    syncs = []
    person_records = 0
    real_records = 0
    carry_records = 0
    no_person_records = 0
    polygon_records = 0
    polygon_persons = 0
    polygon_points = []
    for rec in records:
        sync_i = sync_index_from_record(rec)
        if sync_i is not None:
            syncs.append(sync_i)
        capture_wall = wall_us_from_record(rec, "capture_wall_us", "timestamp_us", "host_read_wall_us")
        record_wall = wall_us_from_record(rec, "record_wall_us", "publish_wall_us", "receiver_wall_us")
        if capture_wall > 0 and record_wall > 0:
            capture_to_record.append((record_wall - capture_wall) / 1000.0)
        if sync_i is not None and sync_i in trigger_wall_by_sync:
            trig_wall = trigger_wall_by_sync[sync_i]
            if capture_wall > 0:
                trigger_to_capture.append((capture_wall - trig_wall) / 1000.0)
            if record_wall > 0:
                trigger_to_record.append((record_wall - trig_wall) / 1000.0)
        yolo_predict.append(rec.get("yolo_predict_ms"))
        yolo_total.append(rec.get("yolo_total_batch_ms"))
        yolo_collect.append(rec.get("yolo_collect_wait_ms"))
        batch_sizes.append(rec.get("yolo_batch_size"))
        persons = rec.get("persons") if isinstance(rec.get("persons"), list) else []
        if persons:
            person_records += 1
        else:
            no_person_records += 1
        rec_has_polygon = False
        for p in persons:
            if not isinstance(p, dict):
                continue
            polys = p.get("mask_polygons")
            if not isinstance(polys, list) or not polys:
                continue
            rec_has_polygon = True
            polygon_persons += 1
            for poly in polys:
                if isinstance(poly, list):
                    polygon_points.append(len(poly))
        if rec_has_polygon:
            polygon_records += 1
        source = str(rec.get("infer_source", rec.get("human_infer_source", "")) or "").lower()
        if source == "carry_last":
            carry_records += 1
        else:
            real_records += 1
    out.update({
        "sync_min": min(syncs) if syncs else None,
        "sync_max": max(syncs) if syncs else None,
        "sync_records": len(syncs),
        "person_records": person_records,
        "no_person_records": no_person_records,
        "polygon_records": polygon_records,
        "polygon_persons": polygon_persons,
        "polygon_points_avg": avg(polygon_points),
        "polygon_points_p50": pct(polygon_points, 0.50),
        "real_yolo_records": real_records,
        "carry_last_records": carry_records,
        "capture_to_record_ms_avg": avg(capture_to_record),
        "capture_to_record_ms_p50": pct(capture_to_record, 0.50),
        "capture_to_record_ms_p90": pct(capture_to_record, 0.90),
        "capture_to_record_ms_p95": pct(capture_to_record, 0.95),
        "trigger_to_capture_ms_p50": pct(trigger_to_capture, 0.50),
        "trigger_to_record_ms_p50": pct(trigger_to_record, 0.50),
        "trigger_to_record_ms_p90": pct(trigger_to_record, 0.90),
        "yolo_predict_ms_p50": pct(yolo_predict, 0.50),
        "yolo_total_batch_ms_p50": pct(yolo_total, 0.50),
        "yolo_collect_wait_ms_avg": avg(yolo_collect),
        "avg_batch_size": avg(batch_sizes),
    })
    return out


def summarize_mask_stats(records: List[Dict[str, Any]], trigger_period_ms: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(records)}
    if not records:
        return out
    deltas = []
    lag_frames = []
    lag_ms = []
    buffer_ages = []
    n_in = []
    n_drop = []
    applied = 0
    missing = 0
    empty = 0
    reason_counts: collections.Counter[str] = collections.Counter()
    release_counts: collections.Counter[str] = collections.Counter()
    for rec in records:
        if rec.get("applied") is True:
            applied += 1
        reason = str(rec.get("reason", "") or "")
        reason_counts[reason_family(reason)] += 1
        if reason.startswith("current_mask_missing"):
            missing += 1
        if reason.startswith("empty_events"):
            empty += 1
        delta = rec.get("current_to_latest_record_sync_delta")
        if is_num(delta):
            d = float(delta)
            deltas.append(d)
            lag = max(0.0, -d)
            lag_frames.append(lag)
            lag_ms.append(lag * trigger_period_ms)
        if is_num(rec.get("buffer_slice_age_ms")):
            buffer_ages.append(float(rec.get("buffer_slice_age_ms")))
        n_in.append(rec.get("n_in"))
        n_drop.append(rec.get("n_drop"))
        rr = str(rec.get("buffer_release_reason", "") or "")
        if rr:
            release_counts[rr] += 1
    total_in = sum(inum(v) for v in n_in)
    total_drop = sum(inum(v) for v in n_drop)
    out.update({
        "applied_ratio": round(applied / max(1, len(records)), 6),
        "current_mask_missing_ratio": round(missing / max(1, len(records)), 6),
        "empty_events_ratio": round(empty / max(1, len(records)), 6),
        "event_drop_ratio": round(total_drop / max(1, total_in), 6),
        "total_in": total_in,
        "total_drop": total_drop,
        "sync_delta_latest_minus_current_p50": pct(deltas, 0.50),
        "sync_lag_frames_p50": pct(lag_frames, 0.50),
        "sync_lag_frames_p90": pct(lag_frames, 0.90),
        "sync_lag_frames_p95": pct(lag_frames, 0.95),
        "sync_lag_ms_p50": pct(lag_ms, 0.50),
        "sync_lag_ms_p90": pct(lag_ms, 0.90),
        "sync_lag_ms_p95": pct(lag_ms, 0.95),
        "buffer_slice_age_ms_p50": pct(buffer_ages, 0.50),
        "buffer_slice_age_ms_p95": pct(buffer_ages, 0.95),
        "reason_family_counts": dict(reason_counts.most_common()),
        "buffer_release_counts": dict(release_counts.most_common()),
    })
    return out


def summarize_worker_status(status: Optional[Dict[str, Any]], trigger_period_ms: float) -> Dict[str, Any]:
    if not status:
        return {}
    hm = status.get("human_mask", {}) if isinstance(status.get("human_mask"), dict) else {}
    delta = hm.get("current_to_latest_record_sync_delta")
    lag_frames = max(0.0, -fnum(delta, 0.0)) if is_num(delta) else None
    return {
        "state": status.get("state"),
        "active_reason": status.get("active_reason"),
        "loop_fps": status.get("loop_fps"),
        "human_mask_applied": hm.get("applied"),
        "human_mask_reason": hm.get("reason"),
        "current_sync_index": hm.get("current_sync_index"),
        "latest_record_sync": hm.get("latest_record_sync"),
        "sync_delta_latest_minus_current": delta,
        "sync_lag_frames": round(lag_frames, 3) if lag_frames is not None else None,
        "sync_lag_ms": round(lag_frames * trigger_period_ms, 3) if lag_frames is not None else None,
        "event_sync_lock_state": hm.get("event_sync_lock_state"),
        "buffer_release_reason": hm.get("buffer_release_reason"),
        "buffer_slice_age_ms": hm.get("buffer_slice_age_ms"),
        "jsonl_poll_count": hm.get("jsonl_poll_count"),
        "jsonl_records_loaded": hm.get("jsonl_records_loaded"),
    }


def summarize_yolo_perf(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": len(rows)}
    if not rows:
        return out
    wall = [inum(r.get("wall_us")) for r in rows if inum(r.get("wall_us")) > 0]
    frames = sum(inum(r.get("batch_size")) for r in rows)
    dur = (max(wall) - min(wall)) / 1_000_000.0 if len(wall) > 1 else 0.0
    batch_sizes = [fnum(r.get("batch_size")) for r in rows]
    out.update({
        "frames": frames,
        "duration_s": round(dur, 3),
        "wall_fps": round(frames / dur, 3) if dur > 0 else None,
        "avg_batch_size": avg(batch_sizes),
        "partial_batch_ratio": round(sum(1 for v in batch_sizes if v < 4) / max(1, len(batch_sizes)), 6),
        "collect_wait_ms_avg": avg(r.get("collect_wait_ms") for r in rows),
        "collect_wait_ms_p90": pct((r.get("collect_wait_ms") for r in rows), 0.90),
        "predict_ms_p50": pct((r.get("predict_ms") for r in rows), 0.50),
        "predict_ms_p90": pct((r.get("predict_ms") for r in rows), 0.90),
        "postprocess_ms_p50": pct((r.get("postprocess_ms") for r in rows), 0.50),
        "total_ms_p50": pct((r.get("total_ms") for r in rows), 0.50),
        "total_ms_p90": pct((r.get("total_ms") for r in rows), 0.90),
    })
    return out


def load_inputs(paths: List[Path], start_us: Optional[int], end_us: Optional[int], margin_us: int) -> Dict[str, Any]:
    human: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    mask_stats: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    worker_status: Dict[str, Dict[str, Any]] = {}
    yolo_perf: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    trigger_wall_by_sync: Dict[int, int] = {}
    trigger_period_ms = 25.0
    sources: collections.Counter[str] = collections.Counter()

    for path in iter_files(paths):
        name = str(path)
        base = path.name
        m = HUMAN_RE.search(name)
        if m:
            cam = m.group(1)
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
                    for raw in fh:
                        rec = load_json_line(raw)
                        if rec is not None and in_time_window(record_wall_for_window(rec), start_us, end_us, margin_us):
                            human[cam].append(rec)
                            sources[str(path)] += 1
            except Exception:
                pass
            continue
        m = MASK_STATS_RE.search(name)
        if m:
            cam = m.group(1)
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
                    for raw in fh:
                        rec = load_json_line(raw)
                        if rec is not None and in_time_window(record_wall_for_window(rec), start_us, end_us, margin_us):
                            mask_stats[cam].append(rec)
                            sources[str(path)] += 1
            except Exception:
                pass
            continue
        m = WORKER_STATUS_RE.search(name)
        if m:
            cam = m.group(1)
            try:
                rec = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
                if isinstance(rec, dict):
                    old = worker_status.get(cam)
                    if old is None or inum(rec.get("wall_time_us")) >= inum(old.get("wall_time_us")):
                        worker_status[cam] = rec
                        sources[str(path)] += 1
            except Exception:
                pass
            continue
        m = YOLO_PERF_RE.search(name)
        if m:
            wid = m.group(1)
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    for row in csv.DictReader(fh):
                        if in_time_window(inum(row.get("wall_us")), start_us, end_us, margin_us):
                            yolo_perf[wid].append(row)
                            sources[str(path)] += 1
            except Exception:
                pass
            continue
        if base == "global_trigger_timeline.jsonl":
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
                    for raw in fh:
                        rec = load_json_line(raw)
                        if rec is None:
                            continue
                        idx, wall = timeline_index_and_wall(rec)
                        if idx is not None and wall > 0 and in_time_window(wall, start_us, end_us, margin_us):
                            trigger_wall_by_sync[int(idx)] = int(wall)
                            sources[str(path)] += 1
            except Exception:
                pass
            continue
        if base == "trigger_status.json":
            try:
                rec = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
                if isinstance(rec, dict):
                    if is_num(rec.get("period_ms")):
                        trigger_period_ms = fnum(rec.get("period_ms"), trigger_period_ms)
                    elif is_num(rec.get("trigger_fps")) and fnum(rec.get("trigger_fps")) > 0:
                        trigger_period_ms = 1000.0 / fnum(rec.get("trigger_fps"))
                    sources[str(path)] += 1
            except Exception:
                pass

    return {
        "human": dict(human),
        "mask_stats": dict(mask_stats),
        "worker_status": worker_status,
        "yolo_perf": dict(yolo_perf),
        "trigger_wall_by_sync": trigger_wall_by_sync,
        "trigger_period_ms": trigger_period_ms,
        "sources": dict(sources),
    }


def build_summary(inputs: List[Path], start_us: Optional[int] = None, end_us: Optional[int] = None,
                  time_margin_sec: float = 2.0) -> Dict[str, Any]:
    auto_start, auto_end, window_sources = discover_run_window(inputs)
    if start_us is None:
        start_us = auto_start
    if end_us is None:
        end_us = auto_end
    margin_us = max(0, int(float(time_margin_sec) * 1_000_000))
    data = load_inputs(inputs, start_us, end_us, margin_us)
    trigger_period_ms = fnum(data.get("trigger_period_ms"), 25.0)
    trigger_wall = data.get("trigger_wall_by_sync", {})
    cams = sorted(set(data["human"]) | set(data["mask_stats"]) | set(data["worker_status"]))
    per_camera: Dict[str, Any] = {}
    all_human: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    for cam in cams:
        human_records = data["human"].get(cam, [])
        stats_records = data["mask_stats"].get(cam, [])
        all_human.extend(human_records)
        all_stats.extend(stats_records)
        per_camera[cam] = {
            "human_result": summarize_human_records(human_records, trigger_wall),
            "event_mask_stats": summarize_mask_stats(stats_records, trigger_period_ms),
            "latest_worker_status": summarize_worker_status(data["worker_status"].get(cam), trigger_period_ms),
        }
    yolo = {wid: summarize_yolo_perf(rows) for wid, rows in sorted(data["yolo_perf"].items())}
    combined_frames = sum(item.get("frames", 0) or 0 for item in yolo.values())
    combined_dur = max((item.get("duration_s", 0) or 0 for item in yolo.values()), default=0)
    root_causes = infer_causes(per_camera, yolo, trigger_period_ms)
    return {
        "inputs": [str(p) for p in inputs],
        "run_window": {
            "start_us": start_us,
            "end_us": end_us,
            "duration_s": round((end_us - start_us) / 1_000_000.0, 3) if start_us and end_us and end_us >= start_us else None,
            "margin_s": round(margin_us / 1_000_000.0, 3),
            "sources": window_sources,
        },
        "trigger_period_ms": round(trigger_period_ms, 3),
        "source_file_count": len(data.get("sources", {})),
        "source_records": data.get("sources", {}),
        "overall": {
            "human_result": summarize_human_records(all_human, trigger_wall),
            "event_mask_stats": summarize_mask_stats(all_stats, trigger_period_ms),
            "combined_yolo_wall_fps": round(combined_frames / combined_dur, 3) if combined_dur > 0 else None,
        },
        "per_camera": per_camera,
        "global_yolo_workers": yolo,
        "analysis": root_causes,
    }


def infer_causes(per_camera: Dict[str, Any], yolo: Dict[str, Any], trigger_period_ms: float) -> Dict[str, Any]:
    lags = []
    applied_ratios = []
    missing_ratios = []
    capture_to_record = []
    for item in per_camera.values():
        ms = item.get("event_mask_stats", {})
        hr = item.get("human_result", {})
        if is_num(ms.get("sync_lag_frames_p50")):
            lags.append(float(ms.get("sync_lag_frames_p50")))
        if is_num(ms.get("applied_ratio")):
            applied_ratios.append(float(ms.get("applied_ratio")))
        if is_num(ms.get("current_mask_missing_ratio")):
            missing_ratios.append(float(ms.get("current_mask_missing_ratio")))
        if is_num(hr.get("capture_to_record_ms_p50")):
            capture_to_record.append(float(hr.get("capture_to_record_ms_p50")))
    yolo_fps = sum((item.get("wall_fps", 0) or 0) for item in yolo.values())
    partial = avg(item.get("partial_batch_ratio") for item in yolo.values())
    collect = avg(item.get("collect_wait_ms_avg") for item in yolo.values())
    lag_p50_frames = pct(lags, 0.50)
    lag_p50_ms = round(lag_p50_frames * trigger_period_ms, 3) if lag_p50_frames is not None else None
    findings = []
    if lag_p50_frames is not None and lag_p50_frames >= 5:
        findings.append("event_worker_current_sync_runs_ahead_of_latest_human_result")
    if avg(applied_ratios) is not None and (avg(applied_ratios) or 0) < 0.05:
        findings.append("current_only_mask_policy_effectively_passes_through")
    if yolo_fps and yolo_fps >= 300:
        findings.append("global_yolo_throughput_is_near_8cam_40hz_target")
    if partial is not None and partial > 0.35:
        findings.append("high_partial_batch_ratio_increases_result_age_jitter")
    if avg(capture_to_record) is not None and (avg(capture_to_record) or 0) > 80:
        findings.append("capture_to_record_latency_is_nontrivial")
    return {
        "lag_p50_frames": lag_p50_frames,
        "lag_p50_ms": lag_p50_ms,
        "applied_ratio_avg": avg(applied_ratios),
        "current_mask_missing_ratio_avg": avg(missing_ratios),
        "capture_to_record_ms_p50_avg": avg(capture_to_record),
        "global_yolo_wall_fps_sum": round(yolo_fps, 3) if yolo_fps else None,
        "global_yolo_partial_batch_ratio_avg": partial,
        "global_yolo_collect_wait_ms_avg": collect,
        "findings": findings,
    }


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Human Mask Latency Analysis",
        "",
        f"- inputs: {', '.join(summary.get('inputs', []))}",
        f"- run_window_duration_s: {summary.get('run_window', {}).get('duration_s')}",
        f"- run_window_margin_s: {summary.get('run_window', {}).get('margin_s')}",
        f"- trigger_period_ms: {summary.get('trigger_period_ms')}",
        f"- source_file_count: {summary.get('source_file_count')}",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(summary.get("analysis", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Per Camera",
        "",
        "| cam | human n | polygon rec/person | cap->record p50 ms | mask n | lag p50 frames | lag p50 ms | applied | missing | latest reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cam, item in summary.get("per_camera", {}).items():
        hr = item.get("human_result", {})
        ms = item.get("event_mask_stats", {})
        ws = item.get("latest_worker_status", {})
        lines.append(
            f"| {cam} | {hr.get('n')} | {hr.get('polygon_records')}/{hr.get('polygon_persons')} | {hr.get('capture_to_record_ms_p50')} | "
            f"{ms.get('n')} | {ms.get('sync_lag_frames_p50')} | {ms.get('sync_lag_ms_p50')} | "
            f"{ms.get('applied_ratio')} | {ms.get('current_mask_missing_ratio')} | "
            f"{str(ws.get('human_mask_reason', ''))[:80]} |"
        )
    lines.extend(["", "## Global YOLO Workers", ""])
    lines.append("| worker | fps | avg batch | partial | collect avg ms | predict p50/p90 ms | total p50/p90 ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for wid, item in summary.get("global_yolo_workers", {}).items():
        lines.append(
            f"| {wid} | {item.get('wall_fps')} | {item.get('avg_batch_size')} | "
            f"{item.get('partial_batch_ratio')} | {item.get('collect_wait_ms_avg')} | "
            f"{item.get('predict_ms_p50')}/{item.get('predict_ms_p90')} | "
            f"{item.get('total_ms_p50')}/{item.get('total_ms_p90')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze human mask latency from a full-system run directory")
    ap.add_argument("inputs", nargs="+", help="Run directory or sync_ipc directory to scan")
    ap.add_argument("--out-json", default="", help="Write JSON summary to this path")
    ap.add_argument("--out-md", default="", help="Write Markdown summary to this path")
    ap.add_argument("--window-start-us", type=int, default=0, help="Optional inclusive wall-clock start in microseconds")
    ap.add_argument("--window-end-us", type=int, default=0, help="Optional inclusive wall-clock end in microseconds")
    ap.add_argument("--time-margin-sec", type=float, default=2.0, help="Margin around run start/end for append-only logs")
    args = ap.parse_args()

    inputs = [Path(p).expanduser() for p in args.inputs]
    summary = build_summary(
        inputs,
        start_us=args.window_start_us or None,
        end_us=args.window_end_us or None,
        time_margin_sec=args.time_margin_sec,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).expanduser().write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(summary, Path(args.out_md).expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
