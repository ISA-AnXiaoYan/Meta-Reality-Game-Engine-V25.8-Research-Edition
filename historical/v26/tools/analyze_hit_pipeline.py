#!/usr/bin/env python3
"""
Analyze the hit-judgement pipeline end-to-end.

Reads sync_ipc logs produced by this project and generates a report that answers:
  1) Did MVS/event/human data exist for the impact sync?
  2) Did hit_judge reject because of camera mapping, trigger sync, human data,
     bullet track quality, trajectory requirements, or geometry/mask mismatch?
  3) Are candidates being generated but later rejected by behavior logic?

Usage:
  python3 tools/analyze_hit_pipeline.py --sync-dir ./sync_ipc --cams A,B,C,D,E
  python3 tools/analyze_hit_pipeline.py --sync-dir ./sync_ipc --cams A,B,C,D,E --make-zip

Outputs are written to ./sync_ipc/hit_pipeline_report/ by default.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _p95(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return 0.0
    vals.sort()
    idx = int(math.ceil(0.95 * len(vals))) - 1
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]


def _avg(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj.setdefault("_src_file", str(path))
                    obj.setdefault("_src_line", ln)
                    out.append(obj)
            except Exception:
                continue
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _infer_cam_from_name(path: Path) -> str:
    stem = path.stem
    # examples: hit_judge_debug_A, human_result_B, frame_bundle_audit_C
    for part in reversed(stem.replace("-", "_").split("_")):
        if len(part) <= 4 and part and part.lower() not in {"all", "debug", "audit", "result", "candidate", "human", "frame", "bundle", "trigger", "mvs"}:
            return part
    return ""


def _load_debug(sync_dir: Path, cams: Sequence[str]) -> List[Dict[str, Any]]:
    all_path = sync_dir / "hit_judge_debug_all.jsonl"
    if all_path.exists() and all_path.stat().st_size > 0:
        rows = _read_jsonl(all_path)
    else:
        rows = []
        for p in sorted(sync_dir.glob("hit_judge_debug_*.jsonl")):
            if p.name.endswith("_all.jsonl"):
                continue
            rows.extend(_read_jsonl(p))
    if cams:
        camset = set(cams)
        rows = [r for r in rows if str(r.get("camera") or r.get("camera_alias") or "") in camset]
    # de-duplicate when both all and per-camera were accidentally concatenated
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        key = (
            r.get("camera"), r.get("bullet_id"), r.get("seq_id"), r.get("event_ts_us"),
            r.get("status"), r.get("reject_reason"), r.get("impact_sync_index"), r.get("_src_line"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def _load_candidates(sync_dir: Path, cams: Sequence[str]) -> List[Dict[str, Any]]:
    all_path = sync_dir / "hit_candidate_all.jsonl"
    if all_path.exists() and all_path.stat().st_size > 0:
        rows = _read_jsonl(all_path)
    else:
        rows = []
        for p in sorted(sync_dir.glob("hit_candidate_*.jsonl")):
            if p.name.endswith("_all.jsonl"):
                continue
            rows.extend(_read_jsonl(p))
    if cams:
        camset = set(cams)
        rows = [r for r in rows if str(r.get("camera") or r.get("camera_alias") or r.get("cam") or "") in camset]
    return rows


def _load_frame_bundle(sync_dir: Path, cams: Sequence[str]) -> Tuple[List[Dict[str, str]], Dict[Tuple[str, int], Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    for cam in cams:
        p = sync_dir / f"frame_bundle_audit_{cam}.csv"
        for r in _read_csv(p):
            r.setdefault("camera", cam)
            rows.append(r)
    if not rows and not cams:
        for p in sorted(sync_dir.glob("frame_bundle_audit_*.csv")):
            cam = _infer_cam_from_name(p)
            for r in _read_csv(p):
                r.setdefault("camera", cam)
                rows.append(r)
    mp: Dict[Tuple[str, int], Dict[str, str]] = {}
    for r in rows:
        cam = str(r.get("camera") or "")
        sync = _as_int(r.get("sync_index"), -999999)
        if cam and sync != -999999:
            mp[(cam, sync)] = r
    return rows, mp


def _load_human_results(sync_dir: Path, cams: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cam in cams:
        p = sync_dir / f"human_result_{cam}.jsonl"
        for r in _read_jsonl(p):
            r.setdefault("camera", cam)
            rows.append(r)
    if not rows and not cams:
        for p in sorted(sync_dir.glob("human_result_*.jsonl")):
            cam = _infer_cam_from_name(p)
            for r in _read_jsonl(p):
                r.setdefault("camera", cam)
                rows.append(r)
    return rows


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return 0


def _stage_for_debug(r: Dict[str, Any]) -> str:
    status = str(r.get("status") or "")
    reason = str(r.get("reject_reason") or "")
    if status == "emitted_candidate":
        return "FINAL_HIT_EMITTED"
    if status in {"pending_impact", "pending_absorb_evidence"}:
        return "IMPACT_BEHAVIOR_PENDING"
    if status in {"impact_absorb_rejected", "near_miss"}:
        return "IMPACT_BEHAVIOR_REJECTED_OR_NEAR_MISS"
    if status in {"no_camera", "no_homography"} or reason in {"camera_not_mapped", "homography_missing"}:
        return "CAMERA_MAPPING_OR_HOMOGRAPHY"
    if status in {"no_trigger_period", "bad_trigger_period"} or "trigger" in reason:
        return "EVENT_MVS_TRIGGER_SYNC"
    if status in {"human_timeout", "no_human_result", "no_valid_human_sync"} or "human" in reason:
        return "HUMAN_DATA_OR_SYNC"
    if status == "no_valid_track":
        return "BULLET_TRACK_QUALITY"
    if status == "no_valid_trajectory":
        return "BULLET_TRAJECTORY_SEGMENT"
    if status == "no_geometry_match":
        return "GEOMETRY_MASK_OR_THRESHOLD"
    if status in {"nonimpact", "impact_duplicate_ignored"}:
        return "NON_IMPACT_OR_DUPLICATE"
    if status in {"drop", "wait_human", "wait_trigger"}:
        return "WAIT_OR_DROP"
    return "OTHER"


def _flatten_debug(r: Dict[str, Any], bundle_map: Dict[Tuple[str, int], Dict[str, str]]) -> Dict[str, Any]:
    cam = str(r.get("camera") or r.get("camera_alias") or "")
    sync = _as_int(r.get("impact_sync_index"), -999999)
    br = bundle_map.get((cam, sync), {}) if cam and sync != -999999 else {}
    cand = r.get("candidate") if isinstance(r.get("candidate"), dict) else {}
    out = {
        "camera": cam,
        "pipeline_stage": _stage_for_debug(r),
        "status": r.get("status", ""),
        "reject_reason": r.get("reject_reason", ""),
        "event_type": r.get("event_type", ""),
        "terminate_reason": r.get("terminate_reason", ""),
        "bullet_id": r.get("bullet_id", ""),
        "seq_id": r.get("seq_id", ""),
        "event_ts_us": r.get("event_ts_us", ""),
        "event_x": (r.get("event_xy") or ["", ""])[0] if isinstance(r.get("event_xy"), list) else "",
        "event_y": (r.get("event_xy") or ["", ""])[1] if isinstance(r.get("event_xy"), list) else "",
        "impact_sync_index": r.get("impact_sync_index", ""),
        "impact_offset_ms": r.get("impact_offset_ms", ""),
        "trigger_period_ms": r.get("trigger_period_ms", ""),
        "trigger_period_ok": r.get("trigger_period_ok", ""),
        "target_person_sync_index": r.get("target_person_sync_index", ""),
        "used_person_sync_index": r.get("used_person_sync_index", ""),
        "person_frame_delta_sync": r.get("person_frame_delta_sync", ""),
        "human_fallback_used": r.get("human_fallback_used", ""),
        "human_fallback_reason": r.get("human_fallback_reason", ""),
        "geometry_tested_persons": r.get("geometry_tested_persons", ""),
        "geometry_best_method": r.get("geometry_best_method", ""),
        "geometry_best_score": r.get("geometry_best_score", ""),
        "evidence_level": r.get("evidence_level", ""),
        "candidate_score": r.get("candidate_score", cand.get("total_score", "")),
        "geometry_mode": r.get("geometry_mode", cand.get("match_method", "")),
        "approach_valid": r.get("approach_valid", ""),
        "approach_len_px": r.get("approach_len_px", ""),
        "trajectory_segment_used": r.get("trajectory_segment_used", ""),
        "track_point_count": r.get("track_point_count", ""),
        "track_path_len_px": r.get("track_path_len_px", ""),
        "track_duration_ms": r.get("track_duration_ms", ""),
        "track_displacement_px": r.get("track_displacement_px", ""),
        "track_straightness": r.get("track_straightness", ""),
        "track_line_error_px": r.get("track_line_error_px", ""),
        "avg_speed_px_per_ms": r.get("avg_speed_px_per_ms", ""),
        "turn_angle_deg": r.get("turn_angle_deg", ""),
        "bundle_status": br.get("status", ""),
        "bundle_person_count": br.get("person_count", ""),
        "bundle_human_record_found": br.get("human_record_found", ""),
        "bundle_event_trigger_found": br.get("event_trigger_found", ""),
        "bundle_human_sync_index": br.get("human_sync_index", ""),
        "bundle_nearest_human_sync_index": br.get("nearest_human_sync_index", ""),
        "bundle_nearest_human_delta_sync": br.get("nearest_human_delta_sync", ""),
        "bundle_notes": br.get("notes", ""),
        "src_file": r.get("_src_file", ""),
        "src_line": r.get("_src_line", ""),
    }
    # Preserve common source/carry fields if present in newer logs.
    for k in [
        "infer_source", "person_infer_source", "human_infer_source", "human_carry", "human_carry_delta_sync",
        "human_carry_source_sync_index", "person_source", "person_source_sync_index", "person_delta_sync",
        "impact_behavior_state", "impact_behavior_reason", "absorb_evidence",
    ]:
        if k in r:
            out[k] = r.get(k)
        elif k in cand:
            out[k] = cand.get(k)
    return out


def _summarize_frame_bundle(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_cam: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_cam[str(r.get("camera") or "")].append(r)
    out: List[Dict[str, Any]] = []
    for cam, rs in sorted(by_cam.items()):
        n = len(rs)
        found = sum(1 for r in rs if _as_bool(r.get("human_record_found")) or str(r.get("status", "")).startswith("OK"))
        person = sum(1 for r in rs if _as_int(r.get("person_count"), 0) > 0)
        event = sum(1 for r in rs if _as_bool(r.get("event_trigger_found")))
        real = sum(1 for r in rs if "REAL_YOLO" in str(r.get("status")) or "REAL_YOLO" in str(r.get("notes")))
        carry = sum(1 for r in rs if "CARRY" in str(r.get("status")) or "CARRY" in str(r.get("notes")))
        status_counts = Counter(str(r.get("status") or "") for r in rs)
        top_status = ";".join(f"{k}:{v}" for k, v in status_counts.most_common(8))
        out.append({
            "camera": cam,
            "frame_bundle_rows": n,
            "human_record_found": found,
            "human_record_coverage_pct": round(found * 100.0 / n, 3) if n else 0,
            "person_count_gt0_rows": person,
            "with_person_pct": round(person * 100.0 / n, 3) if n else 0,
            "event_trigger_found": event,
            "event_trigger_coverage_pct": round(event * 100.0 / n, 3) if n else 0,
            "status_counts_top": top_status,
            "status_real_yolo_hint_rows": real,
            "status_carry_hint_rows": carry,
        })
    return out


def _summarize_human_jsonl(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cam[str(r.get("camera") or r.get("camera_id") or r.get("cam") or "")].append(r)
    out: List[Dict[str, Any]] = []
    for cam, rs in sorted(by_cam.items()):
        n = len(rs)
        pc = []
        syncs = []
        sources = Counter()
        for r in rs:
            pcount = r.get("person_count")
            if pcount is None and isinstance(r.get("persons"), list):
                pcount = len(r.get("persons") or [])
            pc.append(_as_int(pcount, 0))
            syncs.append(_as_int(r.get("sync_index"), -1))
            src = str(r.get("infer_source") or ("carry_last" if r.get("human_carry") else "yolo_or_unknown"))
            sources[src] += 1
        syncs = [s for s in syncs if s >= 0]
        gaps = []
        if syncs:
            last = None
            for s in sorted(set(syncs)):
                if last is not None and s - last > 1:
                    gaps.append(s - last)
                last = s
        out.append({
            "camera": cam,
            "human_jsonl_rows": n,
            "rows_with_person": sum(1 for x in pc if x > 0),
            "with_person_pct": round(sum(1 for x in pc if x > 0) * 100.0 / n, 3) if n else 0,
            "avg_person_count": round(_avg([float(x) for x in pc]), 3),
            "max_person_count": max(pc) if pc else 0,
            "sync_min": min(syncs) if syncs else "",
            "sync_max": max(syncs) if syncs else "",
            "sync_gap_count_gt1": len(gaps),
            "sync_gap_max": max(gaps) if gaps else 0,
            "infer_source_counts": ";".join(f"{k}:{v}" for k, v in sources.most_common()),
        })
    return out


def _summarize_yolo_perf(sync_dir: Path) -> Dict[str, Any]:
    p = sync_dir / "yolo_infer_perf.csv"
    rows = _read_csv(p)
    if not rows:
        return {"available": False}
    nums: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for k in ["batch_size", "predict_ms", "postprocess_batch_ms", "total_batch_ms", "collect_wait_ms", "total_fps_total", "predict_fps_total"]:
            if k in r:
                nums[k].append(_as_float(r.get(k), 0.0))
    settings = rows[-1] if rows else {}
    return {
        "available": True,
        "batches": len(rows),
        "avg_batch": round(_avg(nums.get("batch_size", [])), 3),
        "predict_ms_avg": round(_avg(nums.get("predict_ms", [])), 3),
        "predict_ms_p95": round(_p95(nums.get("predict_ms", [])), 3),
        "postprocess_batch_ms_avg": round(_avg(nums.get("postprocess_batch_ms", [])), 3),
        "postprocess_batch_ms_p95": round(_p95(nums.get("postprocess_batch_ms", [])), 3),
        "collect_wait_ms_avg": round(_avg(nums.get("collect_wait_ms", [])), 3),
        "collect_wait_ms_p95": round(_p95(nums.get("collect_wait_ms", [])), 3),
        "total_ms_avg": round(_avg(nums.get("total_batch_ms", [])), 3),
        "total_ms_p95": round(_p95(nums.get("total_batch_ms", [])), 3),
        "total_fps_avg": round(_avg(nums.get("total_fps_total", [])), 3),
        "settings": f"imgsz={settings.get('imgsz','')} half={settings.get('half','')} retina_masks={settings.get('retina_masks','')} max_det={settings.get('max_det','')} device={settings.get('device','')}",
    }


def _counter_rows(counter: Counter, field1: str, field2: str = "count") -> List[Dict[str, Any]]:
    return [{field1: k, field2: v} for k, v in counter.most_common()]


def _write_text_report(path: Path, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Hit pipeline diagnostic report")
    lines.append(f"generated_at={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Quick interpretation guide")
    lines.append("- CAMERA_MAPPING_OR_HOMOGRAPHY high: camera serial/alias or homography mapping problem.")
    lines.append("- EVENT_MVS_TRIGGER_SYNC high: event trigger CSV missing, bad period, or impact timestamp not bracketed by triggers.")
    lines.append("- HUMAN_DATA_OR_SYNC high: no human result for target sync, fallback rejected, or person sync delta too large.")
    lines.append("- BULLET_TRACK_QUALITY high: event-camera bullet track is rejected before geometry, often not enough points/length/straightness/speed.")
    lines.append("- BULLET_TRAJECTORY_SEGMENT high: approach segment is missing/too short, so geometry cannot be trusted.")
    lines.append("- GEOMETRY_MASK_OR_THRESHOLD high with persons tested > 0: human data exists, but mapped bullet trajectory does not intersect/near mask; likely homography/geometry/threshold issue.")
    lines.append("- FINAL_HIT_EMITTED low while GEOMETRY_MASK_OR_THRESHOLD high: logic/geometry problem, not missing data.")
    lines.append("")
    lines.append("## Top-line counts")
    for k in ["debug_rows", "candidate_rows"]:
        lines.append(f"{k}: {summary.get(k, 0)}")
    y = summary.get("yolo_perf", {}) or {}
    if y.get("available"):
        lines.append("")
        lines.append("## YOLO performance")
        for k, v in y.items():
            if k != "available":
                lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("## Pipeline stage counts")
    for r in summary.get("stage_counts", []):
        lines.append(f"{r['pipeline_stage']}: {r['count']}")
    lines.append("")
    lines.append("## Status/reason top counts")
    for r in summary.get("status_reason_top", [])[:30]:
        lines.append(f"{r['status']} | {r['reject_reason']}: {r['count']}")
    lines.append("")
    lines.append("## Files to send back")
    lines.append("Send hit_pipeline_report.zip plus, if possible, the raw sync_ipc zip used for this report.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-dir", default="./sync_ipc")
    ap.add_argument("--cams", default="A,B,C,D,E")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--make-zip", action="store_true")
    args = ap.parse_args(argv)

    sync_dir = Path(args.sync_dir).expanduser().resolve()
    cams = [x.strip() for x in str(args.cams or "").split(",") if x.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (sync_dir / "hit_pipeline_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    debug = _load_debug(sync_dir, cams)
    candidates = _load_candidates(sync_dir, cams)
    bundle_rows, bundle_map = _load_frame_bundle(sync_dir, cams)
    human_rows = _load_human_results(sync_dir, cams)
    yolo_perf = _summarize_yolo_perf(sync_dir)

    flat_debug = [_flatten_debug(r, bundle_map) for r in debug]
    _write_csv(out_dir / "hit_debug_flat.csv", flat_debug)

    # Aggregate debug status/reason/camera/stage.
    status_reason = Counter((str(r.get("status") or ""), str(r.get("reject_reason") or "")) for r in flat_debug)
    status_reason_rows = [{"status": k[0], "reject_reason": k[1], "count": v} for k, v in status_reason.most_common()]
    _write_csv(out_dir / "hit_status_reject_reason_counts.csv", status_reason_rows, ["status", "reject_reason", "count"])

    stage_counter = Counter(str(r.get("pipeline_stage") or "OTHER") for r in flat_debug)
    stage_rows = _counter_rows(stage_counter, "pipeline_stage")
    _write_csv(out_dir / "hit_pipeline_stage_counts.csv", stage_rows, ["pipeline_stage", "count"])

    cam_stage = Counter((str(r.get("camera") or ""), str(r.get("pipeline_stage") or "OTHER")) for r in flat_debug)
    cam_stage_rows = [{"camera": k[0], "pipeline_stage": k[1], "count": v} for k, v in cam_stage.most_common()]
    _write_csv(out_dir / "hit_pipeline_stage_by_camera.csv", cam_stage_rows, ["camera", "pipeline_stage", "count"])

    # Geometry-specific diagnostics.
    geom_rows = []
    for cam in sorted(set(str(r.get("camera") or "") for r in flat_debug)):
        rs = [r for r in flat_debug if str(r.get("camera") or "") == cam and r.get("pipeline_stage") == "GEOMETRY_MASK_OR_THRESHOLD"]
        if not rs:
            continue
        scores = [_as_float(r.get("geometry_best_score"), 0.0) for r in rs]
        persons = [_as_int(r.get("geometry_tested_persons"), 0) for r in rs]
        geom_rows.append({
            "camera": cam,
            "no_geometry_count": len(rs),
            "geometry_tested_persons_zero": sum(1 for x in persons if x <= 0),
            "geometry_tested_persons_gt0": sum(1 for x in persons if x > 0),
            "geometry_best_score_avg": round(_avg(scores), 4),
            "geometry_best_score_p95": round(_p95(scores), 4),
            "common_methods": ";".join(f"{k}:{v}" for k, v in Counter(str(r.get("geometry_best_method") or "") for r in rs).most_common(8)),
        })
    _write_csv(out_dir / "geometry_no_match_summary.csv", geom_rows)

    # Track-quality diagnostics.
    tr_rows = []
    by_reason: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in flat_debug:
        if r.get("pipeline_stage") == "BULLET_TRACK_QUALITY":
            by_reason[str(r.get("reject_reason") or "")].append(r)
    for reason, rs in sorted(by_reason.items(), key=lambda kv: len(kv[1]), reverse=True):
        tr_rows.append({
            "reject_reason": reason,
            "count": len(rs),
            "track_point_count_avg": round(_avg([_as_float(r.get("track_point_count"), 0) for r in rs]), 3),
            "track_path_len_px_avg": round(_avg([_as_float(r.get("track_path_len_px"), 0) for r in rs]), 3),
            "track_duration_ms_avg": round(_avg([_as_float(r.get("track_duration_ms"), 0) for r in rs]), 3),
            "track_straightness_avg": round(_avg([_as_float(r.get("track_straightness"), 0) for r in rs]), 3),
            "track_line_error_px_avg": round(_avg([_as_float(r.get("track_line_error_px"), 0) for r in rs]), 3),
            "avg_speed_px_per_ms_avg": round(_avg([_as_float(r.get("avg_speed_px_per_ms"), 0) for r in rs]), 3),
        })
    _write_csv(out_dir / "track_quality_reject_summary.csv", tr_rows)

    # Candidate flat summary.
    cand_flat: List[Dict[str, Any]] = []
    for c in candidates:
        cand_flat.append({
            "camera": c.get("camera") or c.get("cam") or c.get("camera_alias") or "",
            "bullet_id": c.get("bullet_id", ""),
            "stable_id": c.get("stable_id", ""),
            "event_ts_us": c.get("event_ts_us", c.get("timestamp_us", "")),
            "impact_sync_index": c.get("impact_sync_index", ""),
            "impact_offset_ms": c.get("impact_offset_ms", ""),
            "total_score": c.get("total_score", ""),
            "match_method": c.get("match_method", ""),
            "human_fallback_used": c.get("human_fallback_used", ""),
            "person_frame_delta_sync": c.get("person_frame_delta_sync", ""),
            "person_source": c.get("person_source", c.get("infer_source", "")),
            "raw_status": c.get("status", ""),
            "src_file": c.get("_src_file", ""),
            "src_line": c.get("_src_line", ""),
        })
    _write_csv(out_dir / "hit_candidate_flat.csv", cand_flat)

    frame_summary = _summarize_frame_bundle(bundle_rows)
    _write_csv(out_dir / "frame_bundle_summary.csv", frame_summary)
    human_summary = _summarize_human_jsonl(human_rows)
    _write_csv(out_dir / "human_result_summary.csv", human_summary)

    sync_counts = []
    for cam in cams:
        sync_counts.append({
            "camera": cam,
            "mvs_frame_audit_rows": _count_csv_rows(sync_dir / f"mvs_frame_audit_{cam}.csv"),
            "event_trigger_rows": _count_csv_rows(sync_dir / f"event_trigger_{cam}.csv"),
            "frame_bundle_rows": sum(1 for r in bundle_rows if str(r.get("camera") or "") == cam),
            "human_jsonl_rows": sum(1 for r in human_rows if str(r.get("camera") or r.get("camera_id") or r.get("cam") or "") == cam),
            "hit_debug_rows": sum(1 for r in flat_debug if str(r.get("camera") or "") == cam),
            "hit_candidate_rows": sum(1 for r in cand_flat if str(r.get("camera") or "") == cam),
        })
    _write_csv(out_dir / "sync_counts_summary.csv", sync_counts)

    # Most important rows to inspect manually: events close to final decision stages.
    important_stages = {"FINAL_HIT_EMITTED", "GEOMETRY_MASK_OR_THRESHOLD", "HUMAN_DATA_OR_SYNC", "BULLET_TRACK_QUALITY", "BULLET_TRAJECTORY_SEGMENT", "IMPACT_BEHAVIOR_REJECTED_OR_NEAR_MISS"}
    important = [r for r in flat_debug if r.get("pipeline_stage") in important_stages]
    # Sort by event timestamp, camera, bullet.
    important.sort(key=lambda r: (_as_int(r.get("event_ts_us"), 0), str(r.get("camera") or ""), _as_int(r.get("bullet_id"), 0), _as_int(r.get("seq_id"), 0)))
    _write_csv(out_dir / "important_hit_events_timeline.csv", important[:20000])

    summary = {
        "debug_rows": len(flat_debug),
        "candidate_rows": len(cand_flat),
        "yolo_perf": yolo_perf,
        "stage_counts": stage_rows,
        "status_reason_top": status_reason_rows,
    }
    _write_text_report(out_dir / "README_HIT_PIPELINE_REPORT.txt", summary)

    # Machine-readable summary.
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = None
    if args.make_zip:
        zip_path = sync_dir / "hit_pipeline_report.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(out_dir.parent)))

    print("========== hit pipeline diagnostic ==========")
    print(f"sync_dir={sync_dir}")
    print(f"out_dir={out_dir}")
    print(f"debug_rows={len(flat_debug)} candidate_rows={len(cand_flat)}")
    if yolo_perf.get("available"):
        print("[YOLO] " + " ".join(f"{k}={v}" for k, v in yolo_perf.items() if k not in {"available"}))
    print("[Pipeline stages]")
    for r in stage_rows[:20]:
        print(f"  {r['pipeline_stage']}: {r['count']}")
    print("[Top status/reason]")
    for r in status_reason_rows[:20]:
        print(f"  {r['status']} | {r['reject_reason']}: {r['count']}")
    if zip_path:
        print(f"[OK] zip={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
