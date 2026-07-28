#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join MVS frame audit, event trigger CSV, and YOLO human JSONL per camera.

This script answers the question used by hit judgement debugging:

    For each captured MVS frame, do we have:
      1) the MVS frame itself,
      2) an event-camera trigger timestamp for the same sync_index,
      3) a human_result JSONL row for the same sync_index?

It writes sync_ipc/frame_bundle_audit_{camera}.csv and prints a compact summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(str(v).strip()))
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(str(v).strip())
    except Exception:
        return default


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_mvs(path: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(path)
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        cam = str(r.get("camera", "") or "")
        fid = _to_int(r.get("frame_id_local"), -1)
        sync = _to_int(r.get("sync_index"), -1)
        key = (cam, fid, sync)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda r: (_to_int(r.get("sync_index"), -1), _to_int(r.get("frame_id_local"), -1)))
    return out


def _read_triggers(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = _read_csv_rows(path)
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        sync = _to_int(r.get("sync_index"), -1)
        ts = _to_int(r.get("event_ts_us", r.get("timestamp_us", r.get("timestamp"))), 0)
        if sync < 0 or ts <= 0:
            continue
        out[sync] = r
    return out


def _read_human(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            sync = _to_int(obj.get("sync_index_hint", obj.get("sync_index", obj.get("mvs_sync_index"))), -1)
            if sync < 0:
                continue
            # If there are duplicate rows for one sync, keep the latest record_wall_us.
            old = out.get(sync)
            if old is None or _to_int(obj.get("record_wall_us"), 0) >= _to_int(old.get("record_wall_us"), 0):
                out[sync] = obj
    return out


def _nearest_key(mapping: Dict[int, Any], target: int, radius: int) -> Tuple[Optional[int], Optional[int]]:
    if not mapping:
        return None, None
    best_key: Optional[int] = None
    best_delta: Optional[int] = None
    for k in mapping.keys():
        d = int(k) - int(target)
        if abs(d) > int(radius):
            continue
        if best_delta is None or abs(d) < abs(best_delta):
            best_key = int(k)
            best_delta = int(d)
    return best_key, best_delta




def _percent(num: int, den: int) -> float:
    return round(100.0 * float(num) / float(den), 2) if den else 0.0


def _read_yolo_perf(path: Path) -> Dict[str, Any]:
    rows = _read_csv_rows(path)
    if not rows:
        return {"rows": 0}
    predict_vals = [_to_float(r.get("predict_ms"), 0.0) for r in rows if _to_float(r.get("predict_ms"), 0.0) > 0]
    total_vals = [_to_float(r.get("total_batch_ms"), 0.0) for r in rows if _to_float(r.get("total_batch_ms"), 0.0) > 0]
    collect_vals = [_to_float(r.get("collect_wait_ms"), 0.0) for r in rows if _to_float(r.get("collect_wait_ms"), 0.0) >= 0]
    bs_vals = [_to_int(r.get("batch_size"), 0) for r in rows if _to_int(r.get("batch_size"), 0) > 0]
    fps_vals = [_to_float(r.get("total_fps_total"), 0.0) for r in rows if _to_float(r.get("total_fps_total"), 0.0) > 0]

    def _mean(vals: List[float]) -> float:
        return round(float(statistics.mean(vals)), 3) if vals else 0.0

    def _p95(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(len(vals) - 1, int(round((len(vals) - 1) * 0.95)))
        return round(float(vals[idx]), 3)

    last = rows[-1]
    return {
        "rows": len(rows),
        "batch_size_avg": _mean([float(x) for x in bs_vals]),
        "predict_ms_avg": _mean(predict_vals),
        "predict_ms_p95": _p95(predict_vals),
        "collect_wait_ms_avg": _mean(collect_vals),
        "collect_wait_ms_p95": _p95(collect_vals),
        "total_ms_avg": _mean(total_vals),
        "total_ms_p95": _p95(total_vals),
        "total_fps_avg": _mean(fps_vals),
        "last_half": last.get("half", ""),
        "last_retina_masks": last.get("retina_masks", ""),
        "last_max_det": last.get("max_det", ""),
        "last_imgsz": last.get("imgsz", ""),
        "last_device": last.get("device", ""),
    }

def analyze_camera(sync_dir: Path, cam: str, out_dir: Path, nearest_radius: int = 3) -> Dict[str, Any]:
    mvs_path = sync_dir / f"mvs_frame_audit_{cam}.csv"
    trig_path = sync_dir / f"event_trigger_{cam}.csv"
    human_path = sync_dir / f"human_result_{cam}.jsonl"
    out_path = out_dir / f"frame_bundle_audit_{cam}.csv"

    mvs_rows = _read_mvs(mvs_path)
    triggers = _read_triggers(trig_path)
    humans = _read_human(human_path)

    trig_keys = sorted(triggers.keys())
    fieldnames = [
        "camera",
        "status",
        "mvs_present",
        "event_trigger_found",
        "human_record_found",
        "human_infer_source",
        "human_carry",
        "human_carry_source_sync_index",
        "human_carry_delta_sync",
        "person_count",
        "sync_index",
        "mvs_frame_id_local",
        "mvs_frame_num",
        "mvs_capture_wall_us",
        "mvs_receiver_wall_us",
        "event_ts_us",
        "event_next_ts_us",
        "event_period_ms",
        "event_trigger_id",
        "human_sync_index",
        "human_frame_id_local",
        "human_mvs_frame_num",
        "human_capture_wall_us",
        "human_record_wall_us",
        "human_dt_from_mvs_capture_ms",
        "human_record_delay_ms",
        "yolo_batch_index",
        "yolo_batch_size",
        "yolo_predict_ms",
        "yolo_postprocess_batch_ms",
        "yolo_total_batch_ms",
        "yolo_half",
        "yolo_retina_masks",
        "yolo_max_det",
        "nearest_human_sync_index",
        "nearest_human_delta_sync",
        "nearest_event_trigger_sync_index",
        "nearest_event_trigger_delta_sync",
        "soft_trigger_cmd_ok",
        "soft_trigger_cmd_spread_us",
        "shape_w",
        "shape_h",
        "notes",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in mvs_rows:
            sync = _to_int(m.get("sync_index"), -1)
            fid = _to_int(m.get("frame_id_local"), -1)
            mvs_frame_num = _to_int(m.get("mvs_frame_num"), -1)
            mvs_ts = _to_int(m.get("capture_wall_us"), 0)
            recv_ts = _to_int(m.get("receiver_wall_us"), 0)
            status_parts: List[str] = []
            notes: List[str] = []

            trig = triggers.get(sync)
            trig_found = trig is not None
            event_ts = _to_int(trig.get("event_ts_us"), 0) if trig else 0
            event_trigger_id = _to_int(trig.get("trigger_id", trig.get("event_trigger_id")), -1) if trig else -1
            next_ts = 0
            event_period_ms = 0.0
            if trig_found:
                # Prefer the next consecutive sync.  If absent, use the next greater sync to expose gaps.
                next_trig = triggers.get(sync + 1)
                if next_trig is None:
                    greater = [k for k in trig_keys if k > sync]
                    if greater:
                        next_trig = triggers.get(greater[0])
                        notes.append(f"next_trigger_not_consecutive:{greater[0]}")
                if next_trig is not None:
                    next_ts = _to_int(next_trig.get("event_ts_us"), 0)
                    event_period_ms = (float(next_ts) - float(event_ts)) / 1000.0 if next_ts > 0 and event_ts > 0 else 0.0
                else:
                    status_parts.append("EVENT_NEXT_TRIGGER_MISSING")
            else:
                status_parts.append("NO_EVENT_TRIGGER")

            human = humans.get(sync)
            human_found = human is not None
            person_count = _to_int(human.get("person_count"), 0) if human else 0
            human_infer_source = str(human.get("infer_source", "") or "") if human else ""
            human_carry = bool(human and (human.get("human_carry") or human_infer_source.startswith("carry")))
            human_carry_source_sync = _to_int(human.get("human_carry_source_sync_index"), -1) if human else -1
            human_carry_delta_sync = _to_int(human.get("human_carry_delta_sync"), 0) if human else 0
            human_fid = _to_int(human.get("frame_id_local"), -1) if human else -1
            human_mvs_frame_num = _to_int(human.get("mvs_frame_num"), -1) if human else -1
            human_capture = _to_int(human.get("capture_wall_us"), 0) if human else 0
            human_record = _to_int(human.get("record_wall_us"), 0) if human else 0
            human_dt_ms = (float(human_capture) - float(mvs_ts)) / 1000.0 if human_capture and mvs_ts else 0.0
            human_delay_ms = (float(human_record) - float(human_capture)) / 1000.0 if human_record and human_capture else 0.0
            if not human_found:
                status_parts.append("NO_HUMAN_RECORD")
            else:
                if human_fid >= 0 and fid >= 0 and human_fid != fid:
                    status_parts.append("HUMAN_FRAME_ID_MISMATCH")
                if human_mvs_frame_num >= 0 and mvs_frame_num >= 0 and human_mvs_frame_num != mvs_frame_num:
                    status_parts.append("HUMAN_MVS_FRAME_NUM_MISMATCH")

            nearest_h_sync, nearest_h_delta = _nearest_key(humans, sync, nearest_radius)
            nearest_e_sync, nearest_e_delta = _nearest_key(triggers, sync, nearest_radius)

            if not status_parts:
                if human_carry:
                    status = "OK_CARRY_WITH_PERSON" if int(person_count) > 0 else "OK_CARRY_EMPTY_PERSON"
                else:
                    status = "OK_WITH_PERSON" if int(person_count) > 0 else "OK_EMPTY_PERSON"
            else:
                if human_found and int(person_count) == 0:
                    status_parts.append("HUMAN_RECORD_EMPTY_PERSON")
                if human_carry:
                    status_parts.append("HUMAN_RECORD_CARRY")
                status = "|".join(status_parts)
            counters[status] += 1
            if human_found:
                counters["HUMAN_RECORD_FOUND"] += 1
                if human_carry:
                    counters["HUMAN_RECORD_CARRY"] += 1
                    if int(person_count) > 0:
                        counters["HUMAN_RECORD_CARRY_WITH_PERSON"] += 1
                    else:
                        counters["HUMAN_RECORD_CARRY_EMPTY_PERSON"] += 1
                else:
                    counters["HUMAN_RECORD_REAL_YOLO"] += 1
                    if int(person_count) > 0:
                        counters["HUMAN_RECORD_REAL_YOLO_WITH_PERSON"] += 1
                    else:
                        counters["HUMAN_RECORD_REAL_YOLO_EMPTY_PERSON"] += 1
                if int(person_count) > 0:
                    counters["HUMAN_RECORD_WITH_PERSON"] += 1
                else:
                    counters["HUMAN_RECORD_EMPTY_PERSON"] += 1
            else:
                counters["HUMAN_RECORD_MISSING_TOTAL"] += 1

            writer.writerow({
                "camera": cam,
                "status": status,
                "mvs_present": 1,
                "event_trigger_found": int(trig_found),
                "human_record_found": int(human_found),
                "human_infer_source": human_infer_source if human_found else "",
                "human_carry": int(human_carry) if human_found else "",
                "human_carry_source_sync_index": int(human_carry_source_sync) if human_found and human_carry_source_sync >= 0 else "",
                "human_carry_delta_sync": int(human_carry_delta_sync) if human_found and human_carry else "",
                "person_count": int(person_count),
                "sync_index": int(sync),
                "mvs_frame_id_local": int(fid),
                "mvs_frame_num": int(mvs_frame_num),
                "mvs_capture_wall_us": int(mvs_ts),
                "mvs_receiver_wall_us": int(recv_ts),
                "event_ts_us": int(event_ts),
                "event_next_ts_us": int(next_ts),
                "event_period_ms": round(float(event_period_ms), 3),
                "event_trigger_id": int(event_trigger_id),
                "human_sync_index": int(sync) if human_found else "",
                "human_frame_id_local": int(human_fid) if human_found else "",
                "human_mvs_frame_num": int(human_mvs_frame_num) if human_found else "",
                "human_capture_wall_us": int(human_capture) if human_found else "",
                "human_record_wall_us": int(human_record) if human_found else "",
                "human_dt_from_mvs_capture_ms": round(float(human_dt_ms), 3) if human_found else "",
                "human_record_delay_ms": round(float(human_delay_ms), 3) if human_found else "",
                "yolo_batch_index": _to_int(human.get("yolo_batch_index"), -1) if human_found else "",
                "yolo_batch_size": _to_int(human.get("yolo_batch_size"), -1) if human_found else "",
                "yolo_predict_ms": round(_to_float(human.get("yolo_predict_ms"), 0.0), 3) if human_found else "",
                "yolo_postprocess_batch_ms": round(_to_float(human.get("yolo_postprocess_batch_ms"), 0.0), 3) if human_found else "",
                "yolo_total_batch_ms": round(_to_float(human.get("yolo_total_batch_ms"), 0.0), 3) if human_found else "",
                "yolo_half": int(bool(human.get("yolo_half"))) if human_found and human.get("yolo_half") is not None else "",
                "yolo_retina_masks": int(bool(human.get("yolo_retina_masks"))) if human_found and human.get("yolo_retina_masks") is not None else "",
                "yolo_max_det": _to_int(human.get("yolo_max_det"), -1) if human_found else "",
                "nearest_human_sync_index": int(nearest_h_sync) if nearest_h_sync is not None else "",
                "nearest_human_delta_sync": int(nearest_h_delta) if nearest_h_delta is not None else "",
                "nearest_event_trigger_sync_index": int(nearest_e_sync) if nearest_e_sync is not None else "",
                "nearest_event_trigger_delta_sync": int(nearest_e_delta) if nearest_e_delta is not None else "",
                "soft_trigger_cmd_ok": _to_int(m.get("soft_trigger_cmd_ok"), 0),
                "soft_trigger_cmd_spread_us": _to_int(m.get("soft_trigger_cmd_spread_us"), 0),
                "shape_w": _to_int(m.get("shape_w"), 0),
                "shape_h": _to_int(m.get("shape_h"), 0),
                "notes": ";".join(notes),
            })

    return {
        "camera": cam,
        "mvs_rows": len(mvs_rows),
        "trigger_rows": len(triggers),
        "human_rows": len(humans),
        "out_path": str(out_path),
        "status_counts": dict(counters),
        "human_coverage_pct": _percent(int(counters.get("HUMAN_RECORD_FOUND", 0)), len(mvs_rows)),
        "human_real_yolo_pct": _percent(int(counters.get("HUMAN_RECORD_REAL_YOLO", 0)), len(mvs_rows)),
        "human_carry_pct": _percent(int(counters.get("HUMAN_RECORD_CARRY", 0)), len(mvs_rows)),
        "human_with_person_pct": _percent(int(counters.get("HUMAN_RECORD_WITH_PERSON", 0)), len(mvs_rows)),
        "missing_mvs_audit": not mvs_path.exists(),
        "missing_event_trigger": not trig_path.exists(),
        "missing_human_result": not human_path.exists(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build per-frame MVS/Event/Human bundle audit CSVs")
    ap.add_argument("--sync-dir", default="./sync_ipc", help="Directory containing sync_ipc files")
    ap.add_argument("--cams", default="A,B,C,D,E", help="Comma-separated camera ids")
    ap.add_argument("--out-dir", default="", help="Output directory. Default: sync-dir")
    ap.add_argument("--nearest-radius", type=int, default=3, help="Nearest human/event sync radius for diagnostics")
    args = ap.parse_args()

    sync_dir = Path(args.sync_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else sync_dir
    cams = [x.strip() for x in str(args.cams).split(",") if x.strip()]

    summaries = []
    for cam in cams:
        s = analyze_camera(sync_dir, cam, out_dir, nearest_radius=int(args.nearest_radius))
        summaries.append(s)

    perf_summary = _read_yolo_perf(sync_dir / "yolo_infer_perf.csv")

    print("========== frame bundle audit summary ==========")
    if perf_summary.get("rows", 0):
        print(
            "[YOLO] batches={rows} avg_batch={batch_size_avg} "
            "predict_ms_avg={predict_ms_avg} predict_ms_p95={predict_ms_p95} "
            "collect_wait_ms_avg={collect_wait_ms_avg} collect_wait_ms_p95={collect_wait_ms_p95} "
            "total_ms_avg={total_ms_avg} total_ms_p95={total_ms_p95} "
            "total_fps_avg={total_fps_avg} "
            "settings: imgsz={last_imgsz} half={last_half} retina_masks={last_retina_masks} max_det={last_max_det} device={last_device}".format(**perf_summary)
        )
    for s in summaries:
        print(f"[{s['camera']}] mvs={s['mvs_rows']} trigger={s['trigger_rows']} human={s['human_rows']} coverage={s.get('human_coverage_pct', 0.0)}% real_yolo={s.get('human_real_yolo_pct', 0.0)}% carry={s.get('human_carry_pct', 0.0)}% with_person={s.get('human_with_person_pct', 0.0)}% -> {s['out_path']}")
        missing = [k for k in ("missing_mvs_audit", "missing_event_trigger", "missing_human_result") if s.get(k)]
        if missing:
            print(f"  missing: {', '.join(missing)}")
        counts = s.get("status_counts", {}) or {}
        if counts:
            for k, v in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:12]:
                print(f"  {k}: {v}")
        else:
            print("  no rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
