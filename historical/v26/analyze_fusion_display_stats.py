#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize OpenGL preview FPS, MVS display continuity, and composite FPS.

Usage:
  python3 analyze_fusion_display_stats.py sync_ipc

Key files produced by fusion_renderer_gl.py:
  gl_paint_fps.csv              actual QOpenGLWidget paint rate
  composite_fps_*.csv           effective MVS/event/human composite rate
  display_audit_*.csv           sampled sync/alignment state
  mvs_display_trace_*.csv       every displayed MVS frame change; proves frame jumps
  gl_stage_perf.csv             poll/paint stage timing; shows where time is spent
  mvs_img_diff_* fields          sampled pixel-content continuity in mvs_display_trace
"""
from __future__ import annotations

import csv
import glob
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional


def _float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _stats(vals: List[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    vals_sorted = sorted(vals)
    p95 = vals_sorted[min(len(vals_sorted) - 1, int(round((len(vals_sorted) - 1) * 0.95)))]
    return f"min={min(vals):.2f}, median={statistics.median(vals):.2f}, p95={p95:.2f}, max={max(vals):.2f}"


def analyze_gl_paint(sync_dir: str) -> None:
    path = os.path.join(sync_dir, "gl_paint_fps.csv")
    rows = _read_csv(path)
    print("\n=== GL paint FPS ===")
    if not rows:
        print("missing gl_paint_fps.csv")
        return
    fps = [_float(r.get("fps_1s")) for r in rows]
    gaps = [_float(r.get("max_gap_ms")) for r in rows]
    low = sum(1 for v in fps if v is not None and v <= 2.0)
    print(f"rows={len(rows)} low_fps_seconds(fps<=2)={low}")
    print("fps_1s:", _stats([v for v in fps if v is not None]))
    print("max_gap_ms:", _stats([v for v in gaps if v is not None]))


def analyze_mvs_trace(sync_dir: str) -> None:
    print("\n=== MVS displayed-frame continuity ===")
    files = sorted(glob.glob(os.path.join(sync_dir, "mvs_display_trace_*.csv")))
    if not files:
        print("missing mvs_display_trace_*.csv")
        return
    for path in files:
        rows = _read_csv(path)
        if not rows:
            continue
        cam = os.path.basename(path).replace("mvs_display_trace_", "").replace(".csv", "")
        d_sync = [_int(r.get("d_sync")) for r in rows]
        d_fid = [_int(r.get("d_frame_id")) for r in rows]
        dt = [_float(r.get("dt_ms")) for r in rows]
        age = [_float(r.get("display_delay_age_ms")) for r in rows]
        full = [_int(r.get("full_composite_near"), 0) for r in rows]
        meta = Counter((r.get("mvs_meta_status") or "") for r in rows)
        ds_valid = [v for v in d_sync if v is not None]
        df_valid = [v for v in d_fid if v is not None]
        dt_valid = [v for v in dt if v is not None]
        n_step = max(1, len(ds_valid))
        sync_1 = sum(1 for v in ds_valid if v == 1)
        sync_2 = sum(1 for v in ds_valid if v == 2)
        sync_ge3 = sum(1 for v in ds_valid if v >= 3)
        sync_le0 = sum(1 for v in ds_valid if v <= 0)
        dt_ge100 = sum(1 for v in dt_valid if v >= 100.0)
        dt_ge250 = sum(1 for v in dt_valid if v >= 250.0)
        print(f"\n[{cam}] rows={len(rows)} frame_steps={len(ds_valid)}")
        print(f"  d_sync=1 continuous: {sync_1/n_step*100:.1f}%  d_sync=2: {sync_2/n_step*100:.1f}%  d_sync>=3: {sync_ge3/n_step*100:.1f}%  d_sync<=0: {sync_le0/n_step*100:.1f}%")
        print(f"  dt>=100ms: {dt_ge100}/{len(dt_valid)}  dt>=250ms: {dt_ge250}/{len(dt_valid)}")
        print("  d_sync stats:", _stats([float(v) for v in ds_valid]))
        print("  d_frame_id stats:", _stats([float(v) for v in df_valid]))
        print("  dt_ms stats:", _stats(dt_valid))
        print("  display_delay_age_ms:", _stats([v for v in age if v is not None]))
        print(f"  full_composite_near on displayed frames: {sum(full)/max(1,len(full))*100:.1f}%")
        print("  top d_sync:", Counter(ds_valid).most_common(8))
        print("  meta status top:", meta.most_common(6))
        jumps = []
        for r in rows:
            ds = _int(r.get("d_sync"))
            dtv = _float(r.get("dt_ms"))
            if (ds is not None and ds >= 3) or (dtv is not None and dtv >= 150.0):
                jumps.append(r)
        if jumps:
            print("  sample visible jumps:")
            for r in jumps[:8]:
                print(
                    "   ",
                    f"fid={r.get('display_mvs_frame_id')} sync={r.get('display_mvs_sync')} "
                    f"dt={r.get('dt_ms')}ms d_sync={r.get('d_sync')} "
                    f"age={r.get('display_delay_age_ms')}ms latest={r.get('mvs_latest_frame_id')} "
                    f"source_gap={r.get('mvs_source_gap_ms')}ms meta={r.get('mvs_meta_status')}"
                )



def analyze_mvs_content_trace(sync_dir: str) -> None:
    print("\n=== MVS displayed-image content continuity ===")
    files = sorted(glob.glob(os.path.join(sync_dir, "mvs_display_trace_*.csv")))
    if not files:
        print("missing mvs_display_trace_*.csv")
        return
    for path in files:
        rows = _read_csv(path)
        if not rows:
            continue
        if "mvs_img_diff_mean" not in rows[0]:
            continue
        cam = os.path.basename(path).replace("mvs_display_trace_", "").replace(".csv", "")
        means = [_float(r.get("mvs_img_diff_mean")) for r in rows]
        p95s = [_float(r.get("mvs_img_diff_p95")) for r in rows]
        changed = [_float(r.get("mvs_img_changed_pct")) for r in rows]
        same = [_int(r.get("mvs_img_same_like"), 0) for r in rows]
        same_run = [_int(r.get("mvs_img_same_run"), 0) for r in rows]
        freeze = [_int(r.get("mvs_img_freeze_like"), 0) for r in rows]
        hashes = [r.get("mvs_img_hash32") or "" for r in rows]
        valid_mean = [v for v in means if v is not None]
        n = max(1, len(rows))
        same_count = sum(1 for v in same if v == 1)
        freeze_count = sum(1 for v in freeze if v == 1)
        max_same_run = max([v for v in same_run if v is not None] or [0])
        consecutive_hash_same = 0
        last_h = None
        for h in hashes:
            if h and last_h and h == last_h:
                consecutive_hash_same += 1
            if h:
                last_h = h
        print(f"\n[{cam}] rows={len(rows)}")
        print("  img_diff_mean:       ", _stats(valid_mean))
        print("  img_diff_p95:        ", _stats([v for v in p95s if v is not None]))
        print("  img_changed_pct:     ", _stats([v for v in changed if v is not None]))
        print(f"  same_like rows:       {same_count}/{n} ({same_count/n*100:.1f}%)")
        print(f"  freeze_like rows:     {freeze_count}/{n} ({freeze_count/n*100:.1f}%)")
        print(f"  max_same_like_run:    {max_same_run}")
        print(f"  exact repeated hash transitions: {consecutive_hash_same}")
        suspicious = []
        for r in rows:
            sr = _int(r.get("mvs_img_same_run"), 0)
            fr = _int(r.get("mvs_img_freeze_like"), 0)
            if fr == 1 or sr >= 5:
                suspicious.append(r)
        if suspicious:
            print("  sample possible content-freeze rows:")
            for r in suspicious[:8]:
                print(
                    "   ",
                    f"fid={r.get('display_mvs_frame_id')} sync={r.get('display_mvs_sync')} "
                    f"dt={r.get('dt_ms')}ms d_sync={r.get('d_sync')} "
                    f"diff_mean={r.get('mvs_img_diff_mean')} p95={r.get('mvs_img_diff_p95')} "
                    f"changed={r.get('mvs_img_changed_pct')}% same_run={r.get('mvs_img_same_run')} hash={r.get('mvs_img_hash32')}"
                )

def analyze_stage_perf(sync_dir: str) -> None:
    path = os.path.join(sync_dir, "gl_stage_perf.csv")
    rows = _read_csv(path)
    print("\n=== GL stage timing ===")
    if not rows:
        print("missing gl_stage_perf.csv")
        return
    by_stage: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_stage[r.get("stage") or ""].append(r)
    # Show stages that dominate either by total time or by max time.
    summaries = []
    for stage, rs in by_stage.items():
        avg_vals = [_float(r.get("avg_ms")) for r in rs]
        max_vals = [_float(r.get("max_ms")) for r in rs]
        total_vals = [_float(r.get("total_ms")) for r in rs]
        med_avg = statistics.median([v for v in avg_vals if v is not None]) if any(v is not None for v in avg_vals) else 0.0
        p95_max_vals = sorted([v for v in max_vals if v is not None])
        p95_max = p95_max_vals[min(len(p95_max_vals)-1, int(round((len(p95_max_vals)-1)*0.95)))] if p95_max_vals else 0.0
        med_total = statistics.median([v for v in total_vals if v is not None]) if any(v is not None for v in total_vals) else 0.0
        summaries.append((med_total, p95_max, med_avg, stage, rs))
    summaries.sort(reverse=True)
    print(f"rows={len(rows)} stages={len(by_stage)}")
    for med_total, p95_max, med_avg, stage, rs in summaries[:18]:
        avg_vals = [_float(r.get("avg_ms")) for r in rs]
        max_vals = [_float(r.get("max_ms")) for r in rs]
        count_vals = [_float(r.get("count")) for r in rs]
        print(f"  {stage:30s} count={_stats([v for v in count_vals if v is not None])} avg_ms={_stats([v for v in avg_vals if v is not None])} max_ms={_stats([v for v in max_vals if v is not None])}")


def analyze_composite(sync_dir: str) -> None:
    print("\n=== Effective composite FPS ===")
    files = sorted(glob.glob(os.path.join(sync_dir, "composite_fps_*.csv")))
    if not files:
        print("missing composite_fps_*.csv")
        return
    for path in files:
        rows = _read_csv(path)
        if not rows:
            continue
        cam = os.path.basename(path).replace("composite_fps_", "").replace(".csv", "")
        full_exact = [_float(r.get("full_composite_exact_fps")) for r in rows]
        full_near = [_float(r.get("full_composite_near_fps")) for r in rows]
        mvs_fps = [_float(r.get("mvs_content_fps")) for r in rows]
        human_fps = [_float(r.get("human_exact_fps")) for r in rows]
        event_fps = [_float(r.get("event_exact_fps")) for r in rows]
        meta_status = Counter((r.get("mvs_meta_status") or "") for r in rows)
        print(f"\n[{cam}] rows={len(rows)}")
        print("  mvs_content_fps:         ", _stats([v for v in mvs_fps if v is not None]))
        print("  event_exact_fps:         ", _stats([v for v in event_fps if v is not None]))
        print("  human_exact_fps:         ", _stats([v for v in human_fps if v is not None]))
        print("  full_composite_exact_fps:", _stats([v for v in full_exact if v is not None]))
        print("  full_composite_near_fps: ", _stats([v for v in full_near if v is not None]))
        print("  meta_status top:", meta_status.most_common(5))


def analyze_audit(sync_dir: str) -> None:
    print("\n=== Display audit delta ratios ===")
    files = sorted(glob.glob(os.path.join(sync_dir, "display_audit_*.csv")))
    if not files:
        print("missing display_audit_*.csv")
        return
    for path in files:
        rows = _read_csv(path)
        if not rows:
            continue
        cam = os.path.basename(path).replace("display_audit_", "").replace(".csv", "")
        event_delta = [_int(r.get("event_sync_delta")) for r in rows]
        human_delta = [_int(r.get("human_sync_delta")) for r in rows]
        full_exact = [_int(r.get("full_composite_exact"), 0) for r in rows]
        full_near = [_int(r.get("full_composite_near"), 0) for r in rows]
        delay = [_float(r.get("display_delay_age_ms")) for r in rows]
        meta = Counter((r.get("mvs_meta_status") or "") for r in rows)
        event_status = Counter((r.get("event_display_status") or "") for r in rows)
        human_status = Counter((r.get("human_display_status") or "") for r in rows)
        n = max(1, len(rows))
        e_exact = sum(1 for v in event_delta if v == 0)
        h_exact = sum(1 for v in human_delta if v == 0)
        e_near = sum(1 for v in event_delta if v is not None and abs(v) <= 1)
        h_near = sum(1 for v in human_delta if v is not None and abs(v) <= 1)
        print(f"\n[{cam}] rows={len(rows)}")
        print(f"  event exact/near ratio: {e_exact/n*100:.1f}% / {e_near/n*100:.1f}%")
        print(f"  human exact/near ratio: {h_exact/n*100:.1f}% / {h_near/n*100:.1f}%")
        print(f"  full exact/near ratio:  {sum(full_exact)/n*100:.1f}% / {sum(full_near)/n*100:.1f}%")
        print("  display_delay_age_ms:", _stats([v for v in delay if v is not None]))
        print("  event status top:", event_status.most_common(6))
        print("  human status top:", human_status.most_common(6))
        print("  meta status top:", meta.most_common(6))


def main() -> int:
    sync_dir = sys.argv[1] if len(sys.argv) >= 2 else "sync_ipc"
    sync_dir = os.path.abspath(sync_dir)
    print(f"sync_dir={sync_dir}")
    analyze_gl_paint(sync_dir)
    analyze_mvs_trace(sync_dir)
    analyze_mvs_content_trace(sync_dir)
    analyze_stage_perf(sync_dir)
    analyze_composite(sync_dir)
    analyze_audit(sync_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
