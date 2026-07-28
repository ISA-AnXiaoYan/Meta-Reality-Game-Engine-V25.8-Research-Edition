#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_mvs_event_sync_quality.py

离线分析 MVS 软触发同步日志 + 事件相机 trigger 日志，用来判断：
  1) 多台 MVS 之间每轮 SoftwareTrigger 的时间差 / spread；
  2) 主 MVS trigger 与事件相机收到 trigger 的对应关系是否稳定；
  3) 是否具备建立 trigger_index -> event_ts_us 时间映射表的数据质量。

输入：
  --mvs-csv    hik_mvs_softsync_2cam_softtrigger_strobe_test.py 生成的 CSV
  --event-csv  test_ext_trigger_in_auto_fps.py 生成的 CSV

重要说明：
  - MVS 间的 trigger_cmd_wall_us spread 反映的是软件触发命令发出时间差，
    不是严格的真实曝光开始时间差。
  - MVS-event 的 wall_delay_us = event_wall_time_us - master_trigger_cmd_wall_us
    包含事件相机 SDK 回调/缓冲/进程调度延迟，不等于纯硬件线缆延迟。
  - 真正判断能否建立时间映射，重点看：
      MVS trigger spread 是否小且稳定；
      event p=1 trigger 周期是否稳定；
      配对覆盖率是否高；
      event_ts_us 对 trigger_index 的线性拟合残差是否小。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    f = pos - lo
    return xs[lo] * (1 - f) + xs[hi] * f


def stat(values: List[float]) -> Dict[str, Any]:
    vals = [float(x) for x in values if x is not None]
    if not vals:
        return {"n": 0, "avg": None, "min": None, "max": None, "std": None, "p50": None, "p95": None, "p99": None, "range": None}
    return {
        "n": len(vals),
        "avg": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "std": statistics.pstdev(vals) if len(vals) >= 2 else 0.0,
        "p50": pct(vals, 0.50),
        "p95": pct(vals, 0.95),
        "p99": pct(vals, 0.99),
        "range": max(vals) - min(vals),
    }


def fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =============================================================================
# MVS 分析
# =============================================================================

def analyze_mvs(rows: List[Dict[str, str]], master_id: str, expected_period_us: int) -> Dict[str, Any]:
    by_idx: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    cam_ids = sorted({str(r.get("camera_id", "")) for r in rows})

    for r in rows:
        idx = to_int(r.get("trigger_index"))
        if idx is not None:
            by_idx[idx].append(r)

    expected_cams = set(cam_ids)
    missing_indices = []
    bad_rows = []
    for idx, items in sorted(by_idx.items()):
        cams = {str(r.get("camera_id", "")) for r in items}
        if cams != expected_cams:
            missing_indices.append((idx, sorted(cams)))
        for r in items:
            ok = to_int(r.get("ok"))
            tr = to_int(r.get("trigger_ret"))
            gr = to_int(r.get("grab_ret"))
            if ok != 1 or tr != 0 or gr != 0:
                bad_rows.append(r)

    frame_jumps: Dict[str, List[Tuple[int, int, int]]] = {}
    for cam in cam_ids:
        rs = [r for r in rows if str(r.get("camera_id", "")) == cam]
        rs.sort(key=lambda x: to_int(x.get("trigger_index")) or -1)
        jumps = []
        last = None
        for r in rs:
            idx = to_int(r.get("trigger_index"))
            fn = to_int(r.get("mvs_frame_num"))
            if idx is None or fn is None:
                continue
            if last is not None and fn != last + 1:
                jumps.append((idx, last, fn))
            last = fn
        frame_jumps[cam] = jumps

    trigger_spreads = []
    grab_start_spreads = []
    grab_end_spreads = []
    cmd_end_spreads = []
    master_rows: Dict[int, Dict[str, str]] = {}
    per_cam_cmd_diff: Dict[str, List[int]] = defaultdict(list)
    per_cam_grab_diff: Dict[str, List[int]] = defaultdict(list)

    for idx, items in sorted(by_idx.items()):
        cmd_vals = []
        cmd_end_vals = []
        grab_vals = []
        grab_end_vals = []
        for r in items:
            v = to_int(r.get("trigger_cmd_wall_us"))
            if v is not None:
                cmd_vals.append(v)
            ve = to_int(r.get("trigger_cmd_end_wall_us"))
            if ve is not None:
                cmd_end_vals.append(ve)
            g = to_int(r.get("grab_wall_us"))
            if g is not None:
                grab_vals.append(g)
            ge = to_int(r.get("grab_end_wall_us"))
            if ge is not None:
                grab_end_vals.append(ge)

        if len(cmd_vals) >= 2:
            trigger_spreads.append(max(cmd_vals) - min(cmd_vals))
        if len(cmd_end_vals) >= 2:
            cmd_end_spreads.append(max(cmd_end_vals) - min(cmd_end_vals))
        if len(grab_vals) >= 2:
            grab_start_spreads.append(max(grab_vals) - min(grab_vals))
        if len(grab_end_vals) >= 2:
            grab_end_spreads.append(max(grab_end_vals) - min(grab_end_vals))

        m = None
        for r in items:
            if str(r.get("camera_id", "")) == str(master_id):
                m = r
                master_rows[idx] = r
                break
        if m is not None:
            m_cmd = to_int(m.get("trigger_cmd_wall_us"))
            m_grab = to_int(m.get("grab_wall_us"))
            for r in items:
                cam = str(r.get("camera_id", ""))
                if cam == str(master_id):
                    continue
                c_cmd = to_int(r.get("trigger_cmd_wall_us"))
                c_grab = to_int(r.get("grab_wall_us"))
                if m_cmd is not None and c_cmd is not None:
                    per_cam_cmd_diff[cam].append(c_cmd - m_cmd)
                if m_grab is not None and c_grab is not None:
                    per_cam_grab_diff[cam].append(c_grab - m_grab)

    master_sorted = sorted(master_rows.items())
    master_cmd_times = [to_int(r.get("trigger_cmd_wall_us")) for _, r in master_sorted]
    master_cmd_times = [x for x in master_cmd_times if x is not None]
    master_periods = [b - a for a, b in zip(master_cmd_times, master_cmd_times[1:])]
    master_period_err = [g - expected_period_us for g in master_periods]

    return {
        "cam_ids": cam_ids,
        "trigger_count": len(by_idx),
        "row_count": len(rows),
        "missing_indices": missing_indices,
        "bad_rows": bad_rows,
        "frame_jumps": frame_jumps,
        "trigger_spread_stats": stat([float(x) for x in trigger_spreads]),
        "cmd_end_spread_stats": stat([float(x) for x in cmd_end_spreads]),
        "grab_start_spread_stats": stat([float(x) for x in grab_start_spreads]),
        "grab_end_spread_stats": stat([float(x) for x in grab_end_spreads]),
        "per_cam_cmd_diff": {k: stat([float(x) for x in v]) for k, v in per_cam_cmd_diff.items()},
        "per_cam_grab_diff": {k: stat([float(x) for x in v]) for k, v in per_cam_grab_diff.items()},
        "master_rows": master_rows,
        "master_period_stats": stat([float(x) for x in master_periods]),
        "master_period_err_stats": stat([float(x) for x in master_period_err]),
        "master_period_drop_like": [g for g in master_periods if g > expected_period_us * 1.5],
        "master_period_too_short": [g for g in master_periods if g < expected_period_us * 0.75],
    }


# =============================================================================
# Event 分析
# =============================================================================

def filter_event_rows(rows: List[Dict[str, str]], polarity: Optional[int]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        ts = to_int(r.get("event_ts_us"))
        if ts is None:
            continue
        if polarity is not None:
            p = to_int(r.get("polarity"))
            if p != polarity:
                continue
        out.append(r)
    # 优先按照 event_ts 排序；如果需要也可按 selected_index，但 timestamp 更直接。
    out.sort(key=lambda x: to_int(x.get("event_ts_us")) or 0)
    return out


def analyze_event(rows: List[Dict[str, str]], expected_period_us: int, polarity: Optional[int]) -> Dict[str, Any]:
    selected = filter_event_rows(rows, polarity)
    ts = [to_int(r.get("event_ts_us")) for r in selected]
    ts = [x for x in ts if x is not None]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    err = [g - expected_period_us for g in gaps]
    drops = [g for g in gaps if g > expected_period_us * 1.5]
    short = [g for g in gaps if g < expected_period_us * 0.75]
    bad_1ms = [g for g in gaps if abs(g - expected_period_us) > 1000]
    return {
        "selected_rows": selected,
        "trigger_count": len(selected),
        "gap_stats": stat([float(x) for x in gaps]),
        "gap_err_stats": stat([float(x) for x in err]),
        "drop_like_count": len(drops),
        "too_short_count": len(short),
        "abs_err_gt_1000us_count": len(bad_1ms),
        "drop_examples": drops[:10],
        "too_short_examples": short[:10],
        "bad_1ms_examples": bad_1ms[:10],
    }


# =============================================================================
# MVS-event 配对与拟合
# =============================================================================

def linear_fit(xs: List[float], ys: List[float]) -> Tuple[Optional[float], Optional[float], List[float]]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None, None, []
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None, None, []
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    b = my - a * mx
    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    return a, b, residuals


def build_pairs(
    master_rows: Dict[int, Dict[str, str]],
    event_rows: List[Dict[str, str]],
    offset: int,
) -> List[Dict[str, Any]]:
    # 定义：event_order = trigger_index + offset
    # offset=0 表示 MVS index=1 对应 event 第1个。
    pairs = []
    for idx in sorted(master_rows.keys()):
        j = idx + offset
        if j < 1 or j > len(event_rows):
            continue
        m = master_rows[idx]
        e = event_rows[j - 1]
        m_wall = to_int(m.get("trigger_cmd_wall_us"))
        e_wall = to_int(e.get("wall_time_us"))
        e_ts = to_int(e.get("event_ts_us"))
        if m_wall is None or e_ts is None:
            continue
        pairs.append({
            "trigger_index": idx,
            "event_order": j,
            "mvs_master_trigger_cmd_wall_us": m_wall,
            "mvs_master_trigger_cmd_end_wall_us": to_int(m.get("trigger_cmd_end_wall_us")),
            "mvs_master_grab_wall_us": to_int(m.get("grab_wall_us")),
            "mvs_master_grab_end_wall_us": to_int(m.get("grab_end_wall_us")),
            "mvs_master_frame_num": to_int(m.get("mvs_frame_num")),
            "event_ts_us": e_ts,
            "event_wall_time_us": e_wall,
            "event_wall_minus_mvs_trigger_wall_us": (e_wall - m_wall) if e_wall is not None else None,
        })
    return pairs


def score_offset(master_rows: Dict[int, Dict[str, str]], event_rows: List[Dict[str, str]], offset: int) -> Tuple[float, int, Optional[float]]:
    pairs = build_pairs(master_rows, event_rows, offset)
    delays = [p["event_wall_minus_mvs_trigger_wall_us"] for p in pairs if p.get("event_wall_minus_mvs_trigger_wall_us") is not None]
    if len(delays) < 5:
        return float("inf"), len(pairs), None
    # 这里 wall delay 包含 SDK buffer，不要求均值很小，但正确 offset 通常 std 更小。
    s = stat([float(x) for x in delays])
    std = s["std"] if s["std"] is not None else float("inf")
    mean = s["avg"]
    # 对非常离谱的均值轻微惩罚，避免错位也得到小 std。
    penalty = 0.0
    if mean is not None and (mean < -200000 or mean > 1000000):
        penalty = 100000.0
    return float(std) + penalty, len(pairs), mean


def choose_offset(master_rows: Dict[int, Dict[str, str]], event_rows: List[Dict[str, str]], search: int) -> int:
    best = None
    for off in range(-search, search + 1):
        score, n, mean = score_offset(master_rows, event_rows, off)
        item = (score, -n, off, mean)
        if best is None or item < best:
            best = item
    return int(best[2]) if best else 0


def analyze_pairs(pairs: List[Dict[str, Any]], expected_period_us: int) -> Dict[str, Any]:
    idxs = [float(p["trigger_index"]) for p in pairs]
    ets = [float(p["event_ts_us"]) for p in pairs]
    a, b, residuals = linear_fit(idxs, ets)
    delays = [p["event_wall_minus_mvs_trigger_wall_us"] for p in pairs if p.get("event_wall_minus_mvs_trigger_wall_us") is not None]
    # 线性 slope 应接近 expected_period_us。
    slope_err = (a - expected_period_us) if a is not None else None
    return {
        "paired_count": len(pairs),
        "fit_slope_event_us_per_trigger": a,
        "fit_slope_err_us": slope_err,
        "fit_intercept_us": b,
        "fit_residual_stats_us": stat([float(x) for x in residuals]),
        "wall_delay_stats_us": stat([float(x) for x in delays]),
    }


def write_pair_csv(path: str, pairs: List[Dict[str, Any]]) -> None:
    if not pairs:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(pairs[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pairs)


def print_stat_block(name: str, s: Dict[str, Any], unit: str = "us") -> None:
    print(f"  {name}: n={s.get('n', 0)} avg={fmt(s.get('avg'))}{unit} min={fmt(s.get('min'))}{unit} max={fmt(s.get('max'))}{unit} std={fmt(s.get('std'))}{unit} p95={fmt(s.get('p95'))}{unit} p99={fmt(s.get('p99'))}{unit}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="分析 MVS 间延迟与 MVS-event 时间映射质量")
    p.add_argument("--mvs-csv", required=True, help="MVS softsync CSV，例如 sync_logs/softsync_masterA_line1_25hz.csv")
    p.add_argument("--event-csv", required=True, help="事件相机 trigger CSV，例如 sync_logs/event_trigger_A_both.csv")
    p.add_argument("--master-id", default="A", help="作为 Line1 输出的主 MVS camera_id")
    p.add_argument("--event-polarity", type=int, default=1, choices=[0, 1], help="只取哪种边沿；默认 p=1")
    p.add_argument("--expected-hz", type=float, default=25.0)
    p.add_argument("--index-offset", type=int, default=None, help="手动指定 event_order = trigger_index + offset；不填则自动搜索")
    p.add_argument("--offset-search", type=int, default=20)
    p.add_argument("--out-pair-csv", default="sync_logs/mvs_event_pairs.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    expected_period_us = int(round(1_000_000.0 / max(1e-9, float(args.expected_hz))))

    mvs_rows = load_csv(args.mvs_csv)
    event_rows_raw = load_csv(args.event_csv)

    mvs = analyze_mvs(mvs_rows, master_id=args.master_id, expected_period_us=expected_period_us)
    ev = analyze_event(event_rows_raw, expected_period_us=expected_period_us, polarity=args.event_polarity)

    event_rows = ev["selected_rows"]
    if args.index_offset is None:
        offset = choose_offset(mvs["master_rows"], event_rows, args.offset_search)
        offset_mode = "auto"
    else:
        offset = int(args.index_offset)
        offset_mode = "manual"

    pairs = build_pairs(mvs["master_rows"], event_rows, offset)
    pair_summary = analyze_pairs(pairs, expected_period_us=expected_period_us)
    write_pair_csv(args.out_pair_csv, pairs)

    print("\n========== MVS 同步质量 ==========")
    print(f"MVS CSV: {args.mvs_csv}")
    print(f"camera_ids: {mvs['cam_ids']}")
    print(f"trigger_count: {mvs['trigger_count']}, row_count: {mvs['row_count']}")
    print(f"bad_ret_count: {len(mvs['bad_rows'])}")
    print(f"missing_index_count: {len(mvs['missing_indices'])}")
    print(f"missing_index_examples: {mvs['missing_indices'][:5]}")
    for cam, jumps in mvs["frame_jumps"].items():
        print(f"frame_jumps[{cam}]: {len(jumps)}, examples={jumps[:5]}")

    print("\n[MVS 周期 / spread]")
    print_stat_block("master_trigger_period", mvs["master_period_stats"])
    print_stat_block("master_period_error", mvs["master_period_err_stats"])
    print(f"  master_drop_like_count: {len(mvs['master_period_drop_like'])}, examples={mvs['master_period_drop_like'][:5]}")
    print(f"  master_too_short_count: {len(mvs['master_period_too_short'])}, examples={mvs['master_period_too_short'][:5]}")
    print_stat_block("trigger_cmd_start_spread_all_mvs", mvs["trigger_spread_stats"])
    print_stat_block("trigger_cmd_end_spread_all_mvs", mvs["cmd_end_spread_stats"])
    print_stat_block("grab_start_spread_all_mvs", mvs["grab_start_spread_stats"])
    print_stat_block("grab_end_spread_all_mvs", mvs["grab_end_spread_stats"])

    print("\n[MVS 相对主相机差值：cam - master，单位 us]")
    for cam, s in mvs["per_cam_cmd_diff"].items():
        print_stat_block(f"trigger_cmd_wall_diff[{cam}-{args.master_id}]", s)
    for cam, s in mvs["per_cam_grab_diff"].items():
        print_stat_block(f"grab_wall_diff[{cam}-{args.master_id}]", s)

    print("\n========== Event trigger 稳定性 ==========")
    print(f"Event CSV: {args.event_csv}")
    print(f"polarity filter: p={args.event_polarity}")
    print(f"event_trigger_count: {ev['trigger_count']}")
    print_stat_block("event_p1_period", ev["gap_stats"])
    print_stat_block("event_p1_period_error", ev["gap_err_stats"])
    print(f"  drop_like_count: {ev['drop_like_count']}, examples={ev['drop_examples']}")
    print(f"  too_short_count: {ev['too_short_count']}, examples={ev['too_short_examples']}")
    print(f"  abs_err_gt_1000us_count: {ev['abs_err_gt_1000us_count']}, examples={ev['bad_1ms_examples']}")

    print("\n========== MVS ↔ Event 配对 / 时间映射质量 ==========")
    print(f"offset_mode: {offset_mode}, chosen_index_offset: {offset}")
    print("定义：event_order = mvs_trigger_index + offset")
    print(f"paired_count: {pair_summary['paired_count']}")
    print(f"pair_csv: {args.out_pair_csv}")
    print(f"fit_slope_event_us_per_trigger: {fmt(pair_summary['fit_slope_event_us_per_trigger'])} us/trigger")
    print(f"fit_slope_err_vs_expected: {fmt(pair_summary['fit_slope_err_us'])} us/trigger")
    print_stat_block("event_ts_fit_residual", pair_summary["fit_residual_stats_us"])
    print_stat_block("event_wall_minus_mvs_trigger_wall", pair_summary["wall_delay_stats_us"])

    print("\n========== 结论建议 ==========")
    spread_p95 = mvs["trigger_spread_stats"].get("p95")
    ev_bad = ev["drop_like_count"] + ev["too_short_count"] + ev["abs_err_gt_1000us_count"]
    residual_std = pair_summary["fit_residual_stats_us"].get("std")
    pair_count = pair_summary["paired_count"]
    min_count = min(mvs["trigger_count"], ev["trigger_count"])
    coverage = pair_count / min_count if min_count else 0

    print(f"coverage: {coverage*100:.2f}%")
    if spread_p95 is not None:
        print(f"MVS trigger spread p95: {spread_p95:.2f} us")
    if residual_std is not None:
        print(f"event_ts mapping residual std: {residual_std:.2f} us")

    if ev_bad == 0 and coverage >= 0.98 and residual_std is not None and residual_std < 500 and spread_p95 is not None and spread_p95 < 2000:
        print("判断：数据质量较好，可以建立 trigger_index -> event_ts_us 时间映射。")
    else:
        print("判断：需要结合上面指标进一步确认；若 coverage 低、event 有漏触发，或 MVS spread 偏大，需要继续排查。")

    print("\n注意：event_wall_minus_mvs_trigger_wall 不是纯硬件延迟，它包含事件 SDK 缓冲和进程调度延迟；")
    print("如果要测真实曝光时刻差，需要硬件示波器、采集卡时间戳，或让所有 MVS 改成同一硬件脉冲触发。")


if __name__ == "__main__":
    main()
