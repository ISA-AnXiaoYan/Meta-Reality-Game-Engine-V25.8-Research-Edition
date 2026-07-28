#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_test607_sync_quality.py

用于验证 test607 当前“事件相机 trigger + rentijiance 人体检测 MVS 帧号”的同步是否跑通，
并给出数据质量报告。

默认读取：
  /home/ysxq/PycharmProjects/Ids_Test_3.9/test607/sync_ipc/event_trigger_A.csv
  /home/ysxq/PycharmProjects/Ids_Test_3.9/test607/sync_ipc/event_trigger_D.csv
  /home/ysxq/PycharmProjects/Ids_Test_3.9/test607/sync_ipc/human_result_A.jsonl
  /home/ysxq/PycharmProjects/Ids_Test_3.9/test607/sync_ipc/human_result_D.jsonl

输出：
  sync_ipc/sync_quality_report.txt
  sync_ipc/sync_quality_summary.json
  sync_ipc/sync_pair_A.csv
  sync_ipc/sync_pair_D.csv

判断核心：
  1) event_trigger_* 是否有 selected trigger，即 sync_index >= 0；
  2) selected trigger 的 sync_index 是否连续，event_ts_us 周期是否接近 40000us；
  3) human_result_* 是否持续输出 mvs_frame_num / sync_index_hint；
  4) sync_index_hint - mvs_frame_num 是否为稳定固定 offset，通常应为 -1；
  5) human_result 的 sync_index_hint 是否能在 event_trigger 的 selected sync_index 中找到。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return default
    try:
        return float(s)
    except Exception:
        return default


def percentile(values: List[float], q: float) -> Optional[float]:
    xs = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def stat(values: Iterable[float]) -> Dict[str, Any]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0, "avg": None, "min": None, "max": None, "std": None, "p50": None, "p95": None, "p99": None}
    return {
        "n": len(xs),
        "avg": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
        "std": statistics.pstdev(xs) if len(xs) >= 2 else 0.0,
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "p99": percentile(xs, 0.99),
    }


def fmt(v: Any, nd: int = 2, unit: str = "") -> str:
    if v is None:
        return "NA"
    try:
        if isinstance(v, float):
            return f"{v:.{nd}f}{unit}"
        return f"{v}{unit}"
    except Exception:
        return str(v)


def fmt_stat_us(st: Dict[str, Any]) -> str:
    if not st or st.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={st['n']} avg={fmt(st.get('avg'), 1)}us "
        f"p50={fmt(st.get('p50'), 1)}us p95={fmt(st.get('p95'), 1)}us "
        f"min={fmt(st.get('min'), 1)}us max={fmt(st.get('max'), 1)}us"
    )


def tail_lines(path: Path, max_lines: int = 0) -> List[str]:
    if not path.exists():
        return []
    if max_lines and max_lines > 0:
        dq: deque[str] = deque(maxlen=max_lines)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    dq.append(line.rstrip("\n"))
        return list(dq)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_event_trigger_csv(path: Path, max_lines: int = 0) -> List[Dict[str, Any]]:
    lines = tail_lines(path, max_lines=max_lines)
    if not lines:
        return []

    # 有 header 时使用 DictReader；无 header 时按当前代码的固定 7 列兜底。
    first = lines[0].strip()
    has_header = any(k in first for k in ["sync_index", "event_ts_us", "polarity", "all_index"])
    rows: List[Dict[str, Any]] = []
    if has_header:
        reader = csv.DictReader(lines)
        for r in reader:
            if not r:
                continue
            rows.append({k.strip(): v for k, v in r.items() if k is not None})
    else:
        names = ["all_index", "sync_index", "event_ts_us", "polarity", "trigger_id", "slice_ts_us", "wall_time_us"]
        for line in lines:
            parts = next(csv.reader([line]))
            if len(parts) < 3:
                continue
            r = {names[i]: parts[i] for i in range(min(len(names), len(parts)))}
            rows.append(r)

    out: List[Dict[str, Any]] = []
    for r in rows:
        # 兼容不同脚本可能使用的字段名。
        sync_index = to_int(r.get("sync_index", r.get("selected_index")))
        event_ts_us = to_int(r.get("event_ts_us", r.get("trigger_ts_us", r.get("t"))))
        polarity = to_int(r.get("polarity", r.get("p")))
        all_index = to_int(r.get("all_index", r.get("idx")))
        trigger_id = to_int(r.get("trigger_id", r.get("id")))
        slice_ts_us = to_int(r.get("slice_ts_us"))
        wall_time_us = to_int(r.get("wall_time_us", r.get("wall_us")))
        if sync_index is None and event_ts_us is None:
            continue
        out.append({
            "all_index": all_index,
            "sync_index": sync_index if sync_index is not None else -999999,
            "event_ts_us": event_ts_us,
            "polarity": polarity,
            "trigger_id": trigger_id,
            "slice_ts_us": slice_ts_us,
            "wall_time_us": wall_time_us,
            "raw": r,
        })
    return out


def load_human_jsonl(path: Path, max_lines: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in tail_lines(path, max_lines=max_lines):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows



def _row_wall_us(row: Dict[str, Any], kind: str) -> Optional[int]:
    """取用于裁剪首尾预热/收尾抖动的 wall time。"""
    if kind == "trigger":
        return to_int(row.get("wall_time_us"))
    # human_result 优先用 record_wall_us，缺失时用 capture_wall_us。
    return to_int(row.get("record_wall_us"), to_int(row.get("capture_wall_us")))


def discard_edge_seconds(rows: List[Dict[str, Any]], kind: str, discard_start_sec: float = 0.0, discard_end_sec: float = 0.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """丢弃文件时间范围内前/后若干秒数据。

    注意：这是离线验证用的“分析裁剪”。实时击中判定的启动预热要用 start 后的 elapsed 判断；
    结束前 30s 只能在离线复盘/验证时丢弃，实时系统无法提前知道何时结束。
    """
    if not rows:
        return rows, {"input_rows": 0, "output_rows": 0, "discarded_start": 0, "discarded_end": 0}
    times = [_row_wall_us(r, kind) for r in rows]
    times_valid = [t for t in times if t is not None]
    if not times_valid:
        return rows, {"input_rows": len(rows), "output_rows": len(rows), "discarded_start": 0, "discarded_end": 0, "warning": "no wall time fields"}
    t0 = min(times_valid)
    t1 = max(times_valid)
    start_cut = t0 + int(max(0.0, float(discard_start_sec)) * 1_000_000)
    end_cut = t1 - int(max(0.0, float(discard_end_sec)) * 1_000_000)
    if start_cut > end_cut:
        return [], {
            "input_rows": len(rows),
            "output_rows": 0,
            "discarded_start": len(rows),
            "discarded_end": 0,
            "time_span_sec": (t1 - t0) / 1_000_000.0,
            "warning": "discard window is longer than available data; run longer or reduce discard seconds",
        }
    kept = []
    disc_start = 0
    disc_end = 0
    for r, t in zip(rows, times):
        if t is None:
            # 没有 wall 时间的行不参与裁剪，保留，避免误删数据。
            kept.append(r)
        elif t < start_cut:
            disc_start += 1
        elif t > end_cut:
            disc_end += 1
        else:
            kept.append(r)
    return kept, {
        "input_rows": len(rows),
        "output_rows": len(kept),
        "discarded_start": disc_start,
        "discarded_end": disc_end,
        "time_span_sec": (t1 - t0) / 1_000_000.0,
        "kept_start_wall_us": start_cut,
        "kept_end_wall_us": end_cut,
    }


def analyze_triggers(rows: List[Dict[str, Any]], expected_period_us: int) -> Dict[str, Any]:
    selected = [r for r in rows if to_int(r.get("sync_index"), -1) is not None and int(r.get("sync_index")) >= 0]
    selected.sort(key=lambda r: int(r.get("sync_index", -1)))
    indices = [int(r["sync_index"]) for r in selected]
    event_ts = [to_int(r.get("event_ts_us")) for r in selected]
    wall_ts = [to_int(r.get("wall_time_us")) for r in selected]

    dup = [idx for idx, c in Counter(indices).items() if c > 1]
    missing: List[int] = []
    if indices:
        sidx = sorted(set(indices))
        full = set(range(sidx[0], sidx[-1] + 1))
        missing = sorted(full - set(sidx))

    event_pairs = [(a, b) for a, b in zip(event_ts, event_ts[1:]) if a is not None and b is not None]
    periods = [b - a for a, b in event_pairs]
    period_err = [p - expected_period_us for p in periods]
    abs_period_err = [abs(x) for x in period_err]
    big_gaps = [(indices[i], periods[i]) for i in range(min(len(indices) - 1, len(periods))) if periods[i] > expected_period_us * 1.5]
    small_gaps = [(indices[i], periods[i]) for i in range(min(len(indices) - 1, len(periods))) if periods[i] < expected_period_us * 0.5]

    duration_event_s = None
    fps_event = None
    ev_nonnull = [x for x in event_ts if x is not None]
    if len(ev_nonnull) >= 2:
        duration_event_s = (ev_nonnull[-1] - ev_nonnull[0]) / 1_000_000.0
        if duration_event_s > 0:
            fps_event = (len(ev_nonnull) - 1) / duration_event_s

    duration_wall_s = None
    fps_wall = None
    wall_nonnull = [x for x in wall_ts if x is not None]
    if len(wall_nonnull) >= 2:
        duration_wall_s = (wall_nonnull[-1] - wall_nonnull[0]) / 1_000_000.0
        if duration_wall_s > 0:
            fps_wall = (len(wall_nonnull) - 1) / duration_wall_s

    polarity_counts = Counter(str(r.get("polarity")) for r in rows)

    return {
        "total_rows": len(rows),
        "selected_rows": len(selected),
        "invalid_edge_rows": len(rows) - len(selected),
        "first_sync_index": min(indices) if indices else None,
        "last_sync_index": max(indices) if indices else None,
        "duplicate_sync_indices": dup[:50],
        "duplicate_count": len(dup),
        "missing_sync_indices": missing[:100],
        "missing_count": len(missing),
        "period_us_stats": stat(periods),
        "period_error_us_stats": stat(period_err),
        "period_abs_error_us_stats": stat(abs_period_err),
        "big_gap_count": len(big_gaps),
        "big_gap_examples": big_gaps[:20],
        "small_gap_count": len(small_gaps),
        "small_gap_examples": small_gaps[:20],
        "duration_event_s": duration_event_s,
        "fps_event_ts": fps_event,
        "duration_wall_s": duration_wall_s,
        "fps_wall": fps_wall,
        "polarity_counts": dict(polarity_counts),
        "selected_by_index": {int(r["sync_index"]): r for r in selected},
    }


def analyze_humans(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows_sorted = sorted(rows, key=lambda r: to_int(r.get("record_wall_us"), to_int(r.get("capture_wall_us"), 0)) or 0)
    mvs_nums = [to_int(r.get("mvs_frame_num")) for r in rows_sorted]
    mvs_nums = [x for x in mvs_nums if x is not None]
    sync_hints = [to_int(r.get("sync_index_hint")) for r in rows_sorted]
    sync_hints = [x for x in sync_hints if x is not None]
    frame_ids = [to_int(r.get("frame_id_local")) for r in rows_sorted]
    frame_ids = [x for x in frame_ids if x is not None]
    offsets: List[int] = []
    fid_vs_mvs: List[int] = []
    lat_ms: List[float] = []
    person_counts: List[int] = []

    for r in rows_sorted:
        m = to_int(r.get("mvs_frame_num"))
        s = to_int(r.get("sync_index_hint"))
        fid = to_int(r.get("frame_id_local"))
        if m is not None and s is not None:
            offsets.append(s - m)
        if m is not None and fid is not None:
            fid_vs_mvs.append(m - fid)
        cap = to_int(r.get("capture_wall_us"))
        rec = to_int(r.get("record_wall_us"))
        if cap is not None and rec is not None and rec >= cap:
            lat_ms.append((rec - cap) / 1000.0)
        pc = to_int(r.get("person_count"))
        if pc is None:
            persons = r.get("persons")
            pc = len(persons) if isinstance(persons, list) else 0
        person_counts.append(int(pc))

    mvs_gaps = [b - a for a, b in zip(mvs_nums, mvs_nums[1:])]
    sync_gaps = [b - a for a, b in zip(sync_hints, sync_hints[1:])]
    nonpositive_mvs_gaps = [(i, mvs_gaps[i]) for i in range(len(mvs_gaps)) if mvs_gaps[i] <= 0]
    huge_mvs_gaps = [(i, mvs_gaps[i]) for i in range(len(mvs_gaps)) if mvs_gaps[i] >= 20]
    offset_counts = Counter(offsets)
    fid_offset_counts = Counter(fid_vs_mvs)

    duration_wall_s = None
    fps_human = None
    rec_ts = [to_int(r.get("record_wall_us")) for r in rows_sorted]
    rec_ts = [x for x in rec_ts if x is not None]
    if len(rec_ts) >= 2:
        duration_wall_s = (rec_ts[-1] - rec_ts[0]) / 1_000_000.0
        if duration_wall_s > 0:
            fps_human = (len(rec_ts) - 1) / duration_wall_s

    most_offset = offset_counts.most_common(1)[0] if offset_counts else (None, 0)
    offset_stability = most_offset[1] / max(1, len(offsets))

    return {
        "total_rows": len(rows_sorted),
        "first_mvs_frame_num": min(mvs_nums) if mvs_nums else None,
        "last_mvs_frame_num": max(mvs_nums) if mvs_nums else None,
        "first_sync_index_hint": min(sync_hints) if sync_hints else None,
        "last_sync_index_hint": max(sync_hints) if sync_hints else None,
        "mvs_frame_gap_stats": stat(mvs_gaps),
        "sync_hint_gap_stats": stat(sync_gaps),
        "nonpositive_mvs_gap_count": len(nonpositive_mvs_gaps),
        "nonpositive_mvs_gap_examples": nonpositive_mvs_gaps[:20],
        "huge_mvs_gap_count": len(huge_mvs_gaps),
        "huge_mvs_gap_examples": huge_mvs_gaps[:20],
        "sync_index_hint_minus_mvs_frame_num_counts": dict(offset_counts),
        "dominant_sync_offset": most_offset[0],
        "dominant_sync_offset_count": most_offset[1],
        "sync_offset_stability": offset_stability,
        "mvs_minus_frame_id_local_counts": dict(fid_offset_counts),
        "record_minus_capture_ms_stats": stat(lat_ms),
        "person_count_stats": stat(person_counts),
        "person_count_distribution": dict(Counter(person_counts)),
        "duration_wall_s": duration_wall_s,
        "fps_human_record": fps_human,
        "rows_sorted": rows_sorted,
    }


def pair_human_to_trigger(human_rows: List[Dict[str, Any]], trig_by_index: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    paired: List[Dict[str, Any]] = []
    missing = 0
    wall_delay_ms: List[float] = []
    event_ts_backwards = 0
    last_event_ts = None

    for r in human_rows:
        hint = to_int(r.get("sync_index_hint"))
        mvs = to_int(r.get("mvs_frame_num"))
        trig = trig_by_index.get(hint) if hint is not None else None
        ok = trig is not None
        if not ok:
            missing += 1
        event_ts = to_int(trig.get("event_ts_us")) if trig else None
        trig_wall = to_int(trig.get("wall_time_us")) if trig else None
        cap_wall = to_int(r.get("capture_wall_us"))
        rec_wall = to_int(r.get("record_wall_us"))
        if trig_wall is not None and rec_wall is not None:
            wall_delay_ms.append((rec_wall - trig_wall) / 1000.0)
        if event_ts is not None and last_event_ts is not None and event_ts < last_event_ts:
            event_ts_backwards += 1
        if event_ts is not None:
            last_event_ts = event_ts
        paired.append({
            "camera": r.get("camera", r.get("camera_name", "")),
            "mvs_frame_num": mvs,
            "frame_id_local": to_int(r.get("frame_id_local")),
            "sync_index_hint": hint,
            "paired": 1 if ok else 0,
            "event_ts_us": event_ts,
            "event_wall_time_us": trig_wall,
            "human_capture_wall_us": cap_wall,
            "human_record_wall_us": rec_wall,
            "record_minus_event_wall_ms": ((rec_wall - trig_wall) / 1000.0) if trig_wall is not None and rec_wall is not None else None,
            "person_count": to_int(r.get("person_count"), len(r.get("persons", [])) if isinstance(r.get("persons"), list) else 0),
            "mvs_serial": r.get("mvs_serial", ""),
            "event_serial": r.get("event_serial", ""),
        })

    coverage = (len(human_rows) - missing) / max(1, len(human_rows))
    return {
        "human_rows": len(human_rows),
        "paired_rows": len(human_rows) - missing,
        "missing_pair_rows": missing,
        "pair_coverage": coverage,
        "record_minus_event_wall_ms_stats": stat(wall_delay_ms),
        "event_ts_backwards_in_paired_order": event_ts_backwards,
    }, paired


def judge_camera(cam: str, trig: Dict[str, Any], human: Dict[str, Any], pair: Dict[str, Any], expected_fps: float) -> Tuple[str, List[str]]:
    issues: List[str] = []
    status = "PASS"

    if trig["selected_rows"] <= 0:
        issues.append("没有 selected trigger 行；事件相机没有收到有效 MVS 曝光脉冲，或 polarity 选择不对")
        return "FAIL", issues
    if human["total_rows"] <= 0:
        issues.append("没有 human_result 行；人体检测没有输出 JSONL")
        return "FAIL", issues

    fps = trig.get("fps_event_ts") or trig.get("fps_wall")
    if fps is not None:
        if abs(fps - expected_fps) > max(1.0, expected_fps * 0.08):
            issues.append(f"trigger FPS 偏离目标 {expected_fps:.1f}fps：当前约 {fps:.2f}fps")
            status = "WARN"
    else:
        issues.append("trigger FPS 无法计算，event_ts_us/wall_time_us 不足")
        status = "WARN"

    if trig.get("duplicate_count", 0) > 0:
        issues.append(f"存在重复 sync_index：{trig.get('duplicate_count')} 个")
        status = "FAIL"
    if trig.get("missing_count", 0) > 0:
        # 少量可能是只分析 tail 截断造成；大量才 FAIL。
        issues.append(f"selected trigger sync_index 中间缺号：{trig.get('missing_count')} 个")
        status = "FAIL"
    if trig.get("big_gap_count", 0) > 0:
        issues.append(f"trigger 周期出现大间隔：{trig.get('big_gap_count')} 次")
        status = "FAIL"
    if trig.get("small_gap_count", 0) > 0:
        issues.append(f"trigger 周期出现异常小间隔：{trig.get('small_gap_count')} 次，可能记录到错误边沿或重复 trigger")
        status = "FAIL"

    if human.get("sync_offset_stability", 0.0) < 0.99:
        issues.append(
            f"sync_index_hint - mvs_frame_num 不是稳定固定值：主 offset={human.get('dominant_sync_offset')}，稳定率={human.get('sync_offset_stability', 0)*100:.2f}%"
        )
        status = "FAIL"
    if human.get("nonpositive_mvs_gap_count", 0) > 0:
        issues.append(f"human_result 的 mvs_frame_num 出现不递增/重复：{human.get('nonpositive_mvs_gap_count')} 次")
        status = "FAIL"
    if human.get("huge_mvs_gap_count", 0) > 0:
        issues.append(f"human_result 的 mvs_frame_num 有较大跳变：{human.get('huge_mvs_gap_count')} 次；如果 process_every_n/推理负载较大，可能是正常降采样，但要关注")
        if status == "PASS":
            status = "WARN"

    if pair.get("pair_coverage", 0.0) < 0.995:
        issues.append(f"human sync_index_hint 在 event_trigger 中匹配覆盖率不足：{pair.get('pair_coverage', 0)*100:.2f}%")
        status = "FAIL"
    if pair.get("event_ts_backwards_in_paired_order", 0) > 0:
        issues.append(f"配对后的 event_ts_us 出现倒退：{pair.get('event_ts_backwards_in_paired_order')} 次")
        status = "FAIL"

    # 如果无问题，给一个明确解释。
    if not issues:
        issues.append("trigger 连续、人体帧 offset 稳定、human_result 能匹配到 event_trigger；当前数据同步质量合格")
    return status, issues


def read_latest_perf(log_dir: Path) -> Dict[str, Any]:
    # 不是核心判断，仅辅助看人体检测性能。
    result = {"log_file": None, "last_perf_lines": [], "last_errors": []}
    if not log_dir.exists():
        return result
    candidates = sorted(log_dir.glob("event_rentijiance_*/rentijiance.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if not candidates:
        return result
    p = candidates[0]
    result["log_file"] = str(p)
    lines = tail_lines(p, max_lines=300)
    perf = [ln for ln in lines if "[perf]" in ln]
    errs = [ln for ln in lines if any(k in ln.lower() for k in ["error", "traceback", "failed", "exception"])]
    result["last_perf_lines"] = perf[-5:]
    result["last_errors"] = errs[-10:]
    # 简单提取最后一条 perf。
    if perf:
        last = perf[-1]
        result["last_perf"] = last
        m = re.search(r"infer_frames/s=([0-9.]+)", last)
        if m:
            result["infer_frames_per_s"] = float(m.group(1))
        m = re.search(r"last_infer_ms=([0-9.]+)", last)
        if m:
            result["last_infer_ms"] = float(m.group(1))
    return result


def write_pair_csv(path: Path, pairs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "camera", "paired", "mvs_frame_num", "frame_id_local", "sync_index_hint",
        "event_ts_us", "event_wall_time_us", "human_capture_wall_us", "human_record_wall_us",
        "record_minus_event_wall_ms", "person_count", "mvs_serial", "event_serial",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in pairs:
            w.writerow({k: r.get(k) for k in fields})


def build_report(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    root = Path(args.root).expanduser().resolve()
    sync_dir = Path(args.sync_dir).expanduser().resolve() if args.sync_dir else root / "sync_ipc"
    out_dir = sync_dir if not args.out_dir else Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "sync_dir": str(sync_dir),
        "expected_fps": args.expected_fps,
        "expected_period_us": args.expected_period_us,
        "tail_lines": args.tail_lines,
        "discard_start_sec": args.discard_start_sec,
        "discard_end_sec": args.discard_end_sec,
        "cameras": {},
    }
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("test607 同步质量验证报告")
    lines.append("=" * 78)
    lines.append(f"生成时间: {summary['generated_at']}")
    lines.append(f"工程目录: {root}")
    lines.append(f"同步目录: {sync_dir}")
    if args.tail_lines > 0:
        lines.append(f"分析范围: 每个文件最后 {args.tail_lines} 行")
    else:
        lines.append("分析范围: 全文件")
    if args.discard_start_sec > 0 or args.discard_end_sec > 0:
        lines.append(f"首尾裁剪: 丢弃开头 {args.discard_start_sec:.1f}s，丢弃结尾 {args.discard_end_sec:.1f}s")
    lines.append("")

    for cam in args.cameras:
        cam = cam.strip()
        if not cam:
            continue
        trig_path = sync_dir / f"event_trigger_{cam}.csv"
        human_path = sync_dir / f"human_result_{cam}.jsonl"
        lines.append("-" * 78)
        lines.append(f"相机 {cam}")
        lines.append("-" * 78)
        lines.append(f"trigger 文件: {trig_path}")
        lines.append(f"human 文件 : {human_path}")

        trig_rows_raw = load_event_trigger_csv(trig_path, max_lines=args.tail_lines)
        human_rows_raw = load_human_jsonl(human_path, max_lines=args.tail_lines)
        trig_rows, trig_discard = discard_edge_seconds(trig_rows_raw, "trigger", args.discard_start_sec, args.discard_end_sec)
        human_rows, human_discard = discard_edge_seconds(human_rows_raw, "human", args.discard_start_sec, args.discard_end_sec)
        trig_an = analyze_triggers(trig_rows, args.expected_period_us)
        human_an = analyze_humans(human_rows)
        pair_an, pairs = pair_human_to_trigger(human_an["rows_sorted"], trig_an["selected_by_index"])
        status, issues = judge_camera(cam, trig_an, human_an, pair_an, args.expected_fps)

        pair_csv = out_dir / f"sync_pair_{cam}.csv"
        write_pair_csv(pair_csv, pairs)

        # selected_by_index / rows_sorted 不适合 JSON 直接输出。
        trig_json = dict(trig_an)
        trig_json.pop("selected_by_index", None)
        human_json = dict(human_an)
        human_json.pop("rows_sorted", None)
        summary["cameras"][cam] = {
            "status": status,
            "issues": issues,
            "trigger_path": str(trig_path),
            "human_path": str(human_path),
            "pair_csv": str(pair_csv),
            "discard": {"trigger": trig_discard, "human": human_discard},
            "trigger": trig_json,
            "human": human_json,
            "pair": pair_an,
        }

        lines.append(f"状态: {status}")
        for msg in issues:
            lines.append(f"  - {msg}")
        if args.discard_start_sec > 0 or args.discard_end_sec > 0:
            lines.append(f"  - trigger 裁剪: 原始 {trig_discard.get('input_rows')} 行 -> 保留 {trig_discard.get('output_rows')} 行；开头丢 {trig_discard.get('discarded_start')}，结尾丢 {trig_discard.get('discarded_end')}")
            lines.append(f"  - human 裁剪 : 原始 {human_discard.get('input_rows')} 行 -> 保留 {human_discard.get('output_rows')} 行；开头丢 {human_discard.get('discarded_start')}，结尾丢 {human_discard.get('discarded_end')}")
            if trig_discard.get('warning'):
                lines.append(f"  - trigger 裁剪警告: {trig_discard.get('warning')}")
            if human_discard.get('warning'):
                lines.append(f"  - human 裁剪警告: {human_discard.get('warning')}")
        lines.append("")
        lines.append("[trigger 质量]")
        lines.append(f"  总边沿行数 total_rows       : {trig_an['total_rows']}")
        lines.append(f"  有效 selected trigger 行数 : {trig_an['selected_rows']}")
        lines.append(f"  无效/非选定边沿行数        : {trig_an['invalid_edge_rows']}")
        lines.append(f"  sync_index 范围            : {trig_an['first_sync_index']} -> {trig_an['last_sync_index']}")
        lines.append(f"  polarity 分布              : {trig_an['polarity_counts']}")
        lines.append(f"  FPS(event_ts)              : {fmt(trig_an.get('fps_event_ts'), 3)}")
        lines.append(f"  FPS(wall)                  : {fmt(trig_an.get('fps_wall'), 3)}")
        lines.append(f"  周期 period_us             : {fmt_stat_us(trig_an['period_us_stats'])}")
        lines.append(f"  周期误差 abs(period-目标)  : {fmt_stat_us(trig_an['period_abs_error_us_stats'])}")
        lines.append(f"  duplicate sync_index       : {trig_an['duplicate_count']}")
        lines.append(f"  missing sync_index         : {trig_an['missing_count']}")
        lines.append(f"  big_gap / small_gap        : {trig_an['big_gap_count']} / {trig_an['small_gap_count']}")
        if trig_an["big_gap_examples"]:
            lines.append(f"  big_gap 示例               : {trig_an['big_gap_examples'][:5]}")
        if trig_an["small_gap_examples"]:
            lines.append(f"  small_gap 示例             : {trig_an['small_gap_examples'][:5]}")

        lines.append("")
        lines.append("[human_result 质量]")
        lines.append(f"  human 行数                 : {human_an['total_rows']}")
        lines.append(f"  mvs_frame_num 范围         : {human_an['first_mvs_frame_num']} -> {human_an['last_mvs_frame_num']}")
        lines.append(f"  sync_index_hint 范围       : {human_an['first_sync_index_hint']} -> {human_an['last_sync_index_hint']}")
        lines.append(f"  human 输出 FPS(record)     : {fmt(human_an.get('fps_human_record'), 3)}")
        lines.append(f"  mvs_frame_num 间隔         : {human_an['mvs_frame_gap_stats']}")
        lines.append(f"  sync_index_hint-mvs_frame  : {human_an['sync_index_hint_minus_mvs_frame_num_counts']}")
        lines.append(f"  主 offset / 稳定率         : {human_an['dominant_sync_offset']} / {human_an['sync_offset_stability']*100:.3f}%")
        lines.append(f"  record-capture 延迟        : {human_an['record_minus_capture_ms_stats']}")
        lines.append(f"  人数统计 person_count      : {human_an['person_count_stats']}")
        lines.append(f"  人数分布                   : {human_an['person_count_distribution']}")
        lines.append(f"  非递增 mvs_frame_num       : {human_an['nonpositive_mvs_gap_count']}")
        lines.append(f"  大跳变 mvs_frame_num       : {human_an['huge_mvs_gap_count']}")

        lines.append("")
        lines.append("[human ↔ trigger 配对]")
        lines.append(f"  human 行数                 : {pair_an['human_rows']}")
        lines.append(f"  配对成功行数               : {pair_an['paired_rows']}")
        lines.append(f"  配对缺失行数               : {pair_an['missing_pair_rows']}")
        lines.append(f"  配对覆盖率                 : {pair_an['pair_coverage']*100:.4f}%")
        lines.append(f"  record_wall - trigger_wall : {pair_an['record_minus_event_wall_ms_stats']}")
        lines.append(f"  配对 CSV                   : {pair_csv}")
        lines.append("")

    # A/D 概览对比。
    if len(args.cameras) >= 2:
        lines.append("-" * 78)
        lines.append("A/D 两路概览")
        lines.append("-" * 78)
        cams = [c for c in args.cameras if c in summary["cameras"]]
        for c in cams:
            ca = summary["cameras"][c]
            tr = ca["trigger"]
            hu = ca["human"]
            lines.append(
                f"{c}: status={ca['status']} trigger_selected={tr.get('selected_rows')} "
                f"human_rows={hu.get('total_rows')} pair_coverage={ca['pair'].get('pair_coverage',0)*100:.3f}% "
                f"offset={hu.get('dominant_sync_offset')}"
            )
        if len(cams) >= 2:
            c1, c2 = cams[0], cams[1]
            t1 = summary["cameras"][c1]["trigger"]
            t2 = summary["cameras"][c2]["trigger"]
            last1 = t1.get("last_sync_index")
            last2 = t2.get("last_sync_index")
            if last1 is not None and last2 is not None:
                lines.append(f"latest trigger sync_index 差值 {c1}-{c2}: {int(last1)-int(last2)}")
            h1 = summary["cameras"][c1]["human"]
            h2 = summary["cameras"][c2]["human"]
            lh1 = h1.get("last_sync_index_hint")
            lh2 = h2.get("last_sync_index_hint")
            if lh1 is not None and lh2 is not None:
                lines.append(f"latest human sync_index_hint 差值 {c1}-{c2}: {int(lh1)-int(lh2)}")
        lines.append("")

    perf = read_latest_perf(root / "launcher_logs")
    summary["latest_rentijiance_perf"] = perf
    lines.append("-" * 78)
    lines.append("人体检测性能日志辅助信息")
    lines.append("-" * 78)
    if perf.get("log_file"):
        lines.append(f"日志文件: {perf.get('log_file')}")
        for ln in perf.get("last_perf_lines", []):
            lines.append(f"  {ln}")
        if perf.get("last_errors"):
            lines.append("最近错误/异常行:")
            for ln in perf.get("last_errors", []):
                lines.append(f"  {ln}")
    else:
        lines.append("没有找到 launcher_logs/event_rentijiance_*/rentijiance.log")

    lines.append("")
    lines.append("=" * 78)
    lines.append("结论说明")
    lines.append("=" * 78)
    lines.append("PASS 表示：当前日志中 trigger 连续、人体帧 offset 稳定、human_result 能匹配 event_trigger。")
    lines.append("WARN 表示：链路基本通，但有性能降采样、FPS 偏移或较大 frame gap，需要结合配置判断。")
    lines.append("FAIL 表示：当前数据不能可靠用于击中判定，需要先处理缺 trigger、错 offset、缺 human_result 等问题。")
    lines.append("注意：A/D 两台事件相机的 event_ts_us 原始零点不同，不能直接互相比较；跨相机只能先看 sync_index/帧号质量。")

    report = "\n".join(lines) + "\n"
    return report, summary


def main() -> None:
    p = argparse.ArgumentParser(description="验证 test607 事件 trigger 与人体检测 MVS 帧号同步质量")
    p.add_argument("--root", default="/home/ysxq/PycharmProjects/Ids_Test_3.9/test607", help="test607 工程目录")
    p.add_argument("--sync-dir", default="", help="sync_ipc 目录；默认 root/sync_ipc")
    p.add_argument("--out-dir", default="", help="报告输出目录；默认 sync_ipc")
    p.add_argument("--cameras", nargs="+", default=["A", "D"], help="要分析的相机 ID，例如 A D")
    p.add_argument("--expected-fps", type=float, default=25.0)
    p.add_argument("--expected-period-us", type=int, default=40000)
    p.add_argument("--tail-lines", type=int, default=0, help="只分析每个文件最后 N 行；0=全文件")
    p.add_argument("--discard-start-sec", type=float, default=0.0, help="离线验证时丢弃每个文件开头 N 秒，建议正式验收用 30")
    p.add_argument("--discard-end-sec", type=float, default=0.0, help="离线验证时丢弃每个文件结尾 N 秒，建议正式验收用 30")
    args = p.parse_args()

    report, summary = build_report(args)
    sync_dir = Path(args.sync_dir).expanduser().resolve() if args.sync_dir else Path(args.root).expanduser().resolve() / "sync_ipc"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else sync_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "sync_quality_report.txt"
    summary_path = out_dir / "sync_quality_summary.json"
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report)
    print(f"[saved] report : {report_path}")
    print(f"[saved] summary: {summary_path}")


if __name__ == "__main__":
    main()
