#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YSXQ hit-sync verifier

用于验证：子弹事件 timestamp_us -> event_trigger_{cam}.csv 的 sync_index ->
human_result_{cam}.jsonl 的人体帧 sync_index_hint 是否正确对应。

运行示例：
  cd /home/ysxq/PycharmProjects/Ids_Test_3.9/test607
  python3 ysxq_verify_hit_sync.py --config hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json --root ./sync_ipc

只看 D/E：
  python3 ysxq_verify_hit_sync.py --cams D,E --root ./sync_ipc

实时刷新：
  python3 ysxq_verify_hit_sync.py --cams D --root ./sync_ipc --watch 1

验证某一个子弹事件时间戳：
  python3 ysxq_verify_hit_sync.py --cam D --root ./sync_ipc --event-ts 1779517004123456
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


def now_us() -> int:
    return int(time.time() * 1_000_000)


def as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def fmt_ms(us: Optional[int]) -> str:
    if us is None:
        return "NA"
    try:
        return f"{us/1000.0:.3f}ms"
    except Exception:
        return "NA"


def status_mark(ok: Optional[bool]) -> str:
    if ok is True:
        return "OK"
    if ok is False:
        return "BAD"
    return "WARN"


def pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def tail_lines(path: str, limit: int) -> List[str]:
    """Return last limit lines. Simple and safe for continuously appended logs."""
    if limit <= 0:
        return []
    dq: Deque[str] = deque(maxlen=limit)
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                if line.strip():
                    dq.append(line.rstrip("\n"))
    except FileNotFoundError:
        return []
    except Exception as exc:
        return [f"__READ_ERROR__ {exc}"]
    return list(dq)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def read_jsonl_tail(path: str, limit: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rows: List[Dict[str, Any]] = []
    err = None
    for line in tail_lines(path, limit):
        if line.startswith("__READ_ERROR__"):
            err = line
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception as exc:
            err = f"json parse failed: {exc}"
    return rows, err


def get_config_value(cfg: Optional[Dict[str, Any]], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def detect_cams(cfg: Optional[Dict[str, Any]], cams_arg: str) -> List[str]:
    if cams_arg:
        return [x.strip() for x in cams_arg.split(",") if x.strip()]
    cams: List[str] = []
    if cfg:
        active = str(cfg.get("active_profile", "") or "")
        profile = get_config_value(cfg, ["profiles", active], {})
        routes = profile.get("routes") if isinstance(profile, dict) else None
        if isinstance(routes, list):
            for r in routes:
                if isinstance(r, dict) and r.get("enabled", True):
                    cid = str(r.get("id", "") or "").strip()
                    if cid:
                        cams.append(cid)
        serials = get_config_value(cfg, ["sync", "event_serials"], {})
        if not cams and isinstance(serials, dict):
            cams = [str(k) for k in serials.keys()]
    return cams or ["A", "B", "C", "D", "E"]


def resolve_root(args: argparse.Namespace, cfg: Optional[Dict[str, Any]]) -> str:
    root = args.root
    if not root and cfg:
        root = str(get_config_value(cfg, ["sync", "root"], "./sync_ipc") or "./sync_ipc")
    if not root:
        root = "./sync_ipc"
    return os.path.abspath(os.path.expanduser(root))


def path_tpl(root: str, template: str, cam: str) -> str:
    return os.path.abspath(template.format(root=root, camera=cam, cam=cam))


def read_trigger_csv(path: str, tail_limit: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
        "rows_raw": 0,
        "rows_selected": 0,
        "rows_helper": 0,
        "bad_rows": 0,
        "selected": [],  # list[(sync_index,event_ts_us,wall_time_us,raw_dict)]
        "helper_period_ms": [],
        "header": None,
        "error": None,
    }
    if not os.path.exists(path):
        return out
    try:
        lines = tail_lines(path, tail_limit + 1)
        if not lines:
            return out
        # 如果 tail 里第一行不是 header，就重新读 header。
        first = lines[0]
        if "sync_index" not in first or "event_ts_us" not in first:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                header_line = f.readline().rstrip("\n")
            if header_line:
                lines = [header_line] + lines[-tail_limit:]
        reader = csv.DictReader(lines)
        out["header"] = list(reader.fieldnames or [])
        prev_wall = None
        for r in reader:
            out["rows_raw"] += 1
            si = as_int(r.get("sync_index"), None)
            ts = as_int(r.get("event_ts_us"), None)
            wall = as_int(r.get("wall_time_us"), None)
            if wall is not None and prev_wall is not None:
                d = (wall - prev_wall) / 1000.0
                if 0 <= d < 500:
                    out["helper_period_ms"].append(d)
            if wall is not None:
                prev_wall = wall
            if si is None or ts is None or ts <= 0:
                out["bad_rows"] += 1
                continue
            if si >= 0:
                out["selected"].append((int(si), int(ts), int(wall or 0), dict(r)))
                out["rows_selected"] += 1
            else:
                out["rows_helper"] += 1
    except Exception as exc:
        out["error"] = str(exc)
    return out


def summarize_trigger(trig: Dict[str, Any], expected_ms: float, min_ms: float, max_ms: float) -> Dict[str, Any]:
    rows: List[Tuple[int, int, int, Dict[str, Any]]] = trig.get("selected") or []
    rows = sorted(rows, key=lambda x: x[0])
    periods_ms: List[float] = []
    sync_jumps: List[int] = []
    bad_periods: List[Tuple[int, float]] = []
    dup_sync = 0
    seen = set()
    for i, (si, ts, wall, _) in enumerate(rows):
        if si in seen:
            dup_sync += 1
        seen.add(si)
        if i == 0:
            continue
        prev_si, prev_ts, _, _ = rows[i-1]
        d_ms = (ts - prev_ts) / 1000.0
        if 0 <= d_ms < 5000:
            periods_ms.append(d_ms)
        sj = si - prev_si
        sync_jumps.append(sj)
        if d_ms < min_ms or d_ms > max_ms:
            bad_periods.append((si, d_ms))
    last = rows[-1] if rows else None
    med = statistics.median(periods_ms) if periods_ms else None
    return {
        "count": len(rows),
        "last_sync": last[0] if last else None,
        "last_ts": last[1] if last else None,
        "last_age_ms": (now_us() - last[1]) / 1000.0 if last and last[1] > 1_000_000_000_000 else None,
        "period_min": min(periods_ms) if periods_ms else None,
        "period_med": med,
        "period_p95": pct(periods_ms, 0.95),
        "period_max": max(periods_ms) if periods_ms else None,
        "bad_period_count": len(bad_periods),
        "bad_period_examples": bad_periods[-5:],
        "dup_sync": dup_sync,
        "sync_jump_counts": Counter(sync_jumps),
        "ok": bool(rows) and med is not None and min_ms <= med <= max_ms and len(bad_periods) <= max(2, len(periods_ms) * 0.05),
    }


def read_human(path: str, tail_limit: int) -> Dict[str, Any]:
    rows, err = read_jsonl_tail(path, tail_limit)
    out: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
        "rows": rows,
        "error": err,
        "count": len(rows),
        "fallback_count": 0,
        "mismatch_count": 0,
        "missing_program_sync_count": 0,
        "person_positive_count": 0,
        "last": rows[-1] if rows else None,
    }
    for r in rows:
        meta = r.get("mvs_meta") if isinstance(r.get("mvs_meta"), dict) else {}
        si = as_int(r.get("sync_index_hint"), None)
        psi = as_int(meta.get("program_sync_index"), None)
        if meta.get("soft_trigger_pending_missing"):
            out["fallback_count"] += 1
        if psi is None:
            out["missing_program_sync_count"] += 1
        if si is not None and psi is not None and si != psi:
            out["mismatch_count"] += 1
        if int(r.get("person_count", len(r.get("persons") or [])) or 0) > 0:
            out["person_positive_count"] += 1
    return out


def summarize_human(hum: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = hum.get("rows") or []
    syncs: List[int] = []
    gaps: List[int] = []
    last = rows[-1] if rows else None
    for r in rows:
        si = as_int(r.get("sync_index_hint"), None)
        if si is not None:
            syncs.append(si)
    syncs_sorted = sorted(syncs)
    for i in range(1, len(syncs_sorted)):
        gaps.append(syncs_sorted[i] - syncs_sorted[i-1])
    meta = last.get("mvs_meta") if isinstance(last, dict) and isinstance(last.get("mvs_meta"), dict) else {}
    return {
        "count": len(rows),
        "last_sync": as_int(last.get("sync_index_hint"), None) if isinstance(last, dict) else None,
        "last_program_sync": as_int(meta.get("program_sync_index"), None),
        "last_mvs_frame_num": as_int(last.get("mvs_frame_num"), None) if isinstance(last, dict) else None,
        "last_person_count": int(last.get("person_count", len(last.get("persons") or [])) or 0) if isinstance(last, dict) else None,
        "last_record_age_ms": (now_us() - as_int(last.get("record_wall_us"), 0)) / 1000.0 if isinstance(last, dict) and as_int(last.get("record_wall_us"), 0) and as_int(last.get("record_wall_us"), 0) > 1_000_000_000_000 else None,
        "fallback_count": int(hum.get("fallback_count", 0)),
        "mismatch_count": int(hum.get("mismatch_count", 0)),
        "missing_program_sync_count": int(hum.get("missing_program_sync_count", 0)),
        "person_positive_count": int(hum.get("person_positive_count", 0)),
        "gap_counts": Counter(gaps),
        "ok": bool(rows) and int(hum.get("fallback_count", 0)) == 0 and int(hum.get("mismatch_count", 0)) == 0 and int(hum.get("missing_program_sync_count", 0)) == 0,
    }


def read_mvs_latest(path: str) -> Dict[str, Any]:
    obj = load_json(path) if os.path.exists(path) else None
    meta = obj.get("mvs_meta") if isinstance(obj, dict) and isinstance(obj.get("mvs_meta"), dict) else {}
    return {
        "path": path,
        "exists": os.path.exists(path),
        "json": obj,
        "sync": as_int(meta.get("program_sync_index"), None),
        "mvs_frame_num": as_int(obj.get("mvs_frame_num"), None) if isinstance(obj, dict) else None,
        "fallback": bool(meta.get("soft_trigger_pending_missing")) if isinstance(meta, dict) else False,
        "age_ms": (now_us() - as_int(obj.get("write_wall_us"), 0)) / 1000.0 if isinstance(obj, dict) and as_int(obj.get("write_wall_us"), 0) and as_int(obj.get("write_wall_us"), 0) > 1_000_000_000_000 else None,
    }


def find_period(rows: List[Tuple[int, int, int, Dict[str, Any]]], event_ts_us: int, expected_us: int) -> Optional[Dict[str, Any]]:
    items = sorted((ts, si) for si, ts, _wall, _raw in rows if si >= 0 and ts > 0)
    if not items:
        return None
    ts_list = [x[0] for x in items]
    i = bisect.bisect_right(ts_list, int(event_ts_us)) - 1
    if i < 0:
        return None
    anchor_ts, sync_index = items[i]
    if i + 1 < len(items):
        next_ts, next_sync = items[i+1]
        estimated = False
    else:
        next_ts = anchor_ts + expected_us
        next_sync = sync_index + 1
        estimated = True
    period_us = max(1, next_ts - anchor_ts)
    offset_us = int(event_ts_us) - int(anchor_ts)
    return {
        "sync_index": int(sync_index),
        "anchor_ts_us": int(anchor_ts),
        "next_sync_index": int(next_sync),
        "next_ts_us": int(next_ts),
        "period_ms": period_us / 1000.0,
        "offset_ms": offset_us / 1000.0,
        "estimated_next": estimated,
    }


def index_human_by_sync(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    d: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        si = as_int(r.get("sync_index_hint"), None)
        if si is not None:
            d[si].append(r)
    return d


def choose_human_for_period(period: Dict[str, Any], human_by_sync: Dict[int, List[Dict[str, Any]]], midpoint_ms: float, radius: int) -> Dict[str, Any]:
    k = int(period["sync_index"])
    k_next = int(period["next_sync_index"])
    target = k_next if float(period["offset_ms"]) >= midpoint_ms else k
    result: Dict[str, Any] = {
        "impact_sync_index": k,
        "target_person_sync_index": target,
        "used_person_sync_index": None,
        "delta_sync": None,
        "fallback_used": False,
        "reason": "",
        "person_count": None,
    }
    # 与 hit_judge 默认 impact_person_strategy="nearest" 的第一选择一致。
    if target in human_by_sync and human_by_sync[target]:
        rec = human_by_sync[target][-1]
        result.update({
            "used_person_sync_index": target,
            "delta_sync": 0,
            "reason": "exact_target_sync",
            "person_count": int(rec.get("person_count", len(rec.get("persons") or [])) or 0),
        })
        return result
    # 严格模式下 radius=0 时到这里就会失败；非严格时看附近。
    best = None
    best_abs = None
    for si in sorted(human_by_sync.keys()):
        delta = si - target
        if abs(delta) <= radius:
            if best is None or abs(delta) < best_abs:
                best = si
                best_abs = abs(delta)
    if best is not None:
        rec = human_by_sync[best][-1]
        result.update({
            "used_person_sync_index": best,
            "delta_sync": best - target,
            "fallback_used": best != target,
            "reason": "nearby_sync",
            "person_count": int(rec.get("person_count", len(rec.get("persons") or [])) or 0),
        })
    else:
        result["reason"] = "no_human_result_near_sync"
    return result


def read_debug(path: str, tail_limit: int) -> Dict[str, Any]:
    rows, err = read_jsonl_tail(path, tail_limit)
    status = Counter()
    reasons = Counter()
    fallback = Counter()
    delta = Counter()
    trig_ok = Counter()
    geom = Counter()
    for r in rows:
        status[str(r.get("status"))] += 1
        reasons[str(r.get("reject_reason"))] += 1
        fallback[str(r.get("human_fallback_used"))] += 1
        delta[str(r.get("person_frame_delta_sync"))] += 1
        trig_ok[str(r.get("trigger_period_ok"))] += 1
        gm = r.get("geometry_mode") or r.get("geometry_best_method")
        geom[str(gm)] += 1
    return {
        "path": path,
        "exists": os.path.exists(path),
        "rows": rows,
        "error": err,
        "count": len(rows),
        "status": status,
        "reasons": reasons,
        "fallback": fallback,
        "delta": delta,
        "trigger_ok": trig_ok,
        "geometry": geom,
    }


def print_counter(name: str, c: Counter, limit: int = 6) -> None:
    if not c:
        print(f"    {name}: <empty>")
        return
    items = c.most_common(limit)
    s = ", ".join([f"{k}:{v}" for k, v in items])
    print(f"    {name}: {s}")


def print_cam_report(cam: str, args: argparse.Namespace, cfg: Optional[Dict[str, Any]], root: str) -> Dict[str, Any]:
    sync_cfg = cfg.get("sync") if isinstance(cfg, dict) and isinstance(cfg.get("sync"), dict) else {}
    hit_cfg = cfg.get("hit_judge") if isinstance(cfg, dict) and isinstance(cfg.get("hit_judge"), dict) else {}
    trig_tpl = str(sync_cfg.get("event_trigger_csv_template", "{root}/event_trigger_{camera}.csv"))
    hum_tpl = str(sync_cfg.get("human_jsonl_template", "{root}/human_result_{camera}.jsonl"))
    mvs_tpl = str(sync_cfg.get("mvs_frame_pub_meta_template", "{root}/mvs_latest_{camera}.json"))
    dbg_tpl = str(hit_cfg.get("hit_debug_jsonl_template", "{root}/hit_judge_debug_{camera}.jsonl"))

    expected_ms = float(hit_cfg.get("impact_anchor_period_ms", sync_cfg.get("mvs_soft_trigger_period_ms", args.expected_ms)) or args.expected_ms)
    midpoint_ms = float(hit_cfg.get("impact_midpoint_ms", expected_ms / 2.0) or (expected_ms / 2.0))
    min_ms = float(hit_cfg.get("trigger_period_min_ms", args.min_period_ms) or args.min_period_ms)
    max_ms = float(hit_cfg.get("trigger_period_max_ms", args.max_period_ms) or args.max_period_ms)
    radius = int(args.radius if args.radius is not None else int(hit_cfg.get("human_search_radius_sync", 0) or 0))

    trig_path = path_tpl(root, trig_tpl, cam)
    hum_path = path_tpl(root, hum_tpl, cam)
    mvs_path = path_tpl(root, mvs_tpl, cam)
    dbg_path = path_tpl(root, dbg_tpl, cam)

    trig = read_trigger_csv(trig_path, args.tail)
    tsum = summarize_trigger(trig, expected_ms, min_ms, max_ms)
    hum = read_human(hum_path, args.tail)
    hsum = summarize_human(hum)
    mvs = read_mvs_latest(mvs_path)
    dbg = read_debug(dbg_path, args.tail_debug)

    print(f"\n================ Camera {cam} ================")
    print(f"[files]")
    print(f"  trigger: {'YES' if trig['exists'] else 'NO '} {trig_path}")
    print(f"  human  : {'YES' if hum['exists'] else 'NO '} {hum_path}")
    print(f"  mvs    : {'YES' if mvs['exists'] else 'NO '} {mvs_path}")
    print(f"  debug  : {'YES' if dbg['exists'] else 'NO '} {dbg_path}")

    print(f"[1] trigger CSV selected rows")
    print(f"  {status_mark(tsum['ok'])} selected={tsum['count']} helper_rows={trig.get('rows_helper')} raw_tail={trig.get('rows_raw')} bad_rows={trig.get('bad_rows')}")
    print(f"  last_sync={tsum['last_sync']} last_event_ts={tsum['last_ts']} last_age={fmt_ms(int(tsum['last_age_ms']*1000) if tsum['last_age_ms'] is not None else None)}")
    if tsum["period_med"] is None:
        print(f"  period_ms: NA  (没有足够的 sync_index>=0 行)")
    else:
        print(f"  period_ms: min={tsum['period_min']:.3f} med={tsum['period_med']:.3f} p95={tsum['period_p95']:.3f} max={tsum['period_max']:.3f} expected={expected_ms:.3f} allowed=[{min_ms:.1f},{max_ms:.1f}]")
    if tsum["bad_period_count"]:
        print(f"  BAD bad_period_count={tsum['bad_period_count']} examples(last sync,ms)={tsum['bad_period_examples']}")
    if tsum["dup_sync"]:
        print(f"  WARN duplicate sync_index rows in tail: {tsum['dup_sync']}")
    if trig.get("helper_period_ms"):
        hm = statistics.median(trig["helper_period_ms"])
        print(f"  raw/helper wall period median={hm:.3f}ms 说明：这个可能对应画面上的 per，不等于 selected trigger 周期")

    print(f"[2] human_result sync quality")
    print(f"  {status_mark(hsum['ok'])} rows={hsum['count']} person_positive={hsum['person_positive_count']} last_sync={hsum['last_sync']} last_program_sync={hsum['last_program_sync']} last_mvs_frame={hsum['last_mvs_frame_num']} last_person={hsum['last_person_count']} age={fmt_ms(int(hsum['last_record_age_ms']*1000) if hsum['last_record_age_ms'] is not None else None)}")
    print(f"  fallback_pending_missing={hsum['fallback_count']} missing_program_sync={hsum['missing_program_sync_count']} sync_mismatch={hsum['mismatch_count']}")
    if hsum["gap_counts"]:
        print_counter("human sync gap counts", hsum["gap_counts"], 5)

    print(f"[3] latest MVS meta")
    print(f"  sync={mvs['sync']} mvs_frame_num={mvs['mvs_frame_num']} fallback={mvs['fallback']} age={fmt_ms(int(mvs['age_ms']*1000) if mvs['age_ms'] is not None else None)}")
    if mvs["sync"] is not None and mvs["mvs_frame_num"] is not None:
        print(f"  mvs_frame_num - program_sync_index = {mvs['mvs_frame_num'] - mvs['sync']}  这个差值大不代表错，但如果 fallback=True 就危险")

    print(f"[4] trigger-human cross check")
    if tsum["last_sync"] is not None and hsum["last_sync"] is not None:
        print(f"  latest_human_sync - latest_trigger_sync = {hsum['last_sync'] - tsum['last_sync']}")
    selected_syncs = [si for si, _ts, _wall, _raw in trig.get("selected", [])]
    human_by_sync = index_human_by_sync(hum.get("rows") or [])
    if selected_syncs:
        last_n = selected_syncs[-min(80, len(selected_syncs)):]
        covered_any = sum(1 for si in last_n if si in human_by_sync)
        covered_person = sum(1 for si in last_n if any(int(r.get("person_count", len(r.get("persons") or [])) or 0) > 0 for r in human_by_sync.get(si, [])))
        print(f"  last_{len(last_n)} trigger sync coverage: human_record={covered_any}/{len(last_n)}, person_positive={covered_person}/{len(last_n)}")
        missing = [si for si in last_n if si not in human_by_sync]
        if missing[:10]:
            print(f"  missing human sync examples: {missing[:10]}{' ...' if len(missing)>10 else ''}")

    if args.event_ts is not None and (args.cam == cam or not args.cam):
        print(f"[5] manual bullet event timestamp mapping")
        period = find_period(trig.get("selected") or [], int(args.event_ts), int(expected_ms * 1000))
        if period is None:
            print(f"  BAD event_ts={args.event_ts} 没有落入已知 trigger 周期；通常是 event_trigger CSV 还没读到、相机映射错、或时间戳不是同一个 event-camera 时钟")
        else:
            ok_period = min_ms <= float(period["period_ms"]) <= max_ms
            print(f"  {status_mark(ok_period)} event_ts={args.event_ts} -> impact_sync={period['sync_index']} next={period['next_sync_index']} offset={period['offset_ms']:.3f}ms period={period['period_ms']:.3f}ms estimated_next={period['estimated_next']}")
            choice = choose_human_for_period(period, human_by_sync, midpoint_ms, radius)
            ok_exact = choice.get("delta_sync") == 0 and not choice.get("fallback_used")
            print(f"  {status_mark(ok_exact)} target_human_sync={choice['target_person_sync_index']} used_human_sync={choice['used_person_sync_index']} delta={choice['delta_sync']} fallback={choice['fallback_used']} person_count={choice['person_count']} reason={choice['reason']} radius={radius}")

    print(f"[6] hit_judge_debug summary")
    if not dbg["exists"] or dbg["count"] == 0:
        print("  WARN 没有 debug 记录。需要 hit_judge_server 正在运行，并且收到子弹 UDP 事件后才会有内容。")
    else:
        print(f"  rows={dbg['count']}")
        print_counter("status", dbg["status"])
        print_counter("reject_reason", dbg["reasons"])
        print_counter("trigger_period_ok", dbg["trigger_ok"])
        print_counter("human_fallback_used", dbg["fallback"])
        print_counter("person_frame_delta_sync", dbg["delta"])
        print_counter("geometry", dbg["geometry"])
        # 给出最后一条关键链路。
        last = dbg["rows"][-1]
        chain_keys = [
            "status", "reject_reason", "event_ts_us", "impact_sync_index", "impact_offset_ms", "trigger_period_ms",
            "target_person_sync_index", "used_person_sync_index", "person_frame_delta_sync", "human_fallback_used",
            "event_point_mvs", "geometry_best_method", "geometry_mode", "candidate_score"
        ]
        chain = {k: last.get(k) for k in chain_keys if k in last}
        print("  last_chain=" + json.dumps(chain, ensure_ascii=False, separators=(",", ":")))

    # 生成机器可读结论，供最后总体结论使用。
    cam_ok = bool(tsum["ok"] and hsum["ok"])
    if tsum["last_sync"] is not None and hsum["last_sync"] is not None:
        # human_result 写入可能略落后，不把小落后直接判死刑。
        if abs(int(hsum["last_sync"]) - int(tsum["last_sync"])) > 5:
            cam_ok = False
    return {
        "cam": cam,
        "trigger_ok": bool(tsum["ok"]),
        "human_ok": bool(hsum["ok"]),
        "cam_ok": bool(cam_ok),
        "trigger_last_sync": tsum["last_sync"],
        "human_last_sync": hsum["last_sync"],
        "debug_status": dbg["status"],
        "fallback_count": hsum["fallback_count"],
        "mismatch_count": hsum["mismatch_count"],
        "missing_program_sync_count": hsum["missing_program_sync_count"],
    }


def print_config_check(cfg: Optional[Dict[str, Any]], root: str, cams: List[str], args: argparse.Namespace) -> None:
    print("================ CONFIG CHECK ================")
    print(f"root={root}")
    print(f"cams={','.join(cams)}")
    if not cfg:
        print("WARN 没有读取 config，只按默认文件名检查。")
        return
    sync = cfg.get("sync") if isinstance(cfg.get("sync"), dict) else {}
    hit = cfg.get("hit_judge") if isinstance(cfg.get("hit_judge"), dict) else {}
    proc = get_config_value(cfg, ["common", "model", "process_every_n"], None)
    print(f"process_every_n={proc}  {'OK' if proc == 1 else 'WARN: 建议调试对应关系时设为 1'}")
    print(f"mvs_soft_trigger_period_ms={sync.get('mvs_soft_trigger_period_ms')} impact_anchor_period_ms={hit.get('impact_anchor_period_ms')} trigger_allowed=[{hit.get('trigger_period_min_ms')},{hit.get('trigger_period_max_ms')}]")
    print(f"impact_sync_enable={hit.get('impact_sync_enable')} impact_person_strategy={hit.get('impact_person_strategy')} human_search_radius_sync={hit.get('human_search_radius_sync')} short_grace_ms={hit.get('short_grace_ms')}")
    if args.strict:
        print("STRICT 检查口径：你传了 --strict，建议配置 human_search_radius_sync=0，用来验证严格同 sync 对应。")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify YSXQ bullet-human impact sync chain")
    ap.add_argument("--config", default="hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json", help="配置文件路径；找不到也能只按 --root 检查")
    ap.add_argument("--root", default="", help="sync_ipc 目录；不填则从 config.sync.root 读取，默认 ./sync_ipc")
    ap.add_argument("--cams", default="", help="例如 A,B,C,D,E")
    ap.add_argument("--cam", default="", help="配合 --event-ts，只验证某一路；也可作为单路简写")
    ap.add_argument("--tail", type=int, default=2000, help="读取 trigger/human 末尾多少行")
    ap.add_argument("--tail-debug", type=int, default=1000, help="读取 hit_judge_debug 末尾多少行")
    ap.add_argument("--watch", type=float, default=0.0, help="大于 0 时循环刷新，单位秒")
    ap.add_argument("--event-ts", type=int, default=None, help="手动输入 BulletEvent.timestamp_us，验证它映射到哪个 sync 和人体帧")
    ap.add_argument("--radius", type=int, default=None, help="手动覆盖 human_search_radius_sync；严格验证用 --radius 0")
    ap.add_argument("--strict", action="store_true", help="输出严格验证提示；等价于建议使用 --radius 0")
    ap.add_argument("--expected-ms", type=float, default=40.0)
    ap.add_argument("--min-period-ms", type=float, default=25.0)
    ap.add_argument("--max-period-ms", type=float, default=60.0)
    args = ap.parse_args()

    cfg = load_json(args.config) if args.config else None
    root = resolve_root(args, cfg)
    cams_arg = args.cams or args.cam
    cams = detect_cams(cfg, cams_arg)
    if args.strict and args.radius is None:
        args.radius = 0

    def once() -> None:
        os.system("clear" if args.watch else ":")
        print(time.strftime("YSXQ hit-sync verify  %Y-%m-%d %H:%M:%S"))
        print_config_check(cfg, root, cams, args)
        results = []
        for cam in cams:
            results.append(print_cam_report(cam, args, cfg, root))
        print("\n================ SUMMARY ================")
        for r in results:
            print(f"{r['cam']}: trigger={status_mark(r['trigger_ok'])} human_sync={status_mark(r['human_ok'])} overall={status_mark(r['cam_ok'])} trigger_last={r['trigger_last_sync']} human_last={r['human_last_sync']} fallback={r['fallback_count']} mismatch={r['mismatch_count']} missing_program_sync={r['missing_program_sync_count']}")
        print("\n判读顺序：")
        print("1. trigger 必须 OK，selected trigger 周期中位数应接近 40ms。画面 per=8ms 不一定代表 selected trigger 错。")
        print("2. human_sync 必须 OK：fallback_pending_missing、missing_program_sync、sync_mismatch 都应为 0。")
        print("3. 有子弹事件后看 hit_judge_debug：理想是 trigger_period_ok=True、person_frame_delta_sync=0、human_fallback_used=False。")
        print("4. 若大量 no_geometry_match，说明时间对应基本过了，下一步查 H 矩阵和 event_x/y 坐标系。")

    if args.watch and args.watch > 0:
        try:
            while True:
                once()
                time.sleep(float(args.watch))
        except KeyboardInterrupt:
            return 0
    else:
        once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
