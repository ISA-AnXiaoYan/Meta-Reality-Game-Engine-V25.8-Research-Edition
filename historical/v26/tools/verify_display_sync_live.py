#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path
from collections import deque

SYNC_KEYS = [
    "sync_index",
    "sync_index_hint",
    "program_sync_index",
    "frame_sync_index",
    "mvs_sync_index",
]

WALL_KEYS = [
    "record_wall_us",
    "capture_wall_us",
    "wall_us",
    "timestamp_wall_us",
    "trigger_wall_us",
    "soft_trigger_tick_wall_us",
    "soft_trigger_cmd_wall_us",
    "receiver_wall_us",
]

def to_int(v, default=None):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default

def to_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default

def get_nested_dicts(obj):
    if not isinstance(obj, dict):
        return []
    out = [obj]
    for k in ["mvs_meta", "meta", "frame_meta", "sync", "trigger"]:
        v = obj.get(k)
        if isinstance(v, dict):
            out.append(v)
    return out

def extract_sync(obj):
    for d in get_nested_dicts(obj):
        for k in SYNC_KEYS:
            v = to_int(d.get(k))
            if v is not None:
                return v
    return None

def extract_wall_us(obj):
    for d in get_nested_dicts(obj):
        for k in WALL_KEYS:
            v = to_int(d.get(k))
            if v is not None and v > 1_000_000:
                return v
    return None

def extract_person_count(obj):
    if not isinstance(obj, dict):
        return None
    pc = to_int(obj.get("person_count"))
    if pc is not None:
        return pc
    persons = obj.get("persons")
    if isinstance(persons, list):
        return len(persons)
    return None

def read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_tail_text(path, max_bytes=4_000_000):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""

def read_recent_jsonl(path, max_bytes=6_000_000):
    text = read_tail_text(path, max_bytes=max_bytes)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        sync = extract_sync(obj)
        wall = extract_wall_us(obj)
        pc = extract_person_count(obj)
        rows.append({
            "obj": obj,
            "sync": sync,
            "wall_us": wall,
            "person_count": pc,
        })
    return rows

def read_recent_csv(path, max_bytes=6_000_000):
    if not os.path.exists(path):
        return []

    # 读取第一行表头
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip()
    except Exception:
        header = ""

    text = read_tail_text(path, max_bytes=max_bytes)
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return []

    # 如果 tail 里没有表头，就补上第一行表头
    if header and lines[0].strip() != header:
        text2 = header + "\n" + "\n".join(lines)
    else:
        text2 = "\n".join(lines)

    rows = []
    try:
        reader = csv.DictReader(text2.splitlines())
        for r in reader:
            if not isinstance(r, dict):
                continue
            sync = extract_sync(r)
            wall = extract_wall_us(r)
            rows.append({
                "obj": r,
                "sync": sync,
                "wall_us": wall,
            })
    except Exception:
        return []
    return rows

def nearest_by_sync(rows, target_sync):
    if target_sync is None:
        return None
    candidates = [r for r in rows if r.get("sync") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r["sync"] - target_sync))

def nearest_by_time(rows, target_wall_us):
    if target_wall_us is None:
        return None
    candidates = [r for r in rows if r.get("wall_us") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r["wall_us"] - target_wall_us))

def row_delta(row, target_sync, target_wall_us):
    if row is None:
        return None, None
    ds = None
    dt = None
    if target_sync is not None and row.get("sync") is not None:
        ds = row["sync"] - target_sync
    if target_wall_us is not None and row.get("wall_us") is not None:
        dt = (row["wall_us"] - target_wall_us) / 1000.0
    return ds, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sync_ipc")
    ap.add_argument("--camera", default="D")
    ap.add_argument("--delay-ms", type=float, default=100.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=60.0, help="0 means forever")
    ap.add_argument("--out", default="RESULT/display_sync_verify_D.csv")
    args = ap.parse_args()

    root = Path(args.root)
    cam = args.camera

    mvs_meta_path = root / f"mvs_latest_{cam}.json"
    human_path = root / f"human_result_{cam}.jsonl"
    event_trigger_path = root / f"event_trigger_{cam}.csv"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== display sync verifier ===")
    print("camera:", cam)
    print("delay_ms:", args.delay_ms)
    print("mvs_meta:", mvs_meta_path)
    print("human:", human_path)
    print("event_trigger:", event_trigger_path)
    print("out:", out_path)
    print()

    fieldnames = [
        "wall_now_us",
        "target_wall_us",
        "delay_ms",
        "target_sync",
        "target_sync_source",

        "mvs_latest_sync",
        "mvs_latest_wall_us",

        "human_sync",
        "human_ds",
        "human_dt_ms",
        "human_person_count",

        "event_sync",
        "event_ds",
        "event_dt_ms",

        "ok_sync",
        "ok_time",
        "note",
    ]

    start = time.time()
    first_write = not out_path.exists() or out_path.stat().st_size == 0

    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()

        while True:
            now_us = int(time.time() * 1_000_000)
            target_wall_us = int(now_us - args.delay_ms * 1000.0)

            mvs = read_json_file(mvs_meta_path)
            mvs_sync = extract_sync(mvs) if mvs else None
            mvs_wall = extract_wall_us(mvs) if mvs else None

            human_rows = read_recent_jsonl(human_path)
            event_rows = read_recent_csv(event_trigger_path)

            # 优先用 event_trigger 里最接近 target_wall 的 sync 作为目标 sync
            evt_by_time = nearest_by_time(event_rows, target_wall_us)
            target_sync = evt_by_time.get("sync") if evt_by_time else None
            target_source = "event_trigger_by_time" if target_sync is not None else ""

            # 如果 event_trigger 没有 wall_us，则用最新 MVS sync 按 40ms 估算
            if target_sync is None and mvs_sync is not None:
                target_sync = int(round(mvs_sync - args.delay_ms / 40.0))
                target_source = "mvs_latest_minus_delay_est"

            # 找 human：优先按 sync 找，找不到再按时间找
            human = nearest_by_sync(human_rows, target_sync)
            if human is None:
                human = nearest_by_time(human_rows, target_wall_us)

            # 找 event：优先按 sync 找，找不到用刚才 by_time
            event = nearest_by_sync(event_rows, target_sync)
            if event is None:
                event = evt_by_time

            human_ds, human_dt = row_delta(human, target_sync, target_wall_us)
            event_ds, event_dt = row_delta(event, target_sync, target_wall_us)

            human_pc = human.get("person_count") if human else None

            # 判定阈值：human 允许 +-2 sync；event 尽量 +-1 sync
            ok_sync = (
                target_sync is not None
                and human_ds is not None and abs(human_ds) <= 2
                and event_ds is not None and abs(event_ds) <= 1
            )

            ok_time = (
                (human_dt is None or abs(human_dt) <= 140.0)
                and (event_dt is None or abs(event_dt) <= 80.0)
            )

            notes = []
            if human is None:
                notes.append("no_human_row")
            elif human_pc == 0:
                notes.append("human_row_no_person")
            if event is None:
                notes.append("no_event_trigger_row")
            if target_sync is None:
                notes.append("no_target_sync")
            if human_ds is not None and abs(human_ds) > 2:
                notes.append("human_sync_far")
            if event_ds is not None and abs(event_ds) > 1:
                notes.append("event_sync_far")
            if human_dt is not None and abs(human_dt) > 140.0:
                notes.append("human_time_far")
            if event_dt is not None and abs(event_dt) > 80.0:
                notes.append("event_time_far")

            row = {
                "wall_now_us": now_us,
                "target_wall_us": target_wall_us,
                "delay_ms": args.delay_ms,
                "target_sync": target_sync,
                "target_sync_source": target_source,

                "mvs_latest_sync": mvs_sync,
                "mvs_latest_wall_us": mvs_wall,

                "human_sync": human.get("sync") if human else None,
                "human_ds": human_ds,
                "human_dt_ms": None if human_dt is None else round(human_dt, 3),
                "human_person_count": human_pc,

                "event_sync": event.get("sync") if event else None,
                "event_ds": event_ds,
                "event_dt_ms": None if event_dt is None else round(event_dt, 3),

                "ok_sync": int(bool(ok_sync)),
                "ok_time": int(bool(ok_time)),
                "note": "|".join(notes),
            }

            writer.writerow(row)
            f.flush()

            print(
                f"target_sync={target_sync} "
                f"human={row['human_sync']} ds={human_ds} dt={row['human_dt_ms']}ms pc={human_pc} "
                f"event={row['event_sync']} ds={event_ds} dt={row['event_dt_ms']}ms "
                f"ok_sync={row['ok_sync']} ok_time={row['ok_time']} "
                f"{row['note']}"
            )

            if args.duration > 0 and (time.time() - start) >= args.duration:
                break

            time.sleep(args.interval)

if __name__ == "__main__":
    main()
