#!/usr/bin/env python3
import csv
import glob
import os
import statistics
import sys
from collections import Counter

def to_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default

def to_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def analyze_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.readline()
        f.seek(0)

        has_header = not sample[:1].isdigit()
        if has_header:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        else:
            reader = csv.reader(f)
            for p in reader:
                if len(p) < 18:
                    continue
                rows.append({
                    "wall_us": p[0],
                    "cam_id": p[1],
                    "display_mvs_frame_id": p[2],
                    "display_mvs_sync": p[3],
                    "display_event_sync": p[7],
                    "event_sync_delta": p[8],
                    "event_match_diff_ms": p[9],
                    "human_sync": p[11],
                    "human_sync_delta": p[12],
                    "mvs_meta_status": p[15],
                    "display_delay_age_ms": p[17],
                })

    if len(rows) < 3:
        print(f"\n{path}: not enough rows")
        return

    wall = [to_int(r.get("wall_us")) for r in rows]
    mvs_sync = [to_int(r.get("display_mvs_sync")) for r in rows]
    mvs_fid = [to_int(r.get("display_mvs_frame_id")) for r in rows]
    event_delta = [to_int(r.get("event_sync_delta")) for r in rows]
    human_delta = [to_int(r.get("human_sync_delta")) for r in rows]
    delay_age = [to_float(r.get("display_delay_age_ms")) for r in rows]
    status = [r.get("mvs_meta_status", "") for r in rows]

    valid = [i for i in range(len(rows)) if wall[i] is not None]
    if len(valid) < 3:
        print(f"\n{path}: no valid wall_us")
        return

    total_s = (wall[valid[-1]] - wall[valid[0]]) / 1_000_000.0
    if total_s <= 0:
        total_s = 1e-6

    def changed_count(values):
        c = 0
        last = None
        for v in values:
            if v is None:
                continue
            if last is not None and v != last:
                c += 1
            last = v
        return c

    mvs_sync_changes = changed_count(mvs_sync)
    mvs_fid_changes = changed_count(mvs_fid)

    event_exact = sum(1 for x in event_delta if x == 0)
    event_near = sum(1 for x in event_delta if x is not None and abs(x) <= 1)
    human_exact = sum(1 for x in human_delta if x == 0)
    human_near = sum(1 for x in human_delta if x is not None and abs(x) <= 1)

    full_exact = 0
    full_near = 0
    missing_event = 0
    missing_human = 0

    for ed, hd in zip(event_delta, human_delta):
        if ed is None:
            missing_event += 1
        if hd is None:
            missing_human += 1
        if ed == 0 and hd == 0:
            full_exact += 1
        if ed is not None and hd is not None and abs(ed) <= 1 and abs(hd) <= 1:
            full_near += 1

    # mvs sync jumps
    jumps = []
    last_wall = None
    last_sync = None
    for w, s in zip(wall, mvs_sync):
        if w is None or s is None:
            continue
        if last_wall is not None and last_sync is not None:
            dt_ms = (w - last_wall) / 1000.0
            ds = s - last_sync
            # 25Hz MVS sync means about 1 sync per 40ms.
            expected = dt_ms / 40.0
            extra = ds - expected
            if ds > expected + 3:
                jumps.append((dt_ms, ds, expected, extra))
        last_wall = w
        last_sync = s

    event_counter = Counter(event_delta)
    human_counter = Counter(human_delta)
    status_counter = Counter(status)

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"rows: {len(rows)}")
    print(f"time span: {total_s:.2f}s")
    print(f"MVS sync change rate from audit: {mvs_sync_changes / total_s:.2f}/s")
    print(f"MVS frame_id change rate from audit: {mvs_fid_changes / total_s:.2f}/s")
    print(f"event exact match ratio: {event_exact / len(rows) * 100:.1f}%")
    print(f"event near  match ratio abs(delta)<=1: {event_near / len(rows) * 100:.1f}%")
    print(f"human exact match ratio: {human_exact / len(rows) * 100:.1f}%")
    print(f"human near  match ratio abs(delta)<=1: {human_near / len(rows) * 100:.1f}%")
    print(f"FULL exact composite ratio: {full_exact / len(rows) * 100:.1f}%")
    print(f"FULL near  composite ratio abs(delta)<=1: {full_near / len(rows) * 100:.1f}%")
    print(f"missing event rows: {missing_event}")
    print(f"missing human rows: {missing_human}")
    print(f"MVS jump rows: {len(jumps)}")
    print(f"mvs_meta_status: {dict(status_counter)}")

    if delay_age:
        da = [x for x in delay_age if x is not None]
        if da:
            print(
                f"display_delay_age_ms: "
                f"min={min(da):.1f}, median={statistics.median(da):.1f}, max={max(da):.1f}"
            )

    print(f"top event_sync_delta: {event_counter.most_common(8)}")
    print(f"top human_sync_delta: {human_counter.most_common(8)}")

    if jumps:
        print("sample MVS jumps:")
        for dt_ms, ds, expected, extra in jumps[:8]:
            print(f"  dt={dt_ms:.1f}ms, sync_step={ds}, expected={expected:.1f}, extra={extra:.1f}")

def main():
    sync_dir = sys.argv[1] if len(sys.argv) > 1 else "sync_ipc"
    files = sorted(glob.glob(os.path.join(sync_dir, "display_audit_*.csv")))
    if not files:
        print(f"no display_audit_*.csv found in {sync_dir}")
        return

    for path in files:
        analyze_file(path)

if __name__ == "__main__":
    main()