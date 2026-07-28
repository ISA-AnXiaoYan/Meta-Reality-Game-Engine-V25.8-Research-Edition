#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import statistics
from pathlib import Path

event_csv = Path("./sync_logs/event_trigger_D.csv")
mvs_csv = Path("./sync_logs/mvs_frame_DB0664083.csv")

event_by_sync = {}
event_rows = []
with event_csv.open("r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        si = int(r["sync_index"])
        if si >= 0:
            event_by_sync[si] = r
            event_rows.append(r)

mvs_rows = []
with mvs_csv.open("r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        mvs_rows.append(r)

paired = []
mvs_without_event = []
for m in mvs_rows:
    mvs_frame_num = int(m["mvs_frame_num"])

    # 严格按硬件帧号对应：frame_num=1 -> sync_index=0
    expected_sync_index = mvs_frame_num - 1

    e = event_by_sync.get(expected_sync_index)
    if e is None:
        mvs_without_event.append((m["frame_index"], mvs_frame_num, expected_sync_index))
        continue

    paired.append({
        "mvs_frame_index": int(m["frame_index"]),
        "mvs_frame_num": mvs_frame_num,
        "expected_sync_index": expected_sync_index,
        "event_ts_us": int(e["event_ts_us"]),
        "mvs_wall_time_us": int(m["wall_time_us"]),
        "event_wall_time_us": int(e["wall_time_us"]),
        "wall_diff_us": int(m["wall_time_us"]) - int(e["wall_time_us"]),
    })

# 检查 MVS SDK 帧号跳跃
frame_nums = [int(r["mvs_frame_num"]) for r in mvs_rows]
gaps = []
for a, b in zip(frame_nums[:-1], frame_nums[1:]):
    if b != a + 1:
        gaps.append((a, b, b - a - 1))

# 检查 event trigger 是否有 MVS 图像对应
mvs_frame_nums_set = set(frame_nums)
event_without_mvs = []
for e in event_rows:
    expected_mvs_frame_num = int(e["sync_index"]) + 1
    if expected_mvs_frame_num not in mvs_frame_nums_set:
        event_without_mvs.append((e["sync_index"], expected_mvs_frame_num, e["event_ts_us"]))

print("event selected trigger count =", len(event_rows))
print("mvs received rows            =", len(mvs_rows))
print("paired rows                  =", len(paired))
print("mvs_without_event rows        =", len(mvs_without_event))
print("event_without_mvs rows        =", len(event_without_mvs))
print("mvs frame_num gaps            =", len(gaps))

if paired:
    wall_diffs = [r["wall_diff_us"] for r in paired]
    print("paired wall_diff median us    =", statistics.median(wall_diffs))
    print("paired wall_diff min/max us   =", min(wall_diffs), max(wall_diffs))

print("\nfirst 5 paired:")
for r in paired[:5]:
    print(r)

print("\nlast 5 paired:")
for r in paired[-5:]:
    print(r)

print("\nfirst 10 mvs frame_num gaps:")
for g in gaps[:10]:
    print("frame_num jump:", g[0], "->", g[1], "missing", g[2], "frames")

print("\nfirst 20 event_without_mvs:")
for x in event_without_mvs[:20]:
    print("sync_index, expected_mvs_frame_num, event_ts_us =", x)

print("\nfirst 20 mvs_without_event:")
for x in mvs_without_event[:20]:
    print("mvs_frame_index, mvs_frame_num, expected_sync_index =", x)
