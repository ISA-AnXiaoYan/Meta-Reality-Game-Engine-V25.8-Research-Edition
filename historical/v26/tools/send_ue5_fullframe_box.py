#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

sync_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sync_ipc")
cam = sys.argv[2] if len(sys.argv) > 2 else "A"
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1624
h = int(sys.argv[4]) if len(sys.argv) > 4 else 1240

path = sync_dir / f"overlay_bullet_point_{cam}.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)

points = [
    (0, 0),
    (w - 1, 0),
    (w - 1, h - 1),
    (0, h - 1),
    (0, 0),
]

run_id = "ue5_fullframe_box_test"
bullet_id = int(time.time()) % 100000 + 900000
base_ts = int(time.time() * 1_000_000)

print(f"write fullframe box to {path} cam={cam} size={w}x{h} bullet_id={bullet_id}")

with path.open("a", encoding="utf-8") as f:
    for i in range(1, len(points)):
        px, py = points[i - 1]
        x, y = points[i]
        obj = {
            "type": "overlay_bullet_point",
            "sync_run_id": run_id,
            "pair_id": cam,
            "camera_alias": cam,
            "bullet_id": bullet_id,
            "bullet_seq_id": i,
            "point_index": i,
            "point_ts_us": base_ts + i * 1000,
            "x": x,
            "y": y,
            "prev_valid": True,
            "prev_x": px,
            "prev_y": py,
            "prev_point_index": i - 1,
            "prev_point_ts_us": base_ts + (i - 1) * 1000,
            "display_width": w,
            "display_height": h,
            "confidence": 1.0,
            "status": "tracking"
        }
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        time.sleep(0.08)

print("done")
