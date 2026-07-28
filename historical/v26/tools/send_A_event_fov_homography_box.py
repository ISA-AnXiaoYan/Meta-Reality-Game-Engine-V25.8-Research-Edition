#!/usr/bin/env python3
import json
import time
from pathlib import Path

cam = "A"
w = 1624
h = 1240
path = Path("sync_ipc") / f"overlay_bullet_point_{cam}.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)

# A 路 event 1280x720 四角经过你提供的 flight homography 映射到 MVS 后的坐标
points = [
    (118.04, 227.48),    # LT
    (1497.64, 211.76),   # RT
    (1508.76, 992.72),   # RB
    (126.48, 1004.34),   # LB
    (118.04, 227.48),    # close
]

bullet_id = int(time.time()) % 100000 + 990000
base_ts = int(time.time() * 1_000_000)

print(f"write A event-FOV homography box to {path}, bullet_id={bullet_id}")

with path.open("a", encoding="utf-8") as f:
    for i in range(1, len(points)):
        px, py = points[i - 1]
        x, y = points[i]
        obj = {
            "type": "overlay_bullet_point",
            "sync_run_id": "A_event_fov_homography_test",
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
