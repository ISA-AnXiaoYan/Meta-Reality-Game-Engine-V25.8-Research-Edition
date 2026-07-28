#!/usr/bin/env python3
import json
import time
import sys
from pathlib import Path

cam = "A"
w = 1624
h = 1240

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
sleep_between_loops = 0.25

path = Path("sync_ipc") / f"overlay_bullet_point_{cam}.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)

# A 路 event 1280x720 四角经过 flight homography 映射到 MVS 后的坐标
points = [
    (118.04, 227.48),    # LT
    (1497.64, 211.76),   # RT
    (1508.76, 992.72),   # RB
    (126.48, 1004.34),   # LB
    (118.04, 227.48),    # close
]

run_id = "A_event_fov_homography_loop_test"
bullet_id = int(time.time()) % 100000 + 990000
seq = 0
end_t = time.time() + duration

print(f"write LOOP A event-FOV homography box to {path}")
print(f"duration={duration}s bullet_id={bullet_id}")

with path.open("a", encoding="utf-8") as f:
    while time.time() < end_t:
        base_ts = int(time.time() * 1_000_000)

        for i in range(1, len(points)):
            seq += 1
            px, py = points[i - 1]
            x, y = points[i]

            obj = {
                "type": "overlay_bullet_point",
                "sync_run_id": run_id,
                "pair_id": cam,
                "camera_alias": cam,
                "bullet_id": bullet_id,
                "bullet_seq_id": seq,
                "point_index": seq,
                "point_ts_us": base_ts + seq * 1000,
                "x": x,
                "y": y,
                "prev_valid": True,
                "prev_x": px,
                "prev_y": py,
                "prev_point_index": seq - 1,
                "prev_point_ts_us": base_ts + (seq - 1) * 1000,
                "display_width": w,
                "display_height": h,
                "confidence": 1.0,
                "status": "tracking"
            }

            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            time.sleep(0.05)

        time.sleep(sleep_between_loops)

print("done")
