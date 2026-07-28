#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

sync_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sync_ipc")
cam = sys.argv[2] if len(sys.argv) > 2 else "A"
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1624
h = int(sys.argv[4]) if len(sys.argv) > 4 else 1240
margin = int(sys.argv[5]) if len(sys.argv) > 5 else 100
duration = float(sys.argv[6]) if len(sys.argv) > 6 else 15.0

path = sync_dir / f"overlay_bullet_point_{cam}.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)

points = [
    (margin, margin),
    (w - 1 - margin, margin),
    (w - 1 - margin, h - 1 - margin),
    (margin, h - 1 - margin),
    (margin, margin),
]

run_id = "ue5_inset_box_loop_test"
bullet_id = int(time.time()) % 100000 + 980000
seq = 0

print(f"write LOOP inset box to {path} cam={cam} size={w}x{h} margin={margin} duration={duration}s bullet_id={bullet_id}")

end_t = time.time() + duration
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
        time.sleep(0.2)

print("done")
