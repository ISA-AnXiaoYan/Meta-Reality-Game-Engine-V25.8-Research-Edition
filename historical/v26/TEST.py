
import json
import numpy as np
import cv2
from pathlib import Path

event_path = Path("/home/ysxq/PycharmProjects/Ids_Test_3.9/test607/calib/E/event_flight.json")
mvs_path = Path("/home/ysxq/PycharmProjects/Ids_Test_3.9/test607/calib/E/mvs_flight.json")
out_path = Path("/home/ysxq/PycharmProjects/Ids_Test_3.9/test607/calib/E/homography_flight_C.json")

def load_points(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        pts = data.get("points", [])
    else:
        pts = data

    out = []
    for p in pts:
        if isinstance(p, dict):
            x = p.get("x", p.get("px", p.get("u")))
            y = p.get("y", p.get("py", p.get("v")))
        else:
            x, y = p[0], p[1]
        out.append([float(x), float(y)])

    return np.asarray(out, dtype=np.float32)

src = load_points(event_path)
dst = load_points(mvs_path)

print("event points:", src.shape)
print("mvs points:  ", dst.shape)

if len(src) != len(dst):
    raise SystemExit("ERROR: event 和 mvs 点数量不一致")
if len(src) < 4:
    raise SystemExit("ERROR: 至少需要 4 对点")

H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

if H is None:
    raise SystemExit("ERROR: Homography 计算失败，请检查点是否对应")

mask = mask.ravel().astype(bool)

src_h = cv2.convertPointsToHomogeneous(src).reshape(-1, 3).T
proj = H @ src_h
proj = (proj[:2] / proj[2]).T

err = np.linalg.norm(proj - dst, axis=1)

inlier_err = err[mask]

print("\n========== Homography C flight ==========")
print(H)

print("\n========== Error ==========")
print("total points:", len(src))
print("inliers:     ", int(mask.sum()))
print("outliers:    ", int((~mask).sum()))

if len(inlier_err) > 0:
    print("inlier mean px:  ", float(np.mean(inlier_err)))
    print("inlier median px:", float(np.median(inlier_err)))
    print("inlier max px:   ", float(np.max(inlier_err)))

print("\n========== Per point ==========")
for i, (e, ok) in enumerate(zip(err, mask), start=1):
    print(f"{i:02d}: error={e:8.3f}px  {'OK' if ok else 'OUTLIER'}")

result = {
    "camera": "C",
    "plane": "flight",
    "source": "event_to_mvs",
    "event_points_file": str(event_path),
    "mvs_points_file": str(mvs_path),
    "homography_event_to_mvs": H.tolist(),
    "ransac_reproj_threshold_px": 5.0,
    "total_points": int(len(src)),
    "inliers": int(mask.sum()),
    "outliers": int((~mask).sum()),
    "error_px": {
        "all": err.tolist(),
        "inlier_mean": float(np.mean(inlier_err)) if len(inlier_err) else None,
        "inlier_median": float(np.median(inlier_err)) if len(inlier_err) else None,
        "inlier_max": float(np.max(inlier_err)) if len(inlier_err) else None,
    },
    "inlier_mask": mask.astype(int).tolist(),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\nSaved:", out_path)
