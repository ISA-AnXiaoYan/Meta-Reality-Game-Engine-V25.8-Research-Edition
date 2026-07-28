#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute event-camera -> MVS-camera homography from corresponding image points.

Input JSON example:
{
  "event_points": [[100,120], [1100,130], [1120,650], [90,640]],
  "mvs_points":   [[230,310], [1390,300], [1410,1030], [210,1040]]
}

Usage:
  python compute_event_mvs_homography.py --points points_A.json

Copy the printed homography_event_to_mvs matrix into:
  rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json
  -> event_overlay -> per_route -> A/B/C/D/...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ev = obj.get("event_points", obj.get("src", obj.get("event")))
    mv = obj.get("mvs_points", obj.get("dst", obj.get("mvs")))
    if ev is None or mv is None:
        raise ValueError("JSON must contain event_points and mvs_points")
    ev_np = np.asarray(ev, dtype=np.float32)
    mv_np = np.asarray(mv, dtype=np.float32)
    if ev_np.ndim != 2 or mv_np.ndim != 2 or ev_np.shape[1] != 2 or mv_np.shape[1] != 2:
        raise ValueError("points must be Nx2 arrays")
    if len(ev_np) != len(mv_np) or len(ev_np) < 4:
        raise ValueError("need the same number of event/MVS points, at least 4 pairs")
    return ev_np, mv_np


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute event->MVS homography")
    ap.add_argument("--points", required=True, help="JSON file with event_points and mvs_points")
    ap.add_argument("--ransac-reproj-threshold", type=float, default=3.0)
    args = ap.parse_args()

    event_pts, mvs_pts = _load_points(Path(args.points))
    if len(event_pts) == 4:
        H = cv2.getPerspectiveTransform(event_pts, mvs_pts)
        inlier_mask = np.ones((4,), dtype=bool)
    else:
        H, mask = cv2.findHomography(event_pts, mvs_pts, cv2.RANSAC, float(args.ransac_reproj_threshold))
        if H is None:
            raise RuntimeError("cv2.findHomography failed")
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones((len(event_pts),), dtype=bool)

    H = H / H[2, 2]
    projected = cv2.perspectiveTransform(event_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(projected - mvs_pts, axis=1)

    print("homography_event_to_mvs:")
    print(json.dumps([[float(x) for x in row] for row in H], ensure_ascii=False, indent=2))
    print(f"point_pairs={len(event_pts)} inliers={int(inlier_mask.sum())}/{len(inlier_mask)}")
    print(f"reprojection_error_px mean={float(err.mean()):.3f} median={float(np.median(err)):.3f} max={float(err.max()):.3f}")
    for i, (src, dst, prj, e, ok) in enumerate(zip(event_pts, mvs_pts, projected, err, inlier_mask), start=1):
        mark = "inlier" if ok else "outlier"
        print(f"#{i:02d} event=({src[0]:.1f},{src[1]:.1f}) mvs=({dst[0]:.1f},{dst[1]:.1f}) projected=({prj[0]:.1f},{prj[1]:.1f}) err={e:.2f} {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
