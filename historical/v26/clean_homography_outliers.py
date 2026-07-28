#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hp = np.hstack([pts, ones]) @ H.T
    hp = hp[:, :2] / hp[:, 2:3]
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--root", default="calib_old")
    ap.add_argument("--plane", default="flight")
    ap.add_argument("--threshold", type=float, default=4.0)
    args = ap.parse_args()

    cam = args.cam.upper()
    root = Path(args.root) / cam

    src = root / f"points_{args.plane}_pair.json"
    dst = root / f"points_{args.plane}_pair_clean.json"
    report = root / f"outliers_{args.plane}.txt"

    obj = json.loads(src.read_text(encoding="utf-8"))
    event_pts = np.asarray(obj["event_points"], dtype=np.float64)
    mvs_pts = np.asarray(obj["mvs_points"], dtype=np.float64)

    H, mask = cv2.findHomography(
        event_pts,
        mvs_pts,
        cv2.RANSAC,
        args.threshold
    )

    if H is None or mask is None:
        raise SystemExit(f"[{cam}] findHomography failed")

    mask = mask.reshape(-1).astype(bool)
    proj = project_points(H, event_pts)
    errs = np.linalg.norm(proj - mvs_pts, axis=1)

    keep_event = []
    keep_mvs = []
    removed = []
    lines = []

    lines.append(f"camera={cam}")
    lines.append(f"threshold={args.threshold}")
    lines.append(f"point_pairs={len(event_pts)} inliers={int(mask.sum())}/{len(mask)}")
    lines.append("")

    for i, (ep, mp, pp, err, ok) in enumerate(zip(event_pts, mvs_pts, proj, errs, mask), start=1):
        status = "inlier" if ok else "outlier"
        lines.append(
            f"#{i:02d} event=({ep[0]:.1f},{ep[1]:.1f}) "
            f"mvs=({mp[0]:.1f},{mp[1]:.1f}) "
            f"projected=({pp[0]:.1f},{pp[1]:.1f}) "
            f"err={err:.3f} {status}"
        )

        if ok:
            keep_event.append([float(ep[0]), float(ep[1])])
            keep_mvs.append([float(mp[0]), float(mp[1])])
        else:
            removed.append(i)

    out = {
        "event_points": keep_event,
        "mvs_points": keep_mvs,
        "removed_outliers_1based": removed,
        "source": str(src),
        "ransac_reproj_threshold": args.threshold
    }

    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{cam}] source pairs: {len(event_pts)}")
    print(f"[{cam}] kept pairs: {len(keep_event)}")
    print(f"[{cam}] removed outliers: {removed}")
    print(f"[{cam}] wrote: {dst}")
    print(f"[{cam}] report: {report}")


if __name__ == "__main__":
    main()
