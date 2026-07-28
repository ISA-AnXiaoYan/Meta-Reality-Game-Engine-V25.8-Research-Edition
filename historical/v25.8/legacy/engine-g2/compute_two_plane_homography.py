#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_two_plane_homography.py

用途：
  从 pick_calibration_points.py 产生的点集 JSON 中，分别计算：
    1) 地面平面 H_ground
    2) 子弹常用飞行高度平面 H_flight
  两个 event -> MVS 的 Homography，并输出可直接复制到 rentijiance 配置里的 JSON 片段。

点集要求：
  同一个平面内，event 和 mvs 两边按同样顺序点击对应点。
  例如 event_ground 第 1 个点，对应 mvs_ground 第 1 个点。

示例：
  python compute_two_plane_homography.py \
    --camera A \
    --event-ground calib/A/event_ground.json \
    --mvs-ground   calib/A/mvs_ground.json \
    --event-flight calib/A/event_flight.json \
    --mvs-flight   calib/A/mvs_flight.json \
    --out calib/A/homography_two_plane_A.json

输出：
  - 终端打印 ground/flight 的重投影误差
  - --out 指定的 JSON 文件
  - 可复制进 hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json 的 event_overlay.per_route.A 片段
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _points_from_pick_json(path: str | Path) -> np.ndarray:
    obj = _read_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("points"), list):
        pts = obj["points"]
        arr = []
        for p in pts:
            if isinstance(p, dict):
                arr.append([float(p["x"]), float(p["y"])])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                arr.append([float(p[0]), float(p[1])])
        return np.asarray(arr, dtype=np.float32)
    if isinstance(obj, list):
        return np.asarray([[float(p[0]), float(p[1])] for p in obj], dtype=np.float32)
    raise ValueError(f"无法从 {path} 读取点；需要 pick_calibration_points.py 输出格式或 [[x,y], ...]")


def _compute_h(src_event: np.ndarray, dst_mvs: np.ndarray, ransac_px: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if src_event.ndim != 2 or dst_mvs.ndim != 2 or src_event.shape[1] != 2 or dst_mvs.shape[1] != 2:
        raise ValueError("点必须是 Nx2")
    if len(src_event) != len(dst_mvs):
        raise ValueError(f"event 点数量 {len(src_event)} != mvs 点数量 {len(dst_mvs)}")
    if len(src_event) < 4:
        raise ValueError("Homography 至少需要 4 对点，建议 12-20 对")

    if len(src_event) == 4:
        H = cv2.getPerspectiveTransform(src_event.astype(np.float32), dst_mvs.astype(np.float32))
        mask = np.ones((len(src_event),), dtype=bool)
    else:
        H, raw_mask = cv2.findHomography(
            src_event.astype(np.float32),
            dst_mvs.astype(np.float32),
            cv2.RANSAC,
            float(ransac_px),
        )
        if H is None:
            raise RuntimeError("cv2.findHomography failed；请检查对应点顺序或是否混入太多错误点")
        mask = raw_mask.ravel().astype(bool) if raw_mask is not None else np.ones((len(src_event),), dtype=bool)

    H = H / H[2, 2]
    projected = cv2.perspectiveTransform(src_event.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(projected - dst_mvs, axis=1)
    return H, mask, err


def _pack_matrix(H: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in H]


def _report(name: str, ev: np.ndarray, mv: np.ndarray, H: np.ndarray, mask: np.ndarray, err: np.ndarray) -> Dict[str, Any]:
    inlier_err = err[mask] if np.any(mask) else err
    print("\n" + "=" * 70)
    print(f"[{name}] point_pairs={len(ev)} inliers={int(mask.sum())}/{len(mask)}")
    print(f"[{name}] error_all_px    mean={float(err.mean()):.3f} median={float(np.median(err)):.3f} max={float(err.max()):.3f}")
    print(f"[{name}] error_inlier_px mean={float(inlier_err.mean()):.3f} median={float(np.median(inlier_err)):.3f} max={float(inlier_err.max()):.3f}")
    print(f"[{name}] H =")
    print(json.dumps(_pack_matrix(H), ensure_ascii=False, indent=2))
    for i, (src, dst, e, ok) in enumerate(zip(ev, mv, err, mask), start=1):
        print(f"  #{i:02d} event=({src[0]:.1f},{src[1]:.1f}) -> mvs=({dst[0]:.1f},{dst[1]:.1f}) err={e:.2f} {'OK' if ok else 'OUTLIER'}")
    return {
        "matrix": _pack_matrix(H),
        "pairs": int(len(ev)),
        "inliers": int(mask.sum()),
        "error_all_px": {
            "mean": float(err.mean()),
            "median": float(np.median(err)),
            "max": float(err.max()),
        },
        "error_inlier_px": {
            "mean": float(inlier_err.mean()),
            "median": float(np.median(inlier_err)),
            "max": float(inlier_err.max()),
        },
        "outlier_indices_1based": [int(i + 1) for i, ok in enumerate(mask) if not bool(ok)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="A", help="路线/相机编号，例如 A/B/C/D/E")
    ap.add_argument("--event-ground", required=True)
    ap.add_argument("--mvs-ground", required=True)
    ap.add_argument("--event-flight", required=True)
    ap.add_argument("--mvs-flight", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ransac-reproj-threshold", type=float, default=4.0, help="RANSAC 重投影阈值，单位像素")
    args = ap.parse_args()

    ev_g = _points_from_pick_json(args.event_ground)
    mv_g = _points_from_pick_json(args.mvs_ground)
    ev_f = _points_from_pick_json(args.event_flight)
    mv_f = _points_from_pick_json(args.mvs_flight)

    H_g, mask_g, err_g = _compute_h(ev_g, mv_g, args.ransac_reproj_threshold)
    H_f, mask_f, err_f = _compute_h(ev_f, mv_f, args.ransac_reproj_threshold)

    ground = _report("ground", ev_g, mv_g, H_g, mask_g, err_g)
    flight = _report("flight", ev_f, mv_f, H_f, mask_f, err_f)

    payload: Dict[str, Any] = {
        "camera": str(args.camera),
        "homographies_event_to_mvs": {
            "ground": ground,
            "flight": flight,
        },
        "event_overlay_per_route_snippet": {
            str(args.camera): {
                "enabled": True,
                "active_homography": "flight",
                "homographies_event_to_mvs": {
                    "ground": ground["matrix"],
                    "flight": flight["matrix"],
                },
                "opacity": 1.0,
                "black_threshold": 8,
                "mask_blur": 3,
                "reuse_last_frame": True,
                "max_stale_ms": 250,
            }
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"已保存: {out}")
    print("\n把下面片段合并到 rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json 的 event_overlay.per_route 里：")
    print(json.dumps(payload["event_overlay_per_route_snippet"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
