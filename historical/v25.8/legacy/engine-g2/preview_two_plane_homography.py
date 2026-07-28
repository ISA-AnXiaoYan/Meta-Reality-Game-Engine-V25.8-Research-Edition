#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_two_plane_homography.py

用途：
  用两张截图预览 ground/flight 两个 Homography 的叠加效果。

示例：
  python preview_two_plane_homography.py \
    --event-image calib/A/event_snapshot.png \
    --mvs-image calib/A/mvs_snapshot.png \
    --homography calib/A/homography_two_plane_A.json

按键：
  g：看 ground 对齐
  f：看 flight 对齐
  +/-：调事件层透明度
  s：保存当前预览图
  q/ESC：退出
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_hs(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    hs = obj.get("homographies_event_to_mvs", {})
    # compute_two_plane_homography.py 输出格式
    if isinstance(hs.get("ground"), dict):
        return {
            "ground": np.asarray(hs["ground"]["matrix"], dtype=np.float64),
            "flight": np.asarray(hs["flight"]["matrix"], dtype=np.float64),
        }
    # 简化格式
    return {
        "ground": np.asarray(hs["ground"], dtype=np.float64),
        "flight": np.asarray(hs["flight"], dtype=np.float64),
    }


def _build_alpha(event_bgr: np.ndarray, threshold: int = 8) -> np.ndarray:
    gray = np.max(event_bgr[:, :, :3], axis=2).astype(np.uint8)
    alpha = (gray.astype(np.float32) - float(threshold)) / max(1.0, 255.0 - float(threshold))
    return np.clip(alpha, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-image", required=True)
    ap.add_argument("--mvs-image", required=True)
    ap.add_argument("--homography", required=True)
    ap.add_argument("--opacity", type=float, default=0.75)
    ap.add_argument("--black-threshold", type=int, default=8)
    ap.add_argument("--display-max-width", type=int, default=1400)
    ap.add_argument("--display-max-height", type=int, default=900)
    args = ap.parse_args()

    event = cv2.imread(args.event_image, cv2.IMREAD_COLOR)
    mvs = cv2.imread(args.mvs_image, cv2.IMREAD_COLOR)
    if event is None:
        raise RuntimeError(f"无法读取事件图: {args.event_image}")
    if mvs is None:
        raise RuntimeError(f"无法读取 MVS 图: {args.mvs_image}")

    hs = _load_hs(args.homography)
    active = "flight"
    opacity = float(args.opacity)
    h, w = mvs.shape[:2]

    while True:
        H = hs[active]
        warped = cv2.warpPerspective(event, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        alpha = cv2.warpPerspective(_build_alpha(event, args.black_threshold), H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = np.clip(alpha * opacity, 0.0, 1.0)[:, :, None]
        out = np.clip(warped.astype(np.float32) * a + mvs.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
        cv2.putText(out, f"plane={active} opacity={opacity:.2f} | g=ground f=flight +/- opacity s=save q=quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

        scale = min(float(args.display_max_width) / max(1, w), float(args.display_max_height) / max(1, h), 1.0)
        vis = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else out
        cv2.imshow("two-plane homography preview", vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('g'):
            active = "ground"
        elif key == ord('f'):
            active = "flight"
        elif key in (ord('+'), ord('=')):
            opacity = min(1.0, opacity + 0.05)
        elif key in (ord('-'), ord('_')):
            opacity = max(0.0, opacity - 0.05)
        elif key == ord('s'):
            out_path = Path(args.homography).with_name(f"preview_{active}.png")
            cv2.imwrite(str(out_path), out)
            print(f"saved {out_path}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
