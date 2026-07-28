#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_calibration_points.py

用途：
  打开一张图片、OpenCV 相机、海康 MVS 相机，或读取你工程里的事件相机共享内存画面；
  鼠标左键点击一次，立即保存一个原始图像坐标点，方便后续 event -> MVS 对应点标定。

常用示例：
  # 1) 从事件相机共享内存取点。先启动你的事件相机程序，让它发布 event_overlay_latest_A。
  python pick_calibration_points.py --source shm --shm-name event_overlay_latest_A --name event_A --out calib/event_A_points

  # 2) 打开海康 MVS 相机取点。serial 可不填，不填会打开枚举到的第一台。
  python pick_calibration_points.py --source hik --serial 你的MVS序列号 --name mvs_A --out calib/mvs_A_points

  # 3) 从截图取点，最稳定。
  python pick_calibration_points.py --source image --image event_A.png --name event_A --out calib/event_A_points
  python pick_calibration_points.py --source image --image mvs_A.png   --name mvs_A   --out calib/mvs_A_points

操作：
  左键：保存一个点
  右键：撤销最后一个点
  空格：冻结/恢复实时画面
  s：保存当前画面截图
  c：清空当前点
  q / ESC：退出

输出：
  --out xxx 会生成 xxx.json 和 xxx.csv。
  注意：保存的是原始图像坐标，不是窗口缩放后的坐标。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Dict[str, Any]


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="milliseconds")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def out_paths(out_prefix: str) -> Tuple[Path, Path]:
    p = Path(out_prefix)
    if p.suffix.lower() in [".json", ".csv"]:
        p = p.with_suffix("")
    return p.with_suffix(".json"), p.with_suffix(".csv")


def load_points(json_path: Path) -> List[Point]:
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pts = data.get("points", data if isinstance(data, list) else [])
        return list(pts) if isinstance(pts, list) else []
    except Exception:
        return []


def save_points(points: List[Point], json_path: Path, csv_path: Path, *, name: str, source: str) -> None:
    ensure_parent(json_path)
    ensure_parent(csv_path)
    payload = {
        "name": name,
        "source": source,
        "updated_at": now_iso(),
        "count": len(points),
        "points": points,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "time", "name", "source", "note"])
        for p in points:
            writer.writerow([
                p.get("id", ""), p.get("x", ""), p.get("y", ""),
                p.get("time", ""), p.get("name", ""), p.get("source", ""), p.get("note", ""),
            ])


def auto_scale_for_display(frame: np.ndarray, max_width: int, max_height: int, user_scale: float) -> float:
    if user_scale > 0:
        return float(user_scale)
    h, w = frame.shape[:2]
    s = min(float(max_width) / max(1, w), float(max_height) / max(1, h), 1.0)
    return max(s, 0.05)


def draw_points(frame: np.ndarray, points: List[Point], scale: float, title: str, frozen: bool) -> np.ndarray:
    vis = frame.copy()
    for p in points:
        x, y = int(p["x"]), int(p["y"])
        cv2.circle(vis, (x, y), 6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(p["id"]), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    status = f"{title} | points={len(points)} | left=save right=undo space=freeze s=screenshot q=quit"
    if frozen:
        status += " | FROZEN"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 1350), 34), (0, 0, 0), -1)
    cv2.putText(vis, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    if abs(scale - 1.0) > 1e-6:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return vis


class ImageSource:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.frame is None:
            raise RuntimeError(f"无法读取图片: {image_path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return True, self.frame.copy()

    def close(self) -> None:
        pass


class VideoSource:
    def __init__(self, index: int, width: int = 0, height: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开 OpenCV 相机 index={index}")
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        return bool(ok), frame if ok else None

    def close(self) -> None:
        self.cap.release()


class ShmSource:
    def __init__(self, shm_name: str, project_dir: str):
        project_dir = str(project_dir or ".")
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        try:
            from latest_frame_shm import SharedFrameReader  # type: ignore
        except Exception as e:
            raise RuntimeError(f"无法导入 latest_frame_shm.py，请确认 --project-dir 指向 test607 目录。原始错误: {e}")
        self.reader = SharedFrameReader(shm_name)
        self.last_frame: Optional[np.ndarray] = None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        item = self.reader.read_latest()
        if item is not None and item.get("frame") is not None:
            frame = item["frame"]
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
            self.last_frame = frame.copy()
        if self.last_frame is not None:
            return True, self.last_frame.copy()
        return False, None

    def close(self) -> None:
        try:
            self.reader.close()
        except Exception:
            pass


class HikSource:
    def __init__(self, serial: str, project_dir: str, mvs_sdk_path: str, fps: float = 25.0):
        # 直接复用你工程里已经写好的 HikFullViewSource，避免另写一套 MVS SDK 代码。
        project_dir = str(project_dir or ".")
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        renti_dir = os.path.join(project_dir, "rentijiance")
        if renti_dir not in sys.path:
            sys.path.insert(0, renti_dir)
        try:
            mod = importlib.import_module("hik_rgb_yolo_seg_multicam_same_model_stableid_v5")
            HikFullViewSource = getattr(mod, "HikFullViewSource")
        except Exception as e:
            raise RuntimeError(
                "无法从 rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py "
                f"导入 HikFullViewSource。原始错误: {type(e).__name__}: {e}"
            )
        self.cam = HikFullViewSource(
            serial=str(serial or ""),
            mvs_sdk_path=mvs_sdk_path,
            timeout_ms=1000,
            width=0,
            height=0,
            fps=float(fps),
            force_full_sensor=True,
            prefer_bgr8=False,
            sync_cfg={},
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame, _ts = self.cam.read()
        return bool(ok), frame if ok else None

    def close(self) -> None:
        self.cam.release()


def build_source(args: argparse.Namespace):
    if args.source == "image":
        if not args.image:
            raise SystemExit("--source image 需要指定 --image")
        return ImageSource(args.image)
    if args.source == "video":
        return VideoSource(args.index, args.width, args.height)
    if args.source == "shm":
        if not args.shm_name:
            raise SystemExit("--source shm 需要指定 --shm-name，例如 event_overlay_latest_A")
        return ShmSource(args.shm_name, args.project_dir)
    if args.source == "hik":
        return HikSource(args.serial, args.project_dir, args.mvs_sdk_path, args.fps)
    raise SystemExit(f"未知 source: {args.source}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["image", "video", "shm", "hik"], required=True)
    ap.add_argument("--name", default="camera", help="点集名称，例如 event_A 或 mvs_A")
    ap.add_argument("--out", required=True, help="输出前缀，例如 calib/event_A_points；会生成 .json 和 .csv")

    ap.add_argument("--image", default="", help="--source image 使用")
    ap.add_argument("--index", type=int, default=0, help="--source video 使用的 OpenCV 相机编号")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)

    ap.add_argument("--shm-name", default="", help="--source shm 使用，例如 event_overlay_latest_A")
    ap.add_argument("--project-dir", default=os.getcwd(), help="你的 test607 工程目录，默认当前目录")

    ap.add_argument("--serial", default="", help="--source hik 使用；不填则打开第一台 MVS")
    ap.add_argument("--mvs-sdk-path", default="/opt/MVS/Samples/64/Python/MvImport")
    ap.add_argument("--fps", type=float, default=25.0)

    ap.add_argument("--display-scale", type=float, default=0.0, help="显示缩放。0=按 max 自动缩小")
    ap.add_argument("--display-max-width", type=int, default=1400)
    ap.add_argument("--display-max-height", type=int, default=900)
    ap.add_argument("--load-existing", action="store_true", help="启动时加载已有点继续追加")
    args = ap.parse_args()

    json_path, csv_path = out_paths(args.out)
    points: List[Point] = load_points(json_path) if args.load_existing else []
    next_id = 1 + max([int(p.get("id", 0)) for p in points] or [0])

    src = build_source(args)
    window = f"pick_points: {args.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state: Dict[str, Any] = {
        "scale": 1.0,
        "last_frame": None,
        "frozen_frame": None,
        "frozen": False,
        "next_id": next_id,
    }

    def on_mouse(event, x_disp, y_disp, flags, userdata):
        nonlocal points
        frame = state.get("frozen_frame") if state.get("frozen") else state.get("last_frame")
        if frame is None:
            return
        scale = float(state.get("scale", 1.0) or 1.0)
        x = int(round(x_disp / scale))
        y = int(round(y_disp / scale))
        h, w = frame.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        if event == cv2.EVENT_LBUTTONDOWN:
            pid = int(state["next_id"])
            state["next_id"] = pid + 1
            p: Point = {
                "id": pid,
                "x": x,
                "y": y,
                "time": now_iso(),
                "name": args.name,
                "source": args.source,
                "note": "",
            }
            points.append(p)
            save_points(points, json_path, csv_path, name=args.name, source=args.source)
            print(f"[SAVE] {args.name} #{pid}: x={x}, y={y} -> {json_path} / {csv_path}", flush=True)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                p = points.pop()
                save_points(points, json_path, csv_path, name=args.name, source=args.source)
                print(f"[UNDO] removed #{p.get('id')}: x={p.get('x')}, y={p.get('y')}", flush=True)

    cv2.setMouseCallback(window, on_mouse)

    print("操作：左键保存点；右键撤销；空格冻结/恢复；s 保存截图；c 清空；q/ESC 退出。", flush=True)
    print(f"输出：{json_path} 和 {csv_path}", flush=True)

    try:
        while True:
            if state["frozen"] and state["frozen_frame"] is not None:
                frame = state["frozen_frame"].copy()
                ok = True
            else:
                ok, frame = src.read()
                if ok and frame is not None:
                    state["last_frame"] = frame.copy()

            if not ok or frame is None:
                blank = np.zeros((360, 900, 3), dtype=np.uint8)
                cv2.putText(blank, "waiting for frame...", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                frame = blank

            scale = auto_scale_for_display(frame, args.display_max_width, args.display_max_height, args.display_scale)
            state["scale"] = scale
            vis = draw_points(frame, points, scale, args.name, bool(state["frozen"]))
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord(" "):
                if state["frozen"]:
                    state["frozen"] = False
                    state["frozen_frame"] = None
                    print("[LIVE] 恢复实时画面", flush=True)
                else:
                    base = state.get("last_frame")
                    if base is not None:
                        state["frozen"] = True
                        state["frozen_frame"] = base.copy()
                        print("[FREEZE] 已冻结当前画面，便于取点", flush=True)
            elif key == ord("s"):
                frame_to_save = state.get("frozen_frame") if state.get("frozen") else state.get("last_frame")
                if frame_to_save is not None:
                    snap = Path(args.out).with_suffix("")
                    snap_path = snap.parent / f"{snap.name}_screenshot_{int(time.time())}.png"
                    ensure_parent(snap_path)
                    cv2.imwrite(str(snap_path), frame_to_save)
                    print(f"[SCREENSHOT] {snap_path}", flush=True)
            elif key == ord("c"):
                points = []
                state["next_id"] = 1
                save_points(points, json_path, csv_path, name=args.name, source=args.source)
                print("[CLEAR] 已清空点", flush=True)
    finally:
        src.close()
        cv2.destroyAllWindows()

    print(f"完成：共保存 {len(points)} 个点。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
